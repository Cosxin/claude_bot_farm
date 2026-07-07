import importlib.metadata
import importlib.util

from werkzeug.test import Client

from losscandles.plugin import LossCandlesPlugin

HAVE_TORCH = importlib.util.find_spec("torch") is not None


def test_entry_point_discovery():
    eps = list(importlib.metadata.entry_points(group="tensorboard_plugins"))
    matches = [ep for ep in eps if ep.name == "losscandles"]
    assert len(matches) == 1
    assert matches[0].value == "losscandles.plugin:LossCandlesPlugin"
    cls = matches[0].load()
    assert cls is LossCandlesPlugin
    assert cls.plugin_name == "losscandles"


def test_frontend_metadata(plugin):
    meta = plugin.frontend_metadata()
    assert meta.es_module_path == "/index.js"
    assert meta.tab_name == "Candles"


def test_is_active_with_data(plugin):
    assert plugin.is_active() is True


def test_serve_js_route(plugin):
    resp = Client(plugin._serve_js).get("/index.js")
    assert resp.status_code == 200
    assert "javascript" in resp.content_type
    body = resp.get_data(as_text=True)
    assert len(body) > 10_000
    assert "render" in body


def test_serve_tags_route(plugin, demo_run_name):
    resp = Client(plugin._serve_tags).get("/tags")
    assert resp.status_code == 200
    data = resp.get_json()
    tags = data["runs"][demo_run_name]
    assert set(["train/loss", "val/loss", "lr", "grad_norm", "epoch"]).issubset(set(tags))
    # set_volume_tag() enrichment is only written by the torch-based LossCandlesWriter;
    # the torch-free demo fallback (tensorboard.summary.Writer) doesn't have it.
    if HAVE_TORCH:
        assert data["config"][demo_run_name]["volume_tag"] == "grad_norm"


def test_serve_ohlc_full_fidelity(plugin, demo_run_name):
    resp = Client(plugin._serve_ohlc).get(f"/ohlc?run={demo_run_name}&tag=train/loss&window=45")
    assert resp.status_code == 200
    candles = resp.get_json()["candles"]
    assert len(candles) == 20
    assert candles[0]["step_start"] == 0
    assert candles[0]["n_points"] == 45
    assert candles[-1]["step_end"] == 899
    for c in candles:
        assert set(c.keys()) == {
            "index", "step_start", "step_end", "open", "high", "low", "close", "volume", "n_points", "has_nonfinite",
        }


def test_serve_ohlc_missing_params(plugin):
    resp = Client(plugin._serve_ohlc).get("/ohlc")
    assert resp.status_code == 400


def test_serve_ohlc_unknown_tag(plugin, demo_run_name):
    resp = Client(plugin._serve_ohlc).get(f"/ohlc?run={demo_run_name}&tag=does_not_exist&window=45")
    assert resp.status_code == 404


def test_serve_ohlc_invalid_window(plugin, demo_run_name):
    resp = Client(plugin._serve_ohlc).get(f"/ohlc?run={demo_run_name}&tag=train/loss&window=0")
    assert resp.status_code == 400


def test_serve_ohlc_volume_tag_changes_volume(plugin, demo_run_name):
    client = Client(plugin._serve_ohlc)
    default_resp = client.get(f"/ohlc?run={demo_run_name}&tag=train/loss&window=45")
    vol_resp = client.get(f"/ohlc?run={demo_run_name}&tag=train/loss&window=45&volume_tag=grad_norm")
    default_candles = default_resp.get_json()["candles"]
    vol_candles = vol_resp.get_json()["candles"]
    assert default_candles[0]["volume"] == 45.0  # point count
    assert vol_candles[0]["volume"] != 45.0  # grad_norm aggregate instead


def test_serve_annotations_missing_params(plugin):
    resp = Client(plugin._serve_annotations).get("/annotations")
    assert resp.status_code == 400


def test_serve_annotations_full(plugin, demo_run_name):
    url = f"/annotations?run={demo_run_name}&tag=train/loss&window=45&lr_tag=lr&val_tag=val/loss"
    resp = Client(plugin._serve_annotations).get(url)
    assert resp.status_code == 200
    data = resp.get_json()

    assert len(data["circuit_breakers"]) == 1
    assert data["circuit_breakers"][0]["label"] == "CIRCUIT BREAKER HALT"

    assert len(data["flash_crashes"]) >= 1
    assert data["flash_crashes"][0]["label"] == "SELL"

    assert len(data["rate_cuts"]) == 1
    assert data["rate_cuts"][0]["label"] == "RATE CUT"
    assert data["rate_cuts"][0]["step"] == 450

    assert len(data["overfit_regions"]) >= 1
    assert data["overfit_regions"][0]["label"] == "OVERFITTING"

    assert data["epoch_boundaries"] == sorted(data["epoch_boundaries"])
    assert 90 in data["epoch_boundaries"]


def test_serve_annotations_without_lr_or_val_tag(plugin, demo_run_name):
    url = f"/annotations?run={demo_run_name}&tag=train/loss&window=45"
    resp = Client(plugin._serve_annotations).get(url)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["rate_cuts"] == []
    assert data["overfit_regions"] == []
    # circuit breaker + epoch boundaries are zero-instrumentation and still work
    assert len(data["circuit_breakers"]) == 1
    assert len(data["epoch_boundaries"]) > 0
