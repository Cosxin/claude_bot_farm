import importlib.metadata
import importlib.util

import pytest
from tensorboard.plugins import base_plugin
from werkzeug.test import Client

from losscandles.plugin import LossCandlesPlugin

HAVE_TORCH = importlib.util.find_spec("torch") is not None


class _StaticDataProvider:
    """Fake data_provider returning a fixed list_scalars() mapping, for exercising
    is_active()'s branches without needing a real logdir.
    """

    def __init__(self, mapping):
        self._mapping = mapping

    def list_scalars(self, *args, **kwargs):
        return self._mapping


class _RaisingDataProvider:
    def list_scalars(self, *args, **kwargs):
        raise RuntimeError("boom")


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


def test_is_active_false_with_no_data_provider():
    context = base_plugin.TBContext(data_provider=None, logdir="/nonexistent")
    assert LossCandlesPlugin(context).is_active() is False


def test_is_active_false_when_list_scalars_raises():
    context = base_plugin.TBContext(data_provider=_RaisingDataProvider(), logdir="/nonexistent")
    assert LossCandlesPlugin(context).is_active() is False


def test_is_active_false_with_empty_mapping():
    context = base_plugin.TBContext(data_provider=_StaticDataProvider({}), logdir="/nonexistent")
    assert LossCandlesPlugin(context).is_active() is False


def test_is_active_false_when_run_has_no_tags():
    context = base_plugin.TBContext(data_provider=_StaticDataProvider({"run": {}}), logdir="/nonexistent")
    assert LossCandlesPlugin(context).is_active() is False


def test_is_active_true_when_run_has_tags():
    # Sanity check on the fake itself: proves the False cases above are really
    # exercising is_active()'s logic and not just a fake that always returns {}.
    context = base_plugin.TBContext(data_provider=_StaticDataProvider({"run": {"tag": object()}}), logdir="/nonexistent")
    assert LossCandlesPlugin(context).is_active() is True


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


def test_serve_ohlc_non_integer_window(plugin, demo_run_name):
    resp = Client(plugin._serve_ohlc).get(f"/ohlc?run={demo_run_name}&tag=train/loss&window=abc")
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


def test_serve_annotations_unknown_tag_404(plugin, demo_run_name):
    resp = Client(plugin._serve_annotations).get(f"/annotations?run={demo_run_name}&tag=does_not_exist&window=45")
    assert resp.status_code == 404


def test_serve_annotations_non_integer_window(plugin, demo_run_name):
    resp = Client(plugin._serve_annotations).get(f"/annotations?run={demo_run_name}&tag=train/loss&window=abc")
    assert resp.status_code == 400


@pytest.mark.parametrize("param", ["flash_crash_threshold", "rate_cut_threshold"])
@pytest.mark.parametrize("bad_value", ["not_a_number", "0", "-1", "nan", "inf"])
def test_serve_annotations_invalid_threshold_400(plugin, demo_run_name, param, bad_value):
    url = f"/annotations?run={demo_run_name}&tag=train/loss&window=45&{param}={bad_value}"
    resp = Client(plugin._serve_annotations).get(url)
    assert resp.status_code == 400


@pytest.mark.parametrize("param", ["flash_crash_threshold", "rate_cut_threshold"])
def test_serve_annotations_valid_custom_threshold_200(plugin, demo_run_name, param):
    url = f"/annotations?run={demo_run_name}&tag=train/loss&window=45&{param}=0.5"
    resp = Client(plugin._serve_annotations).get(url)
    assert resp.status_code == 200


def test_serve_annotations_full(plugin, demo_run_name):
    url = f"/annotations?run={demo_run_name}&tag=train/loss&window=45&lr_tag=lr&val_tag=val/loss"
    resp = Client(plugin._serve_annotations).get(url)
    assert resp.status_code == 200
    data = resp.get_json()

    # Exact values from the seed=0 demo run (deterministic given the fixed
    # seed), not just counts/labels -- pins down index/step, not just "some
    # annotation fired somewhere".
    assert data["circuit_breakers"] == [{"index": 11, "step": 495, "label": "CIRCUIT BREAKER HALT"}]
    assert data["flash_crashes"] == [{"index": 7, "step": 315, "label": "SELL"}]
    assert data["rate_cuts"] == [{"index": 10, "step": 450, "label": "RATE CUT"}]
    assert data["overfit_regions"] == [
        {"start_index": 16, "end_index": 19, "start_step": 720, "end_step": 899, "label": "OVERFITTING"}
    ]
    assert data["epoch_boundaries"] == sorted(data["epoch_boundaries"])
    assert data["epoch_boundaries"] == [0, 90, 180, 270, 360, 450, 540, 630, 720, 810]


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


def test_serve_annotations_typo_lr_and_val_tag_behaves_like_omitted(plugin, demo_run_name):
    # A tag name that doesn't exist must degrade gracefully to "no series
    # found" (empty rate_cuts/overfit_regions) exactly like the omitted-param
    # case above -- not a 404 or 500.
    url = (
        f"/annotations?run={demo_run_name}&tag=train/loss&window=45"
        "&lr_tag=lr_typo_xyz&val_tag=val_typo_xyz"
    )
    resp = Client(plugin._serve_annotations).get(url)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["rate_cuts"] == []
    assert data["overfit_regions"] == []


def test_serve_ohlc_all_nonfinite_window_is_json_safe(tmp_path, make_plugin):
    # json.dumps(float("nan")) emits a literal `NaN` token, which Python's own
    # json.loads happily round-trips back to nan -- so a test that only checks
    # HTTP status or looks for "NaN" in the raw text wouldn't catch a missing
    # _json_safe() call. Asserting `is None` through get_json() would.
    from tensorboard.summary import Writer

    logdir = str(tmp_path)
    writer = Writer(logdir)
    for step, value in enumerate([float("nan"), float("inf"), float("-inf")]):
        writer.add_scalar("train/loss", value, step)
    writer.close()

    plugin = make_plugin(logdir)
    resp = Client(plugin._serve_ohlc).get("/ohlc?run=.&tag=train/loss&window=3")
    assert resp.status_code == 200
    candles = resp.get_json()["candles"]
    assert len(candles) == 1
    c = candles[0]
    assert c["open"] is None
    assert c["high"] is None
    assert c["low"] is None
    assert c["close"] is None
    assert c["has_nonfinite"] is True
    assert c["volume"] == 3  # point count volume stays finite regardless


def test_zero_instrumentation_logdir_served_over_http(tmp_path, make_plugin):
    # Proves the "works on any logdir written by plain SummaryWriter.add_scalar"
    # claim end-to-end over HTTP: no losscandles.torch writer, no plugin_data,
    # no epoch/volume enrichment -- just a bare tensorboard.summary.Writer.
    from tensorboard.summary import Writer

    logdir = str(tmp_path)
    writer = Writer(logdir)
    for step in range(10):
        writer.add_scalar("loss", 1.0 / (step + 1), step)
    writer.close()

    plugin = make_plugin(logdir)

    tags_resp = Client(plugin._serve_tags).get("/tags")
    assert tags_resp.status_code == 200
    tags_data = tags_resp.get_json()
    assert tags_data["runs"]["."] == ["loss"]
    assert tags_data["config"] == {}

    ohlc_resp = Client(plugin._serve_ohlc).get("/ohlc?run=.&tag=loss&window=5")
    assert ohlc_resp.status_code == 200
    candles = ohlc_resp.get_json()["candles"]
    assert len(candles) == 2
    assert candles[0]["open"] == 1.0

    annotations_resp = Client(plugin._serve_annotations).get("/annotations?run=.&tag=loss&window=5")
    assert annotations_resp.status_code == 200
    assert annotations_resp.get_json()["epoch_boundaries"] == []
