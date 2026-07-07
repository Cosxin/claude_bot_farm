import pytest

pytest.importorskip("torch")

from tensorboard.backend.event_processing import data_ingester as di
from tensorboard.backend.event_processing import data_provider as event_data_provider
from tensorboard.backend.event_processing import plugin_event_multiplexer
from tensorboard import context as tb_context
from tensorboard.data import provider

from losscandles.torch import LossCandlesWriter, TradingHalt
from losscandles import metadata


def _read_marker_series(logdir, tag):
    tensor_size_guidance = dict(di.DEFAULT_TENSOR_SIZE_GUIDANCE)
    multiplexer = plugin_event_multiplexer.EventMultiplexer(
        size_guidance=di.DEFAULT_SIZE_GUIDANCE, tensor_size_guidance=tensor_size_guidance
    )
    multiplexer.AddRunsFromDirectory(logdir)
    multiplexer.Reload()
    dp = event_data_provider.MultiplexerDataProvider(multiplexer, logdir)
    ctx = tb_context.RequestContext()
    data = dp.read_scalars(
        ctx,
        experiment_id="",
        plugin_name=metadata.PLUGIN_NAME,
        downsample=1000,
        run_tag_filter=provider.RunTagFilter(tags=[tag]),
    )
    return data.get(".", {}).get(tag)


def test_standard_scalars_pass_through(tmp_path):
    logdir = str(tmp_path)
    writer = LossCandlesWriter(logdir)
    writer.add_scalar("train/loss", 1.0, 0)
    writer.add_scalar("train/loss", 0.5, 1)
    writer.close()

    tensor_size_guidance = dict(di.DEFAULT_TENSOR_SIZE_GUIDANCE)
    multiplexer = plugin_event_multiplexer.EventMultiplexer(
        size_guidance=di.DEFAULT_SIZE_GUIDANCE, tensor_size_guidance=tensor_size_guidance
    )
    multiplexer.AddRunsFromDirectory(logdir)
    multiplexer.Reload()
    dp = event_data_provider.MultiplexerDataProvider(multiplexer, logdir)
    ctx = tb_context.RequestContext()
    data = dp.read_scalars(
        ctx, experiment_id="", plugin_name="scalars", downsample=1000, run_tag_filter=provider.RunTagFilter(tags=["train/loss"])
    )
    series = data["."]["train/loss"]
    assert [(s.step, s.value) for s in series] == [(0, 1.0), (1, 0.5)]


def test_nan_writes_circuit_breaker_marker_without_halting(tmp_path):
    logdir = str(tmp_path)
    writer = LossCandlesWriter(logdir, halt_on_nan=False)
    writer.add_scalar("train/loss", 1.0, 0)
    writer.add_scalar("train/loss", float("nan"), 1)
    writer.close()

    series = _read_marker_series(logdir, metadata.TAG_CIRCUIT_BREAKER)
    assert series is not None
    assert len(series) == 1
    assert series[0].step == 1


def test_halt_on_nan_raises_trading_halt(tmp_path):
    writer = LossCandlesWriter(str(tmp_path), halt_on_nan=True)
    writer.add_scalar("train/loss", 1.0, 0)
    with pytest.raises(TradingHalt):
        writer.add_scalar("train/loss", float("inf"), 1)
    writer.close()


def test_finite_scalars_never_raise(tmp_path):
    writer = LossCandlesWriter(str(tmp_path), halt_on_nan=True)
    for step in range(5):
        writer.add_scalar("train/loss", 1.0 / (step + 1), step)
    writer.close()


def test_add_epoch_boundary_writes_marker(tmp_path):
    logdir = str(tmp_path)
    writer = LossCandlesWriter(logdir)
    writer.add_epoch_boundary(100)
    writer.add_epoch_boundary(200)
    writer.close()

    series = _read_marker_series(logdir, metadata.TAG_EPOCH_BOUNDARY)
    assert [s.step for s in series] == [100, 200]


def test_set_volume_tag_writes_config_marker(tmp_path):
    logdir = str(tmp_path)
    writer = LossCandlesWriter(logdir)
    writer.set_volume_tag("grad_norm")
    writer.close()

    series = _read_marker_series(logdir, metadata.TAG_CONFIG)
    assert series is not None and len(series) == 1
