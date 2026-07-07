import pytest
from tensorboard.backend.event_processing import data_ingester as di
from tensorboard.backend.event_processing import data_provider as event_data_provider
from tensorboard.backend.event_processing import plugin_event_multiplexer
from tensorboard.plugins import base_plugin

from losscandles.demo import write_demo_run
from losscandles.plugin import LossCandlesPlugin


@pytest.fixture(scope="session")
def demo_logdir(tmp_path_factory):
    logdir = str(tmp_path_factory.mktemp("demo_run"))
    write_demo_run(logdir, seed=0)
    return logdir


@pytest.fixture(scope="session")
def demo_run_name():
    # AddRunsFromDirectory treats the logdir passed to it as run "."
    return "."


@pytest.fixture(scope="session")
def data_provider(demo_logdir):
    """Mirrors how TensorBoard's own LocalDataIngester constructs its multiplexer,
    so tests see the same ingestion-time reservoir sampling real users would.
    """
    tensor_size_guidance = dict(di.DEFAULT_TENSOR_SIZE_GUIDANCE)
    multiplexer = plugin_event_multiplexer.EventMultiplexer(
        size_guidance=di.DEFAULT_SIZE_GUIDANCE,
        tensor_size_guidance=tensor_size_guidance,
    )
    multiplexer.AddRunsFromDirectory(demo_logdir)
    multiplexer.Reload()
    return event_data_provider.MultiplexerDataProvider(multiplexer, demo_logdir)


@pytest.fixture(scope="session")
def plugin(data_provider, demo_logdir):
    context = base_plugin.TBContext(data_provider=data_provider, logdir=demo_logdir)
    return LossCandlesPlugin(context)
