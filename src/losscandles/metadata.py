"""Shared constants and summary-proto helpers for the losscandles plugin.

Kept torch-free: building `Summary` protos only needs the `tensorboard` package.
Only writing them through a real file writer requires torch (see `losscandles.torch`).
"""

from __future__ import annotations

import json

from tensorboard.compat.proto.summary_pb2 import DATA_CLASS_SCALAR, Summary, SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto

PLUGIN_NAME = "losscandles"

TAG_EPOCH_BOUNDARY = "losscandles/epoch_boundary"
TAG_CIRCUIT_BREAKER = "losscandles/circuit_breaker"
TAG_CONFIG = "losscandles/config"


def build_marker_summary(tag: str, value: float, extra: dict | None = None) -> Summary:
    """Build a scalar-shaped Summary proto tagged with our own plugin_name, so the
    backend can read it back via `data_provider.read_scalars(plugin_name=PLUGIN_NAME)`
    without mixing it up with the standard "scalars" plugin's data.
    """
    tensor_proto = TensorProto(float_val=[float(value)], dtype="DT_FLOAT")
    content = json.dumps(extra or {}).encode("utf-8")
    plugin_data = SummaryMetadata.PluginData(plugin_name=PLUGIN_NAME, content=content)
    smd = SummaryMetadata(plugin_data=plugin_data, data_class=DATA_CLASS_SCALAR)
    return Summary(value=[Summary.Value(tag=tag, tensor=tensor_proto, metadata=smd)])


def parse_config_content(content: bytes) -> dict:
    if not content:
        return {}
    try:
        return json.loads(content.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
