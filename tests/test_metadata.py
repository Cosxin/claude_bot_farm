from tensorboard.compat.proto.summary_pb2 import DATA_CLASS_SCALAR

from losscandles import metadata


def test_build_marker_summary_shape():
    summary = metadata.build_marker_summary("losscandles/epoch_boundary", 42.0, {"epoch": 3})
    value = summary.value[0]
    assert value.tag == "losscandles/epoch_boundary"
    assert value.tensor.float_val[0] == 42.0
    assert value.metadata.plugin_data.plugin_name == metadata.PLUGIN_NAME
    assert value.metadata.data_class == DATA_CLASS_SCALAR


def test_build_marker_summary_content_roundtrip():
    summary = metadata.build_marker_summary("losscandles/config", 0.0, {"volume_tag": "grad_norm"})
    content = summary.value[0].metadata.plugin_data.content
    assert metadata.parse_config_content(content) == {"volume_tag": "grad_norm"}


def test_parse_config_content_empty_bytes():
    assert metadata.parse_config_content(b"") == {}


def test_parse_config_content_invalid_json():
    assert metadata.parse_config_content(b"not json") == {}
