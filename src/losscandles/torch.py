"""PyTorch extension: a drop-in SummaryWriter subclass with candlestick-plugin extras.

All standard `add_scalar` calls pass through unchanged and are fully readable by the
losscandles TensorBoard plugin with zero instrumentation. The extras here (epoch
gridlines, a declared volume tag, halt-on-nan) are optional enrichment written as
summaries under our own plugin_name, never a requirement.
"""

from __future__ import annotations

import math

from torch.utils.tensorboard import SummaryWriter

from losscandles import metadata


class TradingHalt(RuntimeError):
    """Raised by `LossCandlesWriter` when `halt_on_nan=True` and a NaN/Inf is logged."""


class LossCandlesWriter(SummaryWriter):
    def __init__(self, *args, halt_on_nan: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.halt_on_nan = halt_on_nan
        self._epoch_counter = 0

    def add_scalar(
        self,
        tag,
        scalar_value,
        global_step=None,
        walltime=None,
        new_style=False,
        double_precision=False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step,
            walltime=walltime,
            new_style=new_style,
            double_precision=double_precision,
        )
        try:
            value = float(scalar_value)
        except (TypeError, ValueError):
            return
        if math.isfinite(value):
            return
        step = global_step or 0
        self._write_marker(metadata.TAG_CIRCUIT_BREAKER, step, {"tag": tag})
        if self.halt_on_nan:
            raise TradingHalt(f"losscandles: NaN/Inf logged for {tag!r} at step {step} — halting.")

    def add_epoch_boundary(self, step: int) -> None:
        """Log an epoch gridline the plugin renders across the chart."""
        self._epoch_counter += 1
        self._write_marker(metadata.TAG_EPOCH_BOUNDARY, step, {"epoch": self._epoch_counter})

    def set_volume_tag(self, tag: str) -> None:
        """Declare which scalar tag the plugin's volume bars should aggregate by default."""
        self._write_marker(metadata.TAG_CONFIG, 0, {"volume_tag": tag})

    def _write_marker(self, tag: str, step: int, extra: dict) -> None:
        summary = metadata.build_marker_summary(tag, float(step), extra)
        self._get_file_writer().add_summary(summary, global_step=step)
