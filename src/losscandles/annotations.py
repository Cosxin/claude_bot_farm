"""Always-on auto-annotations: circuit breakers, flash crashes, rate cuts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from losscandles.ohlc import Candle

CIRCUIT_BREAKER = "circuit_breaker"
FLASH_CRASH = "flash_crash"
RATE_CUT = "rate_cut"


@dataclass
class Annotation:
    index: int
    kind: str
    label: str


def detect_circuit_breakers(candles: Sequence[Candle]) -> list[Annotation]:
    """One annotation per candle window that contained a NaN/inf value."""
    return [
        Annotation(index=c.index, kind=CIRCUIT_BREAKER, label="CIRCUIT BREAKER HALT")
        for c in candles
        if c.has_nonfinite
    ]


def detect_flash_crashes(candles: Sequence[Candle], threshold: float = 0.30) -> list[Annotation]:
    """Flag candles whose close rose more than `threshold` (30% by default) versus
    the previous candle's close -- i.e. loss got sharply worse. Labeled as a
    standard chart "SELL" signal rather than "FLASH CRASH" (too alarmist for
    something that fires routinely on noisy runs).
    """
    out = []
    prev_close = None
    for c in candles:
        if prev_close is not None and not math.isnan(prev_close) and not math.isnan(c.close) and prev_close != 0:
            change = (c.close - prev_close) / abs(prev_close)
            if change > threshold:
                out.append(Annotation(index=c.index, kind=FLASH_CRASH, label="SELL"))
        if not math.isnan(c.close):
            prev_close = c.close
    return out


def detect_rate_cuts(lr_candles: Sequence[Candle], min_drop_frac: float = 0.10) -> list[Annotation]:
    """Flag candles where the lr-tag close dropped by more than `min_drop_frac`
    versus the previous candle's close (a discrete scheduler step, not smooth decay).
    """
    out = []
    prev_close = None
    for c in lr_candles:
        if prev_close is not None and not math.isnan(prev_close) and not math.isnan(c.close) and prev_close != 0:
            drop = (prev_close - c.close) / abs(prev_close)
            if drop > min_drop_frac:
                out.append(Annotation(index=c.index, kind=RATE_CUT, label="RATE CUT"))
        if not math.isnan(c.close):
            prev_close = c.close
    return out
