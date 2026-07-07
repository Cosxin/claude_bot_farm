"""Aggregate raw (step, value) scalar streams into OHLC candles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

VolumeAgg = Literal["mean", "sum", "last"]


@dataclass
class Candle:
    index: int
    start_step: int
    end_step: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    n_points: int
    has_nonfinite: bool = False


def _is_finite(v: float) -> bool:
    return v is not None and math.isfinite(v)


def aggregate_ohlc(
    steps: Sequence[float],
    values: Sequence[float],
    window: int = 100,
    epochs: Sequence[float] | None = None,
    volume_values: Sequence[float] | None = None,
    volume_agg: VolumeAgg = "mean",
) -> list[Candle]:
    """Group a scalar stream into candles spanning `window` consecutive *step
    values* each (i.e. candle k covers steps [k*window, (k+1)*window)), or into
    one candle per contiguous run of equal `epochs` value.

    Windowing by step value (not by position in the list) matters whenever two
    tags are logged at different frequencies -- e.g. train/loss every step but
    val/loss once per epoch. Candle k always covers the same step range for
    every tag, so cross-tag comparisons (overfit-index, rate-cut positioning)
    stay meaningful regardless of logging density. A tag with sparse logging
    can simply be missing candles for step ranges it has no points in; callers
    that align two candle lists should match on `start_step // window`, not on
    list position.

    Open/close use the first/last *finite* value in the window (NaN/inf values
    are excluded from O/H/L/C but still counted in `n_points`, and the candle
    is flagged `has_nonfinite`); a window with no finite values yields NaN
    O/H/L/C. `volume_values` (e.g. grad_norm, lr, tokens/sec) is aggregated
    with `volume_agg` instead of the default point-count volume.
    """
    n = len(steps)
    if n != len(values):
        raise ValueError("steps and values must be the same length")
    if volume_values is not None and len(volume_values) != n:
        raise ValueError("volume_values must be the same length as steps")
    if epochs is not None and len(epochs) != n:
        raise ValueError("epochs must be the same length as steps")
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: steps[i])
    steps = [steps[i] for i in order]
    values = [values[i] for i in order]
    if volume_values is not None:
        volume_values = [volume_values[i] for i in order]
    if epochs is not None:
        epochs = [epochs[i] for i in order]

    if epochs is not None:
        window_ids = [0] * n
        for i in range(1, n):
            window_ids[i] = window_ids[i - 1] + (1 if epochs[i] != epochs[i - 1] else 0)
    else:
        if window < 1:
            raise ValueError("window must be >= 1")
        window_ids = [int(steps[i] // window) for i in range(n)]

    candles = []
    start = 0
    while start < n:
        end = start
        w = window_ids[start]
        while end < n and window_ids[end] == w:
            end += 1
        candles.append(
            _build_candle(len(candles), steps, values, range(start, end), volume_values, volume_agg)
        )
        start = end
    return candles


def align_candles(a: list[Candle], b: list[Candle], window: int) -> list[tuple[Candle, Candle]]:
    """Pair up candles from two (possibly differently-sparse) candle lists that
    cover the same step range, for cross-tag comparisons like overfit-index.
    Aligning by list position is wrong whenever the two tags are logged at
    different frequencies -- e.g. train/loss every step, val/loss once/epoch.
    """
    b_by_window_id = {c.start_step // window: c for c in b}
    return [(ca, b_by_window_id[wid]) for ca in a if (wid := ca.start_step // window) in b_by_window_id]


def _build_candle(index, steps, values, idx, volume_values, volume_agg: VolumeAgg):
    idx = list(idx)
    window_vals = [values[i] for i in idx]
    finite_vals = [v for v in window_vals if _is_finite(v)]
    has_nonfinite = len(finite_vals) < len(window_vals)

    if finite_vals:
        open_, close, high, low = finite_vals[0], finite_vals[-1], max(finite_vals), min(finite_vals)
    else:
        open_ = close = high = low = float("nan")

    if volume_values is None:
        volume = float(len(idx))
    else:
        finite_vol = [volume_values[i] for i in idx if _is_finite(volume_values[i])]
        if not finite_vol:
            volume = float("nan")
        elif volume_agg == "sum":
            volume = float(sum(finite_vol))
        elif volume_agg == "last":
            volume = float(finite_vol[-1])
        else:
            volume = float(sum(finite_vol) / len(finite_vol))

    return Candle(
        index=index,
        start_step=int(steps[idx[0]]),
        end_step=int(steps[idx[-1]]),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        n_points=len(idx),
        has_nonfinite=has_nonfinite,
    )
