"""Indicators computed over candle closes: SMA, EMA, Bollinger bands, overfit-index."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def sma(closes: Sequence[float], window: int) -> list[float]:
    """Simple moving average. First `window - 1` points are NaN (insufficient history)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    arr = np.asarray(closes, dtype=float)
    out = np.full(arr.shape, np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = np.nanmean(arr[i - window + 1 : i + 1])
    return out.tolist()


def ema(closes: Sequence[float], span: int) -> list[float]:
    """Exponential moving average with smoothing alpha = 2 / (span + 1).

    NaN closes carry the previous EMA value forward rather than resetting it.
    """
    if span < 1:
        raise ValueError("span must be >= 1")
    alpha = 2.0 / (span + 1)
    out: list[float] = []
    prev = float("nan")
    for v in closes:
        if np.isnan(v):
            out.append(prev)
            continue
        prev = v if np.isnan(prev) else alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


def bollinger(
    closes: Sequence[float], window: int = 20, num_std: float = 2.0
) -> tuple[list[float], list[float], list[float]]:
    """Rolling (mid, upper, lower) bands: mid = SMA, upper/lower = mid +/- num_std * rolling std."""
    if window < 1:
        raise ValueError("window must be >= 1")
    arr = np.asarray(closes, dtype=float)
    mid = np.full(arr.shape, np.nan)
    std = np.full(arr.shape, np.nan)
    for i in range(window - 1, len(arr)):
        chunk = arr[i - window + 1 : i + 1]
        mid[i] = np.nanmean(chunk)
        std[i] = np.nanstd(chunk)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid.tolist(), upper.tolist(), lower.tolist()


def overfit_regions(
    train_closes: Sequence[float],
    val_closes: Sequence[float],
    min_consecutive: int = 3,
) -> list[tuple[int, int]]:
    """Index ranges (inclusive, candle-index space) of `min_consecutive`+ consecutive
    candles where each candle's val close rose and train close fell versus the
    candle immediately before it.
    """
    n = min(len(train_closes), len(val_closes))
    transition = [False] * n  # transition[k]: candle k-1 -> k is val-up/train-down
    for k in range(1, n):
        t0, t1 = train_closes[k - 1], train_closes[k]
        v0, v1 = val_closes[k - 1], val_closes[k]
        if any(np.isnan(x) for x in (t0, t1, v0, v1)):
            continue
        transition[k] = v1 > v0 and t1 < t0

    regions: list[tuple[int, int]] = []
    run_start = None
    run_len = 0
    for k in range(1, n):
        if transition[k]:
            if run_start is None:
                run_start = k
            run_len += 1
        else:
            if run_len >= min_consecutive:
                regions.append((run_start, run_start + run_len - 1))
            run_start, run_len = None, 0
    if run_len >= min_consecutive:
        regions.append((run_start, run_start + run_len - 1))
    return regions
