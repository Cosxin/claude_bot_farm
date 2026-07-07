import math

import numpy as np
import pytest

from losscandles.indicators import bollinger, ema, overfit_regions, sma


def test_sma_basic():
    closes = [1, 2, 3, 4, 5]
    out = sma(closes, window=3)
    assert math.isnan(out[0])
    assert math.isnan(out[1])
    assert out[2:] == pytest.approx([2.0, 3.0, 4.0])


def test_sma_window_one_is_identity():
    closes = [1.0, 2.0, 3.0]
    assert sma(closes, window=1) == pytest.approx(closes)


def test_sma_ignores_internal_nan():
    closes = [1.0, float("nan"), 3.0]
    out = sma(closes, window=3)
    # nanmean of [1, nan, 3] == 2.0
    assert out[2] == pytest.approx(2.0)


def test_ema_matches_hand_computation():
    closes = [1.0, 2.0, 3.0]
    span = 2
    alpha = 2 / (span + 1)  # 2/3
    e0 = 1.0
    e1 = alpha * 2.0 + (1 - alpha) * e0
    e2 = alpha * 3.0 + (1 - alpha) * e1
    out = ema(closes, span=span)
    assert out == pytest.approx([e0, e1, e2])


def test_ema_carries_forward_on_nan():
    closes = [1.0, float("nan"), 5.0]
    out = ema(closes, span=2)
    assert out[1] == pytest.approx(out[0])
    assert out[0] == 1.0


def test_ema_nan_prefix_until_first_finite():
    closes = [float("nan"), float("nan"), 3.0]
    out = ema(closes, span=2)
    assert math.isnan(out[0])
    assert math.isnan(out[1])
    assert out[2] == pytest.approx(3.0)


def test_bollinger_bands_width():
    closes = [1.0, 2.0, 3.0, 4.0, 5.0]
    mid, upper, lower = bollinger(closes, window=5, num_std=2.0)
    expected_mid = 3.0
    expected_std = np.std(closes)  # population std, ddof=0
    assert mid[-1] == pytest.approx(expected_mid)
    assert upper[-1] == pytest.approx(expected_mid + 2 * expected_std)
    assert lower[-1] == pytest.approx(expected_mid - 2 * expected_std)
    assert all(math.isnan(v) for v in mid[:-1])


def test_overfit_regions_exact_three_candle_run():
    train = [10, 9, 8, 7]
    val = [5, 6, 7, 8]
    regions = overfit_regions(train, val, min_consecutive=3)
    assert regions == [(1, 3)]


def test_overfit_regions_below_threshold_not_flagged():
    train = [10, 9, 8, 9]
    val = [5, 6, 7, 6]
    regions = overfit_regions(train, val, min_consecutive=3)
    assert regions == []


def test_overfit_regions_ignores_nan():
    train = [10, 9, float("nan"), 7, 6]
    val = [5, 6, float("nan"), 8, 9]
    regions = overfit_regions(train, val, min_consecutive=3)
    # the NaN candle breaks the run into two length-1 runs, neither reaches 3
    assert regions == []


def test_overfit_regions_no_overlap_when_train_val_both_fall():
    train = [10, 9, 8, 7]
    val = [10, 9, 8, 7]
    assert overfit_regions(train, val, min_consecutive=3) == []
