import math

import pytest

from losscandles.ohlc import aggregate_ohlc, align_candles


def test_basic_two_candles():
    steps = [0, 1, 2, 3]
    values = [1.0, 2.0, 3.0, 0.0]
    candles = aggregate_ohlc(steps, values, window=2)
    assert len(candles) == 2

    c0, c1 = candles
    assert (c0.start_step, c0.end_step) == (0, 1)
    assert (c0.open, c0.close, c0.high, c0.low) == (1.0, 2.0, 2.0, 1.0)
    assert c0.volume == 2
    assert c0.has_nonfinite is False

    assert (c1.start_step, c1.end_step) == (2, 3)
    assert (c1.open, c1.close, c1.high, c1.low) == (3.0, 0.0, 3.0, 0.0)
    assert c1.volume == 2


def test_ragged_final_window():
    steps = [0, 1, 2]
    values = [10.0, 20.0, 30.0]
    candles = aggregate_ohlc(steps, values, window=2)
    assert len(candles) == 2
    assert candles[0].n_points == 2
    assert candles[1].n_points == 1
    # single leftover point forms its own open==high==low==close candle
    c1 = candles[1]
    assert (c1.open, c1.high, c1.low, c1.close) == (30.0, 30.0, 30.0, 30.0)


def test_single_point_windows():
    steps = [0, 1, 2]
    values = [5.0, -3.0, 7.0]
    candles = aggregate_ohlc(steps, values, window=1)
    assert len(candles) == 3
    for c, v in zip(candles, values):
        assert (c.open, c.high, c.low, c.close) == (v, v, v, v)
        assert c.n_points == 1


def test_nan_excluded_from_ohlc_but_flagged():
    steps = [0, 1, 2]
    values = [1.0, float("nan"), 3.0]
    candles = aggregate_ohlc(steps, values, window=3)
    c = candles[0]
    assert (c.open, c.close, c.high, c.low) == (1.0, 3.0, 3.0, 1.0)
    assert c.has_nonfinite is True
    assert c.n_points == 3


def test_inf_excluded_from_ohlc_but_flagged():
    steps = [0, 1, 2]
    values = [1.0, float("inf"), 3.0]
    candles = aggregate_ohlc(steps, values, window=3)
    c = candles[0]
    assert (c.open, c.close, c.high, c.low) == (1.0, 3.0, 3.0, 1.0)
    assert c.has_nonfinite is True


def test_all_nonfinite_window_yields_nan_ohlc():
    steps = [0, 1]
    values = [float("nan"), float("inf")]
    candles = aggregate_ohlc(steps, values, window=2)
    c = candles[0]
    assert math.isnan(c.open)
    assert math.isnan(c.high)
    assert math.isnan(c.low)
    assert math.isnan(c.close)
    assert c.has_nonfinite is True
    assert c.n_points == 2


def test_default_volume_is_point_count():
    steps = list(range(5))
    values = [1.0] * 5
    candles = aggregate_ohlc(steps, values, window=5)
    assert candles[0].volume == 5


def test_volume_tag_mean_aggregation():
    steps = [0, 1, 2, 3]
    values = [1.0, 2.0, 3.0, 4.0]
    volume_values = [10.0, 20.0, 30.0, 40.0]
    candles = aggregate_ohlc(steps, values, window=2, volume_values=volume_values, volume_agg="mean")
    assert candles[0].volume == 15.0
    assert candles[1].volume == 35.0


def test_volume_tag_sum_aggregation():
    steps = [0, 1, 2, 3]
    values = [1.0, 2.0, 3.0, 4.0]
    volume_values = [10.0, 20.0, 30.0, 40.0]
    candles = aggregate_ohlc(steps, values, window=2, volume_values=volume_values, volume_agg="sum")
    assert candles[0].volume == 30.0
    assert candles[1].volume == 70.0


def test_volume_tag_last_aggregation():
    steps = [0, 1, 2, 3]
    values = [1.0, 2.0, 3.0, 4.0]
    volume_values = [10.0, 20.0, 30.0, 40.0]
    candles = aggregate_ohlc(steps, values, window=2, volume_values=volume_values, volume_agg="last")
    assert candles[0].volume == 20.0
    assert candles[1].volume == 40.0


def test_epoch_based_windows():
    steps = list(range(6))
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    epochs = [0, 0, 0, 1, 1, 1]
    candles = aggregate_ohlc(steps, values, epochs=epochs)
    assert len(candles) == 2
    assert (candles[0].start_step, candles[0].end_step) == (0, 2)
    assert (candles[1].start_step, candles[1].end_step) == (3, 5)
    assert (candles[0].open, candles[0].close) == (1.0, 3.0)
    assert (candles[1].open, candles[1].close) == (4.0, 6.0)


def test_unsorted_steps_are_sorted_first():
    steps = [2, 0, 1]
    values = [30.0, 10.0, 20.0]
    candles = aggregate_ohlc(steps, values, window=3)
    c = candles[0]
    assert (c.open, c.close, c.high, c.low) == (10.0, 30.0, 30.0, 10.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        aggregate_ohlc([0, 1], [1.0], window=1)


def test_empty_input_returns_no_candles():
    assert aggregate_ohlc([], [], window=10) == []


def test_window_is_step_range_not_position_count():
    # Sparse, gapped steps: window=100 should bucket by step // 100, not by
    # position in the list -- this is what makes cross-tag alignment valid
    # when two tags are logged at different frequencies.
    steps = [0, 5, 100, 105, 250]
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    candles = aggregate_ohlc(steps, values, window=100)
    assert len(candles) == 3
    assert (candles[0].start_step, candles[0].end_step) == (0, 5)
    assert (candles[1].start_step, candles[1].end_step) == (100, 105)
    assert (candles[2].start_step, candles[2].end_step) == (250, 250)
    assert candles[0].n_points == 2
    assert candles[1].n_points == 2
    assert candles[2].n_points == 1


def test_window_step_range_matches_sparse_and_dense_series():
    # A densely-logged tag (every step) and a sparsely-logged tag (every 10
    # steps) should produce candles covering the SAME step ranges for the
    # same window size, so index-free alignment by step // window is valid.
    dense_steps = list(range(300))
    dense_values = [float(s) for s in dense_steps]
    sparse_steps = list(range(0, 300, 10))
    sparse_values = [float(s) for s in sparse_steps]

    dense_candles = aggregate_ohlc(dense_steps, dense_values, window=100)
    sparse_candles = aggregate_ohlc(sparse_steps, sparse_values, window=100)

    assert [c.start_step // 100 for c in dense_candles] == [0, 1, 2]
    assert [c.start_step // 100 for c in sparse_candles] == [0, 1, 2]


def test_align_candles_matches_by_step_range_across_frequencies():
    # train logged every step, val logged once every 100 steps (once/epoch-like)
    train_steps = list(range(300))
    train_values = [10.0 - s * 0.01 for s in train_steps]  # falling
    val_steps = [0, 100, 200]
    val_values = [5.0, 6.0, 7.0]  # rising

    train_candles = aggregate_ohlc(train_steps, train_values, window=100)
    val_candles = aggregate_ohlc(val_steps, val_values, window=100)

    pairs = align_candles(train_candles, val_candles, window=100)
    assert len(pairs) == 3
    for train_c, val_c in pairs:
        assert train_c.start_step // 100 == val_c.start_step // 100
    assert [v.close for _, v in pairs] == [5.0, 6.0, 7.0]


def test_align_candles_skips_unmatched_windows():
    a = aggregate_ohlc([0, 50], [1.0, 2.0], window=50)
    b = aggregate_ohlc([500, 550], [3.0, 4.0], window=50)
    assert align_candles(a, b, window=50) == []
