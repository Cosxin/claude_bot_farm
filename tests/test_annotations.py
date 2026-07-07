from losscandles.annotations import (
    CIRCUIT_BREAKER,
    FLASH_CRASH,
    RATE_CUT,
    detect_circuit_breakers,
    detect_flash_crashes,
    detect_rate_cuts,
)
from losscandles.ohlc import aggregate_ohlc


def _candles(values, window=1):
    steps = list(range(len(values)))
    return aggregate_ohlc(steps, values, window=window)


def test_circuit_breaker_flags_nonfinite_windows():
    candles = _candles([1.0, float("nan"), 2.0, float("inf"), 3.0], window=1)
    flags = detect_circuit_breakers(candles)
    assert [a.index for a in flags] == [1, 3]
    assert all(a.kind == CIRCUIT_BREAKER for a in flags)
    assert all(a.label == "CIRCUIT BREAKER HALT" for a in flags)


def test_circuit_breaker_none_when_all_finite():
    candles = _candles([1.0, 2.0, 3.0], window=1)
    assert detect_circuit_breakers(candles) == []


def test_flash_crash_detects_big_jump():
    # candle closes: 1.0 -> 1.4 (+40%) should trigger; 1.4 -> 1.5 (~7%) should not
    candles = _candles([1.0, 1.4, 1.5], window=1)
    flags = detect_flash_crashes(candles, threshold=0.30)
    assert [a.index for a in flags] == [1]
    assert flags[0].kind == FLASH_CRASH


def test_flash_crash_ignores_drops():
    candles = _candles([1.0, 0.5], window=1)
    assert detect_flash_crashes(candles, threshold=0.30) == []


def test_flash_crash_skips_nan_neighbors():
    candles = _candles([1.0, float("nan"), 5.0], window=1)
    # prev_close carry logic: nan candle doesn't update prev_close, and its own
    # close is nan so it can't be flagged either.
    flags = detect_flash_crashes(candles, threshold=0.30)
    assert [a.index for a in flags] == [2]


def test_rate_cut_detects_halving():
    candles = _candles([0.1, 0.1, 0.05, 0.05], window=1)
    flags = detect_rate_cuts(candles, min_drop_frac=0.10)
    assert [a.index for a in flags] == [2]
    assert flags[0].kind == RATE_CUT
    assert flags[0].label == "RATE CUT"


def test_rate_cut_ignores_small_decay():
    candles = _candles([0.100, 0.099, 0.098, 0.097], window=1)
    assert detect_rate_cuts(candles, min_drop_frac=0.10) == []


def test_rate_cut_ignores_increase():
    candles = _candles([0.05, 0.10], window=1)
    assert detect_rate_cuts(candles, min_drop_frac=0.10) == []


def test_flash_crash_exact_threshold_not_flagged():
    # change == threshold exactly must NOT fire: detect_flash_crashes uses a
    # strict `>`, not `>=`. Values chosen as exact powers of two so the ratio
    # lands precisely on 0.25 in IEEE754, not just approximately.
    candles = _candles([4.0, 5.0], window=1)
    assert detect_flash_crashes(candles, threshold=0.25) == []


def test_flash_crash_zero_prev_close_does_not_raise():
    candles = _candles([0.0, 5.0], window=1)
    assert detect_flash_crashes(candles, threshold=0.30) == []


def test_rate_cut_exact_threshold_not_flagged():
    # Same exact-boundary reasoning as the flash-crash case above.
    candles = _candles([4.0, 3.0], window=1)
    assert detect_rate_cuts(candles, min_drop_frac=0.25) == []


def test_rate_cut_zero_prev_close_does_not_raise():
    candles = _candles([0.0, -1.0], window=1)
    assert detect_rate_cuts(candles, min_drop_frac=0.10) == []
