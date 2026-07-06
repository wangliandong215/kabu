"""
Unit tests for engine/regime_features.py.
Run:  python -m unittest engine.test_regime_features -v
"""
import unittest

import numpy as np
import pandas as pd

from engine.regime_features import (adx_series, atr_pct_series, atr_series,
                                     bb_width_series, extract_features,
                                     volume_ratio_series)


def _flat_ohlcv(n=30, price=100.0, half_range=1.0, volume=1000.0):
    """Constant daily true range (high-low == 2*half_range), flat close —
    gives an exactly predictable ATR."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "high": price + half_range,
        "low": price - half_range,
        "close": price,
        "volume": volume,
    }, index=idx)


def _zigzag_ohlcv(n=30, price=100.0, amplitude=1.0, volume=1000.0):
    """Alternating up/down moves of equal magnitude — no net directional
    drift (low ADX), non-zero true range (defined, non-NaN ADX)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    offsets = np.array([amplitude if i % 2 == 0 else -amplitude for i in range(n)])
    close = price + offsets
    return pd.DataFrame({
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
        "volume": volume,
    }, index=idx)


def _trend_ohlcv(n=30, start=100.0, step=1.0, half_range=0.1, volume=1000.0):
    """Monotonically increasing close, every day a fresh high -> strongly
    trending (ADX approaches 100)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = start + step * np.arange(n)
    return pd.DataFrame({
        "high": close + half_range,
        "low": close - half_range,
        "close": close,
        "volume": volume,
    }, index=idx)


class TestAtrSeries(unittest.TestCase):

    def test_constant_true_range_gives_exact_atr(self):
        df = _flat_ohlcv(n=20, half_range=1.0)
        atr = atr_series(df["high"], df["low"], df["close"], period=14)
        # TR is exactly 2.0 every bar (high-low, no gaps) -> Wilder ewm of a
        # constant series stays exactly at that constant.
        self.assertTrue(np.allclose(atr.dropna(), 2.0))

    def test_atr_pct_matches_atr_over_close(self):
        df = _flat_ohlcv(n=20, price=50.0, half_range=0.5)
        atr = atr_series(df["high"], df["low"], df["close"], period=14)
        atr_pct = atr_pct_series(df["high"], df["low"], df["close"], period=14)
        expected = atr / df["close"]
        pd.testing.assert_series_equal(atr_pct, expected, check_names=False)


class TestAdxSeries(unittest.TestCase):

    def test_flat_price_gives_no_directional_movement_not_inf(self):
        df = _flat_ohlcv(n=20)
        adx = adx_series(df["high"], df["low"], df["close"], period=14)
        # zero +DM and -DM -> 0/0 -> NaN, never inf/exploding.
        self.assertFalse(np.isinf(adx.dropna()).any())

    def test_strong_uptrend_gives_high_adx(self):
        df = _trend_ohlcv(n=60)
        adx = adx_series(df["high"], df["low"], df["close"], period=14)
        self.assertGreater(adx.iloc[-1], 25.0)

    def test_zigzag_gives_low_adx(self):
        df = _zigzag_ohlcv(n=60, amplitude=1.0)
        adx = adx_series(df["high"], df["low"], df["close"], period=14)
        self.assertLess(adx.iloc[-1], 25.0)


class TestBbWidthSeries(unittest.TestCase):

    def test_constant_price_gives_zero_width(self):
        df = _flat_ohlcv(n=30, price=100.0)
        width = bb_width_series(df["close"], period=20, num_std=2.0)
        self.assertTrue(np.allclose(width.dropna(), 0.0))

    def test_volatile_price_gives_positive_width(self):
        df = _zigzag_ohlcv(n=30, amplitude=5.0)
        width = bb_width_series(df["close"], period=20, num_std=2.0)
        self.assertGreater(width.iloc[-1], 0.0)


class TestVolumeRatioSeries(unittest.TestCase):

    def test_constant_volume_gives_ratio_one(self):
        volume = pd.Series([1000.0] * 30)
        ratio = volume_ratio_series(volume, ma_period=20)
        self.assertTrue(np.allclose(ratio.dropna(), 1.0))

    def test_volume_spike_gives_ratio_above_one(self):
        volume = pd.Series([1000.0] * 29 + [5000.0])
        ratio = volume_ratio_series(volume, ma_period=20)
        self.assertGreater(ratio.iloc[-1], 1.0)


class TestExtractFeatures(unittest.TestCase):

    def test_returns_expected_columns(self):
        df = _trend_ohlcv(n=40)
        out = extract_features(df)
        self.assertEqual(list(out.columns),
                          ["adx", "atr", "atr_pct", "bb_width", "volume_ratio"])
        self.assertEqual(len(out), len(df))

    def test_missing_volume_column_gives_nan_ratio(self):
        df = _trend_ohlcv(n=40).drop(columns=["volume"])
        out = extract_features(df)
        self.assertTrue(out["volume_ratio"].isna().all())


if __name__ == "__main__":
    unittest.main()
