"""
Unit tests for engine/market_regime.py.
Run:  python -m unittest engine.test_market_regime -v
"""
import unittest

import numpy as np
import pandas as pd

import config
from engine.market_regime import (Regime, RegimeClassifier,
                                   RuleBasedRegimeClassifier, _axis_confidence,
                                   classify, classify_series)

N = 90  # comfortably past the classifier's min-bars warmup gate


def _trend_df(n=N, start=100.0, daily_return=0.01, vol_pct=0.002, volume=1000.0):
    """Monotonic compounding uptrend. `vol_pct` controls intraday range as a
    fraction of price, independent of the trend itself, so ADX (trend axis)
    and ATR% (volatility axis) can be tuned separately."""
    close = start * (1 + daily_return) ** np.arange(n)
    half_range = close * vol_pct
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "high": close + half_range,
        "low": close - half_range,
        "close": close,
        "volume": volume,
    }, index=idx)


def _sideways_df(n=N, price=100.0, amp_pct=0.005, volume=1000.0):
    """Alternating up/down moves — no net drift (low ADX). `amp_pct`
    controls the swing size, i.e. the volatility axis."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    offsets = np.array([amp_pct if i % 2 == 0 else -amp_pct for i in range(n)])
    close = price * (1 + offsets)
    return pd.DataFrame({
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": volume,
    }, index=idx)


class TestRegimeClassification(unittest.TestCase):
    """One synthetic scenario per PRD-defined regime; parameters were probed
    empirically to land clearly on each side of MRD_ADX_TREND_MIN=25 /
    MRD_ATR_PCT_HIGH=0.03 with margin, not just barely crossing."""

    def test_low_vol_sideways(self):
        df = _sideways_df(amp_pct=0.005)
        snap = classify(df)
        self.assertEqual(snap.regime, Regime.LOW_VOL_SIDEWAYS)

    def test_high_vol_sideways(self):
        df = _sideways_df(amp_pct=0.05)
        snap = classify(df)
        self.assertEqual(snap.regime, Regime.HIGH_VOL_SIDEWAYS)

    def test_low_vol_trend(self):
        df = _trend_df(daily_return=0.01, vol_pct=0.002)
        snap = classify(df)
        self.assertEqual(snap.regime, Regime.LOW_VOL_TREND)

    def test_high_vol_trend(self):
        df = _trend_df(daily_return=0.01, vol_pct=0.02)
        snap = classify(df)
        self.assertEqual(snap.regime, Regime.HIGH_VOL_TREND)


class TestWarmup(unittest.TestCase):

    def test_short_history_is_unknown_not_a_guess(self):
        df = _trend_df(n=5)
        snap = classify(df)
        self.assertEqual(snap.regime, Regime.UNKNOWN)
        self.assertEqual(snap.confidence, 0.0)

    def test_warmup_rows_all_unknown_in_series(self):
        df = _trend_df(n=N)
        series = classify_series(df)
        min_bars = (max(config.MRD_ADX_PERIOD, config.MRD_ATR_PERIOD) * 3
                    + max(config.MRD_BB_PERIOD, config.MRD_VOLUME_MA_PERIOD))
        self.assertTrue((series["regime"].iloc[:min_bars] == Regime.UNKNOWN).all())
        self.assertTrue((series["regime"].iloc[min_bars:] != Regime.UNKNOWN).all())


class TestConfidence(unittest.TestCase):

    def test_confidence_bounded_0_100(self):
        for df in (_sideways_df(amp_pct=0.005), _sideways_df(amp_pct=0.05),
                   _trend_df(vol_pct=0.002), _trend_df(vol_pct=0.02)):
            snap = classify(df)
            self.assertGreaterEqual(snap.confidence, 0.0)
            self.assertLessEqual(snap.confidence, 100.0)

    def test_axis_confidence_is_50_at_the_boundary(self):
        # Exactly at threshold -> maximally uncertain which side it's on.
        value = pd.Series([25.0])
        self.assertEqual(_axis_confidence(value, threshold=25.0, scale=15.0).iloc[0], 50.0)

    def test_axis_confidence_grows_with_distance_from_threshold(self):
        threshold, scale = 25.0, 15.0
        near = _axis_confidence(pd.Series([26.0]), threshold, scale).iloc[0]
        far = _axis_confidence(pd.Series([40.0]), threshold, scale).iloc[0]
        self.assertGreater(far, near)
        self.assertGreaterEqual(far, near)  # far should reach the 100 ceiling
        self.assertEqual(far, 100.0)

    def test_axis_confidence_symmetric_around_threshold(self):
        threshold, scale = 0.03, 0.02
        below = _axis_confidence(pd.Series([0.02]), threshold, scale).iloc[0]
        above = _axis_confidence(pd.Series([0.04]), threshold, scale).iloc[0]
        self.assertAlmostEqual(below, above)

    def test_volume_confirmation_raises_confidence_on_trend(self):
        base = _trend_df(daily_return=0.01, vol_pct=0.02, volume=1000.0)

        confirmed = base.copy()
        confirmed.loc[confirmed.index[-1], "volume"] = 1000.0 * (config.MRD_VOLUME_RATIO_CONFIRM + 1)

        weak = base.copy()
        weak.loc[weak.index[-1], "volume"] = 1000.0 * (config.MRD_VOLUME_RATIO_WEAK - 0.1)

        neutral_snap = classify(base)
        confirmed_snap = classify(confirmed)
        weak_snap = classify(weak)

        self.assertEqual(neutral_snap.regime, Regime.HIGH_VOL_TREND)
        self.assertEqual(confirmed_snap.regime, Regime.HIGH_VOL_TREND)
        self.assertEqual(weak_snap.regime, Regime.HIGH_VOL_TREND)
        self.assertGreater(confirmed_snap.confidence, neutral_snap.confidence)
        self.assertLess(weak_snap.confidence, neutral_snap.confidence)


class TestClassifierInterface(unittest.TestCase):

    def test_rule_based_classifier_implements_interface(self):
        self.assertIsInstance(RuleBasedRegimeClassifier(), RegimeClassifier)

    def test_custom_classifier_can_be_injected(self):
        """Stand-in for a future stage-2 ML classifier — proves callers can
        swap the classifier without changing classify()/classify_series()."""
        class _AlwaysUnknown(RegimeClassifier):
            def classify_series(self, df):
                out = pd.DataFrame(index=df.index)
                out["regime"] = Regime.UNKNOWN
                out["regime_name"] = "Unknown"
                for col in ("adx", "atr", "atr_pct", "bb_width", "volume_ratio", "confidence"):
                    out[col] = 0.0
                return out

        snap = classify(_trend_df(), classifier=_AlwaysUnknown())
        self.assertEqual(snap.regime, Regime.UNKNOWN)


class TestGenericOverAnySymbol(unittest.TestCase):
    """classify()/classify_series() must not assume a specific data source —
    same call works whether the df represents a market index (e.g. QQQ) or a
    single stock."""

    def test_works_on_market_proxy_and_single_stock_alike(self):
        market_like = _trend_df(daily_return=0.005, vol_pct=0.01)
        stock_like = _sideways_df(amp_pct=0.02)

        market_snap = classify(market_like)
        stock_snap = classify(stock_like)

        self.assertIn(market_snap.regime, list(Regime))
        self.assertIn(stock_snap.regime, list(Regime))
        self.assertEqual(len(classify_series(market_like)), len(market_like))
        self.assertEqual(len(classify_series(stock_like)), len(stock_like))


if __name__ == "__main__":
    unittest.main()
