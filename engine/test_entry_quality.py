"""
Unit tests for engine/entry_quality.py.
Run:  python -m unittest engine.test_entry_quality -v
"""
import unittest

import numpy as np
import pandas as pd

from engine.entry_quality import (build_entry_quality_snapshot,
                                   compute_forward_returns)


def _flat_ohlcv(n=90, price=100.0, volume=1000.0):
    """A boring, flat-price df — high==low==close==price every bar, so
    distance-from-20d-high/EMA computations have an exact, easy-to-assert
    expected value of 0."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "high": price, "low": price, "close": price, "volume": volume,
    }, index=idx)


class TestBuildEntryQualitySnapshot(unittest.TestCase):

    def test_none_df_returns_none(self):
        self.assertIsNone(build_entry_quality_snapshot(None, entry_price=100.0))

    def test_empty_df_returns_none(self):
        self.assertIsNone(build_entry_quality_snapshot(pd.DataFrame(), entry_price=100.0))

    def test_missing_entry_price_returns_none(self):
        self.assertIsNone(build_entry_quality_snapshot(_flat_ohlcv(), entry_price=None))
        self.assertIsNone(build_entry_quality_snapshot(_flat_ohlcv(), entry_price=0.0))

    def test_flat_price_distances_are_zero(self):
        snap = build_entry_quality_snapshot(_flat_ohlcv(), entry_price=100.0,
                                            donchian_breakout_price=100.0,
                                            atr_at_entry=2.0)
        self.assertAlmostEqual(snap["distance_from_20d_high"], 0.0)
        self.assertAlmostEqual(snap["distance_from_20d_ema"], 0.0)
        self.assertAlmostEqual(snap["entry_day_return"], 0.0)
        self.assertAlmostEqual(snap["distance_from_breakout_atr"], 0.0)

    def test_distance_from_breakout_atr_formula(self):
        # entry 106, breakout 100, ATR 2 -> (106-100)/2 = 3.0 ATR extended
        snap = build_entry_quality_snapshot(_flat_ohlcv(price=100.0), entry_price=106.0,
                                            donchian_breakout_price=100.0,
                                            atr_at_entry=2.0)
        self.assertAlmostEqual(snap["distance_from_breakout_atr"], 3.0)

    def test_missing_donchian_price_leaves_distance_none(self):
        snap = build_entry_quality_snapshot(_flat_ohlcv(), entry_price=106.0,
                                            donchian_breakout_price=None,
                                            atr_at_entry=2.0)
        self.assertIsNone(snap["distance_from_breakout_atr"])
        self.assertIsNone(snap["donchian_breakout_price"])

    def test_missing_atr_leaves_distance_none_not_zero_division(self):
        snap = build_entry_quality_snapshot(_flat_ohlcv(), entry_price=106.0,
                                            donchian_breakout_price=100.0,
                                            atr_at_entry=None)
        self.assertIsNone(snap["distance_from_breakout_atr"])

    def test_zero_atr_does_not_raise_and_leaves_distance_none(self):
        snap = build_entry_quality_snapshot(_flat_ohlcv(), entry_price=106.0,
                                            donchian_breakout_price=100.0,
                                            atr_at_entry=0.0)
        self.assertIsNotNone(snap)
        self.assertIsNone(snap["distance_from_breakout_atr"])

    def test_short_df_leaves_20d_fields_none_but_other_fields_present(self):
        # Only 5 bars — below the 20-bar lookback for high/EMA/volume_ratio.
        snap = build_entry_quality_snapshot(_flat_ohlcv(n=5), entry_price=101.0,
                                            donchian_breakout_price=100.0,
                                            atr_at_entry=2.0)
        self.assertIsNotNone(snap)
        self.assertIsNone(snap["distance_from_20d_high"])
        self.assertIsNone(snap["distance_from_20d_ema"])
        self.assertIsNone(snap["entry_volume_ratio"])
        self.assertIsNotNone(snap["entry_day_return"])   # only needs 1 bar
        self.assertAlmostEqual(snap["distance_from_breakout_atr"], 0.5)

    def test_no_volume_column_leaves_volume_ratio_none(self):
        df = _flat_ohlcv().drop(columns=["volume"])
        snap = build_entry_quality_snapshot(df, entry_price=100.0)
        self.assertIsNone(snap["entry_volume_ratio"])

    def test_volume_spike_reflected_in_ratio(self):
        df = _flat_ohlcv(n=90, volume=1000.0)
        df.iloc[-1, df.columns.get_loc("volume")] = 5000.0
        snap = build_entry_quality_snapshot(df, entry_price=100.0)
        self.assertGreater(snap["entry_volume_ratio"], 1.0)

    def test_rule_score_and_confidence_score_pass_through(self):
        snap = build_entry_quality_snapshot(_flat_ohlcv(), entry_price=100.0,
                                            rule_score=87.5, confidence_score=None)
        self.assertEqual(snap["rule_score"], 87.5)
        self.assertIsNone(snap["confidence_score"])

    def test_atr_expansion_ratio_needs_60_bars(self):
        snap_short = build_entry_quality_snapshot(_flat_ohlcv(n=59), entry_price=100.0,
                                                   atr_at_entry=2.0)
        self.assertIsNone(snap_short["atr_expansion_ratio"])
        snap_long = build_entry_quality_snapshot(_flat_ohlcv(n=90), entry_price=100.0,
                                                  atr_at_entry=2.0)
        # Flat price -> true range/ATR is 0 -> baseline 0 -> guarded, stays None
        self.assertIsNone(snap_long["atr_expansion_ratio"])

    def test_atr_expansion_ratio_detects_expansion(self):
        n = 90
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        close = np.full(n, 100.0)
        # Blow the range wide open on the last (entry) bar only.
        high[-1] = 120.0
        low[-1] = 80.0
        df = pd.DataFrame({"high": high, "low": low, "close": close,
                           "volume": 1000.0}, index=idx)
        snap = build_entry_quality_snapshot(df, entry_price=100.0, atr_at_entry=6.0)
        self.assertIsNotNone(snap["atr_expansion_ratio"])
        self.assertGreater(snap["atr_expansion_ratio"], 1.0)

    def test_never_raises_on_malformed_df(self):
        bad_df = pd.DataFrame({"close": [1, 2, 3]})   # missing high/low
        self.assertIsNone(build_entry_quality_snapshot(bad_df, entry_price=100.0))


class TestComputeForwardReturns(unittest.TestCase):

    def _df_after(self, n, start=100.0, daily_return=0.0):
        idx = pd.date_range("2024-02-01", periods=n, freq="D")
        close = start * (1 + daily_return) ** np.arange(1, n + 1)
        return pd.DataFrame({
            "high": close * 1.01, "low": close * 0.99, "close": close,
        }, index=idx)

    def test_none_df_returns_none(self):
        self.assertIsNone(compute_forward_returns(None, entry_price=100.0))

    def test_empty_df_returns_none(self):
        self.assertIsNone(compute_forward_returns(pd.DataFrame(), entry_price=100.0))

    def test_missing_entry_price_returns_none(self):
        self.assertIsNone(compute_forward_returns(self._df_after(10), entry_price=None))

    def test_partial_history_only_fills_reached_horizons(self):
        # Only 8 bars after entry -> 5d reachable, 10d/20d/60d not yet.
        result = compute_forward_returns(self._df_after(8), entry_price=100.0)
        self.assertIsNotNone(result["return_5d"])
        self.assertIsNone(result["return_10d"])
        self.assertIsNone(result["return_20d"])
        self.assertIsNone(result["return_60d"])
        self.assertEqual(result["forward_bars_available"], 8)

    def test_full_history_fills_every_horizon(self):
        result = compute_forward_returns(self._df_after(90, daily_return=0.001),
                                          entry_price=100.0)
        for key in ("return_5d", "return_10d", "return_20d", "return_60d"):
            self.assertIsNotNone(result[key])
        self.assertEqual(result["forward_bars_available"], 60)   # capped at 60

    def test_return_5d_value(self):
        df = self._df_after(10)
        # Force a known 5th-bar close.
        df.iloc[4, df.columns.get_loc("close")] = 110.0
        result = compute_forward_returns(df, entry_price=100.0)
        self.assertAlmostEqual(result["return_5d"], 0.10)

    def test_mfe_mae_over_available_window(self):
        n = 10
        idx = pd.date_range("2024-02-01", periods=n, freq="D")
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        close = np.full(n, 100.0)
        high[3] = 130.0   # best favorable excursion
        low[6] = 70.0      # worst adverse excursion
        df = pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)
        result = compute_forward_returns(df, entry_price=100.0)
        self.assertAlmostEqual(result["max_favorable_excursion"], 0.30)
        self.assertAlmostEqual(result["max_adverse_excursion"], -0.30)

    def test_never_raises_on_malformed_df(self):
        bad_df = pd.DataFrame({"close": ["not", "a", "number"]})
        self.assertIsNone(compute_forward_returns(bad_df, entry_price=100.0))


if __name__ == "__main__":
    unittest.main()
