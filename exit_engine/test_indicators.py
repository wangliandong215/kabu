# -*- coding: utf-8 -*-
"""Unit tests for exit_engine/indicators.py — VWAP/EMA/ATR/trend/volume
helper correctness, plus missing-data/insufficient-bars/NaN edge cases.

Run:  python -m unittest exit_engine.test_indicators -v
"""
import unittest

import pandas as pd

from exit_engine import indicators
import test_support


def _bars(closes, highs=None, lows=None, volumes=None):
    n = len(closes)
    return pd.DataFrame({
        "open": closes,
        "high": highs if highs is not None else [c + 0.5 for c in closes],
        "low": lows if lows is not None else [c - 0.5 for c in closes],
        "close": closes,
        "volume": volumes if volumes is not None else [1000] * n,
    })


class VwapTestCase(unittest.TestCase):
    def test_vwap_matches_hand_computed_value(self):
        df = pd.DataFrame({
            "high": [10.0, 11.0], "low": [9.0, 10.0],
            "close": [9.5, 10.5], "volume": [100.0, 200.0],
        })
        # typical = (h+l+c)/3 -> [9.5, 10.5]; vwap = (9.5*100+10.5*200)/300
        expected = (9.5 * 100 + 10.5 * 200) / 300
        self.assertAlmostEqual(indicators.compute_vwap(df), expected, places=6)

    def test_vwap_none_on_empty_df(self):
        self.assertIsNone(indicators.compute_vwap(pd.DataFrame()))

    def test_vwap_none_on_missing_columns(self):
        self.assertIsNone(indicators.compute_vwap(pd.DataFrame({"close": [1.0]})))

    def test_vwap_none_on_zero_volume(self):
        df = _bars([10.0, 10.0], volumes=[0, 0])
        self.assertIsNone(indicators.compute_vwap(df))

    def test_vwap_none_on_none_input(self):
        self.assertIsNone(indicators.compute_vwap(None))


class EmaTestCase(unittest.TestCase):
    def test_ema_none_when_too_short(self):
        series = pd.Series([1.0, 2.0, 3.0])
        self.assertIsNone(indicators.compute_ema(series, span=5))

    def test_ema_matches_pandas_ewm(self):
        series = pd.Series([float(i) for i in range(1, 21)])
        expected = series.ewm(span=5, adjust=False).mean().iloc[-1]
        self.assertAlmostEqual(indicators.compute_ema(series, span=5), expected, places=9)

    def test_ema_none_on_none_series(self):
        self.assertIsNone(indicators.compute_ema(None, span=5))

    def test_ema_none_on_nonpositive_span(self):
        series = pd.Series([1.0, 2.0, 3.0])
        self.assertIsNone(indicators.ema_series(series, span=0))


class AtrTestCase(unittest.TestCase):
    def test_atr_none_when_too_short(self):
        df = _bars([10.0, 11.0])
        self.assertIsNone(indicators.atr_from_df(df, period=14))

    def test_atr_positive_on_sufficient_data(self):
        closes = [10.0 + (i % 3) for i in range(20)]
        df = _bars(closes)
        atr = indicators.atr_from_df(df, period=14)
        self.assertIsNotNone(atr)
        self.assertGreater(atr, 0)

    def test_atr_none_on_missing_columns(self):
        self.assertIsNone(indicators.atr_from_df(pd.DataFrame({"close": [1.0] * 20}), period=14))

    def test_atr_none_on_none_input(self):
        self.assertIsNone(indicators.atr_from_df(None))


class TrendDirectionTestCase(unittest.TestCase):
    def test_trend_up_on_rising_series(self):
        closes = [100.0 + i for i in range(30)]
        df = _bars(closes)
        self.assertEqual(indicators.trend_direction(df, fast=5, slow=20), "UP")

    def test_trend_down_on_falling_series(self):
        closes = [130.0 - i for i in range(30)]
        df = _bars(closes)
        self.assertEqual(indicators.trend_direction(df, fast=5, slow=20), "DOWN")

    def test_trend_none_when_too_short(self):
        df = _bars([100.0, 101.0, 102.0])
        self.assertIsNone(indicators.trend_direction(df, fast=5, slow=20))

    def test_trend_none_on_none_input(self):
        self.assertIsNone(indicators.trend_direction(None, fast=5, slow=20))

    def test_trend_none_on_missing_close_column(self):
        df = pd.DataFrame({"open": [1.0] * 30})
        self.assertIsNone(indicators.trend_direction(df, fast=5, slow=20))


class AvgVolumeTestCase(unittest.TestCase):
    def test_avg_volume_excludes_last_bar_by_default(self):
        df = _bars([1.0] * 5, volumes=[100, 100, 100, 100, 999999])
        avg = indicators.avg_volume(df, lookback=4, exclude_last=True)
        self.assertAlmostEqual(avg, 100.0, places=6)

    def test_avg_volume_includes_last_bar_when_requested(self):
        df = _bars([1.0] * 2, volumes=[100, 200])
        avg = indicators.avg_volume(df, lookback=2, exclude_last=False)
        self.assertAlmostEqual(avg, 150.0, places=6)

    def test_avg_volume_none_on_empty_df(self):
        self.assertIsNone(indicators.avg_volume(pd.DataFrame()))

    def test_avg_volume_none_on_none_input(self):
        self.assertIsNone(indicators.avg_volume(None))

    def test_avg_volume_none_when_all_zero(self):
        df = _bars([1.0, 1.0, 1.0], volumes=[0, 0, 0])
        self.assertIsNone(indicators.avg_volume(df, lookback=3, exclude_last=False))


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
