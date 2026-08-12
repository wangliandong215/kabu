"""
Unit tests for analyze_entry_quality.py.
Run:  python -m unittest test_analyze_entry_quality -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_entry_quality import (analyze_by_atr_distance,
                                    analyze_composite_overheat,
                                    analyze_overheat_dimension,
                                    backfill_forward_returns, main)
from engine.trade_tracker import TradeTracker


class AnalyzeEntryQualityTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "test_trade_history.db"
        self.tracker = TradeTracker(self.db_path)

    def tearDown(self):
        self.tracker.close()
        self._tmpdir.cleanup()

    def _seed_trade(self, trade_id, symbol="US.MSFT", entry_price=100.0,
                     distance_from_breakout_atr=0.2, **extra_quality):
        self.tracker.log_entry(
            trade_id=trade_id, ticker=symbol, strategy_name="atr_breakout",
            strategy_version="2.9", direction="LONG", price=entry_price,
            shares=10, position_value=entry_price * 10, position_pct=0.1,
            cash=9000.0, equity=10000.0, timestamp="2026-01-01T09:30:00",
            atr_entry=2.0,
        )
        fields = dict(distance_from_breakout_atr=distance_from_breakout_atr)
        fields.update(extra_quality)
        self.tracker.log_entry_quality(trade_id=trade_id, **fields)


class TestBackfillForwardReturns(AnalyzeEntryQualityTestCase):

    def _fake_fetch_kline(self, bars_map):
        def _fetch(code, ktype="K_DAY", bars=300):
            return bars_map.get(code)
        return _fetch

    def test_fills_return_columns_from_fake_fetch(self):
        self._seed_trade("t1", symbol="US.MSFT")
        idx = pd.date_range("2026-01-02", periods=10, freq="D")
        close = 100.0 * (1.01 ** np.arange(1, 11))
        bars_df = pd.DataFrame({
            "time_key": idx.strftime("%Y-%m-%d %H:%M:%S"),
            "high": close * 1.001, "low": close * 0.999, "close": close,
        })
        n = backfill_forward_returns(
            self.tracker, fetch_kline_fn=self._fake_fetch_kline({"US.MSFT": bars_df}))
        self.assertEqual(n, 1)
        row = self.tracker.query_entry_quality().iloc[0]
        self.assertIsNotNone(row["return_5d"])
        self.assertIsNone(row["return_20d"])   # only 10 forward bars available

    def test_skips_rows_already_complete_unless_forced(self):
        self._seed_trade("t1", symbol="US.MSFT")
        self.tracker.update_forward_returns(
            trade_id="t1", return_5d=0.01, return_10d=0.02, return_20d=0.03,
            return_60d=0.04, forward_bars_available=60)
        calls = []

        def _fetch(code, ktype="K_DAY", bars=300):
            calls.append(code)
            return None
        n = backfill_forward_returns(self.tracker, fetch_kline_fn=_fetch)
        self.assertEqual(n, 0)
        self.assertEqual(calls, [])   # never even attempted — already complete

    def test_force_recomputes_even_complete_rows(self):
        self._seed_trade("t1", symbol="US.MSFT")
        self.tracker.update_forward_returns(
            trade_id="t1", return_5d=0.01, forward_bars_available=60)
        calls = []

        def _fetch(code, ktype="K_DAY", bars=300):
            calls.append(code)
            return None
        backfill_forward_returns(self.tracker, fetch_kline_fn=_fetch, force=True)
        self.assertEqual(calls, ["US.MSFT"])

    def test_fetch_failure_is_skipped_not_raised(self):
        self._seed_trade("t1", symbol="US.MSFT")

        def _raises(code, ktype="K_DAY", bars=300):
            raise ConnectionError("no OpenD")
        n = backfill_forward_returns(self.tracker, fetch_kline_fn=_raises)
        self.assertEqual(n, 0)

    def test_empty_tracker_returns_zero(self):
        n = backfill_forward_returns(self.tracker, fetch_kline_fn=lambda *a, **k: None)
        self.assertEqual(n, 0)


class TestAnalyzeByAtrDistance(AnalyzeEntryQualityTestCase):

    def test_buckets_and_flags_insufficient_sample(self):
        # Two trades in "0~0.5 ATR", one in "1.5~2.0 ATR" -> both under
        # the default min_sample of 10, both should be flagged.
        for i in range(2):
            self._seed_trade(f"low{i}", distance_from_breakout_atr=0.2)
            self.tracker.update_forward_returns(
                trade_id=f"low{i}", return_20d=0.05, max_favorable_excursion=0.08,
                max_adverse_excursion=-0.01, forward_bars_available=20)
        self._seed_trade("high0", distance_from_breakout_atr=1.7)
        self.tracker.update_forward_returns(
            trade_id="high0", return_20d=-0.05, max_favorable_excursion=0.02,
            max_adverse_excursion=-0.08, forward_bars_available=20)

        df = self.tracker.query_entry_quality()
        stats = analyze_by_atr_distance(df, horizon_col="return_20d", min_sample=10)
        low_row = stats[stats["bucket"] == "0~0.5 ATR"].iloc[0]
        high_row = stats[stats["bucket"] == "1.5~2.0 ATR"].iloc[0]
        self.assertEqual(low_row["trade_count"], 2)
        self.assertEqual(low_row["flag"], "INSUFFICIENT_SAMPLE")
        self.assertEqual(high_row["trade_count"], 1)
        self.assertEqual(high_row["flag"], "INSUFFICIENT_SAMPLE")

    def test_meets_min_sample_clears_flag(self):
        for i in range(12):
            self._seed_trade(f"t{i}", distance_from_breakout_atr=0.3)
            self.tracker.update_forward_returns(
                trade_id=f"t{i}", return_20d=0.02 if i % 2 == 0 else -0.01,
                forward_bars_available=20)
        df = self.tracker.query_entry_quality()
        stats = analyze_by_atr_distance(df, horizon_col="return_20d", min_sample=10)
        row = stats[stats["bucket"] == "0~0.5 ATR"].iloc[0]
        self.assertEqual(row["flag"], "")
        self.assertEqual(row["sample_count"], 12)

    def test_negative_distance_bucketed_separately(self):
        self._seed_trade("neg", distance_from_breakout_atr=-0.3)
        df = self.tracker.query_entry_quality()
        stats = analyze_by_atr_distance(df, horizon_col="return_20d", min_sample=10)
        self.assertIn("< 0 ATR (filled below breakout)", stats["bucket"].tolist())

    def test_null_distance_bucketed_as_na(self):
        self.tracker.log_entry(
            trade_id="nodata", ticker="US.XYZ", strategy_name="combined",
            strategy_version="2.9", direction="LONG", price=50.0, shares=1,
            position_value=50.0, position_pct=0.01, cash=1.0, equity=1.0,
            timestamp="2026-01-01T00:00:00",
        )
        self.tracker.log_entry_quality(trade_id="nodata")   # no ATR distance known
        df = self.tracker.query_entry_quality()
        stats = analyze_by_atr_distance(df, horizon_col="return_20d", min_sample=10)
        self.assertIn("N/A", stats["bucket"].tolist())

    def test_profit_factor_computation(self):
        self._seed_trade("win", distance_from_breakout_atr=0.1)
        self.tracker.update_forward_returns(trade_id="win", return_20d=0.10,
                                            forward_bars_available=20)
        self._seed_trade("lose", distance_from_breakout_atr=0.1)
        self.tracker.update_forward_returns(trade_id="lose", return_20d=-0.05,
                                            forward_bars_available=20)
        df = self.tracker.query_entry_quality()
        stats = analyze_by_atr_distance(df, horizon_col="return_20d", min_sample=1)
        row = stats[stats["bucket"] == "0~0.5 ATR"].iloc[0]
        self.assertAlmostEqual(row["profit_factor"], 2.0)   # 0.10 / 0.05
        self.assertAlmostEqual(row["win_rate"], 0.5)


class TestOverheatAnalyses(AnalyzeEntryQualityTestCase):

    def test_dimension_analysis_needs_min_quantile_rows(self):
        self._seed_trade("only_one", entry_day_return=0.03)
        df = self.tracker.query_entry_quality()
        result = analyze_overheat_dimension(df, "entry_day_return", "return_20d",
                                            min_sample=10, quantiles=4)
        self.assertIsNone(result)   # 1 row < 4 quantiles

    def test_dimension_analysis_produces_buckets_with_enough_rows(self):
        for i in range(8):
            self._seed_trade(f"t{i}", entry_day_return=0.01 * i)
            self.tracker.update_forward_returns(
                trade_id=f"t{i}", return_20d=0.02, forward_bars_available=20)
        df = self.tracker.query_entry_quality()
        result = analyze_overheat_dimension(df, "entry_day_return", "return_20d",
                                            min_sample=10, quantiles=4)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
        self.assertTrue((result["flag"] == "INSUFFICIENT_SAMPLE").all())

    def test_composite_overheat_needs_dimension_data(self):
        self._seed_trade("only_one")   # no overheat dims populated at all
        df = self.tracker.query_entry_quality()
        result = analyze_composite_overheat(df, "return_20d", min_sample=10, quantiles=4)
        self.assertIsNone(result)

    def test_composite_overheat_produces_buckets(self):
        for i in range(8):
            self._seed_trade(f"t{i}", entry_day_return=0.01 * i,
                             atr_expansion_ratio=1.0 + 0.05 * i,
                             distance_from_20d_high=-0.01 * i)
            self.tracker.update_forward_returns(
                trade_id=f"t{i}", return_20d=0.02, forward_bars_available=20)
        df = self.tracker.query_entry_quality()
        result = analyze_composite_overheat(df, "return_20d", min_sample=10, quantiles=4)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)


class TestMainEntryPoint(AnalyzeEntryQualityTestCase):

    def test_runs_end_to_end_with_no_backfill_on_empty_db(self):
        rc = main(["--db-path", str(self.db_path), "--no-backfill"])
        self.assertEqual(rc, 0)

    def test_runs_end_to_end_with_data(self):
        for i in range(3):
            self._seed_trade(f"t{i}", distance_from_breakout_atr=0.2 * i)
            self.tracker.update_forward_returns(
                trade_id=f"t{i}", return_20d=0.02, forward_bars_available=20)
        self.tracker.close()   # release the file so main()'s own TradeTracker can open it
        rc = main(["--db-path", str(self.db_path), "--no-backfill", "--min-sample", "1"])
        self.assertEqual(rc, 0)
        self.tracker = TradeTracker(self.db_path)   # reopen for tearDown


if __name__ == "__main__":
    unittest.main()
