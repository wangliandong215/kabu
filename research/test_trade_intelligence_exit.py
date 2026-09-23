"""
Unit tests for research/trade_intelligence_exit.py against a throwaway temp
trade_history.db built via engine.trade_tracker.TradeTracker.

Run:  python -m unittest research.test_trade_intelligence_exit -v
"""
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from engine.trade_tracker import TradeTracker
from research import trade_intelligence_data as data
from research import trade_intelligence_exit as exit_analysis

_ENTRY_TS = "2026-01-01T00:00:00"


class TradeIntelligenceExitTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.source_db = Path(self._tmpdir.name) / "fake_trade_history.db"
        self.tracker = TradeTracker(self.source_db)

    def tearDown(self):
        self.tracker.close()
        self._tmpdir.cleanup()

    def _make_trade(self, trade_id, ticker, entry_price, exit_price, high,
                     exit_reason_raw, strategy_name, holding_hours):
        self.tracker.log_entry(
            trade_id=trade_id, ticker=ticker, strategy_name=strategy_name,
            strategy_version="test", direction="LONG", price=entry_price, shares=10,
            position_value=entry_price * 10, position_pct=0.1, cash=9000.0,
            equity=10000.0, timestamp=_ENTRY_TS, execution="REAL",
        )
        self.tracker.update_position_metrics(
            trade_id=trade_id, date="2026-01-01", close=exit_price,
            entry_price=entry_price, shares=10, high=high, low=min(exit_price, entry_price) - 1)
        exit_ts = (datetime.fromisoformat(_ENTRY_TS) + timedelta(hours=holding_hours)).isoformat()
        self.tracker.log_exit(
            trade_id=trade_id, price=exit_price, cash=1.0, equity=1.0,
            timestamp=exit_ts, exit_reason=exit_reason_raw)
        self.tracker.log_exit_diagnostics(
            trade_id=trade_id, stop_loss_triggered=(exit_reason_raw == "STOP_LOSS"),
            strategy_exit_triggered=exit_reason_raw.startswith("STRATEGY_EXIT"))

    def _build_fixture(self):
        # 10 STOP_LOSS losers: entry 100 -> exit 90, tiny MFE (EARLY_FAILURE),
        # held 12h (<2d band), strategy "strat_a".
        for i in range(10):
            self._make_trade(f"loss{i}", f"US.L{i}", entry_price=100.0, exit_price=90.0,
                              high=101.0, exit_reason_raw="STOP_LOSS",
                              strategy_name="strat_a", holding_hours=12)
        # 10 TAKE_PROFIT winners: entry 100 -> exit 112, big MFE (PROFIT_GIVEBACK),
        # held 72h (2-5d band), strategy "strat_b".
        for i in range(10):
            self._make_trade(f"win{i}", f"US.W{i}", entry_price=100.0, exit_price=112.0,
                              high=113.0, exit_reason_raw="TAKE_PROFIT",
                              strategy_name="strat_b", holding_hours=72)

    def _patterns_by_key(self):
        df = data.load_dataset(db_path=self.source_db, execution="REAL")
        return {c.pattern_key: c for c in exit_analysis.analyze(df)}

    def test_pnl_pct_by_exit_reason_matches_hand_computed_means(self):
        self._build_fixture()
        by_key = self._patterns_by_key()
        self.assertIn("pnlpct_by_exit_reason_EXIT_STOP_LOSS", by_key)
        self.assertIn("pnlpct_by_exit_reason_EXIT_TAKE_PROFIT", by_key)
        self.assertAlmostEqual(
            by_key["pnlpct_by_exit_reason_EXIT_STOP_LOSS"].metric_value, -0.10, places=6)
        self.assertAlmostEqual(
            by_key["pnlpct_by_exit_reason_EXIT_TAKE_PROFIT"].metric_value, 0.12, places=6)
        self.assertEqual(by_key["pnlpct_by_exit_reason_EXIT_STOP_LOSS"].n_sample, 10)
        self.assertEqual(by_key["pnlpct_by_exit_reason_EXIT_TAKE_PROFIT"].n_sample, 10)

    def test_holding_days_bands_match_expected_win_rates(self):
        self._build_fixture()
        by_key = self._patterns_by_key()
        self.assertIn("holding_days_<2d", by_key)
        self.assertIn("holding_days_2-5d", by_key)
        self.assertAlmostEqual(by_key["holding_days_<2d"].metric_value, 0.0)
        self.assertAlmostEqual(by_key["holding_days_2-5d"].metric_value, 1.0)
        self.assertEqual(by_key["holding_days_<2d"].state, "OBSERVATION")

    def test_strategy_early_failure_rate_cross_tab(self):
        self._build_fixture()
        by_key = self._patterns_by_key()
        self.assertIn("strategy_strat_a_early_failure_rate", by_key)
        self.assertIn("strategy_strat_b_early_failure_rate", by_key)
        self.assertAlmostEqual(by_key["strategy_strat_a_early_failure_rate"].metric_value, 1.0)
        self.assertAlmostEqual(by_key["strategy_strat_b_early_failure_rate"].metric_value, 0.0)

    def test_single_strategy_produces_no_cross_tab(self):
        for i in range(10):
            self._make_trade(f"only{i}", f"US.O{i}", entry_price=100.0, exit_price=90.0,
                              high=101.0, exit_reason_raw="STOP_LOSS",
                              strategy_name="only_strat", holding_hours=12)
        by_key = self._patterns_by_key()
        self.assertFalse(any(k.startswith("strategy_") for k in by_key))


if __name__ == "__main__":
    unittest.main()
