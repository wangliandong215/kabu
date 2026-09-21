"""
Unit tests for research/early_failure_monitor.py — refresh()/load_history()
against throwaway temp DBs on both ends (a fake trade_history.db built via
engine.trade_tracker.TradeTracker as the source, and a fake monitor DB as
the destination). Never touches the production DBs.

Run:  python -m unittest research.test_early_failure_monitor -v
"""
import tempfile
import unittest
from pathlib import Path

from engine.trade_tracker import TradeTracker
from research.early_failure_monitor import refresh, load_history


class EarlyFailureMonitorTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.source_db = Path(self._tmpdir.name) / "fake_trade_history.db"
        self.monitor_db = Path(self._tmpdir.name) / "fake_monitor.db"
        self.tracker = TradeTracker(self.source_db)

    def tearDown(self):
        self.tracker.close()
        self._tmpdir.cleanup()

    def _make_early_failure_trade(self, trade_id="ef1", ticker="US.AAA"):
        """Peaks tiny on day 1, grinds down every day after, closes via
        STRATEGY_EXIT well under the MFE_day2<1% bar."""
        self.tracker.log_entry(
            trade_id=trade_id, ticker=ticker, strategy_name="atr_breakout",
            strategy_version="test", direction="LONG", price=100.0, shares=10,
            position_value=1000.0, position_pct=0.1, cash=9000.0, equity=10000.0,
            timestamp="2026-01-01T00:00:00",
        )
        # day1: tiny favorable excursion, closes flat-ish
        self.tracker.update_position_metrics(
            trade_id=trade_id, date="2026-01-01", close=100.2,
            entry_price=100.0, shares=10, high=100.5, low=99.5)
        # day2: no new high, drifting down
        self.tracker.update_position_metrics(
            trade_id=trade_id, date="2026-01-02", close=98.0,
            entry_price=100.0, shares=10, high=100.3, low=97.5)
        # day3-5: keeps sliding
        for i, (d, c) in enumerate([("2026-01-03", 96.0), ("2026-01-04", 94.5),
                                     ("2026-01-05", 93.0)]):
            self.tracker.update_position_metrics(
                trade_id=trade_id, date=d, close=c,
                entry_price=100.0, shares=10, high=c + 0.3, low=c - 0.5)
        self.tracker.log_exit(
            trade_id=trade_id, price=93.0, cash=1.0, equity=1.0,
            timestamp="2026-01-05T12:00:00", exit_reason="STRATEGY_EXIT(atr_breakout)")
        self.tracker.log_exit_diagnostics(trade_id=trade_id, strategy_exit_triggered=True)

    def _make_winner_trade(self, trade_id="win1", ticker="US.BBB"):
        """Strong favorable move from day 1 on, gives back some but closes
        solidly positive -> PROFIT_GIVEBACK, well clear of the MFE_day2<1%
        bar (should never be a false positive)."""
        self.tracker.log_entry(
            trade_id=trade_id, ticker=ticker, strategy_name="atr_breakout",
            strategy_version="test", direction="LONG", price=100.0, shares=10,
            position_value=1000.0, position_pct=0.1, cash=9000.0, equity=10000.0,
            timestamp="2026-01-01T00:00:00",
        )
        for d, c, h in [("2026-01-01", 104.0, 105.0), ("2026-01-02", 108.0, 110.0),
                        ("2026-01-03", 112.0, 115.0), ("2026-01-04", 110.0, 116.0),
                        ("2026-01-05", 109.0, 116.0)]:
            self.tracker.update_position_metrics(
                trade_id=trade_id, date=d, close=c,
                entry_price=100.0, shares=10, high=h, low=c - 1.0)
        self.tracker.log_exit(
            trade_id=trade_id, price=109.0, cash=1.0, equity=1.0,
            timestamp="2026-01-05T12:00:00", exit_reason="STRATEGY_EXIT(atr_breakout)")
        self.tracker.log_exit_diagnostics(trade_id=trade_id, strategy_exit_triggered=True)

    def test_refresh_persists_trajectory_rows(self):
        self._make_early_failure_trade()
        self._make_winner_trade()
        df = refresh(db_path=self.monitor_db, source_db_path=self.source_db)
        self.assertEqual(len(df), 2)
        ef_row = df[df["trade_id"] == "ef1"].iloc[0]
        win_row = df[df["trade_id"] == "win1"].iloc[0]
        self.assertEqual(ef_row["exit_category"], "EARLY_FAILURE")
        self.assertEqual(win_row["exit_category"], "PROFIT_GIVEBACK")
        self.assertEqual(ef_row["mfe_day2_lt_1pct"], 1)
        self.assertEqual(win_row["mfe_day2_lt_1pct"], 0)

    def test_signal_summary_history_appends_one_row_per_refresh(self):
        self._make_early_failure_trade()
        self._make_winner_trade()
        refresh(db_path=self.monitor_db, source_db_path=self.source_db)
        refresh(db_path=self.monitor_db, source_db_path=self.source_db)
        hist = load_history(db_path=self.monitor_db)
        self.assertEqual(len(hist), 2, "two refresh() calls -> two history rows, append-only")
        self.assertEqual(hist.iloc[-1]["n_total"], 2)
        self.assertEqual(hist.iloc[-1]["n_early_failure"], 1)
        self.assertEqual(hist.iloc[-1]["n_winner"], 1)

    def test_mfe_day2_lt1pct_signal_stats_no_false_positive_in_fixture(self):
        self._make_early_failure_trade()
        self._make_winner_trade()
        refresh(db_path=self.monitor_db, source_db_path=self.source_db)
        hist = load_history(db_path=self.monitor_db)
        row = hist.iloc[-1]
        self.assertEqual(row["sig_mfe_d2_lt1_n_flagged"], 1)
        self.assertAlmostEqual(row["sig_mfe_d2_lt1_precision"], 1.0)
        self.assertAlmostEqual(row["sig_mfe_d2_lt1_recall"], 1.0)
        self.assertEqual(row["sig_mfe_d2_lt1_false_positives"], 0)

    def test_trade_trajectory_overwritten_not_duplicated_on_repeat_refresh(self):
        self._make_early_failure_trade()
        refresh(db_path=self.monitor_db, source_db_path=self.source_db)
        refresh(db_path=self.monitor_db, source_db_path=self.source_db)
        df = refresh(db_path=self.monitor_db, source_db_path=self.source_db)
        self.assertEqual(len(df), 1)   # not 3 — trade_trajectory is INSERT OR REPLACE

    def test_still_open_trade_excluded(self):
        self.tracker.log_entry(
            trade_id="open1", ticker="US.CCC", strategy_name="atr_breakout",
            strategy_version="test", direction="LONG", price=100.0, shares=10,
            position_value=1000.0, position_pct=0.1, cash=9000.0, equity=10000.0,
            timestamp="2026-01-01T00:00:00",
        )
        self.tracker.update_position_metrics(
            trade_id="open1", date="2026-01-01", close=101.0,
            entry_price=100.0, shares=10, high=102.0, low=99.0)
        df = refresh(db_path=self.monitor_db, source_db_path=self.source_db)
        self.assertEqual(len(df), 0)


if __name__ == "__main__":
    unittest.main()
