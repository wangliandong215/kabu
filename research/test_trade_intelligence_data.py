"""
Unit tests for research/trade_intelligence_data.py against a throwaway temp
trade_history.db built via engine.trade_tracker.TradeTracker. Never touches
the production DB.

Run:  python -m unittest research.test_trade_intelligence_data -v
"""
import tempfile
import unittest
from pathlib import Path

from engine.trade_tracker import TradeTracker
from research import trade_intelligence_data as data


class TradeIntelligenceDataTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.source_db = Path(self._tmpdir.name) / "fake_trade_history.db"
        self.tracker = TradeTracker(self.source_db)

    def tearDown(self):
        self.tracker.close()
        self._tmpdir.cleanup()

    def _open_trade(self, trade_id, ticker="US.AAA", execution="REAL",
                     equity=10000.0, confidence_score=55.0):
        self.tracker.log_entry(
            trade_id=trade_id, ticker=ticker, strategy_name="atr_breakout",
            strategy_version="test", direction="LONG", price=100.0, shares=10,
            position_value=1000.0, position_pct=0.1, cash=9000.0, equity=equity,
            timestamp="2026-01-01T00:00:00", confidence_score=confidence_score,
            execution=execution,
        )

    def _close_trade(self, trade_id, price=105.0):
        self.tracker.update_position_metrics(
            trade_id=trade_id, date="2026-01-01", close=102.0,
            entry_price=100.0, shares=10, high=103.0, low=99.0)
        self.tracker.update_position_metrics(
            trade_id=trade_id, date="2026-01-02", close=105.0,
            entry_price=100.0, shares=10, high=106.0, low=101.0)
        self.tracker.log_exit(
            trade_id=trade_id, price=price, cash=1.0, equity=1.0,
            timestamp="2026-01-02T12:00:00", exit_reason="STRATEGY_EXIT(atr_breakout)")
        self.tracker.log_exit_diagnostics(trade_id=trade_id, strategy_exit_triggered=True)

    def test_load_closed_trades_excludes_open_and_phantom_rows(self):
        # t1: normal closed REAL trade
        self._open_trade("t1")
        self._close_trade("t1")
        # t2: closed PAPER trade
        self._open_trade("t2", execution="PAPER")
        self._close_trade("t2")
        # t3: still open (no exit) REAL trade
        self._open_trade("t3")
        # t4: closed REAL trade but a phantom $50k-equity_before row
        self._open_trade("t4", equity=50000.0)
        self._close_trade("t4")

        real_only = data.load_closed_trades(db_path=self.source_db, execution="REAL")
        self.assertEqual(sorted(real_only["trade_id"]), ["t1"])

        mixed = data.load_closed_trades(db_path=self.source_db, execution=None)
        self.assertEqual(sorted(mixed["trade_id"]), ["t1", "t2"])

    def test_load_dataset_merges_trajectory_columns(self):
        self._open_trade("t1")
        self._close_trade("t1")
        df = data.load_dataset(db_path=self.source_db, execution="REAL")
        self.assertEqual(len(df), 1)
        self.assertIn("MFE_at_day_1", df.columns)
        self.assertIn("MFE_at_day_2", df.columns)
        row = df.iloc[0]
        self.assertAlmostEqual(row["MFE_at_day_1"], 0.03, places=6)  # (103-100)*10/1000

    def test_load_dataset_empty_when_no_trades(self):
        df = data.load_dataset(db_path=self.source_db, execution="REAL")
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
