"""
Unit tests for research/trade_intelligence_position_size.py against a
throwaway temp trade_history.db built via engine.trade_tracker.TradeTracker.

Run:  python -m unittest research.test_trade_intelligence_position_size -v
"""
import tempfile
import unittest
from pathlib import Path

from engine.trade_tracker import TradeTracker
from research import trade_intelligence_data as data
from research import trade_intelligence_position_size as position_size

_ENTRY_TS = "2026-01-01T00:00:00"


class TradeIntelligencePositionSizeTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.source_db = Path(self._tmpdir.name) / "fake_trade_history.db"
        self.tracker = TradeTracker(self.source_db)

    def tearDown(self):
        self.tracker.close()
        self._tmpdir.cleanup()

    def _make_trade(self, trade_id, ticker, position_pct, exit_price,
                     skip_reason=None):
        self.tracker.log_entry(
            trade_id=trade_id, ticker=ticker, strategy_name="atr_breakout",
            strategy_version="test", direction="LONG", price=100.0, shares=10,
            position_value=1000.0, position_pct=position_pct, cash=9000.0,
            equity=10000.0, timestamp=_ENTRY_TS, execution="REAL",
        )
        if skip_reason is not None:
            self.tracker.log_confidence_score(trade_id=trade_id, skip_reason=skip_reason)
        self.tracker.update_position_metrics(
            trade_id=trade_id, date="2026-01-01", close=exit_price,
            entry_price=100.0, shares=10, high=max(exit_price, 101.0), low=95.0)
        self.tracker.log_exit(
            trade_id=trade_id, price=exit_price, cash=1.0, equity=1.0,
            timestamp="2026-01-01T12:00:00", exit_reason="STRATEGY_EXIT(atr_breakout)")

    def _patterns_by_key(self):
        df = data.load_dataset(db_path=self.source_db, execution="REAL")
        return {c.pattern_key: c for c in position_size.analyze(df)}

    def test_position_pct_band_avg_pnl_pct_matches_hand_computed_values(self):
        # Small band (<0.05): losers. Large band (>=0.15): winners.
        for i in range(10):
            self._make_trade(f"small{i}", f"US.S{i}", position_pct=0.03, exit_price=90.0)
        for i in range(10):
            self._make_trade(f"large{i}", f"US.L{i}", position_pct=0.20, exit_price=110.0)

        by_key = self._patterns_by_key()
        self.assertIn("position_pct_band_<0.05_avg_pnl_pct", by_key)
        self.assertIn("position_pct_band_>=0.15_avg_pnl_pct", by_key)
        self.assertAlmostEqual(
            by_key["position_pct_band_<0.05_avg_pnl_pct"].metric_value, -0.10, places=6)
        self.assertAlmostEqual(
            by_key["position_pct_band_>=0.15_avg_pnl_pct"].metric_value, 0.10, places=6)

    def test_skip_reason_frequency_is_always_observation(self):
        for i in range(6):
            self._make_trade(f"skip{i}", f"US.SK{i}", position_pct=0.05, exit_price=101.0,
                              skip_reason="LOW_CONFIDENCE")
        for i in range(2):  # below MIN_OBSERVATION_N -> must be omitted
            self._make_trade(f"rare{i}", f"US.RA{i}", position_pct=0.05, exit_price=101.0,
                              skip_reason="RARE_REASON")

        by_key = self._patterns_by_key()
        self.assertIn("skip_reason_LOW_CONFIDENCE", by_key)
        pattern = by_key["skip_reason_LOW_CONFIDENCE"]
        self.assertEqual(pattern.state, "OBSERVATION")
        self.assertEqual(pattern.n_sample, 6)
        self.assertNotIn("skip_reason_RARE_REASON", by_key)


if __name__ == "__main__":
    unittest.main()
