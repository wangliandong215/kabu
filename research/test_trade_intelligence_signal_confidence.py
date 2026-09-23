"""
Unit tests for research/trade_intelligence_signal_confidence.py against a
throwaway temp trade_history.db built via engine.trade_tracker.TradeTracker.

Run:  python -m unittest research.test_trade_intelligence_signal_confidence -v
"""
import tempfile
import unittest
from pathlib import Path

from engine.trade_tracker import TradeTracker
from research import trade_intelligence_data as data
from research import trade_intelligence_signal_confidence as signal_confidence

_ENTRY_TS = "2026-01-01T00:00:00"


class TradeIntelligenceSignalConfidenceTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.source_db = Path(self._tmpdir.name) / "fake_trade_history.db"
        self.tracker = TradeTracker(self.source_db)

    def tearDown(self):
        self.tracker.close()
        self._tmpdir.cleanup()

    def _make_trade(self, trade_id, ticker, exit_price, confidence_score):
        self.tracker.log_entry(
            trade_id=trade_id, ticker=ticker, strategy_name="atr_breakout",
            strategy_version="test", direction="LONG", price=100.0, shares=10,
            position_value=1000.0, position_pct=0.1, cash=9000.0, equity=10000.0,
            timestamp=_ENTRY_TS, execution="REAL",
        )
        self.tracker.log_confidence_score(trade_id=trade_id, confidence_score=confidence_score)
        self.tracker.update_position_metrics(
            trade_id=trade_id, date="2026-01-01", close=exit_price,
            entry_price=100.0, shares=10, high=max(exit_price, 101.0), low=95.0)
        self.tracker.log_exit(
            trade_id=trade_id, price=exit_price, cash=1.0, equity=1.0,
            timestamp="2026-01-01T12:00:00", exit_reason="STRATEGY_EXIT(atr_breakout)")

    def _patterns_by_key(self):
        df = data.load_dataset(db_path=self.source_db, execution="REAL")
        return {c.pattern_key: c for c in signal_confidence.analyze(df)}

    def test_confidence_band_win_rate_matches_hand_computed_values(self):
        # Low band (<40): all losers. High band (>=60): all winners.
        for i in range(10):
            self._make_trade(f"low{i}", f"US.L{i}", exit_price=90.0, confidence_score=20.0)
        for i in range(10):
            self._make_trade(f"high{i}", f"US.H{i}", exit_price=110.0, confidence_score=80.0)

        by_key = self._patterns_by_key()
        self.assertIn("confidence_score_band_<40", by_key)
        self.assertIn("confidence_score_band_>=60", by_key)
        self.assertAlmostEqual(by_key["confidence_score_band_<40"].metric_value, 0.0)
        self.assertAlmostEqual(by_key["confidence_score_band_>=60"].metric_value, 1.0)

    def test_correlation_is_always_observation_even_with_strong_signal(self):
        for i in range(20):
            score = 10.0 + i * 4.0   # 10..86, strictly increasing
            exit_price = 90.0 + i * 2.0  # monotonically increasing with score -> strong r
            self._make_trade(f"t{i}", f"US.T{i}", exit_price=exit_price, confidence_score=score)

        by_key = self._patterns_by_key()
        self.assertIn("confidence_score_correlation", by_key)
        corr_pattern = by_key["confidence_score_correlation"]
        self.assertEqual(corr_pattern.state, "OBSERVATION")
        self.assertGreater(corr_pattern.metric_value, 0.9)  # strong positive correlation by construction

    def test_missing_component_column_is_skipped_without_error(self):
        # historical_win_rate never logged -> its correlation candidate must
        # simply be absent, not raise.
        for i in range(6):
            self._make_trade(f"t{i}", f"US.T{i}", exit_price=105.0, confidence_score=50.0)
        by_key = self._patterns_by_key()
        self.assertNotIn("historical_win_rate_correlation", by_key)


if __name__ == "__main__":
    unittest.main()
