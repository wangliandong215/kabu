"""
Unit tests for research/trade_intelligence_early_failure.py against a
throwaway temp trade_history.db built via engine.trade_tracker.TradeTracker.

Run:  python -m unittest research.test_trade_intelligence_early_failure -v
"""
import tempfile
import unittest
from pathlib import Path

import config
from engine.trade_tracker import TradeTracker
from research import trade_intelligence_data as data
from research import trade_intelligence_early_failure as early_failure


class TradeIntelligenceEarlyFailureTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.source_db = Path(self._tmpdir.name) / "fake_trade_history.db"
        self.tracker = TradeTracker(self.source_db)

    def tearDown(self):
        self.tracker.close()
        self._tmpdir.cleanup()

    def _make_early_failure_trade(self, trade_id, ticker, confidence_score=30.0,
                                   regime_label=None):
        self.tracker.log_entry(
            trade_id=trade_id, ticker=ticker, strategy_name="atr_breakout",
            strategy_version="test", direction="LONG", price=100.0, shares=10,
            position_value=1000.0, position_pct=0.1, cash=9000.0, equity=10000.0,
            timestamp="2026-01-01T00:00:00", execution="REAL",
            regime_ctx={"regime": 1, "regime_label": regime_label, "confidence": 0.8}
            if regime_label else None,
        )
        self.tracker.log_confidence_score(trade_id=trade_id, confidence_score=confidence_score)
        self.tracker.update_position_metrics(
            trade_id=trade_id, date="2026-01-01", close=100.2,
            entry_price=100.0, shares=10, high=100.5, low=99.5)
        self.tracker.update_position_metrics(
            trade_id=trade_id, date="2026-01-02", close=98.0,
            entry_price=100.0, shares=10, high=100.3, low=97.5)
        for d, c in [("2026-01-03", 96.0), ("2026-01-04", 94.5), ("2026-01-05", 93.0)]:
            self.tracker.update_position_metrics(
                trade_id=trade_id, date=d, close=c, entry_price=100.0, shares=10,
                high=c + 0.3, low=c - 0.5)
        self.tracker.log_exit(
            trade_id=trade_id, price=93.0, cash=1.0, equity=1.0,
            timestamp="2026-01-05T12:00:00", exit_reason="STRATEGY_EXIT(atr_breakout)")
        self.tracker.log_exit_diagnostics(trade_id=trade_id, strategy_exit_triggered=True)

    def _make_winner_trade(self, trade_id, ticker, confidence_score=80.0):
        self.tracker.log_entry(
            trade_id=trade_id, ticker=ticker, strategy_name="atr_breakout",
            strategy_version="test", direction="LONG", price=100.0, shares=10,
            position_value=1000.0, position_pct=0.1, cash=9000.0, equity=10000.0,
            timestamp="2026-01-01T00:00:00", execution="REAL",
        )
        self.tracker.log_confidence_score(trade_id=trade_id, confidence_score=confidence_score)
        for d, c, h in [("2026-01-01", 104.0, 105.0), ("2026-01-02", 108.0, 110.0),
                        ("2026-01-03", 112.0, 115.0), ("2026-01-04", 110.0, 116.0),
                        ("2026-01-05", 109.0, 116.0)]:
            self.tracker.update_position_metrics(
                trade_id=trade_id, date=d, close=c, entry_price=100.0, shares=10,
                high=h, low=c - 1.0)
        self.tracker.log_exit(
            trade_id=trade_id, price=109.0, cash=1.0, equity=1.0,
            timestamp="2026-01-05T12:00:00", exit_reason="STRATEGY_EXIT(atr_breakout)")
        self.tracker.log_exit_diagnostics(trade_id=trade_id, strategy_exit_triggered=True)

    def _patterns_by_key(self):
        df = data.load_dataset(db_path=self.source_db, execution="REAL")
        candidates = early_failure.analyze(df)
        return {c.pattern_key: c for c in candidates}

    def test_mfe_day2_and_low_confidence_flags_reach_candidate_pattern(self):
        n_candidate = config.TRADE_INTELLIGENCE_MIN_CANDIDATE_N
        for i in range(n_candidate):
            self._make_early_failure_trade(f"ef{i}", f"US.EF{i}")
        for i in range(5):
            self._make_winner_trade(f"win{i}", f"US.WIN{i}")

        by_key = self._patterns_by_key()

        self.assertIn("mfe_day2_lt_1pct", by_key)
        ef_pattern = by_key["mfe_day2_lt_1pct"]
        self.assertEqual(ef_pattern.state, "CANDIDATE_PATTERN")
        self.assertAlmostEqual(ef_pattern.precision, 1.0)
        self.assertEqual(ef_pattern.n_sample, n_candidate)

        self.assertIn("low_confidence_score", by_key)
        conf_pattern = by_key["low_confidence_score"]
        self.assertEqual(conf_pattern.state, "CANDIDATE_PATTERN")
        self.assertAlmostEqual(conf_pattern.precision, 1.0)

    def test_slice_below_min_observation_n_is_omitted(self):
        n_candidate = config.TRADE_INTELLIGENCE_MIN_CANDIDATE_N
        for i in range(n_candidate):
            self._make_early_failure_trade(f"ef{i}", f"US.EF{i}")
        for i in range(5):
            self._make_winner_trade(f"win{i}", f"US.WIN{i}")
        # Only 2 trades tagged with a rare regime label -> below
        # TRADE_INTELLIGENCE_MIN_OBSERVATION_N, must not appear at all.
        self._make_early_failure_trade("rare1", "US.RARE1", regime_label="REGIME_RARE")
        self._make_early_failure_trade("rare2", "US.RARE2", regime_label="REGIME_RARE")
        self.assertLess(2, config.TRADE_INTELLIGENCE_MIN_OBSERVATION_N)

        by_key = self._patterns_by_key()
        self.assertNotIn("entry_regime_REGIME_RARE", by_key)

    def test_no_patterns_when_no_early_failures_present(self):
        for i in range(10):
            self._make_winner_trade(f"win{i}", f"US.WIN{i}")
        by_key = self._patterns_by_key()
        self.assertNotIn("mfe_day2_lt_1pct", by_key)


if __name__ == "__main__":
    unittest.main()
