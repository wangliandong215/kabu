"""
Unit tests for engine/confidence_score.py.
Run:  python -m unittest engine.test_confidence_score -v
"""
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine import confidence_score as cs
from engine.trade_tracker import TradeTracker


class TestComputeConfidenceScoreBounds(unittest.TestCase):
    """0-100 boundary + no-NaN/inf guarantees (V2.9 spec section 十四)."""

    def test_rule_only_returns_rule_score_unchanged(self):
        result = cs.compute_confidence_score(rule_based_score=72.0)
        self.assertAlmostEqual(result.confidence_score, 72.0)
        self.assertEqual(result.weights_used, {"rule": cs.W_RULE})

    def test_zero_rule_score_floor(self):
        result = cs.compute_confidence_score(rule_based_score=0.0)
        self.assertEqual(result.confidence_score, 0.0)

    def test_all_components_maxed_gives_high_but_hmm_capped_score(self):
        # HMM_BULL's base score is deliberately capped at 90 (not 100) even
        # at full regime confidence -- a regime read is never treated as
        # certainty -- so the ceiling here is 98.5, not 100. See
        # _HMM_BASE_SCORE's docstring.
        result = cs.compute_confidence_score(
            rule_based_score=100.0,
            hmm_state="HMM_BULL", hmm_confidence=100.0,
            historical_win_rate=1.0, historical_win_rate_n=100,
            historical_expectancy_pct=cs.EXPECTANCY_CLAMP_PCT,
            market_cnn_fear_greed=100.0,
            volatility_atr_pct=0.02,
            volume_feature=100.0,
        )
        self.assertAlmostEqual(result.confidence_score, 98.5)
        self.assertLessEqual(result.confidence_score, 100.0)

    def test_all_components_worst_case_gives_low_but_hmm_floored_score(self):
        # Symmetric floor: HMM_BEAR's base score is 10, not 0, so the floor
        # here is 1.5, not 0.
        result = cs.compute_confidence_score(
            rule_based_score=0.0,
            hmm_state="HMM_BEAR", hmm_confidence=100.0,
            historical_win_rate=0.0, historical_win_rate_n=100,
            historical_expectancy_pct=-cs.EXPECTANCY_CLAMP_PCT,
            market_cnn_fear_greed=0.0,
            volatility_atr_pct=0.20,
            volume_feature=0.0,
        )
        self.assertAlmostEqual(result.confidence_score, 1.5)
        self.assertGreaterEqual(result.confidence_score, 0.0)

    def test_out_of_range_inputs_are_clipped_not_rejected(self):
        result = cs.compute_confidence_score(
            rule_based_score=150.0,          # over 100
            historical_win_rate=1.5,          # over 1.0
            historical_win_rate_n=50,
            market_cnn_fear_greed=-20.0,       # under 0
        )
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 100.0)
        self.assertFalse(math.isnan(result.confidence_score))
        self.assertFalse(math.isinf(result.confidence_score))

    def test_never_nan_or_inf_across_a_grid_of_inputs(self):
        for rule in (0.0, 50.0, 100.0):
            for hmm in (None, "HMM_BULL", "HMM_BEAR", "UNKNOWN_LABEL"):
                for wr in (None, 0.0, 0.5, 1.0):
                    for exp in (None, -1.0, 0.0, 1.0):
                        result = cs.compute_confidence_score(
                            rule_based_score=rule,
                            hmm_state=hmm, hmm_confidence=80.0,
                            historical_win_rate=wr, historical_win_rate_n=20,
                            historical_expectancy_pct=exp,
                        )
                        self.assertFalse(math.isnan(result.confidence_score))
                        self.assertFalse(math.isinf(result.confidence_score))
                        self.assertGreaterEqual(result.confidence_score, 0.0)
                        self.assertLessEqual(result.confidence_score, 100.0)


class TestMissingInputsExcludedNotDefaulted(unittest.TestCase):
    """Missing dimensions must drop out of the weighted average (renormalize),
    never be treated as a fabricated placeholder value — mirrors engine/
    scoring.py's fundamental_score/news_score contract."""

    def test_missing_hmm_excluded(self):
        result = cs.compute_confidence_score(rule_based_score=80.0, hmm_state=None)
        self.assertIsNone(result.hmm_component)
        self.assertNotIn("hmm", result.weights_used)

    def test_unrecognized_hmm_label_excluded(self):
        result = cs.compute_confidence_score(
            rule_based_score=80.0, hmm_state="HMM_UNKNOWN", hmm_confidence=90.0)
        self.assertIsNone(result.hmm_component)

    def test_missing_hmm_confidence_excludes_component(self):
        result = cs.compute_confidence_score(
            rule_based_score=80.0, hmm_state="HMM_BULL", hmm_confidence=None)
        self.assertIsNone(result.hmm_component)

    def test_sample_below_minimum_excludes_win_rate_and_expectancy(self):
        result = cs.compute_confidence_score(
            rule_based_score=80.0,
            historical_win_rate=1.0, historical_win_rate_n=cs.MIN_HISTORICAL_SAMPLE - 1,
            historical_expectancy_pct=0.03,
        )
        self.assertIsNone(result.win_rate_component)
        self.assertIsNone(result.expectancy_component)

    def test_sample_at_minimum_includes_win_rate_and_expectancy(self):
        result = cs.compute_confidence_score(
            rule_based_score=80.0,
            historical_win_rate=0.6, historical_win_rate_n=cs.MIN_HISTORICAL_SAMPLE,
            historical_expectancy_pct=0.01,
        )
        self.assertIsNotNone(result.win_rate_component)
        self.assertIsNotNone(result.expectancy_component)

    def test_missing_market_stats_excluded(self):
        result = cs.compute_confidence_score(rule_based_score=80.0)
        self.assertIsNone(result.market_component)
        self.assertNotIn("market", result.weights_used)

    def test_vix_used_only_when_cnn_missing(self):
        with_cnn = cs.compute_confidence_score(
            rule_based_score=50.0, market_cnn_fear_greed=70.0, market_vix_close=15.0)
        vix_only = cs.compute_confidence_score(
            rule_based_score=50.0, market_cnn_fear_greed=None, market_vix_close=15.0)
        self.assertAlmostEqual(with_cnn.market_component, 70.0)
        self.assertIsNotNone(vix_only.market_component)
        self.assertNotAlmostEqual(vix_only.market_component, 70.0)

    def test_missing_volatility_excluded(self):
        result = cs.compute_confidence_score(rule_based_score=80.0, volatility_atr_pct=None)
        self.assertIsNone(result.volatility_component)

    def test_missing_volume_feature_excluded_phase1_default(self):
        result = cs.compute_confidence_score(rule_based_score=80.0)
        self.assertIsNone(result.volume_component)
        self.assertNotIn("volume", result.weights_used)

    def test_weights_renormalize_when_components_missing(self):
        rule_only = cs.compute_confidence_score(rule_based_score=30.0)
        self.assertAlmostEqual(sum(rule_only.weights_used.values()), cs.W_RULE)
        self.assertAlmostEqual(rule_only.confidence_score, 30.0)


class TestVolatilityBand(unittest.TestCase):
    def test_inside_normal_band_scores_100(self):
        result = cs.compute_confidence_score(rule_based_score=50.0, volatility_atr_pct=0.02)
        self.assertAlmostEqual(result.volatility_component, 100.0)

    def test_extremely_low_or_high_scores_0(self):
        low = cs.compute_confidence_score(rule_based_score=50.0, volatility_atr_pct=0.0001)
        high = cs.compute_confidence_score(rule_based_score=50.0, volatility_atr_pct=1.0)
        self.assertAlmostEqual(low.volatility_component, 0.0)
        self.assertAlmostEqual(high.volatility_component, 0.0)

    def test_negative_atr_pct_excluded(self):
        result = cs.compute_confidence_score(rule_based_score=50.0, volatility_atr_pct=-0.01)
        self.assertIsNone(result.volatility_component)


class TestHmmComponent(unittest.TestCase):
    def test_bull_with_full_confidence_scores_above_neutral(self):
        result = cs.compute_confidence_score(
            rule_based_score=50.0, hmm_state="HMM_BULL", hmm_confidence=100.0)
        self.assertGreater(result.hmm_component, 50.0)

    def test_bear_with_full_confidence_scores_below_neutral(self):
        result = cs.compute_confidence_score(
            rule_based_score=50.0, hmm_state="HMM_BEAR", hmm_confidence=100.0)
        self.assertLess(result.hmm_component, 50.0)

    def test_low_confidence_pulls_toward_neutral(self):
        confident = cs.compute_confidence_score(
            rule_based_score=50.0, hmm_state="HMM_BULL", hmm_confidence=100.0)
        unsure = cs.compute_confidence_score(
            rule_based_score=50.0, hmm_state="HMM_BULL", hmm_confidence=5.0)
        self.assertLess(abs(unsure.hmm_component - 50.0), abs(confident.hmm_component - 50.0))


class TestHistoricalPerformanceNoLookahead(unittest.TestCase):
    """historical_performance() must never use a trade's own future outcome,
    must exclude BOOTSTRAP_TRADE_IDS, and must respect execution/REAL vs
    PAPER separation (V2.9 spec section 七/九)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "test_trade_history.db"
        self.tracker = TradeTracker(self.db_path)

    def tearDown(self):
        self.tracker.close()
        self._tmpdir.cleanup()

    def _close_trade(self, trade_id, code, entry_price, exit_price, entry_ts, exit_ts,
                      execution="REAL"):
        self.tracker.log_entry(
            trade_id=trade_id, ticker=code, strategy_name="atr_breakout",
            strategy_version="2.9", direction="LONG", price=entry_price, shares=10,
            position_value=entry_price * 10, position_pct=0.1,
            cash=9000.0, equity=10000.0, timestamp=entry_ts, execution=execution,
        )
        self.tracker.log_exit(
            trade_id=trade_id, price=exit_price, cash=9500.0, equity=10500.0,
            timestamp=exit_ts, exit_reason="STOP_LOSS",
        )

    def test_no_tracker_returns_none(self):
        self.assertEqual(cs.historical_performance(None), (None, None, 0))

    def test_below_min_sample_returns_none(self):
        for i in range(cs.MIN_HISTORICAL_SAMPLE - 1):
            self._close_trade(f"US.AAA_{i}", "US.AAA", 100.0, 110.0,
                               f"2026-01-0{i+1}T09:30:00", f"2026-01-0{i+1}T16:00:00")
        win_rate, expectancy, n = cs.historical_performance(self.tracker, as_of="2026-02-01T00:00:00")
        self.assertIsNone(win_rate)
        self.assertIsNone(expectancy)
        self.assertEqual(n, cs.MIN_HISTORICAL_SAMPLE - 1)

    def test_at_min_sample_computes_win_rate_and_expectancy(self):
        # 3 winners (+10%), 2 losers (-5%) -> win_rate=0.6,
        # expectancy = mean(+0.10,+0.10,+0.10,-0.05,-0.05) = +4%
        outcomes = [110.0, 110.0, 110.0, 95.0, 95.0]
        for i, exit_price in enumerate(outcomes):
            self._close_trade(f"US.AAA_{i}", "US.AAA", 100.0, exit_price,
                               f"2026-01-{i+1:02d}T09:30:00", f"2026-01-{i+1:02d}T16:00:00")
        win_rate, expectancy, n = cs.historical_performance(self.tracker, as_of="2026-02-01T00:00:00")
        self.assertEqual(n, 5)
        self.assertAlmostEqual(win_rate, 0.6)
        self.assertAlmostEqual(expectancy, 0.04, places=4)

    def test_trades_closing_after_as_of_are_excluded_no_lookahead(self):
        for i in range(cs.MIN_HISTORICAL_SAMPLE):
            self._close_trade(f"US.AAA_{i}", "US.AAA", 100.0, 110.0,
                               f"2026-01-{i+1:02d}T09:30:00", f"2026-01-{i+1:02d}T16:00:00")
        # A trade whose exit is AFTER as_of must not be counted at all.
        self._close_trade("US.FUTURE", "US.FUTURE", 100.0, 200.0,
                           "2026-01-10T09:30:00", "2099-01-01T00:00:00")
        win_rate, expectancy, n = cs.historical_performance(self.tracker, as_of="2026-02-01T00:00:00")
        self.assertEqual(n, cs.MIN_HISTORICAL_SAMPLE)
        self.assertAlmostEqual(win_rate, 1.0)   # would be skewed if US.FUTURE leaked in

    def test_bootstrap_trade_ids_excluded(self):
        bootstrap_id = next(iter(cs.BOOTSTRAP_TRADE_IDS))
        self.tracker.log_entry(
            trade_id=bootstrap_id, ticker="US.QQQ", strategy_name="core_etf",
            strategy_version="2.9", direction="LONG", price=500.0, shares=10,
            position_value=5000.0, position_pct=0.1, cash=9000.0, equity=10000.0,
            timestamp="2026-07-06T22:46:06.692798", execution="REAL",
        )
        self.tracker.log_exit(
            trade_id=bootstrap_id, price=1000.0, cash=9500.0, equity=15000.0,
            timestamp="2026-08-31T22:56:08.792083", exit_reason="STOP_LOSS",
        )
        for i in range(cs.MIN_HISTORICAL_SAMPLE):
            self._close_trade(f"US.AAA_{i}", "US.AAA", 100.0, 95.0,
                               f"2026-01-{i+1:02d}T09:30:00", f"2026-01-{i+1:02d}T16:00:00")
        win_rate, expectancy, n = cs.historical_performance(self.tracker, as_of="2026-09-01T00:00:00")
        # If the bootstrap 100% winner leaked in, win_rate would not be 0.0.
        self.assertEqual(n, cs.MIN_HISTORICAL_SAMPLE)
        self.assertAlmostEqual(win_rate, 0.0)

    def test_paper_and_real_are_kept_separate(self):
        for i in range(cs.MIN_HISTORICAL_SAMPLE):
            self._close_trade(f"US.REAL_{i}", "US.REAL", 100.0, 110.0,
                               f"2026-01-{i+1:02d}T09:30:00", f"2026-01-{i+1:02d}T16:00:00",
                               execution="REAL")
        for i in range(cs.MIN_HISTORICAL_SAMPLE):
            self._close_trade(f"JP.PAPER_{i}", "JP.PAPER", 100.0, 90.0,
                               f"2026-01-{i+1:02d}T09:30:00", f"2026-01-{i+1:02d}T16:00:00",
                               execution="PAPER")
        real_wr, real_exp, real_n = cs.historical_performance(
            self.tracker, as_of="2026-02-01T00:00:00", execution="REAL")
        paper_wr, paper_exp, paper_n = cs.historical_performance(
            self.tracker, as_of="2026-02-01T00:00:00", execution="PAPER")
        self.assertAlmostEqual(real_wr, 1.0)
        self.assertAlmostEqual(paper_wr, 0.0)
        self.assertEqual(real_n, cs.MIN_HISTORICAL_SAMPLE)
        self.assertEqual(paper_n, cs.MIN_HISTORICAL_SAMPLE)

    def test_legacy_null_execution_counts_as_real(self):
        """Rows predating the execution column (NULL) are real US trades
        that simply predate that column — see module docstring's audit
        note. Excluding them would needlessly shrink an already-small
        sample."""
        for i in range(cs.MIN_HISTORICAL_SAMPLE):
            self._close_trade(f"US.LEGACY_{i}", "US.LEGACY", 100.0, 110.0,
                               f"2026-01-{i+1:02d}T09:30:00", f"2026-01-{i+1:02d}T16:00:00",
                               execution=None)
        win_rate, expectancy, n = cs.historical_performance(
            self.tracker, as_of="2026-02-01T00:00:00", execution="REAL")
        self.assertEqual(n, cs.MIN_HISTORICAL_SAMPLE)
        self.assertAlmostEqual(win_rate, 1.0)


class TestGatherAndScoreNeverRaises(unittest.TestCase):
    def test_all_sources_missing_still_returns_rule_only_score(self):
        with mock.patch("engine.regime_store.get_regime_interface",
                         side_effect=Exception("boom")), \
             mock.patch("engine.market_context.get_latest_context",
                         side_effect=Exception("boom")):
            result, detail = cs.gather_and_score(
                code="US.AAPL", rule_based_score=77.0, sig={}, tracker=None)
        self.assertAlmostEqual(result.confidence_score, 77.0)
        self.assertEqual(detail["rule_based_score"], 77.0)
        self.assertIsNone(detail["hmm_state"])
        self.assertIsNone(detail["historical_win_rate"])
        self.assertIsNone(detail["volume_feature"])
        self.assertEqual(detail["formula_version"], cs.FORMULA_VERSION)

    def test_atr_pct_derived_from_sig(self):
        result, detail = cs.gather_and_score(
            code="US.AAPL", rule_based_score=60.0,
            sig={"atr": 2.0, "current_price": 100.0}, tracker=None)
        self.assertAlmostEqual(detail["volatility_atr_pct"], 0.02)

    def test_missing_atr_in_sig_leaves_volatility_none(self):
        result, detail = cs.gather_and_score(
            code="US.AAPL", rule_based_score=60.0, sig={}, tracker=None)
        self.assertIsNone(detail["volatility_atr_pct"])


if __name__ == "__main__":
    unittest.main()
