"""
Unit tests for regime/models.py — MarketContext/MarketRegime validation.

Run:  python -m unittest regime.test_models -v
"""
import unittest

from regime.models import (MarketContext, MarketRegime, VALID_REGIMES,
                            REGIME_BULL, REGIME_NEUTRAL, REGIME_BEAR, REGIME_CRASH,
                            SOURCE_RULES, SOURCE_LLM)
import test_support


class TestMarketRegimeValidation(unittest.TestCase):

    def test_all_four_valid_regimes_construct(self):
        for r in (REGIME_BULL, REGIME_NEUTRAL, REGIME_BEAR, REGIME_CRASH):
            mr = MarketRegime(regime=r, confidence=0.8, risk_level=0.5,
                               reason_codes=["X"], source=SOURCE_RULES)
            self.assertEqual(mr.regime, r)

    def test_invalid_regime_label_rejected(self):
        with self.assertRaises(ValueError):
            MarketRegime(regime="SIDEWAYS", confidence=0.5, risk_level=0.5,
                         reason_codes=[], source=SOURCE_RULES)

    def test_confidence_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            MarketRegime(regime=REGIME_BULL, confidence=1.5, risk_level=0.5,
                         reason_codes=[], source=SOURCE_RULES)
        with self.assertRaises(ValueError):
            MarketRegime(regime=REGIME_BULL, confidence=-0.1, risk_level=0.5,
                         reason_codes=[], source=SOURCE_RULES)

    def test_risk_level_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            MarketRegime(regime=REGIME_BULL, confidence=0.5, risk_level=1.1,
                         reason_codes=[], source=SOURCE_RULES)
        with self.assertRaises(ValueError):
            MarketRegime(regime=REGIME_BULL, confidence=0.5, risk_level=-0.01,
                         reason_codes=[], source=SOURCE_RULES)

    def test_valid_regimes_tuple_matches_constants(self):
        self.assertEqual(set(VALID_REGIMES),
                          {REGIME_BULL, REGIME_NEUTRAL, REGIME_BEAR, REGIME_CRASH})

    def test_source_llm_constant_accepted(self):
        mr = MarketRegime(regime=REGIME_NEUTRAL, confidence=0.5, risk_level=0.5,
                           reason_codes=[], source=SOURCE_LLM)
        self.assertEqual(mr.source, SOURCE_LLM)


class TestMarketContext(unittest.TestCase):

    def test_defaults_and_extra_field(self):
        ctx = MarketContext(weather_code=2, qqq_above_ma=True, drawdown_halt=False)
        self.assertEqual(ctx.extra, {})

    def test_none_tolerant_fields(self):
        ctx = MarketContext(weather_code=None, qqq_above_ma=None, drawdown_halt=False)
        self.assertIsNone(ctx.weather_code)
        self.assertIsNone(ctx.qqq_above_ma)


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
