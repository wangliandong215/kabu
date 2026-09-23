"""
Unit tests for regime/rule_provider.py::RuleRegimeProvider — pins the
decision table documented in that module's docstring (deliberately
mirroring risk/portfolio_position_manager.py::classify_market_regime()'s
priority order, but producing the separate BULL/NEUTRAL/BEAR/CRASH
taxonomy for the Regime Observation Layer).

Run:  python -m unittest regime.test_rule_provider -v
"""
import unittest

from regime.models import (MarketContext, REGIME_BULL, REGIME_NEUTRAL,
                            REGIME_BEAR, REGIME_CRASH, SOURCE_RULES)
from regime.rule_provider import RuleRegimeProvider
import test_support


class TestRuleRegimeProvider(unittest.TestCase):

    def setUp(self):
        self.provider = RuleRegimeProvider()

    def _ctx(self, weather_code, qqq_above_ma, drawdown_halt):
        return MarketContext(weather_code=weather_code, qqq_above_ma=qqq_above_ma,
                             drawdown_halt=drawdown_halt)

    def test_drawdown_halt_always_crash(self):
        # Even with the most bullish weather/MA reading, drawdown_halt wins.
        result = self.provider.evaluate(self._ctx(2, True, True))
        self.assertEqual(result.regime, REGIME_CRASH)
        self.assertEqual(result.risk_level, 0.0)
        self.assertIn("DRAWDOWN_HALT", result.reason_codes)

    def test_weather_crisis_is_crash(self):
        result = self.provider.evaluate(self._ctx(0, True, False))
        self.assertEqual(result.regime, REGIME_CRASH)
        self.assertIn("WEATHER_CRISIS", result.reason_codes)

    def test_weather_chop_is_bear(self):
        result = self.provider.evaluate(self._ctx(1, True, False))
        self.assertEqual(result.regime, REGIME_BEAR)
        self.assertIn("WEATHER_CHOP", result.reason_codes)

    def test_below_ma200_is_bear_even_with_safe_weather(self):
        result = self.provider.evaluate(self._ctx(2, False, False))
        self.assertEqual(result.regime, REGIME_BEAR)
        self.assertIn("QQQ_BELOW_MA200", result.reason_codes)

    def test_safe_weather_and_above_ma200_is_bull(self):
        result = self.provider.evaluate(self._ctx(2, True, False))
        self.assertEqual(result.regime, REGIME_BULL)
        self.assertEqual(result.risk_level, 1.0)
        self.assertIn("WEATHER_SAFE", result.reason_codes)
        self.assertIn("QQQ_ABOVE_MA200", result.reason_codes)

    def test_missing_weather_code_degrades_to_bear_not_bull(self):
        result = self.provider.evaluate(self._ctx(None, True, False))
        self.assertEqual(result.regime, REGIME_BEAR)
        self.assertLess(result.confidence, 0.9)

    def test_missing_qqq_above_ma_degrades_to_bear_not_bull(self):
        result = self.provider.evaluate(self._ctx(2, None, False))
        self.assertEqual(result.regime, REGIME_BEAR)
        self.assertLess(result.confidence, 0.9)

    def test_source_is_always_rules(self):
        result = self.provider.evaluate(self._ctx(2, True, False))
        self.assertEqual(result.source, SOURCE_RULES)

    def test_confidence_full_when_all_signals_present(self):
        result = self.provider.evaluate(self._ctx(2, True, False))
        self.assertEqual(result.confidence, 0.9)

    def test_neutral_is_reachable_only_as_defensive_fallback(self):
        # With today's 3 inputs NEUTRAL is not reachable via any (weather_code,
        # qqq_above_ma, drawdown_halt) combination -- documents that intent
        # explicitly rather than leaving it as an untested code path.
        from itertools import product
        seen_regimes = set()
        for weather_code in (0, 1, 2, None):
            for qqq_above_ma in (True, False, None):
                for drawdown_halt in (True, False):
                    result = self.provider.evaluate(
                        self._ctx(weather_code, qqq_above_ma, drawdown_halt))
                    seen_regimes.add(result.regime)
        self.assertNotIn(REGIME_NEUTRAL, seen_regimes)
        self.assertEqual(seen_regimes, {REGIME_CRASH, REGIME_BEAR, REGIME_BULL})


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
