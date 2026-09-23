"""
Unit tests for ai_decision/llm/null_provider.py and
ai_decision/llm/mock_provider.py.

Run:  python -m unittest ai_decision.test_providers -v
"""
import unittest

from ai_decision.llm.mock_provider import MockAIDecisionProvider
from ai_decision.llm.null_provider import NullAIDecisionProvider
from ai_decision.schema import AIAction, AIDecision


class TestNullProvider(unittest.TestCase):
    def test_always_returns_skip_with_zero_confidence(self):
        result = NullAIDecisionProvider().decide({})
        self.assertIsInstance(result, AIDecision)
        self.assertEqual(result.decision, AIAction.SKIP)
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("AI_DECISION_DISABLED", result.reason_codes)


class TestMockProvider(unittest.TestCase):
    def test_empty_context_returns_hold_with_no_signal(self):
        result = MockAIDecisionProvider().decide({})
        self.assertEqual(result.decision, AIAction.HOLD)
        self.assertIn("NO_SIGNAL", result.reason_codes)

    def test_block_status_maps_to_skip(self):
        context = {"symbol": "US.NVDA",
                   "risk_decision": {"status": "BLOCK", "reason": "exposure exceeded"}}
        result = MockAIDecisionProvider().decide(context)
        self.assertEqual(result.decision, AIAction.SKIP)
        self.assertIn("RISK_ENGINE_BLOCK", result.reason_codes)
        self.assertTrue(any("exposure exceeded" in n for n in result.negative_factors))

    def test_positive_news_with_no_negative_maps_to_buy(self):
        context = {"symbol": "US.NVDA",
                   "risk_decision": {"status": "ALLOW"},
                   "news": {"effective_score": 0.6}}
        result = MockAIDecisionProvider().decide(context)
        self.assertEqual(result.decision, AIAction.BUY)
        self.assertIn("NEWS_POSITIVE", result.reason_codes)

    def test_risk_off_regime_adds_risk_flag(self):
        context = {"symbol": "US.NVDA", "market_regime": "RISK_OFF"}
        result = MockAIDecisionProvider().decide(context)
        self.assertIn("REGIME_RISK_OFF", result.risk_flags)

    def test_large_position_weight_adds_concentration_flag(self):
        context = {"symbol": "US.NVDA", "portfolio": {"weight": 0.12}}
        result = MockAIDecisionProvider().decide(context)
        self.assertIn("POSITION_CONCENTRATION", result.risk_flags)
        self.assertIn("POSITION_ALREADY_LARGE", result.reason_codes)

    def test_confidence_always_in_unit_range(self):
        for context in ({}, {"risk_decision": {"status": "BLOCK"}},
                         {"news": {"effective_score": 0.9}},
                         {"news": {"effective_score": -0.9}}):
            result = MockAIDecisionProvider().decide(context)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
