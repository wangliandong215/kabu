"""
Unit tests for ai_decision/decision_layer.py — the advisory-only AI
Decision Layer. Confirms the OFF/SHADOW contract, that a failing provider
never raises out of AIDecisionLayer.decide(), and that this module has no
call site in engine/runner.py in V3.5 Phase 1 (mirrors
risk/test_llm_advisor.py's existing test conventions).

Run:  python -m unittest ai_decision.test_decision_layer -v
"""
import unittest

import config
from ai_decision.decision_layer import AIDecisionLayer, get_ai_decision_provider
from ai_decision.llm.base import AIDecisionProvider
from ai_decision.llm.mock_provider import MockAIDecisionProvider
from ai_decision.llm.null_provider import NullAIDecisionProvider
from ai_decision.schema import AIDecision, AIDecisionContext


class _RaisingProvider(AIDecisionProvider):
    def decide(self, context):
        raise RuntimeError("boom")


class TestGetAIDecisionProvider(unittest.TestCase):
    def test_off_returns_null_provider(self):
        self.assertIsInstance(get_ai_decision_provider("OFF"), NullAIDecisionProvider)

    def test_shadow_returns_mock_provider(self):
        self.assertIsInstance(get_ai_decision_provider("SHADOW"), MockAIDecisionProvider)


class TestAIDecisionLayer(unittest.TestCase):

    def _ctx(self):
        return AIDecisionContext(symbol="US.NVDA",
                                  risk_decision_snapshot={"status": "ALLOW", "reason": "ok"})

    def test_default_mode_is_off(self):
        self.assertEqual(config.AI_DECISION_MODE, "OFF")

    def test_off_mode_returns_none_without_constructing_provider(self):
        layer = AIDecisionLayer(mode=config.AI_DECISION_MODE)   # default OFF
        result = layer.decide(self._ctx())
        self.assertIsNone(result)

    def test_shadow_mode_returns_mock_decision(self):
        layer = AIDecisionLayer(mode="SHADOW")
        result = layer.decide(self._ctx())
        self.assertIsInstance(result, AIDecision)
        self.assertEqual(result.model, "mock")

    def test_provider_exception_never_raises_returns_none(self):
        layer = AIDecisionLayer(provider=_RaisingProvider(), mode="SHADOW")
        result = layer.decide(self._ctx())
        self.assertIsNone(result)

    def test_invalid_mode_returns_none(self):
        layer = AIDecisionLayer(mode="ACTIVE")   # not a valid mode — see module docstring
        result = layer.decide(self._ctx())
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
