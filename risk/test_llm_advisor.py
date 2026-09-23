"""
Unit tests for risk/llm_advisor.py — the advisory-only LLM layer. Confirms
the OFF/SHADOW contract and that a failing provider never raises out of
LLMRiskAdvisor.analyze() (mirrors position_manager/llm_interface.py's
existing test conventions).

Run:  python -m unittest risk.test_llm_advisor -v
"""
import unittest

import config
from risk.llm.base import LLMProvider, LLMRiskAnalysis
from risk.llm.mock_provider import MockLLMProvider
from risk.llm.null_provider import NullLLMProvider
from risk.llm_advisor import LLMRiskAdvisor, get_llm_provider
from risk.portfolio_state import PortfolioState
from risk.risk_decision import OrderIntent, RiskDecision, RiskStatus


class _RaisingProvider(LLMProvider):
    def analyze(self, context):
        raise RuntimeError("boom")


class TestGetLLMProvider(unittest.TestCase):
    def test_off_returns_null_provider(self):
        self.assertIsInstance(get_llm_provider("OFF"), NullLLMProvider)

    def test_shadow_returns_mock_provider(self):
        self.assertIsInstance(get_llm_provider("SHADOW"), MockLLMProvider)


class TestNullProvider(unittest.TestCase):
    def test_always_succeeds_with_empty_analysis(self):
        result = NullLLMProvider().analyze({})
        self.assertIsInstance(result, LLMRiskAnalysis)
        self.assertEqual(result.risk_flags, [])


class TestLLMRiskAdvisor(unittest.TestCase):

    def _decision(self):
        return RiskDecision(status=RiskStatus.ALLOW, allowed=True, reason="ok")

    def test_off_mode_returns_none_without_constructing_provider(self):
        advisor = LLMRiskAdvisor(mode=config.RISK_ENGINE_LLM_MODE)   # default OFF
        result = advisor.analyze(OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0),
                                  PortfolioState(), self._decision())
        self.assertIsNone(result)

    def test_shadow_mode_returns_mock_analysis(self):
        advisor = LLMRiskAdvisor(mode="SHADOW")
        result = advisor.analyze(OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0),
                                  PortfolioState(), self._decision())
        self.assertIsNotNone(result)
        self.assertIn("US.NVDA", result.summary)

    def test_provider_exception_never_raises_returns_none(self):
        advisor = LLMRiskAdvisor(provider=_RaisingProvider(), mode="SHADOW")
        result = advisor.analyze(OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0),
                                  PortfolioState(), self._decision())
        self.assertIsNone(result)

    def test_analysis_never_affects_the_already_finalized_decision(self):
        # Structural guarantee: analyze() is called with a decision object
        # and only ever reads it -- it has no way to mutate `allowed`/`status`.
        decision = self._decision()
        advisor = LLMRiskAdvisor(mode="SHADOW")
        advisor.analyze(OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0),
                         PortfolioState(), decision)
        self.assertEqual(decision.status, RiskStatus.ALLOW)
        self.assertTrue(decision.allowed)


if __name__ == "__main__":
    unittest.main()
