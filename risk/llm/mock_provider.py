"""
risk/llm/mock_provider.py — canned, deterministic LLMProvider for tests and
for exercising SHADOW mode before any real backend exists. NOT wired to any
real model — same "placeholder, not a fake real answer" contract as
position_manager/llm_interface.py::_PlaceholderReviewer, except this one
returns a fixed analysis instead of raising, since risk/llm_advisor.py's
tests need something concrete to assert against.
"""
from typing import Any, Dict

from risk.llm.base import LLMProvider, LLMRiskAnalysis


class MockLLMProvider(LLMProvider):
    def analyze(self, context: Dict[str, Any]) -> LLMRiskAnalysis:
        decision = context.get("risk_decision", {}) or {}
        order = context.get("order", {}) or {}
        status = decision.get("status", "UNKNOWN")
        code = order.get("code", "?")
        return LLMRiskAnalysis(
            summary=f"[mock] {code} evaluated as {status}",
            risk_flags=list(decision.get("warnings", [])),
            suggested_action="none (mock provider)",
            confidence=0.5,
            model="mock",
        )
