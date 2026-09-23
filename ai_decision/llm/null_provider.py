"""
ai_decision/llm/null_provider.py — the default AIDecisionProvider
(config.AI_DECISION_MODE == "OFF"). Always succeeds, always returns an
explicit SKIP with confidence 0 — never raises, never fabricates a real
opinion. Mirrors risk/llm/null_provider.py's contract exactly.
"""
from typing import Any, Dict

from ai_decision.llm.base import AIDecisionProvider
from ai_decision.schema import AIAction, AIDecision


class NullAIDecisionProvider(AIDecisionProvider):
    def decide(self, context: Dict[str, Any]) -> AIDecision:
        return AIDecision(
            decision=AIAction.SKIP, confidence=0.0,
            reason_codes=["AI_DECISION_DISABLED"],
            reasoning="AI Decision Layer disabled (AI_DECISION_MODE=OFF)",
            model="none",
        )
