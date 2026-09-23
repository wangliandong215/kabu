"""
risk/llm/null_provider.py — the default LLMProvider (config.RISK_ENGINE_LLM_MODE
== "OFF"). Always succeeds, always returns an explicitly-empty analysis —
never raises, never fabricates a real opinion. This is what "runs completely
normally with no LLM" means for the Risk Engine (V3.3 spec section 十四).
"""
from typing import Any, Dict

from risk.llm.base import LLMProvider, LLMRiskAnalysis


class NullLLMProvider(LLMProvider):
    def analyze(self, context: Dict[str, Any]) -> LLMRiskAnalysis:
        return LLMRiskAnalysis(
            summary="LLM advisory disabled (RISK_ENGINE_LLM_MODE=OFF)",
            risk_flags=[], suggested_action="", confidence=0.0, model="none",
        )
