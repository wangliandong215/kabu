"""
ai_decision/llm/base.py — V3.5 AI Decision provider abstraction.

Mirrors risk/llm/base.py::LLMProvider exactly (same author, same project,
proven pattern) — never wire a real backend (OpenAI/Anthropic/local LLM)
directly into ai_decision/decision_layer.py; every real backend implements
this ABC instead (future: OpenAIDecisionProvider/LocalLLMDecisionProvider).

decide() is advisory-only input to ai_decision/decision_layer.py, which is
itself advisory-only to the Rule Engine — see decision_layer.py's module
docstring for the full authority chain. Nothing in this file, or anything
implementing this ABC, can ever place an order or change portfolio state.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

from ai_decision.schema import AIDecision


class AIDecisionProvider(ABC):
    @abstractmethod
    def decide(self, context: Dict[str, Any]) -> AIDecision:
        """context is the plain-dict form of an AIDecisionContext — see
        ai_decision/decision_layer.py::build_context(). Implementations may
        raise; decision_layer.py catches everything and treats a raise the
        same as "no decision available"."""
        raise NotImplementedError
