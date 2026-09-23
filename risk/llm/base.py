"""
risk/llm/base.py — V3.3 LLM Provider abstraction (spec section 十六).

Never wire Ollama/llama.cpp/OpenAI directly into risk/portfolio_risk_engine.py
or risk/llm_advisor.py — every real backend implements this ABC instead
(future: LocalLLMProvider/OllamaProvider/OpenAIProvider), mirroring
regime/provider.py::RegimeProvider's existing ABC pattern in this repo.

LLMProvider.analyze() is advisory-only input to risk/llm_advisor.py, which
is itself advisory-only to risk/portfolio_risk_engine.py — see
llm_advisor.py's module docstring for the full authority chain. Nothing in
this file, or anything implementing this ABC, can ever set
RiskDecision.allowed/status.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class LLMRiskAnalysis:
    """Advisory-only output contract. suggested_action/risk_flags are
    exposition for a human or a log, never applied to any order or
    RiskDecision automatically."""
    summary: str
    risk_flags: List[str] = field(default_factory=list)
    suggested_action: str = ""      # free-text advisory, e.g. "consider reducing NVDA"
    confidence: float = 0.0          # 0..1
    model: str = "none"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LLMProvider(ABC):
    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> LLMRiskAnalysis:
        """context carries the OrderIntent, PortfolioState summary, and the
        already-finalized RiskDecision (as plain dicts) — see
        risk/llm_advisor.py::build_context(). Implementations may raise;
        risk/llm_advisor.py catches everything and treats a raise the same
        as "no analysis available"."""
        raise NotImplementedError
