"""
regime/provider.py — RegimeProvider: the abstract contract every Market
Regime source (rule-based today, local LLM in the future) must implement.

engine/runner.py, and any future consumer, only ever depends on this
interface (via regime.get_regime_provider()/regime.evaluate_and_log()) —
never on RuleRegimeProvider or LocalLLMRegimeProvider directly. That is
what lets a real local model replace LocalLLMRegimeProvider later without
touching engine/runner.py, risk/portfolio_position_manager.py, or
risk/portfolio_risk_manager.py.
"""
from abc import ABC, abstractmethod

from regime.models import MarketContext, MarketRegime


class RegimeProvider(ABC):
    @abstractmethod
    def evaluate(self, context: MarketContext) -> MarketRegime:
        """Return this pass's MarketRegime for the given MarketContext.
        Implementations may raise on failure (timeout, invalid output,
        model unavailable, ...) — callers are responsible for fallback
        handling; see regime/factory.py."""
        raise NotImplementedError
