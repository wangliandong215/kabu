"""
research/trade_intelligence_narrative.py — V3.6-A narrative placeholder.

Mirrors risk/llm/base.py + risk/llm/{null_provider,mock_provider}.py: an
ABC provider + Null/Mock implementations, mode-gated by
config.TRADE_INTELLIGENCE_LLM_MODE (OFF/SHADOW only, no ACTIVE — real LLM
wiring is explicitly deferred to a later V3.6 phase, same convention as
RISK_ENGINE_LLM_MODE / NEWS_ENGINE_LLM_MODE / AI_DECISION_MODE).
MockNarrativeProvider only echoes counts already present in the context
dict it's given — it never invents a claim or number not already computed
by the statistical analysis modules.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import config

MODE_OFF = "OFF"
MODE_SHADOW = "SHADOW"
VALID_MODES = (MODE_OFF, MODE_SHADOW)


@dataclass
class TradeIntelligenceNarrative:
    summary: str
    model: str = "none"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class NarrativeProvider(ABC):
    @abstractmethod
    def generate(self, context: Dict[str, Any]) -> TradeIntelligenceNarrative:
        """context carries only counts already computed by
        research/trade_intelligence_runner.py (n_trades_analyzed,
        n_candidate_patterns, n_observations, analysis_types) — see
        research/trade_intelligence_report.py::build_report(). Implementations
        may raise; the report builder catches everything and treats a raise
        the same as "no narrative available"."""
        raise NotImplementedError


class NullNarrativeProvider(NarrativeProvider):
    def generate(self, context: Dict[str, Any]) -> TradeIntelligenceNarrative:
        return TradeIntelligenceNarrative(
            summary="AI narrative disabled (TRADE_INTELLIGENCE_LLM_MODE=OFF).",
            model="none",
        )


class MockNarrativeProvider(NarrativeProvider):
    def generate(self, context: Dict[str, Any]) -> TradeIntelligenceNarrative:
        n_trades = context.get("n_trades_analyzed", 0)
        n_candidate = context.get("n_candidate_patterns", 0)
        n_observation = context.get("n_observations", 0)
        analysis_types = context.get("analysis_types") or []
        summary = (
            f"This run analyzed {n_trades} closed trades across "
            f"{len(analysis_types)} analysis type(s) "
            f"({', '.join(analysis_types) if analysis_types else 'none'}). "
            f"{n_candidate} candidate pattern(s) and {n_observation} "
            f"observation(s) were recorded. This is a template summary "
            f"(MockNarrativeProvider) — no real LLM is in use yet; see the "
            f"per-analysis sections below for details. Findings are "
            f"statistical associations only, not proven causes, and are not "
            f"auto-applied to any trading logic."
        )
        return TradeIntelligenceNarrative(summary=summary, model="mock-v1")


def get_narrative_provider(mode: Optional[str] = None) -> NarrativeProvider:
    mode = mode if mode is not None else config.TRADE_INTELLIGENCE_LLM_MODE
    if mode == MODE_SHADOW:
        return MockNarrativeProvider()
    return NullNarrativeProvider()
