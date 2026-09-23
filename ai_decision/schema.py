"""
ai_decision/schema.py — V3.5 AI Decision Layer shared types.

Pure data contracts, no logic. AIDecisionContext is the unified input
snapshot (spec section 二) an AIDecisionProvider analyzes; AIDecision is the
structured advisory output (spec section 三). Mirrors
risk/risk_decision.py's dataclass-not-Pydantic convention.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AIAction(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    SKIP = "SKIP"


@dataclass
class AIDecisionContext:
    """Unified input schema (V3.5 spec section 二). Every *_snapshot field is
    a plain dict the caller builds from data that has already been fetched
    and structured upstream (PortfolioState, RiskDecision, NewsEvent list,
    ...) — this module never fetches or fabricates data itself (spec: "AI
    不负责自己寻找事实"). macro/financial/institution_snapshot have no
    producer anywhere in this codebase yet, so they default to None
    (DATA_UNAVAILABLE-safe, same convention as
    risk/portfolio_state.py::PortfolioState.news_events) — never a guessed
    empty dict."""
    symbol: str
    market_regime: Optional[str] = None
    portfolio_snapshot: Dict[str, Any] = field(default_factory=dict)
    risk_decision_snapshot: Dict[str, Any] = field(default_factory=dict)
    news_snapshot: Optional[Dict[str, Any]] = None
    event_snapshot: Optional[Dict[str, Any]] = None
    trade_history_snapshot: Optional[Dict[str, Any]] = None
    macro_snapshot: Optional[Dict[str, Any]] = None
    financial_snapshot: Optional[Dict[str, Any]] = None
    institution_snapshot: Optional[Dict[str, Any]] = None


@dataclass
class AIDecision:
    """Advisory-only output contract (V3.5 spec section 三/四). Nothing in
    this module or ai_decision/decision_layer.py can ever set
    RiskDecision.allowed/status or place an order — see
    decision_layer.py's module docstring for the full authority chain."""
    decision: AIAction
    confidence: float                  # 0..1
    reason_codes: List[str] = field(default_factory=list)
    reasoning: str = ""
    risk_flags: List[str] = field(default_factory=list)
    supporting_factors: List[str] = field(default_factory=list)
    negative_factors: List[str] = field(default_factory=list)
    model: str = "none"
    prompt_version: str = "v1"
    decision_version: str = "v1"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_log_dict(self, symbol: str = "") -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "symbol": symbol,
            "decision": self.decision.value if isinstance(self.decision, AIAction) else self.decision,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "reasoning": self.reasoning,
            "risk_flags": list(self.risk_flags),
            "supporting_factors": list(self.supporting_factors),
            "negative_factors": list(self.negative_factors),
            "model": self.model,
            "prompt_version": self.prompt_version,
            "decision_version": self.decision_version,
        }
