"""
regime/models.py — data structures for the LLM-ready Market Regime
Observation Layer. See regime/__init__.py for the overall architecture.

MarketContext / MarketRegime are the only two shapes any RegimeProvider
(rule-based today, local LLM in the future) is allowed to speak — this is
what lets engine/runner.py, and eventually risk/portfolio_position_manager.py,
depend on a stable contract instead of a specific implementation.

IMPORTANT: this BULL/NEUTRAL/BEAR/CRASH taxonomy is intentionally distinct
from risk/portfolio_position_manager.py's existing BULL/NORMAL/CAUTION/
RISK_OFF classification. The two are NOT interchangeable and this module
does not replace that one — see regime/rule_provider.py module docstring.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

REGIME_BULL = "BULL"
REGIME_NEUTRAL = "NEUTRAL"
REGIME_BEAR = "BEAR"
REGIME_CRASH = "CRASH"

VALID_REGIMES = (REGIME_BULL, REGIME_NEUTRAL, REGIME_BEAR, REGIME_CRASH)

SOURCE_RULES = "rules"
SOURCE_LLM = "llm"


@dataclass
class MarketContext:
    """Inputs to RegimeProvider.evaluate(), built once per run_once() pass
    from values engine/runner.py has already computed for other purposes.
    This layer never recomputes weather_code/qqq_above_ma/drawdown_halt
    itself — see regime/rule_provider.py module docstring for why.

    `extra` exists so a future signal (VIX, market breadth, ...) or a
    future LLM prompt input can be threaded through without changing this
    dataclass's shape or any existing caller.
    """
    weather_code: Optional[int]
    qqq_above_ma: Optional[bool]
    drawdown_halt: bool
    extra: Dict = field(default_factory=dict)


@dataclass
class MarketRegime:
    """Output of RegimeProvider.evaluate(). Observation-only: nothing in
    this codebase may use `regime`/`risk_level` to directly gate or size a
    BUY/SELL — see regime/__init__.py module docstring."""
    regime: str                 # one of VALID_REGIMES
    confidence: float           # 0..1
    risk_level: float           # 0..1, descriptive only
    reason_codes: List[str]
    source: str                 # SOURCE_RULES | SOURCE_LLM

    def __post_init__(self) -> None:
        if self.regime not in VALID_REGIMES:
            raise ValueError(
                f"MarketRegime.regime={self.regime!r} is not one of {VALID_REGIMES}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"MarketRegime.confidence={self.confidence!r} must be in [0, 1]")
        if not (0.0 <= self.risk_level <= 1.0):
            raise ValueError(f"MarketRegime.risk_level={self.risk_level!r} must be in [0, 1]")
