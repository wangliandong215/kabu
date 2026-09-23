"""
risk/risk_decision.py — V3.3 Portfolio Risk Engine shared types.

Pure data contract, no logic. OrderIntent is what a caller (engine/runner.py
_place_order()) hands the engine; RiskDecision is what evaluate() hands back.
Deliberately not a bool — see V3.3 spec section 十三: callers, logs, a future
dashboard/LLM/backtest all need the reason/violations/warnings/metrics, not
just a pass/fail.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RiskStatus(str, Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"


# Sub-check result status — DATA_UNAVAILABLE is distinct from ALLOW/WARN/BLOCK
# so a missing-data case can never be silently treated as "fine" (V3.3 spec
# section 八: "不要猜测，不要默认0，不要直接放行而不记录").
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


@dataclass
class OrderIntent:
    """What the engine evaluates. side is "BUY"/"SELL". qty/price are always
    positive magnitudes — direction comes from side, never from qty's sign."""
    code: str
    side: str
    qty: float
    price: float
    strategy: str = ""

    @property
    def notional(self) -> float:
        return abs(self.qty) * abs(self.price)

    @property
    def signed_notional(self) -> float:
        """+notional for BUY, -notional for SELL — the delta this order
        applies to a market-value figure (position mv, sector mv, total
        exposure mv)."""
        return self.notional if self.side == "BUY" else -self.notional

    @property
    def signed_qty(self) -> float:
        return abs(self.qty) if self.side == "BUY" else -abs(self.qty)


@dataclass
class RiskDecision:
    status: RiskStatus
    allowed: bool
    reason: str
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_log_dict(self, order: Optional[OrderIntent] = None) -> Dict[str, Any]:
        d = {
            "timestamp": self.timestamp,
            "status": self.status.value if isinstance(self.status, RiskStatus) else self.status,
            "allowed": self.allowed,
            "reason": self.reason,
            "violations": list(self.violations),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }
        if order is not None:
            d.update({
                "code": order.code, "side": order.side,
                "qty": order.qty, "price": order.price,
                "strategy": order.strategy,
            })
        return d


def allow(reason: str = "", metrics: Optional[Dict[str, Any]] = None,
          warnings: Optional[List[str]] = None) -> RiskDecision:
    return RiskDecision(status=RiskStatus.ALLOW, allowed=True, reason=reason,
                         warnings=list(warnings or []), metrics=dict(metrics or {}))


def warn(reason: str, metrics: Optional[Dict[str, Any]] = None,
         warnings: Optional[List[str]] = None) -> RiskDecision:
    return RiskDecision(status=RiskStatus.WARN, allowed=True, reason=reason,
                         warnings=list(warnings or [reason]), metrics=dict(metrics or {}))


def block(reason: str, violations: Optional[List[str]] = None,
          metrics: Optional[Dict[str, Any]] = None) -> RiskDecision:
    return RiskDecision(status=RiskStatus.BLOCK, allowed=False, reason=reason,
                         violations=list(violations or [reason]), metrics=dict(metrics or {}))
