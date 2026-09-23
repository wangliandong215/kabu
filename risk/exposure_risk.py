"""
risk/exposure_risk.py — V3.3 projected total-exposure check.

Independent of risk/portfolio_risk_manager.py's own 95/100/105% tiers (that
module stays frozen, see its docstring) — this is the Risk Engine's own
"stop increasing risk further" gate: BLOCK only fires when the order itself
pushes projected exposure over the limit AND the order increases exposure.
A SELL's signed_notional is negative, so projected_exposure_pct can only be
<= current — a SELL can never trip this BLOCK, no special-casing needed,
just correct projection math (see OrderIntent.signed_notional).
"""
from dataclasses import dataclass
from typing import Optional

import config
from risk.risk_decision import DATA_UNAVAILABLE, OrderIntent, allow, block
from risk.portfolio_state import PortfolioState

VIOLATION = "MAX_TOTAL_EXPOSURE"


@dataclass
class ExposureCheckResult:
    status: str
    current_exposure_pct: Optional[float]
    projected_exposure_pct: Optional[float]
    limit_pct: float


def check(order: OrderIntent, state: PortfolioState):
    limit = config.RISK_ENGINE_MAX_TOTAL_EXPOSURE_PCT

    if state.total_assets is None or state.total_assets <= 0 or state.total_exposure_pct is None:
        result = ExposureCheckResult(status=DATA_UNAVAILABLE, current_exposure_pct=None,
                                      projected_exposure_pct=None, limit_pct=limit)
        return result, allow("total_assets unavailable this pass", metrics={"exposure": result})

    projected_mv = state.long_mv + order.signed_notional
    projected_pct = projected_mv / state.total_assets

    result = ExposureCheckResult(status="OK", current_exposure_pct=state.total_exposure_pct,
                                  projected_exposure_pct=projected_pct, limit_pct=limit)

    if order.side == "BUY" and projected_pct > limit:
        result.status = "BLOCK"
        return result, block(
            f"projected total exposure {projected_pct:.1%} exceeds limit {limit:.1%}",
            violations=[VIOLATION],
            metrics={"exposure": result},
        )

    return result, allow(metrics={"exposure": result})
