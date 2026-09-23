"""
risk/position_limits_risk.py — V3.3 max position count + max single-position
weight, both computed against the PROJECTED portfolio (order applied), not
just current state — see V3.3 spec section 六 ("必须模拟 current + proposed
-> projected"). guard.py's existing can_open_position()/check_sector_exposure()
only look at current state; this module is what actually fixes that gap.

Edge cases (V3.3 spec section 二十, this module's share of them):
  - New open (current_qty==0, BUY): counts toward MAX_POSITIONS.
  - Add-to-existing (current_qty>0, BUY): does NOT create a new position,
    never blocked by MAX_POSITIONS, only by MAX_POSITION_WEIGHT.
  - Any SELL (partial or full close): projected_qty <= current_qty, so
    projected weight can only shrink and projected count can only shrink or
    stay the same — MAX_POSITIONS/MAX_POSITION_WEIGHT can mathematically
    never BLOCK a SELL. No side=="SELL" special-casing needed; this falls
    out of the projection math below, and is asserted directly in tests.
  - qty==0: notional==0, projected==current, always ALLOW.
"""
from dataclasses import dataclass
from typing import Optional

import config
from risk.risk_decision import OrderIntent, allow, block
from risk.portfolio_state import PortfolioState

VIOLATION_COUNT = "MAX_POSITIONS"
VIOLATION_WEIGHT = "MAX_POSITION_WEIGHT"


@dataclass
class PositionCountResult:
    status: str
    current_count: int
    projected_count: int
    limit: int


@dataclass
class PositionWeightResult:
    status: str
    current_weight: float
    projected_weight: Optional[float]
    limit_pct: float


def check_position_count(order: OrderIntent, state: PortfolioState):
    limit = config.RISK_ENGINE_MAX_POSITIONS
    pos = state.get(order.code)
    is_new_symbol = pos is None or pos.qty == 0
    is_core = pos.is_core if pos is not None else (order.code == config.QQQ_CORE_CODE)

    opens_new_position = (order.side == "BUY" and order.qty != 0 and is_new_symbol and not is_core)
    projected_count = state.position_count + (1 if opens_new_position else 0)

    result = PositionCountResult(status="OK", current_count=state.position_count,
                                  projected_count=projected_count, limit=limit)

    if opens_new_position and projected_count > limit:
        result.status = "BLOCK"
        return result, block(
            f"opening {order.code} would bring position count to {projected_count} "
            f"(limit {limit})",
            violations=[VIOLATION_COUNT],
            metrics={"position_count": result},
        )

    return result, allow(metrics={"position_count": result})


def check_position_weight(order: OrderIntent, state: PortfolioState):
    limit = config.RISK_ENGINE_MAX_POSITION_WEIGHT_PCT

    if state.total_assets is None or state.total_assets <= 0:
        result = PositionWeightResult(status="OK", current_weight=0.0,
                                       projected_weight=None, limit_pct=limit)
        return result, allow("total_assets unavailable this pass", metrics={"position_weight": result})

    pos = state.get(order.code)
    current_mv = pos.market_val if pos is not None else 0.0
    current_weight = current_mv / state.total_assets
    projected_mv = current_mv + order.signed_notional
    projected_weight = projected_mv / state.total_assets

    result = PositionWeightResult(status="OK", current_weight=current_weight,
                                   projected_weight=projected_weight, limit_pct=limit)

    if order.side == "BUY" and projected_weight > limit:
        result.status = "BLOCK"
        return result, block(
            f"{order.code} projected weight {projected_weight:.1%} exceeds limit {limit:.1%}",
            violations=[VIOLATION_WEIGHT],
            metrics={"position_weight": result},
        )

    return result, allow(metrics={"position_weight": result})
