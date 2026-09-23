"""
risk/sector_risk.py — V3.3 projected sector exposure.

guard.check_sector_exposure() already exists but only checks CURRENT sector
exposure before adding the order's own size — this module is what actually
implements V3.3 spec section 七's "projected_sector_exposure" requirement:
simulate current + proposed -> projected, then check.

Reuses config.SECTOR_MAP (same map guard.py and portfolio/tracker.py already
use) rather than introducing a second sector taxonomy. First-version static
config per V3.3 spec ("第一版允许行业分类数据采用静态配置") — architecture
leaves room for a future dynamic classifier by keeping the sector lookup as
a single seam (_sector_of()) that could later call out instead of dict.get().
"""
from dataclasses import dataclass
from typing import Optional

import config
from risk.risk_decision import OrderIntent, allow, block, warn
from risk.portfolio_state import PortfolioState

VIOLATION = "MAX_SECTOR_EXPOSURE"
WARNING = "SECTOR_EXPOSURE_NEAR_LIMIT"


def _sector_of(code: str) -> str:
    return config.SECTOR_MAP.get(code, "other")


def _limit_for(sector: str) -> float:
    table = config.RISK_ENGINE_MAX_SECTOR_EXPOSURE_PCT
    return table.get(sector, table.get("default", 1.0))


@dataclass
class SectorCheckResult:
    status: str
    sector: str
    current_sector_pct: float
    projected_sector_pct: Optional[float]
    limit_pct: float


def check(order: OrderIntent, state: PortfolioState):
    sector = _sector_of(order.code)
    limit = _limit_for(sector)

    if state.total_assets is None or state.total_assets <= 0:
        result = SectorCheckResult(status="OK", sector=sector, current_sector_pct=0.0,
                                    projected_sector_pct=None, limit_pct=limit)
        return result, allow("total_assets unavailable this pass", metrics={"sector": result})

    sector_mv = sum(p.market_val for p in state.positions.values() if p.sector == sector)
    current_pct = sector_mv / state.total_assets
    projected_mv = sector_mv + order.signed_notional
    projected_pct = projected_mv / state.total_assets

    result = SectorCheckResult(status="OK", sector=sector, current_sector_pct=current_pct,
                                projected_sector_pct=projected_pct, limit_pct=limit)

    if order.side != "BUY" or order.qty == 0:
        return result, allow(metrics={"sector": result})

    if projected_pct > limit:
        result.status = "BLOCK"
        return result, block(
            f"{sector} projected exposure {projected_pct:.1%} exceeds limit {limit:.1%}",
            violations=[VIOLATION],
            metrics={"sector": result},
        )

    if projected_pct > limit * config.RISK_ENGINE_SECTOR_WARN_RATIO:
        result.status = "WARN"
        return result, warn(
            f"{sector} projected exposure {projected_pct:.1%} approaching limit {limit:.1%}",
            warnings=[WARNING],
            metrics={"sector": result},
        )

    return result, allow(metrics={"sector": result})
