"""
risk/market_regime_risk.py — V3.3 Market Regime input.

Deliberately NOT a new regime model — this repo already has one taxonomy per
subsystem (risk/portfolio_position_manager.py's BULL/NORMAL/CAUTION/RISK_OFF,
regime/models.py's BULL/NEUTRAL/BEAR/CRASH observation layer, engine/
market_regime.py's volatility/trend regimes) and V3.3 spec section 十 is
explicit: "如果系统已经有 Market Regime 判断：直接接入现有模块". This module
is a thin adapter over risk.portfolio_position_manager.classify_market_regime()
— the taxonomy whose four states (BULL/NORMAL/CAUTION/RISK_OFF) line up
directly with spec's own BULL/NORMAL/RISK_OFF/UNKNOWN (CAUTION folds into
the WARN tier, an unclassifiable pass folds into UNKNOWN).

PortfolioState.market_regime is attached by engine/runner.py right after it
computes classify_market_regime() for v2.12 (same inputs, no second
computation) — this module only interprets that value, never recomputes it.
"""
from dataclasses import dataclass

import config
from risk.risk_decision import OrderIntent, allow, block, warn
from risk.portfolio_state import PortfolioState

VIOLATION = "MARKET_REGIME_RESTRICTS_NEW_POSITIONS"
WARNING = "MARKET_REGIME_CAUTION"

UNKNOWN = "UNKNOWN"


@dataclass
class RegimeCheckResult:
    status: str
    regime: str


def check(order: OrderIntent, state: PortfolioState):
    regime = state.market_regime or UNKNOWN

    result = RegimeCheckResult(status="OK", regime=regime)

    pos = state.get(order.code)
    opens_new_position = (order.side == "BUY" and order.qty != 0
                           and (pos is None or pos.qty == 0))

    if not opens_new_position:
        return result, allow(metrics={"market_regime": result})

    if regime in config.RISK_ENGINE_REGIME_RESTRICT_NEW_POSITIONS_ON:
        result.status = "BLOCK"
        return result, block(
            f"market regime {regime} restricts opening new positions",
            violations=[VIOLATION],
            metrics={"market_regime": result},
        )

    if regime in config.RISK_ENGINE_REGIME_WARN_ON:
        result.status = "WARN"
        return result, warn(
            f"market regime {regime} — elevated caution on new positions",
            warnings=[WARNING],
            metrics={"market_regime": result},
        )

    return result, allow(metrics={"market_regime": result})
