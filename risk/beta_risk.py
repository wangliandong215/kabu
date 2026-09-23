"""
risk/beta_risk.py — V3.3 portfolio / projected beta.

No beta calculation existed anywhere in this codebase before V3.3 — this is
genuinely new (confirmed by repo-wide search). Computed from real daily
return history (via risk/price_history.py, never a hand-maintained static
table) against config.QQQ_CORE_CODE as the market proxy, since QQQ is
already this system's own reference index (used the same way by
risk/portfolio_position_manager.py's qqq_above_ma regime input).

Per V3.3 spec section 八: insufficient/missing history is NEVER guessed to
0 or 1, and never silently allowed through un-recorded — it produces
DATA_UNAVAILABLE, which config.RISK_ENGINE_BETA_DATA_MODE then maps to
WARN ("safe", default) or BLOCK ("strict").
"""
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

import config
from risk.risk_decision import DATA_UNAVAILABLE, OrderIntent, allow, block, warn
from risk.portfolio_state import PortfolioState

VIOLATION = "MAX_PORTFOLIO_BETA"
WARNING_DATA = "BETA_DATA_UNAVAILABLE"


@dataclass
class BetaCheckResult:
    status: str
    portfolio_beta: Optional[float]
    projected_beta: Optional[float]
    limit: float
    missing_codes: list


def symbol_beta(returns: Dict[str, pd.Series], code: str,
                 market_code: Optional[str] = None) -> Optional[float]:
    """Pure. None if code or the market proxy is missing from `returns`, or
    fewer than config.RISK_ENGINE_BETA_MIN_BARS overlapping bars exist."""
    market_code = market_code or config.QQQ_CORE_CODE
    if code not in returns or market_code not in returns:
        return None
    r = returns[code]
    m = returns[market_code]
    n = min(len(r), len(m))
    if n < config.RISK_ENGINE_BETA_MIN_BARS:
        return None
    r = r.tail(n).reset_index(drop=True)
    m = m.tail(n).reset_index(drop=True)
    market_var = m.var()
    if market_var is None or pd.isna(market_var) or market_var == 0:
        return None
    cov = r.cov(m)
    if cov is None or pd.isna(cov):
        return None
    return float(cov / market_var)


def portfolio_beta(state: PortfolioState) -> "tuple[Optional[float], list]":
    """Market-value-weighted sum of held symbols' betas. Returns
    (beta_or_None, missing_codes) — beta is None if ANY held (non-core, i.e.
    it moves the risk number) position's beta is unavailable, since a
    partial weighted sum would understate real portfolio beta."""
    if not state.price_returns or state.total_assets is None or state.total_assets <= 0:
        return None, [c for c in state.positions.keys()]

    missing = []
    weighted_sum = 0.0
    for code, pos in state.positions.items():
        b = symbol_beta(state.price_returns, code)
        if b is None:
            missing.append(code)
            continue
        weighted_sum += b * (pos.market_val / state.total_assets)

    if missing:
        return None, missing
    return weighted_sum, []


def check(order: OrderIntent, state: PortfolioState):
    limit = config.RISK_ENGINE_MAX_PORTFOLIO_BETA
    strict = config.RISK_ENGINE_BETA_DATA_MODE == "strict"

    current_beta, missing = portfolio_beta(state)

    if current_beta is None or not state.price_returns:
        result = BetaCheckResult(status=DATA_UNAVAILABLE, portfolio_beta=None,
                                  projected_beta=None, limit=limit, missing_codes=missing)
        reason = f"beta data unavailable for: {missing or [order.code]}"
        if strict:
            return result, block(reason, violations=[WARNING_DATA], metrics={"beta": result})
        return result, warn(reason, warnings=[WARNING_DATA], metrics={"beta": result})

    order_beta = symbol_beta(state.price_returns, order.code)
    if order_beta is None:
        result = BetaCheckResult(status=DATA_UNAVAILABLE, portfolio_beta=current_beta,
                                  projected_beta=None, limit=limit, missing_codes=[order.code])
        reason = f"beta data unavailable for {order.code}"
        if strict:
            return result, block(reason, violations=[WARNING_DATA], metrics={"beta": result})
        return result, warn(reason, warnings=[WARNING_DATA], metrics={"beta": result})

    current_mv = sum(p.market_val for p in state.positions.values())
    projected_total_mv = current_mv + order.signed_notional
    if projected_total_mv <= 0 or state.total_assets is None or state.total_assets <= 0:
        projected_beta = current_beta
    else:
        current_beta_mv = current_beta * current_mv
        projected_beta_mv = current_beta_mv + order_beta * order.signed_notional
        projected_beta = projected_beta_mv / projected_total_mv

    result = BetaCheckResult(status="OK", portfolio_beta=current_beta,
                              projected_beta=projected_beta, limit=limit, missing_codes=[])

    if order.side == "BUY" and projected_beta > limit:
        result.status = "BLOCK"
        return result, block(
            f"projected portfolio beta {projected_beta:.2f} exceeds limit {limit:.2f}",
            violations=[VIOLATION],
            metrics={"beta": result},
        )

    return result, allow(metrics={"beta": result})
