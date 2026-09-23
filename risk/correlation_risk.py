"""
risk/correlation_risk.py — V3.3 portfolio correlation monitoring.

No correlation logic existed anywhere in this codebase before V3.3
(confirmed by repo-wide search) — genuinely new. Per spec section 十一, this
is explicitly SOFT RISK for Phase 1: WARN only, never BLOCK, regardless of
config. A future V3.4 (correlation cluster / effective exposure / factor
exposure) is out of scope here.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

import config
from risk.risk_decision import OrderIntent, allow, warn
from risk.portfolio_state import PortfolioState

WARNING = "HIGH_CORRELATION"


@dataclass
class CorrelationCheckResult:
    status: str
    max_correlation: Optional[float]
    against: Optional[str]
    threshold: float
    pairs: Dict[str, float]


def pairwise_correlation(returns: Dict[str, pd.Series], code_a: str, code_b: str) -> Optional[float]:
    """Pure. None if either series is missing or overlap is too short."""
    if code_a not in returns or code_b not in returns:
        return None
    a, b = returns[code_a], returns[code_b]
    n = min(len(a), len(b))
    if n < config.RISK_ENGINE_CORRELATION_MIN_BARS:
        return None
    corr = a.tail(n).reset_index(drop=True).corr(b.tail(n).reset_index(drop=True))
    if corr is None or pd.isna(corr):
        return None
    return float(corr)


def check(order: OrderIntent, state: PortfolioState):
    threshold = config.RISK_ENGINE_CORRELATION_WARN_THRESHOLD

    if order.side != "BUY" or order.qty == 0 or not state.price_returns:
        result = CorrelationCheckResult(status="OK", max_correlation=None, against=None,
                                         threshold=threshold, pairs={})
        return result, allow(metrics={"correlation": result})

    compare_against: List[str] = list(state.positions.keys())
    if config.QQQ_CORE_CODE not in compare_against:
        compare_against.append(config.QQQ_CORE_CODE)

    pairs: Dict[str, float] = {}
    for other in compare_against:
        if other == order.code:
            continue
        c = pairwise_correlation(state.price_returns, order.code, other)
        if c is not None:
            pairs[other] = c

    if not pairs:
        result = CorrelationCheckResult(status="OK", max_correlation=None, against=None,
                                         threshold=threshold, pairs={})
        return result, allow(metrics={"correlation": result})

    worst_code = max(pairs, key=lambda k: abs(pairs[k]))
    worst = pairs[worst_code]

    result = CorrelationCheckResult(status="OK", max_correlation=worst, against=worst_code,
                                     threshold=threshold, pairs=pairs)

    if abs(worst) >= threshold:
        result.status = "WARN"
        return result, warn(
            f"{order.code} correlation with {worst_code} ({worst:.2f}) at/above threshold {threshold:.2f}",
            warnings=[WARNING],
            metrics={"correlation": result},
        )

    return result, allow(metrics={"correlation": result})
