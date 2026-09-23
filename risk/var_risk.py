"""
risk/var_risk.py — V3.3 Portfolio VaR (1-day, configurable confidence).

No VaR logic existed anywhere in this codebase before V3.3 (confirmed by
repo-wide search) — genuinely new. Per spec section 九, first version's job
is calculate/log/monitor/WARN, NOT auto-sell: RISK_ENGINE_VAR_BLOCK_PCT
defaults to None (never hard-blocks), and even when set, BLOCK is the only
thing that can happen here — no order is ever placed by this module.

Calculation method is pluggable (VarCalculator ABC) per spec's "不要把计算
方式写死在 Risk Engine 主逻辑里" — HistoricalVarCalculator (default) and
ParametricVarCalculator both implemented, selected by
config.RISK_ENGINE_VAR_METHOD.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import config
from risk.risk_decision import DATA_UNAVAILABLE, OrderIntent, allow, block, warn
from risk.portfolio_state import PortfolioState

WARNING_LEVEL = "VAR_ABOVE_WARNING_THRESHOLD"
VIOLATION = "VAR_ABOVE_HARD_THRESHOLD"
WARNING_DATA = "VAR_DATA_UNAVAILABLE"


class VarCalculator(ABC):
    @abstractmethod
    def compute(self, portfolio_returns: pd.Series, confidence: float) -> float:
        """Returns 1-day VaR as a POSITIVE fraction of portfolio value
        (e.g. 0.03 == 3% expected worst-case 1-day loss at `confidence`)."""
        raise NotImplementedError


class HistoricalVarCalculator(VarCalculator):
    """Empirical (1 - confidence) percentile of historical daily P&L."""
    def compute(self, portfolio_returns: pd.Series, confidence: float) -> float:
        if portfolio_returns is None or len(portfolio_returns) == 0:
            raise ValueError("empty return series")
        q = portfolio_returns.quantile(1.0 - confidence)
        return float(max(0.0, -q))


class ParametricVarCalculator(VarCalculator):
    """Variance-covariance (Gaussian) VaR: z_alpha * sigma."""
    _Z = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}

    def compute(self, portfolio_returns: pd.Series, confidence: float) -> float:
        if portfolio_returns is None or len(portfolio_returns) < 2:
            raise ValueError("insufficient return history")
        sigma = float(portfolio_returns.std())
        z = self._Z.get(round(confidence, 3), 1.6449)
        return max(0.0, z * sigma)


def _get_calculator() -> VarCalculator:
    if config.RISK_ENGINE_VAR_METHOD == "parametric":
        return ParametricVarCalculator()
    return HistoricalVarCalculator()


def _weighted_portfolio_returns(state: PortfolioState) -> Optional[pd.Series]:
    """Weighted sum of held symbols' daily return series, aligned on the
    shortest common length. None if price_returns is empty/missing, or if
    any held position's return series is unavailable (a partial weighted
    sum would understate real portfolio volatility — same principle as
    beta_risk.portfolio_beta())."""
    if not state.price_returns or state.total_assets is None or state.total_assets <= 0:
        return None
    if not state.positions:
        return pd.Series([0.0])

    series = []
    weights = []
    for code, pos in state.positions.items():
        r = state.price_returns.get(code)
        if r is None or len(r) == 0:
            return None
        series.append(r)
        weights.append(pos.market_val / state.total_assets)

    min_len = min(len(s) for s in series)
    if min_len == 0:
        return None
    aligned = [s.tail(min_len).reset_index(drop=True) for s in series]
    portfolio_returns = sum(w * s for w, s in zip(weights, aligned))
    return portfolio_returns


@dataclass
class VarCheckResult:
    status: str
    var_pct: Optional[float]
    confidence: float
    method: str
    warn_pct: float
    block_pct: Optional[float]


def check(order: OrderIntent, state: PortfolioState):
    confidence = config.RISK_ENGINE_VAR_CONFIDENCE
    warn_pct = config.RISK_ENGINE_VAR_WARN_PCT
    block_pct = config.RISK_ENGINE_VAR_BLOCK_PCT
    method = config.RISK_ENGINE_VAR_METHOD

    portfolio_returns = _weighted_portfolio_returns(state)
    if portfolio_returns is None:
        result = VarCheckResult(status=DATA_UNAVAILABLE, var_pct=None, confidence=confidence,
                                 method=method, warn_pct=warn_pct, block_pct=block_pct)
        return result, warn("VaR data unavailable this pass (missing return history)",
                             warnings=[WARNING_DATA], metrics={"var": result})

    try:
        var_pct = _get_calculator().compute(portfolio_returns, confidence)
    except Exception:
        result = VarCheckResult(status=DATA_UNAVAILABLE, var_pct=None, confidence=confidence,
                                 method=method, warn_pct=warn_pct, block_pct=block_pct)
        return result, warn("VaR calculation failed (insufficient/invalid data)",
                             warnings=[WARNING_DATA], metrics={"var": result})

    result = VarCheckResult(status="OK", var_pct=var_pct, confidence=confidence,
                             method=method, warn_pct=warn_pct, block_pct=block_pct)

    if block_pct is not None and var_pct > block_pct:
        result.status = "BLOCK"
        return result, block(
            f"portfolio VaR {var_pct:.1%} exceeds hard threshold {block_pct:.1%}",
            violations=[VIOLATION],
            metrics={"var": result},
        )

    if var_pct > warn_pct:
        result.status = "WARN"
        return result, warn(
            f"portfolio VaR {var_pct:.1%} above warning threshold {warn_pct:.1%}",
            warnings=[WARNING_LEVEL],
            metrics={"var": result},
        )

    return result, allow(metrics={"var": result})
