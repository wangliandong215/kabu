"""
risk/portfolio_state.py — V3.3 Portfolio Risk Engine's portfolio snapshot.

Built ONCE per engine/runner.py::run_once() pass (never per order) from the
same portfolio/broker_state.py::BrokerState the v2.10 Portfolio Risk Manager
already fetches — mark-to-market ground truth, same convention as
risk/portfolio_risk_manager.py and risk/portfolio_position_manager.py (see
their docstrings for why: broker state, never portfolio/tracker.py's
cost-basis figures, for risk decisions).

strategy tags (used only to replicate guard.can_open_position()'s existing
"QQQ core doesn't count toward MAX_POSITIONS" exclusion) come from
portfolio/tracker.py — same dual-source pattern
risk/portfolio_risk_manager.py::plan_rebalance() already uses
(broker_state + tracker_positions together).

price_returns/market_regime start unset (None) and are attached in place as
the run_once() pass progresses — see risk/price_history.py and
risk/market_regime_risk.py. This lets exit-loop SELLs (which happen before
regime classification in run_once()) still be evaluated correctly: sells
never need regime data to be non-blockable (see position_limits_risk.py /
exposure_risk.py's projection math).
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import config
from portfolio.broker_state import BrokerState


@dataclass
class PositionSnapshot:
    code: str
    qty: float
    market_val: float
    current_price: float
    weight: float           # market_val / total_assets
    sector: str
    is_core: bool            # True for the QQQ core holding (excluded from MAX_POSITIONS)


@dataclass
class PortfolioState:
    positions: Dict[str, PositionSnapshot] = field(default_factory=dict)
    cash: float = 0.0
    total_assets: float = 0.0
    long_mv: float = 0.0
    total_exposure_pct: Optional[float] = None
    position_count: int = 0   # excludes QQQ core, same convention as guard.can_open_position()

    market_regime: Optional[str] = None   # attached later in the pass — see module docstring
    price_returns: Optional[Dict[str, "object"]] = None  # code -> pandas.Series of daily returns, lazily attached

    def get(self, code: str) -> Optional[PositionSnapshot]:
        return self.positions.get(code)


def build_portfolio_state(broker_state: Optional[BrokerState], tracker=None) -> PortfolioState:
    """Pure. broker_state=None (fetch failed this pass) -> an empty/zeroed
    PortfolioState with total_exposure_pct=None, never a guessed number —
    callers (risk_engine sub-checks) must treat that as DATA_UNAVAILABLE,
    not as "0% exposure"."""
    if broker_state is None or broker_state.total_assets is None or broker_state.total_assets <= 0:
        return PortfolioState(
            cash=getattr(broker_state, "cash", 0.0) or 0.0,
            total_assets=getattr(broker_state, "total_assets", 0.0) or 0.0,
            long_mv=getattr(broker_state, "long_mv", 0.0) or 0.0,
            total_exposure_pct=None,
            position_count=0,
        )

    tracker_strategy: Dict[str, str] = {}
    if tracker is not None:
        try:
            for code, pos in tracker.data.get("positions", {}).items():
                tracker_strategy[code] = pos.get("strategy", "")
        except Exception:
            tracker_strategy = {}

    positions: Dict[str, PositionSnapshot] = {}
    position_count = 0
    for code, bp in broker_state.positions.items():
        if bp.qty == 0:
            continue
        is_core = (code == config.QQQ_CORE_CODE) or (tracker_strategy.get(code) == "core_etf")
        sector = config.SECTOR_MAP.get(code, "other")
        weight = bp.market_val / broker_state.total_assets
        positions[code] = PositionSnapshot(
            code=code, qty=bp.qty, market_val=bp.market_val,
            current_price=bp.current_price, weight=weight, sector=sector,
            is_core=is_core,
        )
        if not is_core:
            position_count += 1

    return PortfolioState(
        positions=positions,
        cash=broker_state.cash,
        total_assets=broker_state.total_assets,
        long_mv=broker_state.long_mv,
        total_exposure_pct=broker_state.long_mv / broker_state.total_assets,
        position_count=position_count,
    )
