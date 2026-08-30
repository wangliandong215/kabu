"""
portfolio/broker_state.py — Broker ground-truth state (v2.10).

Pure data-fetch module: queries moomoo OpenD directly for real positions,
cash, and total assets. This is the ONLY source risk/portfolio_risk_manager.py
is allowed to use for exposure/cash decisions — never portfolio/tracker.py's
cost-basis model (see that module's docstring for why: it doesn't move on
unrealized P&L, and has been observed to drift from the real account).

No caching, no writes — fetch_broker_state() hits OpenD fresh every call.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from common import make_trade_ctx, retry, safe_close


@dataclass
class BrokerPosition:
    code: str
    qty: float
    cost_price: float
    market_val: float
    current_price: float


@dataclass
class BrokerState:
    positions: Dict[str, BrokerPosition] = field(default_factory=dict)
    cash: float = 0.0
    total_assets: float = 0.0
    long_mv: float = 0.0
    fetched_at: Optional[datetime] = None


@retry()
def fetch_broker_state(trd_env) -> BrokerState:
    """Query real positions + account info from moomoo OpenD (US market,
    SECURITY_FIRM per config). Raises on API failure — @retry (config.API_RETRY_*)
    retries transient errors, then the exception propagates to the caller,
    which must treat "no broker state" as an UNKNOWN/conservative tier, never
    silently fall back to tracker.json."""
    import moomoo as ft

    ctx = make_trade_ctx("US")
    try:
        pos_ret, pos_data = ctx.position_list_query(trd_env=trd_env)
        if pos_ret != ft.RET_OK:
            raise RuntimeError(f"position_list_query failed: {pos_data}")

        acc_ret, acc_data = ctx.accinfo_query(trd_env=trd_env)
        if acc_ret != ft.RET_OK:
            raise RuntimeError(f"accinfo_query failed: {acc_data}")
    finally:
        safe_close(ctx)

    positions: Dict[str, BrokerPosition] = {}
    for _, row in pos_data.iterrows():
        code = str(row["code"])
        positions[code] = BrokerPosition(
            code=code,
            qty=float(row.get("qty", 0) or 0),
            cost_price=float(row.get("cost_price", 0) or 0),
            market_val=float(row.get("market_val", 0) or 0),
            current_price=float(row.get("nominal_price", 0) or 0),
        )

    acc_row = acc_data.iloc[0]
    return BrokerState(
        positions=positions,
        cash=float(acc_row.get("cash", 0) or 0),
        total_assets=float(acc_row.get("total_assets", 0) or 0),
        long_mv=float(acc_row.get("long_mv", 0) or 0),
        fetched_at=datetime.now(),
    )
