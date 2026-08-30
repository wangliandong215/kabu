"""
engine/market_preflight.py — OpenD execution preflight check.

A second, right-before-submission layer of confidence on top of
engine/market_hours.is_open() (which only reasons about the clock/calendar).
Confirms the live OpenD connection, quote/trade login state, and the
target market's actual reported state, all fetched fresh via
get_global_state() — no caching, since this exists specifically to catch
"the clock says open but OpenD/the broker disagrees" right before an order
goes out.

Pure judgment function, no side effects (same style as
risk/portfolio_risk_manager.py) — callers decide what to log/alert on a
failed result.
"""
from dataclasses import dataclass
from typing import Optional

import common
import config
import engine.market_hours as market_hours

# code market prefix -> get_global_state() dict key
_MARKET_STATE_KEY = {
    "US": "market_us", "HK": "market_hk", "SH": "market_sh", "SZ": "market_sz",
    "JP": "market_jp", "SG": "market_sg", "MY": "market_my",
}


def _is_true(v) -> bool:
    """get_global_state()'s qot_logined/trd_logined come back as a native
    Python bool against a live OpenD (confirmed 2026-08-30) despite the
    SDK's own doc/skill-script comment describing them as '1'/'0' strings —
    accept both forms defensively rather than trusting one."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true")


@dataclass
class PreflightResult:
    ok: bool
    reason: str = ""


def check(code: str) -> PreflightResult:
    """Run all preflight checks for `code`. Any failure returns ok=False
    with a human-readable reason; never raises."""
    if not market_hours.is_open(code):
        return PreflightResult(False, "market_hours: closed")

    try:
        state = _fetch_global_state()
    except Exception as exc:
        return PreflightResult(False, f"get_global_state exception: {exc}")

    if state is None:
        return PreflightResult(False, "get_global_state failed")

    if not _is_true(state.get("qot_logined")):
        return PreflightResult(False, "quote context not logged in")
    if not _is_true(state.get("trd_logined")):
        return PreflightResult(False, "trade context not logged in")

    market = common.infer_market(code)
    key = _MARKET_STATE_KEY.get(market)
    if key:
        market_state = str(state.get(key, ""))
        if market_state in config.PREFLIGHT_BLOCKED_MARKET_STATES:
            return PreflightResult(False, f"{key}={market_state} not tradable")

    return PreflightResult(True, "ok")


@common.retry()
def _fetch_global_state() -> Optional[dict]:
    from moomoo import RET_OK
    ctx = common.make_quote_ctx()
    try:
        ret, data = ctx.get_global_state()
        if ret != RET_OK:
            return None
        return dict(data)
    finally:
        common.safe_close(ctx)
