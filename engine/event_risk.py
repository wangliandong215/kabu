"""
engine/event_risk.py — Earnings event risk judgment layer.

Turns raw earnings-calendar data (data/earnings.py) into a trading-decision
input: is a symbol within its earnings blackout window right now? This
module never talks to OpenD directly — all data acquisition lives in
data/earnings.py.

Fails open on any lookup error (returns False / None) — missing earnings
data must not halt trading, same fail-open philosophy risk/earnings.py used
before it was refactored to delegate here.

Scope: this layer only informs "should we avoid ADDING risk right now"
(new entries + pyramid adds). It never forces an exit of an existing
position — see engine/runner.py call sites.
"""
from datetime import datetime, date as date_cls, timedelta
from typing import Optional

import config
import data.earnings as earnings_data
import engine.market_hours as market_hours
import notify.alert as alert


def has_earnings_risk(code: str, trade_date: Optional[str] = None) -> bool:
    """True if `code` is within config.EVENT_RISK_EARNINGS_BLOCK_DAYS trading
    days (either side) of its nearest known earnings date."""
    if not config.EVENT_RISK_ENABLED:
        return False
    d = days_to_earnings(code, trade_date)
    if d is None:
        return False
    return abs(d) <= config.EVENT_RISK_EARNINGS_BLOCK_DAYS


def days_to_earnings(code: str, trade_date: Optional[str] = None) -> Optional[int]:
    """Signed trading-day distance from `trade_date` (default: today) to
    `code`'s nearest known earnings date (negative = already happened,
    positive = upcoming, 0 = today). None if no earnings date is known."""
    edate = earnings_date(code, trade_date)
    if edate is None:
        return None
    ref = _parse_date(trade_date) or datetime.now().date()
    return _trading_day_distance(code, ref, edate)


def earnings_date(code: str, trade_date: Optional[str] = None) -> Optional[date_cls]:
    """Nearest known earnings date for `code`, or None."""
    try:
        row = earnings_data.get_earnings_risk(code, trade_date)
    except Exception as exc:
        alert.warn(f"event_risk: {code} 财报日历查询失败 — {exc}")
        return None
    if not row:
        return None
    return _parse_date(row.get("earnings_date"))


def earnings_session(code: str) -> Optional[str]:
    """'BMO' (before market open) / 'AMC' (after market close) / None if
    unknown — pub_type's exact raw values weren't confirmed against a live
    OpenD response at implementation time, so this is a best-effort
    substring match, not an authoritative mapping."""
    try:
        row = earnings_data.get_earnings_risk(code)
    except Exception:
        return None
    if not row:
        return None
    pub = str(row.get("pub_type", "")).upper()
    if "BEFORE" in pub or pub == "BMO":
        return "BMO"
    if "AFTER" in pub or pub == "AMC":
        return "AMC"
    return None


# ── Internal ──────────────────────────────────────────────────────────────────

def _parse_date(s) -> Optional[date_cls]:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _trading_day_distance(code: str, start: date_cls, end: date_cls) -> int:
    """Signed count of trading days from `start` to `end` (0 if equal),
    walking calendar days one at a time and counting only the ones
    market_hours.is_trading_day() accepts. Capped by callers' small
    EVENT_RISK_EARNINGS_BLOCK_DAYS/LOOKAHEAD_DAYS values, so this never
    walks more than a couple weeks."""
    if start == end:
        return 0
    step = 1 if end > start else -1
    n = 0
    cur = start
    while cur != end:
        cur += timedelta(days=step)
        if market_hours.is_trading_day(code, cur):
            n += step
    return n
