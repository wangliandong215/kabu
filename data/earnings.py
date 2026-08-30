"""
data/earnings.py — Earnings calendar fetching via moomoo get_earnings_calendar().

Thin OpenD data-acquisition layer only: fetches and caches the raw earnings
calendar. Risk judgment (is a trade blocked because of an upcoming earnings
date?) lives in engine/event_risk.py, which is built on top of this module —
this module must not import anything from engine/, risk/, portfolio/, or
strategies/.

get_earnings_calendar()'s begin_date/end_date window is capped at 7 days
apart by the API, so get_upcoming_earnings() paginates by market instead of
being called with one huge date range.

Timezone note (v2.11.1 audit): `earnings_date` is a plain 'YYYY-MM-DD'
string with no timezone marker from moomoo. Cross-checked against
`pub_type` (BeforeMarket/AfterMarket) — this is the US-market calendar date
of the announcement in US Eastern terms (the convention every US earnings
calendar provider uses), not JST or UTC. engine/event_risk.py's trading-day
math treats it as such.
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import common
import config

# code market prefix -> moomoo Market enum name
_MARKET_ENUM = {
    "US": "US", "HK": "HK", "SH": "SH", "SZ": "SZ",
    "SG": "SG", "JP": "JP", "AU": "AU", "CA": "CA",
}

_cache: Dict[str, dict] = {}   # market -> {"ts": float, "rows": list[dict]}


def get_upcoming_earnings(
    codes: List[str],
    begin_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[dict]:
    """
    Return raw earnings-calendar rows (as dicts) covering all markets present
    in `codes`, restricted to `codes` (by ticker). Results are cached per
    market for config.EVENT_RISK_CACHE_TTL_SECONDS.

    Row shape: {security, name, earnings_date, earnings_timestamp, pub_type,
    period_text, eps_actual, eps_predict, revenue_actual, revenue_predict,
    iv, iv_rank, iv_percentile, market_cap, price}.
    """
    markets = {common.infer_market(c) for c in codes}
    tickers = {_ticker(c) for c in codes}

    rows: List[dict] = []
    for market in markets:
        for row in _get_calendar(market, begin_date, end_date):
            sec = str(row.get("security", ""))
            if _ticker(sec) in tickers:
                rows.append(row)
    return rows


def get_earnings_risk(code: str, trade_date: Optional[str] = None) -> Optional[dict]:
    """
    Return the nearest earnings-calendar row for `code`'s ticker, or None if
    no earnings date is known within the fetched lookahead window.
    """
    market = common.infer_market(code)
    ticker = _ticker(code)
    rows = _get_calendar(market)
    matches = [r for r in rows if _ticker(str(r.get("security", ""))) == ticker]
    if not matches:
        return None
    # Rows are already restricted to the lookahead/lookback window fetched by
    # _get_calendar(); pick the one closest to trade_date (default: today).
    ref = _parse_date(trade_date) or datetime.now().date()
    matches.sort(key=lambda r: abs((_parse_date(r.get("earnings_date")) - ref).days)
                 if _parse_date(r.get("earnings_date")) else 10**6)
    return matches[0]


# ── Internal ──────────────────────────────────────────────────────────────────

def _ticker(code: str) -> str:
    return code.split(".")[-1] if "." in code else code


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _get_calendar(market: str, begin_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> List[dict]:
    cached = _cache.get(market)
    now = time.time()
    if cached and now - cached["ts"] < config.EVENT_RISK_CACHE_TTL_SECONDS and not begin_date:
        return cached["rows"]

    today = datetime.now().date()
    start = begin_date or (today - timedelta(days=1)).strftime("%Y-%m-%d")
    end = end_date or (today + timedelta(days=config.EVENT_RISK_LOOKAHEAD_DAYS)).strftime("%Y-%m-%d")

    rows = _fetch_calendar_window(market, start, end)

    if not begin_date:
        _cache[market] = {"ts": now, "rows": rows}
    return rows


@common.retry()
def _fetch_calendar_window(market: str, start: str, end: str) -> List[dict]:
    import moomoo as ft

    market_enum = getattr(ft.Market, _MARKET_ENUM.get(market, "US"), ft.Market.US)
    ctx = common.make_quote_ctx()
    rows: List[dict] = []
    try:
        # API caps the begin/end span at 7 days — walk the requested window
        # in <=7-day slices.
        cur = datetime.strptime(start, "%Y-%m-%d").date()
        end_d = datetime.strptime(end, "%Y-%m-%d").date()
        while cur <= end_d:
            slice_end = min(cur + timedelta(days=6), end_d)
            ret, data = ctx.get_earnings_calendar(
                market=market_enum,
                begin_date=cur.strftime("%Y-%m-%d"),
                end_date=slice_end.strftime("%Y-%m-%d"),
            )
            if ret == ft.RET_OK and data is not None and not data.empty:
                rows.extend(data.to_dict("records"))
            cur = slice_end + timedelta(days=1)
    finally:
        common.safe_close(ctx)
    return rows
