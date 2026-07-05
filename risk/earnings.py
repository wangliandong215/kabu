"""
risk/earnings.py — Earnings blackout guard.

Forbids opening NEW positions in any stock within BLACKOUT_DAYS before
or after its earnings announcement.  Earnings gaps routinely gap through
stop-losses, making the 2%-risk-per-trade guarantee meaningless.

Data source: moomoo get_earnings_calendar() — no extra cost, uses
the existing OpenD connection.  Results cached for CACHE_TTL seconds.
"""
import time
from datetime import datetime, timedelta
from typing import Set

import notify.alert as alert

BLACKOUT_DAYS = 1      # ban window: ±1 day around earnings date
CACHE_TTL     = 6 * 3600   # refresh every 6 h (earnings dates don't change)
LOOKAHEAD     = 3      # fetch earnings up to N days ahead

_cache: dict = {"ts": 0.0, "blacklist": set()}


def is_earnings_blackout(code: str) -> bool:
    """
    True if the stock has earnings within BLACKOUT_DAYS from today.
    Returns False on any API error (fail-open so trading is not blocked).
    """
    bl = _get_blacklist()
    ticker = code.split(".")[-1] if "." in code else code
    return ticker in bl


def _get_blacklist() -> Set[str]:
    now = time.time()
    if now - _cache["ts"] < CACHE_TTL:
        return _cache["blacklist"]

    blacklist: Set[str] = set()
    today = datetime.now().date()
    start = today - timedelta(days=BLACKOUT_DAYS)
    end   = today + timedelta(days=LOOKAHEAD + BLACKOUT_DAYS)

    try:
        import moomoo as ft
        ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)
        try:
            for market in (ft.Market.US, ft.Market.JP):
                ret, data = ctx.get_earnings_calendar(
                    market=market,
                    begin_date=start.strftime("%Y-%m-%d"),
                    end_date=end.strftime("%Y-%m-%d"),
                )
                if ret != ft.RET_OK or data is None or data.empty:
                    continue
                for _, row in data.iterrows():
                    edate_str = str(row.get("earnings_date", ""))[:10]
                    try:
                        edate = datetime.strptime(edate_str, "%Y-%m-%d").date()
                    except ValueError:
                        continue
                    if abs((edate - today).days) <= BLACKOUT_DAYS:
                        sec = str(row.get("security", ""))
                        ticker = sec.split(".")[-1] if "." in sec else sec
                        if ticker:
                            blacklist.add(ticker)
        finally:
            ctx.close()
    except Exception as exc:
        alert.warn(f"earnings: failed to fetch calendar — {exc}")

    _cache["ts"]        = now
    _cache["blacklist"] = blacklist
    if blacklist:
        alert.info(f"earnings: blackout list ({len(blacklist)} stocks): "
                   f"{sorted(blacklist)[:10]}{'...' if len(blacklist)>10 else ''}")
    return blacklist
