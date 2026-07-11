"""
engine/market_hours.py — Market hours filter.

Prevents wasting API calls on stocks whose markets are closed.
All times are in JST (UTC+9).

Market sessions (JST):
  US stocks  — summer (EDT): 22:30 ~ next day 05:00
             — winter (EST): 23:30 ~ next day 06:00
  JP stocks  — morning:  09:00 ~ 11:30
             — afternoon: 12:30 ~ 15:30

DST: US daylight saving is active from 2nd Sunday of March to
1st Sunday of November (approximately April–October in JST terms).
"""
from datetime import datetime, time as dtime, date, timedelta
import pytz

_JST = pytz.timezone("Asia/Tokyo")
_US_SUMMER_OPEN  = dtime(22, 30)   # EDT
_US_SUMMER_CLOSE = dtime( 5,  0)   # next day
_US_WINTER_OPEN  = dtime(23, 30)   # EST
_US_WINTER_CLOSE = dtime( 6,  0)   # next day
_JP_MORNING_OPEN  = dtime( 9,  0)
_JP_MORNING_CLOSE = dtime(11, 30)
_JP_AFTERNOON_OPEN  = dtime(12, 30)
_JP_AFTERNOON_CLOSE = dtime(15, 30)


def is_us_dst(d: date) -> bool:
    """Approximate US DST: active roughly April 1 – October 31 in JST."""
    return 4 <= d.month <= 10


def is_open(code: str) -> bool:
    """True if the stock's home market is currently in trading hours."""
    now_jst = datetime.now(_JST)
    market = code.split(".")[0].upper() if "." in code else "US"

    if market == "JP":
        # JP sessions never cross midnight, so a blanket weekend guard is safe.
        if now_jst.weekday() >= 5:
            return False
        return _in_jp_session(now_jst.time())

    # US (and default): the session opens 22:30/23:30 JST and runs past
    # midnight, so a naive weekday()>=5 guard here would wrongly block the
    # Saturday-early-morning JST tail of Friday's still-open session.
    # _in_us_session() checks the weekday of the session's *opening* JST
    # calendar day instead of the current one.
    return _in_us_session(now_jst.time(), now_jst.date())


def filter_open(codes: list) -> list:
    """Return only the codes whose market is currently open."""
    return [c for c in codes if is_open(c)]


def is_daytime_jst(now: datetime = None) -> bool:
    """True during the JP trading-day window (09:00-15:30 JST, including
    the lunch break) — used to pick which regional watchlist to scan by
    default. Everything outside this window (evening through next
    morning) is treated as the EU/US trading night. This is a coarse
    day/night split for choosing *which pool* to scan; per-stock
    is_open()/filter_open() still gate whether an order can actually be
    placed."""
    if now is None:
        now = datetime.now(_JST)
    elif now.tzinfo is None:
        now = _JST.localize(now)
    return _JP_MORNING_OPEN <= now.time() < _JP_AFTERNOON_CLOSE


def just_closed(minutes_after: int = 5, window_minutes: int = 10,
                 now: datetime = None) -> bool:
    """True once `now` (defaults to current JST time) is `minutes_after`
    past today's US market close, for a `window_minutes`-wide window —
    used to fire a one-time post-close notification without needing
    exact alignment with the polling interval."""
    if now is None:
        now = datetime.now(_JST)
    elif now.tzinfo is None:
        now = _JST.localize(now)

    close_t = _US_SUMMER_CLOSE if is_us_dst(now.date()) else _US_WINTER_CLOSE
    notify_start = _JST.localize(datetime.combine(now.date(), close_t)) + timedelta(minutes=minutes_after)
    notify_end = notify_start + timedelta(minutes=window_minutes)
    return notify_start <= now < notify_end


_last_close_notified: date = None


def should_notify_close() -> bool:
    """Stateful wrapper around just_closed(): returns True at most once
    per calendar day (JST), the first time a caller polls inside the
    post-close notification window. Callers don't need to track dates
    themselves — just call this once per pass and act if it returns True."""
    global _last_close_notified
    now = datetime.now(_JST)
    if just_closed(now=now) and _last_close_notified != now.date():
        _last_close_notified = now.date()
        return True
    return False


def _in_us_session(t: dtime, d: date) -> bool:
    """Session spans midnight (open_ > close_), so whether `t` falls inside
    it depends on which JST calendar day the *opening* half belongs to —
    not on today's weekday. `t < close_` is the tail end of a session that
    opened on `d - 1 day`; `t >= open_` is a session opening on `d` itself."""
    dst = is_us_dst(d)
    open_  = _US_SUMMER_OPEN  if dst else _US_WINTER_OPEN
    close_ = _US_SUMMER_CLOSE if dst else _US_WINTER_CLOSE

    if t < close_:
        session_day = d - timedelta(days=1)
        return session_day.weekday() < 5
    if t >= open_:
        return d.weekday() < 5
    return False


def _in_jp_session(t: dtime) -> bool:
    morning   = _JP_MORNING_OPEN   <= t < _JP_MORNING_CLOSE
    afternoon = _JP_AFTERNOON_OPEN <= t < _JP_AFTERNOON_CLOSE
    return morning or afternoon
