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
from datetime import datetime, time as dtime, date
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
    if now_jst.weekday() >= 5:   # Saturday / Sunday
        return False

    market = code.split(".")[0].upper() if "." in code else "US"

    if market == "JP":
        return _in_jp_session(now_jst.time())

    # US (and default)
    return _in_us_session(now_jst.time(), now_jst.date())


def filter_open(codes: list) -> list:
    """Return only the codes whose market is currently open."""
    return [c for c in codes if is_open(c)]


def _in_us_session(t: dtime, d: date) -> bool:
    dst = is_us_dst(d)
    open_  = _US_SUMMER_OPEN  if dst else _US_WINTER_OPEN
    close_ = _US_SUMMER_CLOSE if dst else _US_WINTER_CLOSE

    # Session spans midnight: open_ > close_
    if open_ > close_:
        return t >= open_ or t < close_
    return open_ <= t < close_


def _in_jp_session(t: dtime) -> bool:
    morning   = _JP_MORNING_OPEN   <= t < _JP_MORNING_CLOSE
    afternoon = _JP_AFTERNOON_OPEN <= t < _JP_AFTERNOON_CLOSE
    return morning or afternoon
