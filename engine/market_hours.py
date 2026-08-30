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

Holidays: _JP_HOLIDAYS and _US_HOLIDAYS below are hand-maintained lists of
JPX / NYSE market closure dates, currently covering 2026-2030. Both only
cover the year(s) explicitly listed — sources:
  JP: https://www.jpx.co.jp/english/corporate/about-jpx/calendar/
  US: https://www.nyse.com/markets/hours-calendars
Extend them before this range runs out, or is_open()/is_daytime_jst() will
silently treat holidays outside the listed range as regular trading days
(weekday+session-time check only, same as before these lists existed).
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

_JP_HOLIDAYS = frozenset(date.fromisoformat(d) for d in [
    # 2026 JPX market closures (national holidays + year-end/New Year)
    "2026-01-01", "2026-01-02", "2026-01-03",   # New Year
    "2026-01-12",   # Coming of Age Day
    "2026-02-11",   # National Foundation Day
    "2026-02-23",   # Emperor's Birthday
    "2026-03-20",   # Vernal Equinox Day
    "2026-04-29",   # Showa Day
    "2026-05-04",   # Greenery Day
    "2026-05-05",   # Children's Day
    "2026-05-06",   # Constitution Memorial Day (observed)
    "2026-07-20",   # Marine Day
    "2026-08-11",   # Mountain Day
    "2026-09-21",   # Respect for the Aged Day
    "2026-09-22",   # Citizens' Holiday (between two holidays)
    "2026-09-23",   # Autumnal Equinox Day
    "2026-10-12",   # Sports Day
    "2026-11-03",   # Culture Day
    "2026-11-23",   # Labor Thanksgiving Day
    "2026-12-31",   # Year-End market closure

    # 2027 — Vernal Equinox/National Foundation Day fall on Sunday this
    # year, so both carry a Monday substitute holiday (furikae kyujitsu).
    "2027-01-01", "2027-01-02", "2027-01-03",   # New Year
    "2027-01-11",   # Coming of Age Day
    "2027-02-11",   # National Foundation Day
    "2027-02-23",   # Emperor's Birthday
    "2027-03-22",   # Vernal Equinox Day (observed; Mar 21 is Sunday)
    "2027-04-29",   # Showa Day
    "2027-05-03",   # Constitution Memorial Day
    "2027-05-04",   # Greenery Day
    "2027-05-05",   # Children's Day
    "2027-07-19",   # Marine Day
    "2027-08-11",   # Mountain Day
    "2027-09-20",   # Respect for the Aged Day
    "2027-09-23",   # Autumnal Equinox Day
    "2027-10-11",   # Sports Day
    "2027-11-03",   # Culture Day
    "2027-11-23",   # Labor Thanksgiving Day
    "2027-12-31",   # Year-End market closure

    # 2028 — Jan 1 and several fixed holidays land on Saturday (no
    # substitute under JP law — furikae only applies to Sunday), so those
    # dates are redundant with the weekend guard but harmless to list.
    "2028-01-01", "2028-01-02", "2028-01-03",   # New Year
    "2028-01-10",   # Coming of Age Day
    "2028-02-11",   # National Foundation Day
    "2028-02-23",   # Emperor's Birthday
    "2028-03-20",   # Vernal Equinox Day
    "2028-04-29",   # Showa Day (Saturday)
    "2028-05-03",   # Constitution Memorial Day
    "2028-05-04",   # Greenery Day
    "2028-05-05",   # Children's Day
    "2028-07-17",   # Marine Day
    "2028-08-11",   # Mountain Day
    "2028-09-18",   # Respect for the Aged Day
    "2028-09-22",   # Autumnal Equinox Day
    "2028-10-09",   # Sports Day
    "2028-11-03",   # Culture Day
    "2028-11-23",   # Labor Thanksgiving Day
    "2028-12-31",   # Year-End market closure

    # 2029 — National Foundation Day, Showa Day, and Autumnal Equinox Day
    # all fall on Sunday, each with a Monday substitute.
    "2029-01-01", "2029-01-02", "2029-01-03",   # New Year
    "2029-01-08",   # Coming of Age Day
    "2029-02-11", "2029-02-12",   # National Foundation Day + substitute
    "2029-02-23",   # Emperor's Birthday
    "2029-03-20",   # Vernal Equinox Day
    "2029-04-29", "2029-04-30",   # Showa Day + substitute
    "2029-05-03",   # Constitution Memorial Day
    "2029-05-04",   # Greenery Day
    "2029-05-05",   # Children's Day (Saturday)
    "2029-07-16",   # Marine Day
    "2029-08-11",   # Mountain Day (Saturday)
    "2029-09-17",   # Respect for the Aged Day
    "2029-09-23", "2029-09-24",   # Autumnal Equinox Day + substitute
    "2029-10-08",   # Sports Day
    "2029-11-03",   # Culture Day (Saturday)
    "2029-11-23",   # Labor Thanksgiving Day
    "2029-12-31",   # Year-End market closure

    # 2030 — Children's Day, Mountain Day, and Culture Day all fall on
    # Sunday, each with a Monday substitute.
    "2030-01-01", "2030-01-02", "2030-01-03",   # New Year
    "2030-01-14",   # Coming of Age Day
    "2030-02-11",   # National Foundation Day
    "2030-02-23",   # Emperor's Birthday (Saturday)
    "2030-03-20",   # Vernal Equinox Day
    "2030-04-29",   # Showa Day
    "2030-05-03",   # Constitution Memorial Day
    "2030-05-04",   # Greenery Day (Saturday)
    "2030-05-05", "2030-05-06",   # Children's Day + substitute
    "2030-07-15",   # Marine Day
    "2030-08-11", "2030-08-12",   # Mountain Day + substitute
    "2030-09-16",   # Respect for the Aged Day
    "2030-09-23",   # Autumnal Equinox Day
    "2030-10-14",   # Sports Day
    "2030-11-03", "2030-11-04",   # Culture Day + substitute
    "2030-11-23",   # Labor Thanksgiving Day (Saturday)
    "2030-12-31",   # Year-End market closure
])

# 2026-2030 NYSE/NASDAQ full-day market closures (source:
# https://www.nyse.com/markets/hours-calendars — fixed holidays applied
# with NYSE's own-day observed rule: falls on Saturday -> observed the
# preceding Friday, falls on Sunday -> observed the following Monday;
# Good Friday derived from each year's computed Easter Sunday). Dates are
# the US calendar day the market is closed on — checked against
# _in_us_session()'s session_day (the JST-session's US opening day), not
# the raw JST date. Full closures only; NYSE early-close half days (day
# after Thanksgiving, Jul 3 when it isn't the observed holiday itself,
# Christmas Eve) are not modeled — those sessions are still treated as
# open all the way through. Extend this list each December (or sooner)
# once NYSE has published the next year, same maintenance cadence as
# _JP_HOLIDAYS above.
_US_HOLIDAYS = frozenset(date.fromisoformat(d) for d in [
    # 2026
    "2026-01-01",   # New Year's Day
    "2026-01-19",   # Martin Luther King Jr. Day
    "2026-02-16",   # Washington's Birthday (Presidents Day)
    "2026-04-03",   # Good Friday
    "2026-05-25",   # Memorial Day
    "2026-06-19",   # Juneteenth National Independence Day
    "2026-07-03",   # Independence Day (observed; Jul 4 falls on Saturday)
    "2026-09-07",   # Labor Day
    "2026-11-26",   # Thanksgiving Day
    "2026-12-25",   # Christmas Day

    # 2027
    "2027-01-01",   # New Year's Day
    "2027-01-18",   # Martin Luther King Jr. Day
    "2027-02-15",   # Washington's Birthday (Presidents Day)
    "2027-03-26",   # Good Friday
    "2027-05-31",   # Memorial Day
    "2027-06-18",   # Juneteenth (observed; Jun 19 falls on Saturday)
    "2027-07-05",   # Independence Day (observed; Jul 4 falls on Sunday)
    "2027-09-06",   # Labor Day
    "2027-11-25",   # Thanksgiving Day
    "2027-12-24",   # Christmas (observed; Dec 25 falls on Saturday)
    "2027-12-31",   # New Year's Day 2028 (observed; Jan 1, 2028 falls on Saturday)

    # 2028
    "2028-01-17",   # Martin Luther King Jr. Day
    "2028-02-21",   # Washington's Birthday (Presidents Day)
    "2028-04-14",   # Good Friday
    "2028-05-29",   # Memorial Day
    "2028-06-19",   # Juneteenth National Independence Day
    "2028-07-04",   # Independence Day
    "2028-09-04",   # Labor Day
    "2028-11-23",   # Thanksgiving Day
    "2028-12-25",   # Christmas Day

    # 2029
    "2029-01-01",   # New Year's Day
    "2029-01-15",   # Martin Luther King Jr. Day
    "2029-02-19",   # Washington's Birthday (Presidents Day)
    "2029-03-30",   # Good Friday
    "2029-05-28",   # Memorial Day
    "2029-06-19",   # Juneteenth National Independence Day
    "2029-07-04",   # Independence Day
    "2029-09-03",   # Labor Day
    "2029-11-22",   # Thanksgiving Day
    "2029-12-25",   # Christmas Day

    # 2030
    "2030-01-01",   # New Year's Day
    "2030-01-21",   # Martin Luther King Jr. Day
    "2030-02-18",   # Washington's Birthday (Presidents Day)
    "2030-04-19",   # Good Friday
    "2030-05-27",   # Memorial Day
    "2030-06-19",   # Juneteenth National Independence Day
    "2030-07-04",   # Independence Day
    "2030-09-02",   # Labor Day
    "2030-11-28",   # Thanksgiving Day
    "2030-12-25",   # Christmas Day
])


def is_us_dst(d: date) -> bool:
    """Approximate US DST: active roughly April 1 – October 31 in JST."""
    return 4 <= d.month <= 10


def is_open(code: str) -> bool:
    """True if the stock's home market is currently in trading hours."""
    now_jst = datetime.now(_JST)
    market = code.split(".")[0].upper() if "." in code else "US"

    if market == "JP":
        # JP sessions never cross midnight, so a blanket weekend guard is safe.
        if now_jst.weekday() >= 5 or now_jst.date() in _JP_HOLIDAYS:
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


def _is_us_trading_day(d: date) -> bool:
    """True if `d` (a US calendar day) is a weekday and not a US market
    holiday — shared by just_opened()/just_closed() so the "美股开盘/收盘"
    notifications only fire on days a session actually ran, same trading-day
    definition _in_us_session() uses for the real scan gate."""
    return d.weekday() < 5 and d not in _US_HOLIDAYS


def is_trading_day(code: str, d: date) -> bool:
    """True if `d` is a trading day (weekday, not a market holiday) for the
    market `code` belongs to. Holiday-only check, no time-of-day — additive
    helper for callers that need to count trading-day distance (e.g.
    engine/event_risk.py's earnings-window math), does not touch is_open()."""
    market = code.split(".")[0] if "." in code else "US"
    if market == "JP":
        return d.weekday() < 5 and d not in _JP_HOLIDAYS
    return _is_us_trading_day(d)


def market_date_for(code: str, now: datetime = None) -> date:
    """The trading-calendar date `code`'s home market data should be
    attributed to at moment `now` (default: current JST time) — for
    Research snapshot rows, NOT for is_open()'s live go/no-go decision.

    Distinct from is_open()'s question ("can I trade right now") this
    answers "which calendar day does this observation belong to", which
    still has a sensible answer while the market is closed (research
    collection runs once daily at 07:15 JST, squarely in the JST daytime
    gap between one US session's close and the next one's open — see
    module docstring for the session-crossing-midnight shape).

    US: reuses _US_SUMMER/WINTER_OPEN/CLOSE + is_us_dst() +
    _is_us_trading_day() (no new holiday/DST logic). If a session is live
    right now, returns that session's own opening date (mirrors
    _in_us_session()'s branching). Otherwise walks backward from "yesterday
    if we haven't reached tonight's open yet, else today" to the nearest
    actual US trading day — i.e. the most recently completed session.

    JP: sessions never cross midnight, so this is just is_trading_day()
    walked backward to the nearest match."""
    if now is None:
        now = datetime.now(_JST)
    elif now.tzinfo is None:
        now = _JST.localize(now)

    market = code.split(".")[0] if "." in code else "US"
    d = now.date()

    if market == "JP":
        while not is_trading_day(code, d):
            d -= timedelta(days=1)
        return d

    dst = is_us_dst(d)
    open_ = _US_SUMMER_OPEN if dst else _US_WINTER_OPEN
    candidate = d if now.time() >= open_ else d - timedelta(days=1)
    while not _is_us_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def just_closed(minutes_after: int = 5, now: datetime = None) -> bool:
    """True once `now` (defaults to current JST time) is at or past
    `minutes_after` past today's US market close — used to fire a one-time
    post-close notification without needing exact alignment with the
    polling interval. No window end: should_notify_close()'s per-day dedup
    is what makes this fire only once, so a notification isn't silently
    lost if a scan pass runs long and the check happens well after close
    (observed 2026-07-24: a stuck moomoo OpenD connection delayed a pass by
    ~20min, long enough to miss a fixed-width window entirely). The close
    check lands on the JST morning *after* the session opened, so the
    trading-day check is against `now.date() - 1 day` (the US day the
    session opened on), not `now.date()` itself."""
    if now is None:
        now = datetime.now(_JST)
    elif now.tzinfo is None:
        now = _JST.localize(now)

    if not _is_us_trading_day(now.date() - timedelta(days=1)):
        return False

    close_t = _US_SUMMER_CLOSE if is_us_dst(now.date()) else _US_WINTER_CLOSE
    notify_start = _JST.localize(datetime.combine(now.date(), close_t)) + timedelta(minutes=minutes_after)
    return now >= notify_start


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


def just_opened(minutes_after: int = 0, now: datetime = None) -> bool:
    """True once `now` (defaults to current JST time) is at or past
    `minutes_after` past today's US market open — mirrors just_closed() for
    a one-time post-open notification. Default is 0 (fire right at the
    open, not some buffer after it — 2026-07-28 user wants to know the
    instant 9:30 hits, not minutes later) since this only gates a
    notification, not a trading action, so there's no risk in not waiting.
    No window end: should_notify_open()'s per-day dedup is what makes this
    fire only once, so a notification isn't silently lost if a scan pass
    runs long and the check happens well after open (observed 2026-07-24: a
    stuck moomoo OpenD connection delayed a pass by ~20min, long enough to
    miss a fixed-width window entirely). The open check lands on the JST
    evening of the session's own opening day, so the trading-day check is
    against `now.date()` directly."""
    if now is None:
        now = datetime.now(_JST)
    elif now.tzinfo is None:
        now = _JST.localize(now)

    if not _is_us_trading_day(now.date()):
        return False

    open_t = _US_SUMMER_OPEN if is_us_dst(now.date()) else _US_WINTER_OPEN
    notify_start = _JST.localize(datetime.combine(now.date(), open_t)) + timedelta(minutes=minutes_after)
    return now >= notify_start


_last_open_notified: date = None


def should_notify_open() -> bool:
    """Stateful wrapper around just_opened(): returns True at most once
    per calendar day (JST), the first time a caller polls inside the
    post-open notification window — mirrors should_notify_close()."""
    global _last_open_notified
    now = datetime.now(_JST)
    if just_opened(now=now) and _last_open_notified != now.date():
        _last_open_notified = now.date()
        return True
    return False


def _in_us_session(t: dtime, d: date) -> bool:
    """Session spans midnight (open_ > close_), so whether `t` falls inside
    it depends on which JST calendar day the *opening* half belongs to —
    not on today's weekday. `t < close_` is the tail end of a session that
    opened on `d - 1 day`; `t >= open_` is a session opening on `d` itself.
    Holiday check uses the same session_day (the US calendar day the
    session actually opened on), so a US holiday correctly blocks both its
    JST evening leg and its JST-next-morning tail."""
    dst = is_us_dst(d)
    open_  = _US_SUMMER_OPEN  if dst else _US_WINTER_OPEN
    close_ = _US_SUMMER_CLOSE if dst else _US_WINTER_CLOSE

    if t < close_:
        session_day = d - timedelta(days=1)
        return session_day.weekday() < 5 and session_day not in _US_HOLIDAYS
    if t >= open_:
        return d.weekday() < 5 and d not in _US_HOLIDAYS
    return False


def _in_jp_session(t: dtime) -> bool:
    morning   = _JP_MORNING_OPEN   <= t < _JP_MORNING_CLOSE
    afternoon = _JP_AFTERNOON_OPEN <= t < _JP_AFTERNOON_CLOSE
    return morning or afternoon
