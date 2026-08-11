"""
engine/market_context.py — V2.9.x Market Context Logging (market environment
parameter acquisition + storage).

Side-effect-only, same discipline as engine/trade_tracker.py: this module
only ever COLLECTS and RECORDS market-wide sentiment/volatility indicators —
it never feeds into BUY/SELL, Confidence Score, or position sizing. Every
call site wiring this into engine/runner.py wraps it in try/except so a
fetch failure (network outage, source changed its page layout, etc.) can
never break a trading pass. See project spec "V2.9.x — Market Context
Logging" for the full rationale: this data is accumulated for 1-3 months
(or enough trade samples) before any statistical analysis decides whether
it's worth feeding into a future V3.0 Dynamic Position Sizing / V3.1 Risk
Engine.

Five candidate indicators, three implemented (verified against a live
fetch 2026-08-08 — see fetch_* docstrings for the exact endpoint), two
deliberately left as stubs:
  vix_close        — CBOE VIX_History.csv (daily)
  cnn_fear_greed   — CNN's internal fear & greed JSON feed (daily)
  naaim_exposure   — NAAIM's embeddable chart JSON (weekly)
  aaii_bullish/bearish/neutral/spread — STUB, always None. aaii.com blocks
      scripted requests with a bot-wall (403 on every header combination
      tried) — no free reliable endpoint found. Revisit fetch_aaii() if a
      source turns up; until then every row's aaii_* columns stay NULL
      with data_status["aaii"]="NOT_IMPLEMENTED".
  put_call_ratio   — STUB, always None. CBOE's daily options statistics
      page (cboe.com/us/options/market_statistics/daily/) is a client-side
      rendered SPA with no discoverable public data endpoint (tried the
      obvious CBOE CDN CSV filename patterns — all 403). Same treatment as
      AAII: NULL + data_status["put_call"]="NOT_IMPLEMENTED".

Look-ahead-bias note (see spec section 八): every fetch function returns
its OWN source date alongside the value (vix_date/cnn_fear_greed_date/
naaim_date), separate from `observation_date` (the ET calendar day the
daily update job ran on) — so a backtest/analysis joining trades to market
context can always tell "when was this number actually published" instead
of assuming same-day availability.

Data integrity (2026-08-08, see DataIntegrityError/DataStaleError below):
every value goes through a pure parser (never trusts file/array position —
always parses every candidate row and picks max(date)), a numeric sanity
range check, and a per-source freshness check before a fetch_* function
will return it. Added after an LLM web-fetch tool reading CBOE's
VIX_History.csv silently reported a 1997-10-03 row as "the latest" instead
of the true 2026-08-07 row — a wrong-but-plausible answer, not a visible
failure. fetch_* still never raises (same contract as before: any failure,
including these new integrity/staleness checks, collapses to returning
None) — see the "Data integrity / freshness validation" section for
details.
"""
import functools
import html
import json
import math
import re
import sqlite3
import threading
from datetime import datetime, date, time as dtime
from pathlib import Path
from typing import Optional

import pytz
import requests

from engine.market_hours import _US_HOLIDAYS   # reuse the maintained NYSE
                                                 # holiday calendar instead of
                                                 # keeping a second copy that
                                                 # would drift out of sync
from engine.trade_tracker import _DEFAULT_DB_PATH   # same trade_history.db —
                                                      # market_context is a
                                                      # new table in the same
                                                      # file, not a new DB

_ET = pytz.timezone("America/New_York")
_HTTP_TIMEOUT = 15
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

_DAILY_UPDATE_TIME = dtime(8, 30)   # ET — "before US market open" per spec

_write_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_context (
    observation_date     TEXT PRIMARY KEY,
    vix_close             REAL,
    vix_date              TEXT,
    vix_source             TEXT,
    cnn_fear_greed         REAL,
    cnn_fear_greed_label   TEXT,
    cnn_fear_greed_date    TEXT,
    aaii_bullish           REAL,
    aaii_bearish           REAL,
    aaii_neutral           REAL,
    aaii_spread            REAL,
    aaii_date              TEXT,
    put_call_ratio         REAL,
    put_call_date          TEXT,
    put_call_source        TEXT,
    naaim_exposure         REAL,
    naaim_date             TEXT,
    data_status            TEXT,
    created_at             TEXT,
    updated_at             TEXT
);
"""


# ── Data integrity / freshness validation ─────────────────────────────────────
# Added after a real incident (2026-08-08): asking an LLM web-fetch tool to
# read CBOE's VIX_History.csv (a ~36-year, multi-thousand-row file) had it
# confidently report an 1997-10-03 row as "the latest" instead of the true
# 2026-08-07 row — a wrong-but-plausible-looking answer, not a visible
# failure. Our OWN parsers never trusted file position either (fetch_vix()
# used to just take lines[-1]) — that happened to be correct for this one
# file's current layout, but "assume the last line is newest" is exactly
# the same class of bug, just not yet triggered. Every parser below now
# explicitly parses every row, discards anything malformed, and picks
# max(date) — plus a freshness check calibrated per source, so a parser
# bug that ever again picks a stale row gets caught here instead of
# silently becoming "today's VIX".

class DataIntegrityError(Exception):
    """Raised by the pure parsing/validation helpers below on a structural
    problem: no valid rows at all, an unparseable date/value, a non-finite
    (NaN/inf) value, a value outside that indicator's own sane range, or
    conflicting values reported for what should be a single latest date.
    Always caught by the fetch_* function that calls these helpers (same
    try/except Exception that already handles network/HTTP failures) —
    never propagates to update_market_context() or callers. Kept as a
    distinct raise (instead of just falling into that catch-all) purely so
    these specific failure modes are independently unit-testable without
    needing a live/mocked HTTP call."""


class DataStaleError(DataIntegrityError):
    """A value parsed successfully and passed its own structural checks,
    but its reported date is older than that source's expected update
    cadence allows — the exact shape of the 1997-vs-2026 VIX incident
    above, now caught in OUR parser rather than relying on an LLM
    summarizer to notice. Each source's max-lag threshold (see the
    *_MAX_LAG_DAYS constants next to each fetch function) is deliberately
    generous enough to tolerate that source's own normal cadence —
    weekends, holidays, and NAAIM's known post-2026-08-01
    subscription-paywall lag (see fetch_naaim()) — so a real,
    correctly-identified-but-legitimately-old reading is never confused
    with this failure mode."""


def _check_freshness(source_name: str, data_date: date, now: date,
                      max_lag_days: int) -> None:
    age_days = (now - data_date).days
    if age_days < 0:
        raise DataStaleError(
            f"{source_name}: data_date {data_date} is in the future "
            f"relative to {now} — likely a parsing/timezone bug")
    if age_days > max_lag_days:
        raise DataStaleError(
            f"{source_name}: data_date {data_date} is {age_days} days old "
            f"(max allowed {max_lag_days}) — refusing to treat this as "
            f"fresh data, most likely a parser picked the wrong row")


def _validate_numeric(source_name: str, value, lo: float, hi: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) \
            or not math.isfinite(value):
        raise DataIntegrityError(f"{source_name}: value {value!r} is not a "
                                  f"finite number")
    if not (lo <= value <= hi):
        raise DataIntegrityError(f"{source_name}: value {value} outside "
                                  f"sane range [{lo}, {hi}]")


def _parse_vix_csv(text: str) -> dict:
    """Pure parser for CBOE's VIX_History.csv (DATE,OPEN,HIGH,LOW,CLOSE).
    Explicitly parses every row (skipping the header/blank/malformed
    lines) and picks max(date) — never assumes the file is sorted or that
    the last line is newest. Raises DataIntegrityError if no valid rows
    parse at all, or if the max date has more than one distinct CLOSE
    value (a genuinely conflicting/corrupt file, not just a duplicate
    row)."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.upper().startswith("DATE"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            d = datetime.strptime(parts[0].strip(), "%m/%d/%Y").date()
            close = float(parts[4])
        except (ValueError, IndexError):
            continue
        if not math.isfinite(close):
            continue
        rows.append((d, close))

    if not rows:
        raise DataIntegrityError("VIX CSV: no valid rows parsed")

    max_date = max(d for d, _ in rows)
    candidates = {c for d, c in rows if d == max_date}
    if len(candidates) > 1:
        raise DataIntegrityError(
            f"VIX CSV: conflicting CLOSE values for {max_date}: {candidates}")

    return {"vix_close": candidates.pop(), "vix_date": max_date}


def _parse_cnn_payload(payload: dict) -> dict:
    """Pure parser/validator for CNN's fear_and_greed JSON payload."""
    try:
        fg = payload["fear_and_greed"]
        score = float(fg["score"])
        rating = str(fg["rating"])
        timestamp = str(fg["timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataIntegrityError(
            f"CNN fear&greed: unexpected JSON shape — {exc}") from exc

    try:
        d = datetime.strptime(timestamp[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise DataIntegrityError(
            f"CNN fear&greed: unparseable timestamp {timestamp!r}") from exc

    return {"cnn_fear_greed": round(score, 2), "cnn_fear_greed_label": rating,
            "cnn_fear_greed_date": d}


def _parse_naaim_chart_json(chart: dict) -> dict:
    """Pure parser for NAAIM's embeddable chart JSON payload. Explicitly
    parses every (label, value) pair in the "NAAIM Number" dataset and
    picks the one with max(date) — never trusts the arrays' existing
    order/[-1] the way the original implementation did."""
    try:
        labels = chart["data"]["labels"]
        datasets = chart["data"]["datasets"]
    except (KeyError, TypeError) as exc:
        raise DataIntegrityError(
            f"NAAIM chart: unexpected JSON shape — {exc}") from exc

    naaim_ds = next((d for d in datasets if d.get("label") == "NAAIM Number"),
                     None)
    if naaim_ds is None:
        raise DataIntegrityError("NAAIM chart: no 'NAAIM Number' dataset found")
    values = naaim_ds.get("data") or []
    if len(labels) != len(values):
        raise DataIntegrityError(
            f"NAAIM chart: labels/data length mismatch "
            f"({len(labels)} vs {len(values)})")

    pairs = []
    for label, raw_value in zip(labels, values):
        try:
            d = datetime.strptime(str(label), "%Y-%m-%d").date()
            v = float(raw_value)
        except (ValueError, TypeError):
            continue
        if not math.isfinite(v):
            continue
        pairs.append((d, v))

    if not pairs:
        raise DataIntegrityError("NAAIM chart: no valid (date, value) pairs parsed")

    max_date = max(d for d, _ in pairs)
    candidates = {v for d, v in pairs if d == max_date}
    if len(candidates) > 1:
        raise DataIntegrityError(
            f"NAAIM chart: conflicting values for {max_date}: {candidates}")

    return {"naaim_exposure": candidates.pop(), "naaim_date": max_date}


# ── Per-source fetchers ───────────────────────────────────────────────────────
# Each returns a small dict of its own fields on success, or None on ANY
# failure (network, parse, integrity, staleness — never raises). update_
# market_context() is what decides what happens when a fetch returns None
# (carry forward the last stored value, see below). `now` (a plain date,
# defaulting to today's ET date) is the freshness-check reference point —
# accepting it as a parameter (rather than always computing it internally)
# is what makes the staleness checks deterministically testable.

_VIX_MAX_LAG_DAYS = 7   # daily source; comfortably covers any realistic
                        # holiday-cluster gap (e.g. Thu Thanksgiving + long
                        # weekend) plus CBOE's own ~1-day publish lag,
                        # while still being orders of magnitude tighter
                        # than a genuine parser bug (the 1997 incident was
                        # ~10,500 days stale)
_VIX_MIN, _VIX_MAX = 0.0, 200.0   # VIX has never printed <=0 or briefly
                                   # traded above ~90 intraday historically;
                                   # 200 is a generous ceiling that still
                                   # catches a "parsed the wrong column" bug


def fetch_vix(now: Optional[date] = None) -> Optional[dict]:
    """CBOE's own public VIX history CSV — confirmed working 2026-08-08,
    no API key, updated daily (one row per NYSE trading day, ~1 day lag for
    the most recent close). Columns: DATE(MM/DD/YYYY),OPEN,HIGH,LOW,CLOSE."""
    now = now or datetime.now(_ET).date()
    try:
        resp = requests.get(
            "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
            headers={"User-Agent": _UA}, timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        parsed = _parse_vix_csv(resp.text)
        _validate_numeric("VIX", parsed["vix_close"], _VIX_MIN, _VIX_MAX)
        _check_freshness("VIX", parsed["vix_date"], now, _VIX_MAX_LAG_DAYS)
        return {"vix_close": parsed["vix_close"],
                "vix_date": parsed["vix_date"].isoformat(),
                "vix_source": "CBOE"}
    except Exception:
        return None


_CNN_MAX_LAG_DAYS = 7   # daily source, same reasoning as VIX above
_CNN_MIN, _CNN_MAX = 0.0, 100.0   # CNN defines the score as bounded [0,100]


def fetch_cnn_fear_greed(now: Optional[date] = None) -> Optional[dict]:
    """CNN's internal (undocumented) fear & greed JSON feed — confirmed
    working 2026-08-08. Requires a Referer header matching the public page
    or CNN's edge returns HTTP 418; no API key needed otherwise."""
    now = now or datetime.now(_ET).date()
    try:
        resp = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={"User-Agent": _UA,
                     "Referer": "https://www.cnn.com/markets/fear-and-greed"},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        parsed = _parse_cnn_payload(resp.json())
        _validate_numeric("CNN Fear & Greed", parsed["cnn_fear_greed"],
                           _CNN_MIN, _CNN_MAX)
        _check_freshness("CNN Fear & Greed", parsed["cnn_fear_greed_date"],
                          now, _CNN_MAX_LAG_DAYS)
        return {"cnn_fear_greed": parsed["cnn_fear_greed"],
                "cnn_fear_greed_label": parsed["cnn_fear_greed_label"],
                "cnn_fear_greed_date": parsed["cnn_fear_greed_date"].isoformat()}
    except Exception:
        return None


_NAAIM_ATTR_RE = re.compile(
    r'data-symfony--ux-chartjs--chart-view-value="([^"]*)"')

_NAAIM_MAX_LAG_DAYS = 120   # weekly source, but NAAIM's free embeddable
                            # widget has been stuck ~100 days stale since
                            # NAAIM moved to a subscription model on
                            # 2026-08-01 (see project notes) — this is
                            # documented, legitimate staleness of the free
                            # feed itself, not a parsing bug, so the
                            # threshold must tolerate it. Still tight
                            # enough (120 days, not "no limit") to catch a
                            # genuine parser regression.
_NAAIM_MIN, _NAAIM_MAX = -300.0, 300.0   # NAAIM allows leveraged
                                          # short/long exposure by design;
                                          # historically observed roughly
                                          # -50..+150, 300 is a generous
                                          # ceiling


def fetch_naaim(now: Optional[date] = None) -> Optional[dict]:
    """NAAIM Exposure Index — confirmed working 2026-08-08. NAAIM's own
    "NAAIM Number" iframe (index.naaim.org/embeddable/number) is rendered
    client-side with no server-embedded data, but the sibling chart iframe
    (index.naaim.org/embeddable/chart) server-renders its Chart.js dataset
    into a `data-symfony--ux-chartjs--chart-view-value` HTML attribute
    (JSON, HTML-entity-escaped) — that's the only place the actual number
    is available without executing JS. Weekly data: naturally "carries
    forward" on its own between weekly publishes since the source simply
    hasn't appended a new point yet — see _NAAIM_MAX_LAG_DAYS above for why
    that's tolerated rather than rejected as stale."""
    now = now or datetime.now(_ET).date()
    try:
        resp = requests.get(
            "https://index.naaim.org/embeddable/chart",
            headers={"User-Agent": _UA}, timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        m = _NAAIM_ATTR_RE.search(resp.text)
        if not m:
            return None
        chart = json.loads(html.unescape(m.group(1)))
        parsed = _parse_naaim_chart_json(chart)
        _validate_numeric("NAAIM", parsed["naaim_exposure"],
                           _NAAIM_MIN, _NAAIM_MAX)
        _check_freshness("NAAIM", parsed["naaim_date"], now, _NAAIM_MAX_LAG_DAYS)
        return {"naaim_exposure": parsed["naaim_exposure"],
                "naaim_date": parsed["naaim_date"].isoformat()}
    except Exception:
        return None


def fetch_aaii() -> Optional[dict]:
    """STUB — aaii.com returns HTTP 403 (bot-wall) for every request-header
    combination tried; no free reliable endpoint found as of 2026-08-08.
    Always returns None — see module docstring."""
    return None


def fetch_put_call_ratio() -> Optional[dict]:
    """STUB — CBOE's daily options statistics page is a client-side
    rendered SPA with no discoverable public data endpoint as of
    2026-08-08. Always returns None — see module docstring."""
    return None


# ── Storage ───────────────────────────────────────────────────────────────────

class MarketContextStore:
    """SQLite-backed recorder for the market_context table — one row per
    ET calendar day (observation_date), in the same DB file
    engine/trade_tracker.py uses (separate table, per spec section 六)."""

    def __init__(self, db_path: Optional[Path] = None):
        self.path = Path(db_path or _DEFAULT_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with _write_lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get_latest_context(self) -> Optional[dict]:
        """Most recent row (by observation_date), for snapshotting into a
        trade at ENTRY time (engine/trade_tracker.py's log_market_context).
        None if no update has ever run yet."""
        row = self._conn.execute(
            "SELECT * FROM market_context ORDER BY observation_date DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row is not None else None

    def get_context_for_date(self, observation_date: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM market_context WHERE observation_date=?",
            (observation_date,),
        ).fetchone()
        return dict(row) if row is not None else None

    def save(self, record: dict) -> None:
        """Upsert one observation_date row — safe to call more than once
        for the same day (e.g. a re-run after a partial failure), later
        calls simply overwrite."""
        now_iso = datetime.now().isoformat()
        with _write_lock:
            existing = self._conn.execute(
                "SELECT created_at FROM market_context WHERE observation_date=?",
                (record["observation_date"],),
            ).fetchone()
            created_at = existing["created_at"] if existing else now_iso
            self._conn.execute(
                """INSERT OR REPLACE INTO market_context (
                    observation_date, vix_close, vix_date, vix_source,
                    cnn_fear_greed, cnn_fear_greed_label, cnn_fear_greed_date,
                    aaii_bullish, aaii_bearish, aaii_neutral, aaii_spread, aaii_date,
                    put_call_ratio, put_call_date, put_call_source,
                    naaim_exposure, naaim_date, data_status, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (record["observation_date"], record.get("vix_close"),
                 record.get("vix_date"), record.get("vix_source"),
                 record.get("cnn_fear_greed"), record.get("cnn_fear_greed_label"),
                 record.get("cnn_fear_greed_date"),
                 record.get("aaii_bullish"), record.get("aaii_bearish"),
                 record.get("aaii_neutral"), record.get("aaii_spread"),
                 record.get("aaii_date"),
                 record.get("put_call_ratio"), record.get("put_call_date"),
                 record.get("put_call_source"),
                 record.get("naaim_exposure"), record.get("naaim_date"),
                 json.dumps(record.get("data_status", {}), ensure_ascii=False),
                 created_at, now_iso),
            )
            self._conn.commit()


# ── Orchestration ─────────────────────────────────────────────────────────────

# name -> (fetch function name, fields) per indicator — used by _resolve()
# below so update_market_context() doesn't repeat the same
# fetch-or-carry-forward-or-null branch five times. Looked up by NAME (via
# globals()) rather than holding the function object directly, so
# mock.patch("engine.market_context.fetch_vix", ...) in tests actually
# takes effect — a dict built at import time with bound function references
# would keep calling the original, unpatched function forever.
_SOURCES = {
    "vix":      ("fetch_vix",            ["vix_close", "vix_date", "vix_source"]),
    "cnn":      ("fetch_cnn_fear_greed",  ["cnn_fear_greed", "cnn_fear_greed_label",
                                            "cnn_fear_greed_date"]),
    "naaim":    ("fetch_naaim",           ["naaim_exposure", "naaim_date"]),
}
# aaii/put_call are NOT in _SOURCES — they're not fetched at all (see stubs
# above), so update_market_context() marks them NOT_IMPLEMENTED directly
# rather than running them through the carry-forward path.
_NOT_IMPLEMENTED_FIELDS = {
    "aaii": ["aaii_bullish", "aaii_bearish", "aaii_neutral", "aaii_spread", "aaii_date"],
    "put_call": ["put_call_ratio", "put_call_date", "put_call_source"],
}


def _resolve(name: str, fetch_fn, fields: list, previous: Optional[dict],
             record: dict, status: dict) -> None:
    """Fetch one indicator; on failure, carry forward the value(s) from
    `previous` (the last stored market_context row) if one exists — per
    spec section 二/七: weekly data legitimately hasn't changed between
    publishes, and a transient fetch failure on daily data is better
    represented as "last known good" than a hard NULL gap. Never fabricates
    a value that was never actually observed (status makes clear which
    happened)."""
    fresh = fetch_fn()
    if fresh is not None:
        record.update(fresh)
        status[name] = "OK"
        return
    if previous is not None and previous.get(fields[0]) is not None:
        for f in fields:
            record[f] = previous.get(f)
        status[name] = "CARRIED_FORWARD"
        return
    for f in fields:
        record[f] = None
    status[name] = "UNAVAILABLE"


def update_market_context(now: Optional[datetime] = None) -> dict:
    """Fetch all implemented indicators, resolve carry-forward/NULL for any
    that failed, persist one market_context row for today's ET calendar
    date, and return the saved record (including data_status). This is the
    single daily entry point — engine/runner.py calls it once per ET
    calendar day (see should_run_daily_update()), never on a fixed
    5-minute cadence, per spec section 三/四."""
    now = now or datetime.now(_ET)
    observation_day = now.date()
    observation_date = observation_day.isoformat()

    store = MarketContextStore()
    try:
        previous = store.get_latest_context()
        record = {"observation_date": observation_date}
        status = {}

        for name, (fetch_fn_name, fields) in _SOURCES.items():
            fetch_fn = functools.partial(globals()[fetch_fn_name], now=observation_day)
            _resolve(name, fetch_fn, fields, previous, record, status)
        for name, fields in _NOT_IMPLEMENTED_FIELDS.items():
            for f in fields:
                record[f] = None
            status[name] = "NOT_IMPLEMENTED"

        record["data_status"] = status
        store.save(record)
        return record
    finally:
        store.close()


def get_latest_context() -> Optional[dict]:
    """Convenience wrapper for callers (engine/runner.py at BUY/ENTRY) that
    just want today's-or-most-recent snapshot without managing a
    MarketContextStore instance themselves."""
    store = MarketContextStore()
    try:
        return store.get_latest_context()
    finally:
        store.close()


# ── Daily ET scheduling gate ──────────────────────────────────────────────────
# Mirrors engine/market_hours.py's should_notify_open()/should_notify_close()
# pattern: fires True at most once per ET calendar day, the first time a
# caller polls at/after _DAILY_UPDATE_TIME.

_last_update_date: Optional[date] = None


def should_run_daily_update(now: Optional[datetime] = None) -> bool:
    global _last_update_date
    if now is None:
        now = datetime.now(_ET)
    elif now.tzinfo is None:
        now = _ET.localize(now)

    if now.weekday() >= 5 or now.date() in _US_HOLIDAYS:
        return False
    if now.time() < _DAILY_UPDATE_TIME:
        return False
    if _last_update_date == now.date():
        return False

    _last_update_date = now.date()
    return True
