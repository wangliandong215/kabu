"""
engine/order_audit.py — V3.6 P5 Execution Audit (observation only).

One row per engine/runner.py::_place_order() call in order_audit, plus an
append-only order_audit_events trail of every lifecycle stage:

    DECISION -> RISK_EVALUATED -> PREFLIGHT -> SUBMIT_SENT -> ACK | SUBMIT_REJECTED
      -> POLL (xN) / POLL_FAILED -> CANCEL_SENT / CANCEL_FAILED -> EXCEPTION -> FINISH

_place_order() calls begin()/stage()/finish(); engine/broker.py's
MoomooBroker calls event() from inside place_order(). The "current order"
travels in a ContextVar, so neither the broker's signature nor any return
value changes — a broker (or a test fake) that never calls event() simply
produces a row with fewer stages.

finish() classifies the outcome (see _classify). The two outcomes that
matter most for safety:
  UNCONFIRMED_AFTER_ACK   the broker accepted the order (order_id known) but
                          no terminal status was ever confirmed — polling
                          failed or raised. The caller still gets whatever
                          dealt_qty the last successful poll saw (often 0),
                          so local state may disagree with the broker.
  SUBMIT_OUTCOME_UNKNOWN  place_order itself raised after the request was
                          sent — the order may or may not exist at the broker.
Both are marked fill_confirmed=0 / needs_reconcile=1, reconcile_status=
PENDING and an unknown_reason (why the state is unknown). V3.6 P7 acts on
them: engine/execution_guard.py blocks further BUYs of that code until the
row is reconciled against the broker (read_unresolved()/record_reconcile()
below are its only DB access — the schema stays owned by this module).

Contract: NEVER raises and never changes what _place_order() returns. Every
public function swallows its own errors; after a write failure the audit
just goes quiet for that order. Writes go to a separate SQLite DB
(_DB_PATH), never trade_history.db, so audit I/O can't contend with or
corrupt the trade record.

commission: not available from order_list_query — recorded as NULL with
commission_status='NOT_QUERIED' (a post-hoc broker fee query is a
follow-up, not done in the trading path).
"""
import contextvars
import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

_DB_PATH = Path(r"C:\KabuData\execution\order_audit.db")
_lock = threading.Lock()
_initialized: set = set()   # DB paths whose schema has been created this process

_current_order: contextvars.ContextVar = contextvars.ContextVar("order_audit_current", default=None)
_current_pass: contextvars.ContextVar = contextvars.ContextVar("order_audit_pass", default=None)

# Outcomes
DRY_RUN = "DRY_RUN"
RISK_BLOCKED = "RISK_BLOCKED"
PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
NOT_SUBMITTED_ERROR = "NOT_SUBMITTED_ERROR"
SUBMIT_REJECTED = "SUBMIT_REJECTED"
SUBMIT_OUTCOME_UNKNOWN = "SUBMIT_OUTCOME_UNKNOWN"
FILLED = "FILLED"
PARTIAL_FILLED = "PARTIAL_FILLED"
NOT_FILLED = "NOT_FILLED"
UNCONFIRMED_AFTER_ACK = "UNCONFIRMED_AFTER_ACK"
UNCONFIRMED = (SUBMIT_OUTCOME_UNKNOWN, UNCONFIRMED_AFTER_ACK)
EXECUTION_GUARD_BLOCKED = "EXECUTION_GUARD_BLOCKED"   # P7: BUY refused by engine/execution_guard.py

# Same set as engine/broker.py MoomooBroker._TERMINAL (not imported: broker
# imports this module). A test asserts the two stay identical.
# CANCELLED_PART (P7.3): part filled, remainder cancelled — dealt_qty is final.
TERMINAL_STATUSES = frozenset({"FILLED_ALL", "CANCELLED_ALL", "CANCELLED_PART", "FAILED",
                               "DISABLED", "DELETED", "SUBMIT_FAILED", "FILL_CANCELLED"})

# P7 reconcile_status values. Only RELEASED_STATUSES stop a row from
# blocking new BUYs (see engine/execution_guard.py).
RECONCILE_PENDING = "PENDING"                     # unknown, not yet reconciled
RECONCILE_UNRESOLVED = "UNRESOLVED"               # reconciled, broker state still not provable
RECONCILE_RESOLVED = "RESOLVED"                   # broker final state confirmed, local book agrees
RECONCILE_NEEDS_MANUAL = "RESOLVED_NEEDS_MANUAL"  # final state confirmed, local book disagrees
RECONCILE_MANUAL_CLEARED = "MANUAL_CLEARED"       # an operator cleared it (execution_guard CLI)
RELEASED_STATUSES = (RECONCILE_RESOLVED, RECONCILE_MANUAL_CLEARED)

# Process-local mirror of every unconfirmed order finish() saw, so the P7
# guard still knows about them if the audit DB itself is unreadable.
# audit_id -> {"code", "side", "env_label", "order_id", "outcome"}
_unresolved_mem: dict = {}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS order_audit (
    audit_id              TEXT PRIMARY KEY,
    pass_id               TEXT,
    run_id                TEXT,
    pass_started_at       TEXT,
    decision_at           TEXT,
    code                  TEXT,
    side                  TEXT,
    intent                TEXT,
    requested_qty         REAL,
    expected_price        REAL,
    confirmed             INTEGER,
    env_label             TEXT,
    trd_env               TEXT,
    broker                TEXT,
    risk_status           TEXT,
    risk_reason           TEXT,
    risk_mode             TEXT,
    preflight_status      TEXT,
    preflight_reason      TEXT,
    submit_sent_at        TEXT,
    submitted_price       REAL,
    ack_at                TEXT,
    order_id              TEXT,
    submit_error          TEXT,
    broker_create_time    TEXT,
    broker_updated_time   TEXT,
    first_fill_seen_at    TEXT,
    last_poll_at          TEXT,
    poll_count            INTEGER DEFAULT 0,
    poll_failures         INTEGER DEFAULT 0,
    broker_status         TEXT,
    status_confirmed      INTEGER DEFAULT 0,
    dealt_qty             REAL,
    fill_price            REAL,
    cancel_sent_at        TEXT,
    cancel_error          TEXT,
    exception             TEXT,
    finished_at           TEXT,
    returned_order_id     TEXT,
    returned_status       TEXT,
    returned_dealt_qty    REAL,
    returned_fill_price   REAL,
    outcome               TEXT,
    fill_confirmed        INTEGER,
    needs_reconcile       INTEGER,
    partial_fill          INTEGER,
    slippage_pct          REAL,
    slippage_usd          REAL,
    commission            REAL,
    commission_status     TEXT
    -- P7 columns are added by _P7_COLUMNS below (ALTER TABLE ADD COLUMN),
    -- so an existing DB migrates in place.
);
CREATE INDEX IF NOT EXISTS order_audit_code_time ON order_audit (code, decision_at);
CREATE INDEX IF NOT EXISTS order_audit_outcome ON order_audit (outcome);

CREATE TABLE IF NOT EXISTS order_audit_events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_id     TEXT,
    at           TEXT,
    stage        TEXT,
    detail_json  TEXT
);
CREATE INDEX IF NOT EXISTS order_audit_events_audit ON order_audit_events (audit_id);
"""


# V3.6 P7 — additive columns.
_P7_COLUMNS = (
    ("unknown_reason", "TEXT"),           # why the outcome is unconfirmed
    ("guard_status", "TEXT"),             # PASS / BLOCK (BUYs only)
    ("guard_reason", "TEXT"),
    ("reconcile_status", "TEXT"),         # see RECONCILE_* above
    ("reconcile_attempts", "INTEGER DEFAULT 0"),
    ("reconciled_at", "TEXT"),
    ("reconciled_order_id", "TEXT"),      # order_id found at the broker (the only one we
                                          # have when the submit response was lost)
    ("reconciled_broker_status", "TEXT"),
    ("reconciled_outcome", "TEXT"),       # FILLED / PARTIAL_FILLED / NOT_FILLED / NOT_FOUND
    ("reconciled_dealt_qty", "REAL"),
    ("reconciled_fill_price", "REAL"),
    ("needs_manual", "INTEGER DEFAULT 0"),
    ("manual_reason", "TEXT"),
)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _connect() -> sqlite3.Connection:
    """Short-lived connection per write (opened and closed around each
    statement, so no handle is ever left open on the file). _DB_PATH is read
    at call time, so tests can repoint it."""
    path = Path(_DB_PATH)
    if path not in _initialized:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=2.0)
    if path not in _initialized:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        have = {r[1] for r in conn.execute("PRAGMA table_info(order_audit)")}
        for col, decl in _P7_COLUMNS:
            if col not in have:
                conn.execute(f"ALTER TABLE order_audit ADD COLUMN {col} {decl}")
        conn.commit()
        _initialized.add(path)
    return conn


def _warn_once(where: str, exc: Exception) -> None:
    try:
        import notify.alert as alert
        alert.log(f"order_audit: {where} failed ({exc.__class__.__name__}: {exc}) — "
                  f"audit skipped, order flow unaffected")
    except Exception:
        pass


def _write(sql: str, params: tuple) -> None:
    with _lock:
        conn = _connect()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()


def _update(audit_id: str, fields: dict) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    _write(f"UPDATE order_audit SET {sets} WHERE audit_id=?", (*fields.values(), audit_id))


def _append_event(audit_id: str, stage: str, detail: dict) -> None:
    _write("INSERT INTO order_audit_events (audit_id, at, stage, detail_json) VALUES (?,?,?,?)",
           (audit_id, _now(), stage, json.dumps(detail, ensure_ascii=False, default=str)))


def _num(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ── public API ────────────────────────────────────────────────────────────────

def start_pass(run_id: Optional[str] = None, env_label: Optional[str] = None) -> None:
    """Called once at the top of engine/runner.py::run_once()."""
    try:
        _current_pass.set({"pass_id": uuid.uuid4().hex[:12], "run_id": run_id,
                           "started_at": _now(), "env_label": env_label})
    except Exception:
        pass


def begin(code: str, side: str, qty, price, trd_env, env_label: str, confirmed: bool,
          intent: Optional[str] = None) -> Optional[str]:
    """DECISION stage. Returns the audit_id (None if the audit is unavailable)."""
    try:
        p = _current_pass.get() or {}
        audit_id = uuid.uuid4().hex
        state = {"audit_id": audit_id, "requested_qty": _num(qty), "expected_price": _num(price),
                 "side": side, "code": code, "env_label": env_label, "order_id": None, "ack": False, "sent": False, "status_confirmed": False,
                 "broker_status": None, "dealt_qty": None, "fill_price": None,
                 "exception": None, "submit_error": None, "poll_count": 0, "poll_failures": 0}
        _current_order.set(state)
        _write(
            """INSERT INTO order_audit (audit_id, pass_id, run_id, pass_started_at, decision_at,
                   code, side, intent, requested_qty, expected_price, confirmed, env_label,
                   trd_env, commission_status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (audit_id, p.get("pass_id"), p.get("run_id"), p.get("started_at"), _now(),
             code, side, intent or "UNSPECIFIED", _num(qty), _num(price), int(bool(confirmed)),
             env_label, None if trd_env is None else str(trd_env), "NOT_QUERIED"))
        _append_event(audit_id, "DECISION", {"code": code, "side": side, "qty": qty,
                                             "price": price, "intent": intent})
        return audit_id
    except Exception as exc:
        _current_order.set(None)
        _warn_once("begin", exc)
        return None


def stage(name: str, **fields) -> None:
    """_place_order()-side stages that map onto columns (RISK_EVALUATED,
    PREFLIGHT, BROKER_SELECTED)."""
    state = _current_order.get()
    if not state:
        return
    try:
        cols = {k: v for k, v in fields.items() if k in (
            "risk_status", "risk_reason", "risk_mode", "preflight_status",
            "preflight_reason", "broker", "guard_status", "guard_reason")}
        _update(state["audit_id"], cols)
        _append_event(state["audit_id"], name, fields)
    except Exception as exc:
        _warn_once(f"stage {name}", exc)


def event(name: str, **detail) -> None:
    """Broker-side lifecycle events (engine/broker.py). No-op without a
    current order (e.g. trade.py / scripts calling a broker directly)."""
    state = _current_order.get()
    if not state:
        return
    try:
        now = _now()
        cols: dict = {}
        if name == "SUBMIT_SENT":
            state["sent"] = True
            cols = {"submit_sent_at": now, "submitted_price": _num(detail.get("price"))}
        elif name == "ACK":
            state["ack"] = True
            state["order_id"] = detail.get("order_id")
            cols = {"ack_at": now, "order_id": detail.get("order_id")}
        elif name == "SUBMIT_REJECTED":
            state["submit_error"] = str(detail.get("error"))
            cols = {"submit_error": state["submit_error"]}
        elif name == "POLL":
            state["poll_count"] += 1
            state["broker_status"] = detail.get("status")
            state["dealt_qty"] = _num(detail.get("dealt_qty"))
            state["fill_price"] = _num(detail.get("dealt_avg_price"))
            if detail.get("terminal"):
                state["status_confirmed"] = True
            cols = {"last_poll_at": now, "poll_count": state["poll_count"],
                    "broker_status": state["broker_status"], "dealt_qty": state["dealt_qty"],
                    "fill_price": state["fill_price"],
                    "status_confirmed": int(state["status_confirmed"])}
            if detail.get("create_time"):
                cols["broker_create_time"] = str(detail["create_time"])
            if detail.get("updated_time"):
                cols["broker_updated_time"] = str(detail["updated_time"])
            if (state["dealt_qty"] or 0) > 0 and not state.get("first_fill_seen"):
                state["first_fill_seen"] = True
                cols["first_fill_seen_at"] = now
        elif name == "POLL_FAILED":
            state["poll_failures"] += 1
            cols = {"poll_failures": state["poll_failures"]}
        elif name == "CANCEL_SENT":
            cols = {"cancel_sent_at": now}
        elif name == "CANCEL_FAILED":
            state["cancel_failed"] = True
            cols = {"cancel_error": str(detail.get("error"))}
        elif name == "EXCEPTION":
            state["exception"] = str(detail.get("error"))
            cols = {"exception": state["exception"]}
        _update(state["audit_id"], cols)
        _append_event(state["audit_id"], name, detail)
    except Exception as exc:
        _warn_once(f"event {name}", exc)


def _classify(state: dict, confirmed: bool, result: Optional[dict],
              exc: Optional[BaseException]) -> str:
    returned = (result or {}).get("status")
    if returned == "RISK_BLOCKED":
        return RISK_BLOCKED
    if returned == "PREFLIGHT_FAILED":
        return PREFLIGHT_FAILED
    if returned == EXECUTION_GUARD_BLOCKED:
        return EXECUTION_GUARD_BLOCKED
    if not confirmed or returned == "DRY_RUN":
        return DRY_RUN
    order_id = state.get("ack") or bool((result or {}).get("order_id"))
    if exc is not None or state.get("exception"):
        if order_id:
            return UNCONFIRMED_AFTER_ACK
        return SUBMIT_OUTCOME_UNKNOWN if state.get("sent") else NOT_SUBMITTED_ERROR
    if state.get("submit_error") and not order_id:
        return SUBMIT_REJECTED
    if not order_id:
        return SUBMIT_REJECTED if state.get("sent") else NOT_SUBMITTED_ERROR
    dealt = _num((result or {}).get("dealt_qty")) or 0.0
    qty = state.get("requested_qty") or 0.0
    # Brokers that never report stages (PaperBroker, fakes) are judged on
    # their returned status alone — and only a known terminal status counts.
    confirmed_terminal = state.get("status_confirmed") or (
        state.get("poll_count", 0) == 0 and not state.get("sent")
        and returned in TERMINAL_STATUSES)
    if not confirmed_terminal:
        return UNCONFIRMED_AFTER_ACK
    if dealt <= 0:
        return NOT_FILLED
    return FILLED if dealt >= qty else PARTIAL_FILLED


def _unknown_reason(state: dict, outcome: str) -> str:
    """Why an unconfirmed outcome is unconfirmed (P7.8), most specific first."""
    if outcome == SUBMIT_OUTCOME_UNKNOWN:
        return "SUBMIT_EXCEPTION_AFTER_SEND"
    if state.get("exception"):
        return "EXCEPTION_AFTER_ACK"
    all_polls_failed = state.get("poll_count", 0) == 0 and state.get("poll_failures", 0) > 0
    if all_polls_failed and state.get("cancel_failed"):
        return "ALL_POLLS_FAILED_AND_CANCEL_FAILED"
    if all_polls_failed:
        return "ALL_POLLS_FAILED"
    if state.get("cancel_failed"):
        return "CANCEL_FAILED"
    return f"NON_TERMINAL_AFTER_CANCEL:{state.get('broker_status')}"


def finish(result: Optional[dict] = None, exc: Optional[BaseException] = None,
           confirmed: bool = True) -> None:
    """Called by _place_order() after the broker returns (or raises)."""
    state = _current_order.get()
    if not state:
        return
    _current_order.set(None)
    try:
        outcome = _classify(state, confirmed, result, exc)
        if outcome in UNCONFIRMED:
            # Mirror first, so the P7 guard knows even if the DB write below fails.
            _unresolved_mem[state["audit_id"]] = {
                "code": state.get("code"), "side": state.get("side"),
                "env_label": state.get("env_label"), "outcome": outcome,
                "order_id": (result or {}).get("order_id") or state.get("order_id")}
        r = result or {}
        dealt = _num(r.get("dealt_qty"))
        fill = _num(r.get("dealt_avg_price"))
        expected = state.get("expected_price")
        slip_pct = slip_usd = None
        if outcome in (FILLED, PARTIAL_FILLED) and fill and expected:
            sign = 1.0 if state.get("side") == "BUY" else -1.0
            slip_pct = sign * (fill - expected) / expected       # >0 = worse than expected
            slip_usd = sign * (fill - expected) * (dealt or 0.0)
        fields = {
            "finished_at": _now(), "outcome": outcome,
            "returned_order_id": r.get("order_id"), "returned_status": r.get("status"),
            "returned_dealt_qty": dealt, "returned_fill_price": fill,
            "fill_confirmed": int(outcome not in UNCONFIRMED),
            "needs_reconcile": int(outcome in UNCONFIRMED),
            "partial_fill": int(outcome == PARTIAL_FILLED),
            "slippage_pct": slip_pct, "slippage_usd": slip_usd,
        }
        if exc is not None:
            fields["exception"] = f"{exc.__class__.__name__}: {exc}"
        if outcome in UNCONFIRMED:
            fields["unknown_reason"] = _unknown_reason(state, outcome)
            fields["reconcile_status"] = RECONCILE_PENDING
        _update(state["audit_id"], fields)
        finish_detail = {"outcome": outcome, "returned": r, "exception": fields.get("exception")}
        if outcome == PARTIAL_FILLED:
            # P7.3: partial fill + remainder cancelled at the broker.
            finish_detail["remainder_cancelled"] = r.get("status") in (
                "CANCELLED_PART", "FILL_CANCELLED", "CANCELLED_ALL")
        _append_event(state["audit_id"], "FINISH", finish_detail)
        if outcome in UNCONFIRMED:
            _append_event(state["audit_id"], "UNKNOWN", {
                "reason": fields["unknown_reason"], "order_id": r.get("order_id") or None,
                "returned_dealt_qty": dealt, "reconcile_status": RECONCILE_PENDING,
                "blocks_new_buy": True})
        if outcome in UNCONFIRMED:
            try:
                import notify.alert as alert
                alert.log(f"order_audit: {outcome} audit_id={state['audit_id']} — broker fill "
                          f"state not confirmed; returned dealt_qty={dealt}")
            except Exception:
                pass
    except Exception as e:
        _warn_once("finish", e)


# ── V3.6 P7 access for engine/execution_guard.py ──────────────────────────────

def current_pass_id() -> Optional[str]:
    p = _current_pass.get()
    return p.get("pass_id") if p else None


def current_audit_id() -> Optional[str]:
    state = _current_order.get()
    return state.get("audit_id") if state else None


def read_unresolved(env_label: Optional[str] = None, code: Optional[str] = None,
                    exclude_audit_id: Optional[str] = None) -> list:
    """Confirmed (real-submission) rows whose broker outcome is not proven
    final and agreed with the local book: an unconfirmed outcome not yet
    released by reconciliation, a reconciled row still needing manual
    action, or a row sent to the broker that never finished (the process
    died mid-order).

    RAISES on DB errors — unlike every other function here — so the guard
    can tell "nothing unresolved" from "could not look"."""
    released = ",".join("?" * len(RELEASED_STATUSES))
    sql = (f"SELECT * FROM order_audit WHERE confirmed=1 "
           f"AND COALESCE(reconcile_status,'{RECONCILE_PENDING}') NOT IN ({released}) "
           f"AND (needs_reconcile=1 OR reconcile_status IS NOT NULL "
           f"     OR (finished_at IS NULL AND submit_sent_at IS NOT NULL))")
    params: list = list(RELEASED_STATUSES)
    if env_label is not None:
        sql += " AND env_label=?"
        params.append(env_label)
    if code is not None:
        sql += " AND code=?"
        params.append(code)
    if exclude_audit_id is not None:
        sql += " AND audit_id<>?"
        params.append(exclude_audit_id)
    sql += " ORDER BY decision_at"
    with _lock:
        conn = _connect()
        try:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql, params)]
        finally:
            conn.close()


def read_known_order_ids(env_label: Optional[str]) -> set:
    """Every broker order_id already attributed to some audit row (so a
    lost-response search never adopts another row's order). Raises on DB errors."""
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT order_id, reconciled_order_id FROM order_audit WHERE env_label IS ?",
                (env_label,)).fetchall()
        finally:
            conn.close()
    return {str(v) for r in rows for v in r if v}


def unresolved_in_memory(env_label: Optional[str] = None, code: Optional[str] = None) -> dict:
    return {aid: v for aid, v in _unresolved_mem.items()
            if (env_label is None or v.get("env_label") == env_label)
            and (code is None or v.get("code") == code)}


def record_reconcile(audit_id: str, fields: dict, stage_name: str, detail: dict) -> None:
    """Persist one reconciliation step: update the P7 columns and append an
    order_audit_events row. Never raises (same contract as the rest)."""
    if fields.get("reconcile_status") in RELEASED_STATUSES:
        _unresolved_mem.pop(audit_id, None)
    try:
        _update(audit_id, fields)
        _append_event(audit_id, stage_name, detail)
    except Exception as exc:
        _warn_once(f"record_reconcile {stage_name}", exc)


def log_event(audit_id: str, stage_name: str, detail: dict) -> None:
    try:
        _append_event(audit_id, stage_name, detail)
    except Exception as exc:
        _warn_once(f"log_event {stage_name}", exc)
