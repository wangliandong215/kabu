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
Both are marked fill_confirmed=0 / needs_reconcile=1. Nothing here acts on
them (that is P7); this module only records.

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

# Same set as engine/broker.py MoomooBroker._TERMINAL (not imported: broker
# imports this module). A test asserts the two stay identical.
TERMINAL_STATUSES = frozenset({"FILLED_ALL", "CANCELLED_ALL", "FAILED", "DISABLED",
                               "DELETED", "SUBMIT_FAILED", "FILL_CANCELLED"})

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
                 "side": side, "ack": False, "sent": False, "status_confirmed": False,
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
            "preflight_reason", "broker")}
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


def finish(result: Optional[dict] = None, exc: Optional[BaseException] = None,
           confirmed: bool = True) -> None:
    """Called by _place_order() after the broker returns (or raises)."""
    state = _current_order.get()
    if not state:
        return
    _current_order.set(None)
    try:
        outcome = _classify(state, confirmed, result, exc)
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
        _update(state["audit_id"], fields)
        _append_event(state["audit_id"], "FINISH", {"outcome": outcome, "returned": r,
                                                    "exception": fields.get("exception")})
        if outcome in UNCONFIRMED:
            try:
                import notify.alert as alert
                alert.log(f"order_audit: {outcome} audit_id={state['audit_id']} — broker fill "
                          f"state not confirmed; returned dealt_qty={dealt}")
            except Exception:
                pass
    except Exception as e:
        _warn_once("finish", e)
