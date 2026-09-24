"""
research/proposal_store.py — V3.6 Validation Gate: Proposal registry.

A Proposal is a candidate change to trading parameters (typically motivated
by a research/trade_intelligence_* pattern) that must walk the full
validation ladder before a human may apply it:

    DRAFT -> BACKTESTED -> OOS_PASSED -> PAPER -> GATE_PASSED -> APPROVED
      \\__________\\____________\\__________\\___________\\__-> REJECTED

Rules enforced here (twice: in Python, and again by SQLite triggers so a
raw UPDATE from anywhere else is refused too):
  * one step at a time — no skipping (e.g. DRAFT -> APPROVED is refused);
  * any non-terminal state may go to REJECTED, with a non-empty reason;
  * APPROVED and REJECTED rows are immutable; proposal_events is
    append-only (UPDATE/DELETE refused);
  * APPROVED requires approved_by — and approve() is only ever called by
    research/proposal_cli.py behind an interactive confirmation (a
    structural test asserts no other module calls it), so no automated
    process (AI included) can approve a proposal;
  * OOS must be strictly after the backtest period — the same data may not
    be used to both fit and validate.

APPROVED means "a human may now apply this change by hand". Nothing in this
module or proposal_cli.py writes config.py or changes any live trading
setting: config is only ever READ (getattr) to snapshot current values.
Applying an approved proposal stays a manual edit + commit.

Tables live in config.TRADE_INTELLIGENCE_DB_PATH (next to the
patterns/pattern_history tables source_pattern_id refers to), never in
trade_history.db. Never imports engine.*, risk.*, strategies.*,
portfolio.*, position_manager.*, exit_engine.*, ai_decision.*.
"""
import hashlib
import json
import sqlite3
import subprocess
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

import config

DRAFT = "DRAFT"
BACKTESTED = "BACKTESTED"
OOS_PASSED = "OOS_PASSED"
PAPER = "PAPER"
GATE_PASSED = "GATE_PASSED"
APPROVED = "APPROVED"
REJECTED = "REJECTED"

LADDER = (DRAFT, BACKTESTED, OOS_PASSED, PAPER, GATE_PASSED, APPROVED)
TERMINAL = (APPROVED, REJECTED)
ALL_STATUSES = LADDER + (REJECTED,)

# The one legal forward step out of each non-terminal state. REJECTED is
# additionally reachable from every non-terminal state.
_NEXT = {LADDER[i]: LADDER[i + 1] for i in range(len(LADDER) - 1)}

# Paper Trading must show at least this many closed trades before the gate
# will pass — the same "don't decide on a handful of trades" principle as
# the 10/20/30/50 sample ladder. Deliberately a module constant, not config.
MIN_PAPER_TRADES = 10

# approved_by must name a person, not a process.
_NON_HUMAN_APPROVERS = {"ai", "llm", "claude", "gpt", "bot", "system", "auto",
                        "automation", "script", "runner", "research"}

_DEFAULT_DB_PATH = Path(config.TRADE_INTELLIGENCE_DB_PATH)
_write_lock = threading.Lock()


class ProposalError(Exception):
    """Base class — every refusal from this module raises a subclass."""


class InvalidTransition(ProposalError):
    pass


class ValidationFailed(ProposalError):
    pass


def _legal_pairs() -> list:
    pairs = [(a, b) for a, b in _NEXT.items()]
    pairs += [(s, REJECTED) for s in LADDER if s not in TERMINAL]
    return pairs


def _trigger_sql() -> str:
    legal = " OR ".join(f"(OLD.status='{a}' AND NEW.status='{b}')" for a, b in _legal_pairs())
    statuses = ", ".join(f"'{s}'" for s in ALL_STATUSES)
    terminal = ", ".join(f"'{s}'" for s in TERMINAL)
    return f"""
CREATE TRIGGER IF NOT EXISTS proposals_insert_draft_only
BEFORE INSERT ON proposals WHEN NEW.status != '{DRAFT}'
BEGIN SELECT RAISE(ABORT, 'proposal must be created as {DRAFT}'); END;

CREATE TRIGGER IF NOT EXISTS proposals_terminal_immutable
BEFORE UPDATE ON proposals WHEN OLD.status IN ({terminal})
BEGIN SELECT RAISE(ABORT, 'proposal is terminal (APPROVED/REJECTED) and immutable'); END;

CREATE TRIGGER IF NOT EXISTS proposals_status_ladder
BEFORE UPDATE OF status ON proposals
WHEN NEW.status != OLD.status AND NOT ({legal})
BEGIN SELECT RAISE(ABORT, 'illegal proposal status transition'); END;

CREATE TRIGGER IF NOT EXISTS proposals_status_known
BEFORE UPDATE OF status ON proposals WHEN NEW.status NOT IN ({statuses})
BEGIN SELECT RAISE(ABORT, 'unknown proposal status'); END;

CREATE TRIGGER IF NOT EXISTS proposals_approval_needs_human
BEFORE UPDATE OF status ON proposals
WHEN NEW.status = '{APPROVED}' AND (NEW.approved_by IS NULL OR trim(NEW.approved_by) = '')
BEGIN SELECT RAISE(ABORT, 'APPROVED requires approved_by'); END;

CREATE TRIGGER IF NOT EXISTS proposals_no_delete
BEFORE DELETE ON proposals
BEGIN SELECT RAISE(ABORT, 'proposals are never deleted'); END;

CREATE TRIGGER IF NOT EXISTS proposal_events_no_update
BEFORE UPDATE ON proposal_events
BEGIN SELECT RAISE(ABORT, 'proposal_events is append-only'); END;

CREATE TRIGGER IF NOT EXISTS proposal_events_no_delete
BEFORE DELETE ON proposal_events
BEGIN SELECT RAISE(ABORT, 'proposal_events is append-only'); END;
"""


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id            TEXT NOT NULL,
    version                INTEGER NOT NULL,
    title                  TEXT NOT NULL,
    rationale              TEXT,
    source_pattern_id      TEXT,
    created_by             TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL,
    status                 TEXT NOT NULL CHECK (status IN ({", ".join(f"'{s}'" for s in ALL_STATUSES)})),
    params_before_json     TEXT NOT NULL,
    params_before_hash     TEXT NOT NULL,
    params_after_json      TEXT NOT NULL,
    params_after_hash      TEXT NOT NULL,
    git_commit             TEXT,
    backtest_result_json   TEXT,
    oos_result_json        TEXT,
    paper_result_json      TEXT,
    gate_result_json       TEXT,
    rejection_reason       TEXT,
    rejected_from_status   TEXT,
    rejected_by            TEXT,
    rejected_at            TEXT,
    approved_by            TEXT,
    approved_at            TEXT,
    PRIMARY KEY (proposal_id, version)
);

CREATE TABLE IF NOT EXISTS proposal_events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id   TEXT NOT NULL,
    version       INTEGER NOT NULL,
    at            TEXT NOT NULL,
    actor         TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    from_status   TEXT,
    to_status     TEXT,
    payload_json  TEXT,
    note          TEXT
);
{_trigger_sql()}
"""


# ── pure helpers ──────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    """config holds tuples/sets/dicts; normalize so the JSON (and its hash)
    is stable regardless of set iteration order."""
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(v) for v in value), key=lambda v: json.dumps(v, default=str))
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _dumps(obj) -> str:
    return json.dumps(_jsonable(obj), sort_keys=True, ensure_ascii=False, default=str)


def params_hash(params: dict) -> str:
    """Stable short hash — same recipe as engine/trade_tracker.py's
    compute_parameter_hash() (reimplemented, research must not import engine)."""
    return hashlib.sha256(_dumps(params).encode("utf-8")).hexdigest()[:16]


def snapshot_params(names: Iterable[str]) -> dict:
    """Current values of the named config constants. READ ONLY (getattr).
    Unknown names raise — a proposal may only change parameters that exist."""
    out = {}
    for name in names:
        if not hasattr(config, name):
            raise ValidationFailed(f"config has no parameter {name!r}")
        out[name] = _jsonable(getattr(config, name))
    return out


def detect_git_commit() -> Optional[str]:
    """Short HEAD, suffixed "-dirty" when the working tree has uncommitted
    changes (so a snapshot taken on unreviewed code is visibly marked).
    Never raises."""
    repo = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo),
                              capture_output=True, text=True, timeout=5)
        if head.returncode != 0 or not head.stdout.strip():
            return None
        commit = head.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo),
                               capture_output=True, text=True, timeout=5)
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += "-dirty"
        return commit
    except Exception:
        return None


def _require_text(value, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationFailed(f"{what} must be a non-empty string")
    return value.strip()


def _require_period(result: dict, what: str) -> tuple:
    if not isinstance(result, dict) or not result:
        raise ValidationFailed(f"{what} result must be a non-empty dict")
    try:
        start = date.fromisoformat(str(result["period_start"])[:10])
        end = date.fromisoformat(str(result["period_end"])[:10])
    except (KeyError, ValueError) as exc:
        raise ValidationFailed(
            f"{what} result needs ISO period_start/period_end ({exc})") from None
    if end < start:
        raise ValidationFailed(f"{what} period_end {end} is before period_start {start}")
    return start, end


def _require_bool(result: dict, key: str, what: str) -> bool:
    if not isinstance(result.get(key), bool):
        raise ValidationFailed(f"{what} result needs boolean {key!r}")
    return result[key]


def _require_human(name) -> str:
    name = _require_text(name, "approved_by")
    if name.lower() in _NON_HUMAN_APPROVERS:
        raise ValidationFailed(f"approved_by={name!r} is not a person")
    return name


def evaluate_gate(row: dict) -> dict:
    """Pure deterministic Validation Gate over a proposal row. Returns
    {"passed": bool, "reasons": [...failures...], "checked_at": ...}."""
    reasons = []
    bt = json.loads(row["backtest_result_json"]) if row.get("backtest_result_json") else None
    oos = json.loads(row["oos_result_json"]) if row.get("oos_result_json") else None
    paper = json.loads(row["paper_result_json"]) if row.get("paper_result_json") else None
    if bt is None:
        reasons.append("missing backtest result")
    if oos is None or oos.get("passed") is not True:
        reasons.append("OOS not passed")
    if paper is None:
        reasons.append("missing paper result")
    else:
        if paper.get("passed") is not True:
            reasons.append("paper result not passed")
        n = paper.get("n_trades")
        if not isinstance(n, int) or isinstance(n, bool) or n < MIN_PAPER_TRADES:
            reasons.append(f"paper n_trades={n!r} < {MIN_PAPER_TRADES}")
        if oos is not None:
            try:
                if date.fromisoformat(str(paper["period_start"])[:10]) < \
                        date.fromisoformat(str(oos["period_end"])[:10]):
                    reasons.append("paper period overlaps OOS period")
            except (KeyError, ValueError):
                reasons.append("paper/OOS period missing")
    return {"passed": not reasons, "reasons": reasons, "checked_at": _now()}


# ── store ─────────────────────────────────────────────────────────────────────

class ProposalStore:
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

    # ── reads ──
    def get(self, proposal_id: str, version: Optional[int] = None) -> dict:
        if version is None:
            row = self._conn.execute(
                "SELECT * FROM proposals WHERE proposal_id=? ORDER BY version DESC LIMIT 1",
                (proposal_id,)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM proposals WHERE proposal_id=? AND version=?",
                (proposal_id, version)).fetchone()
        if row is None:
            raise ProposalError(f"no proposal {proposal_id!r} version {version!r}")
        return dict(row)

    def list(self, status: Optional[str] = None) -> pd.DataFrame:
        q = ("SELECT proposal_id, version, status, title, source_pattern_id, "
             "created_by, created_at, updated_at, approved_by, rejection_reason "
             "FROM proposals")
        if status is None:
            return pd.read_sql_query(q + " ORDER BY created_at", self._conn)
        return pd.read_sql_query(q + " WHERE status=? ORDER BY created_at",
                                 self._conn, params=(status,))

    def events(self, proposal_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM proposal_events WHERE proposal_id=? ORDER BY event_id",
            self._conn, params=(proposal_id,))

    # ── internals ──
    def _event(self, proposal_id, version, actor, event_type, from_status,
               to_status, payload=None, note=None) -> None:
        self._conn.execute(
            """INSERT INTO proposal_events (proposal_id, version, at, actor, event_type,
                   from_status, to_status, payload_json, note)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (proposal_id, version, _now(), actor, event_type, from_status, to_status,
             _dumps(payload) if payload is not None else None, note))

    def _transition(self, proposal_id: str, version: int, expected: str, to: str,
                    actor: str, event_type: str, fields: Optional[dict] = None,
                    payload=None, note=None) -> dict:
        """Compare-and-set: moves expected -> to only if the row is still in
        `expected` (so two concurrent callers can't both advance it), writes
        `fields` in the same UPDATE, and appends the event atomically."""
        if to != REJECTED and _NEXT.get(expected) != to:
            raise InvalidTransition(f"{expected} -> {to} is not a legal step")
        if to == REJECTED and expected in TERMINAL:
            raise InvalidTransition(f"{expected} is terminal")
        fields = dict(fields or {})
        fields["status"] = to
        fields["updated_at"] = _now()
        sets = ", ".join(f"{k}=?" for k in fields)
        with _write_lock:
            try:
                cur = self._conn.execute(
                    f"UPDATE proposals SET {sets} WHERE proposal_id=? AND version=? AND status=?",
                    (*fields.values(), proposal_id, version, expected))
                if cur.rowcount != 1:
                    self._conn.rollback()
                    current = self.get(proposal_id, version)["status"]
                    raise InvalidTransition(
                        f"{proposal_id} v{version} is {current}, expected {expected}")
                self._event(proposal_id, version, actor, event_type, expected, to,
                            payload, note)
                self._conn.commit()
            except sqlite3.DatabaseError as exc:
                self._conn.rollback()
                raise InvalidTransition(str(exc)) from exc
        return self.get(proposal_id, version)

    def _require_status(self, proposal_id, version, status) -> dict:
        row = self.get(proposal_id, version)
        if row["status"] != status:
            raise InvalidTransition(
                f"{proposal_id} v{row['version']} is {row['status']}, needs {status}")
        return row

    def _check_before_still_current(self, row: dict) -> None:
        """A proposal is only meaningful against the parameter values it was
        written against — if config has moved on since, it's stale."""
        before = json.loads(row["params_before_json"])
        now = snapshot_params(before.keys())
        if params_hash(now) != row["params_before_hash"]:
            changed = sorted(k for k in before if before[k] != now.get(k))
            raise ValidationFailed(
                f"config changed since proposal was drafted ({changed}) — "
                f"reject and revise it against current values")

    def _pattern_exists(self, pattern_id: str) -> bool:
        has_table = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='patterns'").fetchone()
        if not has_table:
            return False
        return self._conn.execute(
            "SELECT 1 FROM patterns WHERE pattern_id=?", (pattern_id,)).fetchone() is not None

    # ── lifecycle ──
    def create(self, title: str, changes: dict, created_by: str,
               rationale: str = "", source_pattern_id: Optional[str] = None,
               _proposal_id: Optional[str] = None, _version: int = 1) -> dict:
        """New proposal in DRAFT. `changes` = {CONFIG_NAME: proposed_value};
        the before snapshot is read from config right now, the after
        snapshot is before + changes. Nothing is written anywhere but the DB."""
        title = _require_text(title, "title")
        created_by = _require_text(created_by, "created_by")
        if not isinstance(changes, dict) or not changes:
            raise ValidationFailed("changes must be a non-empty {name: value} dict")
        before = snapshot_params(changes.keys())
        after = dict(before)
        after.update({k: _jsonable(v) for k, v in changes.items()})
        if params_hash(after) == params_hash(before):
            raise ValidationFailed("changes do not differ from current config values")
        if source_pattern_id is not None and not self._pattern_exists(source_pattern_id):
            raise ValidationFailed(f"source_pattern_id {source_pattern_id!r} not found in patterns")

        proposal_id = _proposal_id or (
            f"PR-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:6]}")
        ts = _now()
        with _write_lock:
            self._conn.execute(
                """INSERT INTO proposals (proposal_id, version, title, rationale,
                       source_pattern_id, created_by, created_at, updated_at, status,
                       params_before_json, params_before_hash, params_after_json,
                       params_after_hash, git_commit)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (proposal_id, _version, title, rationale, source_pattern_id, created_by,
                 ts, ts, DRAFT, _dumps(before), params_hash(before), _dumps(after),
                 params_hash(after), detect_git_commit()))
            self._event(proposal_id, _version, created_by, "CREATED", None, DRAFT,
                        {"changes": changes})
            self._conn.commit()
        return self.get(proposal_id, _version)

    def revise(self, proposal_id: str, changes: dict, created_by: str,
               rationale: str = "") -> dict:
        """New version (DRAFT) of a REJECTED proposal — the rejected version
        stays untouched, so old/new/reason are all kept side by side."""
        latest = self.get(proposal_id)
        if latest["status"] != REJECTED:
            raise InvalidTransition(
                f"only a REJECTED proposal can be revised ({proposal_id} is {latest['status']})")
        return self.create(latest["title"], changes, created_by,
                           rationale=rationale or latest["rationale"],
                           source_pattern_id=latest["source_pattern_id"],
                           _proposal_id=proposal_id, _version=latest["version"] + 1)

    def record_backtest(self, proposal_id: str, version: int, result: dict,
                        actor: str) -> dict:
        _require_period(result, "backtest")
        return self._transition(proposal_id, version, DRAFT, BACKTESTED,
                                _require_text(actor, "actor"), "BACKTEST_RECORDED",
                                {"backtest_result_json": _dumps(result)}, result)

    def record_oos(self, proposal_id: str, version: int, result: dict,
                   actor: str) -> dict:
        """OOS period must start strictly after the backtest period ends. A
        failed OOS (result["passed"] is False) rejects the proposal."""
        actor = _require_text(actor, "actor")
        row = self._require_status(proposal_id, version, BACKTESTED)
        oos_start, _ = _require_period(result, "OOS")
        passed = _require_bool(result, "passed", "OOS")
        _, bt_end = _require_period(json.loads(row["backtest_result_json"]), "backtest")
        if oos_start <= bt_end:
            raise ValidationFailed(
                f"OOS period_start {oos_start} must be after backtest period_end {bt_end} "
                f"(same data may not be used to fit and validate)")
        if not passed:
            return self._transition(
                proposal_id, version, BACKTESTED, REJECTED, actor, "OOS_FAILED",
                {"oos_result_json": _dumps(result), "rejection_reason": "OOS failed",
                 "rejected_from_status": BACKTESTED, "rejected_by": actor,
                 "rejected_at": _now()}, result)
        return self._transition(proposal_id, version, BACKTESTED, OOS_PASSED, actor,
                                "OOS_RECORDED", {"oos_result_json": _dumps(result)}, result)

    def start_paper(self, proposal_id: str, version: int, actor: str,
                    note: Optional[str] = None) -> dict:
        return self._transition(proposal_id, version, OOS_PASSED, PAPER,
                                _require_text(actor, "actor"), "PAPER_STARTED", note=note)

    def record_paper_result(self, proposal_id: str, version: int, result: dict,
                            actor: str) -> dict:
        """May be called repeatedly while in PAPER (latest wins, every call
        is kept in proposal_events). Does not change status."""
        actor = _require_text(actor, "actor")
        _require_period(result, "paper")
        _require_bool(result, "passed", "paper")
        with _write_lock:
            cur = self._conn.execute(
                "UPDATE proposals SET paper_result_json=?, updated_at=? "
                "WHERE proposal_id=? AND version=? AND status=?",
                (_dumps(result), _now(), proposal_id, version, PAPER))
            if cur.rowcount != 1:
                self._conn.rollback()
                raise InvalidTransition(
                    f"{proposal_id} v{version} is not in {PAPER}")
            self._event(proposal_id, version, actor, "PAPER_RESULT_RECORDED",
                        PAPER, PAPER, result)
            self._conn.commit()
        return self.get(proposal_id, version)

    def pass_gate(self, proposal_id: str, version: int, actor: str) -> dict:
        """Deterministic Validation Gate (evaluate_gate) + config staleness
        check. On failure the status is unchanged, a GATE_CHECK_FAILED event
        is appended, and ValidationFailed is raised with the reasons."""
        actor = _require_text(actor, "actor")
        row = self._require_status(proposal_id, version, PAPER)
        gate = evaluate_gate(row)
        try:
            self._check_before_still_current(row)
        except ValidationFailed as exc:
            gate["passed"] = False
            gate["reasons"].append(str(exc))
        if not gate["passed"]:
            with _write_lock:
                self._event(proposal_id, version, actor, "GATE_CHECK_FAILED",
                            PAPER, PAPER, gate)
                self._conn.commit()
            raise ValidationFailed("; ".join(gate["reasons"]))
        return self._transition(proposal_id, version, PAPER, GATE_PASSED, actor,
                                "GATE_PASSED", {"gate_result_json": _dumps(gate)}, gate)

    def approve(self, proposal_id: str, version: int, approved_by: str,
                note: Optional[str] = None) -> dict:
        """GATE_PASSED -> APPROVED. ONLY research/proposal_cli.py may call
        this (enforced by a structural test) — see module docstring.
        Approval does not apply anything: it records that a named person
        signed off. The config staleness check is re-run here."""
        approved_by = _require_human(approved_by)
        row = self._require_status(proposal_id, version, GATE_PASSED)
        self._check_before_still_current(row)
        return self._transition(proposal_id, version, GATE_PASSED, APPROVED, approved_by,
                                "APPROVED", {"approved_by": approved_by,
                                             "approved_at": _now()}, note=note)

    def reject(self, proposal_id: str, version: int, reason: str, actor: str) -> dict:
        reason = _require_text(reason, "rejection reason")
        actor = _require_text(actor, "actor")
        row = self.get(proposal_id, version)
        if row["status"] in TERMINAL:
            raise InvalidTransition(f"{proposal_id} v{version} is {row['status']} (terminal)")
        return self._transition(proposal_id, version, row["status"], REJECTED, actor,
                                "REJECTED", {"rejection_reason": reason,
                                             "rejected_from_status": row["status"],
                                             "rejected_by": actor, "rejected_at": _now()},
                                note=reason)
