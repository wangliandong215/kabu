"""
engine/execution_guard.py — V3.6 P7 Execution Safety: duplicate-BUY
protection + reconciliation of orders whose broker outcome is unknown.

The problem (reproduced end-to-end by engine/test_order_audit.py): pass N
submits a BUY, the broker ACKs it, polling then fails, MoomooBroker returns
dealt_qty=0 and the local book records nothing. Pass N+1 sees "not held"
and submits the same BUY again although the broker already filled the first.

Two independent protections, both checked by engine/runner.py::_place_order()
for every BUY (ENTRY / PYRAMID_* / QQQ_CORE_BUY / replacement), before
risk/preflight/broker — a blocked BUY returns the same zero-fill shape as
RISK_BLOCKED, so every caller's existing `dealt_qty > 0` gate handles it:

  1. Unresolved order (order_audit.db, P5): any confirmed order for the
     same code+env whose outcome is not proven final — SUBMIT_OUTCOME_UNKNOWN,
     UNCONFIRMED_AFTER_ACK (poll exception / all polls failed / cancel
     failed / still open after cancel), a row sent but never finished
     (process died mid-order), or a reconciled row whose broker fill the
     local book does not reflect (RESOLVED_NEEDS_MANUAL). Side-agnostic: an
     unresolved SELL also means the local book for that code is not trusted.
  2. Position mismatch (broker ground truth): this pass's broker-vs-local
     qty diff for the code (risk/portfolio_risk_manager.reconcile(), already
     computed every pass). Catches a fill the audit never saw. Codes in
     config.RECONCILIATION_IGNORE_CODES (US.0000 delisted EA ghost, the
     long-standing US.QQQ +124 book gap) are exempt, so today's known,
     stable diffs don't newly block the QQQ Beta top-up; rule 1 still
     covers any NEW unknown QQQ order.

Reconciliation (reconcile_pending(), top of every non-JP run_once() pass,
before any order): for each unresolved row, query the broker for the order
(by order_id; or, when the submit response was lost, by code/side/qty/price
among orders no other audit row owns), then:
  terminal status        -> final dealt_qty known. RESOLVED if the local
                            book already reflects it (dealt == what the pass
                            returned to the caller), else RESOLVED_NEEDS_MANUAL
                            — P7 never writes positions.json/trade_history.
  still open             -> cancel (confirmed passes only — the original
                            order path would have cancelled it too), re-query
                            once; still not terminal -> UNRESOLVED.
  no order_id + no match -> NOT_FOUND: the request never reached the broker
                            (RESOLVED, 0 filled; rule 2 still guards a miss).
  query failed / acked
  order_id not found /
  ambiguous match        -> UNRESOLVED; needs_manual after
                            MANUAL_AFTER_ATTEMPTS passes.
Never assumed NOT_FILLED without broker evidence. Every step is an
order_audit_events row (RECONCILE_START / RECONCILE_CANCEL_* /
RECONCILE_END with protection_released + needs_manual), and the order_audit
row carries reconcile_status / reconciled_* / needs_manual / manual_reason.

Operator CLI (clearing is the only way out of a needs_manual state):
    python -m engine.execution_guard list [--env SIMULATE]
    python -m engine.execution_guard clear AUDIT_ID --note "fixed positions.json"

Never raises into the trading pass. Does not touch Risk (V2.10/V3.3),
strategy, sizing or any normal FILLED/PARTIAL/NOT_FILLED/CANCELLED path: with
no unresolved rows and no mismatch it only adds one audit-DB read per BUY.
"""
import argparse
from datetime import datetime
from typing import Iterable, List, Optional

import notify.alert as alert
from engine import order_audit as oa

MANUAL_AFTER_ATTEMPTS = 3

# Broker-vs-local qty diffs noted for the current pass only (keyed by
# order_audit's pass_id, so a stale snapshot can never leak into a later pass).
_pass_mismatch: dict = {"pass_id": None, "env_label": None, "codes": {}}


def _now() -> str:
    return datetime.now().astimezone().isoformat()


# ── guard ─────────────────────────────────────────────────────────────────────

def note_position_diffs(diffs: Iterable, ignore_codes: Iterable[str], env_label: str) -> None:
    """Called by run_once() right after portfolio_risk_manager.reconcile()."""
    ignore = set(ignore_codes or ())
    _pass_mismatch.update(
        pass_id=oa.current_pass_id(), env_label=env_label,
        codes={d.code: (d.broker_qty, d.tracker_qty) for d in (diffs or ())
               if d.code not in ignore})


def _mismatch_for(code: str, env_label: str):
    pid = _pass_mismatch.get("pass_id")
    if pid is None or pid != oa.current_pass_id() or _pass_mismatch.get("env_label") != env_label:
        return None
    return _pass_mismatch["codes"].get(code)


def check_buy(code: str, env_label: str) -> Optional[str]:
    """None = safe to submit; otherwise the human-readable block reason."""
    reasons: List[str] = []
    own = oa.current_audit_id()
    db_ok = True
    seen = set()
    try:
        for r in oa.read_unresolved(env_label=env_label, code=code, exclude_audit_id=own):
            seen.add(r["audit_id"])
            reasons.append(
                f"UNRESOLVED_ORDER side={r.get('side')} audit_id={r['audit_id'][:12]} "
                f"order_id={r.get('order_id') or r.get('reconciled_order_id') or '-'} "
                f"outcome={r.get('outcome') or 'IN_FLIGHT'} "
                f"reconcile={r.get('reconcile_status') or oa.RECONCILE_PENDING}"
                + (f" unknown_reason={r['unknown_reason']}" if r.get("unknown_reason") else ""))
    except Exception as exc:
        db_ok = False
        alert.error(f"execution_guard: order_audit.db unreadable ({exc.__class__.__name__}: "
                    f"{exc}) — {code} BUY checked against in-process record only")
    mem = {aid: v for aid, v in oa.unresolved_in_memory(env_label, code).items()
           if aid != own and aid not in seen}
    if mem and db_ok:
        # The DB is authoritative for rows it has (e.g. cleared from the CLI
        # in another process); memory only speaks for rows it never got.
        mem = {aid: v for aid, v in mem.items() if not _audit_row_exists(aid)}
    for aid, v in mem.items():
        reasons.append(f"UNRESOLVED_ORDER side={v.get('side')} audit_id={aid[:12]} "
                       f"order_id={v.get('order_id') or '-'} outcome={v.get('outcome')} "
                       f"(in-process record)")
    mm = _mismatch_for(code, env_label)
    if mm is not None:
        reasons.append(f"POSITION_MISMATCH broker={mm[0]:g} local={mm[1]:g}")
    return "; ".join(reasons) or None


def _audit_row_exists(audit_id: str) -> bool:
    try:
        with oa._lock:
            conn = oa._connect()
            try:
                return conn.execute("SELECT 1 FROM order_audit WHERE audit_id=?",
                                    (audit_id,)).fetchone() is not None
            finally:
                conn.close()
    except Exception:
        return False


# ── reconciliation ────────────────────────────────────────────────────────────

def _since_date(row: dict) -> Optional[str]:
    ts = row.get("submit_sent_at") or row.get("decision_at")
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d") if ts else None
    except ValueError:
        return None


def _final_outcome(dealt: float, requested: float) -> str:
    if dealt <= 0:
        return oa.NOT_FILLED
    return oa.FILLED if dealt >= (requested or 0) else oa.PARTIAL_FILLED


def reconcile_pending(trd_env, env_label: str, confirmed: bool, broker=None) -> dict:
    """Reconcile every unresolved order of this env. Never raises. Makes no
    broker call at all when nothing is unresolved."""
    summary = {"checked": 0, "resolved": [], "unresolved": [], "needs_manual": []}
    try:
        rows = oa.read_unresolved(env_label=env_label)
    except Exception as exc:
        alert.error(f"execution_guard: reconcile skipped — order_audit.db unreadable ({exc})")
        return summary
    if not rows:
        return summary
    if broker is None:
        from engine.broker import MoomooBroker as broker
    for row in rows:
        try:
            if row.get("reconcile_status") == oa.RECONCILE_NEEDS_MANUAL:
                summary["needs_manual"].append(row["audit_id"])   # final state known; waiting on operator
                continue
            summary["checked"] += 1
            status = _reconcile_row(row, trd_env, env_label, confirmed, broker)
        except Exception as exc:
            alert.error(f"execution_guard: reconcile {row.get('code')} "
                        f"audit_id={row.get('audit_id')} crashed — {exc}")
            status = oa.RECONCILE_UNRESOLVED
        key = {oa.RECONCILE_RESOLVED: "resolved",
               oa.RECONCILE_NEEDS_MANUAL: "needs_manual"}.get(status, "unresolved")
        summary[key].append(row["audit_id"])
    if summary["needs_manual"]:
        alert.error(f"execution_guard: {len(summary['needs_manual'])} order(s) need manual "
                    f"reconciliation — BUY blocked for those codes "
                    f"(python -m engine.execution_guard list)")
    return summary


def _reconcile_row(row: dict, trd_env, env_label: str, confirmed: bool, broker) -> str:
    aid, code, side = row["audit_id"], row["code"], row["side"]
    attempt = int(row.get("reconcile_attempts") or 0) + 1
    order_id = row.get("order_id") or row.get("returned_order_id") or row.get("reconciled_order_id")
    oa.log_event(aid, "RECONCILE_START", {
        "attempt": attempt, "code": code, "side": side, "order_id": order_id,
        "outcome": row.get("outcome") or "IN_FLIGHT",
        "unknown_reason": row.get("unknown_reason"), "confirmed_pass": confirmed})

    ok, orders = broker.query_orders(code, trd_env, order_id=order_id or "",
                                     since_date=_since_date(row))
    if not ok:
        return _unresolved(row, attempt, "BROKER_QUERY_FAILED")

    if order_id:
        match = next((o for o in orders if o["order_id"] == str(order_id)), None)
        if match is None:
            return _unresolved(row, attempt, "ACKED_ORDER_NOT_FOUND_AT_BROKER")
    else:
        # Submit response lost: find the order among ones no audit row owns.
        known = oa.read_known_order_ids(env_label)
        req = row.get("requested_qty") or 0
        px = row.get("submitted_price")
        cands = [o for o in orders
                 if o["order_id"] not in known
                 and o["trd_side"].upper().startswith(str(side).upper())
                 and abs(o["qty"] - req) < 1e-6
                 and (px is None or abs(o["price"] - px) < 0.005)]
        if len(cands) > 1:
            return _unresolved(row, attempt, f"AMBIGUOUS_BROKER_MATCH:{len(cands)}",
                               needs_manual=True)
        if not cands:
            return _resolve(row, attempt, None, oa.RECONCILE_RESOLVED, "NOT_FOUND", 0.0, None,
                            note="no matching order at the broker — request never reached it")
        match = cands[0]

    status = match["order_status"]
    if status not in oa.TERMINAL_STATUSES:
        if not confirmed:
            return _unresolved(row, attempt, f"BROKER_ORDER_STILL_OPEN:{status} (dry-run pass, "
                                             f"not cancelled)")
        c_ok, c_err = broker.cancel_order(code, match["order_id"], trd_env)
        oa.log_event(aid, "RECONCILE_CANCEL_SENT" if c_ok else "RECONCILE_CANCEL_FAILED",
                     {"order_id": match["order_id"], "status_before": status, "error": c_err or None})
        ok2, again = broker.query_orders(code, trd_env, order_id=match["order_id"],
                                         since_date=_since_date(row))
        again_match = next((o for o in again if o["order_id"] == match["order_id"]), None) if ok2 else None
        if again_match is None or again_match["order_status"] not in oa.TERMINAL_STATUSES:
            now_status = again_match["order_status"] if again_match else "QUERY_FAILED"
            return _unresolved(row, attempt, f"BROKER_ORDER_STILL_OPEN:{now_status}")
        match = again_match

    dealt = match["dealt_qty"]
    outcome = _final_outcome(dealt, row.get("requested_qty"))
    returned = row.get("returned_dealt_qty") or 0.0
    if abs(dealt - returned) > 1e-6:
        return _resolve(row, attempt, match, oa.RECONCILE_NEEDS_MANUAL, outcome, dealt,
                        match["dealt_avg_price"],
                        note=(f"broker dealt {dealt:g} but the pass returned {returned:g} to the "
                              f"caller (local book short by {dealt - returned:+g}) — P7 does not "
                              f"adjust positions.json/trade_history; fix the book, then "
                              f"`python -m engine.execution_guard clear {aid}`"))
    return _resolve(row, attempt, match, oa.RECONCILE_RESOLVED, outcome, dealt,
                    match["dealt_avg_price"])


def _resolve(row, attempt, match, status, outcome, dealt, fill_price, note=None) -> str:
    aid = row["audit_id"]
    manual = status == oa.RECONCILE_NEEDS_MANUAL
    fields = {"reconcile_status": status, "reconcile_attempts": attempt,
              "reconciled_at": _now(),
              "reconciled_order_id": match["order_id"] if match else None,
              "reconciled_broker_status": match["order_status"] if match else "NOT_FOUND",
              "reconciled_outcome": outcome, "reconciled_dealt_qty": dealt,
              "reconciled_fill_price": fill_price, "needs_manual": int(manual),
              "manual_reason": note if manual else None}
    oa.record_reconcile(aid, fields, "RECONCILE_END", {
        "result": status, "broker_status": fields["reconciled_broker_status"],
        "final_outcome": outcome, "final_dealt_qty": dealt, "fill_price": fill_price,
        "returned_dealt_qty": row.get("returned_dealt_qty"),
        "protection_released": status in oa.RELEASED_STATUSES,
        "needs_manual": manual, "note": note})
    msg = (f"execution_guard: {row['code']} {row['side']} audit_id={aid[:12]} reconciled -> "
           f"{outcome} dealt={dealt:g} ({fields['reconciled_broker_status']})")
    if manual:
        alert.error(msg + f" — NEEDS MANUAL: {note}")
    else:
        alert.warn(msg + " — protection released")
    return status


def _unresolved(row, attempt, reason, needs_manual=False) -> str:
    aid = row["audit_id"]
    manual = needs_manual or attempt >= MANUAL_AFTER_ATTEMPTS
    oa.record_reconcile(aid, {
        "reconcile_status": oa.RECONCILE_UNRESOLVED, "reconcile_attempts": attempt,
        "reconciled_at": _now(), "needs_manual": int(manual),
        "manual_reason": reason if manual else None,
    }, "RECONCILE_END", {"result": oa.RECONCILE_UNRESOLVED, "reason": reason,
                         "attempt": attempt, "protection_released": False,
                         "needs_manual": manual})
    msg = (f"execution_guard: {row['code']} {row['side']} audit_id={aid[:12]} still unresolved "
           f"({reason}, attempt {attempt}) — BUY protection kept")
    if manual and not row.get("needs_manual"):
        alert.error(msg + " — NEEDS MANUAL")
    else:
        alert.warn(msg)
    return oa.RECONCILE_UNRESOLVED


# ── operator CLI ──────────────────────────────────────────────────────────────

def manual_clear(audit_id: str, note: str) -> bool:
    """Release one row's protection after a human has verified the broker
    and fixed the local book. Returns False if the row is not unresolved."""
    rows = [r for r in oa.read_unresolved() if r["audit_id"] == audit_id]
    if not rows:
        return False
    oa.record_reconcile(audit_id, {"reconcile_status": oa.RECONCILE_MANUAL_CLEARED,
                                   "reconciled_at": _now()},
                        "MANUAL_CLEAR", {"note": note, "previous": rows[0].get("reconcile_status"),
                                         "protection_released": True})
    return True


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m engine.execution_guard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="unresolved orders that block BUYs")
    ls.add_argument("--env")
    cl = sub.add_parser("clear", help="manually release one order's protection")
    cl.add_argument("audit_id")
    cl.add_argument("--note", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "list":
        rows = oa.read_unresolved(env_label=args.env)
        for r in rows:
            print(f"{r['audit_id']}  {r.get('env_label')}  {r['code']} {r['side']} "
                  f"qty={r.get('requested_qty')}  outcome={r.get('outcome') or 'IN_FLIGHT'}  "
                  f"reconcile={r.get('reconcile_status') or 'PENDING'}  "
                  f"attempts={r.get('reconcile_attempts') or 0}  "
                  f"order_id={r.get('order_id') or r.get('reconciled_order_id') or '-'}  "
                  f"manual={r.get('manual_reason') or '-'}")
        print(f"{len(rows)} unresolved")
        return 0
    if manual_clear(args.audit_id, args.note):
        print(f"cleared {args.audit_id}")
        return 0
    print(f"{args.audit_id}: not an unresolved order")
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
