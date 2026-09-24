"""
research/execution_audit_report.py — V3.6 P5 order lifecycle report
(research-only, read-only).

Joins every engine/order_audit.py row to what happened AFTER the broker
call, so each order can be followed end to end:

    Signal/Decision -> Risk -> Preflight -> Submit -> Ack -> Fill   (order_audit)
      -> Local position record   (trade_history.db trades, by code + time)
      -> Reconcile               (next risk_state_log pass + reconciliation_diffs)

local_record:
  MATCHED            a trades row opened (BUY entry) / closed (full exit) for
                     this code within the window, with shares == dealt_qty
  QTY_MISMATCH       such a row exists but its shares differ from dealt_qty
  MISSING            the broker reported a fill but no trades row followed
  UNKNOWN_FILL       outcome unconfirmed — nobody knows what filled
  NOT_TRACKED        intent does not create/close a trades row (pyramid adds,
                     partial reductions, trims, rebalances)
  NO_FILL            nothing filled, so nothing to record
reconcile:
  CONSISTENT / MISMATCH (broker vs local qty diff for this code at the next
  reconcile pass) / NOT_YET_RECONCILED (no later pass on record).

Read-only everywhere; never imports engine.*, risk.*, strategies.*.
"""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

_AUDIT_DB = Path(r"C:\KabuData\execution\order_audit.db")
_TRADE_DB = Path(r"C:\KabuData\trade_history\trade_history.db")
_RISK_STATE_LOG = Path(r"C:\KabuData\portfolio\risk_state_log.jsonl")
_DIFF_LOG = Path(r"C:\KabuData\portfolio\reconciliation_diffs.jsonl")

_LOCAL_WINDOW = timedelta(minutes=5)
_RECONCILE_WINDOW = timedelta(days=1)
_DIFF_BEFORE_SNAPSHOT = timedelta(seconds=30)

_OPENS_TRADE = ("ENTRY", "QQQ_CORE_BUY")
_CLOSES_TRADE_PREFIX = ("EXIT:", "CORE_ETF_MA_EXIT", "REPLACEMENT_EXIT")
_FILLED = ("FILLED", "PARTIAL_FILLED")
_UNCONFIRMED = ("UNCONFIRMED_AFTER_ACK", "SUBMIT_OUTCOME_UNKNOWN")


def _naive(ts) -> Optional[datetime]:
    """Aware audit timestamps -> naive local, to compare with trade_history /
    risk-log timestamps (naive local, same machine)."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    try:
        dt = datetime.fromisoformat(str(ts))
    except ValueError:
        return None
    return dt.astimezone().replace(tzinfo=None) if dt.tzinfo else dt


def _read_sql(path: Path, sql: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, con)
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        return pd.DataFrame()
    finally:
        con.close()


def _read_jsonl(path: Path) -> list:
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def _local_record(order: dict, trades: pd.DataFrame) -> tuple:
    outcome, intent = order.get("outcome"), str(order.get("intent") or "")
    if outcome in _UNCONFIRMED:
        return "UNKNOWN_FILL", None
    if outcome not in _FILLED or not order.get("confirmed"):
        return "NO_FILL", None
    opens = intent in _OPENS_TRADE
    closes = intent.startswith(_CLOSES_TRADE_PREFIX)
    if not (opens or closes) or trades.empty:
        return ("NOT_TRACKED", None) if not (opens or closes) else ("MISSING", None)
    start = _naive(order.get("decision_at"))
    end = _naive(order.get("finished_at")) or start
    if start is None:
        return "MISSING", None
    col = "entry_time" if opens else "exit_time"
    cand = trades[trades["ticker"] == order.get("code")]
    for _, t in cand.iterrows():
        ts = _naive(t[col])
        if ts is not None and start - timedelta(seconds=5) <= ts <= end + _LOCAL_WINDOW:
            same = abs(float(t["shares"] or 0) - float(order.get("dealt_qty") or 0)) < 1e-6
            return ("MATCHED" if same else "QTY_MISMATCH"), t["trade_id"]
    return "MISSING", None


def _reconcile(order: dict, snapshots: list, diffs: list) -> tuple:
    end = _naive(order.get("finished_at"))
    if end is None:
        return "NOT_YET_RECONCILED", None, None
    nxt = next((s for s in snapshots if end < s <= end + _RECONCILE_WINDOW), None)
    if nxt is None:
        return "NOT_YET_RECONCILED", None, None
    hit = [d for d in diffs if d["code"] == order.get("code")
           and nxt - _DIFF_BEFORE_SNAPSHOT <= d["ts"] <= nxt]
    if hit:
        return "MISMATCH", nxt.isoformat(), hit[-1].get("diff_qty")
    return "CONSISTENT", nxt.isoformat(), None


def build_lifecycle(audit_db: Optional[Path] = None, trade_db: Optional[Path] = None,
                    risk_state_log: Optional[Path] = None,
                    diff_log: Optional[Path] = None) -> pd.DataFrame:
    orders = _read_sql(audit_db or _AUDIT_DB, "SELECT * FROM order_audit ORDER BY decision_at")
    if orders.empty:
        return orders
    trades = _read_sql(trade_db or _TRADE_DB,
                       "SELECT trade_id, ticker, entry_time, exit_time, shares FROM trades")
    snapshots = sorted(t for t in (_naive(r.get("timestamp"))
                                   for r in _read_jsonl(risk_state_log or _RISK_STATE_LOG)) if t)
    diffs = []
    for r in _read_jsonl(diff_log or _DIFF_LOG):
        ts = _naive(r.get("timestamp"))
        if ts is not None and r.get("code"):
            diffs.append({**r, "ts": ts})
    local, local_tid, rec, rec_at, rec_diff = [], [], [], [], []
    for order in orders.to_dict("records"):
        status, tid = _local_record(order, trades)
        local.append(status)
        local_tid.append(tid)
        r, at, d = _reconcile(order, snapshots, diffs) if order.get("confirmed") else (
            "NOT_APPLICABLE", None, None)
        rec.append(r)
        rec_at.append(at)
        rec_diff.append(d)
    orders["local_record"] = local
    orders["local_trade_id"] = local_tid
    orders["reconcile"] = rec
    orders["reconcile_at"] = rec_at
    orders["reconcile_diff_qty"] = rec_diff
    return orders


def summarize(lifecycle: pd.DataFrame) -> dict:
    if lifecycle.empty:
        return {"orders": 0}
    vc = lambda c: {str(k): int(v) for k, v in lifecycle[c].value_counts(dropna=False).items()}
    live = lifecycle[lifecycle["confirmed"] == 1]
    slip = live["slippage_pct"].dropna()
    return {
        "orders": int(len(lifecycle)),
        "confirmed_orders": int(len(live)),
        "by_outcome": vc("outcome"),
        "by_intent": vc("intent"),
        "by_env": vc("env_label"),
        "local_record": vc("local_record"),
        "reconcile": vc("reconcile"),
        "needs_reconcile": int(lifecycle["needs_reconcile"].fillna(0).sum()),
        "slippage_pct_mean": float(slip.mean()) if len(slip) else None,
        "slippage_n": int(len(slip)),
        "commission_recorded": int(lifecycle["commission"].notna().sum()),
    }


def render_markdown(lifecycle: pd.DataFrame) -> str:
    s = summarize(lifecycle)
    lines = ["# Execution Audit — order lifecycle", "", "```",
             json.dumps(s, indent=2, ensure_ascii=False), "```", ""]
    if lifecycle.empty:
        lines.append("_No audited orders yet._")
        return "\n".join(lines)
    cols = ["decision_at", "env_label", "code", "side", "intent", "requested_qty", "dealt_qty",
            "expected_price", "fill_price", "slippage_pct", "outcome", "local_record", "reconcile"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for r in lifecycle[cols].to_dict("records"):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values()) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_markdown(build_lifecycle()))
