# -*- coding: utf-8 -*-
"""
research/position_manager_v31_review.py — manual-refresh Observation Dataset
report for V3.1-A Position Manager (see position_manager/ package).

Pure research tooling, deliberately kept OUT of engine/ and NOT wired into
engine/runner.py's live trading pass — same "observation only" boundary as
research/early_failure_monitor.py / research/store.py's Mean Reversion
observation DB. It only ever READS
C:\\KabuData\\portfolio\\position_manager_v31_log.jsonl (written by
position_manager/__init__.py::log_decision() every run_once() pass) and, for
the forward-return check, fetches read-only kline data via
data.fetcher.fetch_kline(). It never imports engine.runner, never calls
_place_order, and never changes config.POSITION_MANAGER_V31_MODE or any
other trading parameter — nothing in this file can influence a live decision.

Two independent reports:

  summarize(rows)      — aggregate counts straight from the log: HOLD vs
      REDUCE, which of the four modules (confidence/hmm/volatility/drawdown)
      triggered how often, repeated-REDUCE runs (same symbol REDUCEd on
      consecutive evaluations with no HOLD in between — the "重复减仓" the
      V3.1 spec explicitly worries about), how often multiple modules fired
      simultaneously ("多信号聚合"/"conflicting signals"), and the
      distribution of reduction size.

  forward_check(rows)  — for REDUCE decisions old enough to have T+1/T+3/T+5
      trading days of price history since, fetches that history and reuses
      research/forward_outcomes.py::compute_forward_outcomes() (same
      machinery entry_quality.py/mean-reversion research already use) to
      classify each REDUCE as POTENTIAL_PREMATURE (price kept rising —
      trend possibly cut short too early) / LIKELY_PROTECTIVE (price fell —
      the reduction would have helped) / NEUTRAL / PENDING (not enough
      elapsed trading days yet). This is exactly the "REDUCE 之后 1/3/5 个
      交易日表现" the V3.1-A observation plan asks for — never a rule
      change by itself, just a dataset to review once enough samples exist.

Usage:
    python -m research.position_manager_v31_review              # summary only
    python -m research.position_manager_v31_review --forward     # + forward-return check
"""
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Optional

LOG_PATH = Path(r"C:\KabuData\portfolio\position_manager_v31_log.jsonl")

# A forward return at this horizon or better is "the trend kept going" —
# same order of magnitude as risk/sizing.py's per-trade risk budget, not
# fitted to any V3.1-A outcome (there isn't any data yet to fit to).
_PREMATURE_RETURN_THRESHOLD = 0.02
_PROTECTIVE_RETURN_THRESHOLD = -0.02


def load_log(path: Path = LOG_PATH) -> List[dict]:
    """Never raises — a missing/corrupt log degrades to an empty report,
    same discipline as every observation-layer reader elsewhere in this
    codebase (e.g. regime/__init__.py's own failure handling)."""
    if not path.exists():
        return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    rows.sort(key=lambda r: r.get("timestamp", ""))
    return rows


def summarize(rows: List[dict]) -> dict:
    total = len(rows)
    hold = sum(1 for r in rows if r.get("final_decision") == "HOLD")
    reduce_ = sum(1 for r in rows if r.get("final_decision") == "REDUCE")

    module_counts = {"confidence": 0, "hmm": 0, "volatility": 0, "drawdown": 0}
    conflicting = 0
    for r in rows:
        triggered = r.get("triggered_modules") or []
        for m in triggered:
            if m in module_counts:
                module_counts[m] += 1
        if len(triggered) >= 2:
            conflicting += 1

    # Repeated REDUCE: same symbol REDUCEd on two or more CONSECUTIVE
    # evaluations for that symbol (no HOLD in between) — rows are already
    # sorted by timestamp, so a per-symbol "last decision" scan is enough.
    last_decision = {}
    repeated_reduce = 0
    for r in rows:
        sym = r.get("symbol")
        decision = r.get("final_decision")
        if decision == "REDUCE" and last_decision.get(sym) == "REDUCE":
            repeated_reduce += 1
        last_decision[sym] = decision

    reduction_pcts = []
    for r in rows:
        if r.get("final_decision") != "REDUCE":
            continue
        cur = r.get("current_position") or 0
        amt = r.get("reduction_amount") or 0
        if cur > 0:
            reduction_pcts.append(amt / cur * 100.0)

    return {
        "total_evaluations": total,
        "hold": hold,
        "reduce": reduce_,
        "confidence_triggered": module_counts["confidence"],
        "hmm_triggered": module_counts["hmm"],
        "volatility_triggered": module_counts["volatility"],
        "drawdown_triggered": module_counts["drawdown"],
        "repeated_reduce": repeated_reduce,
        "conflicting_signals": conflicting,
        "avg_reduction_pct": statistics.mean(reduction_pcts) if reduction_pcts else None,
        "median_reduction_pct": statistics.median(reduction_pcts) if reduction_pcts else None,
        "max_reduction_pct": max(reduction_pcts) if reduction_pcts else None,
        # Structural invariants position_manager/__init__.py::evaluate()
        # guarantees by construction (see its docstring) — reported here as
        # an ongoing sanity check on the REAL log, not just the unit tests.
        "target_below_zero_count": sum(1 for r in rows if (r.get("target_position") or 0) < 0),
        "target_exceeds_current_count": sum(
            1 for r in rows if (r.get("target_position") or 0) > (r.get("current_position") or 0)),
    }


def print_summary(stats: dict) -> None:
    print("== V3.1-A Position Manager — Observation Summary ==")
    print(f"Total evaluations: {stats['total_evaluations']}")
    print(f"  HOLD:   {stats['hold']}")
    print(f"  REDUCE: {stats['reduce']}")
    print()
    print(f"Confidence triggered: {stats['confidence_triggered']}")
    print(f"HMM triggered:        {stats['hmm_triggered']}")
    print(f"Volatility triggered: {stats['volatility_triggered']}")
    print(f"Drawdown triggered:   {stats['drawdown_triggered']}")
    print()
    print(f"Repeated REDUCE (same symbol, no HOLD in between): {stats['repeated_reduce']}")
    print(f"Conflicting/multi-signal evaluations (>=2 modules): {stats['conflicting_signals']}")
    print()

    def _fmt(v):
        return f"{v:.2f}%" if v is not None else "n/a"
    print(f"Average reduction: {_fmt(stats['avg_reduction_pct'])}")
    print(f"Median reduction:  {_fmt(stats['median_reduction_pct'])}")
    print(f"Maximum reduction: {_fmt(stats['max_reduction_pct'])}")
    print()
    print(f"target_position < 0 (should always be 0):            {stats['target_below_zero_count']}")
    print(f"target_position > current_position (should be 0):    {stats['target_exceeds_current_count']}")


def _trading_days_elapsed(timestamp_iso: str) -> int:
    try:
        then = datetime.fromisoformat(timestamp_iso)
    except (ValueError, TypeError):
        return 0
    return max(0, (datetime.now() - then).days)


def forward_check(rows: List[dict]) -> List[dict]:
    """For every REDUCE row, fetch daily kline since its decision date and
    classify how price moved afterward. Read-only market data fetch only —
    see module docstring. Skips (PENDING) rows too recent to judge yet."""
    from data.fetcher import fetch_kline
    from research.forward_outcomes import compute_forward_outcomes

    results = []
    for r in rows:
        if r.get("final_decision") != "REDUCE":
            continue
        symbol = r.get("symbol")
        ts = r.get("timestamp")
        days_elapsed = _trading_days_elapsed(ts)
        if not symbol or not ts or days_elapsed < 1:
            results.append({**r, "forward_outcome": "PENDING", "forward_reason": "too recent"})
            continue

        try:
            df = fetch_kline(symbol, ktype="1d", bars=90)
            if df is None or "time_key" not in df.columns:
                results.append({**r, "forward_outcome": "PENDING", "forward_reason": "no kline data"})
                continue
            decision_date = ts[:10]
            df_after = df[df["time_key"].astype(str).str[:10] > decision_date].reset_index(drop=True)
            decision_price = r.get("current_price")
            forward = compute_forward_outcomes(df_after, decision_price)
        except Exception as exc:
            results.append({**r, "forward_outcome": "PENDING", "forward_reason": f"fetch failed: {exc}"})
            continue

        if not forward or forward.get("forward_bars_available", 0) < 1:
            results.append({**r, "forward_outcome": "PENDING", "forward_reason": "not enough bars yet"})
            continue

        best_return = None
        for h in (5, 3, 1):
            v = forward.get(f"return_fwd_t{h}")
            if v is not None:
                best_return = v
                break

        if best_return is None:
            outcome = "PENDING"
        elif best_return >= _PREMATURE_RETURN_THRESHOLD:
            outcome = "POTENTIAL_PREMATURE"
        elif best_return <= _PROTECTIVE_RETURN_THRESHOLD:
            outcome = "LIKELY_PROTECTIVE"
        else:
            outcome = "NEUTRAL"

        results.append({
            **r,
            "forward_outcome": outcome,
            "return_fwd_t1": forward.get("return_fwd_t1"),
            "return_fwd_t3": forward.get("return_fwd_t3"),
            "return_fwd_t5": forward.get("return_fwd_t5"),
        })
    return results


def print_forward_check(rows: List[dict]) -> None:
    print()
    print("== V3.1-A REDUCE — Forward Return Check (T+1/T+3/T+5) ==")
    if not rows:
        print("(no REDUCE decisions logged yet)")
        return
    for r in rows:
        outcome = r.get("forward_outcome", "PENDING")
        t1 = r.get("return_fwd_t1")
        t3 = r.get("return_fwd_t3")
        t5 = r.get("return_fwd_t5")
        fmt = lambda v: f"{v:+.2%}" if v is not None else "  n/a "
        print(f"{r.get('timestamp', '')[:19]}  {r.get('symbol', ''):<10}  "
              f"reduce={r.get('reduction_amount')}  "
              f"T1={fmt(t1)} T3={fmt(t3)} T5={fmt(t5)}  -> {outcome}")

    counts = {}
    for r in rows:
        outcome = r.get("forward_outcome", "PENDING")
        counts[outcome] = counts.get(outcome, 0) + 1
    print()
    for k in ("POTENTIAL_PREMATURE", "LIKELY_PROTECTIVE", "NEUTRAL", "PENDING"):
        print(f"{k}: {counts.get(k, 0)}")


if __name__ == "__main__":
    import sys

    all_rows = load_log()
    print_summary(summarize(all_rows))

    if "--forward" in sys.argv:
        print_forward_check(forward_check(all_rows))
