# -*- coding: utf-8 -*-
"""
position_manager/state_store.py — independent per-symbol baseline state for
V3.1-A Position Manager.

Deliberately NOT part of portfolio/tracker.py's position schema
(entry_price/avg_cost/qty/strategy/entry_atr/...) — that schema is read by
portfolio/capacity_manager.py, risk/portfolio_risk_manager.py's reconcile(),
and broker diff-growth checks; adding fields to it risks touching all of
those. This store is 100% additive: Position Manager only ever reads
Portfolio via its existing get_position()/data["positions"] API, and keeps
its own bookkeeping (peak_price, confidence/HMM baselines, the "largest
confirmed size this holding period" reference) entirely separately.

Same persistence pattern as portfolio/tracker.py and engine/regime_store.py:
one JSON file, atomic write (temp + os.replace), module-level lock.

Keyed by (symbol, entry_time) — entry_time is Portfolio's own microsecond-
precision ISO string, set fresh by Portfolio.open_position() and never
touched by add_to_position() (pyramid/promotion adds keep it). A stored
entry whose entry_time no longer matches the live position's entry_time
means the previous holding period fully closed and this is a new one, so
get_or_init() resets all baselines instead of reusing stale values from an
unrelated past trade — no separate "clear on close" hook is needed anywhere
else in the codebase for this to stay correct.
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

_STORE_PATH = Path(r"C:\KabuData\portfolio\position_manager_state.json")
_write_lock = threading.Lock()


def _load() -> dict:
    if not _STORE_PATH.exists():
        return {"symbols": {}}
    try:
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"symbols": {}}
    data.setdefault("symbols", {})
    return data


def _save(data: dict) -> None:
    with _write_lock:
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STORE_PATH.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, _STORE_PATH)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise


def get_or_init(symbol: str, entry_time: str, entry_price: float,
                 entry_qty: int, current_price: float,
                 current_confidence: Optional[float],
                 current_hmm_state: Optional[str]) -> dict:
    """Return this holding period's baseline entry, (re)creating it when
    either this is the first time Position Manager has seen `symbol`, or
    the stored entry's entry_time doesn't match `entry_time` (previous
    holding period closed, this is a fresh one)."""
    data = _load()
    entry = data["symbols"].get(symbol)
    if entry is None or entry.get("entry_time") != entry_time:
        entry = {
            "entry_time": entry_time,
            "baseline_qty": entry_qty,
            "peak_price": max(entry_price, current_price),
            "confidence_baseline": current_confidence,
            "hmm_state_baseline": current_hmm_state,
            "first_seen_at": datetime.now().isoformat(),
        }
        data["symbols"][symbol] = entry
        _save(data)
    return entry


def update(symbol: str, *, peak_price: Optional[float] = None,
           baseline_qty: Optional[int] = None) -> dict:
    """Monotonic merge-update: peak_price and baseline_qty only ever grow
    (max() with the stored value), so a past Position Manager REDUCE (or a
    price dip) can never un-grow either one. confidence_baseline/
    hmm_state_baseline are intentionally NOT updated here — they stay fixed
    for the whole holding period (see get_or_init) so confidence_change/
    hmm_state_change always read as "vs. entry", matching the spec's own
    worked example (e.g. "Confidence: 0.82 -> 0.67"). No-op if `symbol` has
    no entry yet (caller must get_or_init() first). Returns the entry."""
    data = _load()
    entry = data["symbols"].get(symbol)
    if entry is None:
        return {}
    if peak_price is not None:
        entry["peak_price"] = max(entry.get("peak_price", peak_price), peak_price)
    if baseline_qty is not None:
        entry["baseline_qty"] = max(entry.get("baseline_qty", baseline_qty), baseline_qty)
    _save(data)
    return entry
