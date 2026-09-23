# -*- coding: utf-8 -*-
"""
exit_engine/state_store.py — independent per-symbol peak-price/peak-date
baseline for V3.2-A Minute Exit Engine.

Deliberately NOT part of portfolio/tracker.py's position schema and NOT
position_manager/state_store.py's own store — this package's peak tracking
serves a different purpose (ATR-stop/trailing-stop reference price, and
"days since last new high" for the time-exit signal), so it keeps entirely
separate bookkeeping under its own file, exactly the same reasoning
position_manager/state_store.py's own docstring gives for not reusing
Portfolio's schema.

Same persistence pattern as position_manager/state_store.py: one JSON
file, atomic write (temp + os.replace), module-level lock, keyed by
(symbol, entry_time) so a closed-then-reopened position starts a fresh
baseline instead of inheriting a stale peak from an unrelated past trade.
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

_STORE_PATH = Path(r"C:\KabuData\portfolio\exit_engine_state.json")
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
                 current_price: float, today_iso: str) -> dict:
    """Return this holding period's baseline entry, (re)creating it when
    either this is the first time this package has seen `symbol`, or the
    stored entry's entry_time doesn't match `entry_time` (previous holding
    period closed, this is a fresh one)."""
    data = _load()
    entry = data["symbols"].get(symbol)
    if entry is None or entry.get("entry_time") != entry_time:
        entry = {
            "entry_time": entry_time,
            "peak_price": max(entry_price, current_price),
            "peak_date": today_iso,
            "first_seen_at": datetime.now().isoformat(),
        }
        data["symbols"][symbol] = entry
        _save(data)
    return entry


def update(symbol: str, *, price: Optional[float] = None,
           today_iso: Optional[str] = None) -> dict:
    """Monotonic peak_price high-water mark: only ever grows. peak_date
    only advances alongside it (a genuine new high made "today"), so "days
    since last new high" is simply today - peak_date. No-op if `symbol`
    has no entry yet (caller must get_or_init() first). Returns the entry."""
    data = _load()
    entry = data["symbols"].get(symbol)
    if entry is None:
        return {}
    if price is not None and price > entry.get("peak_price", price):
        entry["peak_price"] = price
        entry["peak_date"] = today_iso
    _save(data)
    return entry
