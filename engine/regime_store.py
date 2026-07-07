# -*- coding: utf-8 -*-
"""
engine/regime_store.py — durable per-stock regime state (V2.7 Stage 3, Step 2).

This is the "Regime Interface" the task spec asks for: current_regime,
regime_confidence, regime_duration, regime_changed_today. All future
consumers should read it via get_regime_interface(code) — never call
engine/hmm_regime.py's MLRegimeClassifier directly — so the HMM/label scheme
can change later without touching every call site. Same discipline as
portfolio/tracker.py's Portfolio class isolating trading state behind a
read/write API instead of raw JSON access everywhere.

Persistence follows portfolio/tracker.py's exact pattern: a single JSON file,
atomic write (temp file + os.replace, which is atomic on both NTFS and
POSIX), guarded by a module-level lock so concurrent passes can't corrupt it.

NOT wired into any trading decision as of V2.7 Stage 3. The only writer as of
this stage is engine/hmm_shadow.py's daily Shadow Mode pass; nothing in
engine/runner.py, risk/, or strategies/ reads from this store yet.
"""
import json
import os
import threading
from pathlib import Path

_STORE_PATH = Path(__file__).parent / "regime_state.json"
_write_lock = threading.Lock()


def _load() -> dict:
    if not _STORE_PATH.exists():
        return {"regimes": {}}
    try:
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"regimes": {}}
    data.setdefault("regimes", {})
    return data


def _save(data: dict) -> None:
    with _write_lock:
        tmp = _STORE_PATH.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, _STORE_PATH)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise


def update_regime(
    code: str, as_of: str, regime_label: str, confidence: float, hmm_version: str
) -> dict:
    """
    Record today's (`as_of`, ISO date string) decoded regime for `code`,
    updating duration_days/changed_today relative to whatever was previously
    stored. Returns the updated entry.

    Idempotent per (code, as_of): re-running the same day's Shadow Mode pass
    (e.g. after a crash/retry) returns the already-stored entry unchanged
    instead of double-incrementing duration_days.
    """
    data = _load()
    prev = data["regimes"].get(code)

    if prev is not None and prev.get("as_of") == as_of:
        return prev

    if prev is None or prev.get("regime") != regime_label:
        since = as_of
        duration_days = 1
        # A stock's very first-ever recorded regime is an observation, not a
        # "change" (there's nothing to have changed from).
        changed_today = prev is not None
    else:
        since = prev.get("since", as_of)
        duration_days = prev.get("duration_days", 0) + 1
        changed_today = False

    entry = {
        "regime": regime_label,
        "confidence": confidence,
        "as_of": as_of,
        "since": since,
        "duration_days": duration_days,
        "changed_today": changed_today,
        "hmm_version": hmm_version,
    }
    data["regimes"][code] = entry
    _save(data)
    return entry


def get_regime_interface(code: str) -> dict:
    """
    The read-only Regime Interface: current_regime, regime_confidence,
    regime_duration, regime_changed_today. All-None/zero defaults if `code`
    has no recorded regime yet (e.g. Shadow Mode hasn't run for it, or its
    HMM model hasn't been trained) — callers should treat that as "no
    opinion," not as a specific regime.
    """
    entry = _load()["regimes"].get(code)
    if entry is None:
        return {
            "current_regime": None,
            "regime_confidence": None,
            "regime_duration": 0,
            "regime_changed_today": False,
        }
    return {
        "current_regime": entry["regime"],
        "regime_confidence": entry["confidence"],
        "regime_duration": entry["duration_days"],
        "regime_changed_today": entry["changed_today"],
    }


def all_regimes() -> dict:
    """Full snapshot of every stock's stored regime entry — for Shadow Mode
    reporting/analysis, not intended for per-stock decision call sites
    (use get_regime_interface(code) there)."""
    return _load()["regimes"]
