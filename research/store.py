"""
research/store.py — SQLite persistence for Mean Reversion observation
events. Same conventions as engine/market_context.py's MarketContextStore /
engine/trade_tracker.py (WAL mode, module-level write lock, INSERT OR
REPLACE upsert). Deliberately a SEPARATE database file from
trade_history.db — this table's rows mostly have nothing to do with an
actual bot trade (case B/C/D candidates are never bought), so it doesn't
belong coupled to the trade lifecycle DB.

upsert() merges into any existing row rather than blindly replacing it,
because one event_id gets written in phases over time: the daily scan writes
categories 1-8 at detection, a later daily backfill pass fills in category 9
(forward returns) once enough time has passed, and research/annotate_event.py
may add manual-labeling fields at any point — none of those phases should
ever null out fields a previous phase already filled in.
"""
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

import config

_DEFAULT_DB_PATH = Path(config.MR_DB_PATH)

_write_lock = threading.Lock()

_COLUMNS = [
    "event_id", "code", "event_date", "detected_at", "trigger_reason",
    # Category 1: price extremity
    "return_1d", "return_3d", "return_5d", "return_10d",
    "dist_from_20d_high", "dist_from_60d_high", "atr",
    "daily_return_over_atr", "gap_down_pct",
    # Category 2: oversold
    "rsi14", "rsi5", "stoch_k", "stoch_d", "williams_r",
    "dist_from_ma20", "dist_from_ma50",
    # Category 3: volume / panic
    "volume", "volume_ratio_20d", "volume_zscore", "down_day_volume",
    "volume_spike",
    # Category 4: candle stabilization
    "low_to_close_position", "close_location_value", "lower_wick_pct",
    "body_pct", "is_hammer", "recovered_prior_low",
    "made_new_low_t1", "made_new_low_t2", "higher_low_within_3d",
    # Category 5: HMM regime
    "hmm_regime", "hmm_confidence", "hmm_duration_days", "hmm_changed_today",
    "hmm_prev_regime", "transition_bear_to_sideways", "transition_bear_to_bull",
    # Category 6: market environment
    "spy_return_1d", "spy_return_5d", "qqq_return_1d", "qqq_return_5d",
    "vix_close", "sector", "sector_etf", "sector_etf_return_1d",
    "sector_etf_return_5d", "stock_vs_sector_return_5d",
    # Category 7: fundamentals
    "market_cap", "pe_ttm", "revenue_growth_yoy", "eps_growth_yoy", "roe",
    "profit_margin", "gross_margin", "debt_ratio", "roic",
    "free_cash_flow", "total_debt", "earnings_estimate_revision",
    "fundamentals_status",
    # Category 8: crash reason
    "crash_reason", "crash_reason_source",
    # Category 9: forward returns
    "return_fwd_t1", "return_fwd_t3", "return_fwd_t5", "return_fwd_t10",
    "return_fwd_t20", "return_fwd_t60", "max_favorable_excursion",
    "max_adverse_excursion", "forward_bars_available", "outcome_label",
    # Manual labeling / case grouping
    "was_manually_bought", "manual_entry_price", "manual_entry_date",
    "manual_note",
    # Bookkeeping
    "created_at", "updated_at",
]

_TEXT_COLUMNS = {
    "code", "event_date", "detected_at", "trigger_reason", "hmm_regime",
    "hmm_prev_regime", "sector", "sector_etf", "fundamentals_status",
    "crash_reason", "crash_reason_source", "outcome_label",
    "manual_entry_date", "manual_note", "created_at", "updated_at",
}
_INTEGER_COLUMNS = {
    "volume_spike", "is_hammer", "recovered_prior_low",
    "made_new_low_t1", "made_new_low_t2", "higher_low_within_3d",
    "hmm_changed_today", "transition_bear_to_sideways",
    "transition_bear_to_bull", "was_manually_bought",
    "forward_bars_available",
}


def _column_def(name: str) -> str:
    if name == "event_id":
        return "event_id TEXT PRIMARY KEY"
    if name in _TEXT_COLUMNS:
        return f"{name} TEXT"
    if name in _INTEGER_COLUMNS:
        return f"{name} INTEGER"
    return f"{name} REAL"


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS mean_reversion_events (
    {", ".join(_column_def(c) for c in _COLUMNS)}
);
CREATE INDEX IF NOT EXISTS idx_mre_code ON mean_reversion_events(code);
CREATE INDEX IF NOT EXISTS idx_mre_event_date ON mean_reversion_events(event_date);
"""


class ObservationStore:
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

    def get(self, event_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM mean_reversion_events WHERE event_id=?", (event_id,)
        ).fetchone()
        return self._deserialize(dict(row)) if row is not None else None

    def find_latest_event_date(self, code: str) -> Optional[str]:
        """Most recent event_date already recorded for `code`, for the
        caller (scan script) to compute a trading-day cooldown gap against
        the OHLCV df it already has in hand."""
        row = self._conn.execute(
            "SELECT event_date FROM mean_reversion_events WHERE code=? "
            "ORDER BY event_date DESC LIMIT 1", (code,)
        ).fetchone()
        return row["event_date"] if row is not None else None

    def find_near(self, code: str, date_iso: str, within_days: int = 3) -> Optional[dict]:
        """Closest existing row for `code` within `within_days` calendar
        days of `date_iso` — used by research/annotate_event.py to decide
        whether a manual label should update an already auto-detected event
        or create a fresh one."""
        target = pd.Timestamp(date_iso)
        best = None
        best_gap = None
        for row in self._conn.execute(
            "SELECT * FROM mean_reversion_events WHERE code=?", (code,)
        ).fetchall():
            gap = abs((pd.Timestamp(row["event_date"]) - target).days)
            if gap <= within_days and (best_gap is None or gap < best_gap):
                best, best_gap = dict(row), gap
        return self._deserialize(best) if best is not None else None

    def rows_needing_backfill(self) -> list:
        rows = self._conn.execute(
            "SELECT * FROM mean_reversion_events WHERE "
            "forward_bars_available IS NULL OR forward_bars_available < 60"
        ).fetchall()
        return [self._deserialize(dict(r)) for r in rows]

    def all_rows(self) -> list:
        rows = self._conn.execute(
            "SELECT * FROM mean_reversion_events ORDER BY event_date"
        ).fetchall()
        return [self._deserialize(dict(r)) for r in rows]

    @staticmethod
    def _deserialize(row: dict) -> dict:
        if row.get("fundamentals_status"):
            try:
                row["fundamentals_status"] = json.loads(row["fundamentals_status"])
            except (TypeError, ValueError):
                pass
        return row

    def upsert(self, record: dict) -> None:
        """Merge `record` into any existing row with the same event_id
        (existing fields not present in `record` are preserved — see module
        docstring). `record` must include event_id, code, event_date."""
        if "fundamentals_status" in record and isinstance(record["fundamentals_status"], dict):
            record = dict(record)
            record["fundamentals_status"] = json.dumps(
                record["fundamentals_status"], ensure_ascii=False)

        now_iso = datetime.now().isoformat()
        with _write_lock:
            existing = self._conn.execute(
                "SELECT * FROM mean_reversion_events WHERE event_id=?",
                (record["event_id"],),
            ).fetchone()
            merged = dict(existing) if existing is not None else {}
            merged.update({k: v for k, v in record.items() if k in _COLUMNS})
            merged["created_at"] = merged.get("created_at") or now_iso
            merged["updated_at"] = now_iso

            placeholders = ", ".join("?" for _ in _COLUMNS)
            self._conn.execute(
                f"INSERT OR REPLACE INTO mean_reversion_events "
                f"({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                tuple(merged.get(c) for c in _COLUMNS),
            )
            self._conn.commit()
