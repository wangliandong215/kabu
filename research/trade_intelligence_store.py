"""
research/trade_intelligence_store.py — V3.6-A SQLite persistence layer.

Own SQLite DB (config.TRADE_INTELLIGENCE_DB_PATH), separate from
trade_history.db, mirroring research/store.py::ObservationStore's
conventions (WAL mode, module-level write lock, upsert-merge-by-primary-key
— not a blind overwrite, since a pattern's `state` can move
OBSERVATION -> CANDIDATE_PATTERN as more data accumulates across refreshes,
but the row itself is never deleted so its history stays visible).

Three tables:
  patterns          — one row per (analysis_type, pattern_key), upserted.
  pattern_history    — append-only, one row per (pattern, refresh) snapshot,
      mirroring research/early_failure_monitor.py's signal_summary_history
      convention so promotions/regressions are visible as a time series.
  research_runs      — one row per refresh() call (run metadata + status).

`state` is restricted to OBSERVATION/CANDIDATE_PATTERN in Python
(upsert_pattern raises ValueError otherwise) — matching
research.trade_intelligence_patterns.classify_state's structural guarantee
that no other value is ever produced upstream.
"""
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

import config
from research.trade_intelligence_patterns import PatternCandidate

_DEFAULT_DB_PATH = Path(config.TRADE_INTELLIGENCE_DB_PATH)
_VALID_STATES = ("OBSERVATION", "CANDIDATE_PATTERN")

_write_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patterns (
    pattern_id              TEXT PRIMARY KEY,
    analysis_type           TEXT,
    pattern_key              TEXT,
    state                    TEXT,
    description              TEXT,
    slice_definition_json    TEXT,
    metric_name              TEXT,
    metric_value             REAL,
    comparison_metric_value  REAL,
    effect_size              REAL,
    n_sample                 INTEGER,
    n_baseline               INTEGER,
    precision_               REAL,
    recall_                  REAL,
    false_positive_count     INTEGER,
    first_observed_at        TEXT,
    last_observed_at         TEXT,
    last_refresh_run_id      TEXT,
    times_observed           INTEGER,
    source_trade_ids_json    TEXT,
    generator_version        TEXT,
    notes                    TEXT
);

CREATE TABLE IF NOT EXISTS pattern_history (
    history_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id    TEXT,
    run_id        TEXT,
    snapshot_at   TEXT,
    state         TEXT,
    metric_value  REAL,
    n_sample      INTEGER,
    precision_    REAL,
    recall_       REAL
);

CREATE TABLE IF NOT EXISTS research_runs (
    run_id                TEXT PRIMARY KEY,
    started_at            TEXT,
    finished_at           TEXT,
    source_db_path        TEXT,
    n_trades_analyzed     INTEGER,
    n_patterns_total      INTEGER,
    n_candidate_patterns  INTEGER,
    n_observations        INTEGER,
    report_path           TEXT,
    status                TEXT,
    error_message         TEXT
);
"""


class TradeIntelligenceStore:
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

    def upsert_pattern(self, candidate: PatternCandidate, run_id: str,
                        generator_version: str = "") -> None:
        if candidate.state not in _VALID_STATES:
            raise ValueError(
                f"trade_intelligence_store: invalid state {candidate.state!r} "
                f"for pattern {candidate.pattern_key!r} — must be one of {_VALID_STATES}")
        pattern_id = f"{candidate.analysis_type}:{candidate.pattern_key}"
        ts = datetime.now().isoformat()
        with _write_lock:
            existing = self._conn.execute(
                "SELECT times_observed, first_observed_at FROM patterns WHERE pattern_id=?",
                (pattern_id,),
            ).fetchone()
            times_observed = (existing["times_observed"] + 1) if existing else 1
            first_observed_at = existing["first_observed_at"] if existing else ts

            self._conn.execute(
                """INSERT INTO patterns (
                    pattern_id, analysis_type, pattern_key, state, description,
                    slice_definition_json, metric_name, metric_value,
                    comparison_metric_value, effect_size, n_sample, n_baseline,
                    precision_, recall_, false_positive_count, first_observed_at,
                    last_observed_at, last_refresh_run_id, times_observed,
                    source_trade_ids_json, generator_version, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(pattern_id) DO UPDATE SET
                    state=excluded.state,
                    description=excluded.description,
                    slice_definition_json=excluded.slice_definition_json,
                    metric_name=excluded.metric_name,
                    metric_value=excluded.metric_value,
                    comparison_metric_value=excluded.comparison_metric_value,
                    effect_size=excluded.effect_size,
                    n_sample=excluded.n_sample,
                    n_baseline=excluded.n_baseline,
                    precision_=excluded.precision_,
                    recall_=excluded.recall_,
                    false_positive_count=excluded.false_positive_count,
                    last_observed_at=excluded.last_observed_at,
                    last_refresh_run_id=excluded.last_refresh_run_id,
                    times_observed=excluded.times_observed,
                    source_trade_ids_json=excluded.source_trade_ids_json,
                    generator_version=excluded.generator_version""",
                (pattern_id, candidate.analysis_type, candidate.pattern_key,
                 candidate.state, candidate.description,
                 json.dumps(candidate.slice_definition, ensure_ascii=False),
                 candidate.metric_name, candidate.metric_value,
                 candidate.comparison_metric_value, candidate.effect_size,
                 candidate.n_sample, candidate.n_baseline, candidate.precision,
                 candidate.recall, candidate.false_positive_count,
                 first_observed_at, ts, run_id, times_observed,
                 json.dumps(candidate.source_trade_ids, ensure_ascii=False),
                 generator_version, None),
            )
            self._conn.execute(
                """INSERT INTO pattern_history (
                    pattern_id, run_id, snapshot_at, state, metric_value,
                    n_sample, precision_, recall_
                ) VALUES (?,?,?,?,?,?,?,?)""",
                (pattern_id, run_id, ts, candidate.state, candidate.metric_value,
                 candidate.n_sample, candidate.precision, candidate.recall),
            )
            self._conn.commit()

    def record_run(self, run_id: str, started_at: str, finished_at: str,
                    source_db_path: str, n_trades_analyzed: int,
                    n_patterns_total: int, n_candidate_patterns: int,
                    n_observations: int, report_path: Optional[str],
                    status: str, error_message: Optional[str] = None) -> None:
        with _write_lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO research_runs (
                    run_id, started_at, finished_at, source_db_path,
                    n_trades_analyzed, n_patterns_total, n_candidate_patterns,
                    n_observations, report_path, status, error_message
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, started_at, finished_at, source_db_path,
                 n_trades_analyzed, n_patterns_total, n_candidate_patterns,
                 n_observations, report_path, status, error_message),
            )
            self._conn.commit()

    def all_patterns(self, state: Optional[str] = None) -> pd.DataFrame:
        if state is None:
            return pd.read_sql_query("SELECT * FROM patterns", self._conn)
        return pd.read_sql_query(
            "SELECT * FROM patterns WHERE state=?", self._conn, params=(state,))

    def pattern_history(self, pattern_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM pattern_history WHERE pattern_id=? ORDER BY history_id",
            self._conn, params=(pattern_id,))
