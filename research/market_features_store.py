"""
research/market_features_store.py — SQLite persistence for Priority 3/4
research-only market feature snapshots (news/macro/FedWatch/options), the
Priority 4 indicator shadow-validation log, and (v2.11.1) the collection
health log + the Look-ahead-safe time model.

Same conventions as research/store.py (WAL mode, module-level write lock,
INSERT OR REPLACE upsert per table) — kept as a SEPARATE database file from
mean_reversion_observations.db since these rows aren't trade-case
observations, they're daily environment snapshots on an unrelated schedule.

Every table here is observation-only: nothing in this module is imported by
engine/runner.py, strategies/*, risk/*, or portfolio/*.

── Time model (v2.11.1) ─────────────────────────────────────────────────────
Every feature table carries two NEW columns on top of the original
`created_at`/`snapshot_date`:
  observed_at  — UTC ISO 8601, the moment the collector actually fetched and
                 wrote this row (common.utc_now_iso()). Use this, not
                 created_at, for any "was this known by time X" comparison —
                 see latest_before() below.
  market_date  — the US trading-calendar date (engine.market_hours.
                 market_date_for()) this row's data should be attributed to.
`created_at`/`snapshot_date` are kept, unrenamed, for backward compatibility
and debugging — but `snapshot_date` is just the JST calendar day the
collector happened to run on, which is OFF BY ONE from the correct
market_date on every normal day: the daily collection runs at 07:15 JST,
which is in the gap AFTER the previous US session's close and BEFORE that
evening's next session opens, so "today's JST date" is not the trading day
the data reflects — market_date is. New code should always read
market_date, never snapshot_date, for anything date-sensitive.

`macro_snapshots` has an extra wrinkle: its natural key is
(region, indicator_id, data_time) where data_time is the ECONOMIC PERIOD an
indicator covers (e.g. "2026-07-01" for July CPI), not the day it was
fetched. Government statistics get revised weeks/months later — re-fetching
the same data_time after a revision must NOT silently overwrite the
originally-observed value (that would be exactly the look-ahead risk this
whole module exists to prevent), so its primary key includes `observed_at`,
making every fetch of the same economic period its own permanent row.
"""
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pytz

import common
import config
import engine.market_hours as market_hours

_DEFAULT_DB_PATH = Path(config.RESEARCH_MARKET_FEATURES_DB_PATH)
_JST = pytz.timezone("Asia/Tokyo")

_write_lock = threading.Lock()

_TIME_COLUMNS = ["observed_at", "market_date"]

_TABLES = {
    "news_snapshots": {
        "key": ("code", "snapshot_date"),
        "columns": ["code", "snapshot_date", "news_count_24h",
                    "major_news_flag", "announcement_flag", "created_at",
                    *_TIME_COLUMNS],
        "int_columns": {"news_count_24h", "major_news_flag", "announcement_flag"},
    },
    "macro_snapshots": {
        # observed_at is part of the key — see module docstring on why
        # revised macro data must never overwrite an earlier vintage.
        "key": ("region", "indicator_id", "data_time", "observed_at"),
        "columns": ["region", "indicator_id", "category_name", "name",
                    "data_time", "release_time", "value", "predict_value",
                    "previous_value", "unit_type", "snapshot_date", "created_at",
                    *_TIME_COLUMNS],
        "int_columns": set(),
    },
    "fedwatch_target_rate_snapshots": {
        "key": ("snapshot_date", "meeting_date"),
        "columns": ["snapshot_date", "meeting_date", "target_range",
                    "probability", "created_at", *_TIME_COLUMNS],
        "int_columns": set(),
    },
    "fedwatch_dot_plot_snapshots": {
        "key": ("snapshot_date", "year", "rate"),
        "columns": ["snapshot_date", "year", "rate", "vote_count",
                    "is_median", "median_rate", "current_rate", "created_at",
                    *_TIME_COLUMNS],
        "int_columns": {"is_median"},
    },
    "options_snapshots": {
        "key": ("code", "snapshot_date"),
        "columns": ["code", "snapshot_date", "iv", "iv_rank", "iv_percentile",
                    "hv_30d", "call_volume", "put_volume", "put_call_ratio",
                    "created_at", *_TIME_COLUMNS],
        "int_columns": {"call_volume", "put_volume"},
    },
    "market_pcr_snapshots": {
        "key": ("market", "time"),
        "columns": ["market", "time", "timestamp", "call_value", "put_value",
                    "total_value", "ratio", "snapshot_date", "created_at",
                    *_TIME_COLUMNS],
        "int_columns": {"call_value", "put_value", "total_value"},
    },
    "indicator_shadow_log": {
        "key": ("code", "snapshot_date", "indicator"),
        "columns": ["code", "snapshot_date", "indicator", "our_bucket",
                    "moomoo_bucket", "match", "created_at", *_TIME_COLUMNS],
        "int_columns": {"match"},
    },
    # v2.11.1 Priority 2 — append-only collection health log. NOT upserted
    # via upsert()/_TABLES-driven schema (no natural business key to dedupe
    # on — every collection attempt is its own row) — has its own CREATE
    # TABLE + insert method below.
}

_COLLECTION_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS collection_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    market_date TEXT,
    category    TEXT,
    target      TEXT,
    status      TEXT,
    detail      TEXT,
    observed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_collection_log_market_date
    ON collection_log(market_date, category);
"""


def _column_def(table: str, name: str) -> str:
    spec = _TABLES[table]
    if name in spec["int_columns"]:
        return f"{name} INTEGER"
    if name in ("value", "predict_value", "previous_value", "rate", "vote_count",
                "median_rate", "current_rate", "probability", "iv", "iv_rank",
                "iv_percentile", "hv_30d", "put_call_ratio", "ratio", "timestamp",
                "year"):
        return f"{name} REAL"
    return f"{name} TEXT"


def _build_schema() -> str:
    stmts = [_COLLECTION_LOG_SCHEMA]
    for table, spec in _TABLES.items():
        cols_sql = ", ".join(_column_def(table, c) for c in spec["columns"])
        key_sql = ", ".join(spec["key"])
        stmts.append(
            f"CREATE TABLE IF NOT EXISTS {table} ({cols_sql}, "
            f"PRIMARY KEY ({key_sql}));"
        )
    return "\n".join(stmts)


def _utc_from_naive_jst(naive_iso: Optional[str]) -> Optional[str]:
    """Best-effort: a pre-migration created_at was always datetime.now().
    isoformat() on a server whose OS timezone is JST (confirmed 2026-08-30 —
    see common.utc_now_iso()'s docstring) — localize it as JST and convert
    to UTC. Returns None if `naive_iso` is missing/unparseable rather than
    raising, since this only ever runs inside a best-effort backfill."""
    if not naive_iso:
        return None
    try:
        naive = datetime.fromisoformat(naive_iso)
        return _JST.localize(naive).astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError):
        return None


def _market_date_from_utc_iso(utc_iso: Optional[str]) -> Optional[str]:
    if not utc_iso:
        return None
    try:
        utc_dt = datetime.fromisoformat(utc_iso)
        jst_dt = utc_dt.astimezone(_JST)
        return market_hours.market_date_for("US", now=jst_dt).isoformat()
    except Exception:
        return None


class MarketFeaturesStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.path = Path(db_path or _DEFAULT_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with _write_lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._migrate_macro_snapshots_pk()
            self._conn.executescript(_build_schema())
            self._migrate_missing_columns()
            self._backfill_missing_time_fields()
            self._conn.commit()

    def _migrate_macro_snapshots_pk(self) -> None:
        """macro_snapshots' primary key gained `observed_at` in v2.11.1 (see
        module docstring). SQLite can't ALTER a primary key in place, so a
        table created under the old (region, indicator_id, data_time) key
        must be rebuilt: rename aside, recreate under the new schema, copy
        rows across (backfilling observed_at/market_date the same way
        _backfill_missing_time_fields() does for every other table), drop
        the old copy. No-ops on a fresh DB (table doesn't exist yet —
        _build_schema() right after this call creates it with the new key
        directly) and on an already-migrated DB (new key already present)."""
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='macro_snapshots'"
        ).fetchone()
        if row is None or "observed_at" in row["sql"]:
            return

        self._conn.execute("ALTER TABLE macro_snapshots RENAME TO macro_snapshots_old_pk")
        spec = _TABLES["macro_snapshots"]
        cols_sql = ", ".join(_column_def("macro_snapshots", c) for c in spec["columns"])
        key_sql = ", ".join(spec["key"])
        self._conn.execute(
            f"CREATE TABLE macro_snapshots ({cols_sql}, PRIMARY KEY ({key_sql}))")

        old_rows = self._conn.execute("SELECT * FROM macro_snapshots_old_pk").fetchall()
        for old in old_rows:
            old = dict(old)
            observed_at = old.get("observed_at") or _utc_from_naive_jst(old.get("created_at"))
            market_date = old.get("market_date") or _market_date_from_utc_iso(observed_at)
            merged = {**old, "observed_at": observed_at, "market_date": market_date}
            placeholders = ", ".join("?" for _ in spec["columns"])
            self._conn.execute(
                f"INSERT OR REPLACE INTO macro_snapshots ({', '.join(spec['columns'])}) "
                f"VALUES ({placeholders})",
                tuple(merged.get(c) for c in spec["columns"]),
            )
        self._conn.execute("DROP TABLE macro_snapshots_old_pk")

    def _migrate_missing_columns(self) -> None:
        """CREATE TABLE IF NOT EXISTS never widens an existing table — heal
        pre-v2.11.1 DBs with ALTER TABLE ADD COLUMN (same pattern as
        engine/trade_tracker.py's _migrate_missing_columns()). macro_snapshots
        already has both columns by the time this runs (via the PK rebuild
        above, or because it's a fresh table), so this is a no-op for it."""
        for table in _TABLES:
            existing = {r["name"] for r in
                        self._conn.execute(f"PRAGMA table_info({table})")}
            for col in _TIME_COLUMNS:
                if col not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")

    def _backfill_missing_time_fields(self) -> None:
        """One-time best-effort backfill for rows written before v2.11.1:
        derives observed_at from the legacy created_at (JST-naive) and
        market_date from that. Leaves both NULL if created_at is itself
        unparseable rather than guessing. macro_snapshots rows are already
        fully populated by the PK-rebuild step above, so this loop finds
        nothing left to do for that table."""
        for table, spec in _TABLES.items():
            rows = self._conn.execute(
                f"SELECT * FROM {table} WHERE observed_at IS NULL"
            ).fetchall()
            for row in rows:
                row = dict(row)
                observed_at = _utc_from_naive_jst(row.get("created_at"))
                if observed_at is None:
                    continue
                market_date = _market_date_from_utc_iso(observed_at)
                where_clause = " AND ".join(f"{k}=?" for k in spec["key"])
                self._conn.execute(
                    f"UPDATE {table} SET observed_at=?, market_date=? WHERE {where_clause}",
                    (observed_at, market_date, *[row[k] for k in spec["key"]]),
                )

    def close(self) -> None:
        self._conn.close()

    def upsert(self, table: str, record: dict) -> None:
        """Insert or replace one row in `table`. `record` must include all
        of that table's primary-key columns (see _TABLES[table]['key'])."""
        spec = _TABLES[table]
        columns = spec["columns"]
        with _write_lock:
            placeholders = ", ".join("?" for _ in columns)
            self._conn.execute(
                f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) "
                f"VALUES ({placeholders})",
                tuple(record.get(c) for c in columns),
            )
            self._conn.commit()

    def upsert_many(self, table: str, records: List[dict]) -> None:
        for record in records:
            self.upsert(table, record)

    def all_rows(self, table: str) -> list:
        rows = self._conn.execute(f"SELECT * FROM {table}").fetchall()
        return [dict(r) for r in rows]

    def latest_before(self, table: str, filters: Dict[str, object],
                       as_of_utc_iso: str) -> Optional[dict]:
        """The most recent row in `table` matching `filters` (exact-match
        equality) whose observed_at <= as_of_utc_iso — the Look-ahead-safe
        read path: "what did we already know as of this moment", never "what
        do we know now". Returns None if nothing qualifies (never falls back
        to a later row). ISO 8601 UTC strings compare correctly as plain
        text since they're fixed-width and zero-padded, so this doesn't need
        to parse dates to order them."""
        where_parts = ["observed_at IS NOT NULL", "observed_at <= ?"]
        params: list = [as_of_utc_iso]
        for k, v in filters.items():
            where_parts.append(f"{k} = ?")
            params.append(v)
        query = (
            f"SELECT * FROM {table} WHERE {' AND '.join(where_parts)} "
            f"ORDER BY observed_at DESC LIMIT 1"
        )
        row = self._conn.execute(query, tuple(params)).fetchone()
        return dict(row) if row is not None else None

    def log_collection(self, market_date: str, category: str, target: str,
                        status: str, detail: str = "",
                        observed_at: Optional[str] = None) -> None:
        """Append one collection-attempt outcome row (v2.11.1 Priority 2
        Health Check). Never upserted/deduped — every attempt, including
        repeats within the same market_date, gets its own row; the health
        check aggregates over all rows for a market_date+category."""
        with _write_lock:
            self._conn.execute(
                """INSERT INTO collection_log
                   (market_date, category, target, status, detail, observed_at)
                   VALUES (?,?,?,?,?,?)""",
                (market_date, category, target, status, detail,
                 observed_at or common.utc_now_iso()),
            )
            self._conn.commit()

    def collection_log_rows(self, market_date: str,
                             category: Optional[str] = None) -> list:
        if category is None:
            rows = self._conn.execute(
                "SELECT * FROM collection_log WHERE market_date=?", (market_date,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM collection_log WHERE market_date=? AND category=?",
                (market_date, category),
            ).fetchall()
        return [dict(r) for r in rows]
