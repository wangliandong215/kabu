"""
research/early_failure_monitor.py — long-term EARLY_FAILURE signal monitor
(2026-09 research-only investigation, see engine/exit_diagnostics.py and
research/early_failure_trajectory.py).

Pure research tooling, deliberately kept OUT of engine/ and NOT wired into
engine/runner.py's live trading pass — same "observation only" boundary as
research/store.py's Mean Reversion observation DB. This module answers one
question repeatedly as more real closed trades accumulate: does the
MFE_day2<1% candidate signal (found 2026-09, currently 18/18 precision on a
35-trade sample) keep holding up, or does it break once the tiny 8-trade
winner comparison group grows?

Writes to a SEPARATE SQLite DB (C:\\KabuData\\research\\early_failure_monitor.db)
— never to trade_history.db — for the same reason research/store.py uses a
separate DB from trade_history.db: this data has nothing to do with a live
trade's lifecycle and must never get coupled to it.

Two tables:
  trade_trajectory        — one row per closed trade (overwritten on every
      refresh() call — this is a derived/recomputed view, not raw
      observations, so INSERT OR REPLACE is correct here unlike
      research/store.py's upsert-merge convention).
  signal_summary_history  — one row PER refresh() CALL (append-only,
      timestamped) — the precision/recall/n trend for the three candidate
      signals over time, so "has 0 false positives held up" etc. can be
      read directly off a time series instead of re-derived by hand.

NOT wired into any automatic trigger. Per the 2026-09 investigation's
explicit research-only boundary, this file does not touch engine/runner.py
— refresh() must be invoked manually (or by an external scheduler you set
up yourself, e.g. Windows Task Scheduler / a cron-like wrapper) after new
trades close. Wiring an automatic call into the live trading pass would
mean editing engine/runner.py again, which this round of work was
explicitly told to avoid.

Usage:
    python -m research.early_failure_monitor            # refresh + print
    python -m research.early_failure_monitor --history   # print signal_summary_history
"""
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from research.early_failure_trajectory import build_trajectory_table

_DB_PATH = Path(r"C:\KabuData\research\early_failure_monitor.db")
_write_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_trajectory (
    trade_id              TEXT PRIMARY KEY,
    ticker                TEXT,
    entry_time            TEXT,
    exit_time             TEXT,
    holding_days          REAL,
    exit_category         TEXT,
    n_daily_rows          INTEGER,
    mfe_day1              REAL,
    mfe_day2              REAL,
    mfe_day3              REAL,
    mfe_day5              REAL,
    mae_day3              REAL,
    mae_day5              REAL,
    mfe_day2_lt_1pct      INTEGER,
    mfe_day2_lt_2pct      INTEGER,
    mae_day5_lte_neg4pct  INTEGER,
    entry_rank            INTEGER,
    sector                TEXT,
    recorded_at           TEXT
);

CREATE TABLE IF NOT EXISTS signal_summary_history (
    snapshot_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at       TEXT,
    n_total           INTEGER,
    n_early_failure   INTEGER,
    n_winner          INTEGER,
    n_trend_reversal  INTEGER,
    sig_mfe_d2_lt1_n_flagged      INTEGER,
    sig_mfe_d2_lt1_precision      REAL,
    sig_mfe_d2_lt1_recall         REAL,
    sig_mfe_d2_lt1_false_positives INTEGER,
    sig_mfe_d2_lt2_n_flagged      INTEGER,
    sig_mfe_d2_lt2_precision      REAL,
    sig_mfe_d2_lt2_recall         REAL,
    sig_mfe_d2_lt2_false_positives INTEGER,
    sig_mae_d5_lte4_n_flagged      INTEGER,
    sig_mae_d5_lte4_precision      REAL,
    sig_mae_d5_lte4_recall         REAL,
    sig_mae_d5_lte4_false_positives INTEGER
);
"""

# Candidate signal thresholds under long-term tracking — see the 2026-09
# EARLY_FAILURE investigation's final read: MFE_day2<1% is the standout
# (100% precision / 75% recall / 0 false positives on the first 35-trade
# sample), <2% and MAE_day5<=-4% kept as secondary trackers. Deliberately
# not "the best parameters" — these are the exact three the investigation
# ended on, kept fixed so the trend across refreshes is apples-to-apples.
_MFE_D2_TIGHT = 0.01
_MFE_D2_LOOSE = 0.02
_MAE_D5_BAR = -0.04


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    with _write_lock:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
    return conn


def _signal_stats(df: pd.DataFrame, flag_col: str) -> dict:
    """precision/recall/false-positive-count for one boolean flag column
    against exit_category, restricted to the three well-defined outcome
    categories (EARLY_FAILURE / winner / TREND_REVERSAL) — matches the
    2026-09 report's own accounting exactly."""
    flagged = df[df[flag_col] == 1]
    n_flagged = len(flagged)
    tp = int((flagged["exit_category"] == "EARLY_FAILURE").sum())
    fp = int(flagged["is_winner"].sum())
    n_ef_total = int((df["exit_category"] == "EARLY_FAILURE").sum())
    precision = (tp / n_flagged) if n_flagged else None
    recall = (tp / n_ef_total) if n_ef_total else None
    return dict(n_flagged=n_flagged, precision=precision, recall=recall, false_positives=fp)


def refresh(db_path: Optional[Path] = None,
            source_db_path: Optional[Path] = None) -> pd.DataFrame:
    """Recompute the trajectory table from trade_history.db (via
    research.early_failure_trajectory.build_trajectory_table), persist it
    (overwrite) to trade_trajectory, append one row to
    signal_summary_history, and return the per-trade DataFrame with the
    flag columns attached.

    source_db_path lets tests point at a temp trade_history.db instead of
    the production one — see research/test_early_failure_monitor.py."""
    df = build_trajectory_table(source_db_path)
    if df.empty:
        return df
    df = df[df["exit_category"].notna()].copy()
    df["is_winner"] = df["exit_category"].isin(["PROFIT_GIVEBACK", "EXTREME_GIVEBACK"])
    df["mfe_day2_lt_1pct"] = (df["MFE_at_day_2"] < _MFE_D2_TIGHT).astype(int)
    df["mfe_day2_lt_2pct"] = (df["MFE_at_day_2"] < _MFE_D2_LOOSE).astype(int)
    df["mae_day5_lte_neg4pct"] = (df["MAE_at_day_5"] <= _MAE_D5_BAR).astype(int)

    ts = datetime.now().isoformat()
    conn = _connect(db_path)
    with _write_lock:
        for _, r in df.iterrows():
            conn.execute(
                """INSERT OR REPLACE INTO trade_trajectory (
                    trade_id, ticker, entry_time, exit_time, holding_days,
                    exit_category, n_daily_rows, mfe_day1, mfe_day2, mfe_day3,
                    mfe_day5, mae_day3, mae_day5, mfe_day2_lt_1pct,
                    mfe_day2_lt_2pct, mae_day5_lte_neg4pct, entry_rank,
                    sector, recorded_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r["trade_id"], r["ticker"], r["entry_time"], r["exit_time"],
                 r["holding_days"], r["exit_category"], r["n_daily_rows"],
                 r["MFE_at_day_1"], r["MFE_at_day_2"], r["MFE_at_day_3"],
                 r["MFE_at_day_5"], r["MAE_at_day_3"], r["MAE_at_day_5"],
                 int(r["mfe_day2_lt_1pct"]), int(r["mfe_day2_lt_2pct"]),
                 int(r["mae_day5_lte_neg4pct"]), r["entry_rank"], r["sector"], ts),
            )

        s1 = _signal_stats(df, "mfe_day2_lt_1pct")
        s2 = _signal_stats(df, "mfe_day2_lt_2pct")
        s3 = _signal_stats(df, "mae_day5_lte_neg4pct")
        conn.execute(
            """INSERT INTO signal_summary_history (
                snapshot_at, n_total, n_early_failure, n_winner, n_trend_reversal,
                sig_mfe_d2_lt1_n_flagged, sig_mfe_d2_lt1_precision,
                sig_mfe_d2_lt1_recall, sig_mfe_d2_lt1_false_positives,
                sig_mfe_d2_lt2_n_flagged, sig_mfe_d2_lt2_precision,
                sig_mfe_d2_lt2_recall, sig_mfe_d2_lt2_false_positives,
                sig_mae_d5_lte4_n_flagged, sig_mae_d5_lte4_precision,
                sig_mae_d5_lte4_recall, sig_mae_d5_lte4_false_positives
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ts, len(df), int((df["exit_category"] == "EARLY_FAILURE").sum()),
             int(df["is_winner"].sum()),
             int((df["exit_category"] == "TREND_REVERSAL").sum()),
             s1["n_flagged"], s1["precision"], s1["recall"], s1["false_positives"],
             s2["n_flagged"], s2["precision"], s2["recall"], s2["false_positives"],
             s3["n_flagged"], s3["precision"], s3["recall"], s3["false_positives"]),
        )
        conn.commit()
    conn.close()
    return df


def load_history(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Every past refresh() snapshot, oldest first — the time series
    section A/B/C of the 2026-09 report asked to track."""
    conn = _connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM signal_summary_history ORDER BY snapshot_id", conn)
    conn.close()
    return df


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)

    if "--history" in sys.argv:
        print(load_history().to_string())
    else:
        df = refresh()
        print(f"refreshed trade_trajectory: {len(df)} rows -> {_DB_PATH}")
        latest = load_history().iloc[-1]
        print("\nlatest snapshot:")
        print(latest.to_string())
