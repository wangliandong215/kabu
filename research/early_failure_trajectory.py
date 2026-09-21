"""
research/early_failure_trajectory.py — Early Failure trajectory metrics
(2026-09 research-only investigation, see engine/exit_diagnostics.py).

Pure research tooling: reads trade_history.db (trades / trade_daily /
trade_exit_diagnostics), computes a set of day-indexed trajectory metrics
per closed trade, and returns/prints/exports them. Nothing here writes back
to trade_history.db, touches engine/*.py, or feeds any trading decision —
it exists purely so the "does EARLY_FAILURE have a stable, identifiable
post-entry trajectory in the first 3-5 days" question can be re-answered
as more samples accumulate, without hand-recomputing the day-by-day walk
every time (see the 2026-09 EARLY_FAILURE giveback-vs-failure investigation
this was built for).

Definitions (all computed from trade_daily, one row per trade per calendar
date the position was open; day_idx=1 is the first day a trade_daily row
exists, NOT the entry day itself — update_position_metrics() is called once
per live scan pass starting the pass after entry):

  running_mfe_pct(day N) : cummax of that day's OWN mfe (favorable excursion
      using that day's high, per trade_daily.mfe) over days 1..N, divided
      by position_value. This is "the best the trade had shown by day N",
      not day N's own high in isolation.
  running_mae_pct(day N) : cummin of trade_daily.mae over days 1..N,
      divided by position_value — "the worst the trade had shown by day N".
  days_since_last_high   : as of the trade's LAST logged day, how many days
      since running_mfe_pct last increased (0 = still making new highs on
      the final day).
  days_since_last_positive_close : as of the last logged day, how many days
      since floating_pnl_pct was last > 0 (None if never positive).
  days_to_first_MFE_Xpct : first day_idx where running_mfe_pct >= X%
      (None if never reached within the trade's logged days).
  consecutive_days_without_new_high (at exit) : count of trailing days
      (ending at the last logged day) where running_mfe_pct did NOT exceed
      its own running_mfe_pct as of the previous day.
  consecutive_weak_close_days (at exit) : count of trailing days where
      floating_pnl_pct stayed below a small "weak" bar (default 0.5%).

Usage:
    python -m research.early_failure_trajectory                # prints + exports CSV
    python -m research.early_failure_trajectory --min-day3      # section 2 hit-rate tables
"""
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

_DB_PATH = Path(r"C:\KabuData\trade_history\trade_history.db")
_WEAK_CLOSE_THRESHOLD = 0.005   # 0.5% — "barely positive/negative" bar for consecutive_weak_close_days


def _load(db_path: Optional[Path] = None):
    con = sqlite3.connect(str(db_path or _DB_PATH))
    trades = pd.read_sql_query("SELECT * FROM trades", con)
    tdaily = pd.read_sql_query("SELECT * FROM trade_daily", con)
    try:
        diag = pd.read_sql_query("SELECT trade_id, exit_category, mfe_pct, giveback_pct "
                                  "FROM trade_exit_diagnostics", con)
    except Exception:
        diag = pd.DataFrame(columns=["trade_id", "exit_category", "mfe_pct", "giveback_pct"])
    con.close()
    return trades, tdaily, diag


def build_trajectory_table(db_path: Optional[Path] = None) -> pd.DataFrame:
    """One row per closed real trade (excludes the $50k-equity_before phantom
    rows the same way every other 2026-09 analysis in this investigation
    did), with the trajectory metrics described in the module docstring."""
    trades, tdaily, diag = _load(db_path)
    closed = trades[(trades["equity_before"] != 50000.0) & (trades["exit_time"].notna())].copy()

    rows = []
    for _, t in closed.iterrows():
        tid = t["trade_id"]
        pos_val = t["position_value"]
        sub = tdaily[tdaily["trade_id"] == tid].sort_values("date").reset_index(drop=True)
        n = len(sub)
        if n == 0 or not pos_val:
            continue
        sub["day_idx"] = range(1, n + 1)
        sub["running_mfe_pct"] = sub["mfe"].cummax() / pos_val
        sub["running_mae_pct"] = sub["mae"].cummin() / pos_val

        def _at_day(col, day):
            hit = sub[sub["day_idx"] == day]
            if len(hit):
                return float(hit.iloc[0][col])
            # right-censored: trade closed before this day existed
            return float(sub.iloc[-1][col]) if day > n else None

        # days_to_first_MFE_Xpct
        def _days_to_mfe(pct):
            hit = sub[sub["running_mfe_pct"] >= pct]
            return int(hit.iloc[0]["day_idx"]) if len(hit) else None

        # days_since_last_high (as of the trade's last logged day). Day 1
        # always counts as "setting" the first high (nothing to compare
        # against yet), so diff().iloc[0]'s NaN is overwritten to True.
        new_high = sub["running_mfe_pct"].diff() > 0
        new_high.iloc[0] = True
        last_high_day = sub.loc[new_high, "day_idx"].max()
        days_since_last_high = int(n - last_high_day)

        # consecutive_days_without_new_high, trailing from the last day
        consec_no_high = 0
        for i in range(n - 1, -1, -1):
            if new_high.iloc[i]:
                break
            consec_no_high += 1

        # days_since_last_positive_close
        pos_close = sub["floating_pnl_pct"] > 0
        if pos_close.any():
            last_pos_day = sub.loc[pos_close, "day_idx"].max()
            days_since_last_positive = int(n - last_pos_day)
        else:
            days_since_last_positive = None

        # consecutive_weak_close_days, trailing from the last day
        weak = sub["floating_pnl_pct"] < _WEAK_CLOSE_THRESHOLD
        consec_weak = 0
        for i in range(n - 1, -1, -1):
            if not weak.iloc[i]:
                break
            consec_weak += 1

        rows.append({
            "trade_id": tid, "ticker": t["ticker"],
            "entry_time": t["entry_time"], "exit_time": t["exit_time"],
            "holding_days": t["holding_days"], "pnl_pct": t["pnl_pct"],
            "exit_reason_code": t["exit_reason_code"],
            "entry_rank": t["entry_rank"], "confidence_score": t["confidence_score"],
            "sector": t["sector"], "n_daily_rows": n,
            "max_MFE_first_2_days": float(sub[sub["day_idx"] <= 2]["running_mfe_pct"].max())
                if n >= 1 else None,
            "MFE_at_day_1": _at_day("running_mfe_pct", 1),
            "MFE_at_day_2": _at_day("running_mfe_pct", 2),
            "MFE_at_day_3": _at_day("running_mfe_pct", 3),
            "MFE_at_day_5": _at_day("running_mfe_pct", 5),
            "MAE_at_day_3": _at_day("running_mae_pct", 3),
            "MAE_at_day_5": _at_day("running_mae_pct", 5),
            "final_MFE_pct": float(sub["running_mfe_pct"].iloc[-1]),
            "final_MAE_pct": float(sub["running_mae_pct"].iloc[-1]),
            "days_to_first_MFE_1pct": _days_to_mfe(0.01),
            "days_to_first_MFE_2pct": _days_to_mfe(0.02),
            "days_since_last_high": days_since_last_high,
            "consecutive_days_without_new_high": consec_no_high,
            "days_since_last_positive_close": days_since_last_positive,
            "consecutive_weak_close_days": consec_weak,
        })

    out = pd.DataFrame(rows)
    if len(diag):
        out = out.merge(diag, on="trade_id", how="left")
    return out


if __name__ == "__main__":
    import sys
    df = build_trajectory_table()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", None)
    print(df.to_string())
    out_path = Path("early_failure_trajectory_export.csv")
    df.to_csv(out_path, index=False)
    print(f"\nexported {len(df)} rows -> {out_path}")
