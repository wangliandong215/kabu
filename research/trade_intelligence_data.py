"""
research/trade_intelligence_data.py — V3.6-A Trade Intelligence data access
layer (research-only, see research/trade_intelligence_runner.py's module
docstring for the full V3.6-A boundary).

Read-only. Joins trades + trade_attribution + trade_exit_diagnostics +
trade_confidence_score + trade_entry_quality from trade_history.db via a
plain sqlite3.connect() — same precedent as
research/early_failure_trajectory.py::_load(), deliberately NOT
engine.trade_tracker.TradeTracker, to avoid taking its write lock just to
read and to do one multi-table LEFT JOIN instead of four separate
TradeTracker.query_*() round trips + pandas merges.

load_dataset() additionally merges in
research.early_failure_trajectory.build_trajectory_table() (day-indexed
MFE/MAE) — REUSED, not re-derived — producing the single DataFrame every
research/trade_intelligence_*.py analysis module consumes.

Never writes to trade_history.db. Never imports engine.runner, risk.*,
portfolio.*, strategies.*, position_manager.*, exit_engine.*, ai_decision.*.
"""
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from research.early_failure_trajectory import build_trajectory_table

_DB_PATH = Path(r"C:\KabuData\trade_history\trade_history.db")

# Excludes the equity_before==50000.0 phantom rows and open (no exit_time)
# trades — same convention as early_failure_trajectory.build_trajectory_table.
_JOIN_QUERY = """
    SELECT
        t.trade_id, t.ticker, t.strategy_name, t.strategy_version,
        t.direction, t.entry_time, t.entry_price, t.exit_time, t.exit_price,
        t.shares, t.position_value, t.position_pct, t.holding_days,
        t.holding_hours, t.exit_reason, t.exit_reason_code, t.pnl,
        t.pnl_pct, t.atr_entry, t.sector, t.market_environment,
        t.entry_rank, t.confidence_score AS entry_confidence_score,
        t.risk_per_trade, t.mfe, t.mae, t.market, t.execution,
        a.entry_regime, a.entry_regime_label, a.entry_confidence,
        a.exit_regime, a.exit_regime_label, a.exit_confidence,
        a.regime_drifted,
        d.exit_category, d.mfe_pct, d.mae_pct, d.giveback_pct,
        d.mfe_capture, d.stop_loss_triggered, d.strategy_exit_triggered,
        c.rule_based_score, c.hmm_state, c.hmm_confidence, c.hmm_component,
        c.historical_win_rate, c.historical_win_rate_n,
        c.historical_expectancy_pct, c.market_component,
        c.volatility_atr_pct, c.volatility_component, c.volume_feature,
        c.confidence_score, c.position_multiplier, c.base_position,
        c.requested_position, c.risk_adjusted_position, c.final_position,
        c.skip_reason,
        q.distance_from_breakout_atr, q.distance_from_20d_high,
        q.distance_from_20d_ema, q.entry_day_return, q.entry_volume_ratio,
        q.atr_expansion_ratio, q.rule_score AS entry_quality_rule_score
    FROM trades t
    LEFT JOIN trade_attribution a ON a.trade_id = t.trade_id
    LEFT JOIN trade_exit_diagnostics d ON d.trade_id = t.trade_id
    LEFT JOIN trade_confidence_score c ON c.trade_id = t.trade_id
    LEFT JOIN trade_entry_quality q ON q.trade_id = t.trade_id
    WHERE t.exit_time IS NOT NULL AND t.equity_before != 50000.0
"""

_TRAJECTORY_COLUMNS = [
    "trade_id", "MFE_at_day_1", "MFE_at_day_2", "MFE_at_day_3", "MFE_at_day_5",
    "MAE_at_day_3", "MAE_at_day_5", "n_daily_rows", "days_since_last_high",
    "consecutive_days_without_new_high", "days_since_last_positive_close",
    "consecutive_weak_close_days",
]


def load_closed_trades(db_path: Optional[Path] = None,
                        execution: Optional[str] = "REAL") -> pd.DataFrame:
    """One row per closed trade, joined against every table V3.6-A's four
    in-scope analyses need. execution="REAL"/"PAPER" filters to one fill
    kind (mirrors engine.trade_tracker.TradeTracker.query_trades());
    execution=None returns both mixed."""
    con = sqlite3.connect(str(db_path or _DB_PATH))
    try:
        query = _JOIN_QUERY
        params: tuple = ()
        if execution is not None:
            query += " AND t.execution = ?"
            params = (execution,)
        return pd.read_sql_query(query, con, params=params)
    finally:
        con.close()


def load_dataset(db_path: Optional[Path] = None,
                  execution: Optional[str] = "REAL") -> pd.DataFrame:
    """load_closed_trades() left-joined with
    research.early_failure_trajectory.build_trajectory_table(db_path) — the
    single unified DataFrame all four V3.6-A analysis modules consume."""
    trades = load_closed_trades(db_path, execution)
    if trades.empty:
        return trades
    traj = build_trajectory_table(db_path)
    if traj.empty:
        return trades
    cols = [c for c in _TRAJECTORY_COLUMNS if c in traj.columns]
    return trades.merge(traj[cols], on="trade_id", how="left")
