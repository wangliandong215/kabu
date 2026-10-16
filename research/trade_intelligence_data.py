"""
research/trade_intelligence_data.py — V3.6-A Trade Intelligence data access
layer (research-only, see research/trade_intelligence_runner.py's module
docstring for the full V3.6-A boundary).

Read-only. One row per trade from trade_history.db, LEFT-joined against
every per-trade table that already exists (trade_attribution,
trade_exit_diagnostics, trade_confidence_score, trade_entry_quality,
trade_market_context, trade_research_snapshot, metadata via run_id), plus
per-trade aggregates of trade_daily and two recorded-at-the-time context
sources:
  * QQQ / SPY state — the live system's own trade_daily rows for its
    US.QQQ / US.SPY positions (close + HMM regime label), taken from the
    last recorded day STRICTLY BEFORE the entry date (no same-day
    look-ahead), and only if that day is within _MARKET_STATE_MAX_AGE_DAYS.
  * Portfolio exposure / risk tier — risk_state_log.jsonl (written by the
    v2.10 Portfolio Risk Manager at the start of every pass since
    2026-08-29), the last snapshot at or before entry_time within
    _RISK_SNAPSHOT_MAX_AGE.
Nothing is estimated or reconstructed: where the history does not contain a
value, the column is NULL. Plain sqlite3/json reads — deliberately NOT
engine.trade_tracker.TradeTracker (no write lock needed to read).

Every row is also classified (classify_population) so no trade is silently
dropped: is_phantom, is_closed, execution_class (+ reason), account_env,
pre_fill_confirmation_fix, and — for a given selection — inclusion_status /
inclusion_reason. Raw trades.execution is never modified; the 15 legacy
rows with execution=NULL get a research-side execution_class instead:
  INFERRED_REAL — NULL, US-listed, entered before engine/broker.py's
      PaperBroker existed (2026-07-26, commit b9148d7). Before that date
      every US order could only have been routed to moomoo, i.e. the same
      channel the later "REAL" rows record. Kept distinguishable from REAL.
  UNKNOWN       — NULL and that inference does not hold.
Note "REAL" here means "routed to moomoo", not "real money": account_env
(metadata.market, via run_id) says which moomoo account — every run on
record is SIMULATE.

Never writes to trade_history.db. Never imports engine.runner, risk.*,
portfolio.*, strategies.*, position_manager.*, exit_engine.*, ai_decision.*.
"""
import bisect
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from research.early_failure_trajectory import build_trajectory_table
import config

_DB_PATH = Path(config.DATA_ROOT, "trade_history", "trade_history.db")
# Written by risk/portfolio_risk_manager.py (_RISK_LOG_PATH) — read only.
_RISK_STATE_LOG_PATH = Path(config.DATA_ROOT, "portfolio", "risk_state_log.jsonl")

_PHANTOM_EQUITY_BEFORE = 50000.0
# engine/broker.py PaperBroker (JP virtual fills) landed in b9148d7.
_PAPER_BROKER_INTRODUCED = "2026-07-26"
# 58d29d1 (2026-08-12 JST): fills gated on broker dealt_qty. Before it, a
# recorded entry/exit may be an order that never actually filled (see
# engine/broker.py's module docstring). Flagged, never excluded.
_FILL_CONFIRMATION_FIX = "2026-08-13"

_MARKET_STATE_MAX_AGE_DAYS = 5
_RISK_SNAPSHOT_MAX_AGE = timedelta(minutes=15)

REAL = "REAL"
PAPER = "PAPER"
INFERRED_REAL = "INFERRED_REAL"
UNKNOWN = "UNKNOWN"

# (table, alias, source column or SQL expression, output column).
# A missing table/column yields NULL, never an error.
_COLUMNS = [
    ("trades", "t", "trade_id", "trade_id"),
    ("trades", "t", "ticker", "ticker"),
    ("trades", "t", "strategy_name", "strategy_name"),
    ("trades", "t", "strategy_version", "strategy_version"),
    ("trades", "t", "direction", "direction"),
    ("trades", "t", "entry_time", "entry_time"),
    ("trades", "t", "entry_price", "entry_price"),
    ("trades", "t", "exit_time", "exit_time"),
    ("trades", "t", "exit_price", "exit_price"),
    ("trades", "t", "shares", "shares"),
    ("trades", "t", "position_value", "position_value"),
    ("trades", "t", "position_pct", "position_pct"),
    ("trades", "t", "cash_before", "cash_before"),
    ("trades", "t", "equity_before", "equity_before"),
    ("trades", "t", "holding_days", "holding_days"),
    ("trades", "t", "holding_hours", "holding_hours"),
    ("trades", "t", "exit_reason", "exit_reason"),
    ("trades", "t", "exit_reason_code", "exit_reason_code"),
    ("trades", "t", "pnl", "pnl"),
    ("trades", "t", "pnl_pct", "pnl_pct"),
    ("trades", "t", "run_id", "run_id"),
    ("trades", "t", "atr_entry", "atr_entry"),
    ("trades", "t", "sector", "sector"),
    ("trades", "t", "market_environment", "market_environment"),
    ("trades", "t", "entry_rank", "entry_rank"),
    ("trades", "t", "confidence_score", "entry_confidence_score"),
    ("trades", "t", "risk_per_trade", "risk_per_trade"),
    ("trades", "t", "commission", "commission"),
    ("trades", "t", "slippage", "slippage"),
    ("trades", "t", "mfe", "mfe"),
    ("trades", "t", "mae", "mae"),
    ("trades", "t", "market", "market"),
    ("trades", "t", "execution", "execution"),
    ("trade_attribution", "a", "entry_regime", "entry_regime"),
    ("trade_attribution", "a", "entry_regime_label", "entry_regime_label"),
    ("trade_attribution", "a", "entry_confidence", "entry_confidence"),
    ("trade_attribution", "a", "exit_regime", "exit_regime"),
    ("trade_attribution", "a", "exit_regime_label", "exit_regime_label"),
    ("trade_attribution", "a", "exit_confidence", "exit_confidence"),
    ("trade_attribution", "a", "regime_drifted", "regime_drifted"),
    ("trade_attribution", "a", "mrd_version", "mrd_version"),
    ("trade_attribution", "a", "json:entry_context_json:$.features.adx", "entry_adx"),
    ("trade_attribution", "a", "json:entry_context_json:$.features.atr_pct", "entry_atr_pct"),
    ("trade_attribution", "a", "json:entry_context_json:$.features.bb_width", "entry_bb_width"),
    ("trade_attribution", "a", "json:entry_context_json:$.features.volume_ratio",
     "entry_regime_volume_ratio"),
    ("trade_exit_diagnostics", "d", "exit_category", "exit_category"),
    ("trade_exit_diagnostics", "d", "mfe_pct", "mfe_pct"),
    ("trade_exit_diagnostics", "d", "mae_pct", "mae_pct"),
    ("trade_exit_diagnostics", "d", "giveback_pct", "giveback_pct"),
    ("trade_exit_diagnostics", "d", "mfe_capture", "mfe_capture"),
    ("trade_exit_diagnostics", "d", "stop_loss_triggered", "stop_loss_triggered"),
    ("trade_exit_diagnostics", "d", "strategy_exit_triggered", "strategy_exit_triggered"),
    ("trade_confidence_score", "c", "rule_based_score", "rule_based_score"),
    ("trade_confidence_score", "c", "hmm_state", "hmm_state"),
    ("trade_confidence_score", "c", "hmm_confidence", "hmm_confidence"),
    ("trade_confidence_score", "c", "hmm_component", "hmm_component"),
    ("trade_confidence_score", "c", "historical_win_rate", "historical_win_rate"),
    ("trade_confidence_score", "c", "historical_win_rate_n", "historical_win_rate_n"),
    ("trade_confidence_score", "c", "historical_expectancy_pct", "historical_expectancy_pct"),
    ("trade_confidence_score", "c", "market_component", "market_component"),
    ("trade_confidence_score", "c", "volatility_atr_pct", "volatility_atr_pct"),
    ("trade_confidence_score", "c", "volatility_component", "volatility_component"),
    ("trade_confidence_score", "c", "volume_feature", "volume_feature"),
    ("trade_confidence_score", "c", "confidence_score", "confidence_score"),
    ("trade_confidence_score", "c", "position_multiplier", "position_multiplier"),
    ("trade_confidence_score", "c", "base_position", "base_position"),
    ("trade_confidence_score", "c", "requested_position", "requested_position"),
    ("trade_confidence_score", "c", "risk_adjusted_position", "risk_adjusted_position"),
    ("trade_confidence_score", "c", "final_position", "final_position"),
    ("trade_confidence_score", "c", "skip_reason", "skip_reason"),
    ("trade_entry_quality", "q", "distance_from_breakout_atr", "distance_from_breakout_atr"),
    ("trade_entry_quality", "q", "distance_from_20d_high", "distance_from_20d_high"),
    ("trade_entry_quality", "q", "distance_from_20d_ema", "distance_from_20d_ema"),
    ("trade_entry_quality", "q", "entry_day_return", "entry_day_return"),
    ("trade_entry_quality", "q", "entry_volume_ratio", "entry_volume_ratio"),
    ("trade_entry_quality", "q", "atr_expansion_ratio", "atr_expansion_ratio"),
    ("trade_entry_quality", "q", "rule_score", "entry_quality_rule_score"),
    # Post-entry OUTCOME labels — never usable as entry-time features.
    ("trade_entry_quality", "q", "return_5d", "outcome_fwd_return_5d"),
    ("trade_entry_quality", "q", "return_10d", "outcome_fwd_return_10d"),
    ("trade_entry_quality", "q", "return_20d", "outcome_fwd_return_20d"),
    ("trade_entry_quality", "q", "return_60d", "outcome_fwd_return_60d"),
    ("trade_market_context", "mc", "observation_date", "mc_observation_date"),
    ("trade_market_context", "mc", "vix_close", "mc_vix_close"),
    ("trade_market_context", "mc", "cnn_fear_greed", "mc_cnn_fear_greed"),
    ("trade_market_context", "mc", "cnn_fear_greed_label", "mc_cnn_fear_greed_label"),
    ("trade_market_context", "mc", "aaii_bullish", "mc_aaii_bullish"),
    ("trade_market_context", "mc", "aaii_bearish", "mc_aaii_bearish"),
    ("trade_market_context", "mc", "aaii_spread", "mc_aaii_spread"),
    ("trade_market_context", "mc", "put_call_ratio", "mc_put_call_ratio"),
    ("trade_market_context", "mc", "naaim_exposure", "mc_naaim_exposure"),
    ("trade_market_context", "mc", "data_status", "mc_data_status"),
    ("trade_research_snapshot", "rs", "signal_time", "rs_signal_time"),
    ("trade_research_snapshot", "rs", "rsi14", "rs_rsi14"),
    ("trade_research_snapshot", "rs", "atr", "rs_atr"),
    ("trade_research_snapshot", "rs", "signal_strength", "rs_signal_strength"),
    ("trade_research_snapshot", "rs", "total_score", "rs_total_score"),
    ("trade_research_snapshot", "rs", "has_earnings_risk", "rs_has_earnings_risk"),
    ("trade_research_snapshot", "rs", "days_to_earnings", "rs_days_to_earnings"),
    ("trade_research_snapshot", "rs", "news_count_24h", "rs_news_count_24h"),
    ("trade_research_snapshot", "rs", "iv", "rs_iv"),
    ("trade_research_snapshot", "rs", "hv_30d", "rs_hv_30d"),
    ("trade_research_snapshot", "rs", "put_call_ratio", "rs_put_call_ratio"),
    ("trade_research_snapshot", "rs", "fedwatch_target_range", "rs_fedwatch_target_range"),
    ("trade_research_snapshot", "rs", "fedwatch_probability", "rs_fedwatch_probability"),
    ("metadata", "m", "market", "account_env"),
    ("metadata", "m", "strategy_version", "run_strategy_version"),
    ("metadata", "m", "parameter_hash", "parameter_hash"),
    ("metadata", "m", "git_commit", "git_commit"),
]

_JOINS = [
    ("trade_attribution", "a", "a.trade_id = t.trade_id"),
    ("trade_exit_diagnostics", "d", "d.trade_id = t.trade_id"),
    ("trade_confidence_score", "c", "c.trade_id = t.trade_id"),
    ("trade_entry_quality", "q", "q.trade_id = t.trade_id"),
    ("trade_market_context", "mc", "mc.trade_id = t.trade_id"),
    ("trade_research_snapshot", "rs", "rs.trade_id = t.trade_id"),
    ("metadata", "m", "m.run_id = t.run_id"),
]
_JOIN_KEYS = {"a": "trade_id", "d": "trade_id", "c": "trade_id", "q": "trade_id",
              "mc": "trade_id", "rs": "trade_id", "m": "run_id"}

_DAILY_COLUMNS = ["daily_n_rows", "daily_realized_vol", "daily_regime_changes",
                  "daily_first_regime_label", "daily_last_regime_label",
                  "daily_max_floating_pnl_pct", "daily_min_floating_pnl_pct"]
_MARKET_STATE_COLUMNS = [f"{p}_{c}" for p in ("qqq", "spy")
                         for c in ("state_date", "prev_close", "prev_regime_label")]
_RISK_COLUMNS = ["entry_exposure_pct", "entry_risk_tier", "entry_risk_snapshot_at"]
_CLASS_COLUMNS = ["is_phantom", "is_closed", "execution_class", "execution_class_reason",
                  "pre_fill_confirmation_fix"]

ALL_COLUMNS = ([c[3] for c in _COLUMNS] + _DAILY_COLUMNS + _MARKET_STATE_COLUMNS
               + _RISK_COLUMNS + _CLASS_COLUMNS)

_TRAJECTORY_COLUMNS = [
    "trade_id", "MFE_at_day_1", "MFE_at_day_2", "MFE_at_day_3", "MFE_at_day_5",
    "MAE_at_day_3", "MAE_at_day_5", "n_daily_rows", "days_since_last_high",
    "consecutive_days_without_new_high", "days_since_last_positive_close",
    "consecutive_weak_close_days",
]


# ── raw loading ───────────────────────────────────────────────────────────────

def _table_columns(con: sqlite3.Connection) -> dict:
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    return {t: {r[1] for r in con.execute(f'PRAGMA table_info("{t}")')} for t in tables}


def _build_query(schema: dict) -> str:
    joined = {"t"}
    joins = []
    for table, alias, on in _JOINS:
        if _JOIN_KEYS[alias] in schema.get(table, set()) and (
                alias != "m" or "run_id" in schema.get("trades", set())):
            joins.append(f"LEFT JOIN {table} {alias} ON {on}")
            joined.add(alias)
    selects = []
    for table, alias, src, out in _COLUMNS:
        cols = schema.get(table, set())
        if alias not in joined:
            selects.append(f"NULL AS {out}")
        elif src.startswith("json:"):
            _, col, path = src.split(":", 2)
            selects.append(
                f"CASE WHEN json_valid({alias}.{col}) THEN json_extract({alias}.{col}, '{path}') "
                f"END AS {out}" if col in cols else f"NULL AS {out}")
        else:
            selects.append(f"{alias}.{src} AS {out}" if src in cols else f"NULL AS {out}")
    return f"SELECT {', '.join(selects)} FROM trades t {' '.join(joins)}"


def _daily_features(con: sqlite3.Connection, schema: dict,
                    trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(per-trade trade_daily aggregates, QQQ/SPY recorded daily series)."""
    need = {"trade_id", "date", "close", "floating_pnl_pct", "regime_label"}
    if not need <= schema.get("trade_daily", set()):
        return pd.DataFrame(columns=["trade_id"] + _DAILY_COLUMNS), pd.DataFrame()
    daily = pd.read_sql_query(
        "SELECT trade_id, date, close, floating_pnl_pct, regime_label FROM trade_daily "
        "ORDER BY trade_id, date", con)
    if daily.empty:
        return pd.DataFrame(columns=["trade_id"] + _DAILY_COLUMNS), daily

    def _agg(g: pd.DataFrame) -> pd.Series:
        labels = g["regime_label"].dropna().tolist()
        rets = g["close"].astype(float).pct_change().dropna()
        return pd.Series({
            "daily_n_rows": len(g),
            "daily_realized_vol": float(rets.std()) if len(rets) >= 2 else None,
            "daily_regime_changes": sum(1 for a, b in zip(labels, labels[1:]) if a != b),
            "daily_first_regime_label": labels[0] if labels else None,
            "daily_last_regime_label": labels[-1] if labels else None,
            "daily_max_floating_pnl_pct": g["floating_pnl_pct"].max(),
            "daily_min_floating_pnl_pct": g["floating_pnl_pct"].min(),
        })

    agg = daily.groupby("trade_id")[["close", "floating_pnl_pct", "regime_label"]] \
               .apply(_agg).reset_index()
    ticker_of = dict(zip(trades["trade_id"], trades["ticker"]))
    daily["ticker"] = daily["trade_id"].map(
        lambda tid: ticker_of.get(tid) or str(tid).split("_", 1)[0])
    return agg, daily


def _market_state(trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """QQQ/SPY close + regime from the last recorded day strictly before the
    entry date, if no older than _MARKET_STATE_MAX_AGE_DAYS. Otherwise NULL."""
    out = pd.DataFrame({"trade_id": trades["trade_id"]})
    for prefix, code in (("qqq", "US.QQQ"), ("spy", "US.SPY")):
        series = (daily[daily["ticker"] == code].sort_values("date")
                  .drop_duplicates("date", keep="last") if not daily.empty else daily)
        dates = list(series["date"]) if not series.empty else []
        vals = {c: [] for c in ("state_date", "prev_close", "prev_regime_label")}
        for entry_time in trades["entry_time"]:
            hit = None
            try:
                entry_date = str(entry_time)[:10]
                i = bisect.bisect_left(dates, entry_date) - 1
                if i >= 0:
                    age = (datetime.fromisoformat(entry_date)
                           - datetime.fromisoformat(dates[i])).days
                    if age <= _MARKET_STATE_MAX_AGE_DAYS:
                        hit = series.iloc[i]
            except (TypeError, ValueError):
                hit = None
            vals["state_date"].append(hit["date"] if hit is not None else None)
            vals["prev_close"].append(hit["close"] if hit is not None else None)
            vals["prev_regime_label"].append(hit["regime_label"] if hit is not None else None)
        for c, v in vals.items():
            out[f"{prefix}_{c}"] = v
    return out


def _load_risk_snapshots(path: Path) -> list:
    """[(timestamp, row)] sorted. Skips unparseable lines and test-pollution
    rows (integer total_assets — see test_support.py / 2026-09-23 cleanup)."""
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    total = float(r["total_assets"])
                    if total.is_integer():
                        continue
                    rows.append((datetime.fromisoformat(r["timestamp"]), r))
                except (ValueError, KeyError, TypeError):
                    continue
    except OSError:
        return []
    rows.sort(key=lambda x: x[0])
    return rows


def _risk_state(trades: pd.DataFrame, risk_log_path: Path) -> pd.DataFrame:
    snaps = _load_risk_snapshots(risk_log_path)
    stamps = [s[0] for s in snaps]
    recs = []
    for entry_time in trades["entry_time"]:
        rec = dict.fromkeys(_RISK_COLUMNS)
        try:
            et = datetime.fromisoformat(str(entry_time))
            i = bisect.bisect_right(stamps, et) - 1
            if i >= 0 and et - stamps[i] <= _RISK_SNAPSHOT_MAX_AGE:
                r = snaps[i][1]
                rec = {"entry_exposure_pct": r.get("exposure_pct"),
                       "entry_risk_tier": r.get("tier"),
                       "entry_risk_snapshot_at": r.get("timestamp")}
        except (TypeError, ValueError):
            pass
        recs.append(rec)
    return pd.concat([trades[["trade_id"]].reset_index(drop=True),
                      pd.DataFrame(recs, columns=_RISK_COLUMNS)], axis=1)


# ── classification ────────────────────────────────────────────────────────────

def _execution_class(row) -> Tuple[str, str]:
    raw = row.get("execution")
    if raw == REAL:
        return REAL, "recorded execution=REAL (routed to moomoo)"
    if raw == PAPER:
        return PAPER, "recorded execution=PAPER (engine/broker.py PaperBroker)"
    if raw is not None and not pd.isna(raw):
        return UNKNOWN, f"unrecognized recorded execution={raw!r}"
    ticker = str(row.get("ticker") or "")
    entry = str(row.get("entry_time") or "")
    if ticker.startswith("US.") and entry and entry[:10] < _PAPER_BROKER_INTRODUCED:
        return INFERRED_REAL, (f"execution NULL; US ticker entered before PaperBroker existed "
                               f"({_PAPER_BROKER_INTRODUCED}) so it could only route to moomoo")
    return UNKNOWN, "execution NULL and routing cannot be inferred"


def classify_population(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        for c in _CLASS_COLUMNS:
            df[c] = pd.Series(dtype=object)
        return df
    df["is_phantom"] = df["equity_before"] == _PHANTOM_EQUITY_BEFORE
    df["is_closed"] = df["exit_time"].notna()
    cls = df.apply(_execution_class, axis=1, result_type="expand")
    df["execution_class"], df["execution_class_reason"] = cls[0], cls[1]
    entry = df["entry_time"].astype(str).str[:10]
    exit_ = df["exit_time"].where(df["exit_time"].notna(), "9999").astype(str).str[:10]
    df["pre_fill_confirmation_fix"] = (entry < _FILL_CONFIRMATION_FIX) | (exit_ < _FILL_CONFIRMATION_FIX)
    return df


def selected_classes(execution: Optional[str], include_inferred: bool) -> Optional[set]:
    """execution=None -> every class (None). "REAL" -> REAL (+ INFERRED_REAL
    unless include_inferred=False). Any other value -> that class only."""
    if execution is None:
        return None
    classes = {execution}
    if execution == REAL and include_inferred:
        classes.add(INFERRED_REAL)
    return classes


def mark_inclusion(df: pd.DataFrame, execution: Optional[str] = REAL,
                   include_inferred: bool = True) -> pd.DataFrame:
    """Adds inclusion_status (INCLUDED/EXCLUDED) + inclusion_reason. Order of
    precedence: PHANTOM, OPEN, then execution selection."""
    df = df.copy()
    classes = selected_classes(execution, include_inferred)

    def _reason(row) -> str:
        if row["is_phantom"]:
            return "PHANTOM"
        if not row["is_closed"]:
            return "OPEN"
        if classes is not None and row["execution_class"] not in classes:
            if row["execution_class"] == UNKNOWN:
                return "EXECUTION_UNKNOWN"
            if row["execution_class"] == INFERRED_REAL:
                return "INFERRED_NOT_SELECTED"
            return "EXECUTION_NOT_SELECTED"
        return "INCLUDED"

    reasons = df.apply(_reason, axis=1) if not df.empty else pd.Series(dtype=object)
    df["inclusion_reason"] = reasons
    df["inclusion_status"] = reasons.map(lambda r: "INCLUDED" if r == "INCLUDED" else "EXCLUDED") \
        if not df.empty else pd.Series(dtype=object)
    return df


# ── public API ────────────────────────────────────────────────────────────────

def load_all_trades(db_path: Optional[Path] = None,
                    risk_log_path: Optional[Path] = None) -> pd.DataFrame:
    """Every trade (open, closed, phantom, any execution), fully joined and
    classified. Missing DB/tables/columns degrade to NULL columns or an empty
    frame — never an exception."""
    path = Path(db_path or _DB_PATH)
    empty = pd.DataFrame(columns=ALL_COLUMNS)
    if not path.exists():
        return empty
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        schema = _table_columns(con)
        if "trade_id" not in schema.get("trades", set()):
            return empty
        trades = pd.read_sql_query(_build_query(schema), con)
        if trades.empty:
            return empty
        daily_agg, daily = _daily_features(con, schema, trades)
    finally:
        con.close()
    out = trades.merge(daily_agg, on="trade_id", how="left")
    out = out.merge(_market_state(trades, daily), on="trade_id", how="left")
    out = out.merge(_risk_state(trades, Path(risk_log_path or _RISK_STATE_LOG_PATH)),
                    on="trade_id", how="left")
    return classify_population(out)


def load_population(db_path: Optional[Path] = None, execution: Optional[str] = REAL,
                    include_inferred: bool = True,
                    risk_log_path: Optional[Path] = None) -> pd.DataFrame:
    """load_all_trades() + inclusion_status/inclusion_reason for one selection."""
    return mark_inclusion(load_all_trades(db_path, risk_log_path), execution, include_inferred)


def load_closed_trades(db_path: Optional[Path] = None,
                       execution: Optional[str] = REAL,
                       include_inferred: bool = True,
                       risk_log_path: Optional[Path] = None) -> pd.DataFrame:
    """Closed, non-phantom trades in the selected execution classes (see
    selected_classes). execution=None returns every class mixed."""
    pop = load_population(db_path, execution, include_inferred, risk_log_path)
    if pop.empty:
        return pop
    return pop[pop["inclusion_status"] == "INCLUDED"].reset_index(drop=True)


def load_dataset(db_path: Optional[Path] = None,
                 execution: Optional[str] = REAL,
                 include_inferred: bool = True,
                 risk_log_path: Optional[Path] = None) -> pd.DataFrame:
    """load_closed_trades() left-joined with
    research.early_failure_trajectory.build_trajectory_table(db_path) — the
    single unified DataFrame all V3.6-A analysis modules consume."""
    return attach_trajectory(
        load_closed_trades(db_path, execution, include_inferred, risk_log_path), db_path)


def attach_trajectory(trades: pd.DataFrame, db_path: Optional[Path] = None) -> pd.DataFrame:
    if trades.empty:
        return trades
    try:
        traj = build_trajectory_table(db_path)
    except Exception:
        return trades
    if traj.empty:
        return trades
    cols = [c for c in _TRAJECTORY_COLUMNS if c in traj.columns]
    return trades.merge(traj[cols], on="trade_id", how="left")


def included(pop: pd.DataFrame) -> pd.DataFrame:
    if pop.empty or "inclusion_status" not in pop.columns:
        return pop.iloc[0:0]
    return pop[pop["inclusion_status"] == "INCLUDED"].reset_index(drop=True)
