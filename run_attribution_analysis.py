"""
run_attribution_analysis.py — offline analysis for the v2.6 Trade
Intelligence Database (engine/trade_tracker.py).

Reads engine/trade_history.db and attributes closed-trade P&L to the market
regime (engine/market_regime.py, "MRD") that was active at trade entry.
Read-only — never touches trading state, portfolio state, or the DB schema
beyond a SELECT.

Usage:
    cd D:\\workspace\\moomoo\\kabu
    python run_attribution_analysis.py
"""
import json
import os
import sqlite3

import pandas as pd

from engine.trade_tracker import _DEFAULT_DB_PATH

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "analytics", "attribution_summary.json")


def _load_closed_trades(db_path) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            """
            SELECT t.trade_id, t.ticker, t.strategy_name, t.pnl, t.pnl_pct,
                   t.holding_days, t.position_pct, t.exit_time,
                   a.entry_regime_label
            FROM trades t
            LEFT JOIN trade_attribution a ON a.trade_id = t.trade_id
            WHERE t.exit_time IS NOT NULL
            """,
            conn,
        )
    finally:
        conn.close()
    df["entry_regime_label"] = df["entry_regime_label"].fillna("UNKNOWN")
    return df


def _profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in df.groupby("entry_regime_label"):
        n = len(g)
        wins = int((g["pnl"] > 0).sum())
        rows.append({
            "entry_regime_label": label,
            "trades": n,
            "win_rate": wins / n if n else 0.0,
            "total_pnl": g["pnl"].sum(),
            "avg_pnl": g["pnl"].mean(),
            "avg_return": g["pnl_pct"].mean(),
            "profit_factor": _profit_factor(g["pnl"]),
            "avg_holding_days": g["holding_days"].mean(),
            "avg_position_pct": g["position_pct"].mean(),
        })
    return pd.DataFrame(rows).sort_values("trades", ascending=False)


def main():
    df = _load_closed_trades(_DEFAULT_DB_PATH)
    if df.empty:
        print(f"没有已平仓交易记录（{_DEFAULT_DB_PATH}）——先跑一段实盘/模拟盘，"
              f"或用 engine.trade_tracker.ingest_backtest_trade_log() 导入一次回测结果。")
        return

    summary = _summarize(df)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n" + "=" * 110)
    print("按开仓市场状态（entry_regime_label，engine/market_regime.py MRD）分组的交易归因报表")
    print("=" * 110)
    display = summary.copy()
    display["win_rate"]         = display["win_rate"].map("{:.1%}".format)
    display["total_pnl"]        = display["total_pnl"].map("{:,.2f}".format)
    display["avg_pnl"]          = display["avg_pnl"].map("{:,.2f}".format)
    display["avg_return"]       = display["avg_return"].map("{:.2%}".format)
    display["profit_factor"]    = display["profit_factor"].map(
        lambda v: "inf" if v == float("inf") else f"{v:.2f}")
    display["avg_holding_days"] = display["avg_holding_days"].map("{:.1f}".format)
    display["avg_position_pct"] = display["avg_position_pct"].map("{:.1%}".format)
    print(display.to_string(index=False))

    out = {}
    for _, row in summary.iterrows():
        out[row["entry_regime_label"]] = {
            "trades":           int(row["trades"]),
            "win_rate":         round(float(row["win_rate"]), 4),
            "profit_factor":    (round(float(row["profit_factor"]), 4)
                                 if row["profit_factor"] != float("inf") else None),
            "avg_return":       round(float(row["avg_return"]), 4),
            "avg_holding_days": round(float(row["avg_holding_days"]), 2),
            "avg_position_pct": round(float(row["avg_position_pct"]), 4),
        }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 已导出: {OUT_PATH}")


if __name__ == "__main__":
    main()
