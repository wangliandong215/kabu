"""
_regime_analytics_v27.py -- V2.7 Stage 4, "Regime Analytics" phase
(2026-07-08). Answers "which HMM regime does this strategy make/lose money
in" using historical trade data, WITHOUT changing any trading decision --
all HMM-based trading-rule experiments (Round 1's Bear-entry gate, the
planned Bull-position-size Round 2) are paused pending this analysis.

Runs the pure V2.6/V2.7 baseline (config.WATCHLIST_EUROPE_US, 139 stocks,
2015-01-01~2026-07-07, $50,000) with regime_lookup supplied but
enable_bear_gate left at its default False -- backtest_portfolio.py tags
every trade's entry/exit regime, confidence, and duration without
suppressing anything, so this run's Total Return/Sharpe/trade count/etc.
should be identical to _yearly_report_v27_locked.py's numbers. Only the
per-trade regime tags are new information, layered on top of the unchanged
baseline.

Coverage caveat: regime tags only exist for the 8 pilot stocks with a
trained engine/hmm_regime.py model (see project memory, V2.7 Stage 3) --
AAPL/QQQ/MSFT/NVDA/AMZN/GOOGL/TSLA/SPY. The other 131 stocks' trades show up
in the "N/A (ungated stock)" bucket. Conclusions here describe those 8
stocks' historical behavior, not the full 139-stock universe -- expanding
model coverage is a separate, not-yet-authorized step.

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _regime_analytics_v27.py
"""
import contextlib

import numpy as np
import pandas as pd

import backtest
import backtest_portfolio as bp
import config
from _yearly_report import _build_table
from engine.hmm_regime import decode_regime_series_causal, load_stock_hmm, prepare_ohlcv

START = "2015-01-01"
END = "2026-07-07"
CASH = 50_000.0

PILOT_CODES = [
    "US.AAPL", "US.QQQ", "US.MSFT", "US.NVDA",
    "US.AMZN", "US.GOOGL", "US.TSLA", "US.SPY",
]

LOG_PATH = "_regime_analytics_v27.log"
REGIME_ORDER = ["Bear", "Correction", "Sideways", "Bull", "N/A (ungated stock)"]


def build_regime_lookup(codes, start, end) -> dict:
    lookup = {}
    for code in codes:
        payload = load_stock_hmm(code)
        if payload is None:
            print(f"  [skip] {code}: no trained HMM model")
            continue
        raw = backtest.fetch_kline(code, start, end)
        df = prepare_ohlcv(raw)
        regime_df = decode_regime_series_causal(payload, df)
        lookup[code] = regime_df
        counts = regime_df["regime"].dropna().value_counts().to_dict()
        print(f"  [ok] {code}: {counts}")
    return lookup


def _sell_trades_df(result: dict) -> pd.DataFrame:
    sell = [t for t in result["trade_log"] if t["side"] == "SELL"]
    df = pd.DataFrame(sell)
    if len(df) == 0:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["pnl"] = df["pnl"].astype(float)
    df["holding_days"] = (df["date"] - df["entry_date"]).dt.days
    # Cost-basis % return per trade -- normalizes across position sizes, the
    # right unit for a mean/std ("Sharpe-like") comparison across regimes.
    # entry_price is populated for every trade now (analytics-only field
    # added in this phase), not just the 8 pilot stocks.
    cost = df["entry_price"] * df["qty"]
    df["pct_return"] = np.where(cost > 0, df["pnl"] / cost, np.nan)
    return df


def _trade_sharpe_like(pct_returns: pd.Series, avg_holding_days: float) -> float:
    """
    Trade-level Sharpe approximation: mean/std of per-trade % returns,
    annualized by assuming ~252/avg_holding_days independent trades/year.
    This is NOT the same methodology as _yearly_report.py's portfolio-level
    daily-equity-curve Sharpe -- regime is a per-stock, per-day label, so a
    single portfolio-wide daily return can't be cleanly attributed to one
    regime on days multiple positions in different regimes are held at once.
    Trade-level is the right granularity for "which regime's trades looked
    better," but the two Sharpe numbers are not directly comparable -- don't
    quote this figure next to the portfolio Sharpe in _yearly_report.py's
    table without this caveat.
    """
    n = len(pct_returns)
    if n < 2 or pct_returns.std() == 0 or avg_holding_days <= 0:
        return float("nan")
    trades_per_year = 252.0 / avg_holding_days
    return float(pct_returns.mean() / pct_returns.std() * np.sqrt(trades_per_year))


def regime_breakdown(trades_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if len(trades_df) == 0:
        return pd.DataFrame()
    df = trades_df.copy()
    df[group_col] = df[group_col].fillna("N/A (ungated stock)")

    total_pnl_all = df["pnl"].sum()
    rows = {}
    for label, sub in df.groupby(group_col):
        n = len(sub)
        wins = sub[sub["pnl"] > 0]
        gross_win = sub.loc[sub["pnl"] > 0, "pnl"].sum()
        gross_loss = abs(sub.loc[sub["pnl"] < 0, "pnl"].sum())
        if gross_loss == 0:
            profit_factor = 99.9 if gross_win > 0 else 0.0
        else:
            profit_factor = gross_win / gross_loss
        avg_hold = sub["holding_days"].mean()
        # group_col is "regime_at_entry" or "regime_at_exit" -> matching
        # confidence/duration columns are "confidence_at_entry"/"duration_at_entry"
        # or their _at_exit counterparts (see backtest_portfolio.py's trade_log tagging).
        suffix = group_col.replace("regime_", "")   # "at_entry" / "at_exit"
        conf_col = f"confidence_{suffix}"
        dur_col = f"duration_{suffix}"
        rows[label] = {
            "trades": n,
            "contribution_%_of_total_pnl": round(sub["pnl"].sum() / total_pnl_all * 100, 1)
            if total_pnl_all != 0 else float("nan"),
            "total_pnl": round(sub["pnl"].sum(), 2),
            "win_rate_%": round(len(wins) / n * 100, 1) if n else 0.0,
            "profit_factor": round(profit_factor, 2),
            "mean_pct_return_%": round(sub["pct_return"].mean() * 100, 2),
            "trade_sharpe_like": round(_trade_sharpe_like(sub["pct_return"].dropna(), avg_hold), 3),
            "avg_holding_days": round(avg_hold, 1),
            "avg_confidence": round(sub[conf_col].mean(), 1)
            if conf_col in sub.columns and sub[conf_col].notna().any() else None,
            "avg_duration_days": round(sub[dur_col].mean(), 1)
            if dur_col in sub.columns and sub[dur_col].notna().any() else None,
        }
    out = pd.DataFrame(rows).T
    order = [l for l in REGIME_ORDER if l in out.index] + [
        l for l in out.index if l not in REGIME_ORDER
    ]
    return out.loc[order]


def transition_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """8-pilot-stock-only cross-tab: entered in regime X, exited in regime Y
    -> how did that combination perform. Only rows where BOTH entry and exit
    regime are known (i.e. the 8 pilot stocks)."""
    df = trades_df.dropna(subset=["regime_at_entry", "regime_at_exit"]).copy()
    if len(df) == 0:
        return pd.DataFrame()
    rows = {}
    for (entry_r, exit_r), sub in df.groupby(["regime_at_entry", "regime_at_exit"]):
        n = len(sub)
        wins = sub[sub["pnl"] > 0]
        rows[f"{entry_r} -> {exit_r}"] = {
            "trades": n,
            "total_pnl": round(sub["pnl"].sum(), 2),
            "win_rate_%": round(len(wins) / n * 100, 1) if n else 0.0,
            "mean_pct_return_%": round(sub["pct_return"].mean() * 100, 2),
            "avg_holding_days": round(sub["holding_days"].mean(), 1),
        }
    return pd.DataFrame(rows).T.sort_values("total_pnl", ascending=False)


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")

    print(f"拉取 {START} ~ {END} 纯美股（config.WATCHLIST_EUROPE_US，"
          f"{len(config.WATCHLIST_EUROPE_US)}只）历史数据...")
    with contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(config.WATCHLIST_EUROPE_US, START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    print(f"\n为 {len(PILOT_CODES)} 只试点股票构建因果（无未来函数）regime序列（regime/confidence/duration）...")
    regime_lookup = build_regime_lookup(PILOT_CODES, START, END)

    print(f"\n跑纯基线（regime_lookup 已传入用于打标签，enable_bear_gate=False，"
          f"不改变任何交易决策——应与V2.6/V2.7锁定基线数字一致）...")
    with contextlib.redirect_stdout(log_f):
        result = bp.simulate_from_prepared(prepared, cash=CASH, currency="$", regime_lookup=regime_lookup)
    log_f.close()

    df_summary = _build_table(result, prepared)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    print(f"\n{'='*100}")
    print("全期汇总（应与 _yearly_report_v27_locked_result.csv 一致 -- 验证tagging未改变任何交易决策）")
    print(f"{'='*100}")
    print(df_summary.loc[["全期汇总"]].to_string())

    trades_df = _sell_trades_df(result)
    print(f"\n全部平仓交易数={len(trades_df)}，其中8只试点股票（有regime标签）"
          f"={trades_df['regime_at_entry'].notna().sum()}")

    print(f"\n{'='*100}")
    print("按【入场时 Regime】分组（回答：策略在哪个Regime下开仓，后续表现如何）")
    print(f"{'='*100}")
    entry_bd = regime_breakdown(trades_df, "regime_at_entry")
    print(entry_bd.to_string())

    print(f"\n{'='*100}")
    print("按【离场时 Regime】分组（回答：在哪个Regime下平仓，盈亏表现如何）")
    print(f"{'='*100}")
    exit_bd = regime_breakdown(trades_df, "regime_at_exit")
    print(exit_bd.to_string())

    print(f"\n{'='*100}")
    print("【入场Regime -> 离场Regime】转移组合表现（仅8只试点股票）")
    print(f"{'='*100}")
    trans_bd = transition_breakdown(trades_df)
    print(trans_bd.to_string() if len(trans_bd) else "(无数据)")

    print(f"\n{'='*100}")
    print("8只试点股票 vs 其余131只（N/A）—— 逐股票细分（仅入场Regime非N/A的交易）")
    print(f"{'='*100}")
    pilot_only = trades_df[trades_df["regime_at_entry"].notna()]
    for code in PILOT_CODES:
        sub = pilot_only[pilot_only["code"] == code]
        if len(sub) == 0:
            print(f"  {code}: 0笔（无trained model或期间内未产生任何已标注入场regime的交易）")
            continue
        wins = sub[sub["pnl"] > 0]
        print(f"  {code}: {len(sub)}笔, 胜率={len(wins)/len(sub)*100:.1f}%, "
              f"总盈亏={sub['pnl'].sum():.2f}, 平均持仓={sub['holding_days'].mean():.1f}天")

    df_summary.to_csv("_regime_analytics_v27_summary.csv", encoding="utf-8-sig")
    trades_df.to_csv("_regime_analytics_v27_trades.csv", index=False, encoding="utf-8-sig")
    entry_bd.to_csv("_regime_analytics_v27_by_entry_regime.csv", encoding="utf-8-sig")
    exit_bd.to_csv("_regime_analytics_v27_by_exit_regime.csv", encoding="utf-8-sig")
    if len(trans_bd):
        trans_bd.to_csv("_regime_analytics_v27_transitions.csv", encoding="utf-8-sig")
    print("\n结果已保存：_regime_analytics_v27_*.csv")


if __name__ == "__main__":
    main()
