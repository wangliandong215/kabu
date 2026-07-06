"""
_yearly_report.py -- year-by-year performance breakdown for the current
LOCKED default configuration (V2 market weather + dynamic allocator,
2026-07-04), matching the report format the user requested: 收益率/QQQ/
简单Alpha/期末金额/交易数/胜率/Sharpe/Sortino/MDD/Calmar/Profit Factor/
Beta/CAPM Alpha, per calendar year 2015-2026 + full-period summary.

Pure-USD run ($50,000 starting cash, no JPY conversion) over
config.WATCHLIST_EUROPE_US only -- mixing JPY-priced JP.* stocks into a
USD cash pool without an FX rate would be economically meaningless, so this
report intentionally does NOT use the full mixed WATCHLIST like
backtest_portfolio.py's default multi-region JPY runs do.

Methodology notes (no single canonical definition exists for some of these
across finance libraries, so being explicit here):
  - 收益率 / QQQ / 简单Alpha: RAW (non-annualized) return over the calendar
    year (or partial year for 2026). Alpha = portfolio - QQQ, both raw.
  - Sharpe / Sortino: annualized from that period's own daily returns using
    the standard sqrt(252) daily-frequency scalar (not scaled by how many
    days are actually in the slice -- that would double-count with the
    period length, which is already reflected in the input return series).
    Sortino's downside deviation only uses days with a negative return.
  - MDD: peak-to-trough within that calendar year only, with the year's
    starting equity (prior year's close, or initial cash for the first
    year) as the day-0 anchor -- so a drawdown can't be hidden just because
    the year happened to start already below a prior peak.
  - Calmar: annualized return for that specific period (compounded up to a
    365-day year, so partial years like 2026 are extrapolated) divided by
    that period's raw MDD magnitude. This is why Calmar for a partial year
    looks larger than a naive return/MDD ratio would suggest.
  - Beta / CAPM Alpha: Beta = Cov(daily port ret, daily QQQ ret) / Var(daily
    QQQ ret) over that period's daily returns. CAPM Alpha = annualized
    portfolio return - Beta * annualized QQQ return, assuming a 0%
    risk-free rate (this codebase has no risk-free-rate series to draw on;
    treat CAPM Alpha here as an approximation, not a precise Jensen's alpha).
  - Profit Factor: sum(winning SELL trade P&L) / abs(sum(losing SELL trade
    P&L)) for trades closed within that calendar year.

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _yearly_report.py
"""
import contextlib

import numpy as np
import pandas as pd

import backtest_portfolio as bp
import config

START = "2015-01-01"
END   = "2026-07-03"
CASH  = 50_000.0

LOG_PATH = "_yearly_report.log"


def _annualize(period_return: float, days: int) -> float:
    if days <= 0:
        return 0.0
    return (1 + period_return) ** (365.0 / days) - 1


def _sharpe(daily: np.ndarray) -> float:
    if len(daily) < 2 or daily.std() == 0:
        return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(252))


def _sortino(daily: np.ndarray) -> float:
    downside = daily[daily < 0]
    if len(daily) < 2 or len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(daily.mean() / downside.std() * np.sqrt(252))


def _mdd(values: np.ndarray) -> float:
    roll_max = np.maximum.accumulate(values)
    dd = (values - roll_max) / roll_max
    return float(dd.min())


def _period_metrics(eq: pd.Series, qqq: pd.Series, mask: np.ndarray,
                     start_equity: float, start_qqq: float,
                     trades_df: pd.DataFrame, year_label) -> dict:
    eq_y = eq[mask]
    qqq_y = qqq[mask]
    if len(eq_y) == 0:
        return None

    end_equity = float(eq_y.iloc[-1])
    end_qqq = float(qqq_y.iloc[-1])
    days = (eq_y.index[-1] - eq_y.index[0]).days + 1

    port_ret = end_equity / start_equity - 1
    qqq_ret = end_qqq / start_qqq - 1
    simple_alpha = port_ret - qqq_ret

    eq_ext = pd.concat([pd.Series([start_equity]), eq_y.reset_index(drop=True)])
    qqq_ext = pd.concat([pd.Series([start_qqq]), qqq_y.reset_index(drop=True)])
    daily_port = eq_ext.pct_change().dropna().values
    daily_qqq = qqq_ext.pct_change().dropna().values

    mdd = _mdd(eq_ext.values)
    sharpe = _sharpe(daily_port)
    sortino = _sortino(daily_port)

    ann_port = _annualize(port_ret, days)
    ann_qqq = _annualize(qqq_ret, days)
    calmar = ann_port / abs(mdd) if mdd != 0 else float("nan")

    if daily_qqq.std() > 0 and len(daily_qqq) > 1:
        beta = float(np.cov(daily_port, daily_qqq)[0, 1] / np.var(daily_qqq))
    else:
        beta = float("nan")
    capm_alpha = ann_port - beta * ann_qqq if not np.isnan(beta) else float("nan")

    if len(trades_df) == 0 or year_label == "ALL":
        yr_trades = trades_df
    else:
        yr_trades = trades_df[trades_df["year"] == year_label]
    n_trades = len(yr_trades)
    if n_trades:
        wins = yr_trades[yr_trades["pnl"] > 0]
        win_rate = len(wins) / n_trades
        gross_win = yr_trades.loc[yr_trades["pnl"] > 0, "pnl"].sum()
        gross_loss = yr_trades.loc[yr_trades["pnl"] < 0, "pnl"].sum()
        profit_factor = (gross_win / abs(gross_loss)) if gross_loss != 0 else float("inf")
    else:
        win_rate = 0.0
        profit_factor = float("nan")

    return {
        "收益率%": round(port_ret * 100, 2),
        "QQQ%": round(qqq_ret * 100, 2),
        "简单Alpha%": round(simple_alpha * 100, 2),
        "期末金额": round(end_equity, 2),
        "交易数": n_trades,
        "胜率%": round(win_rate * 100, 1),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "MDD%": round(mdd * 100, 2),
        "Calmar": round(calmar, 3) if not np.isnan(calmar) else None,
        "Profit Factor": round(profit_factor, 2) if np.isfinite(profit_factor) else None,
        "Beta": round(beta, 3) if not np.isnan(beta) else None,
        "CAPM Alpha%": round(capm_alpha * 100, 2) if not np.isnan(capm_alpha) else None,
    }, end_equity, end_qqq


def _build_table(result: dict, prepared: dict) -> pd.DataFrame:
    eq = result["equity_curve"]
    qqq = prepared["all_data"]["US.QQQ"]["close"].astype(float).reindex(eq.index).ffill()

    sell_trades = [t for t in result["trade_log"] if t["side"] == "SELL"]
    trades_df = pd.DataFrame(sell_trades)
    if len(trades_df):
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df["year"] = trades_df["date"].dt.year
        trades_df["pnl"] = trades_df["pnl"].astype(float)

    years = sorted(eq.index.year.unique())
    rows = {}
    start_equity = CASH
    start_qqq = float(qqq.iloc[0])   # first available QQQ close at/after START

    for y in years:
        mask = np.asarray(eq.index.year == y)
        metrics, end_equity, end_qqq = _period_metrics(
            eq, qqq, mask, start_equity, start_qqq, trades_df, y)
        if metrics is None:
            continue
        rows[str(y)] = metrics
        start_equity = end_equity
        start_qqq = end_qqq

    # 全期汇总
    full_mask = np.ones(len(eq), dtype=bool)
    full_metrics, _, _ = _period_metrics(
        eq, qqq, full_mask, CASH, float(qqq.iloc[0]), trades_df, "ALL")
    if len(trades_df):
        n_all = len(trades_df)
        wins_all = trades_df[trades_df["pnl"] > 0]
        full_metrics["交易数"] = n_all
        full_metrics["胜率%"] = round(len(wins_all) / n_all * 100, 1) if n_all else 0.0
    rows["全期汇总"] = full_metrics

    return pd.DataFrame(rows).T


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")

    print(f"拉取 {START} ~ {END} 纯美股（config.WATCHLIST_EUROPE_US）历史数据...")
    with contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(config.WATCHLIST_EUROPE_US, START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    # ── V2（当前锁定配置：市场天气模块一/二全部生效）───────────────────────────
    print(f"模拟 V2 当前锁定配置（MARKET_WEATHER_RISK_MULTIPLIER={config.MARKET_WEATHER_RISK_MULTIPLIER}，"
          f"cash=${CASH:,.0f}）...")
    with contextlib.redirect_stdout(log_f):
        result_v2 = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")
    if not result_v2:
        print("NO RESULT (V2) -- aborting")
        return
    df_v2 = _build_table(result_v2, prepared)

    # ── V1（升级前系统：市场天气/动态风险预算完全关闭，只保留原有机制——
    #    ATR=5.5跟踪止损/V_REVERSAL_MODE/QQQ Beta底仓/qqq_macro_halt硬熔断/
    #    半Kelly/TRENDING_EARLY试错仓位等）。用同一份 prepared 数据，只把
    #    qqq_weather 系列整体替换成"永远状态2"，这样 weather_code 永远不等
    #    于0（buy_candidates 的硬拦截门永远不会因为市场天气触发）、
    #    RISK_PER_TRADE_PCT 的乘数永远是1.0（不缩放）——等价于完全没有
    #    Market Weather 模块，不需要重新拉取数据。 ─────────────────────────
    print("模拟 V1（升级前系统，禁用 Market Weather 模块一/二）...")
    prepared_v1 = dict(prepared)
    prepared_v1["qqq_weather"] = pd.Series(2, index=prepared["qqq_weather"].index)
    with contextlib.redirect_stdout(log_f):
        result_v1 = bp.simulate_from_prepared(prepared_v1, cash=CASH, currency="$")
    log_f.close()
    if not result_v1:
        print("NO RESULT (V1) -- aborting")
        return
    df_v1 = _build_table(result_v1, prepared_v1)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print(f"\n{'='*140}")
    print(f"V2（当前锁定配置） {START} ~ {END} 逐年完整指标（${CASH:,.0f}起点，纯美股 WATCHLIST_EUROPE_US）")
    print(f"{'='*140}")
    print(df_v2.to_string())
    df_v2.to_csv("_yearly_report_v2_result.csv", encoding="utf-8-sig")

    print(f"\n{'='*140}")
    print(f"V1（升级前系统，无 Market Weather） {START} ~ {END} 逐年完整指标（${CASH:,.0f}起点，纯美股）")
    print(f"{'='*140}")
    print(df_v1.to_string())
    df_v1.to_csv("_yearly_report_v1_result.csv", encoding="utf-8-sig")

    print(f"\n结果已保存: _yearly_report_v2_result.csv / _yearly_report_v1_result.csv")


if __name__ == "__main__":
    main()
