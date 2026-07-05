"""
analytics/capacity_manager_report.py -- v2.3 回归验证报告：Portfolio Capacity
Manager（主动容量置换）对82只纯美股窄池的实际效果。

对比两组（同一份 prepared 数据，唯一变量是 config.ENABLE_ACTIVE_REPLACEMENT）：
  v2.2（关闭主动置换，原有 Capacity Block 行为——上一版本默认状态）
  v2.3（开启主动置换，本次新增的 Portfolio Capacity Manager）

产出：
  - 总收益/Sharpe/MDD/交易数 对比，及相对 v2.0（208.55%）的最终位置
  - Crowding_Out_Strong_Signals_Count 前后对比（复用
    analytics/portfolio_gap_study.py 的口径）
  - NVDA/TSLA/AMD/AMZN 首次FULL建仓是否成功执行（不再被容量拦截）
  - Replacement Count / Success Rate / Average Score Delta /
    Average Holding Days Before Replacement / PnL Attribution

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python analytics/capacity_manager_report.py
"""
import contextlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest_portfolio as bp
import config
import engine.scoring as scoring
from analytics.portfolio_gap_study import analyze_crowding_out

START = "2015-01-01"
END   = "2026-07-03"
CASH  = 50_000.0
POOL  = config.WATCHLIST_EUROPE_US

WATCH_STOCKS = ["US.NVDA", "US.TSLA", "US.AMD", "US.AMZN"]

LOG_PATH = os.path.join(os.path.dirname(__file__), "_capacity_manager_report.log")
OUT_DIR  = os.path.dirname(__file__)


def _run(prepared, enable_replacement: bool):
    orig = config.ENABLE_ACTIVE_REPLACEMENT
    config.ENABLE_ACTIVE_REPLACEMENT = enable_replacement
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            r = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")
    finally:
        config.ENABLE_ACTIVE_REPLACEMENT = orig
    return r


def _first_full_buy(trade_log, code):
    buys = [t for t in trade_log
            if t["code"] == code and t["side"] == "BUY" and t.get("reason") == "SIGNAL"
            and t.get("score_label") == scoring.LABEL_FULL]
    return buys[0] if buys else None


def _replacement_stats(result: dict) -> dict:
    replacements = result["replacements"]
    if not replacements:
        return {"count": 0}

    # 把每次置换后新开的FULL仓位配对到它自己最终平仓时的pnl，用于success rate
    # 和PnL归因——按 (code, entry_date=replacement date) 唯一定位。
    sell_by_key = {}
    for t in result["trade_log"]:
        if t["side"] == "SELL" and t.get("entry_date") is not None:
            sell_by_key[(t["code"], t["entry_date"])] = t

    rows = []
    for rep in replacements:
        incoming_sell = sell_by_key.get((rep["replacement_to"], rep["date"]))
        incoming_pnl = incoming_sell["pnl"] if incoming_sell else None   # None = 仍持有中，未平仓
        rows.append({**rep, "incoming_final_pnl": incoming_pnl,
                     "incoming_closed": incoming_sell is not None})

    df = pd.DataFrame(rows)
    closed = df[df["incoming_closed"]]
    success_rate = (float((closed["incoming_final_pnl"] > 0).mean()) * 100
                    if len(closed) else None)

    return {
        "count": len(df),
        "avg_score_delta": round(float(df["score_delta"].mean()), 2),
        "avg_holding_days_before_replacement": round(
            float(df["holding_days_before_replacement"].mean()), 2),
        "n_closed": len(closed),
        "n_still_open_at_backtest_end": len(df) - len(closed),
        "success_rate_pct": round(success_rate, 1) if success_rate is not None else None,
        "victim_pnl_total": round(float(df["victim_pnl"].sum()), 2),
        "incoming_pnl_total_closed_only": round(float(closed["incoming_final_pnl"].sum()), 2)
                                            if len(closed) else 0.0,
        "detail": df,
    }


def main():
    open(LOG_PATH, "w", encoding="utf-8").close()

    print(f"拉取 {START} ~ {END}，纯美股池 config.WATCHLIST_EUROPE_US（{len(POOL)}只）...")
    with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        prepared = bp._prepare_backtest_data(POOL, START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    print("模拟 v2.2（关闭主动置换）...")
    r_v22 = _run(prepared, enable_replacement=False)
    print("模拟 v2.3（开启主动置换，Portfolio Capacity Manager）...")
    r_v23 = _run(prepared, enable_replacement=True)

    print("\n" + "=" * 100)
    print("一、组合层面对比")
    print("=" * 100)
    print(f"{'指标':<12}{'v2.0基线':>12}{'v2.2':>12}{'v2.3':>12}")
    print(f"{'总收益%':<12}{'208.55':>12}{r_v22['total_return_pct']:>12.2f}{r_v23['total_return_pct']:>12.2f}")
    print(f"{'Sharpe':<12}{'0.850':>12}{r_v22['sharpe']:>12.3f}{r_v23['sharpe']:>12.3f}")
    print(f"{'最大回撤%':<12}{'-18.72':>12}{r_v22['max_drawdown_pct']:>12.2f}{r_v23['max_drawdown_pct']:>12.2f}")
    print(f"{'交易数':<12}{'840':>12}{r_v22['num_trades']:>12}{r_v23['num_trades']:>12}")
    gap_to_v20 = 208.55 - r_v23["total_return_pct"]
    print(f"\nv2.3 相对 v2.0 的缺口: {gap_to_v20:+.2f}pp"
          f"（{'仍落后' if gap_to_v20 > 0 else '已反超'}）")

    print("\n" + "=" * 100)
    print("二、Crowding_Out_Strong_Signals_Count 前后对比")
    print("=" * 100)
    d2_v22 = analyze_crowding_out(r_v22, prepared["signals"], prepared["all_data"])
    d2_v23 = analyze_crowding_out(r_v23, prepared["signals"], prepared["all_data"])
    print(f"v2.2（关闭主动置换）: {d2_v22['distinct_codes']} 只不同股票被挡"
          f"（{d2_v22['Crowding_Out_Strong_Signals_Count']} 次原始事件）")
    print(f"v2.3（开启主动置换）: {d2_v23['distinct_codes']} 只不同股票被挡"
          f"（{d2_v23['Crowding_Out_Strong_Signals_Count']} 次原始事件）")
    if d2_v22["distinct_codes"]:
        reduction = (1 - d2_v23["distinct_codes"] / d2_v22["distinct_codes"]) * 100
        print(f"下降幅度: {reduction:.1f}%")

    print("\n" + "=" * 100)
    print("三、超级牛股首次FULL建仓验证")
    print("=" * 100)
    for code in WATCH_STOCKS:
        b22 = _first_full_buy(r_v22["trade_log"], code)
        b23 = _first_full_buy(r_v23["trade_log"], code)
        d22 = f"{b22['date'].date()} @ {b22['price']:.2f}" if b22 else "从未成功建仓(FULL)"
        d23 = f"{b23['date'].date()} @ {b23['price']:.2f}" if b23 else "从未成功建仓(FULL)"
        earlier = ""
        if b22 and b23 and b23["date"] < b22["date"]:
            earlier = "  <- v2.3更早建仓"
        elif b22 and b23 and b23["date"] == b22["date"]:
            earlier = "  (相同日期，本来就没被挡)"
        print(f"{code:10s}  v2.2首次FULL: {d22:28s}  v2.3首次FULL: {d23}{earlier}")

    print("\n" + "=" * 100)
    print("四、主动置换统计报告")
    print("=" * 100)
    stats = _replacement_stats(r_v23)
    if stats["count"] == 0:
        print("v2.3没有发生任何主动置换。")
    else:
        print(f"Replacement Count: {stats['count']}")
        print(f"Average Replacement Score Delta: {stats['avg_score_delta']}")
        print(f"Average Holding Days Before Replacement: {stats['avg_holding_days_before_replacement']}")
        print(f"已平仓 / 回测结束时仍持有: {stats['n_closed']} / {stats['n_still_open_at_backtest_end']}")
        if stats["success_rate_pct"] is not None:
            print(f"Replacement Success Rate（新FULL仓最终盈利的比例，仅统计已平仓）: "
                  f"{stats['success_rate_pct']}%")
        print(f"PnL Attribution -- 被换出观察仓的已实现盈亏合计: {stats['victim_pnl_total']:,.2f}")
        print(f"PnL Attribution -- 新FULL仓（已平仓部分）已实现盈亏合计: "
              f"{stats['incoming_pnl_total_closed_only']:,.2f}")
        net = stats["victim_pnl_total"] + stats["incoming_pnl_total_closed_only"]
        print(f"净归因（观察仓平仓损益 + 新FULL仓已平仓损益）: {net:,.2f}")
        stats["detail"].to_csv(
            os.path.join(OUT_DIR, "capacity_manager_replacements.csv"),
            index=False, encoding="utf-8-sig")
        print(f"\n逐笔明细已保存: {os.path.join(OUT_DIR, 'capacity_manager_replacements.csv')}")


if __name__ == "__main__":
    main()
