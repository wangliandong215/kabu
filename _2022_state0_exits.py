"""
_2022_state0_exits.py -- 导出 2022 年里，在 Market Weather 状态0（危机/全面
禁买）生效当天发生的所有 SELL 记录。

明确口径（用户已确认，2026-07-04）：state0 只挡新开仓，从不强平已持仓
（v1.0-RELEASE-FINAL 锁定设计，"强平已持仓"这条路此前已被用户否决过）。
这里导出的不是"被state0强制平仓"的交易——这类交易在当前设计下根本不存在
——而是"state0生效当天，已持仓位按自己正常的止损/止盈/趋势反转规则自然
离场"的记录，用于观察2022年那种阴跌熊市里，已持仓位在危机期间的微观止损
行为规律（是不是集中在某几个策略/某几周、止损幅度分布如何）。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _2022_state0_exits.py
"""
import pandas as pd

import backtest_portfolio as bp

FULL_START = "2015-01-01"   # 从更早开始拉取，让 MA200/波动率基准提前预热
FULL_END   = "2026-07-03"
CASH       = 7_000_000.0
USD_JPY    = 140.0

OUT_CSV = "_2022_state0_exits.csv"


def main():
    print(f"Fetching full history {FULL_START} ~ {FULL_END} "
          f"(v1.1-RELEASE-FINAL locked config, no overrides)...")
    prepared = bp._prepare_backtest_data(bp.BACKTEST_STOCKS, FULL_START, FULL_END, usd_to_jpy=USD_JPY)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    print("Simulating...")
    r = bp.simulate_from_prepared(prepared, cash=CASH, currency="JPY ")
    if not r:
        print("Simulation returned no result -- aborting")
        return

    qqq_weather = prepared["qqq_weather"]
    trade_log   = r["trade_log"]

    rows = []
    for t in trade_log:
        if t["side"] != "SELL":
            continue
        sell_date = pd.Timestamp(t["date"])
        if sell_date.year != 2022:
            continue
        weather_code = int(qqq_weather.get(sell_date, 1))
        if weather_code != 0:
            continue

        entry_date = t.get("entry_date")
        held_days = (
            (sell_date - pd.Timestamp(entry_date)).days
            if entry_date is not None else None
        )
        rows.append({
            "code": t["code"],
            "entry_date": entry_date,
            "sell_date": t["date"],
            "held_days": held_days,
            "strategy": t.get("strategy"),
            "exit_reason": t["reason"],
            "qty": t["qty"],
            "sell_price": t["price"],
            "pnl": t["pnl"],
            "fee": t.get("fee"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("2022 年内，state0 生效当天没有任何 SELL 记录（该情形可能确实没有发生）。")
        return

    df = df.sort_values("sell_date")
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n共 {len(df)} 笔，2022年 state0 期间自然离场记录：")
    print(f"  按 exit_reason 分布：\n{df['exit_reason'].value_counts().to_string()}")
    print(f"  按 strategy 分布：\n{df['strategy'].value_counts().to_string()}")
    print(f"  pnl 合计: {df['pnl'].sum():,.0f}  pnl 中位数: {df['pnl'].median():,.0f}")
    print(f"\n结果已保存: {OUT_CSV}")


if __name__ == "__main__":
    main()
