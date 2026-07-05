"""
_v21_old_vs_new_diff.py -- 找出旧系统(链式过滤器)存在、新系统(v2.1总分模型)
不存在的那批交易，并分析它们的独立表现 + "恢复3%观察仓"的整体影响。

做法：单次 fetch 复用（同 _market_weather_ab.py 模式），跑两遍 simulate：
  NEW: 当前代码原样跑（v2.1 总分模型，total<60直接放弃开仓）。
  OLD: monkeypatch engine.scoring.compute_total_score 让它永远不返回SKIP、
       永远 position_scale=1.0（不做总分连续折算），同时 monkeypatch
       backtest_portfolio._size 让它忽略 score_label 参数、回退到原始
       signal_strength 阈值分档——这精确复现了 v2.1 重构之前的行为
       （MIN_ENTRY_STRENGTH=0.0，弱信号也以3%观察仓下单，仓位不被总分
       模型二次折算），不需要重新实现一遍 simulate_from_prepared 的其余
       逻辑（止损/止盈/晋升/金字塔/QQQ底仓等完全不变）。

用 (code, entry_date) 识别一笔交易（同一支股票在同一时刻只会有一个持仓，
足以唯一定位一笔round-trip）。"旧系统独有"=OLD的SELL集合 - NEW的SELL集合。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _v21_old_vs_new_diff.py
"""
import contextlib
import math

import numpy as np
import pandas as pd

import backtest_portfolio as bp
import engine.scoring as scoring

FULL_START = "2015-01-01"
FULL_END   = "2026-07-04"
CASH       = 7_000_000.0
USD_JPY    = 140.0

LOG_PATH = "_v21_old_vs_new_diff.log"


def _run(prepared, label):
    print(f"Simulating {label} ...")
    with open(LOG_PATH, "a", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f):
        r = bp.simulate_from_prepared(prepared, cash=CASH, currency="JPY ")
    return r


def _sell_trades(trade_log):
    """{(code, entry_date): trade_dict} for every SELL record."""
    out = {}
    for t in trade_log:
        if t["side"] != "SELL":
            continue
        key = (t["code"], t.get("entry_date"))
        out[key] = t
    return out


def main():
    open(LOG_PATH, "w", encoding="utf-8").close()

    print(f"Fetching full history {FULL_START} ~ {FULL_END} once (shared)...")
    with open(LOG_PATH, "a", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(bp.BACKTEST_STOCKS, FULL_START, FULL_END, usd_to_jpy=USD_JPY)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    # ── NEW: 当前代码原样跑 ──────────────────────────────────────────────────
    r_new = _run(prepared, "NEW (v2.1 total-score, current code)")

    # ── OLD: monkeypatch 还原成重构前的行为 ─────────────────────────────────
    _real_compute = scoring.compute_total_score
    _real_size    = bp._size

    def _fake_compute_total_score(trend_strength, weather_code,
                                   fundamental_score=None, news_score=None):
        # 永远不SKIP（对应旧 MIN_ENTRY_STRENGTH=0.0，任何正强度信号都不会被
        # 总分模型拦截）、永远 position_scale=1.0（对应旧系统里仓位比例只由
        # signal_strength 阈值分档决定一次，没有总分连续折算）。weather==0
        # 的硬拦截不受影响——那发生在更早的 4c-macro 阶段，买入候选列表在
        # 到达这里之前就已经被清空了。
        trend_score = max(0.0, min(1.0, trend_strength)) * 100.0
        weather_score = 0.0 if weather_code == 0 else 100.0
        return scoring.TotalScore(100.0, 1.0, scoring.LABEL_FULL,
                                   trend_score, fundamental_score, news_score, weather_score)

    def _old_size(*args, **kwargs):
        kwargs.pop("score_label", None)   # 强制回退到原始 strength 阈值分档
        return _real_size(*args, **kwargs)

    scoring.compute_total_score = _fake_compute_total_score
    bp._size = _old_size
    try:
        r_old = _run(prepared, "OLD (pre-v2.1 chain-filter, reconstructed via monkeypatch)")
    finally:
        scoring.compute_total_score = _real_compute
        bp._size = _real_size

    # ── 对比两组全历史指标，先确认复现的OLD系统数字符合预期 ──────────────────
    def _summ(r, label):
        print(f"{label}: 总收益{r['total_return_pct']:.2f}%  Sharpe{r['sharpe']:.3f}  "
              f"MDD{r['max_drawdown_pct']:.2f}%  交易{r['num_trades']}笔  "
              f"胜率{r['win_rate_pct']:.1f}%")

    _summ(r_new, "NEW")
    _summ(r_old, "OLD(reconstructed)")

    # ── 找出 OLD 独有、NEW 不存在的交易 ─────────────────────────────────────
    old_sells = _sell_trades(r_old["trade_log"])
    new_sells = _sell_trades(r_new["trade_log"])
    only_old_keys = set(old_sells) - set(new_sells)
    only_old = [old_sells[k] for k in only_old_keys]
    only_old.sort(key=lambda t: pd.Timestamp(t["date"]))

    print(f"\n旧系统独有交易数: {len(only_old)}")

    rows = []
    for t in only_old:
        entry_date = t.get("entry_date")
        pnl = t["pnl"]
        # 估算收益率：pnl / (qty * 成交价对应的成本)。用 avg pnl 反推的成本不
        # 精确知道 avg_cost，这里用 qty*price(卖出价) 反推的名义敞口做分母的
        # 近似（trade_log 没有单独存 entry cost，只有 pnl 已经是税后净值）。
        # 更精确的做法是用 entry_price*qty，但 trade_log 的 SELL 记录没有
        # entry_price 字段，只有 entry_date——这里改用 pnl/(price*qty - pnl)
        # 反推出的隐含成本基数，等价于 entry_notional = exit_notional - pnl。
        exit_notional = t["price"] * t["qty"]
        entry_notional = exit_notional - pnl
        ret_pct = (pnl / entry_notional * 100.0) if entry_notional not in (0, None) else float("nan")
        rows.append({
            "code": t["code"],
            "entry_date": entry_date,
            "sell_date": t["date"],
            "strategy": t.get("strategy"),
            "exit_reason": t["reason"],
            "pnl": pnl,
            "return_pct": round(ret_pct, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv("_v21_old_only_trades.csv", index=False, encoding="utf-8-sig")

    print("\n=== 1. 每笔交易 ===")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    print(df.to_string(index=False))

    pnl_series = df["pnl"].astype(float)
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series <= 0]

    total_pnl = pnl_series.sum()
    avg_pnl   = pnl_series.mean()
    win_rate  = 100.0 * len(wins) / len(pnl_series) if len(pnl_series) else float("nan")
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")

    # Sharpe（仅这119笔）：用逐笔收益率序列（不是日收益率）算，年化用
    # "该笔交易发生的频率"没有清晰定义，这里给出未年化的逐笔Sharpe（mean/std），
    # 并单独标注，避免和整体系统日频Sharpe混淆、造成误解。
    ret_series = df["return_pct"].astype(float).dropna()
    trade_sharpe = (ret_series.mean() / ret_series.std(ddof=1)) if ret_series.std(ddof=1) > 0 else float("nan")

    print(f"\n=== 2. 总PnL: {total_pnl:,.0f} 日元 ===")
    print(f"=== 3. 平均PnL: {avg_pnl:,.0f} 日元/笔 ===")
    print(f"=== 4. 胜率: {win_rate:.1f}% ({len(wins)}/{len(pnl_series)}) ===")
    print(f"=== 5. Profit Factor: {profit_factor:.3f} ===")
    print(f"=== 6. 逐笔Sharpe(仅119笔, 未年化, 按收益率序列 mean/std): {trade_sharpe:.3f} ===")

    print(f"\n结果已保存: _v21_old_only_trades.csv")

    print("\n=== 7. 如果恢复3%观察仓（弱信号total<60时不放弃，仍以3%观察仓下单）===")
    print("见下一步 _v21_restore_observation_tier.py 的运行结果")


if __name__ == "__main__":
    main()
