"""
analytics/replacement_stability_report.py -- v2.4 Replacement Stabilization
Layer（RSL）回归验证报告：在v2.3主动置换机制之外叠加冷却/预算/自适应门槛/
新增LOW_PARTIAL+WEAK_FULL两档优先级/稳定性保护后，对82只纯美股窄池的实际
效果。v2.4不追求提升收益上限，重点看PRD"九、回归验证"要求的5个稳定性指标。

对比三组（同一份 prepared 数据，只跑一次数据拉取）：
  v2.0（旧链式过滤器——复用 analytics/portfolio_gap_study.py::_run_v20，
        额外强制关闭ENABLE_ACTIVE_REPLACEMENT/ENABLE_REPLACEMENT_STABILIZATION，
        因为v2.0所在的历史时期这两个机制都还不存在）
  v2.3（开启主动置换，关闭RSL —— config.ENABLE_REPLACEMENT_STABILIZATION=False，
        100%复现v2.3发布时的行为）
  v2.4（开启主动置换 + RSL —— 本次新增）

产出 PRD 第九节的5个核心指标（v2.3 vs v2.4）：
  1. Replacement Count（预期 v2.4 相对 v2.3 下降20~40%）
  2. Replacement Success Rate（预期 v2.4 相对 v2.3 提升到45~60%）
  3. Sharpe（组合层面，波动应该下降）
  4. Max Drawdown（组合层面，结构应该更稳定）
  5. Tail Contribution Ratio（Top10%盈利交易占总正PnL比例——避免极端依赖，
     口径见 _tail_contribution_ratio()，不是PRD原文定义的正式统计量，是
     针对"收益是否集中在少数极端事件"这个问题设计的一个近似代理指标）

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python analytics/replacement_stability_report.py
"""
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest_portfolio as bp
import config
from analytics.capacity_manager_report import _replacement_stats
from analytics.portfolio_gap_study import _run_v20

START = "2015-01-01"
END   = "2026-07-03"
CASH  = 50_000.0
POOL  = config.WATCHLIST_EUROPE_US

LOG_PATH = os.path.join(os.path.dirname(__file__), "_replacement_stability_report.log")
OUT_DIR  = os.path.dirname(__file__)


def _run(prepared, enable_replacement: bool, enable_rsl: bool):
    orig_rep = config.ENABLE_ACTIVE_REPLACEMENT
    orig_rsl = config.ENABLE_REPLACEMENT_STABILIZATION
    config.ENABLE_ACTIVE_REPLACEMENT = enable_replacement
    config.ENABLE_REPLACEMENT_STABILIZATION = enable_rsl
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            r = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")
    finally:
        config.ENABLE_ACTIVE_REPLACEMENT = orig_rep
        config.ENABLE_REPLACEMENT_STABILIZATION = orig_rsl
    return r


def _run_v20_authentic(prepared):
    """v2.0所在的历史时期v2.3/v2.4都还不存在，强制关闭，不依赖"总分永远是
    100分打平所以置换永远不会触发"这个巧合来保证v2.0的纯净性。"""
    orig_rep = config.ENABLE_ACTIVE_REPLACEMENT
    orig_rsl = config.ENABLE_REPLACEMENT_STABILIZATION
    config.ENABLE_ACTIVE_REPLACEMENT = False
    config.ENABLE_REPLACEMENT_STABILIZATION = False
    try:
        return _run_v20(prepared)
    finally:
        config.ENABLE_ACTIVE_REPLACEMENT = orig_rep
        config.ENABLE_REPLACEMENT_STABILIZATION = orig_rsl


def _tail_contribution_ratio(trade_log, top_pct: float = 0.10) -> float:
    """Top top_pct(默认10%)盈利交易的PnL之和 / 全部正PnL之和。越接近1，说明
    组合收益越依赖少数极端事件；越低说明盈利来源越分散。"""
    pnls = sorted((t["pnl"] for t in trade_log
                   if t["side"] == "SELL" and (t.get("pnl") or 0) > 0), reverse=True)
    if not pnls:
        return 0.0
    total_positive = sum(pnls)
    n_top = max(1, round(len(pnls) * top_pct))
    top_sum = sum(pnls[:n_top])
    return top_sum / total_positive if total_positive > 0 else 0.0


def main():
    open(LOG_PATH, "w", encoding="utf-8").close()

    print(f"拉取 {START} ~ {END}，纯美股池 config.WATCHLIST_EUROPE_US（{len(POOL)}只）...")
    with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        prepared = bp._prepare_backtest_data(POOL, START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    print("模拟 v2.0（旧链式过滤器）...")
    r_v20 = _run_v20_authentic(prepared)
    print("模拟 v2.3（开启主动置换，关闭RSL）...")
    r_v23 = _run(prepared, enable_replacement=True, enable_rsl=False)
    print("模拟 v2.4（开启主动置换 + RSL）...")
    r_v24 = _run(prepared, enable_replacement=True, enable_rsl=True)

    print("\n" + "=" * 100)
    print("一、组合层面对比")
    print("=" * 100)
    print(f"{'指标':<12}{'v2.0':>12}{'v2.3(RSL关)':>14}{'v2.4(RSL开)':>14}")
    print(f"{'总收益%':<12}{r_v20['total_return_pct']:>12.2f}"
          f"{r_v23['total_return_pct']:>14.2f}{r_v24['total_return_pct']:>14.2f}")
    print(f"{'Sharpe':<12}{r_v20['sharpe']:>12.3f}"
          f"{r_v23['sharpe']:>14.3f}{r_v24['sharpe']:>14.3f}")
    print(f"{'最大回撤%':<12}{r_v20['max_drawdown_pct']:>12.2f}"
          f"{r_v23['max_drawdown_pct']:>14.2f}{r_v24['max_drawdown_pct']:>14.2f}")
    print(f"{'交易数':<12}{r_v20['num_trades']:>12}"
          f"{r_v23['num_trades']:>14}{r_v24['num_trades']:>14}")

    print("\n" + "=" * 100)
    print("二、PRD第九节核心指标：v2.3(RSL关) vs v2.4(RSL开)")
    print("=" * 100)
    s23 = _replacement_stats(r_v23)
    s24 = _replacement_stats(r_v24)

    count_change = (f"{(s24['count'] - s23['count']) / s23['count'] * 100:+.1f}%"
                    if s23["count"] else "N/A")
    print(f"1. Replacement Count        v2.3={s23['count']:>4}   v2.4={s24['count']:>4}   变化={count_change}")
    print(f"2. Replacement Success Rate v2.3={s23.get('success_rate_pct')}%   "
          f"v2.4={s24.get('success_rate_pct')}%")
    print(f"3. Sharpe                   v2.3={r_v23['sharpe']:.3f}   v2.4={r_v24['sharpe']:.3f}")
    print(f"4. Max Drawdown%            v2.3={r_v23['max_drawdown_pct']:.2f}   "
          f"v2.4={r_v24['max_drawdown_pct']:.2f}")

    tcr23 = _tail_contribution_ratio(r_v23["trade_log"])
    tcr24 = _tail_contribution_ratio(r_v24["trade_log"])
    print(f"5. Tail Contribution Ratio(Top10% wins) v2.3={tcr23*100:.1f}%   v2.4={tcr24*100:.1f}%")

    print("\n" + "=" * 100)
    print("三、v2.4新增字段分布（replacement_type breakdown + 拦截原因）")
    print("=" * 100)
    if s24["count"] > 0:
        type_counts = s24["detail"]["replacement_type"].value_counts()
        for t, n in type_counts.items():
            print(f"  {t:<28}{n}")
    else:
        print("  v2.4没有发生任何置换。")

    cooldown_blocked_count = sum(1 for b in r_v24["capacity_blocks"] if b.get("cooldown_blocked"))
    budget_blocked_count   = sum(1 for b in r_v24["capacity_blocks"] if b.get("budget_blocked"))
    print(f"\n因Cooldown/同日链式阻断被拒绝的容量拦截次数: {cooldown_blocked_count}")
    print(f"因Budget耗尽被拒绝的容量拦截次数: {budget_blocked_count}")

    if s23["count"] > 0:
        s23["detail"].to_csv(os.path.join(OUT_DIR, "replacement_stability_v23_detail.csv"),
                              index=False, encoding="utf-8-sig")
    if s24["count"] > 0:
        s24["detail"].to_csv(os.path.join(OUT_DIR, "replacement_stability_v24_detail.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"\n逐笔明细已保存: replacement_stability_v23_detail.csv / replacement_stability_v24_detail.csv"
          f"（存在于 {OUT_DIR}，仅当对应组有置换发生时才会写出）")


if __name__ == "__main__":
    main()
