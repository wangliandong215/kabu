"""
analytics/weak_full_standalone_report.py -- v2.4（本轮，2026-07-05）回归验证：
独立 WEAK_FULL 通道（config.REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED，不带
v2.4 RSL的cooldown/预算/自适应门槛/Stability Score）对82只纯美股窄池的效果。

对比两组（同一份 prepared 数据，唯一变量是
config.REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED，RSL全程保持关闭 =
config.ENABLE_REPLACEMENT_STABILIZATION=False，即当前生产锁定的v2.3行为）：
  基线（WEAK_FULL关闭，纯v2.3 OBSERVATION_EVICT）
  新增（WEAK_FULL开启，OBSERVATION优先，找不到候选时才看WEAK_FULL）

产出：
  - 总收益/Sharpe/最大回撤/交易数/平均持仓天数 对比
  - Replacement Attempts / Success / Blocked / Average Score Difference /
    Average Holding Days Before Replacement（仅新增组，用于评估
    REPLACEMENT_MARGIN 是否合理）

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python analytics/weak_full_standalone_report.py
"""
import contextlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest_portfolio as bp
import config

START = "2015-01-01"
END   = "2026-07-03"
CASH  = 50_000.0
POOL  = config.WATCHLIST_EUROPE_US

LOG_PATH = os.path.join(os.path.dirname(__file__), "_weak_full_standalone_report.log")
OUT_DIR  = os.path.dirname(__file__)


def _run(prepared, standalone_weak_full: bool):
    orig = config.REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED
    config.REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED = standalone_weak_full
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            r = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")
    finally:
        config.REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED = orig
    return r


def _avg_holding_days(trade_log) -> float:
    days = [(t["date"] - t["entry_date"]).days
            for t in trade_log
            if t["side"] == "SELL" and t.get("strategy") != "core_etf"
            and t.get("entry_date") is not None]
    return round(sum(days) / len(days), 2) if days else float("nan")


def _replacement_margin_stats(result: dict) -> dict:
    attempts = result.get("replacement_attempts", [])
    if not attempts:
        return {"attempts": 0}
    df = pd.DataFrame(attempts)
    success = df[df["decision"] == "REPLACE"]
    blocked = df[df["decision"] == "KEEP"]
    return {
        "attempts": len(df),
        "success": len(success),
        "blocked": len(blocked),
        "avg_score_difference": round(float(df["score_difference"].mean()), 2),
        "avg_holding_days_success": (
            round(float(_holding_days_for(result, success)), 2) if len(success) else None),
        "by_type": success["replacement_type"].value_counts().to_dict() if len(success) else {},
        "detail": df,
    }


def _holding_days_for(result: dict, success_df) -> float:
    reps = {(r["date"], r["replacement_to"]): r["holding_days_before_replacement"]
            for r in result["replacements"]}
    vals = [reps.get((row["date"], row["incoming_code"]))
            for _, row in success_df.iterrows()]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    open(LOG_PATH, "w", encoding="utf-8").close()

    print(f"拉取 {START} ~ {END}，纯美股池 config.WATCHLIST_EUROPE_US（{len(POOL)}只）...")
    with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        prepared = bp._prepare_backtest_data(POOL, START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    print("模拟基线（WEAK_FULL关闭，纯v2.3 OBSERVATION_EVICT）...")
    r_off = _run(prepared, standalone_weak_full=False)
    print("模拟新增（WEAK_FULL开启，独立通道，不带RSL约束）...")
    r_on = _run(prepared, standalone_weak_full=True)

    print("\n" + "=" * 100)
    print("一、组合层面对比")
    print("=" * 100)
    print(f"{'指标':<16}{'WEAK_FULL关闭':>16}{'WEAK_FULL开启':>16}")
    print(f"{'总收益%':<16}{r_off['total_return_pct']:>16.2f}{r_on['total_return_pct']:>16.2f}")
    print(f"{'Sharpe':<16}{r_off['sharpe']:>16.3f}{r_on['sharpe']:>16.3f}")
    print(f"{'最大回撤%':<16}{r_off['max_drawdown_pct']:>16.2f}{r_on['max_drawdown_pct']:>16.2f}")
    print(f"{'交易数':<16}{r_off['num_trades']:>16}{r_on['num_trades']:>16}")
    print(f"{'平均持仓天数':<16}{_avg_holding_days(r_off['trade_log']):>16.2f}"
          f"{_avg_holding_days(r_on['trade_log']):>16.2f}")

    print("\n" + "=" * 100)
    print("二、Replacement Margin 统计报告（WEAK_FULL开启组）")
    print("=" * 100)
    stats = _replacement_margin_stats(r_on)
    if stats["attempts"] == 0:
        print("没有发生任何容量审查（Replacement Attempts=0）。")
    else:
        print(f"Replacement Attempts: {stats['attempts']}")
        print(f"Replacement Success:  {stats['success']}")
        print(f"Replacement Blocked（因 REPLACEMENT_MARGIN={config.REPLACEMENT_MARGIN} 被阻止）: "
              f"{stats['blocked']}")
        print(f"Average Score Difference（全部Attempts，含Blocked）: {stats['avg_score_difference']}")
        if stats["avg_holding_days_success"] is not None:
            print(f"Average Holding Days（仅成功换仓的victim持仓）: {stats['avg_holding_days_success']}")
        print(f"成功换仓按类型拆分: {stats['by_type']}")
        stats["detail"].to_csv(
            os.path.join(OUT_DIR, "weak_full_standalone_attempts.csv"),
            index=False, encoding="utf-8-sig")
        print(f"\n逐笔明细已保存: {os.path.join(OUT_DIR, 'weak_full_standalone_attempts.csv')}")


if __name__ == "__main__":
    main()
