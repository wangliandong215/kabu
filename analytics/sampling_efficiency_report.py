"""
analytics/sampling_efficiency_report.py -- Signal Sampling Scheduler view of
v2.4's replacement mechanism.

背景：用户把三档置换（OBSERVATION_EVICT/LOW_PARTIAL_EVICT/
FULL_DOWNGRADE_REPLACE）统一重新表述为"信号重采样事件"（sampling event）
——不再问"要不要换仓"，而是问"这次资金重新配置，划不划算"。这是一次纯
分析层的扩展：**不改变任何决策逻辑**（tier1/2/3+cooldown+budget+子预算
的判断代码原封不动），只是在已经跑过、已经稳健验证过的决策引擎之上，用
v2.4新增的三档shadow mode（config.REPLACEMENT_OBSERVATION_TIER_ENABLED /
REPLACEMENT_LOW_PARTIAL_TIER_ENABLED / REPLACEMENT_WEAK_FULL_ENABLED 均设
False）拿到"如果不换、让它自然走完"的反事实数据，做ROI分解。

ROI分解（每一次sampling event）：
  early_exit_loss / avoided_loss ： mark-to-market在审查那一刻的浮盈浮亏，
      跟最终真实平仓PnL的差值——正值＝继续持有会多赚（换仓的机会成本），
      负值＝继续持有会多亏（换仓避免的损失）。这比简单用"整笔交易最终
      PnL的正负"更精确：只归因"审查这一刻之后"发生的部分，不把建仓到
      审查之间已经落袋的盈亏也算进决策的功劳/代价。
  re_entry_gain ： ON跑法里真实发生的置换，受益方(beneficiary)最终已实现
      PnL（复用 analytics/capacity_manager_report.py::_replacement_stats
      的既有匹配逻辑）。

sampling_efficiency = realized_alpha(re_entry_gain合计) / signal_opportunity_cost
      (early_exit_loss合计，只算正值部分)。>1 说明重新配置的资金赚回的比
      放弃的机会成本更多；<1 说明单看这一层归因是净损耗的——但组合整体
      收益可能依然因为路径依赖/复利效应而改善（v2.4 WEAK_FULL的ablation
      已经证明过这种"单笔归因是负的，组合整体是正的"的情况真实存在，见
      analytics/weak_full_ablation.py）。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python analytics/sampling_efficiency_report.py
"""
import contextlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest_portfolio as bp
import config
from analytics.capacity_manager_report import _replacement_stats
from portfolio.replacement_stabilizer import (
    REPL_FULL_DOWNGRADE_REPLACE, REPL_LOW_PARTIAL_EVICT, REPL_OBSERVATION_EVICT,
)

FETCH_START = "2015-01-01"
END         = "2026-07-03"
CASH        = 50_000.0
POOL        = config.WATCHLIST_EUROPE_US
WINDOWS     = [("2015-01-01~2026-07-03 (全历史)", "2015-01-01", END),
               ("2019-01-01~2026-07-03 (剔除早期窄仓位时代)", "2019-01-01", END)]
TIERS       = [REPL_OBSERVATION_EVICT, REPL_LOW_PARTIAL_EVICT, REPL_FULL_DOWNGRADE_REPLACE]

LOG_PATH = os.path.join(os.path.dirname(__file__), "_sampling_efficiency_report.log")
OUT_DIR  = os.path.dirname(__file__)


def _run(prepared, all_tiers_enabled: bool, sim_start: str, sim_end: str):
    orig = {
        "REPLACEMENT_OBSERVATION_TIER_ENABLED": config.REPLACEMENT_OBSERVATION_TIER_ENABLED,
        "REPLACEMENT_LOW_PARTIAL_TIER_ENABLED": config.REPLACEMENT_LOW_PARTIAL_TIER_ENABLED,
        "REPLACEMENT_WEAK_FULL_ENABLED":        config.REPLACEMENT_WEAK_FULL_ENABLED,
    }
    config.ENABLE_ACTIVE_REPLACEMENT = True
    config.ENABLE_REPLACEMENT_STABILIZATION = True
    config.REPLACEMENT_OBSERVATION_TIER_ENABLED = all_tiers_enabled
    config.REPLACEMENT_LOW_PARTIAL_TIER_ENABLED = all_tiers_enabled
    config.REPLACEMENT_WEAK_FULL_ENABLED        = all_tiers_enabled
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            r = bp.simulate_from_prepared(prepared, cash=CASH, currency="$",
                                           sim_start=sim_start, sim_end=sim_end)
    finally:
        for k, v in orig.items():
            setattr(config, k, v)
    return r


def _early_exit_decomposition(r_shadow: dict) -> pd.DataFrame:
    """对FULL-SHADOW跑法里的每一次would-be置换，找到该would_be_victim在
    同一次跑法里真实的后续SELL（(code, entry_date)精确匹配），算出
    foregone_gain = 最终已实现PnL - 审查那一刻的mark-to-market浮盈浮亏。

    去重关键：shadow mode不会真正清仓，同一个持仓在被举牌资格期内会连续
    很多天都满足"would-be victim"条件，每天都会被记一条——如果不去重，
    同一笔交易的机会成本会被按"该持仓在shadow池里待了多少天"重复计入N次，
    严重虚增总量。这里按(would_be_victim, entry_date, replacement_type)
    只保留**最早**的一次观测（对应真实机制里"第一次满足条件就会被换出"
    的时点），其余同一持仓后续天数的重复举牌全部丢弃，只统计一次。"""
    shadow = r_shadow.get("shadow_replacements", [])
    if not shadow:
        return pd.DataFrame()

    sell_by_key = {}
    for t in r_shadow["trade_log"]:
        if t["side"] == "SELL" and t.get("entry_date") is not None:
            sell_by_key[(t["code"], t["entry_date"])] = t

    dedup_by_key = {}
    for ev in shadow:
        key = (ev["would_be_victim"], ev["would_be_victim_entry_date"], ev["replacement_type"])
        if key not in dedup_by_key or ev["day_idx"] < dedup_by_key[key]["day_idx"]:
            dedup_by_key[key] = ev

    rows = []
    for ev in dedup_by_key.values():
        key = (ev["would_be_victim"], ev["would_be_victim_entry_date"])
        actual_exit = sell_by_key.get(key)
        final_pnl = actual_exit["pnl"] if actual_exit else None
        mtm_now = ev.get("would_be_victim_mtm_pnl_at_review")
        foregone_gain = (final_pnl - mtm_now) if (final_pnl is not None and mtm_now is not None) else None
        rows.append({**ev, "final_pnl": final_pnl, "foregone_gain_from_review_point": foregone_gain,
                     "still_held_at_end": actual_exit is None})
    return pd.DataFrame(rows)


def _tier_summary(shadow_df: pd.DataFrame, replace_stats: dict, tier: str) -> dict:
    tier_shadow = (shadow_df[shadow_df["replacement_type"] == tier]
                   if len(shadow_df) else pd.DataFrame())
    tier_shadow = tier_shadow[tier_shadow["foregone_gain_from_review_point"].notna()] \
        if len(tier_shadow) else tier_shadow

    signal_opportunity_cost = (tier_shadow.loc[tier_shadow["foregone_gain_from_review_point"] > 0,
                                                "foregone_gain_from_review_point"].sum()
                                if len(tier_shadow) else 0.0)
    avoided_loss_value = (-tier_shadow.loc[tier_shadow["foregone_gain_from_review_point"] < 0,
                                            "foregone_gain_from_review_point"].sum()
                           if len(tier_shadow) else 0.0)

    if replace_stats["count"] == 0:
        realized_alpha, n_real = 0.0, 0
    else:
        d = replace_stats["detail"]
        d_tier = d[(d["replacement_type"] == tier) & (d["incoming_closed"])]
        realized_alpha = float(d_tier["incoming_final_pnl"].sum())
        n_real = len(d_tier)

    efficiency = (realized_alpha / signal_opportunity_cost
                  if signal_opportunity_cost > 0 else None)

    return {
        "tier": tier,
        "n_shadow_events": len(tier_shadow),
        "signal_opportunity_cost": round(float(signal_opportunity_cost), 2),
        "avoided_loss_value": round(float(avoided_loss_value), 2),
        "n_real_replacements_closed": n_real,
        "realized_alpha (re_entry_gain)": round(realized_alpha, 2),
        "sampling_efficiency": round(efficiency, 3) if efficiency is not None else None,
    }


def main():
    open(LOG_PATH, "w", encoding="utf-8").close()

    print(f"拉取 {FETCH_START} ~ {END}，纯美股池 config.WATCHLIST_EUROPE_US（{len(POOL)}只）...")
    with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        prepared = bp._prepare_backtest_data(POOL, FETCH_START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    for label, sim_start, sim_end in WINDOWS:
        print("\n" + "=" * 100)
        print(f"窗口: {label}")
        print("=" * 100)

        print("模拟 ON（三档全部正常执行——当前v2.4默认配置）...")
        r_on = _run(prepared, True, sim_start, sim_end)
        print("模拟 FULL-SHADOW（三档全部只记录不执行）...")
        r_shadow = _run(prepared, False, sim_start, sim_end)

        s_on = _replacement_stats(r_on)
        shadow_df = _early_exit_decomposition(r_shadow)

        print(f"\n{'Tier':<24}{'Shadow事件数':>12}{'机会成本':>14}{'规避亏损值':>14}"
              f"{'真实置换数':>12}{'再配置收益':>14}{'效率比':>10}")
        for tier in TIERS:
            row = _tier_summary(shadow_df, s_on, tier)
            eff = f"{row['sampling_efficiency']:.3f}" if row['sampling_efficiency'] is not None else "N/A"
            print(f"{tier:<24}{row['n_shadow_events']:>12}{row['signal_opportunity_cost']:>14,.0f}"
                  f"{row['avoided_loss_value']:>14,.0f}{row['n_real_replacements_closed']:>12}"
                  f"{row['realized_alpha (re_entry_gain)']:>14,.0f}{eff:>10}")

        print(f"\n组合层面（供对照）：ON 总收益={r_on['total_return_pct']:.2f}% "
              f"Sharpe={r_on['sharpe']:.3f} MDD={r_on['max_drawdown_pct']:.2f}%   |   "
              f"FULL-SHADOW（零置换执行）总收益={r_shadow['total_return_pct']:.2f}% "
              f"Sharpe={r_shadow['sharpe']:.3f} MDD={r_shadow['max_drawdown_pct']:.2f}%")

        if len(shadow_df):
            safe_label = label.split(" ")[0].replace("~", "_").replace("-", "")
            out_path = os.path.join(OUT_DIR, f"sampling_efficiency_{safe_label}.csv")
            shadow_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"\n逐笔明细已保存: {out_path}")


if __name__ == "__main__":
    main()
