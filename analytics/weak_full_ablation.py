"""
analytics/weak_full_ablation.py -- WEAK FULL(FULL_DOWNGRADE_REPLACE) tier
ablation + shadow-mode analysis.

背景：v2.4实测（82只池，两个不同历史窗口）显示Replacement Count在两个窗口
里都稳健上升，但收益/Sharpe/回撤谁更好高度依赖窗口选择，不能一概而论。
用户明确要求**不删除任何置换类型**，改为对WEAK FULL这一档单独做
ablation+shadow实验，判断它是"低频高价值的尾部风险截断"还是"冗余路径"：

  ON            : config.REPLACEMENT_WEAK_FULL_ENABLED=True（当前默认）——
                  正常执行FULL_DOWNGRADE_REPLACE。
  OFF+SHADOW    : REPLACEMENT_WEAK_FULL_ENABLED=False——组合按"这一档不存在"
                  往下走（victim不会被换出，继续持有），但
                  portfolio/replacement_stabilizer.py仍会把"如果放行本来会
                  换出谁"记录进 result["shadow_replacements"]（不影响任何
                  交易决策的纯观测埋点）。

产出三类分析（对应用户要求的三个诊断维度）：
  1. Tail loss contribution：ON vs OFF 的 Tail Contribution Ratio（Top10%
     盈利交易占比）+ 该档"如果放行"的victim位置，事后看是被换出后规避了
     后续亏损，还是打断了后续盈利——用OFF run里shadow事件对应victim的
     真实后续走势（entry_date精确匹配的那笔SELL）做反事实评估。
  2. Regime sensitivity：分别在2015-01-01~2026-07-03（含早期低价股窄仓位
     时代）和2019-01-01~2026-07-03（剔除早期极端窗口）两个窗口跑一遍，
     对比ON/OFF在不同regime下的相对表现是否一致。
  3. 补位效应（backfill）：对比ON vs OFF跑法里OBSERVATION_EVICT+
     LOW_PARTIAL_EVICT的总次数——如果OFF跑法这两档次数明显更高，说明
     WEAK FULL关闭后其他档位"补位"承接了部分置换压力；如果基本不变，说明
     WEAK FULL是纯粹独立增量，跟其他档位没有互相挤压关系。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python analytics/weak_full_ablation.py
"""
import contextlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest_portfolio as bp
import config
from analytics.capacity_manager_report import _replacement_stats
from analytics.replacement_stability_report import _tail_contribution_ratio
from portfolio.replacement_stabilizer import REPL_LOW_PARTIAL_EVICT, REPL_OBSERVATION_EVICT

FETCH_START = "2015-01-01"
END         = "2026-07-03"
CASH        = 50_000.0
POOL        = config.WATCHLIST_EUROPE_US
WINDOWS     = [("2015-01-01~2026-07-03 (全历史)", "2015-01-01", END),
               ("2019-01-01~2026-07-03 (剔除早期窄仓位时代)", "2019-01-01", END)]

LOG_PATH = os.path.join(os.path.dirname(__file__), "_weak_full_ablation.log")
OUT_DIR  = os.path.dirname(__file__)


def _run(prepared, weak_full_enabled: bool, sim_start: str, sim_end: str):
    orig = config.REPLACEMENT_WEAK_FULL_ENABLED
    config.ENABLE_ACTIVE_REPLACEMENT = True
    config.ENABLE_REPLACEMENT_STABILIZATION = True
    config.REPLACEMENT_WEAK_FULL_ENABLED = weak_full_enabled
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            r = bp.simulate_from_prepared(prepared, cash=CASH, currency="$",
                                           sim_start=sim_start, sim_end=sim_end)
    finally:
        config.REPLACEMENT_WEAK_FULL_ENABLED = orig
    return r


def _shadow_counterfactual(r_off: dict) -> pd.DataFrame:
    """对OFF+SHADOW跑法里每一次shadow事件，找到该would_be_victim在同一次
    跑法里真实的后续SELL（用 (code, entry_date) 精确匹配持仓——同一持仓
    在没被置换掉的情况下，最终会走正常的止损/止盈/策略SELL退出），标注
    这笔"如果放行会被换掉"的持仓最终是盈利离场还是亏损离场。

    去重关键：shadow mode不会真正清仓，同一个持仓在满足WEAK FULL条件期间
    会连续多天都被记一条would-be事件——不去重会把同一笔交易的结果按
    "在shadow池里待了多少天"重复计入N次，严重虚增样本量和金额。这里按
    (would_be_victim, entry_date)只保留最早的一次观测（对应真实机制里
    "第一次满足条件就会被换出"的时点）。"""
    shadow = r_off.get("shadow_replacements", [])
    if not shadow:
        return pd.DataFrame()

    sell_by_key = {}
    for t in r_off["trade_log"]:
        if t["side"] == "SELL" and t.get("entry_date") is not None:
            sell_by_key[(t["code"], t["entry_date"])] = t

    dedup_by_key = {}
    for ev in shadow:
        key = (ev["would_be_victim"], ev["would_be_victim_entry_date"])
        if key not in dedup_by_key or ev["day_idx"] < dedup_by_key[key]["day_idx"]:
            dedup_by_key[key] = ev

    rows = []
    for ev in dedup_by_key.values():
        key = (ev["would_be_victim"], ev["would_be_victim_entry_date"])
        actual_exit = sell_by_key.get(key)
        rows.append({
            **ev,
            "actual_exit_reason": actual_exit["reason"] if actual_exit else None,
            "actual_exit_pnl":    actual_exit["pnl"] if actual_exit else None,
            "still_held_at_end":  actual_exit is None,
        })
    return pd.DataFrame(rows)


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

        print("模拟 ON（WEAK FULL正常执行）...")
        r_on = _run(prepared, True, sim_start, sim_end)
        print("模拟 OFF+SHADOW（WEAK FULL不执行，只记录would-be victim）...")
        r_off = _run(prepared, False, sim_start, sim_end)

        print(f"\n{'指标':<28}{'ON':>14}{'OFF+SHADOW':>14}")
        print(f"{'总收益%':<28}{r_on['total_return_pct']:>14.2f}{r_off['total_return_pct']:>14.2f}")
        print(f"{'Sharpe':<28}{r_on['sharpe']:>14.3f}{r_off['sharpe']:>14.3f}")
        print(f"{'最大回撤%':<28}{r_on['max_drawdown_pct']:>14.2f}{r_off['max_drawdown_pct']:>14.2f}")
        print(f"{'交易数':<28}{r_on['num_trades']:>14}{r_off['num_trades']:>14}")

        s_on  = _replacement_stats(r_on)
        s_off = _replacement_stats(r_off)
        tcr_on  = _tail_contribution_ratio(r_on["trade_log"])
        tcr_off = _tail_contribution_ratio(r_off["trade_log"])
        print(f"{'Tail Contribution Ratio(Top10%)':<28}{tcr_on*100:>13.1f}%{tcr_off*100:>13.1f}%")

        def _count(stats, type_):
            if stats["count"] == 0:
                return 0
            return int((stats["detail"]["replacement_type"] == type_).sum())

        obs_on,  low_on  = _count(s_on, REPL_OBSERVATION_EVICT),  _count(s_on, REPL_LOW_PARTIAL_EVICT)
        obs_off, low_off = _count(s_off, REPL_OBSERVATION_EVICT), _count(s_off, REPL_LOW_PARTIAL_EVICT)
        print(f"\n【1. Tail loss contribution】ON Tail Ratio={tcr_on*100:.1f}%  "
              f"OFF Tail Ratio={tcr_off*100:.1f}%")

        print(f"\n【3. 补位效应】OBSERVATION_EVICT: ON={obs_on} OFF={obs_off}  "
              f"|  LOW_PARTIAL_EVICT: ON={low_on} OFF={low_off}")
        backfill = (obs_off + low_off) - (obs_on + low_on)
        print(f"OFF比ON多出的OBS+LOW置换次数: {backfill:+d}"
              f"（{'有补位迹象' if backfill > 0 else '无补位，纯独立增量' if backfill == 0 else '反而更少，需进一步排查'}）")

        shadow_df = _shadow_counterfactual(r_off)
        n_shadow = len(shadow_df)
        print(f"\n【WEAK FULL shadow事件数】{n_shadow}（OFF跑法里本来会被换出的次数）")
        if n_shadow > 0:
            closed = shadow_df[~shadow_df["still_held_at_end"]]
            n_closed = len(closed)
            n_would_have_avoided_loss = int((closed["actual_exit_pnl"] < 0).sum())
            n_would_have_cut_winner   = int((closed["actual_exit_pnl"] > 0).sum())
            print(f"已平仓可评估: {n_closed}/{n_shadow}")
            if n_closed > 0:
                print(f"  -> 如果放行置换，本可规避后续亏损的次数: {n_would_have_avoided_loss} "
                      f"({n_would_have_avoided_loss/n_closed*100:.1f}%)")
                print(f"  -> 如果放行置换，本会打断后续盈利的次数: {n_would_have_cut_winner} "
                      f"({n_would_have_cut_winner/n_closed*100:.1f}%)")
                avoided_loss_total = closed.loc[closed["actual_exit_pnl"] < 0, "actual_exit_pnl"].sum()
                cut_winner_total   = closed.loc[closed["actual_exit_pnl"] > 0, "actual_exit_pnl"].sum()
                print(f"  -> 规避的亏损金额合计: {avoided_loss_total:,.2f}")
                print(f"  -> 打断的盈利金额合计: {cut_winner_total:,.2f}")
                print(f"  -> 净效果（正=WEAK FULL整体上是净有益的尾部截断）: "
                      f"{-avoided_loss_total - cut_winner_total:,.2f}")
            safe_label = label.split(" ")[0].replace("~", "_").replace("-", "")
            out_path = os.path.join(OUT_DIR, f"weak_full_shadow_{safe_label}.csv")
            shadow_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"逐笔明细已保存: {out_path}")

        print(f"\n【2. Regime sensitivity 提示】本窗口(ON-OFF)总收益差="
              f"{r_on['total_return_pct']-r_off['total_return_pct']:+.2f}pp, "
              f"Sharpe差={r_on['sharpe']-r_off['sharpe']:+.3f}, "
              f"MDD差={r_on['max_drawdown_pct']-r_off['max_drawdown_pct']:+.2f}pp"
              f"——跨窗口对比见脚本完整输出末尾的两组数字是否同向。")


if __name__ == "__main__":
    main()
