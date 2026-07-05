"""
analytics/margin_optimization_report.py -- v2.4 优化阶段 第一~四阶段：
REPLACEMENT_MARGIN 参数扫描 + Replacement Quality Analysis (Replace Alpha)
+ Blocked Opportunity Analysis + 基于规则的参数推荐。

前提：三轮WEAK_FULL实验（完整RSL/RSL消融/独立WEAK_FULL）已确认WEAK_FULL
方向为负贡献，本次优化以当前生产版v2.3为基线（config.
REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED=False，
config.ENABLE_REPLACEMENT_STABILIZATION=False，均为默认值，本脚本不改动
这两个开关）——唯一变量是 config.REPLACEMENT_MARGIN。

方法论：
  阶段一：对 MARGIN_VALUES 里的每个取值跑一次82只美股/2015-2026全历史
    回测，产出总收益/年化收益/Sharpe/Sortino/Calmar/最大回撤/交易次数/
    胜率/平均持仓天数/Replacement Attempts/Success/Blocked/Blocked
    Ratio/Average Score Difference。Sharpe/Sortino/Calmar/MDD/胜率复用
    `_yearly_report.py::_period_metrics()`（不重新定义指标口径）。
  阶段二：对每次真正执行的Replace，用真实价格数据计算"如果不替换、继续
    持有旧股票到（新仓位平仓日或回测结束日）"的价格收益，跟新仓位自己
    实际实现的价格收益相减，得到 Replace Alpha；只用已有回测产出的
    trade_log/replacements + all_data 真实价格，不新增策略信号、不重新
    模拟止损/止盈逻辑。
  阶段三：对每次因margin被拦截(decision=="KEEP")的候选，用真实价格数据
    计算旧持仓/候选股票从审查当天起、未来5/10/20/60个交易日的价格收益，
    统计"阻止正确"（旧持仓表现更好）vs"阻止错误"（候选股表现更好）的
    比例。
  阶段四：基于阶段一的表格，用明确规则（不是权重打分黑盒）筛选/排序：
    先筛出总收益不低于baseline(margin=10)减去容忍区间的候选，再按
    (Sharpe降序, 最大回撤降序即幅度更小优先, 交易次数升序) 排序，取第一
    名作为推荐，同时把完整候选表打印出来供人工复核推荐理由。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python analytics/margin_optimization_report.py
"""
import contextlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest_portfolio as bp
import config
import _yearly_report as yr

START = "2015-01-01"
END   = "2026-07-03"
CASH  = 50_000.0
POOL  = config.WATCHLIST_EUROPE_US

MARGIN_VALUES     = [6.0, 8.0, 10.0, 12.0, 15.0, 20.0,
                     # 2026-07-05 扩展：诊断发现OBSERVATION_EVICT候选的
                     # 真实score_difference分布 min=27.19 / 25th=42.6 /
                     # median=49.19 / max=59.82（82只池全历史，见
                     # margin_optimization_report.log附带的诊断输出）——
                     # PRD原定的6~20区间margin完全不生效（Blocked恒为0，
                     # 组合层面全部指标逐位相同），必须把网格延伸到
                     # margin真正开始起作用的区间，否则整个参数扫描是
                     # 一次没有信息量的空跑。PRD"如有必要可增加更多中间值"
                     # 条款覆盖这次延伸。
                     25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
BASELINE_MARGIN   = 10.0
RETURN_TOLERANCE_PP = 15.0   # baseline总收益243.49%左右，15pp约等于6%相对容忍度
HORIZONS = [5, 10, 20, 60]

LOG_PATH = os.path.join(os.path.dirname(__file__), "_margin_optimization_report.log")
OUT_DIR  = os.path.dirname(__file__)


def _avg_holding_days(trade_log) -> float:
    days = [(t["date"] - t["entry_date"]).days
            for t in trade_log
            if t["side"] == "SELL" and t.get("strategy") != "core_etf"
            and t.get("entry_date") is not None]
    return round(sum(days) / len(days), 2) if days else float("nan")


def _stage1_metrics(result: dict, prepared: dict, margin: float) -> dict:
    eq = result["equity_curve"]
    qqq = prepared["all_data"]["US.QQQ"]["close"].astype(float).reindex(eq.index).ffill()

    sell_trades = [t for t in result["trade_log"] if t["side"] == "SELL"]
    trades_df = pd.DataFrame(sell_trades)
    if len(trades_df):
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df["year"] = trades_df["date"].dt.year
        trades_df["pnl"] = trades_df["pnl"].astype(float)

    full_mask = np.ones(len(eq), dtype=bool)
    metrics, _, _ = yr._period_metrics(eq, qqq, full_mask, CASH, float(qqq.iloc[0]), trades_df, "ALL")
    # 全期"交易数"/"胜率%"需要覆盖（_period_metrics内部按year_label过滤，
    # "ALL"不是真实年份匹配不到任何行）——跟 _yearly_report.py::_build_table
    # 里对"全期汇总"行的处理完全同款，不是这里新发明的补丁。
    if len(trades_df):
        n_all = len(trades_df)
        wins_all = trades_df[trades_df["pnl"] > 0]
        metrics["交易数"] = n_all
        metrics["胜率%"] = round(len(wins_all) / n_all * 100, 1) if n_all else 0.0

    attempts = result["replacement_attempts"]
    n_attempts = len(attempts)
    n_success = sum(1 for a in attempts if a["decision"] == "REPLACE")
    n_blocked = sum(1 for a in attempts if a["decision"] == "KEEP")
    # 2026-07-05修复：margin通过但因敞口/板块/qty<=0被下游回滚的事件标记为
    # "ROLLED_BACK"（跟backtest_portfolio.py的_undo_replacement同步），
    # 不计入Success也不计入Blocked——它不是margin的问题，混进Blocked会
    # 误导"margin是否过严"的判断，混进Success会高估真实换仓成功率
    # （之前版本此处有bug：205次margin通过误报成205次成功，实际只有121次
    # 真正落地，见对应CSV核对）。
    n_rolled_back = sum(1 for a in attempts if a["decision"] == "ROLLED_BACK")
    avg_score_diff = (sum(a["score_difference"] for a in attempts) / n_attempts
                       if n_attempts else float("nan"))

    return {
        "REPLACEMENT_MARGIN":      margin,
        "总收益%":                  metrics["收益率%"],
        "年化收益%":                round(result["annualized_return_pct"], 2),
        "Sharpe":                  metrics["Sharpe"],
        "Sortino":                 metrics["Sortino"],
        "Calmar":                  metrics["Calmar"],
        "最大回撤%":                metrics["MDD%"],
        "交易次数":                 metrics["交易数"],
        "胜率%":                   metrics["胜率%"],
        "平均持仓天数":              _avg_holding_days(result["trade_log"]),
        "Replacement Attempts":    n_attempts,
        "Replacement Success":     n_success,
        "Replacement Blocked":     n_blocked,
        "Replacement Rolled Back（非margin原因，敞口/板块/qty<=0）": n_rolled_back,
        "Blocked Ratio%":          round(n_blocked / n_attempts * 100, 1) if n_attempts else None,
        "Average Score Difference": round(avg_score_diff, 2) if n_attempts else None,
    }


def _replace_alpha_detail(result: dict, all_data: dict, all_dates: list) -> pd.DataFrame:
    """阶段二：每次真正执行的Replace，计算Replace Alpha = 新仓位实际价格
    收益 - 旧仓位"如果继续持有到同一个窗口终点"的价格收益。窗口终点 =
    新仓位自己实际平仓日（若回测结束时仍未平仓，则用回测最后一天+当时
    最新可得收盘价近似）。"""
    reps = result["replacements"]
    if not reps:
        return pd.DataFrame()

    trade_log = result["trade_log"]
    victim_exit_price = {(t["code"], t["date"]): t["price"]
                          for t in trade_log
                          if t["side"] == "SELL" and t.get("reason") == "ACTIVE_REPLACEMENT"}
    incoming_buy_price = {(t["code"], t["date"]): t["price"]
                           for t in trade_log if t["side"] == "BUY"}
    incoming_sell = {(t["code"], t["entry_date"]): t
                      for t in trade_log if t["side"] == "SELL"}
    backtest_end = pd.Timestamp(all_dates[-1])

    rows = []
    for rep in reps:
        date, victim_code, incoming_code = rep["date"], rep["replacement_from"], rep["replacement_to"]
        v_exit_price = victim_exit_price.get((victim_code, date))
        i_entry_price = incoming_buy_price.get((incoming_code, date))
        if v_exit_price is None or i_entry_price is None:
            continue   # 防御性：正常情况下不会发生，同一事件必然两条trade_log记录都在

        sell = incoming_sell.get((incoming_code, date))
        if sell is not None:
            window_end, i_exit_price, still_open = sell["date"], sell["price"], False
        else:
            window_end, still_open = backtest_end, True
            df_i = all_data.get(incoming_code)
            if df_i is None:
                continue
            i_exit_price = float(df_i["close"].reindex([window_end], method="ffill").iloc[0])

        df_v = all_data.get(victim_code)
        if df_v is None:
            continue
        v_end_price = float(df_v["close"].reindex([window_end], method="ffill").iloc[0])

        replace_return = i_exit_price / i_entry_price - 1
        keep_return = v_end_price / v_exit_price - 1
        rows.append({
            "date": date, "victim_code": victim_code, "incoming_code": incoming_code,
            "window_end": window_end, "still_open_at_backtest_end": still_open,
            "replace_return_pct": round(replace_return * 100, 3),
            "keep_return_pct": round(keep_return * 100, 3),
            "replace_alpha_pct": round((replace_return - keep_return) * 100, 3),
        })
    return pd.DataFrame(rows)


def _stage2_summary(detail: pd.DataFrame) -> dict:
    if detail.empty:
        return {"n": 0}
    return {
        "n": len(detail),
        "mean_alpha_pct": round(float(detail["replace_alpha_pct"].mean()), 3),
        "median_alpha_pct": round(float(detail["replace_alpha_pct"].median()), 3),
        "win_rate_pct": round(float((detail["replace_alpha_pct"] > 0).mean()) * 100, 1),
    }


def _blocked_detail(result: dict, all_data: dict) -> pd.DataFrame:
    """阶段三：每次因margin被拦截(decision=='KEEP')的候选，记录旧持仓/
    候选股票从审查当天起未来 HORIZONS 个交易日的真实价格收益（用各自
    股票自己的交易日索引找第N根K线，不假设跨股票交易日历完全对齐）。"""
    attempts = [a for a in result["replacement_attempts"] if a["decision"] == "KEEP"]
    if not attempts:
        return pd.DataFrame()

    rows = []
    for a in attempts:
        date, victim_code, incoming_code = a["date"], a["victim_code"], a["incoming_code"]
        df_v, df_i = all_data.get(victim_code), all_data.get(incoming_code)
        if df_v is None or df_i is None:
            continue
        ts = pd.Timestamp(date)
        if ts not in df_v.index or ts not in df_i.index:
            continue
        v_pos, i_pos = df_v.index.get_loc(ts), df_i.index.get_loc(ts)
        v0, i0 = float(df_v["close"].iloc[v_pos]), float(df_i["close"].iloc[i_pos])

        row = {"date": date, "victim_code": victim_code, "incoming_code": incoming_code}
        for h in HORIZONS:
            vt, it = v_pos + h, i_pos + h
            if vt >= len(df_v) or it >= len(df_i):
                row[f"victim_ret_{h}d_pct"] = None
                row[f"candidate_ret_{h}d_pct"] = None
                continue
            row[f"victim_ret_{h}d_pct"] = round(float(df_v["close"].iloc[vt]) / v0 * 100 - 100, 3)
            row[f"candidate_ret_{h}d_pct"] = round(float(df_i["close"].iloc[it]) / i0 * 100 - 100, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _blocked_summary(detail: pd.DataFrame) -> dict:
    out = {}
    for h in HORIZONS:
        vcol, ccol = f"victim_ret_{h}d_pct", f"candidate_ret_{h}d_pct"
        if detail.empty or vcol not in detail.columns:
            out[h] = {"n": 0}
            continue
        sub = detail[[vcol, ccol]].dropna()
        n = len(sub)
        if n == 0:
            out[h] = {"n": 0}
            continue
        correct = int((sub[vcol] > sub[ccol]).sum())
        wrong = int((sub[ccol] > sub[vcol]).sum())
        out[h] = {"n": n,
                   "blocked_correct_pct": round(correct / n * 100, 1),
                   "blocked_wrong_pct": round(wrong / n * 100, 1)}
    return out


def _recommend(df1: pd.DataFrame) -> pd.DataFrame:
    """阶段四：规则式筛选/排序，不是权重打分黑盒。规则：
      1) 先筛出总收益 >= baseline总收益 - RETURN_TOLERANCE_PP 的候选
         （"收益基本不下降"）。
      2) 候选池内按 (Sharpe降序, 最大回撤%降序[越接近0/幅度越小越靠前],
         交易次数升序) 排序——对应PRD"若多个参数综合表现接近，优先选择
         Sharpe更高、回撤更小、交易次数更少的方案"。
    返回排好序的候选DataFrame（第一行 = 推荐结果），如果没有候选通过
    筛选则返回按同一排序规则排的全量表（并在打印时提示筛选条件未命中）。
    """
    baseline_row = df1[df1["REPLACEMENT_MARGIN"] == BASELINE_MARGIN]
    baseline_return = float(baseline_row["总收益%"].iloc[0]) if len(baseline_row) else df1["总收益%"].max()

    candidates = df1[df1["总收益%"] >= baseline_return - RETURN_TOLERANCE_PP].copy()
    fell_back = False
    if candidates.empty:
        candidates = df1.copy()
        fell_back = True

    candidates = candidates.sort_values(
        by=["Sharpe", "最大回撤%", "交易次数"],
        ascending=[False, False, True]).reset_index(drop=True)
    candidates.attrs["baseline_return"] = baseline_return
    candidates.attrs["fell_back"] = fell_back
    return candidates


def main():
    open(LOG_PATH, "w", encoding="utf-8").close()

    print(f"拉取 {START} ~ {END}，纯美股池 config.WATCHLIST_EUROPE_US（{len(POOL)}只），仅拉取一次...")
    with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        prepared = bp._prepare_backtest_data(POOL, START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    orig_margin = config.REPLACEMENT_MARGIN
    stage1_rows, stage2_summaries, stage3_summaries = [], {}, {}
    detail2_frames, detail3_frames = {}, {}
    try:
        for margin in MARGIN_VALUES:
            print(f"模拟 REPLACEMENT_MARGIN={margin} ...")
            config.REPLACEMENT_MARGIN = margin
            with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
                result = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")

            stage1_rows.append(_stage1_metrics(result, prepared, margin))

            d2 = _replace_alpha_detail(result, prepared["all_data"], prepared["all_dates"])
            detail2_frames[margin] = d2
            stage2_summaries[margin] = _stage2_summary(d2)

            d3 = _blocked_detail(result, prepared["all_data"])
            detail3_frames[margin] = d3
            stage3_summaries[margin] = _blocked_summary(d3)
    finally:
        config.REPLACEMENT_MARGIN = orig_margin

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    df1 = pd.DataFrame(stage1_rows)
    print("\n" + "=" * 130)
    print("阶段一：REPLACEMENT_MARGIN 参数扫描对比表")
    print("=" * 130)
    print(df1.to_string(index=False))
    df1.to_csv(os.path.join(OUT_DIR, "margin_sweep_stage1.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 130)
    print("阶段二：Replacement Quality Analysis（Replace Alpha，每个margin取值分别统计）")
    print("=" * 130)
    df2 = pd.DataFrame([{"REPLACEMENT_MARGIN": m, **s} for m, s in stage2_summaries.items()])
    print(df2.to_string(index=False))
    for m, d in detail2_frames.items():
        if not d.empty:
            d.to_csv(os.path.join(OUT_DIR, f"replace_alpha_detail_margin{m}.csv"),
                      index=False, encoding="utf-8-sig")

    print("\n" + "=" * 130)
    print("阶段三：Blocked Opportunity Analysis（按 margin x 观察周期）")
    print("=" * 130)
    rows3 = []
    for m, per_h in stage3_summaries.items():
        for h, s in per_h.items():
            rows3.append({"REPLACEMENT_MARGIN": m, "horizon_days": h, **s})
    df3 = pd.DataFrame(rows3)
    print(df3.to_string(index=False))
    for m, d in detail3_frames.items():
        if not d.empty:
            d.to_csv(os.path.join(OUT_DIR, f"blocked_opportunity_detail_margin{m}.csv"),
                      index=False, encoding="utf-8-sig")

    print("\n" + "=" * 130)
    print("阶段四：基于规则的参数推荐")
    print("=" * 130)
    ranked = _recommend(df1)
    baseline_return = ranked.attrs["baseline_return"]
    if ranked.attrs["fell_back"]:
        print(f"警告：没有margin取值的总收益落在baseline({BASELINE_MARGIN})的"
              f"{baseline_return:.2f}% - {RETURN_TOLERANCE_PP}pp 容忍区间内，"
              f"已回退为对全部取值按(Sharpe降序,最大回撤降序,交易次数升序)排序。")
    else:
        print(f"筛选条件：总收益% >= baseline({BASELINE_MARGIN})的{baseline_return:.2f}% "
              f"- {RETURN_TOLERANCE_PP}pp = {baseline_return - RETURN_TOLERANCE_PP:.2f}%")
        print(f"通过筛选的候选数：{len(ranked)} / {len(df1)}")
    print("\n候选排序（第一行为推荐）：")
    print(ranked.to_string(index=False))
    ranked.to_csv(os.path.join(OUT_DIR, "margin_recommendation_ranked.csv"),
                  index=False, encoding="utf-8-sig")

    best = ranked.iloc[0]
    print(f"\n>>> 推荐 REPLACEMENT_MARGIN = {best['REPLACEMENT_MARGIN']}"
          f"（总收益{best['总收益%']}% / Sharpe {best['Sharpe']} / "
          f"最大回撤{best['最大回撤%']}% / 交易次数{best['交易次数']} / "
          f"Blocked Ratio {best['Blocked Ratio%']}%）")
    print("\n结果已保存到 analytics/ 目录：margin_sweep_stage1.csv / "
          "replace_alpha_detail_margin*.csv / blocked_opportunity_detail_margin*.csv / "
          "margin_recommendation_ranked.csv")


if __name__ == "__main__":
    main()
