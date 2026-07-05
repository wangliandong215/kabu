"""
analytics/portfolio_gap_study.py -- 离线诊断脚本，回答"82只纯美股窄池上
v2.2 为什么还落后 v2.0 约55pp"这个问题的三个结构性假设：

  维度一（饿死）：资金空置率——v2.2 是否在某些窗口里比 v2.0 明显更空仓，
                 错过了那段时间的大盘涨幅？
  维度二（挤跑）：位置挤占——3%的 OBSERVATION 观察仓复活后，是否在
                 MAX_POSITIONS/总敞口上限触发时，挤占了本该给 FULL 强
                 信号的仓位名额？
  维度三（变慢）：开仓延迟——四档状态机比旧的链式过滤器"更严格"，是否让
                 同一只股票、同一波行情的开仓时间点系统性推迟、买得更贵？

**不修改 engine/ 或 risk/ 任何生产代码。** 唯一改动是给
`backtest_portfolio.py::simulate_from_prepared()` 加了三处纯增量、不影响
任何交易决策的埋点（已用单测+既有回测数字核对过前后逐位一致）：
  1. `positions[code]["score_label"]` -- 开仓时记录该仓位的决策标签，用于
     判断"当前占着仓位名额的是不是弱信号"。
  2. `capacity_blocks` -- 当 MAX_POSITIONS 或 MAX_TOTAL_EXPOSURE 触发导致
     后续候选信号被整体放弃（原代码里的 `break`）时，记录被放弃的候选
     （含 code/score_label/total_score/price）和当时占用仓位的构成。
  3. `deployed_curve` -- 逐日已部署资金（mark-to-market），跟 equity_curve
     并列返回，用于算资金利用率。

这三处埋点都只是"多记一份日志"，不改变任何 qty/仓位/交易时机的计算——
已用 `python -m unittest ...`（48个单测）+ 加埋点前后重跑同一份82只$50k
回测确认总收益/交易数逐位不变来验证。

方法论说明（近似之处，务必读完再看数字）：
  - 维度一的"机会成本"：把窗口内的平均闲置资金 × 同期QQQ涨幅算作"本可以
    赚到但没赚到"的钱——这是一个粗略代理（假设闲置资金原本会获得市场beta
    收益），不是严格的反事实模拟。
  - 维度二的"潜在利润"：对每笔被挤占的FULL信号，用
    `config.SIZING_PCT_STRONG_DYNAMIC × 起始资金` 作为"如果没被挤占，正常
    应该拿到的仓位金额"的近似值（不是真实反事实资金曲线），持有期用信号
    序列里下一次SELL信号（或60个交易日封顶）近似退出时点。
  - 维度三的"配对"：同一只股票在两个版本里开仓日期相差在
    `MATCH_TOLERANCE_DAYS`（默认15个交易日）内的两笔BUY视为"同一个交易
    机会"，不是完全形式化证明。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python analytics/portfolio_gap_study.py
"""
import contextlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest_portfolio as bp
import config
import engine.scoring as scoring

START = "2015-01-01"
END   = "2026-07-03"
CASH  = 50_000.0
POOL  = config.WATCHLIST_EUROPE_US   # 82只纯美股——55pp缺口出现的那个池子

VACANCY_GAP_THRESHOLD  = 0.30   # 资金利用率差值超过这个值才算"空置"
VACANCY_MIN_DAYS       = 5      # 且要连续这么多个交易日以上才算"真空期"
CROWD_LOOKFORWARD_DAYS = 60     # 被挤占信号的反事实持有期上限（交易日）
MATCH_TOLERANCE_DAYS   = 15     # 判定"同一笔交易机会"的开仓日期容差
TOP_N_WINNERS          = 50

LOG_PATH = os.path.join(os.path.dirname(__file__), "_portfolio_gap_study.log")
OUT_DIR  = os.path.dirname(__file__)


# ───────────────────────── 两个变体的回测执行 ──────────────────────────────

def _run_v20(prepared):
    """v2.0：旧链式过滤器——compute_total_score 永远返回 FULL/scale=1.0
    （不SKIP、不做总分连续折算），_size() 忽略 score_label、回退到原始
    signal_strength 阈值分档（weak<0.4观察仓3%/medium 0.4-0.7为10%/
    strong>=0.7为20-30%）。跟 _yearly_report_v22.py 同一套已验证手法。"""
    _real_compute = scoring.compute_total_score
    _real_size    = bp._size

    def _fake_compute_total_score(trend_strength, weather_code,
                                   fundamental_score=None, news_score=None):
        trend_score = max(0.0, min(1.0, trend_strength)) * 100.0
        weather_score = 0.0 if weather_code == 0 else 100.0
        return scoring.TotalScore(100.0, 1.0, scoring.LABEL_FULL,
                                   trend_score, fundamental_score, news_score, weather_score)

    def _old_size(*args, **kwargs):
        kwargs.pop("score_label", None)
        return _real_size(*args, **kwargs)

    scoring.compute_total_score = _fake_compute_total_score
    bp._size = _old_size
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            r = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")
    finally:
        scoring.compute_total_score = _real_compute
        bp._size = _real_size
    return r


def _run_v22(prepared):
    """v2.2：当前代码原样跑（四档状态机，含 OBSERVATION）。"""
    with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        r = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")
    return r


# ───────────────────────── 维度一：资金空置率 ──────────────────────────────

def analyze_vacancy(r_v20: dict, r_v22: dict, qqq_close: pd.Series) -> dict:
    util_v20 = r_v20["deployed_curve"] / r_v20["equity_curve"]
    util_v22 = r_v22["deployed_curve"] / r_v22["equity_curve"]
    gap = (util_v20 - util_v22).dropna()

    windows, run_start = [], None
    dates = list(gap.index)
    for i, date in enumerate(dates):
        g = gap.loc[date]
        if g > VACANCY_GAP_THRESHOLD:
            if run_start is None:
                run_start = date
        else:
            if run_start is not None:
                windows.append((run_start, dates[i - 1]))
                run_start = None
    if run_start is not None:
        windows.append((run_start, dates[-1]))

    detail_rows = []
    total_loss_estimate = 0.0
    for start, end in windows:
        span = gap.loc[start:end]
        if len(span) < VACANCY_MIN_DAYS:
            continue
        avg_gap = float(span.mean())
        avg_equity_v22 = float(r_v22["equity_curve"].loc[start:end].mean())
        idle_capital = avg_gap * avg_equity_v22
        qqq_slice = qqq_close.loc[start:end]
        qqq_ret = float(qqq_slice.iloc[-1] / qqq_slice.iloc[0] - 1) if len(qqq_slice) > 1 else 0.0
        loss_estimate = idle_capital * qqq_ret
        total_loss_estimate += loss_estimate
        detail_rows.append({
            "start": start, "end": end, "trading_days": len(span),
            "avg_utilization_gap_pct": round(avg_gap * 100, 1),
            "avg_idle_capital": round(idle_capital, 0),
            "qqq_return_pct_during_window": round(qqq_ret * 100, 2),
            "estimated_opportunity_cost": round(loss_estimate, 0),
        })

    return {
        "windows": detail_rows,
        "n_windows": len(detail_rows),
        "total_estimated_loss": round(total_loss_estimate, 0),
        "Vacant_Time_Loss_Pct": round(total_loss_estimate / CASH * 100, 2),
    }


# ───────────────────────── 维度二：强信号被挤占 ────────────────────────────

def analyze_crowding_out(r_v22: dict, signals: dict, all_data: dict) -> dict:
    blocks = r_v22["capacity_blocks"]
    rows = []
    for b in blocks:
        full_candidates = [c for c in b["blocked_candidates"]
                            if c["score_label"] == scoring.LABEL_FULL]
        if not full_candidates:
            continue
        occ = b["occupied_positions"]
        n_occ = len(occ)
        n_obs = sum(1 for o in occ if o["score_label"] == scoring.LABEL_OBSERVATION)
        obs_pct = (n_obs / n_occ * 100) if n_occ else 0.0
        for c in full_candidates:
            rows.append({
                "date": b["date"], "reason": b["reason"], "code": c["code"],
                "total_score": c["total_score"], "price": c.get("price"),
                "occupied_slots": n_occ, "observation_occupied": n_obs,
                "observation_occupancy_pct": round(obs_pct, 1),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return {"Crowding_Out_Strong_Signals_Count": 0, "distinct_codes": 0,
                "avg_observation_occupancy_pct": None,
                "potential_profit_estimate": 0.0,
                "potential_profit_estimate_pct_of_cash": 0.0,
                "detail": df}

    df = df.sort_values("date")
    first_block = df.drop_duplicates(subset="code", keep="first").copy()

    nominal_size = config.SIZING_PCT_STRONG_DYNAMIC * CASH   # 近似值，见模块docstring
    potential_profit = 0.0
    for _, row in first_block.iterrows():
        code, block_date, entry_price = row["code"], row["date"], row["price"]
        if code not in all_data or entry_price is None or entry_price <= 0:
            continue
        sig = signals.get(code)
        if sig is None or block_date not in sig.index:
            continue
        future = sig.loc[sig.index > block_date].head(CROWD_LOOKFORWARD_DAYS)
        if future.empty:
            continue
        sell_days = future[future["signal"] == "SELL"]
        exit_date = sell_days.index[0] if len(sell_days) else future.index[-1]
        if exit_date not in all_data[code].index:
            continue
        exit_price = float(all_data[code].loc[exit_date, "close"])
        ret_pct = exit_price / entry_price - 1
        potential_profit += nominal_size * ret_pct

    return {
        "Crowding_Out_Strong_Signals_Count": len(df),
        "distinct_codes": len(first_block),
        "avg_observation_occupancy_pct": round(float(df["observation_occupancy_pct"].mean()), 1),
        "potential_profit_estimate": round(potential_profit, 0),
        "potential_profit_estimate_pct_of_cash": round(potential_profit / CASH * 100, 2),
        "detail": df,
    }


# ───────────────────────── 维度三：开仓时间戳延迟 ──────────────────────────

def analyze_entry_delay(r_v20: dict, r_v22: dict) -> dict:
    date_index = {d: i for i, d in enumerate(r_v22["equity_curve"].index)}

    def _buy_map(trade_log):
        m = {}
        for t in trade_log:
            if t["side"] == "BUY" and t.get("reason") == "SIGNAL":
                m.setdefault(t["code"], []).append(t)
        return m

    v20_buys = _buy_map(r_v20["trade_log"])
    v22_buys = _buy_map(r_v22["trade_log"])
    v20_sells = sorted(
        [t for t in r_v20["trade_log"] if t["side"] == "SELL" and t.get("pnl") is not None],
        key=lambda t: t["pnl"], reverse=True,
    )

    matched = []
    for sell in v20_sells:
        if len(matched) >= TOP_N_WINNERS:
            break
        code, entry_date = sell["code"], sell.get("entry_date")
        if sell["pnl"] <= 0 or code not in v20_buys or code not in v22_buys:
            continue
        v20_buy = next((b for b in v20_buys[code] if b["date"] == entry_date), None)
        if v20_buy is None or entry_date not in date_index:
            continue

        best, best_delay = None, None
        for b in v22_buys[code]:
            if b["date"] not in date_index:
                continue
            delay = date_index[b["date"]] - date_index[entry_date]
            if abs(delay) <= MATCH_TOLERANCE_DAYS and (best_delay is None or abs(delay) < abs(best_delay)):
                best, best_delay = b, delay
        if best is None:
            continue

        slippage_pct = (best["price"] / v20_buy["price"] - 1) * 100
        matched.append({
            "code": code, "v20_entry_date": entry_date, "v22_entry_date": best["date"],
            "delay_trading_days": best_delay, "v20_price": v20_buy["price"],
            "v22_price": best["price"], "slippage_pct": round(slippage_pct, 2),
            "v20_pnl": sell["pnl"],
        })

    df = pd.DataFrame(matched)
    if df.empty:
        return {"Timestamp_Delay_Avg_Cost": None, "n_matched": 0, "detail": df}

    return {
        "n_matched": len(df),
        "avg_delay_trading_days": round(float(df["delay_trading_days"].mean()), 2),
        "avg_slippage_pct": round(float(df["slippage_pct"].mean()), 2),
        "pct_entered_later_and_higher": round(
            float(((df["delay_trading_days"] > 0) & (df["slippage_pct"] > 0)).mean() * 100), 1),
        "Timestamp_Delay_Avg_Cost": round(float(df["slippage_pct"].mean()), 2),
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

    print("模拟 v2.0（旧链式过滤器）...")
    r_v20 = _run_v20(prepared)
    print("模拟 v2.2（四档状态机，当前代码）...")
    r_v22 = _run_v22(prepared)

    gap_pct = r_v20["total_return_pct"] - r_v22["total_return_pct"]
    print(f"\nv2.0: 总收益={r_v20['total_return_pct']:.2f}%  交易数={r_v20['num_trades']}")
    print(f"v2.2: 总收益={r_v22['total_return_pct']:.2f}%  交易数={r_v22['num_trades']}")
    print(f"缺口 (v2.0 - v2.2): {gap_pct:.2f}pp\n")

    qqq_close = (prepared["benchmark_df"]["close"].astype(float)
                 .reindex(r_v22["equity_curve"].index).ffill())

    print("=" * 100)
    print("维度一：资金空置率与机会成本（饿死）")
    print("=" * 100)
    d1 = analyze_vacancy(r_v20, r_v22, qqq_close)
    print(f"真空期窗口数（v2.2利用率比v2.0低{VACANCY_GAP_THRESHOLD*100:.0f}pp+，"
          f"持续{VACANCY_MIN_DAYS}个交易日以上）: {d1['n_windows']}")
    print(f"估算合计机会成本: {d1['total_estimated_loss']:,.0f} "
          f"（Vacant_Time_Loss_Pct = {d1['Vacant_Time_Loss_Pct']}% of 起始资金）")
    if gap_pct:
        print(f"对55pp缺口的解释占比: {d1['Vacant_Time_Loss_Pct'] / gap_pct * 100:.1f}%")
    if d1["windows"]:
        pd.DataFrame(d1["windows"]).to_csv(
            os.path.join(OUT_DIR, "dimension1_vacancy_windows.csv"), index=False, encoding="utf-8-sig")
        print(f"明细已保存: {os.path.join(OUT_DIR, 'dimension1_vacancy_windows.csv')}")

    print("\n" + "=" * 100)
    print("维度二：强弱信号的位置挤占（挤跑）")
    print("=" * 100)
    d2 = analyze_crowding_out(r_v22, prepared["signals"], prepared["all_data"])
    print(f"Crowding_Out_Strong_Signals_Count（去重后不同股票数）: {d2['distinct_codes']}")
    print(f"（原始被挡事件数，含同股票多日重复计数）: {d2['Crowding_Out_Strong_Signals_Count']}")
    if d2["avg_observation_occupancy_pct"] is not None:
        print(f"挤占事件发生时，持仓席位平均 {d2['avg_observation_occupancy_pct']}% 被 OBSERVATION 弱信号占用")
        print(f"这批被挤占FULL信号的潜在利润估算（近似）: {d2['potential_profit_estimate']:,.0f} "
              f"（{d2['potential_profit_estimate_pct_of_cash']}% of 起始资金）")
        if not d2["detail"].empty:
            d2["detail"].to_csv(
                os.path.join(OUT_DIR, "dimension2_crowded_full_signals.csv"), index=False, encoding="utf-8-sig")
            print(f"明细已保存: {os.path.join(OUT_DIR, 'dimension2_crowded_full_signals.csv')}")
    else:
        print("没有发生任何FULL信号因容量上限被放弃的事件。")

    print("\n" + "=" * 100)
    print("维度三：开仓时间戳延迟与滑点（变慢）")
    print("=" * 100)
    d3 = analyze_entry_delay(r_v20, r_v22)
    if d3["n_matched"] == 0:
        print("没有找到可配对的交易（可能是同代码/时间窗口重叠不足）。")
    else:
        print(f"配对到 {d3['n_matched']} 笔交易（v2.0/v2.2 同代码、"
              f"开仓日期在{MATCH_TOLERANCE_DAYS}个交易日内）")
        print(f"平均开仓延迟: {d3['avg_delay_trading_days']:.2f} 个交易日")
        print(f"平均开仓滑点（Timestamp_Delay_Avg_Cost）: {d3['avg_slippage_pct']:.2f}%")
        print(f"其中 {d3['pct_entered_later_and_higher']}% 的配对交易同时满足'更晚+更高价'（追高特征）")
        d3["detail"].to_csv(
            os.path.join(OUT_DIR, "dimension3_entry_delay.csv"), index=False, encoding="utf-8-sig")
        print(f"明细已保存: {os.path.join(OUT_DIR, 'dimension3_entry_delay.csv')}")

    print("\n" + "=" * 100)
    print(f"汇总：55pp量级缺口归因（本次实测缺口 {gap_pct:.2f}pp）")
    print("=" * 100)
    print(f"  饿死 Vacant_Time_Loss_Pct                = {d1['Vacant_Time_Loss_Pct']:.2f}pp")
    if d2["avg_observation_occupancy_pct"] is not None:
        print(f"  挤跑 潜在利润损失（近似）                 = {d2['potential_profit_estimate_pct_of_cash']:.2f}pp"
              f"  (Crowding_Out_Strong_Signals_Count={d2['distinct_codes']})")
    if d3["n_matched"]:
        print(f"  变慢 Timestamp_Delay_Avg_Cost（单笔均值） = {d3['Timestamp_Delay_Avg_Cost']:.2f}%"
              f"  （基于{d3['n_matched']}笔配对交易，不是全样本外推到pp口径的加总量）")


if __name__ == "__main__":
    main()
