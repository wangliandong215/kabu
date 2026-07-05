"""
analytics/replace_ablation_report.py -- v2.4 优化阶段三：候选变量消融验证。

背景：阶段二（analytics/replace_decision_feature_analysis.py）用相关性/
互信息分析找到三个候选变量——n_observation_pool_size（观察池拥挤度，负
相关）、new_score（incoming绝对分数，正相关）、same_sector（同板块换仓
明显更差）——但用户明确要求"不要因为某个相关系数略高就加入过滤条件"，
必须先做完整的历史A/B消融回测，验证是否能稳定改善收益/Sharpe/回撤/
Replace Alpha，才允许正式进入Replace决策规则。

方法论：
  - 8组配置：baseline（不加任何过滤，=当前生产v2.3行为）+ 3个单变量
    + 3个两两组合 + 1个三变量全开，每组只改capacity_manager.py里已经
    实现好的3个消融开关（config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE/
    REPLACEMENT_MIN_NEW_SCORE/REPLACEMENT_BLOCK_SAME_SECTOR），其余
    所有逻辑（评分/信号/风控/仓位计算）完全不变。
  - 阈值选取：MAX_OBSERVATION_POOL_SIZE=2（生产margin=10基线121次换仓
    的pool_size中位数）、MIN_NEW_SCORE=95.0（121次换仓new_score分布的
    近似中位数区间，25th=92.36/50th=100caps较多，取95作为折中）、
    BLOCK_SAME_SECTOR=True（布尔，无需选阈值）——都是阶段二分析时从
    真实生产基线分布里定的，不是拍脑袋。
  - 稳健性：不只跑一个历史窗口——2015-01-01~2026-07-03（全历史）+
    2019-01-01~2026-07-03（近期窗口，跟RSL章节验证时用的第二窗口一致）
    各跑一遍全部8组配置，只有两个窗口方向一致才算"稳定"，避免重复
    RSL那次"2015-2026说A更好、2019-2026说B更好"的regime flip教训。
  - 判定标准（用户设定）：只有消融回测证明能同时稳定改善收益、Sharpe、
    回撤、Replace Alpha（在两个窗口都成立，不是只在一个窗口凑巧更好）
    的变量/组合，才允许转正维默认值；否则维持v2.3现状，把v2.4定义为
    Replace Engine重构完成版，后续方向转向v3.0多维决策模型。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python analytics/replace_ablation_report.py
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
from margin_optimization_report import _avg_holding_days, _replace_alpha_detail, _stage2_summary

CASH = 50_000.0
POOL = config.WATCHLIST_EUROPE_US
WINDOWS = [
    ("2015-01-01", "2026-07-03", "full_2015_2026"),
    ("2019-01-01", "2026-07-03", "recent_2019_2026"),
]

GATE_KEYS = ["REPLACEMENT_MAX_OBSERVATION_POOL_SIZE", "REPLACEMENT_MIN_NEW_SCORE",
             "REPLACEMENT_BLOCK_SAME_SECTOR"]
GATE_DEFAULTS = {"REPLACEMENT_MAX_OBSERVATION_POOL_SIZE": None,
                  "REPLACEMENT_MIN_NEW_SCORE": None,
                  "REPLACEMENT_BLOCK_SAME_SECTOR": False}

ABLATION_CONFIGS = {
    "baseline":          {},
    "pool_size":         {"REPLACEMENT_MAX_OBSERVATION_POOL_SIZE": 2},
    "new_score":         {"REPLACEMENT_MIN_NEW_SCORE": 95.0},
    "same_sector":       {"REPLACEMENT_BLOCK_SAME_SECTOR": True},
    "pool+score":        {"REPLACEMENT_MAX_OBSERVATION_POOL_SIZE": 2, "REPLACEMENT_MIN_NEW_SCORE": 95.0},
    "pool+sector":       {"REPLACEMENT_MAX_OBSERVATION_POOL_SIZE": 2, "REPLACEMENT_BLOCK_SAME_SECTOR": True},
    "score+sector":      {"REPLACEMENT_MIN_NEW_SCORE": 95.0, "REPLACEMENT_BLOCK_SAME_SECTOR": True},
    "pool+score+sector": {"REPLACEMENT_MAX_OBSERVATION_POOL_SIZE": 2, "REPLACEMENT_MIN_NEW_SCORE": 95.0,
                           "REPLACEMENT_BLOCK_SAME_SECTOR": True},
}

LOG_PATH = os.path.join(os.path.dirname(__file__), "_replace_ablation_report.log")
OUT_DIR  = os.path.dirname(__file__)


def _run(prepared, overrides: dict):
    for key in GATE_KEYS:
        setattr(config, key, GATE_DEFAULTS[key])
    for key, val in overrides.items():
        setattr(config, key, val)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            return bp.simulate_from_prepared(prepared, cash=CASH, currency="$")
    finally:
        for key in GATE_KEYS:
            setattr(config, key, GATE_DEFAULTS[key])


def _row_metrics(name: str, window_label: str, result: dict, prepared: dict) -> dict:
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
    if len(trades_df):
        n_all = len(trades_df)
        wins_all = trades_df[trades_df["pnl"] > 0]
        metrics["交易数"] = n_all
        metrics["胜率%"] = round(len(wins_all) / n_all * 100, 1) if n_all else 0.0

    attempts = result["replacement_attempts"]
    n_success = sum(1 for a in attempts if a["decision"] == "REPLACE")
    n_blocked = sum(1 for a in attempts if a["decision"] == "KEEP")
    n_rolled_back = sum(1 for a in attempts if a["decision"] == "ROLLED_BACK")

    alpha_detail = _replace_alpha_detail(result, prepared["all_data"], prepared["all_dates"])
    alpha_summary = _stage2_summary(alpha_detail)

    return {
        "config": name, "window": window_label,
        "总收益%": metrics["收益率%"], "年化收益%": round(result["annualized_return_pct"], 2),
        "Sharpe": metrics["Sharpe"], "Sortino": metrics["Sortino"], "Calmar": metrics["Calmar"],
        "最大回撤%": metrics["MDD%"], "交易次数": metrics["交易数"], "胜率%": metrics["胜率%"],
        "平均持仓天数": _avg_holding_days(result["trade_log"]),
        "Replacement Success": n_success, "Replacement Blocked": n_blocked,
        "Replacement Rolled Back": n_rolled_back,
        "Replace Alpha 均值%": alpha_summary.get("mean_alpha_pct"),
        "Replace Alpha 中位数%": alpha_summary.get("median_alpha_pct"),
        "Replace Alpha 胜率%": alpha_summary.get("win_rate_pct"),
        "Replace Alpha n": alpha_summary.get("n", 0),
    }


def _verdict(df: pd.DataFrame) -> pd.DataFrame:
    """跟baseline比较，看每个非baseline配置是否在两个窗口都同时满足
    "四项都不变差"（收益容忍-5pp、Sharpe/Replace Alpha均值不降、回撤
    不恶化）——用户设定的"稳定改善"标准，不是"综合打分"。"""
    rows = []
    for cfg in ABLATION_CONFIGS:
        if cfg == "baseline":
            continue
        stable_pass = True
        detail = {}
        for _, _, wl in WINDOWS:
            base = df[(df["config"] == "baseline") & (df["window"] == wl)].iloc[0]
            cur = df[(df["config"] == cfg) & (df["window"] == wl)].iloc[0]
            ret_ok = cur["总收益%"] >= base["总收益%"] - 5.0
            sharpe_ok = cur["Sharpe"] >= base["Sharpe"]
            mdd_ok = cur["最大回撤%"] >= base["最大回撤%"]   # 负数，越接近0越好
            alpha_ok = (cur["Replace Alpha 均值%"] is not None and base["Replace Alpha 均值%"] is not None
                        and cur["Replace Alpha 均值%"] >= base["Replace Alpha 均值%"])
            window_pass = ret_ok and sharpe_ok and mdd_ok and alpha_ok
            stable_pass = stable_pass and window_pass
            detail[wl] = {"收益达标": ret_ok, "Sharpe达标": sharpe_ok,
                           "回撤达标": mdd_ok, "AlphaC达标": alpha_ok, "本窗口通过": window_pass}
        rows.append({"config": cfg, "两窗口均稳定改善": stable_pass, **{
            f"{wl}_{k}": v for wl in detail for k, v in detail[wl].items()}})
    return pd.DataFrame(rows)


def main():
    open(LOG_PATH, "w", encoding="utf-8").close()
    all_rows = []

    for start, end, window_label in WINDOWS:
        print(f"拉取 {start} ~ {end}（{window_label}），纯美股池（{len(POOL)}只）...")
        with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            prepared = bp._prepare_backtest_data(POOL, start, end, usd_to_jpy=None)
        if prepared is None:
            print(f"NO DATA for {window_label} -- skipping")
            continue

        for name, overrides in ABLATION_CONFIGS.items():
            print(f"  模拟 [{window_label}] config={name} overrides={overrides} ...")
            result = _run(prepared, overrides)
            all_rows.append(_row_metrics(name, window_label, result, prepared))

    df = pd.DataFrame(all_rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    print("\n" + "=" * 130)
    print("一、全部8组配置 x 2个窗口 完整指标对比")
    print("=" * 130)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT_DIR, "replace_ablation_full_results.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 130)
    print("二、跨窗口稳定性判定（收益容忍-5pp内 + Sharpe不降 + 回撤不恶化 + Replace Alpha均值不降，两个窗口都要成立）")
    print("=" * 130)
    verdict = _verdict(df)
    print(verdict.to_string(index=False))
    verdict.to_csv(os.path.join(OUT_DIR, "replace_ablation_verdict.csv"), index=False, encoding="utf-8-sig")

    passed = verdict[verdict["两窗口均稳定改善"]]["config"].tolist() if len(verdict) else []
    print("\n" + "=" * 130)
    print("三、最终结论")
    print("=" * 130)
    if passed:
        print(f">>> 通过两窗口稳定性验证、可以考虑转正的配置: {passed}")
    else:
        print(">>> 没有任何配置在两个窗口都同时稳定改善收益/Sharpe/回撤/Replace Alpha。")
        print(">>> 按用户设定的判定标准，v2.4 Replace Engine 重构到此为止，")
        print(">>> 维持v2.3现状（不加任何新过滤条件），后续方向转向v3.0多维决策模型。")


if __name__ == "__main__":
    main()
