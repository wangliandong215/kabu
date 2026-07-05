"""
analytics/replace_decision_feature_analysis.py -- v2.4 优化阶段继续完善
（不升级版本号）：REPLACEMENT_MARGIN 参数扫描已确认触及能力边界（见
margin_optimization_report.py 的结论：6~20区间完全不生效，27~60区间
非单调、没有单一取值同时命中四项目标）。这一步不再搜参数，转向纯数据
分析：为什么 score_difference 对 Replace 是否有价值的预测力弱？除了
score_difference，还有哪些候选特征值得未来考虑纳入 Replace 决策？

**严格边界（用户明确要求）**：不修改任何策略规则/评分逻辑，不训练任何
模型，只做统计分析（相关性/分组统计/互信息），产出的是"候选变量"报告，
不是新的决策规则本身。

方法论：
  1. 数据来源：单一margin取值(=45)的一次完整回测——PRD原始生产margin
     (10)的Blocked样本数为0（见上一阶段发现，score_difference min=27.19，
     margin<=25时从未真正拦截过任何候选），无法用来分析"被拦截的候选
     后来表现如何"；margin=45这一次回测里REPLACE(108)/KEEP(361)/
     ROLLED_BACK(34)三类样本都有足够数量，且全部来自同一个内部自洽的
     单次模拟（不跨margin混合，避免路径依赖导致的样本非独立性问题）。
  2. 统一结果标签：不用"实际最终平仓收益"（只有REPLACE类才有，KEEP/
     ROLLED_BACK类候选从未真正建仓，没有实际交易可言），改用对全部
     三类样本都能算的统一代理指标——从审查当天起，candidate和victim
     各自未来N个交易日的真实价格收益之差（Alpha_Nd = candidate_ret_Nd -
     victim_ret_Nd，N∈{5,10,20,60}，主报告用20日作为标题指标）。这跟
     margin_optimization_report.py 阶段三 Blocked Opportunity Analysis
     的方法完全同源，只是这次同时套用在全部三类决策上，让三类样本可以
     放在同一张表里比较。
  3. 候选特征（全部来自已有数据的观测统计，不引入新指标/新模型）：
     score_difference/old_score/new_score（现有）、victim_holding_days/
     victim_strategy/incoming_strategy/n_observation_pool_size/
     weather_code（2026-07-05在backtest_portfolio.py里新增的纯观测
     字段）、victim_atr_pct（victim_entry_atr/victim_avg_cost）、
     same_sector（config.SECTOR_MAP）、victim/candidate各自的近20日
     动量与波动率（从all_data价格序列直接算，观察窗口在审查日之前，
     不引入未来信息）、day_of_week/month（季节性）。
  4. 统计方法：数值特征 vs Alpha_20d 算 Pearson/Spearman相关系数 + 经验
     互信息（等频分箱估计，纯统计诊断，不是训练模型）；类别特征算分组
     均值/中位数/样本量表。全部特征按 |Spearman| 排序给出候选变量清单。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python analytics/replace_decision_feature_analysis.py
"""
import contextlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest_portfolio as bp
import config

START = "2015-01-01"
END   = "2026-07-03"
CASH  = 50_000.0
POOL  = config.WATCHLIST_EUROPE_US

ANALYSIS_MARGIN = 45.0   # 见模块docstring：唯一在单次回测里三类样本都充足的取值
HORIZONS = [5, 10, 20, 60]
HEADLINE_HORIZON = 20
MI_BINS = 5

LOG_PATH = os.path.join(os.path.dirname(__file__), "_replace_decision_feature_analysis.log")
OUT_DIR  = os.path.dirname(__file__)

CATEGORICAL_FEATURES = ["decision", "victim_strategy", "incoming_strategy",
                          "same_sector", "weather_code"]
NUMERIC_FEATURES = ["score_difference", "old_score", "new_score",
                     "victim_holding_days", "victim_atr_pct",
                     "n_observation_pool_size",
                     "victim_recent_return_20d", "victim_recent_vol_20d",
                     "candidate_recent_return_20d", "candidate_recent_vol_20d",
                     "day_of_week", "month"]


def _lookback_return_vol(df: pd.DataFrame, date, window: int = 20):
    """审查日之前(不含当天)window个交易日的价格收益率+日收益率标准差
    （年化前的原始日波动率）。用于捕捉"这只股票审查前处于什么动量/波动
    状态"，只用历史数据，不引入未来信息。"""
    ts = pd.Timestamp(date)
    if ts not in df.index:
        return None, None
    pos = df.index.get_loc(ts)
    start = pos - window
    if start < 0:
        return None, None
    window_close = df["close"].iloc[start:pos + 1].astype(float)
    if len(window_close) < window + 1:
        return None, None
    ret = float(window_close.iloc[-1] / window_close.iloc[0] - 1)
    daily_ret = window_close.pct_change().dropna()
    vol = float(daily_ret.std()) if len(daily_ret) > 1 else None
    return ret, vol


def _forward_return(df: pd.DataFrame, date, horizon: int):
    ts = pd.Timestamp(date)
    if ts not in df.index:
        return None
    pos = df.index.get_loc(ts)
    target = pos + horizon
    if target >= len(df):
        return None
    return float(df["close"].iloc[target]) / float(df["close"].iloc[pos]) - 1


def _build_sample(result: dict, all_data: dict) -> pd.DataFrame:
    attempts = result["replacement_attempts"]
    rows = []
    for a in attempts:
        victim_code, incoming_code, date = a["victim_code"], a["incoming_code"], a["date"]
        df_v, df_i = all_data.get(victim_code), all_data.get(incoming_code)
        if df_v is None or df_i is None:
            continue

        row = dict(a)   # 拷贝所有已有字段（score_difference/decision/victim_holding_days等）
        row["same_sector"] = (config.SECTOR_MAP.get(victim_code, "other")
                               == config.SECTOR_MAP.get(incoming_code, "other"))
        row["victim_atr_pct"] = (
            a["victim_entry_atr"] / a["victim_avg_cost"]
            if a.get("victim_entry_atr") and a.get("victim_avg_cost") else None)
        row["day_of_week"] = pd.Timestamp(date).dayofweek
        row["month"] = pd.Timestamp(date).month

        v_ret, v_vol = _lookback_return_vol(df_v, date)
        i_ret, i_vol = _lookback_return_vol(df_i, date)
        row["victim_recent_return_20d"] = v_ret
        row["victim_recent_vol_20d"] = v_vol
        row["candidate_recent_return_20d"] = i_ret
        row["candidate_recent_vol_20d"] = i_vol

        for h in HORIZONS:
            v_fwd = _forward_return(df_v, date, h)
            i_fwd = _forward_return(df_i, date, h)
            row[f"alpha_{h}d"] = (i_fwd - v_fwd) if (v_fwd is not None and i_fwd is not None) else None

        rows.append(row)
    return pd.DataFrame(rows)


def _pearson_spearman(df: pd.DataFrame, feature: str, label: str):
    sub = df[[feature, label]].dropna()
    if len(sub) < 10 or sub[feature].nunique() < 2:
        return None, None, len(sub)
    pearson = float(sub[feature].corr(sub[label], method="pearson"))
    # pandas .corr(method="spearman") 内部会 import scipy.stats.spearmanr，
    # 这台环境没装 scipy——Spearman本质就是对两列先取秩(rank)再算Pearson，
    # 手动实现等价，不需要额外依赖。
    spearman = float(sub[feature].rank().corr(sub[label].rank(), method="pearson"))
    return pearson, spearman, len(sub)


def _mutual_info(df: pd.DataFrame, feature: str, label: str, bins: int = MI_BINS) -> float:
    """经验互信息（bits），等频分箱估计——纯统计诊断，不是训练模型。"""
    sub = df[[feature, label]].dropna()
    if len(sub) < bins * 4:
        return float("nan")
    try:
        xb = pd.qcut(sub[feature], q=bins, duplicates="drop")
        yb = pd.qcut(sub[label], q=bins, duplicates="drop")
    except (ValueError, IndexError):
        return float("nan")
    if xb.nunique() < 2 or yb.nunique() < 2:
        return float("nan")
    joint = pd.crosstab(xb, yb, normalize=True)
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    mi = 0.0
    for i in joint.index:
        for j in joint.columns:
            pij = joint.loc[i, j]
            if pij > 0:
                mi += pij * np.log2(pij / (px[i] * py[j]))
    return float(mi)


def _numeric_feature_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for feat in NUMERIC_FEATURES:
        pearson, spearman, n = _pearson_spearman(df, feat, label)
        mi = _mutual_info(df, feat, label)
        rows.append({"feature": feat, "n": n,
                      "pearson_r": round(pearson, 4) if pearson is not None else None,
                      "spearman_rho": round(spearman, 4) if spearman is not None else None,
                      "mutual_info_bits": round(mi, 4) if not np.isnan(mi) else None})
    out = pd.DataFrame(rows)
    out["abs_spearman"] = out["spearman_rho"].abs()
    return out.sort_values("abs_spearman", ascending=False).drop(columns="abs_spearman")


def _categorical_group_table(df: pd.DataFrame, feature: str, label: str) -> pd.DataFrame:
    sub = df[[feature, label]].dropna()
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(feature)[label].agg(["count", "mean", "median", "std"])
    g["win_rate_pct"] = sub.groupby(feature)[label].apply(lambda s: (s > 0).mean() * 100)
    return g.reset_index().sort_values("mean", ascending=False)


def main():
    open(LOG_PATH, "w", encoding="utf-8").close()

    print(f"拉取 {START} ~ {END}，纯美股池 config.WATCHLIST_EUROPE_US（{len(POOL)}只）...")
    with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        prepared = bp._prepare_backtest_data(POOL, START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    orig_margin = config.REPLACEMENT_MARGIN
    print(f"模拟 REPLACEMENT_MARGIN={ANALYSIS_MARGIN}（唯一三类样本都充足的取值）...")
    config.REPLACEMENT_MARGIN = ANALYSIS_MARGIN
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            result = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")
    finally:
        config.REPLACEMENT_MARGIN = orig_margin

    df = _build_sample(result, prepared["all_data"])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\n" + "=" * 110)
    print("一、样本总览")
    print("=" * 110)
    print(df["decision"].value_counts().to_string())
    df.to_csv(os.path.join(OUT_DIR, "replace_decision_samples.csv"), index=False, encoding="utf-8-sig")
    print(f"\n全量样本已保存: analytics/replace_decision_samples.csv（{len(df)}行）")

    label = f"alpha_{HEADLINE_HORIZON}d"
    print("\n" + "=" * 110)
    print(f"二、score_difference 单独核对：跟 {label} 的相关性（回答'为什么预测力弱'）")
    print("=" * 110)
    pearson, spearman, n = _pearson_spearman(df, "score_difference", label)
    mi = _mutual_info(df, "score_difference", label)
    print(f"n={n}  Pearson r={pearson}  Spearman rho={spearman}  互信息={round(mi, 4) if not np.isnan(mi) else None} bits")
    for h in HORIZONS:
        p, s, nh = _pearson_spearman(df, "score_difference", f"alpha_{h}d")
        print(f"  horizon={h}d: n={nh}  Pearson={p}  Spearman={s}")

    print("\n" + "=" * 110)
    print(f"三、全部数值候选特征 vs {label}（按|Spearman|降序）")
    print("=" * 110)
    numeric_table = _numeric_feature_table(df, label)
    print(numeric_table.to_string(index=False))
    numeric_table.to_csv(os.path.join(OUT_DIR, "replace_feature_numeric_correlations.csv"),
                          index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print(f"四、类别特征分组统计 vs {label}")
    print("=" * 110)
    cat_tables = {}
    for feat in CATEGORICAL_FEATURES:
        t = _categorical_group_table(df, feat, label)
        cat_tables[feat] = t
        print(f"\n-- {feat} --")
        print(t.to_string(index=False) if not t.empty else "(无有效样本)")
        if not t.empty:
            t.to_csv(os.path.join(OUT_DIR, f"replace_feature_group_{feat}.csv"),
                      index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print("五、按四个观察周期重新核对Top候选特征的稳健性")
    print("=" * 110)
    top_features = numeric_table.head(5)["feature"].tolist()
    rows = []
    for feat in top_features:
        row = {"feature": feat}
        for h in HORIZONS:
            _, s, _ = _pearson_spearman(df, feat, f"alpha_{h}d")
            row[f"spearman_{h}d"] = s
        rows.append(row)
    robustness = pd.DataFrame(rows)
    print(robustness.to_string(index=False))
    robustness.to_csv(os.path.join(OUT_DIR, "replace_feature_robustness_by_horizon.csv"),
                       index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print("六、结论：最值得加入 Replace 决策的候选变量")
    print("=" * 110)
    print(f"score_difference 在 {label} 上的 Spearman={spearman}（{'弱' if abs(spearman or 0) < 0.15 else '中等'}相关），")
    print("对照下面Top候选特征表判断是否有明显更强的信号：")
    print(numeric_table.head(5).to_string(index=False))
    print("\n(以上仅为候选变量的统计证据，是否真正纳入Replace决策规则需要用户拍板，")
    print(" 本阶段严格不改动任何策略/评分逻辑，产出的是数据分析报告，不是新规则。)")


if __name__ == "__main__":
    main()
