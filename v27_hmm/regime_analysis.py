# -*- coding: utf-8 -*-
"""
V2.7 第二阶段 —— 市场状态深度验证分析

本模块只负责"验证 HMM 是否学到了真实市场状态"相关的统计分析（状态统计表、
持续时间、年度分布、转移概率、特征画像、质量自检），不涉及模型训练、回测或
收益率优化，与 hmm_model.py / backtest.py 职责分离，不改变整体流水线架构。
"""

import numpy as np
import pandas as pd

from config import FEATURE_LIST, PERSISTENCE_TARGET_DURATION, CHATTERING_THRESHOLD


def full_state_statistics(
    df: pd.DataFrame, state_col: str = "state", label_col: str = "label"
) -> pd.DataFrame:
    """
    每个状态输出：样本数量、占全部样本比例、平均收益率、收益率标准差、
    平均ATR、平均波动率、平均成交量变化、平均日振幅（hl_range）。
    """
    total = len(df)
    stats = df.groupby(state_col).agg(
        label=(label_col, "first"),
        samples=(state_col, "count"),
        mean_return=("log_return", "mean"),
        return_std=("log_return", "std"),
        avg_atr=("atr", "mean"),
        avg_volatility=("volatility", "mean"),
        avg_volume_change=("volume_change", "mean"),
        avg_hl_range=("hl_range", "mean"),
    )
    stats["pct_of_total"] = stats["samples"] / total
    ordered_cols = [
        "label",
        "samples",
        "pct_of_total",
        "mean_return",
        "return_std",
        "avg_atr",
        "avg_volatility",
        "avg_volume_change",
        "avg_hl_range",
    ]
    return stats[ordered_cols].sort_values("mean_return", ascending=False)


def print_state_statistics(stats_df: pd.DataFrame):
    print("\n======== State Statistics ========")
    for state_id, row in stats_df.iterrows():
        print(f"\nState {state_id}")
        print(f"Label：{row['label']}")
        print(f"Samples：{int(row['samples'])}")
        print(f"Pct of Total：{row['pct_of_total']:.2%}")
        print(f"Mean Return：{row['mean_return']:.2%}")
        print(f"Return Std：{row['return_std']:.2%}")
        print(f"ATR：{row['avg_atr']:.2%}")
        print(f"Volatility：{row['avg_volatility']:.2%}")
        print(f"Volume Change：{row['avg_volume_change']:.2%}")
        print(f"Avg Daily Range (HL)：{row['avg_hl_range']:.2%}")
    print("===================================")


def compute_state_durations(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    统计每个状态"连续持续段"的天数分布：平均/中位/最长/最短持续天数。

    做法：找出 label 连续不变的分段，每一段的长度即一次"状态持续期"，
    再按 label 汇总这些持续期样本的统计量，用于分析状态稳定性。
    """
    labels = df[label_col].values
    durations = []
    seg_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[seg_start]:
            durations.append((labels[seg_start], i - seg_start))
            seg_start = i

    dur_df = pd.DataFrame(durations, columns=["label", "duration"])
    if dur_df.empty:
        return pd.DataFrame(
            columns=["avg_duration", "median_duration", "max_duration", "min_duration", "n_episodes"]
        )

    summary = dur_df.groupby("label")["duration"].agg(
        avg_duration="mean",
        median_duration="median",
        max_duration="max",
        min_duration="min",
        n_episodes="count",
    )
    return summary


def print_duration_statistics(duration_df: pd.DataFrame):
    print("\n======== State Duration Statistics ========")
    for label, row in duration_df.iterrows():
        print(f"\n{label}")
        print(f"Average Duration：{row['avg_duration']:.1f} Days")
        print(f"Median Duration：{row['median_duration']:.1f} Days")
        print(f"Longest：{int(row['max_duration'])} Days")
        print(f"Shortest：{int(row['min_duration'])} Days")
        print(f"Episodes：{int(row['n_episodes'])}")
    print("=============================================")


def annual_state_distribution(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    按年份统计各状态占比，用于观察是否符合真实市场（例如某年 Bear 占比是否明显升高）。
    """
    years = df.index.year
    table = pd.crosstab(years, df[label_col], normalize="index")
    table.index.name = "Year"
    return table


def print_annual_distribution(annual_df: pd.DataFrame):
    print("\n======== Annual State Distribution ========")
    for year, row in annual_df.iterrows():
        print(f"\n{year}")
        for label in annual_df.columns:
            print(f"{label}：{row[label]:.0%}")
    print("=============================================")


def label_transition_matrix(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    在"人类可读的 Bull/Bear/Sideways 标签"层面统计转移概率。

    直接从解码后的状态序列统计"今天状态X -> 明天状态Y"的经验频率，而不是
    直接照搬 model.transmat_：当隐藏状态数 > 3 时，多个原始隐藏状态编号可能
    被映射到同一个 label（如多个 Sideways 子状态），需要先按 label 聚合
    再统计转移频率，才能得到"Bull整体转到Bear的概率"这种可解释指标。
    """
    labels = df[label_col].values
    trans_df = pd.DataFrame({"from": labels[:-1], "to": labels[1:]})
    counts = pd.crosstab(trans_df["from"], trans_df["to"])
    probs = counts.div(counts.sum(axis=1), axis=0)
    return probs


def print_transition_probabilities(trans_df: pd.DataFrame):
    print("\n======== Label Transition Probabilities ========")
    for from_label in trans_df.index:
        print(f"\n{from_label}")
        row = trans_df.loc[from_label].sort_values(ascending=False)
        for to_label, prob in row.items():
            print(f"  -> {to_label}：{prob:.0%}")
    print("==================================================")


def feature_means_by_state(
    df: pd.DataFrame,
    feature_list: list = None,
    state_col: str = "state",
    label_col: str = "label",
) -> pd.DataFrame:
    """输出每个状态在原始特征尺度（未标准化）下的均值。"""
    if feature_list is None:
        feature_list = FEATURE_LIST
    means = df.groupby(state_col)[feature_list].mean()
    means.insert(0, "label", df.groupby(state_col)[label_col].first())
    return means


def qualitative_feature_profile(
    means_df: pd.DataFrame, feature_list: list = None
) -> pd.DataFrame:
    """
    把每个状态的特征均值转换成 High/Mid/Low 的定性描述（基于跨状态的 z-score），
    方便直接读出"Bull: Log Return 高、Volatility 低、Volume 高"这样的结论，
    帮助理解 HMM 为什么划分出这些状态。
    """
    if feature_list is None:
        feature_list = FEATURE_LIST
    z = (means_df[feature_list] - means_df[feature_list].mean()) / means_df[feature_list].std(ddof=0)

    def qualitative(v):
        if pd.isna(v):
            return "Mid"
        if v > 0.3:
            return "High"
        if v < -0.3:
            return "Low"
        return "Mid"

    profile = z.map(qualitative)
    profile.insert(0, "label", means_df["label"])
    return profile


def print_feature_profile(profile_df: pd.DataFrame, means_df: pd.DataFrame, feature_list: list = None):
    if feature_list is None:
        feature_list = FEATURE_LIST
    print("\n======== Feature Importance / State Profile ========")
    for state_id in profile_df.index:
        print(f"\nState {state_id}（{profile_df.loc[state_id, 'label']}）")
        for feature in feature_list:
            qual = profile_df.loc[state_id, feature]
            raw = means_df.loc[state_id, feature]
            sign = "+" if raw > 0 else ("-" if raw < 0 else "0")
            print(f"{feature}：{qual} ({sign})")
    print("======================================================")


def regime_quality_report(
    state_stats: pd.DataFrame,
    duration_stats: pd.DataFrame,
    transition_probs: pd.DataFrame,
) -> dict:
    """
    自动化验收检查（对应任务书第9/10节：与 V2.6 对比 + 验收标准）。

    本阶段不比较收益率，只判断 HMM 是否"稳定识别出 Bull/Bear/Sideways
    且具有统计学意义"，输出布尔检查项，供人工最终确认 Market Regime
    Detection 是否初步成功。
    """
    checks = {}
    labels_present = set(state_stats["label"])

    checks["Bull 状态存在且平均收益率为正"] = (
        "Bull" in labels_present
        and state_stats.loc[state_stats["label"] == "Bull", "mean_return"].mean() > 0
    )
    checks["Bear 状态存在且平均收益率为负"] = (
        "Bear" in labels_present
        and state_stats.loc[state_stats["label"] == "Bear", "mean_return"].mean() < 0
    )

    if "Sideways" in labels_present and "Bull" in labels_present and "Bear" in labels_present:
        bear_ret = state_stats.loc[state_stats["label"] == "Bear", "mean_return"].mean()
        side_ret = state_stats.loc[state_stats["label"] == "Sideways", "mean_return"].mean()
        bull_ret = state_stats.loc[state_stats["label"] == "Bull", "mean_return"].mean()
        checks["Sideways 平均收益率介于 Bull/Bear 之间"] = bear_ret <= side_ret <= bull_ret
    else:
        checks["Sideways 平均收益率介于 Bull/Bear 之间"] = True

    checks["各状态样本占比均 >= 5%（无退化为噪声的极小状态）"] = bool(
        (state_stats["pct_of_total"] >= 0.05).all()
    )
    checks["各状态平均持续天数均 >= 3 天（非逐日跳变的噪声状态）"] = bool(
        (duration_stats["avg_duration"] >= 3).all()
    )
    checks["转移矩阵对角线（状态自持概率）均 >= 50%（状态具有粘滞性/稳定性）"] = all(
        transition_probs.loc[l, l] >= 0.5 for l in transition_probs.index if l in transition_probs.columns
    )

    return checks


def print_regime_quality_report(checks: dict):
    print("\n======== Regime Detection Quality Check（对应第9/10节验收标准） ========")
    for name, passed in checks.items():
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}")
    print("==========================================================================")


def regime_persistence_score(duration_stats: pd.DataFrame, target_duration: int = None) -> dict:
    """
    第2节：状态稳定性（Regime Persistence）。

    对每个状态的平均持续天数，相对 PERSISTENCE_TARGET_DURATION（默认20个交易日，
    约等于一个月）做归一化打分：avg_duration 达到或超过目标天数记满分100，
    不足则按比例线性打分。整体 Persistence Score 取各状态得分的简单平均——
    不按样本占比加权，因为占比小但极不稳定的状态（如日内反复跳变的 Sideways）
    正是 State Chattering 最需要暴露出来的信号，加权平均会把它稀释掉。
    """
    target = target_duration if target_duration is not None else PERSISTENCE_TARGET_DURATION
    if duration_stats.empty:
        return {"per_label": {}, "overall": 0.0}
    per_label = {
        label: float(min(100.0, 100.0 * row["avg_duration"] / target))
        for label, row in duration_stats.iterrows()
    }
    overall = float(np.mean(list(per_label.values())))
    return {"per_label": per_label, "overall": overall}


def print_persistence_score(score: dict):
    print("\n======== Regime Persistence Score ========")
    for label, s in score["per_label"].items():
        print(f"{label}：{s:.0f}")
    print(f"\nPersistence Score：{score['overall']:.0f}")
    print("=============================================")


def state_separation_score(
    df: pd.DataFrame, feature_list: list = None, state_col: str = "state"
) -> dict:
    """
    第3节：状态分离程度（State Separation）。

    对每个特征，计算"状态间均值极差 / 状态内混合（pooled）标准差"，这是单因素
    方差分析里效应量（effect size）的简化版本：ratio 越大说明该特征在不同状态间
    区分度越强。再用 100*(1 - 1/(1+ratio)) 压缩到 [0,100) 区间，ratio=0（各状态
    该特征均值完全相同）对应0分，ratio 越大越接近100分。

    多个状态在全部特征上都高度重叠（Separation Score 很低）意味着状态数量可能
    设置过多，HMM 把本质上同一种市场行为拆成了多个统计特征无法区分的状态。
    """
    if feature_list is None:
        feature_list = FEATURE_LIST
    grouped = df.groupby(state_col)
    per_feature = {}
    for feature in feature_list:
        means = grouped[feature].mean()
        stds = grouped[feature].std().fillna(0.0)
        counts = grouped[feature].count()
        dof = (counts - 1).clip(lower=0)
        total_dof = dof.sum()
        pooled_var = (stds**2 * dof).sum() / total_dof if total_dof > 0 else 0.0
        pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 0.0
        gap = means.max() - means.min()
        ratio = gap / (pooled_std + 1e-9)
        per_feature[feature] = float(100.0 * (1.0 - 1.0 / (1.0 + ratio)))
    overall = float(np.mean(list(per_feature.values()))) if per_feature else 0.0
    return {"per_feature": per_feature, "overall": overall}


def print_separation_score(score: dict):
    print("\n======== State Separation Score ========")
    for feature, s in score["per_feature"].items():
        print(f"{feature}：{s:.0f}")
    print(f"\nSeparation Score：{score['overall']:.0f}")
    if score["overall"] < 40:
        print("[WARNING] Separation Score 偏低，多个状态统计特征高度重叠，状态数量可能过多")
    print("============================================")


def transition_stability_report(transition_probs: pd.DataFrame, threshold: float = None) -> dict:
    """
    第4节：Transition Matrix 稳定性分析。

    统计每个状态的自持概率（今天处于该状态，明天仍处于该状态的经验概率），
    以及全部状态的平均自持概率（Self Transition Probability）。平均自持概率
    低于 CHATTERING_THRESHOLD（默认50%）时判定存在 State Chattering（状态抖动）风险——
    状态在不同时刻之间反复跳变，缺乏可交易的持续性。
    """
    threshold = threshold if threshold is not None else CHATTERING_THRESHOLD
    self_probs = {
        label: float(transition_probs.loc[label, label])
        for label in transition_probs.index
        if label in transition_probs.columns
    }
    avg_self_prob = float(np.mean(list(self_probs.values()))) if self_probs else 0.0
    return {
        "self_probs": self_probs,
        "avg_self_prob": avg_self_prob,
        "chattering": avg_self_prob < threshold,
        "score": avg_self_prob * 100.0,
    }


def print_transition_stability(report: dict):
    print("\n======== Transition Matrix Stability ========")
    for label, prob in report["self_probs"].items():
        print(f"{label} 自持概率：{prob:.0%}")
    print(f"\n平均自持概率（Self Transition Probability）：{report['avg_self_prob']:.0%}")
    if report["chattering"]:
        print("[WARNING] 平均自持概率低于 50%，存在 State Chattering（状态抖动）风险")
    print("===============================================")


def interpretability_score(profile_df: pd.DataFrame, feature_list: list = None) -> float:
    """
    可解释性打分（第6节 Regime Quality Report 的一个维度）。

    qualitative_feature_profile() 把每个 (状态, 特征) 的相对水平压缩成
    High/Mid/Low。"Mid" 代表该特征在这个状态下与其他状态区分度不高，
    可解释性弱；High/Low 占比越高，说明各状态的特征画像越鲜明、越容易向人解释。
    """
    if feature_list is None:
        feature_list = FEATURE_LIST
    values = profile_df[feature_list].values.flatten()
    if len(values) == 0:
        return 0.0
    non_mid = sum(1 for v in values if v != "Mid")
    return 100.0 * non_mid / len(values)


def backtest_performance_score(sharpe: float) -> float:
    """
    用 Sharpe 比率经 sigmoid 压缩到 [0,100]，作为 Regime Quality Report 里
    "Backtest Performance" 维度的打分：Sharpe=0（无风险调整后超额收益）对应50分，
    Sharpe 越高越接近100分，越低（含负数）越接近0分，避免极端 Sharpe 把分数
    直接顶到边界之外，也不需要额外设定人为的归一化上下限。
    """
    return float(100.0 / (1.0 + np.exp(-sharpe)))


def regime_quality_scorecard(
    separation_score: float,
    persistence_score: float,
    transition_stability_score: float,
    interpretability_score: float,
    backtest_score: float,
) -> dict:
    """
    第6节：状态质量评分（Regime Quality Report）。

    把 State Separation / Persistence / Transition Stability / Interpretability /
    Backtest Performance 五个 0~100 分的子分数统一换算成 0~10 分制，并取算术平均
    作为 Overall 总分，用于快速判断当前状态数量下的 HMM 是否达到可用水平。
    """
    scores_100 = {
        "State Separation": separation_score,
        "Persistence": persistence_score,
        "Transition Stability": transition_stability_score,
        "Interpretability": interpretability_score,
        "Backtest Performance": backtest_score,
    }
    scores_10 = {k: v / 10.0 for k, v in scores_100.items()}
    scores_10["Overall"] = float(np.mean(list(scores_10.values())))
    return scores_10


def print_regime_quality_scorecard(scores_10: dict):
    print("\n================================")
    print("Regime Quality Report")
    print("================================")
    for name, score in scores_10.items():
        if name == "Overall":
            continue
        print(f"{name}：{score:.1f} / 10")
    print("--------------------------------")
    print(f"Overall：{scores_10['Overall']:.1f} / 10")
    print("================================")


def _rank_words(series: pd.Series) -> dict:
    """
    把一组按 label 索引的数值，在"当前状态数量下的全部状态"之间做相对排名，
    转成 最高/较高/中等/较低/最低 的定性描述（第7节：状态可解释性增强）。
    """
    n = len(series)
    if n == 0:
        return {}
    ranks = series.rank(method="first")
    words = {}
    for label, r in ranks.items():
        pct = (r - 1) / (n - 1) if n > 1 else 0.5
        if pct >= 0.999:
            words[label] = "最高"
        elif pct <= 0.001:
            words[label] = "最低"
        elif pct >= 0.66:
            words[label] = "较高"
        elif pct <= 0.34:
            words[label] = "较低"
        else:
            words[label] = "中等"
    return words


def _narrative_for_label(label: str, vol_word: str) -> str:
    if label == "Bull":
        return "典型趋势上涨市场"
    if label == "Bear":
        return "风险市场（下跌趋势，波动率通常偏高）"
    # Sideways / Sideways_k
    if vol_word in ("最高", "较高"):
        return "高波动震荡市场"
    if vol_word in ("最低", "较低"):
        return "低波动盘整市场"
    return "震荡整理市场"


def regime_interpretation(state_stats: pd.DataFrame, duration_stats: pd.DataFrame) -> dict:
    """
    第7节：状态可解释性增强。

    综合 mean_return / avg_volatility / avg_atr / avg_volume_change（来自
    full_state_statistics，按 label 索引）与 avg_duration（来自
    compute_state_durations，同样按 label 索引），对每个指标在"当前状态数量下
    的全部状态"之间做相对排名，转成定性描述，再拼出一句概括性解释，
    生成类似"Bull：收益最高，波动率较低……解释：典型趋势上涨市场"的画像。
    """
    by_label = state_stats.reset_index().set_index("label")
    return_words = _rank_words(by_label["mean_return"])
    vol_words = _rank_words(by_label["avg_volatility"])
    atr_words = _rank_words(by_label["avg_atr"])
    volume_words = _rank_words(by_label["avg_volume_change"])
    duration_words = _rank_words(duration_stats["avg_duration"]) if not duration_stats.empty else {}

    interpretations = {}
    for label in by_label.index:
        interpretations[label] = {
            "mean_return": return_words.get(label, "中等"),
            "volatility": vol_words.get(label, "中等"),
            "atr": atr_words.get(label, "中等"),
            "volume": volume_words.get(label, "中等"),
            "duration": duration_words.get(label, "中等"),
            "narrative": _narrative_for_label(label, vol_words.get(label, "中等")),
        }
    return interpretations


def print_regime_interpretation(interpretations: dict):
    print("\n======== Regime Interpretability Report ========")
    for label, info in interpretations.items():
        print(f"\n{label}")
        print(f"平均收益：{info['mean_return']}")
        print(f"波动率：{info['volatility']}")
        print(f"ATR：{info['atr']}")
        print(f"成交量：{info['volume']}")
        print(f"持续时间：{info['duration']}")
        print(f"解释：{info['narrative']}")
    print("===================================================")
