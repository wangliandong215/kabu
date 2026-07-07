# -*- coding: utf-8 -*-
"""
V2.7 HMM 市场状态识别模块 —— 主程序调度入口

流程：加载数据（无真实CSV则自动生成模拟K线）-> 清洗/特征工程 -> 训练/测试集切分
      -> HMM 训练与自适应状态标签映射 -> 回测 -> 风控指标评估 -> 四合一可视化。
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    TRAIN_RATIO,
    N_STATES,
    FEATURE_LIST,
    ANNUAL_PERIODS,
    STATES_SEARCH_RANGE,
    OVERALL_SCORE_WEIGHTS,
    get_state_label_order,
    get_state_colors,
)
from features import prepare_dataset, scale_features, add_atr
from hmm_model import (
    train_hmm,
    decode_states,
    print_model_diagnostics,
    map_states_to_labels,
    get_state_statistics,
    compute_aic_bic,
    search_best_n_states,
    print_model_selection_report,
)
from regime_analysis import (
    full_state_statistics,
    print_state_statistics,
    compute_state_durations,
    print_duration_statistics,
    annual_state_distribution,
    print_annual_distribution,
    label_transition_matrix,
    print_transition_probabilities,
    feature_means_by_state,
    qualitative_feature_profile,
    print_feature_profile,
    regime_quality_report,
    print_regime_quality_report,
    regime_persistence_score,
    print_persistence_score,
    state_separation_score,
    print_separation_score,
    transition_stability_report,
    print_transition_stability,
    interpretability_score,
    backtest_performance_score,
    regime_quality_scorecard,
    print_regime_quality_scorecard,
    regime_interpretation,
    print_regime_interpretation,
)
from backtest import run_backtest
from metrics import compute_strategy_and_benchmark_metrics
from visualization import plot_full_report


def generate_mock_kline(n_days: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    生成一段包含牛市/熊市/震荡市轮动的模拟 K 线数据，仅用于在没有真实历史数据时
    让整套系统能够直接跑通演示。规则：随机切分若干段，每段随机赋予一种"真实regime"
    （牛/熊/震荡），并用不同的日收益率均值/波动率生成该段价格路径，最后拼接成完整序列。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2019-01-01", periods=n_days)

    # regime 参数：(日均收益率, 日波动率)
    regime_params = {
        "bull": (0.0009, 0.010),
        "bear": (-0.0009, 0.016),
        "sideways": (0.0000, 0.007),
    }
    regime_names = list(regime_params.keys())

    log_returns = np.empty(n_days)
    volumes = np.empty(n_days)
    idx = 0
    while idx < n_days:
        seg_len = rng.integers(40, 120)
        seg_len = min(seg_len, n_days - idx)
        regime = regime_names[rng.integers(0, len(regime_names))]
        mu, sigma = regime_params[regime]
        log_returns[idx : idx + seg_len] = rng.normal(mu, sigma, seg_len)
        vol_base = {"bull": 1.2, "bear": 1.6, "sideways": 1.0}[regime]
        volumes[idx : idx + seg_len] = rng.lognormal(
            mean=np.log(1_000_000 * vol_base), sigma=0.25, size=seg_len
        )
        idx += seg_len

    close = 100.0 * np.exp(np.cumsum(log_returns))
    open_ = close * (1 + rng.normal(0, 0.001, n_days))
    intraday_range = np.abs(rng.normal(0, 0.006, n_days)) + 0.002
    high = np.maximum(open_, close) * (1 + intraday_range)
    low = np.minimum(open_, close) * (1 - intraday_range)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volumes,
        },
        index=dates,
    )
    return df


def standardize_raw_csv(csv_path: str) -> pd.DataFrame:
    """
    将任意来源的真实历史 K 线 CSV 统一整理成系统内部约定的 schema：
    DatetimeIndex（按日期）+ Open/High/Low/Close/Volume 五列。

    兼容：
      - 日期列名为 Date/date/Datetime/time_key 等常见写法，若均未命中则退回第一列；
      - OHLCV 列名大小写不完全一致（如 open/Open/OPEN）；
    这样只需保证 CSV 含"日期 + OHLCV"信息，换一只股票的数据文件无需改代码即可跑通。
    """
    raw = pd.read_csv(csv_path)

    date_candidates = ["Date", "date", "Datetime", "datetime", "time_key", "Time", "time"]
    date_col = next((c for c in raw.columns if c in date_candidates), raw.columns[0])
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.set_index(date_col)
    raw.index.name = "Date"

    col_map = {}
    for target in ["Open", "High", "Low", "Close", "Volume"]:
        match = next((c for c in raw.columns if c.strip().lower() == target.lower()), None)
        if match:
            col_map[match] = target
    raw = raw.rename(columns=col_map)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in raw.columns]
    if missing:
        raise ValueError(
            f"CSV 缺少必要列: {missing}（需要 日期 + Open/High/Low/Close/Volume，列名大小写不敏感）"
        )

    df = raw[required_cols]
    df = df[df.index.notna()]
    return df


def load_kline_data(csv_path: str) -> pd.DataFrame:
    """
    尝试从 csv_path 读取真实历史 K 线数据。若文件不存在，自动生成模拟数据
    以保证系统可以直接跑通。真实数据的日期排序/缺失值处理/类型转换/异常数据
    剔除统一交给 features.clean_data 完成（prepare_dataset 会调用它）。
    """
    if csv_path and os.path.exists(csv_path):
        df = standardize_raw_csv(csv_path)
        print(f"已加载真实历史数据: {csv_path}, 共 {len(df)} 条（清洗前）")
    else:
        print(f"未找到有效 CSV 路径（{csv_path}），自动生成模拟 K 线数据用于演示。")
        df = generate_mock_kline()
    return df


def split_train_test(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    """
    按时间顺序切分训练集/测试集（不能随机打乱，否则训练集会掺入"未来"的测试期数据）。
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def prepare_train_test(csv_path: str = None):
    """
    数据准备的公共前半段（加载 -> 清洗/特征工程 -> train/test 切分 -> 标准化），
    与具体用几个隐藏状态训练无关，因此从 run_pipeline 中抽出来，供
    run_pipeline（单一状态数）和 run_state_optimization（2~6 状态数量扫描）共用，
    避免每个候选状态数都重复一遍数据加载和特征工程。
    """
    raw_df = load_kline_data(csv_path)
    df = prepare_dataset(raw_df)
    # ATR 仅作为状态统计的展示性指标（第3节），不参与 HMM 训练特征。
    # add_atr 输出的是原始价格单位的绝对波幅，不同股票/不同价位区间不可比，
    # 这里除以 Close 换算成相对振幅（与 hl_range 同口径），才能用 :.2% 格式展示。
    df = add_atr(df)
    df["atr"] = df["atr"] / df["Close"]

    train_df, test_df = split_train_test(df, TRAIN_RATIO)

    # 【硬性要求：防信息泄露】StandardScaler 只在训练集上 fit，测试集只 transform
    train_scaled, test_scaled, scaler = scale_features(train_df, test_df, FEATURE_LIST)
    return train_df, test_df, train_scaled, test_scaled, scaler


def run_single_n_states(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_scaled,
    test_scaled,
    n_states: int,
    verbose: bool = True,
) -> dict:
    """
    对固定的 n_states 完整执行一次：HMM训练 -> 状态解码/自适应标签映射 -> 回测 ->
    全部统计分析（状态统计/持续时间/年度分布/转移概率/特征画像/Persistence/
    Separation/Transition Stability/可解释性/质量自检）。

    是 run_pipeline（单次固定状态数运行）与 run_state_optimization（2~6 状态数量
    扫描比较）共用的核心步骤，保证"单独跑一次"和"扫描比较里的某一次"结果完全一致。
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    if verbose:
        print(f"\n\n################ N_STATES = {n_states} ################")
        print("======== HMM Training ========")
        print(f"Training Samples：{len(train_df)}")
        print(f"Testing Samples：{len(test_df)}")
        print(f"Features：{FEATURE_LIST}")
        print(f"Hidden States：{n_states}")

    model = train_hmm(train_scaled, n_states=n_states)

    if verbose:
        log_likelihood = print_model_diagnostics(model, train_scaled, FEATURE_LIST)
        print(f"Log Likelihood：{log_likelihood:.4f}")
        print("Training Complete")
        print("==============================")

    # ---- 解码状态：训练集/测试集分别用各自的标准化特征做 Viterbi 解码 ----
    train_df["state"] = decode_states(model, train_scaled)
    test_df["state"] = decode_states(model, test_scaled)

    # 【自适应标签映射】映射规则只从训练集统计得出，测试集复用同一映射，
    # 避免用测试集的未来表现反过来定义"什么是牛市"这种信息泄露
    label_map = map_states_to_labels(train_df, state_col="state")
    train_df["label"] = train_df["state"].map(label_map)
    test_df["label"] = test_df["state"].map(label_map)

    full_df = pd.concat([train_df, test_df])

    full_state_stats = full_state_statistics(full_df, state_col="state", label_col="label")
    duration_stats = compute_state_durations(full_df, label_col="label")
    annual_dist = annual_state_distribution(full_df, label_col="label")
    transition_probs = label_transition_matrix(full_df, label_col="label")
    means_by_state = feature_means_by_state(full_df, FEATURE_LIST, state_col="state", label_col="label")
    profile_by_state = qualitative_feature_profile(means_by_state, FEATURE_LIST)

    # ==== 状态稳定性 / 分离程度 / 转移稳定性 / 可解释性 ====
    persistence = regime_persistence_score(duration_stats)
    separation = state_separation_score(full_df, FEATURE_LIST, state_col="state")
    stability = transition_stability_report(transition_probs)
    interp_score = interpretability_score(profile_by_state, FEATURE_LIST)
    interpretation = regime_interpretation(full_state_stats, duration_stats)

    aic_bic = compute_aic_bic(model, train_scaled)

    # ---- 回测（含 shift(1) 执行延迟与交易成本） ----
    full_bt = run_backtest(full_df)
    test_bt = run_backtest(test_df)
    full_metrics = compute_strategy_and_benchmark_metrics(full_bt)
    test_metrics = compute_strategy_and_benchmark_metrics(test_bt)

    # 用样本外（测试集）Sharpe 评估 Backtest Performance：
    # 样本内更多状态数天然更容易过拟合出更高的 Sharpe，只有样本外表现才能真实
    # 反映该状态数量的泛化能力，这也是后续"状态数量推荐"要用测试集指标的原因。
    backtest_score = backtest_performance_score(test_metrics.loc["sharpe", "Strategy"])

    quality_scorecard = regime_quality_scorecard(
        separation_score=separation["overall"],
        persistence_score=persistence["overall"],
        transition_stability_score=stability["score"],
        interpretability_score=interp_score,
        backtest_score=backtest_score,
    )

    if verbose:
        print("\n状态统计信息（训练集，简版）：")
        print(get_state_statistics(train_df, state_col="state"))
        print_state_statistics(full_state_stats)
        print_duration_statistics(duration_stats)
        print_annual_distribution(annual_dist)
        print_transition_probabilities(transition_probs)
        print_feature_profile(profile_by_state, means_by_state, FEATURE_LIST)
        print_persistence_score(persistence)
        print_separation_score(separation)
        print_transition_stability(stability)
        print_regime_interpretation(interpretation)

        quality_checks = regime_quality_report(full_state_stats, duration_stats, transition_probs)
        print_regime_quality_report(quality_checks)
        print_regime_quality_scorecard(quality_scorecard)

        print(
            f"\nAIC：{aic_bic['aic']:.2f}    BIC：{aic_bic['bic']:.2f}    "
            f"Log Likelihood：{aic_bic['log_likelihood']:.4f}"
        )
        print("\n全样本（训练+测试）回测指标：")
        print(full_metrics)
        print("\n测试集（样本外）回测指标：")
        print(test_metrics)

    summary_row = {
        "n_states": n_states,
        "total_return": test_metrics.loc["cumulative_return", "Strategy"],
        "cagr": test_metrics.loc["cagr", "Strategy"],
        "sharpe": test_metrics.loc["sharpe", "Strategy"],
        "sortino": test_metrics.loc["sortino", "Strategy"],
        "calmar": test_metrics.loc["calmar", "Strategy"],
        "max_drawdown": test_metrics.loc["max_drawdown", "Strategy"],
        "avg_duration": float(np.mean(duration_stats["avg_duration"])) if not duration_stats.empty else 0.0,
        "transition_stability": stability["score"],
        "persistence_score": persistence["overall"],
        "separation_score": separation["overall"],
        "aic": aic_bic["aic"],
        "bic": aic_bic["bic"],
        "quality_overall": quality_scorecard["Overall"],
    }

    return {
        "n_states": n_states,
        "model": model,
        "label_map": label_map,
        "full_df": full_bt,
        "test_df": test_bt,
        "state_stats": full_state_stats,
        "duration_stats": duration_stats,
        "annual_distribution": annual_dist,
        "transition_probs": transition_probs,
        "means_by_state": means_by_state,
        "profile_by_state": profile_by_state,
        "persistence": persistence,
        "separation": separation,
        "transition_stability": stability,
        "interpretation": interpretation,
        "quality_scorecard": quality_scorecard,
        "aic_bic": aic_bic,
        "full_metrics": full_metrics,
        "test_metrics": test_metrics,
        "summary_row": summary_row,
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """把各状态数量的 summary_row 汇总成以 n_states 为索引的比较表（第1节）。"""
    rows = [r["summary_row"] for r in results.values()]
    return pd.DataFrame(rows).set_index("n_states").sort_index()


def _min_max_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    在"本次参与比较的候选状态数量集合内部"做 min-max 归一化打分（0~100）。
    AIC/BIC/Sharpe/MaxDD 的绝对数值量纲各不相同、单独看没有意义，只有放在
    2~6 States 的候选集合里互相比较排名，才能转换成可加权求和的可比分数。
    """
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(100.0, index=series.index)
    norm = (series - lo) / (hi - lo)
    if not higher_is_better:
        norm = 1 - norm
    return norm * 100.0


def compute_overall_scores(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    第5节：自动推荐最佳状态数量。

    综合 AIC / BIC / Sharpe / MaxDD（候选集合内部 min-max 归一化）与
    Persistence / Separation / Transition Stability（本身已是 0~100 绝对分数），
    按 config.OVERALL_SCORE_WEIGHTS 加权求和得到 Overall Score。
    """
    df = comparison_df.copy()
    df["aic_score"] = _min_max_score(df["aic"], higher_is_better=False)
    df["bic_score"] = _min_max_score(df["bic"], higher_is_better=False)
    df["sharpe_score"] = _min_max_score(df["sharpe"], higher_is_better=True)
    # max_drawdown 本身是负数（越接近0越好），数值上"越大越好"，可直接 higher_is_better=True
    df["mdd_score"] = _min_max_score(df["max_drawdown"], higher_is_better=True)

    w = OVERALL_SCORE_WEIGHTS
    df["overall_score"] = (
        df["aic_score"] * w["aic"]
        + df["bic_score"] * w["bic"]
        + df["sharpe_score"] * w["sharpe"]
        + df["mdd_score"] * w["max_drawdown"]
        + df["persistence_score"] * w["persistence"]
        + df["separation_score"] * w["separation"]
        + df["transition_stability"] * w["transition_stability"]
    )
    return df


def print_comparison_table(df: pd.DataFrame):
    """打印第1节要求的完整比较表。"""
    print("\n======== State Number Comparison Table (2~6 States) ========")
    display_df = pd.DataFrame(index=df.index)
    display_df["Total Return"] = df["total_return"].map(lambda v: f"{v:.2%}")
    display_df["CAGR"] = df["cagr"].map(lambda v: f"{v:.2%}")
    display_df["Sharpe"] = df["sharpe"].map(lambda v: f"{v:.2f}")
    display_df["Sortino"] = df["sortino"].map(lambda v: f"{v:.2f}")
    display_df["Calmar"] = df["calmar"].map(lambda v: f"{v:.2f}")
    display_df["MaxDD"] = df["max_drawdown"].map(lambda v: f"{v:.2%}")
    display_df["Avg Duration"] = df["avg_duration"].map(lambda v: f"{v:.1f}d")
    display_df["Transition Stability"] = df["transition_stability"].map(lambda v: f"{v:.0f}%")
    display_df["Persistence"] = df["persistence_score"].map(lambda v: f"{v:.0f}")
    display_df["Separation"] = df["separation_score"].map(lambda v: f"{v:.0f}")
    display_df["AIC"] = df["aic"].map(lambda v: f"{v:.0f}")
    display_df["BIC"] = df["bic"].map(lambda v: f"{v:.0f}")
    display_df["Overall"] = df["overall_score"].map(lambda v: f"{v:.1f}")
    print(display_df)
    print("================================================================")


def print_recommendation(df: pd.DataFrame) -> int:
    """打印第5节要求的推荐报告，返回 Overall Score 最高的状态数量（仅推荐，不修改 config）。"""
    print("\n======== Recommended State Number ========")
    for n in df.index:
        print(f"\n{n} States")
        print(f"Overall：{df.loc[n, 'overall_score']:.1f}")
        print("--------------")
    best_n = int(df["overall_score"].idxmax())
    print(f"\nRecommended")
    print(f"Best State Number：{best_n}")
    print("\n注意：以上仅为自动推荐，不会自动修改 config.py 中的 N_STATES，需人工确认后手动调整。")
    print("=============================================")
    return best_n


def run_state_optimization(
    csv_path: str = None,
    show_plot: bool = True,
    states_range=None,
    verbose_per_n: bool = True,
) -> dict:
    """
    V2.7 状态数量优化主流程（对应任务书第1~7节）：对 states_range（默认2~6）
    中每个候选状态数分别完整执行一次 HMM训练/状态映射/回测/全部统计分析，
    最终输出比较表、自动推荐、以及推荐状态数量的 Regime Quality Report。
    """
    states_range = states_range if states_range is not None else STATES_SEARCH_RANGE
    train_df, test_df, train_scaled, test_scaled, scaler = prepare_train_test(csv_path)

    results = {
        n: run_single_n_states(train_df, test_df, train_scaled, test_scaled, n, verbose=verbose_per_n)
        for n in states_range
    }

    comparison_df = build_comparison_table(results)
    comparison_df = compute_overall_scores(comparison_df)
    print_comparison_table(comparison_df)
    best_n = print_recommendation(comparison_df)

    best_result = results[best_n]
    print(f"\n======== Regime Quality Report（Best：{best_n} States） ========")
    print_regime_quality_scorecard(best_result["quality_scorecard"])

    fig = None
    if show_plot:
        labels_order = get_state_label_order(best_n)
        colors = get_state_colors(labels_order)
        fig = plot_full_report(
            best_result["full_df"],
            best_result["model"].transmat_,
            state_labels=labels_order,
            state_colors=colors,
        )
        plt.show()

    return {
        "results": results,
        "comparison_table": comparison_df,
        "best_n_states": best_n,
        "best_result": best_result,
        "scaler": scaler,
        "fig": fig,
    }


def run_pipeline(csv_path: str = None, show_plot: bool = True, n_states: int = None) -> dict:
    """
    单一固定状态数运行（默认沿用 config.N_STATES=3），保留原有单次运行入口，
    内部复用 run_single_n_states，与 run_state_optimization 共享同一套逻辑。
    """
    n_states = n_states if n_states is not None else N_STATES
    train_df, test_df, train_scaled, test_scaled, scaler = prepare_train_test(csv_path)
    result = run_single_n_states(train_df, test_df, train_scaled, test_scaled, n_states, verbose=True)

    # 诊断性状态数搜索（仅供参考对比，不影响本次实际训练所用的 n_states）
    selection_result, best_n_states, _ = search_best_n_states(train_scaled)
    print_model_selection_report(selection_result, best_n_states)

    labels_order = get_state_label_order(n_states)
    colors = get_state_colors(labels_order)
    fig = plot_full_report(
        result["full_df"], result["model"].transmat_, state_labels=labels_order, state_colors=colors
    )
    if show_plot:
        plt.show()

    quality_checks = regime_quality_report(
        result["state_stats"], result["duration_stats"], result["transition_probs"]
    )

    return {
        "model": result["model"],
        "scaler": scaler,
        "label_map": result["label_map"],
        "full_df": result["full_df"],
        "test_df": result["test_df"],
        "fig": fig,
        "state_stats": result["state_stats"],
        "duration_stats": result["duration_stats"],
        "annual_distribution": result["annual_distribution"],
        "transition_probs": result["transition_probs"],
        "model_selection": selection_result,
        "best_n_states": best_n_states,
        "quality_checks": quality_checks,
        "quality_scorecard": result["quality_scorecard"],
    }


def main():
    parser = argparse.ArgumentParser(description="V2.7 HMM 市场状态识别模块")
    parser.add_argument(
        "--csv", type=str, default=None, help="真实历史K线CSV路径（可选）"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="不弹出可视化窗口（仅打印指标）"
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=None,
        help="仅使用固定的隐藏状态数运行单次流程（不做2~6状态数量比较）。不传则默认执行完整的状态数量优化流程（2~6）。",
    )
    args = parser.parse_args()

    if args.n_states is not None:
        run_pipeline(csv_path=args.csv, show_plot=not args.no_plot, n_states=args.n_states)
    else:
        run_state_optimization(csv_path=args.csv, show_plot=not args.no_plot)


if __name__ == "__main__":
    main()
