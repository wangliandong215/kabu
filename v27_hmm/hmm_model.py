# -*- coding: utf-8 -*-
"""
V2.7 HMM 市场状态识别模块 —— 模型训练与自适应状态标签映射
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from config import N_STATES, RANDOM_STATE, N_ITER, COVARIANCE_TYPE


def train_hmm(
    train_scaled: np.ndarray,
    n_states: int = N_STATES,
    random_state: int = RANDOM_STATE,
    n_iter: int = N_ITER,
    covariance_type: str = COVARIANCE_TYPE,
) -> GaussianHMM:
    """
    训练 GaussianHMM。

    n_states 是隐藏状态数（对应 Bull/Bear/Sideways），covariance_type='full'
    允许每个隐藏状态下各特征之间存在独立的协方差结构（而不是假设特征互相独立），
    更贴近真实市场中"高波动率常伴随大振幅"这类特征相关性。
    random_state 固定后，EM 算法的初始化和迭代路径可复现，保证同一份数据每次
    训练出的模型（在参数层面）完全一致。
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(train_scaled)
    return model


def decode_states(model: GaussianHMM, features_scaled: np.ndarray) -> np.ndarray:
    """
    使用 Viterbi 算法解码出最可能的隐藏状态序列（每个时间点一个状态编号 0..n_states-1）。
    注意：此时的状态编号是 HMM 内部随机分配的，不代表任何固定含义，
    必须配合 map_states_to_labels() 做自适应映射后才能解读为 Bull/Bear/Sideways。
    """
    return model.predict(features_scaled)


def print_model_diagnostics(
    model: GaussianHMM,
    features_scaled: np.ndarray,
    feature_list: list,
) -> float:
    """
    打印 HMM 原厂指标，并加入量化含义注释：

    - 初始状态概率 (startprob_)  : 序列第一天落在每个隐藏状态的先验概率。
    - 状态转移矩阵 (transmat_)   : transmat_[i, j] 表示"今天处于状态 i，明天转移到
                                    状态 j"的概率，对角线元素越大说明该状态越"粘滞"
                                    （趋势/状态一旦形成越不容易切换，即状态的平均
                                    持续期 ≈ 1 / (1 - 对角线概率)）。
    - 各状态均值 (means_)        : 每个隐藏状态下，各标准化特征的均值向量，
                                    用于判断该状态"长什么样"（例如均值收益率高/波动率低）。
    - 各状态协方差 (covars_)     : 每个隐藏状态下特征的协方差矩阵，反映该状态内部
                                    特征的离散程度及特征间的相关性。
    - Log Likelihood (score)     : 给定当前模型参数，观测到这批训练数据的对数似然，
                                    数值越大（越接近0）说明模型对训练数据的拟合度越高，
                                    可用于同一份数据下比较不同 n_states/协方差类型的优劣。
    """
    log_likelihood = model.score(features_scaled)

    print("原始状态编号（训练随机分配，不代表固定含义）：", list(range(model.n_components)))
    print("初始状态概率 startprob_:")
    print(np.round(model.startprob_, 4))
    print("状态转移矩阵 transmat_:")
    print(np.round(model.transmat_, 4))
    for i in range(model.n_components):
        avg_duration = 1.0 / (1.0 - model.transmat_[i, i] + 1e-12)
        print(f"状态 {i} 平均持续期（交易日）≈ {avg_duration:.2f}")
    print("各状态均值 means_ (标准化特征空间, 列顺序=%s):" % feature_list)
    print(np.round(model.means_, 4))
    print("各状态协方差 covars_:")
    print(np.round(model.covars_, 4))
    print(f"Log Likelihood: {log_likelihood:.4f}")

    return log_likelihood


def map_states_to_labels(df: pd.DataFrame, state_col: str = "state") -> dict:
    """
    【自适应状态标签映射，禁止使用固定编号映射】

    HMM 每次训练分配给隐藏状态的编号 (0, 1, 2...) 是随机的，同一份数据换个
    random_state 或换一批数据重新训练，"状态1"完全可能从上次的 Bull 变成这次的
    Bear。因此不能写死 {0: 'Bull', 1: 'Bear', 2: 'Sideways'} 这种映射。

    正确做法：按各状态在训练期内的统计特征（平均对数收益率）排序——
    平均收益率最高 -> Bull；平均收益率最低 -> Bear；其余（介于两者之间）-> Sideways。

    支持任意状态数（V2.7 状态数量优化需要在 2~6 之间切换）：
      1 State  -> 全部归为 Sideways（无法区分牛熊的退化情形）
      2 States -> Bear / Bull
      3 States -> Bear / Sideways / Bull
      N States (N>3) -> Bear / Sideways_1 ... Sideways_{N-2}（按平均收益率从低到高
                        依次编号，Sideways_1 最接近 Bear，Sideways_{N-2} 最接近 Bull）
                        / Bull

    命名规则必须与 config.get_state_label_order() 保持一致。
    """
    stats = df.groupby(state_col)["log_return"].mean().sort_values()
    state_ids = stats.index.tolist()
    n = len(state_ids)

    label_map = {}
    if n == 1:
        # 边界情况：只有一个状态，无法区分牛熊，统一归为 Sideways
        label_map[state_ids[0]] = "Sideways"
    elif n == 2:
        label_map[state_ids[0]] = "Bear"   # 平均收益率最低
        label_map[state_ids[1]] = "Bull"   # 平均收益率最高
    else:
        bear_state = state_ids[0]          # 平均收益率最低
        bull_state = state_ids[-1]         # 平均收益率最高
        middle_states = state_ids[1:-1]    # 按平均收益率升序排列的中间状态
        label_map[bear_state] = "Bear"
        label_map[bull_state] = "Bull"
        if len(middle_states) == 1:
            label_map[middle_states[0]] = "Sideways"
        else:
            for i, s in enumerate(middle_states, start=1):
                label_map[s] = f"Sideways_{i}"
    return label_map


def get_state_statistics(df: pd.DataFrame, state_col: str = "state") -> pd.DataFrame:
    """
    输出每个隐藏状态的统计信息：样本数、平均对数收益率、平均波动率，
    供分析师核对自适应映射是否合理（例如 Bull 状态的平均收益率应明显为正）。
    """
    stats = df.groupby(state_col).agg(
        sample_count=("log_return", "count"),
        avg_log_return=("log_return", "mean"),
        avg_volatility=("volatility", "mean"),
    )
    stats["label"] = stats.index.map(map_states_to_labels(df, state_col))
    return stats.sort_values("avg_log_return", ascending=False)


def compute_aic_bic(model: GaussianHMM, features_scaled: np.ndarray) -> dict:
    """
    计算 AIC / BIC，用于在不同隐藏状态数之间做模型选择。

    自由参数数量估算（covariance_type='full'）：
      - 初始状态概率：n_states - 1（各分量之和为1）
      - 转移矩阵：n_states * (n_states - 1)（每行之和为1）
      - 各状态均值：n_states * n_features
      - 各状态协方差（对称矩阵）：n_states * n_features * (n_features + 1) / 2

    AIC = -2*logL + 2*k， BIC = -2*logL + k*ln(n_samples)。
    BIC 对参数数量的惩罚比 AIC 更重，更适合金融时间序列这种噪声大、
    容易过拟合的场景，因此后续状态数选择以 BIC 最小者为准。
    """
    log_likelihood = model.score(features_scaled)
    n_states = model.n_components
    n_features = model.n_features
    n_samples = features_scaled.shape[0]

    n_params = (
        (n_states - 1)
        + n_states * (n_states - 1)
        + n_states * n_features
        + n_states * n_features * (n_features + 1) // 2
    )

    return {
        "n_states": n_states,
        "log_likelihood": log_likelihood,
        "n_params": n_params,
        "aic": -2 * log_likelihood + 2 * n_params,
        "bic": -2 * log_likelihood + n_params * np.log(n_samples),
    }


def search_best_n_states(
    train_scaled: np.ndarray,
    states_range=range(2, 7),
    random_state: int = RANDOM_STATE,
    n_iter: int = N_ITER,
    covariance_type: str = COVARIANCE_TYPE,
):
    """
    对 states_range（默认 2~6）中每个候选隐藏状态数分别训练 GaussianHMM，
    汇总 Log Likelihood / AIC / BIC，用于决定"最终采用几个状态"。

    这是一次独立的诊断性搜索，不影响本次流程实际使用的 config.N_STATES —
    最终是否切换状态数由分析师依据本报告人工决定（V2.7 阶段暂不自动切换）。

    返回：(result_df, best_n_states, best_model)
      result_df     : 以 n_states 为索引的诊断表
      best_n_states : BIC 最小对应的状态数
      best_model    : best_n_states 对应的已训练模型
    """
    records = []
    models = {}
    for k in states_range:
        model = train_hmm(
            train_scaled,
            n_states=k,
            random_state=random_state,
            n_iter=n_iter,
            covariance_type=covariance_type,
        )
        records.append(compute_aic_bic(model, train_scaled))
        models[k] = model

    result_df = pd.DataFrame(records).set_index("n_states")
    best_n_states = int(result_df["bic"].idxmin())
    best_model = models[best_n_states]

    return result_df, best_n_states, best_model


def print_model_selection_report(result_df: pd.DataFrame, best_n_states: int):
    """打印状态数自动搜索报告（对应任务书第7节）。"""
    print("\n======== Model Selection: Hidden States Search ========")
    print(result_df.round(4))
    print(f"\nBest Model：{best_n_states} States")
    print(f"AIC：{result_df.loc[best_n_states, 'aic']:.2f}")
    print(f"BIC：{result_df.loc[best_n_states, 'bic']:.2f}")
    print("=========================================================")


def walk_forward_train(
    df: pd.DataFrame,
    train_window: int,
    test_window: int,
    step: int,
):
    """
    【预留接口】Walk-Forward（滚动训练/滚动窗口重训）。

    V2.7 当前版本仅实现"一次性 train/test 切分"训练，尚未实现滚动重训练。
    未来 V2.8 风险引擎如需应对市场状态的结构性漂移（regime 本身的统计特征
    随时间变化），可在此函数中实现：按 train_window 训练 -> 在紧随其后的
    test_window 上解码 -> 窗口整体前移 step -> 重复，从而得到"随时间滚动更新"
    的状态序列，而不是用一份固定不变的模型贯穿全部历史。

    当前保持未实现，避免在没有真实滚动验证需求前过度设计。
    """
    raise NotImplementedError(
        "walk_forward_train 是 V2.8 预留接口，V2.7 暂不实现滚动训练。"
    )
