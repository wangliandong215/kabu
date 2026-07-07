# -*- coding: utf-8 -*-
"""
V2.7 HMM 市场状态识别模块 —— 全局参数配置
所有可调参数集中在这里管理，其余模块只 import 不写死数字。
"""

import numpy as np

# ======================== HMM 模型参数 ========================
N_STATES = 3            # 隐藏状态数：Bull / Bear / Sideways
RANDOM_STATE = 42        # 固定随机种子，保证 GaussianHMM 训练结果可复现
N_ITER = 1000            # HMM 训练最大迭代次数（EM 算法收敛上限）
COVARIANCE_TYPE = "full"  # 协方差矩阵类型：full 允许每个状态特征间有独立相关性

# ======================== 数据切分参数 ========================
TRAIN_RATIO = 0.8        # 训练集占比，其余为测试集（按时间顺序切分，不做随机打乱）

# ======================== 特征工程参数 ========================
LOOKBACK_PERIOD = 5       # 滚动波动率 / 成交量变化率的回看窗口（交易日）
ATR_PERIOD = 14           # ATR 计算周期（可选扩展特征）

# 参与 HMM 训练的特征列表，顺序即模型输入的列顺序
FEATURE_LIST = [
    "log_return",       # 对数收益率
    "volatility",       # 滚动历史波动率
    "hl_range",         # 高低价振幅
    "volume_change",    # 成交量变化率
]

# ======================== 回测参数 ========================
TRANSACTION_COST = 0.0005  # 单边交易成本（手续费+滑点），双边买卖各扣一次
INITIAL_CAPITAL = 1_000_000.0  # 回测起始资金，仅用于展示财富曲线绝对值

# ======================== 风险指标参数 ========================
ANNUAL_PERIODS = 252      # 年化交易日数，用于 CAGR / Sharpe / Sortino 年化换算
RISK_FREE_RATE = 0.02     # 年化无风险利率，Sharpe/Sortino 计算超额收益时使用

# ======================== 状态标签 ========================
STATE_LABELS = ["Bull", "Bear", "Sideways"]  # 自适应映射后使用的标准状态名

# ======================== 可视化参数 ========================
STATE_COLORS = {
    "Bull": "#2ca02c",       # 绿色
    "Bear": "#d62728",       # 红色
    "Sideways": "#ffbb00",   # 黄色
}

# ======================== 状态数量优化参数（V2.7 State Optimization） ========================
STATES_SEARCH_RANGE = range(2, 7)  # 状态数量搜索范围：2~6

# Persistence Score 的目标持续天数：状态平均持续天数达到/超过该值记满分100，
# 不足则按比例线性打分。20个交易日约等于一个自然月，作为"稳定regime"的基准线。
PERSISTENCE_TARGET_DURATION = 20

# State Chattering（状态抖动）判定阈值：转移矩阵的平均自持概率低于该值即判定为抖动
CHATTERING_THRESHOLD = 0.5

# 综合推荐评分（Overall Score）各维度权重，需和为 1。
# BIC 对参数数量的惩罚更重、更适合金融时间序列，因此权重略高于 AIC；
# Persistence / Separation / Transition Stability 是本阶段最关心的"状态质量"，
# 三者权重合计 0.5，与传统回测指标（Sharpe/MaxDD）及信息准则（AIC/BIC）平分秋色。
OVERALL_SCORE_WEIGHTS = {
    "aic": 0.10,
    "bic": 0.15,
    "sharpe": 0.15,
    "max_drawdown": 0.10,
    "persistence": 0.20,
    "separation": 0.15,
    "transition_stability": 0.15,
}


def get_state_label_order(n_states: int) -> list:
    """
    根据隐藏状态数量生成"按平均收益率从低到高"排列的标准标签顺序，
    用于替代写死的 3 状态 STATE_LABELS，支持 2~6 甚至任意状态数：

      2 States: [Bear, Bull]
      3 States: [Bear, Sideways, Bull]
      5 States: [Bear, Sideways_1, Sideways_2, Sideways_3, Bull]

    必须与 hmm_model.map_states_to_labels() 的命名规则保持一致。
    """
    if n_states <= 1:
        return ["Sideways"]
    if n_states == 2:
        return ["Bear", "Bull"]
    middle_count = n_states - 2
    if middle_count == 1:
        middles = ["Sideways"]
    else:
        middles = [f"Sideways_{i}" for i in range(1, middle_count + 1)]
    return ["Bear"] + middles + ["Bull"]


def get_state_colors(labels: list) -> dict:
    """
    按"Bear -> ... -> Bull"的标签顺序生成 红->黄->绿 三段线性插值渐变色，
    用于任意状态数量（2~6）的可视化着色，替代写死 3 色的 STATE_COLORS。
    """
    n = len(labels)
    if n == 1:
        return {labels[0]: "#ffbb00"}
    red = np.array([214, 39, 40])
    yellow = np.array([255, 187, 0])
    green = np.array([44, 160, 44])
    colors = {}
    for i, label in enumerate(labels):
        t = i / (n - 1)
        if t <= 0.5:
            rgb = red + (yellow - red) * (t / 0.5)
        else:
            rgb = yellow + (green - yellow) * ((t - 0.5) / 0.5)
        colors[label] = "#{:02x}{:02x}{:02x}".format(*rgb.astype(int))
    return colors
