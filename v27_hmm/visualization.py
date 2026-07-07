# -*- coding: utf-8 -*-
"""
V2.7 HMM 市场状态识别模块 —— 四合一量化评估可视化

四个子图：
    1. 收盘价走势图，按 Bull/Bear/Sideways 状态动态背景染色
    2. 策略净值曲线 vs Buy & Hold 基准净值曲线
    3. HMM 状态随时间切换的时间序列图
    4. 状态转移矩阵热力图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import STATE_COLORS, STATE_LABELS


def _shade_regime_background(ax, df: pd.DataFrame, label_col: str = "label", state_colors: dict = None):
    """
    按状态标签把背景切成一段段色块。
    做法：找出 label 发生变化的分段区间，对每一段用对应状态颜色 axvspan 着色。
    """
    if state_colors is None:
        state_colors = STATE_COLORS
    labels = df[label_col].values
    index = df.index

    seg_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[seg_start]:
            color = state_colors.get(labels[seg_start], "#cccccc")
            ax.axvspan(index[seg_start], index[i - 1], color=color, alpha=0.15)
            seg_start = i


def plot_full_report(
    df: pd.DataFrame,
    model_transmat: np.ndarray,
    price_col: str = "Close",
    label_col: str = "label",
    figsize: tuple = (16, 14),
    state_labels: list = None,
    state_colors: dict = None,
):
    """
    绘制四合一评估大图并返回 fig 对象（由调用方决定 show() 或 savefig()）。

    state_labels / state_colors 支持传入任意状态数量（2~6）下的标签顺序与配色
    （见 config.get_state_label_order / get_state_colors），不传则退回默认的
    3 状态 STATE_LABELS / STATE_COLORS，保持向后兼容。
    """
    if state_labels is None:
        state_labels = STATE_LABELS
    if state_colors is None:
        state_colors = STATE_COLORS

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # ---------------- 子图1：收盘价 + 状态背景染色 ----------------
    ax1 = axes[0, 0]
    ax1.plot(df.index, df[price_col], color="black", linewidth=1.0, label="Close")
    _shade_regime_background(ax1, df, label_col, state_colors=state_colors)
    ax1.set_title("Close Price with HMM Regime Shading")
    ax1.set_ylabel("Price")
    legend_patches = [
        mpatches.Patch(color=state_colors.get(s, "#cccccc"), alpha=0.4, label=s) for s in state_labels
    ]
    ax1.legend(handles=legend_patches, loc="upper left")

    # ---------------- 子图2：策略净值 vs Buy & Hold ----------------
    ax2 = axes[0, 1]
    ax2.plot(df.index, df["strategy_equity"], label="Strategy Equity", color="#1f77b4")
    ax2.plot(
        df.index,
        df["benchmark_equity"],
        label="Buy & Hold Benchmark",
        color="#7f7f7f",
        linestyle="--",
    )
    ax2.set_title("Equity Curve: Strategy vs Buy & Hold")
    ax2.set_ylabel("Equity")
    ax2.legend(loc="upper left")

    # ---------------- 子图3：状态时间序列 ----------------
    ax3 = axes[1, 0]
    label_to_code = {label: i for i, label in enumerate(state_labels)}
    state_codes = df[label_col].map(label_to_code)
    colors = df[label_col].map(state_colors).fillna("#cccccc")
    ax3.scatter(df.index, state_codes, c=colors, s=8)
    ax3.set_yticks(list(label_to_code.values()))
    ax3.set_yticklabels(list(label_to_code.keys()))
    ax3.set_title("HMM Regime Over Time")

    # ---------------- 子图4：状态转移矩阵热力图 ----------------
    ax4 = axes[1, 1]
    # 注意：model_transmat 的行列顺序是训练时的原始状态编号顺序，
    # 这里仅做数值展示，不代表原始编号已按 Bull/Bear/Sideways 排序。
    sns.heatmap(
        model_transmat,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        ax=ax4,
        cbar=True,
    )
    ax4.set_title("HMM State Transition Matrix")
    ax4.set_xlabel("To State")
    ax4.set_ylabel("From State")

    fig.tight_layout()
    return fig
