# -*- coding: utf-8 -*-
"""
V2.7 HMM 市场状态识别模块 —— 专业量化风控指标计算

统一约定：所有输入的 returns 均为「对数收益率」序列，equity 为对应的资产净值曲线。
"""

import numpy as np
import pandas as pd

from config import ANNUAL_PERIODS, RISK_FREE_RATE


def cumulative_return(equity: pd.Series) -> float:
    """累计收益率 = 期末净值 / 期初净值 - 1"""
    return equity.iloc[-1] / equity.iloc[0] - 1.0


def cagr(equity: pd.Series, periods_per_year: int = ANNUAL_PERIODS) -> float:
    """
    年化收益率（Compound Annual Growth Rate）。
    n_periods 为实际交易日数，按 ANNUAL_PERIODS 折算成"年"的分数单位再开方年化。
    """
    n_periods = len(equity)
    if n_periods == 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = n_periods / periods_per_year
    if years <= 0 or total_return <= 0:
        return 0.0
    return total_return ** (1.0 / years) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """
    最大回撤：净值曲线相对历史最高点的最大跌幅（负数，越接近0越好）。
    running_max 是"截止当前时刻"的历史最高净值，drawdown 是当前净值相对它的回撤幅度。
    """
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = ANNUAL_PERIODS,
) -> float:
    """
    夏普比率 = 年化超额收益 / 年化波动率。
    先把年化无风险利率折算成单期无风险收益，再算超额收益的均值/标准差，最后年化。
    """
    if returns.std() == 0 or len(returns) == 0:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = ANNUAL_PERIODS,
) -> float:
    """
    索提诺比率：与夏普比率类似，但分母只统计"下行波动率"（只考虑亏损日的离散程度），
    不惩罚正向的高波动，更贴近投资者真正在意的"下跌风险"。
    """
    if len(returns) == 0:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)


def calmar_ratio(cagr_value: float, mdd_value: float) -> float:
    """
    卡玛比率 = 年化收益率 / |最大回撤|，衡量"每承受一单位最大回撤能换回多少年化收益"。
    """
    if mdd_value == 0:
        return 0.0
    return cagr_value / abs(mdd_value)


def compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    periods_per_year: int = ANNUAL_PERIODS,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """汇总计算单条净值曲线的全部核心指标。"""
    cagr_value = cagr(equity, periods_per_year)
    mdd_value = max_drawdown(equity)
    return {
        "cumulative_return": cumulative_return(equity),
        "cagr": cagr_value,
        "max_drawdown": mdd_value,
        "sharpe": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "calmar": calmar_ratio(cagr_value, mdd_value),
    }


def compute_strategy_and_benchmark_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    同时计算策略与 Buy & Hold 基准的全部指标，便于对比展示。
    """
    strategy_metrics = compute_metrics(df["strategy_equity"], df["strategy_return"])
    benchmark_metrics = compute_metrics(df["benchmark_equity"], df["log_return"])
    result = pd.DataFrame(
        {"Strategy": strategy_metrics, "Buy & Hold": benchmark_metrics}
    )
    return result
