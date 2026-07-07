# -*- coding: utf-8 -*-
"""
V2.7 HMM 市场状态识别模块 —— 信号生成与回测引擎

策略逻辑：
    Bull 状态 -> 满仓做多（信号 = 1）
    Bear / Sideways 状态 -> 空仓（信号 = 0）
"""

import numpy as np
import pandas as pd

from config import TRANSACTION_COST, INITIAL_CAPITAL


def generate_signal(df: pd.DataFrame, label_col: str = "label") -> pd.Series:
    """
    根据状态标签生成目标仓位信号：Bull=1，Bear/Sideways=0。
    这里生成的是"当日识别出的目标仓位"，尚未做执行延迟处理。
    """
    return (df[label_col] == "Bull").astype(int)


def run_backtest(
    df: pd.DataFrame,
    label_col: str = "label",
    transaction_cost: float = TRANSACTION_COST,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """
    执行回测：

    【硬性要求：执行延迟】今天收盘才能确认 HMM 识别出的状态，因此今天识别出的
    目标仓位信号必须 shift(1) 后才作为"明天实际持仓"，不能用当天状态直接算
    当天收益，否则等于用收盘价的信息去交易当天的开盘/收盘，构成未来函数。

    交易成本：每当持仓状态发生变化（开仓或平仓）时，扣除一次 transaction_cost。
    """
    df = df.copy()

    df["target_position"] = generate_signal(df, label_col)

    # 【硬性要求：执行延迟】今天的目标仓位，明天才真正持有
    df["position"] = df["target_position"].shift(1).fillna(0)

    # 仓位变化产生交易成本：|今日仓位 - 昨日仓位| != 0 即视为发生了一次调仓
    position_change = df["position"].diff().abs().fillna(df["position"].abs())
    df["transaction_cost"] = position_change * transaction_cost

    # 策略当日收益 = 持仓 * 当日对数收益 - 当日交易成本
    df["strategy_return"] = df["position"] * df["log_return"] - df["transaction_cost"]

    # 资产曲线：对数收益连乘等价于 exp(累计对数收益)，从 initial_capital 起步
    df["strategy_equity"] = initial_capital * np.exp(df["strategy_return"].cumsum())

    # Buy & Hold 基准：全程满仓、不扣成本（仅期初建仓一次的成本量级很小，此处忽略）
    df["benchmark_equity"] = initial_capital * np.exp(df["log_return"].cumsum())

    return df
