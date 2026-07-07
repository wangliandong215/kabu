# -*- coding: utf-8 -*-
"""
V2.7 HMM 市场状态识别模块 —— 数据清洗与特征工程

【防未来函数（Look-ahead Bias）总原则】
    HMM 在时刻 t 做状态判断时，只允许使用「时刻 t 收盘时已经确定」的信息。
    - log_return / hl_range：直接由当日 OHLC 计算，当日收盘后即可获得，不存在未来函数问题。
    - volatility / volume_change：属于"滚动统计量"，一旦不小心把当天自身也纳入统计口径，
      在某些资金曲线回放场景下容易被误用成"提前偷看未来"，因此本模块严格要求这类滚动特征
      使用 rolling(..., closed='left') —— 即窗口不包含当前行，只使用截止到 t-1 的历史数据，
      等价于对 rolling 结果整体 shift(1)，但写法更直接、不容易漏加。
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import LOOKBACK_PERIOD, ATR_PERIOD, FEATURE_LIST


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗原始 K 线数据（真实历史数据与模拟数据共用同一套清洗规则）。

    要求输入 df 至少包含 Open, High, Low, Close, Volume 列，索引为日期。
    依次完成：类型转换 -> 缺失值处理 -> 日期排序去重 -> 异常数据剔除。
    """
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"输入数据缺少必要列: {missing}")

    df = df.copy()

    # ---- 类型转换：真实 CSV 中数值列可能被读成字符串（如带千分位逗号），
    # 统一转成数值类型，无法转换的值变为 NaN，交给下一步缺失值处理 ----
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- 缺失值处理：直接删除任意 OHLCV 列存在缺失的行。金融时间序列前向填充
    # 容易掩盖真实停牌/数据缺口，因此不做 fillna，宁可丢弃也不伪造数据 ----
    df = df.dropna(subset=required_cols)

    # ---- 日期排序 + 去重：按索引（日期）升序排列；同一天出现多条记录时保留最后一条 ----
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # ---- 去除异常数据：价格/成交量必须为正，且 High/Low 必须满足基本 OHLC 约束 ----
    valid = (
        (df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
        & (df["Volume"] >= 0)
        & (df["High"] >= df[["Open", "Close", "Low"]].max(axis=1))
        & (df["Low"] <= df[["Open", "Close", "High"]].min(axis=1))
    )
    n_removed = (~valid).sum()
    if n_removed > 0:
        print(f"数据清洗：剔除 {n_removed} 条异常/不合法 OHLC 记录")
    df = df[valid]

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建 HMM 训练所需的基础特征。

    特征说明：
      log_return    : 对数收益率 ln(Close_t / Close_{t-1})，反映当日涨跌幅度。
      volatility    : 过去 LOOKBACK_PERIOD 日（不含当日）对数收益率的滚动标准差，
                      反映"进入当日之前"市场的历史波动水平。
      hl_range      : 当日振幅 (High - Low) / Close，反映当日日内波动剧烈程度。
      volume_change : 当日成交量相对于过去 LOOKBACK_PERIOD 日（不含当日）平均成交量的变化率，
                      反映当日成交量是否出现异常放量/缩量。
    """
    df = df.copy()

    # ---- 对数收益率：直接使用当日与前一日收盘价，当日收盘后即可计算，无未来函数 ----
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # ---- 高低价振幅：完全基于当日 OHLC，当日收盘后即可获得，无未来函数 ----
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

    # ---- 滚动波动率：【硬性防未来函数】rolling(closed='left') 窗口不含当日 ----
    # 即 t 时刻的 volatility 只使用 [t-LOOKBACK_PERIOD, t-1] 区间的历史收益率
    df["volatility"] = (
        df["log_return"].rolling(window=LOOKBACK_PERIOD, closed="left").std()
    )

    # ---- 成交量变化率：【硬性防未来函数】基准均值同样 closed='left'，不含当日成交量 ----
    volume_baseline = (
        df["Volume"].rolling(window=LOOKBACK_PERIOD, closed="left").mean()
    )
    df["volume_change"] = df["Volume"] / volume_baseline - 1

    return df


def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """
    【可扩展特征】计算 ATR（Average True Range，平均真实波幅）。

    True Range 本身基于当日 High/Low 与前一日 Close，当日收盘后即可获得；
    对 True Range 做滚动平均时同样使用 closed='left'，保证 ATR_t 不掺入当日 True Range，
    避免未来函数。当前默认 FEATURE_LIST 未启用该特征，仅作为后续扩展接口预留。
    """
    df = df.copy()
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = true_range.rolling(window=period, closed="left").mean()
    return df


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_list: list = None,
):
    """
    对特征做标准化（均值0，方差1）。

    【硬性要求：防信息泄露】StandardScaler 必须只在训练集上 fit，
    测试集只能 transform，不能重新 fit，否则测试集的均值/方差信息会
    提前泄露给"未来"的训练过程，导致回测结果虚高。

    返回：
      train_scaled, test_scaled : 标准化后的 ndarray（顺序与 feature_list 一致）
      scaler                    : 已 fit 好的 StandardScaler，供后续 Walk-Forward 复用
    """
    if feature_list is None:
        feature_list = FEATURE_LIST

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_list])
    test_scaled = scaler.transform(test_df[feature_list])
    return train_scaled, test_scaled, scaler


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    一站式数据准备：清洗 -> 构建特征 -> 丢弃因 rolling/shift 产生的 NaN 行。
    """
    df = clean_data(df)
    df = build_features(df)
    df = df.dropna(subset=FEATURE_LIST)
    return df
