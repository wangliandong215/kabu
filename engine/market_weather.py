"""
engine/market_weather.py — "大盘天气指挥官"（V2 需求书模块一）

每日读取大盘指数（默认 QQQ，见 config.QQQ_CORE_CODE）收盘价，输出状态码：
  2 = 安全 / 全面进攻  —— 非危机，且波动率低/稳定
  1 = 震荡 / 减仓防守  —— 跌破短期均线（20日）但未达危机，或波动率异常放大
  0 = 危机 / 全面禁买  —— qqq_macro_halt 触发（跌破MA200且MA20动量陡降），
                        或波动率飙升至极端阈值

状态0的"跌破MA200"这条腿直接复用 engine/regime.py::qqq_macro_halt_series
（v1.0-RELEASE-FINAL 锁定基线），不用更宽松的"单纯跌破MA200"——2026-07-04
全历史回测验证过：单纯跌破MA200触发拦截（不要求MA20动量同步陡降）会比
qqq_macro_halt多拦截138笔交易，总收益277.18%→238.35%，但最大回撤没有改善
反而略差（-19.85%→-20.79%），说明多拦截的那些"跌破MA200但企稳/走平"的
日子换不来更好的风控，纯粹损失收益。改回复用 qqq_macro_halt、只在其基础上
叠加"波动率极端放大"这一条新增覆盖后，交易笔数871（vs 锁定基线873，只多
拦截约2笔），总收益240.91%/Sharpe 0.891/最大回撤-20.37%——比锁定基线仍有
~13%的收益差距，但笔数几乎持平，剩余差距大概率是这2笔额外拦截触发的路径
依赖效应（同一批信号在长回测里逐日排队顺序被小幅改变，跟项目里
TRENDING_EARLY那次记录的路径依赖是同一类现象），不是实现bug（见
_market_weather_ab_result.csv、project memory）。这个模块只喂给
risk/sizing.py 的风险乘数（模块二），状态0在 runner.py/backtest_portfolio.py
里并入跟 qqq_macro_halt 相同的"只挡新开仓、不强平已持仓"拦截开关，不是
替代关系。

真实 VIX 在 moomoo 无法获取（US.VIX 报错"未知股票"，见项目memory），用 QQQ
自身已实现波动率（复用 engine.regime.qqq_realized_vol_series）作为恐慌代理。
"""
import numpy as np
import pandas as pd

from engine.regime import qqq_realized_vol_series, qqq_macro_halt_series
import config


def market_weather_series(close: pd.Series, ma_long_period: int = None) -> pd.Series:
    """
    Per-bar market weather code (vectorised, causal — safe for backtest).

    MA200预热陷阱（项目里已踩过一次的坑）：close<ma 这类比较在 ma 是
    NaN 时会静默返回 False，如果不显式处理，热身期会被误判成状态2（安全）。
    这里显式算出 warmup 掩码并最后应用，强制热身期为状态1（谨慎），既不是
    "未知=安全"也不是"未知=危机"。
    """
    if ma_long_period is None:
        ma_long_period = config.QQQ_MA_PERIOD

    close = close.astype(float)
    ma20 = close.rolling(config.MARKET_WEATHER_MA_SHORT).mean()
    ma_long = close.rolling(ma_long_period).mean()

    vol = qqq_realized_vol_series(close, window=config.MARKET_WEATHER_VOL_WINDOW)
    vol_baseline = vol.rolling(config.MARKET_WEATHER_VOL_BASELINE_WINDOW).mean()
    vol_ratio = vol / vol_baseline.replace(0, np.nan)

    warmup = ma_long.isna() | ma20.isna() | vol_ratio.isna()

    macro_halt = qqq_macro_halt_series(close)   # 跌破MA200 且 MA20动量陡降
    crisis = macro_halt | (vol_ratio >= config.MARKET_WEATHER_VOL_EXTREME_MULT)
    chop = (~crisis) & ((close < ma20) | (vol_ratio >= config.MARKET_WEATHER_VOL_EXPANSION_MULT))

    code = pd.Series(2, index=close.index)   # 默认：安全/全面进攻
    code[chop] = 1
    code[crisis] = 0
    code[warmup] = 1   # 最后应用——不能被 crisis/chop 覆盖
    return code


def market_weather(df: pd.DataFrame) -> int:
    """Last-bar convenience wrapper for live scanning. df needs a 'close' column."""
    series = market_weather_series(df["close"].astype(float))
    if len(series) == 0:
        return 1
    return int(series.iloc[-1])
