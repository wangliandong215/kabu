"""
backtest_portfolio.py — 多仓位投资组合回测（实盘级模拟）

模拟真实交易系统的完整逻辑：
  ✓ 最多5只同时持仓
  ✓ v2.1 横向多因子总分（趋势40%+基本面20%+新闻20%+天气20%，见
    engine/scoring.py）决定放行/折算仓位/放弃，取代旧的链式分档仓位
  ✓ 单笔风险预算 = 总资金 × 2%（反算最大持仓）
  ✓ 硬止损 -5% / 硬止盈 +15%（使用当天 Low/High 检查）
  ✓ 市场形态识别 → 策略路由（atr_breakout / boll_mr / ma_rsi）
  ✓ TRENDING_DOWN 跳过，不做空
  ✓ 总敞口上限 65%，板块上限 25%
  ✓ Half-Kelly：回撤 ≥ 5% 时仓位系数 × 0.5
  ✓ ATR 跟踪止损（针对 atr_breakout 持仓）

不模拟（需实时数据）：
  - 财报熔断
  - 盘中5分钟扫描（日线策略，以日收盘信号为准）

止损/止盈使用当天最低/最高价，比收盘价更真实。
信号以当天收盘时生成，次日（当天收盘）成交——保守起见用同日收盘价成交。

基本面评分（总分模型20%分量）：moomoo 的基本面/评级接口跟新闻一样不支持
按历史日期查询（见 engine/fundamental.py 模块docstring），回测里该分量
恒为 None——不编造历史数据，也不塞固定占位分，直接从加权总分里剔除、
剩余权重重新归一化（engine.scoring.compute_total_score()）。

新闻评分（总分模型20%分量，可选真实数据源）：moomoo 的 get_search_news()
不支持按历史日期查询，只能拿到"当前"新闻，因此无法从 moomoo 直接获得逐日
历史新闻。若提供 --news-file（CSV: date,code,title,source），则用跟实盘
完全相同的三级分类器（engine.news_filter.classify_headlines，Tier1黑天鹅
一票否决/Tier2预期差恶化打50分/Tier3正常100分）逐日按发布日期过滤，不看
未来数据，真实得到的分数参与加权。不提供该参数时，新闻分量同样是 None
（剔除，不参与加权，不是编造历史新闻）。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python backtest_portfolio.py
  python backtest_portfolio.py --start 2022-01-01 --cash 50000
  python backtest_portfolio.py --start 2022-01-01 --news-file news_history.csv
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import engine.news_filter as news_filter
import engine.scoring as scoring
from backtest import fetch_kline as _moomoo_fetch
from engine.indicators import rsi_series
from engine.news_backtest import HistoricalNewsFeed
from engine.regime import detect_series as _regime_series
from engine.regime import qqq_macro_halt_series as _qqq_macro_halt_series
from engine.regime import qqq_realized_vol_series as _qqq_realized_vol_series
from engine.regime import market_volatility_percentile as _market_volatility_percentile
from engine.market_weather import market_weather_series as _market_weather_series
from portfolio import capacity_manager
from portfolio import replacement_stabilizer as rsl
from risk.sizing import rsi_multiplier
from strategies.boll import BollStrategy

# v2.1 横向多因子总分模型：回测环境没有真实历史基本面数据源（跟新闻同一个
# 坑，见 engine/fundamental.py 模块docstring）。固定分（不管是70还是100）
# 参与加权平均都是编造不存在的信息去影响仓位——已用全历史回测实测验证过
# 两版固定分都有副作用（70分：240.91%->207.09%明显拖累；100分：->219.75%
# 更接近但仍是"缺失当满分"的错误概念）。改为 None，交给
# engine.scoring.compute_total_score() 剔除该维度、重新归一化剩余权重
# （2026-07-04，用户指出并采纳 Method A）。
_FUNDAMENTAL_SCORE_BACKTEST = None

# ── 回测参数 ──────────────────────────────────────────────────────────────────

# 2026-07-03: 用户决定不分地区结算，合并回测——欧美+亚太(含台湾/韩国ADR)
# 全部132+10只放进同一个资金池、同一批仓位名额里抢。账户本位币是日元，
# 所以默认用 --usd-jpy 把 US.* 代码换算成日元（见 run_portfolio_backtest
# 的 usd_to_jpy 参数），固定汇率 1 USD = 140 JPY（用户指定，静态汇率，
# 不反映历史汇率波动，见函数 docstring）。
# 若只想跑纯美股，用 config.WATCHLIST_EUROPE_US；纯日股用
# [s for s in config.WATCHLIST_ASIA_PACIFIC if s.startswith("JP.")]。

BACKTEST_STOCKS = list(config.WATCHLIST)

SECTOR_MAP = {k: v for k, v in config.SECTOR_MAP.items() if k in BACKTEST_STOCKS}

MAX_POSITIONS        = 10   # 最多同时持有的活跃信号仓位（不含 QQQ 底仓）
STOP_LOSS_PCT        = 0.05
TAKE_PROFIT_PCT      = 0.15  # 均值回归策略止盈；趋势策略用 ATR 跟踪止损
MAX_TOTAL_EXPOSURE   = 0.95  # 扩大上限：QQQ 底仓 + 10只活跃仓位，需要更高敞口
MAX_SECTOR_EXPOSURE  = 0.50  # 半导体17只，50%允许同时持2-3只龙头
RISK_PER_TRADE_PCT   = 0.02
HALFKELLY_THRESHOLD  = 0.05
HALFKELLY_RECOVERY   = 0.02

MIN_ENTRY_STRENGTH  = 0.0   # 不过滤，所有信号均可进场；动态仓位区分强弱

# 动态仓位：强信号 + 活跃仓位少时，单笔加到 30%
SIZING_PCT = {0.7: 0.20, 0.4: 0.10, 0.0: 0.03}
SIZING_PCT_STRONG_DYNAMIC = 0.30   # 活跃仓位 ≤ DYNAMIC_MAX_OPEN 时强信号用 30%
DYNAMIC_MAX_OPEN     = 3           # ≤3 只活跃仓位时触发动态扩仓
STOP_MAX   = {0.7: 0.10, 0.4: 0.12}

COMMISSION  = 0.001
ATR_TRAIL_MULT  = 5.5   # 趋势策略跟踪止损距离 — 2026-07-04 从 4.0 上调锁定为
                        # 5.5，全历史{4.0,5.0,5.5,6.0,6.5}梯度对比中的单点
                        # 最优（见 config.py ATR_MULT_BASE 同一条记录、
                        # _atr_fixed_comparison.py）

# ── QQQ 大盘 Beta 动态垫底参数（与 config.QQQ_CORE_TARGET_PCT/QQQ_MA_PERIOD 同步）─
QQQ_CODE = "US.QQQ"

BOLL_PERIOD = 20
MA_FAST     = 5
MA_SLOW     = 20
ADX_PERIOD  = 14
ATR_PERIOD  = 14
CHAN_PERIOD  = 20


# ── 向量化技术指标 ────────────────────────────────────────────────────────────

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (EWM with alpha=1/period)."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _atr_series(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return _wilder_smooth(tr, period)


def _atr_breakout_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised ATR Donchian breakout entry signals."""
    c    = df["close"].astype(float)
    h    = df["high"].astype(float)
    atr  = _atr_series(df)
    chan = h.rolling(CHAN_PERIOD).max().shift(1)   # previous period high

    # Strength = (close - channel_high) / ATR — breakout magnitude relative to vol.
    # Typical range 0.1-0.5; values above 1.0 are capped later.
    # All breakouts pass MIN_ENTRY_STRENGTH=0.0 (any positive value triggers).
    norm_strength = ((c - chan) / atr.replace(0, np.nan)).clip(lower=0.0, upper=1.0).fillna(0.0)

    signal   = pd.Series("HOLD", index=df.index)
    strength = pd.Series(0.0, index=df.index)

    buy = c > chan
    signal[buy]   = "BUY"
    strength[buy] = norm_strength[buy]

    # Stop pct: 2×ATR from close
    stop = (2 * atr / c.replace(0, np.nan)).clip(lower=0.01, upper=0.30)

    return pd.DataFrame({"signal": signal, "strength": strength,
                         "atr": atr, "stop_pct": stop})


def _ma_rsi_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised EMA golden cross + RSI filter."""
    c    = df["close"].astype(float)
    emaf = c.ewm(span=MA_FAST,  adjust=False).mean()
    emas = c.ewm(span=MA_SLOW, adjust=False).mean()
    atr  = _atr_series(df)

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    golden = (emaf > emas) & (emaf.shift(1) <= emas.shift(1))  # crossover
    above  = emaf > emas

    signal   = pd.Series("HOLD", index=df.index)
    strength = pd.Series(0.0, index=df.index)

    buy = golden & (rsi > 50)
    signal[buy] = "BUY"
    sell = (emaf < emas) | (rsi < 40)
    signal[sell & ~buy] = "SELL"

    strength[buy] = (rsi[buy] / 100.0).clip(0, 1)   # RSI 50 → 0.50, RSI 70 → 0.70

    stop = (2 * atr / c.replace(0, np.nan)).clip(lower=0.02, upper=0.25)

    return pd.DataFrame({"signal": signal, "strength": strength,
                         "atr": atr, "stop_pct": stop})


_BOLL_STRATEGY = BollStrategy(period=BOLL_PERIOD, stop_loss_pct=STOP_LOSS_PCT)

STRATEGY_MAP = {
    "TRENDING_UP":    _atr_breakout_signals,
    "TRENDING_EARLY": _atr_breakout_signals,  # same signal calc, sized down below
    "TRENDING_DOWN":  None,
    "RANGING":        _BOLL_STRATEGY.compute_series,
    "VOLATILE":       None,
}

STRATEGY_NAME_MAP = {
    "TRENDING_UP":    "atr_breakout",
    "TRENDING_EARLY": "atr_breakout_early",
    "TRENDING_DOWN":  None,
    "RANGING":        "boll_mr",
    "VOLATILE":       None,
}

STRATEGY_MAX_PCT = {
    "atr_breakout":       0.20,   # original 20% cap
    "atr_breakout_early": 0.08,   # TRENDING_EARLY light-trial cap — mirrors
                                   # config.STRATEGY_MAX_SIZE_PCT so live and
                                   # backtest never drift apart
    "boll_mr":            0.10,
    "ma_rsi":             0.15,
}

_TREND_STRATS = {"atr_breakout", "atr_breakout_early", "ema_rsi", "ma", "macd"}


def _compute_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each bar, pick the right strategy based on regime,
    return merged signal/strength/atr/stop_pct/strategy_name.
    """
    regime = _regime_series(df)
    rsi14  = rsi_series(df["close"].astype(float))

    # Pre-compute all strategy signals (atr_breakout_early reuses the same
    # breakout calc as atr_breakout — only the position cap differs, applied
    # later via STRATEGY_MAX_PCT)
    _atr_sig = _atr_breakout_signals(df)
    sigs = {
        "atr_breakout":       _atr_sig,
        "atr_breakout_early": _atr_sig,
        "boll_mr":            _BOLL_STRATEGY.compute_series(df),
        "ma_rsi":             _ma_rsi_signals(df),
    }

    out_signal   = pd.Series("HOLD", index=df.index)
    out_strength = pd.Series(0.0, index=df.index)
    out_atr      = pd.Series(np.nan, index=df.index)
    out_stop     = pd.Series(STOP_LOSS_PCT, index=df.index)
    out_strat    = pd.Series("", index=df.index)

    for reg, strat_name in STRATEGY_NAME_MAP.items():
        mask = regime == reg
        if strat_name is None or not mask.any():
            continue
        s = sigs[strat_name]
        out_signal[mask]   = s["signal"][mask]
        out_strength[mask] = s["strength"][mask]
        out_atr[mask]      = s["atr"][mask]
        out_stop[mask]     = s["stop_pct"][mask]
        out_strat[mask]    = strat_name

    return pd.DataFrame({
        "signal":   out_signal,
        "strength": out_strength,
        "atr":      out_atr,
        "stop_pct": out_stop,
        "strategy": out_strat,
        "regime":   regime,
        "rsi14":    rsi14,
    })


# ── 仓位计算 ──────────────────────────────────────────────────────────────────

def _size(available_cash, price, strength, stop_pct, total_capital,
          strategy, kelly=1.0, open_positions=0, rsi_val=None,
          position_scale=1.0, market_weather_code=2, score_label=None) -> int:
    if price <= 0:
        return 0

    # 动态大盘风险系数（模块二，2026-07-04 重构）—— 直接缩放 RISK_PER_TRADE_PCT
    # 本身，不再另开一条平行的 ATR 风险预算（第一版曾对 atr_breakout/ma_rsi
    # 这类本来就用 2×ATR 做止损的策略重复叠加同一个止损距离算出来的风险上限，
    # 全历史回测验证是纯粹的负面回归，见 config.py 里 MARKET_WEATHER_RISK_MULTIPLIER
    # 的注释和 _market_weather_ab_result.csv）。stop_pct 已经是策略类型感知的
    # （ATR跟踪止损策略传入2×ATR/price，均值回归策略传入固定5%），所以下面
    # q_risk 公式对两类策略天然都是对的，不需要额外的 atr 参数。
    weather_mult = config.MARKET_WEATHER_RISK_MULTIPLIER.get(market_weather_code, 1.0)

    if score_label is not None:
        # v2.1/v2.2：engine/scoring.py 的总分模型已经把趋势(40%)+基本面/新闻/
        # 天气揉进一个 FULL/OBSERVATION/PARTIAL 档位（SKIP 已在调用方过滤掉，
        # 不会传进来），这里直接按 label 选仓位比例，而不是在 fund/news 真实参与加权、
        # 把某个中等趋势信号拉到FULL（或把强趋势信号拉到PARTIAL）时，仍然
        # 用脱节的原始 strength 阈值(0.7/0.4)重新分档一次。注意：fund/news
        # 都缺失时（回测无--news-file的常态），label 的边界和 strength 的
        # 0.7/0.4 阈值恰好重合，这里改用 label 选档不会改变任何结果（实测
        # 验证过，byte-for-byte一致）——真正让PARTIAL档产生折扣的是下面
        # position_scale 的取值，且已经改为"fund/news都缺失时强制1.0，不
        # 折算"（见 engine/scoring.py，2026-07-04 用户二次确认）。
        if score_label == scoring.LABEL_FULL:
            pct = (SIZING_PCT_STRONG_DYNAMIC if open_positions <= DYNAMIC_MAX_OPEN
                   else SIZING_PCT[0.7]) if stop_pct <= STOP_MAX[0.7] else SIZING_PCT[0.4]
        elif score_label == scoring.LABEL_OBSERVATION:
            # v2.2：技术面弱信号观察仓，独立配置项，不复用 SIZING_PCT[0.0]
            # （那是给不传 score_label 的旧 strength 阈值路径用的）。
            pct = config.OBSERVATION_POSITION_PCT
        else:   # LABEL_PARTIAL
            pct = SIZING_PCT[0.4] if stop_pct <= STOP_MAX[0.4] else SIZING_PCT[0.0]
    elif strength >= 0.7:
        if stop_pct <= STOP_MAX[0.7]:
            # Dynamic: fewer positions → bigger slice
            pct = (SIZING_PCT_STRONG_DYNAMIC if open_positions <= DYNAMIC_MAX_OPEN
                   else SIZING_PCT[0.7])
        else:
            pct = SIZING_PCT[0.4]   # downgrade: stop too wide
    elif strength >= 0.4:
        pct = SIZING_PCT[0.4] if stop_pct <= STOP_MAX[0.4] else SIZING_PCT[0.0]
    else:
        pct = SIZING_PCT[0.0]

    # Momentum-weighted tier: same helper used by live risk/sizing.py, so the
    # backtest and live paths apply the identical multiplier logic. Only the
    # cash tier is scaled — q_risk and q_strat below stay as hard caps, so
    # this can only redistribute within the existing risk budget, not expand it.
    pct *= rsi_multiplier(rsi_val, strategy)

    q_tier  = int(available_cash * pct / price)
    q_risk  = int(total_capital * RISK_PER_TRADE_PCT * weather_mult / (price * max(stop_pct, 0.01)))
    q_strat = int(available_cash * STRATEGY_MAX_PCT.get(strategy, 0.15) / price)
    qty     = min(q_tier, q_risk, q_strat)
    qty     = max(0, int(qty * kelly * position_scale))

    if weather_mult <= 0:
        return 0   # 状态0：新开仓硬拦截（q_risk 已经是0，这里主要是兜底）

    # 单股仓位硬顶——跟风险预算缩放无关，是独立的资金集中度上限。
    if total_capital > 0:
        qty = min(qty, int(config.MARKET_WEATHER_MAX_POSITION_PCT * total_capital / price))

    return max(0, qty)


# ── 投资组合模拟器 ────────────────────────────────────────────────────────────

def run_portfolio_backtest(stocks: List[str], start: str, end: str,
                           cash: float = 50_000.0,
                           news_feed: Optional[HistoricalNewsFeed] = None,
                           usd_to_jpy: Optional[float] = None) -> dict:
    """
    usd_to_jpy : if set, every US.* code's OHLC is multiplied by this fixed
                 rate right after fetch, before any signal/sizing logic runs.
                 Lets a single JPY-denominated cash pool trade US.* and JP.*
                 codes side by side, competing for the same slots.
                 NOTE: this is a STATIC rate applied across the whole backtest
                 window — it does not model real historical USD/JPY movement,
                 so any FX-driven gain/loss a real JPY account would have seen
                 is not captured. Regime/RSI/%B signals are unaffected (all
                 scale-invariant under a constant price multiplier); only cash
                 amounts, share counts and the printed $ /¥ P&L are affected.
    """
    currency = "JPY " if usd_to_jpy else "$"
    # ASCII-only label — Windows consoles running a GBK codepage raise
    # UnicodeEncodeError on '¥' (and other non-GBK glyphs) inside print().
    print(f"\n{'='*70}")
    print(f"  多仓位组合回测  {start} ~ {end}  初始资金 {currency}{cash:,.0f}")
    print(f"  股票池: {len(stocks)} 只   最大持仓: {MAX_POSITIONS} 只")
    print(f"  新闻过滤: {'启用' if news_feed else '未启用'}")
    if usd_to_jpy:
        print(f"  美元换算: 固定汇率 1 USD = {usd_to_jpy} JPY（静态，不反映历史汇率波动）")
    print(f"{'='*70}\n")

    prepared = _prepare_backtest_data(stocks, start, end, usd_to_jpy)
    if prepared is None:
        print("无数据，回测终止")
        return {}

    return simulate_from_prepared(prepared, cash=cash, news_feed=news_feed,
                                  currency=currency)


def _prepare_backtest_data(stocks: List[str], start: str, end: str,
                            usd_to_jpy: Optional[float] = None) -> Optional[dict]:
    """Fetch history + precompute signals/benchmark series (Steps 1-3).
    Independent of ATR_TRAIL_MULT and every other simulate_from_prepared()
    parameter, so a parameter sweep over the same (stocks, start, end) window
    can call this once and reuse the result across many simulated variants
    instead of re-fetching + re-computing signals per variant (see
    _atr_walkforward_sweep.py). Returns None if no data could be fetched.
    """
    print("拉取历史数据...")
    all_data: Dict[str, pd.DataFrame] = {}
    benchmark_df = None

    # QQQ gets extra pre-`start` history so its own MA200 (QQQ Beta底仓死命令
    # + 宏观技术熔断，两者都用 config.QQQ_MA_PERIOD=200) AND the dynamic-ATR
    # historical-vol-mean baseline (trailing 252-day mean of a 20-day realized
    # vol series -> needs ~272 prior trading days of its own) are already
    # warmed up by the time the actual simulation window begins — without
    # this, MA200/vol-mean stay NaN (-> checks read as False / dynamic mult
    # falls back to base) for the first several hundred trading days of ANY
    # backtest, which for an early `start` can silently blind these mechanisms
    # through the exact period they'd matter most.
    _QQQ_WARMUP_DAYS = 460  # calendar days: covers both MA200 (~320d) and the
                            # dynamic-ATR historical-vol-mean warmup (~400d)

    for code in stocks:
        try:
            fetch_start = start
            if code == QQQ_CODE:
                fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=_QQQ_WARMUP_DAYS)).strftime("%Y-%m-%d")
            df = _moomoo_fetch(code, fetch_start, end)
            df["date"] = pd.to_datetime(df["time_key"].str[:10])
            df = df.set_index("date").sort_index()
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            if usd_to_jpy and code.startswith("US."):
                df[["open", "high", "low", "close"]] *= usd_to_jpy
            all_data[code] = df
            if code == "US.QQQ":
                benchmark_df = df.copy()
            print(f"  {code:<12}  {len(df)} bars")
            time.sleep(0.5)   # OpenD rate limit: 60 req/30s
        except Exception as e:
            print(f"  {code:<12}  SKIP ({e})")

    if not all_data:
        return None

    # QQQ Beta 底仓的唯一进出场信号：收盘价是否高于 MA(config.QQQ_MA_PERIOD)。
    # 预计算一次，日常循环里直接按日期查表，和 regime/boll 等指标同样的模式
    # （避免在4.5万+行的逐日循环里重复算 rolling mean）。
    qqq_above_ma200 = pd.Series(dtype=bool)
    qqq_macro_halt  = pd.Series(dtype=bool)
    qqq_vol20       = pd.Series(dtype=float)
    qqq_weather     = pd.Series(dtype=int)
    qqq_vol_pct     = pd.Series(dtype=float)
    if benchmark_df is not None:
        qqq_close = benchmark_df["close"].astype(float)
        qqq_above_ma200 = qqq_close > qqq_close.rolling(config.QQQ_MA_PERIOD).mean()
        # QQQ macro technical halt: below MA200 AND own MA20 momentum falling
        # steeply -> block new stock-engine entries (see engine/regime.py).
        # Blocks new entries only, does NOT force-close existing positions —
        # same behavior as the news-based macro circuit breaker below.
        # v1.0-RELEASE-FINAL: locked as a hard block — position-scale and
        # dual-watchlist variants were tried and reverted (see project memory).
        # A Risk-Off half-size overlay (position_scale x0.5 on high-vol/
        # below-MA252 days) was also tried and rolled back 2026-07-04: it
        # cut commission-stress sensitivity for real (-35.6%->-19.2% relative
        # return drop under 2x commission) but cost 28.5% of baseline return
        # and 9.6% of baseline Sharpe (277.18%/0.920 -> 198.29%/0.832), and
        # only modestly improved tail risk once compared on a like-for-like
        # Monte Carlo basis (5th-pct worst drawdown -19.55%->-17.93%). Decision
        # was to keep the pure hard-halt-only baseline instead of taking that
        # trade-off. See engine.regime.qqq_risk_off_series() docstring.
        qqq_macro_halt = _qqq_macro_halt_series(qqq_close)
        # Raw 20-day realized-vol series for QQQ (causal). This is the
        # "current_vol_t" input to the dynamic-ATR formula -- kept as a raw
        # series here (not yet turned into a multiplier) so a hyperparameter
        # sweep (base/alpha/min/max) can build many different multiplier
        # series from this one fetch without re-fetching. See
        # engine.regime.dynamic_atr_multiplier_series() for the transform.
        qqq_vol20 = _qqq_realized_vol_series(qqq_close, window=20)
        # Market Weather 状态码（模块一，2026-07-04 新增）——只喂给下面模块二
        # 的仓位风险乘数，不改变/不替代上面 qqq_macro_halt 的硬拦截判断。
        qqq_weather = _market_weather_series(qqq_close, ma_long_period=config.QQQ_MA_PERIOD)
        # v2.4 RSL: 最近N日(默认5)市场波动率相对其自身历史的因果百分位排名
        # ——用于Replacement Confidence Buffer的volatility_factor，见
        # portfolio/replacement_stabilizer.py。
        qqq_vol_pct = _market_volatility_percentile(
            qqq_close, window=config.REPLACEMENT_VOL_LOOKBACK_DAYS)

    # ── Step 2: 预计算信号 ───────────────────────────────────────────────────
    print("\n计算信号序列...")
    signals: Dict[str, pd.DataFrame] = {}
    for code, df in all_data.items():
        if code == "US.QQQ":
            continue
        try:
            sig = _compute_all_signals(df)
            signals[code] = sig
        except Exception as e:
            print(f"  {code} signal error: {e}")

    # ── Step 3: 交易日历 = QQQ 交易日（基准），个股按各自有无数据处理 ────────────
    # 不用全体交集——新股上市晚会把回测截断到2023，丢失2022年数据
    if benchmark_df is not None:
        all_dates = sorted(d for d in benchmark_df.index
                           if pd.Timestamp(d) >= pd.Timestamp(start))
    else:
        all_union = set()
        for df in all_data.values():
            all_union.update(df.index)
        all_dates = sorted(d for d in all_union
                           if pd.Timestamp(d) >= pd.Timestamp(start))

    return {
        "stocks": stocks, "start": start, "end": end,
        "all_data": all_data,
        "signals": signals,
        "benchmark_df": benchmark_df,
        "qqq_above_ma200": qqq_above_ma200,
        "qqq_macro_halt": qqq_macro_halt,
        "qqq_vol20": qqq_vol20,
        "qqq_weather": qqq_weather,
        "qqq_vol_pct": qqq_vol_pct,
        "all_dates": all_dates,
    }


def simulate_from_prepared(prepared: dict, cash: float,
                           news_feed: Optional[HistoricalNewsFeed] = None,
                           currency: str = "$",
                           sim_start: Optional[str] = None,
                           sim_end: Optional[str] = None,
                           atr_mult_series: Optional[pd.Series] = None) -> dict:
    """Steps 4-5: day-by-day simulation + stats, run against data already
    fetched by _prepare_backtest_data(). Split out so a parameter sweep (e.g.
    over ATR_TRAIL_MULT) can re-simulate many times against one fetch instead
    of re-fetching + re-computing signals for every variant.

    sim_start/sim_end : optional sub-window inside the fetched range (must
        both be within [prepared['start'], prepared['end']]). Lets an
        IS/OOS walk-forward harness fetch the FULL history once and then
        simulate many disjoint folds against it without re-fetching. Falls
        back to the full prepared start/end when omitted. Every fold starts
        fresh (cash reset, no carried positions) -- callers that want a
        stitched multi-fold equity curve should compound the returned daily
        % returns across folds themselves.
    atr_mult_series : optional date-indexed Series giving the ATR trailing-
        stop multiplier to use on each day (see
        engine.regime.dynamic_atr_multiplier_series()). If omitted, falls
        back to the flat module constant ATR_TRAIL_MULT everywhere, exactly
        as before this parameter existed.
    """
    stocks          = prepared["stocks"]
    start           = sim_start or prepared["start"]
    end             = sim_end or prepared["end"]
    all_data        = prepared["all_data"]
    signals         = prepared["signals"]
    benchmark_df    = prepared["benchmark_df"]
    qqq_above_ma200 = prepared["qqq_above_ma200"]
    qqq_macro_halt  = prepared["qqq_macro_halt"]
    qqq_weather     = prepared["qqq_weather"]
    qqq_vol_pct     = prepared.get("qqq_vol_pct", pd.Series(dtype=float))

    all_dates = [d for d in prepared["all_dates"]
                 if pd.Timestamp(start) <= pd.Timestamp(d) <= pd.Timestamp(end)]

    print(f"\n共 {len(all_dates)} 个交易日，开始模拟...\n")

    # ── Step 4: 逐日模拟 ─────────────────────────────────────────────────────
    portfolio_cash  = cash
    positions: Dict[str, dict] = {}   # code → position record
    equity_curve: List[float] = []
    deployed_curve: List[float] = []   # analytics-only, parallels equity_curve
    date_list:    List = []
    trade_log:    List[dict] = []
    # analytics-only（不影响任何交易决策/仓位计算）：记录因 MAX_POSITIONS /
    # MAX_TOTAL_EXPOSURE 容量上限而被迫放弃的候选，用于离线分析"强信号被
    # 弱信号挤占仓位名额"这类问题（见 analytics/portfolio_gap_study.py）。
    capacity_blocks: List[dict] = []
    # v2.3 Portfolio Capacity Manager：记录每一次主动置换（换出OBSERVATION
    # 腾出名额给FULL信号），用于回归验证报告（见 analytics/ 或调用方统计）。
    replacements: List[dict] = []
    # 2026-07-05 Explain Layer：非RSL分支下每一次容量审查（MAX_POSITIONS已满
    # 且新信号是FULL）只要找到候选（不论margin是否满足）就记一条完整决策
    # 记录，供 Replacement Attempts/Success/Blocked 等统计使用（见
    # portfolio/capacity_manager.py::evaluate_replacement()）。RSL开启时
    # （config.ENABLE_REPLACEMENT_STABILIZATION=True）不写入此列表——RSL有
    # 自己的 shadow_replacements/decide() 诊断路径，两套机制不混用同一份
    # 统计口径。
    replacement_attempts: List[dict] = []
    peak_equity   = cash
    realized_pnl  = 0.0
    news_stats    = {"blocked": 0, "boosted": 0, "macro_block_days": 0}
    cooldowns:    Dict[str, int] = {}   # code -> day_idx before which new entries are blocked

    # ── v2.4 Replacement Stabilization Layer（RSL）状态 ──────────────────────
    # replacement_cooldown: code -> 冷却到期day_idx，victim/beneficiary双方
    #   一次置换后都会被设置，跨天生效（见 config.REPLACEMENT_COOLDOWN_DAYS）。
    # replaced_today: 当天已经参与过置换的code集合，每个day_idx开头清空，
    #   用于阻断同一天内的链式置换（A->B->C->D）。
    # full_score_history: 滚动的历史FULL标签总分列表，供WEAK FULL档位算
    #   百分位cutoff用（见 portfolio/replacement_stabilizer.py）。
    replacement_cooldown: Dict[str, int] = {}
    full_score_history: List[float] = []
    _FULL_SCORE_HISTORY_CAP = 500
    # WEAK FULL shadow mode（config.REPLACEMENT_WEAK_FULL_ENABLED=False）：
    # 记录"如果放行会换出谁"，但不执行——用于ablation实验判断这个低频档位
    # 是尾部风险截断还是冗余路径，见 analytics/weak_full_ablation.py。
    shadow_replacements: List[dict] = []

    def _deployed() -> float:
        return sum(p["avg_cost"] * p["qty"] for p in positions.values())

    def _equity() -> float:
        return cash + realized_pnl  # realized equity for Kelly

    def _sector(code):
        return SECTOR_MAP.get(code, "other")

    def _undo_replacement(victim_code, victim_pos, v_exit_price, v_qty, v_fee, v_pnl):
        """v2.3: 撤销一次已经tentatively执行的 Active Replacement——用在换出
        OBSERVATION腾出名额后，FULL信号本身却因为敞口/板块/风险上限拿不到
        任何仓位（qty<=0）的情况，避免"卖了但没买成"的净损失。恢复持仓、
        撤回trade_log/replacements记录，返回需要从portfolio_cash/
        realized_pnl里减掉的金额（调用方自己更新这两个变量，因为它们不是
        闭包能直接改写的外层局部变量）。"""
        positions[victim_code] = victim_pos
        trade_log.pop()
        replacements.pop()
        return v_exit_price * v_qty - v_fee, v_pnl - v_fee

    def _sector_exposure(sector: str) -> float:
        dep = sum(p["avg_cost"] * p["qty"]
                  for c, p in positions.items() if _sector(c) == sector)
        return dep / cash if cash > 0 else 0.0

    for day_idx, today in enumerate(all_dates):
        ts = pd.Timestamp(today)
        replaced_today: set = set()   # v2.4 RSL: 同日链式置换阻断，逐日重置
        cur_atr_mult = (atr_mult_series.get(ts, ATR_TRAIL_MULT)
                        if atr_mult_series is not None else ATR_TRAIL_MULT)
        weather_code = int(qqq_weather.get(ts, 1))   # 缺失日期默认状态1（谨慎）

        # ── 4a. 检查退出 ─────────────────────────────────────────────────────
        to_close = []
        for code, pos in positions.items():
            if ts not in all_data[code].index:
                continue
            bar  = all_data[code].loc[ts]
            low  = float(bar["low"])
            high = float(bar["high"])
            clo  = float(bar["close"])
            ep   = pos["avg_cost"]
            strat = pos.get("strategy", "")

            reason = ""
            exit_price = clo

            # ── QQQ Beta 底仓："死仓"，唯一退出条件是跌破 MA200 ─────────────
            # 死命令（用户明确要求）：continue 提前跳出，不会走到下面的
            # is_trend/ATR跟踪止损分支——哪怕日内剧烈震荡（4×ATR级别）也
            # 绝对不会触发卖出，只有跌破MA200才清仓。
            if strat == "core_etf":
                if not qqq_above_ma200.get(ts, False):
                    reason = "MA200_BREAK"
                    exit_price = clo
                if reason:
                    to_close.append((code, reason, exit_price))
                continue   # 跳过普通逻辑

            # Priority 1: hard stop-loss (check LOW of the day)
            stop_price = ep * (1 - STOP_LOSS_PCT)
            if low <= stop_price:
                reason = "STOP_LOSS"
                exit_price = min(clo, stop_price)

            is_trend = strat in _TREND_STRATS

            if is_trend:
                # 趋势策略：无硬止盈，4×ATR 棘轮跟踪止损让利润奔跑
                atr_now = (signals[code].loc[ts, "atr"]
                           if code in signals and ts in signals[code].index else 0.0)
                if not np.isnan(atr_now) and atr_now > 0:
                    new_trail = clo - cur_atr_mult * atr_now
                    old_trail = pos.get("trail_stop", ep - cur_atr_mult * pos.get("entry_atr", atr_now))
                    pos["trail_stop"] = max(old_trail, new_trail)

                trail = pos.get("trail_stop")
                if not reason and trail and low <= trail:
                    reason = "ATR_TRAIL"
                    exit_price = min(clo, trail)
            else:
                # 均值回归策略：硬止盈 +15%
                if not reason and high >= ep * (1 + TAKE_PROFIT_PCT):
                    reason = "TAKE_PROFIT"
                    exit_price = ep * (1 + TAKE_PROFIT_PCT)

            # 策略 SELL 信号（所有策略）
            if not reason:
                if code in signals and ts in signals[code].index:
                    if signals[code].loc[ts, "signal"] == "SELL":
                        reason = "STRAT_EXIT"
                        exit_price = clo

            if reason:
                to_close.append((code, reason, exit_price))
            else:
                pos["highest_close"] = max(pos.get("highest_close", ep), clo)

        for code, reason, exit_price in to_close:
            pos = positions.pop(code)
            qty = pos["qty"]
            pnl = (exit_price - pos["avg_cost"]) * qty
            fee = exit_price * qty * COMMISSION
            portfolio_cash += exit_price * qty - fee
            realized_pnl += pnl - fee

            # TRENDING_EARLY trial stopped out before it could confirm/promote
            # -> cool the stock down so we don't whipsaw-re-enter the range top.
            if reason == "STOP_LOSS" and pos.get("strategy") == "atr_breakout_early":
                held_days = day_idx - pos.get("entry_day_idx", day_idx)
                if held_days <= config.TRENDING_EARLY_STOPOUT_LOOKBACK_DAYS:
                    cooldowns[code] = day_idx + config.TRENDING_EARLY_COOLDOWN_DAYS

            round_trip_fee = round(fee + pos.get("avg_cost", exit_price) * qty * COMMISSION, 2)
            trade_log.append({
                "date": today, "code": code, "side": "SELL",
                "qty": qty, "price": exit_price,
                "pnl": round(pnl - fee, 2), "reason": reason,
                "strategy": pos.get("strategy"),
                "entry_date": pos.get("entry_date"),
                "fee": round_trip_fee,
            })
            eq = _equity()
            if eq > peak_equity:
                peak_equity = eq

        # ── 活跃仓位数（不含 QQQ 底仓）─────────────────────────────────────
        def _active_count():
            return sum(1 for p in positions.values() if p.get("strategy") != "core_etf")

        # ── v2.4 RSL：滚动窗口内的置换计数（Budget/Crowding两个机制共用）───────
        def _replacement_count_within(window_days: int) -> int:
            return sum(1 for r in replacements if day_idx - r["day_idx"] < window_days)

        # Tier2(LOW_PARTIAL)+Tier3(WEAK_FULL)子预算专用计数——只统计这两个
        # 新增通道，跟上面的总预算计数分开算，见
        # config.REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS。
        def _new_tier_replacement_count_within(window_days: int) -> int:
            return sum(1 for r in replacements
                       if day_idx - r["day_idx"] < window_days
                       and r.get("replacement_type") in
                       (rsl.REPL_LOW_PARTIAL_EVICT, rsl.REPL_FULL_DOWNGRADE_REPLACE))

        # ── 4b. 计算 Kelly 系数 ───────────────────────────────────────────────
        eq = _equity()
        if peak_equity > 0 and (peak_equity - eq) / peak_equity >= HALFKELLY_THRESHOLD:
            kelly = 0.5
        elif peak_equity > 0 and (peak_equity - eq) / peak_equity < HALFKELLY_RECOVERY:
            kelly = 1.0
        else:
            kelly = 1.0

        # ── 4b-2. 晋升：TRENDING_EARLY 试错仓位一旦 regime 确认为 TRENDING_UP，
        #        补仓到满仓（30% -> 100%），策略标签升级为 atr_breakout ─────────
        for code, pos in list(positions.items()):
            if pos.get("strategy") != "atr_breakout_early":
                continue
            if code not in signals or ts not in signals[code].index:
                continue
            row = signals[code].loc[ts]
            if row.get("regime") != "TRENDING_UP":
                continue
            if ts not in all_data[code].index:
                continue
            price = float(all_data[code].loc[ts, "close"])
            if price <= 0:
                continue
            row_strength = row["strength"]
            strength = (float(row_strength) if not np.isnan(row_strength) and row_strength > 0
                       else pos.get("signal_strength", 0.5))
            stop_pct = float(row["stop_pct"]) if not np.isnan(row["stop_pct"]) else STOP_LOSS_PCT
            rsi_val  = float(row["rsi14"]) if not np.isnan(row["rsi14"]) else None

            pos_value  = pos["avg_cost"] * pos["qty"]
            target_qty = _size(portfolio_cash + pos_value, price, strength, stop_pct, cash,
                               "atr_breakout", kelly, open_positions=_active_count(),
                               rsi_val=rsi_val, market_weather_code=weather_code)
            add_qty = target_qty - pos["qty"]
            if add_qty <= 0:
                continue
            if (_deployed() / cash) >= MAX_TOTAL_EXPOSURE:
                continue   # portfolio-wide exposure cap — don't grow into it
            cost = price * add_qty
            fee  = cost * COMMISSION
            if cost + fee > portfolio_cash:
                add_qty = int((portfolio_cash * 0.98) / (price * (1 + COMMISSION)))
                if add_qty <= 0:
                    continue
                cost = price * add_qty
                fee  = cost * COMMISSION

            portfolio_cash -= (cost + fee)
            old_qty, old_cost = pos["qty"], pos["avg_cost"]
            total_qty = old_qty + add_qty
            pos["avg_cost"] = (old_qty * old_cost + add_qty * price) / total_qty
            pos["qty"]      = total_qty
            pos["strategy"] = "atr_breakout"   # confirmed — promoted out of trial
            trade_log.append({
                "date": today, "code": code, "side": "BUY",
                "qty": add_qty, "price": price,
                "pnl": None, "reason": "PROMOTE",
                "strategy": "atr_breakout",
            })

        # ── 4c. 收集 BUY 信号并排序 ──────────────────────────────────────────
        buy_candidates = []
        for code, sig_df in signals.items():
            if code in positions:
                continue
            if cooldowns.get(code, -1) > day_idx:
                continue
            if ts not in sig_df.index:
                continue
            row = sig_df.loc[ts]
            if row["signal"] != "BUY" or row["strength"] <= 0:
                continue
            if ts not in all_data[code].index:
                continue
            price = float(all_data[code].loc[ts, "close"])
            if price <= 0:
                continue
            buy_candidates.append({
                "code": code, "price": price,
                "strength": float(row["strength"]),
                "stop_pct": float(row["stop_pct"]) if not np.isnan(row["stop_pct"]) else STOP_LOSS_PCT,
                "strategy": str(row["strategy"]),
                "atr": float(row["atr"]) if not np.isnan(row["atr"]) else 0.0,
                "rsi14": float(row["rsi14"]) if not np.isnan(row["rsi14"]) else None,
            })

        buy_candidates = [c for c in buy_candidates if c["strength"] >= MIN_ENTRY_STRENGTH]

        # ── 4c-macro. QQQ 宏观技术熔断：跌破MA200且自身MA20动量严重向下 ──────
        # 只挡新开仓，已持仓位不受影响（正常走自己的止损/跟踪止损/策略SELL）。
        # Market Weather 状态0（模块一）并入同一个拦截开关，不新建平行分支。
        if qqq_macro_halt.get(ts, False) or weather_code == 0:
            news_stats["macro_block_days"] += 1
            buy_candidates = []

        # ── 4c-news. 新闻黑天鹅熔断（可选，仅在提供 --news-file 时生效）───────
        # 沿用历史新闻CSV里的真实 headlines 判断circuit breaker（跟实盘
        # engine.news.check_circuit_breaker 是同一层含义：财务造假/退市/SEC
        # 调查这类事件不管总分多高都要拦截，不是靠总分模型的权重稀释掉的）。
        if news_feed is not None:
            macro_blocked = ""
            for idx_code in ("US.QQQ", "US.SPY"):
                _, blocked = news_feed.score_asof(idx_code, ts)
                if blocked:
                    macro_blocked = f"{idx_code}:{blocked}"
                    break
            if macro_blocked:
                news_stats["macro_block_days"] += 1
                buy_candidates = []

        # ── 4c-score. v2.1 横向多因子总分：趋势(40%)+基本面(20%)+新闻(20%)+
        #    天气(20%)，见 engine/scoring.py。基本面在回测里没有真实历史数据
        #    源，恒为 None（剔除，不参与加权，见 engine/fundamental.py）。
        #    新闻：提供 --news-file 时用真实历史 headlines 跑三级分类器
        #    （跟实盘同一套 engine.news_filter 分类逻辑，真实参与加权），
        #    否则同样是 None（剔除，不是编造历史新闻数据）。
        #    v2.2 四档状态机：总分<40（或天气state0，已在上面 4c-macro 拦截）
        #    彻底放弃开仓；40-59 OBSERVATION（技术面弱信号观察仓，固定
        #    config.OBSERVATION_POSITION_PCT，见 _size()）；60-79 PARTIAL
        #    按总分比例折算仓位；>=80 FULL 满仓。
        scored_candidates = []
        for c in buy_candidates:
            if news_feed is not None:
                titles = news_feed.headlines_asof(c["code"], ts)
                tier, _matched = news_filter.classify_headlines(titles)
                if tier == 1:
                    news_stats["blocked"] += 1
                    continue   # Tier1黑天鹅一票否决，不进入总分模型
                news_score_val = news_filter.TIER2_SCORE if tier == 2 else news_filter.TIER3_SCORE
                if tier == 2:
                    news_stats["boosted"] += 1   # 复用既有统计字段：这次跑发生了新闻降级
            else:
                news_score_val = None

            result = scoring.compute_total_score(
                trend_strength=c["strength"],
                weather_code=weather_code,
                fundamental_score=_FUNDAMENTAL_SCORE_BACKTEST,
                news_score=news_score_val,
            )
            if result.label == scoring.LABEL_SKIP:
                continue
            c["news_score"]        = news_score_val
            c["fundamental_score"] = _FUNDAMENTAL_SCORE_BACKTEST
            c["weather_score"]     = result.weather_score
            c["trend_score"]       = result.trend_score
            c["total_score"]       = result.total
            c["total_score_scale"] = result.position_scale
            c["score_label"]       = result.label
            scored_candidates.append(c)
            # v2.4 RSL: 滚动记录历史FULL分数分布（不论最终是否成交），供
            # WEAK FULL档位算百分位cutoff用，见 portfolio/replacement_stabilizer.py。
            if result.label == scoring.LABEL_FULL:
                full_score_history.append(result.total)
                if len(full_score_history) > _FULL_SCORE_HISTORY_CAP:
                    del full_score_history[:-_FULL_SCORE_HISTORY_CAP]
        buy_candidates = scored_candidates

        buy_candidates.sort(key=lambda x: x["strength"], reverse=True)

        # ── 4e. 开仓（活跃信号仓位）─────────────────────────────────────────
        total_cap = cash
        for idx, cand in enumerate(buy_candidates):
            replacement_from = None
            replacement_score_delta = None

            if _active_count() >= MAX_POSITIONS:
                # ── v2.3 Portfolio Capacity Manager：Active Replacement ──
                # 只有新信号是FULL时才进入Capacity Review。v2.4 RSL（若开启，
                # config.ENABLE_REPLACEMENT_STABILIZATION）在v2.3原判断之外
                # 叠加冷却/预算/自适应门槛/LOW_PARTIAL+WEAK_FULL两档新增优先级
                # /稳定性保护，见 portfolio/replacement_stabilizer.py；关闭时
                # 完全回退到v2.3原始调用方式（100%复现，用于A/B回归对比）。
                victim_code = None
                replacement_type = None
                replacement_stability_score = None
                cooldown_blocked = False
                budget_blocked = False
                new_tier_budget_blocked = False

                if config.ENABLE_ACTIVE_REPLACEMENT and cand["score_label"] == scoring.LABEL_FULL:
                    held = {c: p for c, p in positions.items()
                            if p.get("strategy") != "core_etf"}

                    if config.ENABLE_REPLACEMENT_STABILIZATION:
                        vol_pct = float(qqq_vol_pct.get(ts, 0.0))
                        vol_factor = (vol_pct / 100.0) * config.REPLACEMENT_VOL_FACTOR_MAX
                        crowd_factor = min(
                            config.REPLACEMENT_CROWDING_FACTOR_MAX,
                            _replacement_count_within(config.REPLACEMENT_CROWDING_WINDOW_DAYS)
                            * config.REPLACEMENT_CROWDING_FACTOR_PER_EVENT)
                        decision = rsl.decide(
                            incoming_code=cand["code"], incoming_score=cand["total_score"],
                            held_positions=held, current_day_idx=day_idx,
                            volatility_factor=vol_factor, crowding_factor=crowd_factor,
                            full_score_history=full_score_history,
                            cooldown_until=replacement_cooldown, replaced_today=replaced_today,
                            recent_replacement_count=_replacement_count_within(100),
                            recent_new_tier_replacement_count=_new_tier_replacement_count_within(
                                config.REPLACEMENT_NEW_TIER_BUDGET_WINDOW_DAYS))
                        victim_code                 = decision.victim_code
                        replacement_type             = decision.replacement_type
                        replacement_stability_score  = decision.stability_score
                        cooldown_blocked              = decision.cooldown_blocked
                        budget_blocked                = decision.budget_blocked
                        new_tier_budget_blocked        = decision.new_tier_budget_blocked
                        # v2.4 shadow mode：三档都可能各自产生一次"如果放行
                        # 会换出谁"的观测（同一次审查里最多各出现一次，互不
                        # 排斥——见 portfolio/replacement_stabilizer.py）。
                        for shadow_type, shadow_victim, shadow_stability in (
                            (rsl.REPL_OBSERVATION_EVICT, decision.shadow_observation_victim, None),
                            (rsl.REPL_LOW_PARTIAL_EVICT, decision.shadow_low_partial_victim,
                             decision.shadow_low_partial_stability_score),
                            (rsl.REPL_FULL_DOWNGRADE_REPLACE, decision.shadow_weak_full_victim,
                             decision.shadow_weak_full_stability_score),
                        ):
                            if shadow_victim is None:
                                continue
                            v_pos = held[shadow_victim]
                            v_price_now = (float(all_data[shadow_victim].loc[ts, "close"])
                                           if ts in all_data[shadow_victim].index
                                           else v_pos.get("avg_cost"))
                            # 审查这一刻的mark-to-market浮盈浮亏——用于事后跟
                            # "如果不换、真正走完"的最终已实现PnL做差，得到
                            # "从这一刻起，多让它走一段"这个决策本身贡献的
                            # 那部分盈亏（而不是把整笔交易从建仓开始的PnL都
                            # 算成这个决策的功劳/代价）。
                            v_mtm_pnl_now = (v_price_now - v_pos.get("avg_cost", v_price_now)) * v_pos.get("qty", 0)
                            shadow_replacements.append({
                                "date": today, "day_idx": day_idx,
                                "replacement_type": shadow_type,
                                "would_be_victim": shadow_victim,
                                "would_be_victim_score": v_pos.get("total_score"),
                                "would_be_victim_entry_date": v_pos.get("entry_date"),
                                "would_be_victim_entry_day_idx": v_pos.get("entry_day_idx"),
                                "would_be_victim_mtm_pnl_at_review": round(v_mtm_pnl_now, 2),
                                "incoming_code": cand["code"],
                                "incoming_score": cand["total_score"],
                                "stability_score": shadow_stability,
                            })
                    else:
                        # 2026-07-05 独立 WEAK_FULL 通道 + Explain Layer：
                        # evaluate_replacement() 内部先看OBSERVATION候选，
                        # 找不到时（若 config.REPLACEMENT_STANDALONE_
                        # WEAK_FULL_ENABLED）再看WEAK_FULL候选，不带RSL的
                        # cooldown/预算/自适应门槛/Stability Score。只要找到
                        # 候选（不论margin是否满足）就记一条完整决策记录，
                        # 供 Replacement Attempts/Success/Blocked 等统计使用。
                        evaluation = capacity_manager.evaluate_replacement(
                            incoming_code=cand["code"], incoming_score=cand["total_score"],
                            held_positions=held, current_day_idx=day_idx,
                            full_score_history=full_score_history)
                        if evaluation.decision != "NO_CANDIDATE":
                            replacement_attempts.append({
                                "date": today, "day_idx": day_idx,
                                "incoming_code": evaluation.incoming_code,
                                "victim_code": evaluation.victim_code,
                                "replacement_type": evaluation.replacement_type,
                                "old_score": evaluation.old_score,
                                "new_score": evaluation.new_score,
                                "score_difference": evaluation.score_difference,
                                "margin": evaluation.margin,
                                "margin_satisfied": evaluation.margin_satisfied,
                                "decision": evaluation.decision,
                            })
                        if evaluation.decision == "REPLACE":
                            victim_code = evaluation.victim_code
                            replacement_type = evaluation.replacement_type

                if victim_code is None:
                    capacity_blocks.append({
                        "date": today, "reason": "MAX_POSITIONS",
                        "cooldown_blocked": cooldown_blocked,
                        "budget_blocked": budget_blocked,
                        "new_tier_budget_blocked": new_tier_budget_blocked,
                        "blocked_candidates": [
                            {"code": c["code"], "score_label": c["score_label"],
                             "total_score": c["total_score"], "strength": c["strength"],
                             "price": c["price"]}
                            for c in buy_candidates[idx:]
                        ],
                        "occupied_positions": [
                            {"code": c, "score_label": p.get("score_label")}
                            for c, p in positions.items() if p.get("strategy") != "core_etf"
                        ],
                    })
                    break

                # ── 原子执行：close(victim) -> release slot -> open(FULL)
                # 全部发生在同一次 for-loop 迭代里（同一个 Decision Tick），
                # 不拆分到多个事件循环——紧接着的开仓逻辑（下面）会在本次
                # 迭代内直接用刚释放出来的名额和现金，不存在"名额已释放但
                # 还没建仓"的中间态。
                victim_pos = positions.pop(victim_code)
                v_qty = victim_pos["qty"]
                v_exit_price = (float(all_data[victim_code].loc[ts, "close"])
                                if ts in all_data[victim_code].index else victim_pos["avg_cost"])
                v_pnl = (v_exit_price - victim_pos["avg_cost"]) * v_qty
                v_fee = v_exit_price * v_qty * COMMISSION
                portfolio_cash += v_exit_price * v_qty - v_fee
                realized_pnl += v_pnl - v_fee
                replacement_score_delta = round(
                    cand["total_score"] - (victim_pos.get("total_score") or 0.0), 2)
                round_trip_fee = round(
                    v_fee + victim_pos.get("avg_cost", v_exit_price) * v_qty * COMMISSION, 2)
                trade_log.append({
                    "date": today, "code": victim_code, "side": "SELL",
                    "qty": v_qty, "price": v_exit_price,
                    "pnl": round(v_pnl - v_fee, 2), "reason": "ACTIVE_REPLACEMENT",
                    "strategy": victim_pos.get("strategy"),
                    "entry_date": victim_pos.get("entry_date"),
                    "fee": round_trip_fee,
                    "replacement_to": cand["code"],
                    "replacement_score_delta": replacement_score_delta,
                    "replacement_type": replacement_type,
                    "replacement_stability_score": replacement_stability_score,
                })
                replacements.append({
                    "date": today, "day_idx": day_idx,
                    "replacement_from": victim_code, "replacement_to": cand["code"],
                    "victim_score": victim_pos.get("total_score"),
                    "incoming_score": cand["total_score"],
                    "score_delta": replacement_score_delta,
                    "holding_days_before_replacement":
                        day_idx - victim_pos.get("entry_day_idx", day_idx),
                    "victim_pnl": round(v_pnl - v_fee, 2),
                    "replacement_type": replacement_type,
                    "replacement_stability_score": replacement_stability_score,
                })
                replacement_from = victim_code

            deployed = _deployed()
            if (deployed / total_cap) >= MAX_TOTAL_EXPOSURE:
                if replacement_from is not None:
                    # 换出了OBSERVATION，但敞口上限仍然拦住了FULL——撤销
                    # 这次置换，不留下"卖了但没买成"的净损失。
                    cash_undo, pnl_undo = _undo_replacement(
                        replacement_from, victim_pos, v_exit_price, v_qty, v_fee, v_pnl)
                    portfolio_cash -= cash_undo
                    realized_pnl -= pnl_undo
                    replacement_from = None
                capacity_blocks.append({
                    "date": today, "reason": "MAX_TOTAL_EXPOSURE",
                    "blocked_candidates": [
                        {"code": c["code"], "score_label": c["score_label"],
                         "total_score": c["total_score"], "strength": c["strength"],
                         "price": c["price"]}
                        for c in buy_candidates[idx:]
                    ],
                    "occupied_positions": [
                        {"code": c, "score_label": p.get("score_label")}
                        for c, p in positions.items() if p.get("strategy") != "core_etf"
                    ],
                })
                break

            code     = cand["code"]
            price    = cand["price"]
            strength = cand["strength"]
            stop_pct = cand["stop_pct"]
            strategy = cand["strategy"]
            sector   = _sector(code)

            if _sector_exposure(sector) >= MAX_SECTOR_EXPOSURE:
                if replacement_from is not None:
                    cash_undo, pnl_undo = _undo_replacement(
                        replacement_from, victim_pos, v_exit_price, v_qty, v_fee, v_pnl)
                    portfolio_cash -= cash_undo
                    realized_pnl -= pnl_undo
                    replacement_from = None
                continue

            avail = portfolio_cash
            # v2.1 总分模型算出的 position_scale 和既有 TRENDING_EARLY 试错
            # 仓位系数相乘——两者都只裁剪、不放大，跟 risk/sizing.py::
            # calculate() 里 position_scale 参数"trims, never expands"的既有
            # 语义一致，不新增平行的仓位裁剪路径。
            trial_scale = (config.TRENDING_EARLY_POSITION_SCALE
                           if strategy == "atr_breakout_early" else 1.0)
            position_scale = trial_scale * cand["total_score_scale"]
            qty = _size(avail, price, strength, stop_pct, total_cap, strategy, kelly,
                        open_positions=_active_count(), rsi_val=cand.get("rsi14"),
                        position_scale=position_scale, market_weather_code=weather_code,
                        score_label=cand["score_label"])
            if qty <= 0:
                if replacement_from is not None:
                    cash_undo, pnl_undo = _undo_replacement(
                        replacement_from, victim_pos, v_exit_price, v_qty, v_fee, v_pnl)
                    portfolio_cash -= cash_undo
                    realized_pnl -= pnl_undo
                    replacement_from = None
                continue
            cost = price * qty
            fee  = cost * COMMISSION
            if cost + fee > portfolio_cash:
                qty = int((portfolio_cash * 0.98) / (price * (1 + COMMISSION)))
                if qty <= 0:
                    if replacement_from is not None:
                        cash_undo, pnl_undo = _undo_replacement(
                            replacement_from, victim_pos, v_exit_price, v_qty, v_fee, v_pnl)
                        portfolio_cash -= cash_undo
                        realized_pnl -= pnl_undo
                        replacement_from = None
                    continue
                cost = price * qty
                fee  = cost * COMMISSION

            portfolio_cash -= (cost + fee)
            atr_val = cand["atr"]
            is_trend_entry = strategy in _TREND_STRATS
            trail = (price - cur_atr_mult * atr_val
                     if is_trend_entry and atr_val > 0 else None)

            positions[code] = {
                "qty":             qty,
                "entry_price":     price,
                "avg_cost":        price,
                "strategy":        strategy,
                "entry_date":      today,
                "entry_day_idx":   day_idx,
                "signal_strength": strength,
                "highest_close":   price,
                "entry_atr":       atr_val,
                "trail_stop":      trail,
                "score_label":     cand["score_label"],   # analytics-only
                "total_score":     cand["total_score"],   # analytics-only + v2.3 capacity manager input
            }
            trade_log.append({
                "date": today, "code": code, "side": "BUY",
                "qty": qty, "price": price,
                "pnl": None, "reason": "SIGNAL",
                "strategy": strategy,
                "trend_score":       cand["trend_score"],
                "fundamental_score": cand["fundamental_score"],
                "news_score":        cand["news_score"],
                "weather_score":     cand["weather_score"],
                "total_score":       cand["total_score"],
                "score_label":       cand["score_label"],
                "replacement_from":        replacement_from,
                "replacement_score_delta": replacement_score_delta,
            })

            # v2.4 RSL：置换在此处才算真正落地（前面任何一处undo都不会走到
            # 这里），此时才设置冷却/同日阻断状态——避免了给_undo_replacement
            # 额外增加RSL状态回滚逻辑：没落地就永远不会写入这两个结构。
            if replacement_from is not None and config.ENABLE_REPLACEMENT_STABILIZATION:
                replacement_cooldown[replacement_from] = day_idx + config.REPLACEMENT_COOLDOWN_DAYS
                replacement_cooldown[code]              = day_idx + config.REPLACEMENT_COOLDOWN_DAYS
                replaced_today.add(replacement_from)
                replaced_today.add(code)

        # ── 4f. QQQ Beta 底仓：固定目标仓位，只要空仓且 QQQ>MA200 就买回 ──────
        # 不再看活跃仓位数量——这是永远划出的固定死仓，不是"信号不够时的填充"。
        if (QQQ_CODE not in positions
                and qqq_above_ma200.get(ts, False)
                and ts in all_data.get(QQQ_CODE, pd.DataFrame()).index):
            qprice = float(all_data[QQQ_CODE].loc[ts, "close"])
            deployed = _deployed()
            target_value = config.QQQ_CORE_TARGET_PCT * cash
            remaining = min(
                portfolio_cash * 0.98,
                target_value,
                total_cap * MAX_TOTAL_EXPOSURE - deployed,
            )
            if remaining > qprice:
                qty = int(remaining / (qprice * (1 + COMMISSION)))
                if qty > 0:
                    cost = qprice * qty
                    fee  = cost * COMMISSION
                    portfolio_cash -= (cost + fee)
                    positions[QQQ_CODE] = {
                        "qty":         qty,
                        "entry_price": qprice,
                        "avg_cost":    qprice,
                        "strategy":    "core_etf",
                        "entry_date":  today,
                        "trail_stop":  None,
                        "entry_atr":   0.0,
                    }
                    trade_log.append({
                        "date": today, "code": QQQ_CODE, "side": "BUY",
                        "qty": qty, "price": qprice,
                        "pnl": None, "reason": "QQQ_BETA_FLOOR",
                        "strategy": "core_etf",
                    })

        # ── 4e. 记录每日净值 ─────────────────────────────────────────────────
        mark_to_market = sum(
            float(all_data[c].loc[ts, "close"]) * p["qty"]
            for c, p in positions.items()
            if ts in all_data[c].index
        )
        daily_equity = portfolio_cash + mark_to_market
        equity_curve.append(daily_equity)
        deployed_curve.append(mark_to_market)   # analytics-only
        date_list.append(ts)

    # ── Step 5: 统计 ─────────────────────────────────────────────────────────
    eq_series = pd.Series(equity_curve, index=date_list)
    ret_total = (eq_series.iloc[-1] - cash) / cash
    years = (eq_series.index[-1] - eq_series.index[0]).days / 365.25
    ann   = (1 + ret_total) ** (1 / years) - 1 if years > 0 else 0

    roll_max = eq_series.cummax()
    drawdown = (eq_series - roll_max) / roll_max
    max_dd   = float(drawdown.min())

    daily_ret = eq_series.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0

    sell_trades = [t for t in trade_log if t["side"] == "SELL"]
    n_trades = len(sell_trades)
    wins     = [t for t in sell_trades if (t["pnl"] or 0) > 0]
    win_rate = len(wins) / n_trades if n_trades else 0

    # Benchmark: buy-and-hold QQQ
    bm_ret = None
    if benchmark_df is not None:
        bm_start = benchmark_df.loc[(benchmark_df.index >= pd.Timestamp(start))
                                     & (benchmark_df.index <= pd.Timestamp(end)), "close"]
        if len(bm_start) > 1:
            bm_ret = (bm_start.iloc[-1] - bm_start.iloc[0]) / bm_start.iloc[0]

    result = {
        "start": start, "end": end, "stocks": len(stocks),
        "initial_cash": cash,
        "final_equity": round(float(eq_series.iloc[-1]), 2),
        "total_return_pct": round(ret_total * 100, 2),
        "annualized_return_pct": round(ann * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "num_trades": n_trades,
        "win_rate_pct": round(win_rate * 100, 1),
        "benchmark_qqq_pct": round(bm_ret * 100, 2) if bm_ret is not None else None,
        "equity_curve": eq_series,
        "deployed_curve": pd.Series(deployed_curve, index=date_list),   # analytics-only
        "capacity_blocks": capacity_blocks,   # analytics-only
        "replacements": replacements,   # v2.3 Portfolio Capacity Manager 主动置换记录
        "replacement_attempts": replacement_attempts,   # 2026-07-05 Explain Layer：非RSL分支下
                                                          # 每一次容量审查的完整决策记录（含被
                                                          # margin拦截的KEEP），仅在
                                                          # ENABLE_REPLACEMENT_STABILIZATION=False
                                                          # 时写入
        "shadow_replacements": shadow_replacements,   # v2.4 WEAK FULL shadow mode 观测记录
        "trade_log": trade_log,
        # Exposed whenever there's something to show — macro_block_days now
        # also counts QQQ technical halts (see 4c-macro above), which fire
        # regardless of whether a news_feed was supplied.
        "news_stats": news_stats if (news_feed is not None or news_stats["macro_block_days"] > 0) else None,
        "currency": currency,
    }

    _print_result(result)
    return result


def _print_result(r: dict) -> None:
    eq  = r["equity_curve"]
    tr  = r["total_return_pct"]
    ann = r["annualized_return_pct"]
    sh  = r["sharpe"]
    dd  = r["max_drawdown_pct"]
    bm  = r["benchmark_qqq_pct"]
    cur = r.get("currency", "$")

    print(f"\n{'='*70}")
    print(f"  回测结果汇总  {r['start']} ~ {r['end']}")
    print(f"{'='*70}")
    print(f"  初始资金:         {cur}{r['initial_cash']:>12,.2f}")
    print(f"  最终净值:         {cur}{r['final_equity']:>12,.2f}")
    print(f"{'─'*70}")
    print(f"  总收益率:         {tr:>+10.2f}%")
    print(f"  年化收益率:       {ann:>+10.2f}%")
    print(f"  Sharpe Ratio:     {sh:>10.3f}")
    print(f"  最大回撤:         {dd:>+10.2f}%")
    print(f"{'─'*70}")
    print(f"  总交易次数:       {r['num_trades']:>10}")
    print(f"  胜率:             {r['win_rate_pct']:>10.1f}%")
    if r.get("news_stats"):
        ns = r["news_stats"]
        print(f"{'─'*70}")
        print(f"  新闻Tier1否决次数: {ns['blocked']:>10}")
        print(f"  新闻Tier2降级次数: {ns['boosted']:>10}")
        print(f"  宏观熔断天数:     {ns['macro_block_days']:>10}")
    if bm is not None:
        alpha = tr - bm
        print(f"{'─'*70}")
        print(f"  基准 QQQ 收益:    {bm:>+10.2f}%  (同期买入持有)")
        print(f"  超额收益 Alpha:   {alpha:>+10.2f}%")
    print(f"{'='*70}")
    print("  * 过去表现不代表未来收益")
    print(f"{'='*70}\n")

    # 按年份分解收益
    try:
        yearly = eq.resample("YE").last().pct_change().dropna()
        eq_yearly = eq.resample("YE").last()
        first_val = eq.iloc[0]
        print("  各年度收益:")
        print(f"  {'年份':<6} {'区间收益':>10}  {'年末净值':>12}")
        print(f"  {'─'*32}")
        prev = first_val
        for d, v in eq_yearly.items():
            yr = d.year
            yr_ret = (v - prev) / prev * 100
            print(f"  {yr:<6} {yr_ret:>+10.2f}%  ${v:>12,.2f}")
            prev = v
        print()
    except Exception:
        pass

    # Top 10 交易
    sells = sorted([t for t in r["trade_log"] if t["side"] == "SELL" and t["pnl"]],
                   key=lambda x: x["pnl"], reverse=True)
    if sells:
        print("  最赚钱的前5笔交易:")
        for t in sells[:5]:
            entry_d = str(t.get("entry_date", "?"))[:10]
            print(f"    建仓{entry_d}  退出{str(t['date'])[:10]}  {t['code']:<8} {(t.get('strategy') or '?'):<14}  +${t['pnl']:,.0f}")
        print()
        print("  亏损最大的前5笔交易:")
        for t in sells[-5:]:
            entry_d = str(t.get("entry_date", "?"))[:10]
            print(f"    建仓{entry_d}  退出{str(t['date'])[:10]}  {t['code']:<8} {(t.get('strategy') or '?'):<14}   ${t['pnl']:,.0f}  [{t.get('reason','')}]")
        print()

    # ── 策略分布分析 ──────────────────────────────────────────────────────────
    strat_stats: dict = {}
    for t in r["trade_log"]:
        if t["side"] != "SELL" or t.get("pnl") is None:
            continue
        s = t.get("strategy") or "unknown"
        if s not in strat_stats:
            strat_stats[s] = {"n": 0, "wins": 0, "pnl": 0.0, "fees": 0.0}
        strat_stats[s]["n"]    += 1
        strat_stats[s]["pnl"]  += t["pnl"]
        strat_stats[s]["fees"] += t.get("fee", 0.0)
        if t["pnl"] > 0:
            strat_stats[s]["wins"] += 1

    if strat_stats:
        total_fees_all = sum(v["fees"] for v in strat_stats.values())
        print("  ── 策略分布分析 ─────────────────────────────────────────────")
        print(f"  {'策略':<14} {'笔数':>6} {'胜率':>8} {'净盈亏':>11} {'手续费(估)':>11} {'费用占比':>8}")
        print(f"  {'─'*62}")
        for s, st in sorted(strat_stats.items(), key=lambda x: -x[1]["n"]):
            wr      = st["wins"] / st["n"] * 100 if st["n"] else 0
            fee_pct = st["fees"] / total_fees_all * 100 if total_fees_all else 0
            print(f"  {s:<14} {st['n']:>6} {wr:>7.1f}% {st['pnl']:>+11,.0f} {st['fees']:>11,.0f} {fee_pct:>7.1f}%")
        print(f"  {'─'*62}")
        print(f"  {'合计':<14} {sum(v['n'] for v in strat_stats.values()):>6} {'':>8}"
              f" {sum(v['pnl'] for v in strat_stats.values()):>+11,.0f}"
              f" {total_fees_all:>11,.0f}")
        print()

    # ── 亏损 > $400 明细（财报闪崩复盘）────────────────────────────────────
    big_losers = [t for t in r["trade_log"]
                  if t["side"] == "SELL" and (t.get("pnl") or 0) < -400]
    big_losers.sort(key=lambda x: x["pnl"])
    if big_losers:
        print("  ── 亏损 > $400 交易明细（建仓~退出）────────────────────────")
        print(f"  {'建仓日':>12} {'退出日':>12} {'代码':<8} {'策略':<13} {'净盈亏':>8}  原因")
        print(f"  {'─'*62}")
        for t in big_losers:
            entry_d = str(t.get("entry_date", "?"))[:10]
            print(f"  {entry_d:>12} {str(t['date'])[:10]:>12} {t['code']:<8}"
                  f" {(t.get('strategy') or '?'):<13} {t['pnl']:>+8,.0f}  {t.get('reason','')}")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="多仓位组合回测（实盘级模拟，欧美+亚太合并结算）")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end",   default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--cash",  type=float, default=7_000_000.0,
                    help="账户本金，默认按日元本位币计（7,000,000 JPY ≈ 50,000 USD @140）")
    p.add_argument("--usd-jpy", type=float, default=140.0,
                    help="USD/JPY 固定换算汇率（静态，不反映历史汇率波动）。传0则不换算，"
                         "此时账户不能同时装美股和日股金额（单位不一致）")
    p.add_argument("--news-file", default=None,
                    help="历史新闻 CSV (date,code,title,source) — moomoo 不提供历史新闻，"
                         "需自行提供数据源，见 engine/news_backtest.py 文档")
    args = p.parse_args()

    feed = HistoricalNewsFeed.load(args.news_file) if args.news_file else None

    run_portfolio_backtest(
        stocks=BACKTEST_STOCKS,
        start=args.start,
        end=args.end,
        cash=args.cash,
        news_feed=feed,
        usd_to_jpy=(args.usd_jpy or None),
    )
