"""
engine/regime_features.py — independent Feature Extraction module for
Market Regime Detection (v2.5, see engine/market_regime.py).

Deliberately self-contained: does NOT import engine/regime.py. That module's
_adx/_atr helpers are already tuned and locked into the v1.0~v2.4 production
backtest baselines (see project memory) — duplicating the standard formulas
here instead of importing keeps this purely-descriptive module from ever
accidentally drifting the trading-decision code, and vice versa.

All functions are vectorised (Series/DataFrame in, Series/DataFrame out) and
causal — every point only depends on bars up to and including it, so this is
safe to use inside a backtest without look-ahead bias.
"""
import pandas as pd

import config


def atr_series(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = None) -> pd.Series:
    """Wilder-smoothed Average True Range."""
    period = period or config.MRD_ATR_PERIOD
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def atr_pct_series(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = None) -> pd.Series:
    """ATR expressed as a fraction of price (ATR / close) — a cross-stock
    comparable volatility measure (unlike raw ATR, which scales with price)."""
    atr = atr_series(high, low, close, period)
    return atr / close


def adx_series(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = None) -> pd.Series:
    """Wilder's Average Directional Index (trend-strength, direction-agnostic)."""
    period = period or config.MRD_ADX_PERIOD

    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)
    plus_dm[(up > down) & (up > 0)] = up
    minus_dm[(down > up) & (down > 0)] = down

    atr = atr_series(high, low, close, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def bb_width_series(close: pd.Series, period: int = None,
                     num_std: float = None) -> pd.Series:
    """Bollinger Band width, normalised by the mid band: (upper-lower)/mid.
    Low values -> compression (squeeze), high values -> expansion."""
    period = period or config.MRD_BB_PERIOD
    num_std = num_std if num_std is not None else config.MRD_BB_STD

    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return (upper - lower) / mid


def volume_ratio_series(volume: pd.Series, ma_period: int = None) -> pd.Series:
    """Current volume relative to its trailing moving average — used to
    corroborate whether a volatility/trend reading is backed by real
    participation or not."""
    ma_period = ma_period or config.MRD_VOLUME_MA_PERIOD
    vol_ma = volume.rolling(ma_period).mean()
    return volume / vol_ma


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bundle all five features for the whole df into one DataFrame with columns
    adx, atr, atr_pct, bb_width, volume_ratio.

    df must have columns: high, low, close (float). volume (float) is
    optional — without it, volume_ratio is all-NaN.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    out = pd.DataFrame(index=df.index)
    out["adx"] = adx_series(high, low, close)
    out["atr"] = atr_series(high, low, close)
    out["atr_pct"] = atr_pct_series(high, low, close)
    out["bb_width"] = bb_width_series(close)

    if "volume" in df.columns:
        out["volume_ratio"] = volume_ratio_series(df["volume"].astype(float))
    else:
        out["volume_ratio"] = float("nan")

    return out
