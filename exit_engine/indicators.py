# -*- coding: utf-8 -*-
"""
exit_engine/indicators.py — shared vectorised helpers for V3.2-A signal
modules (VWAP/EMA/ATR/trend/volume). Kept self-contained to exit_engine/
rather than added to engine/indicators.py (which today only holds RSI) to
keep this feature low-blast-radius; can be promoted later if other
features want them.

Every function is defensive about short/missing/malformed input (returns
None rather than raising) since intraday bars can legitimately be thin or
absent — pre-market, an illiquid symbol, a moomoo hiccup, or simply not
enough history yet for the requested window.
"""
from typing import Optional

import pandas as pd


def compute_vwap_series(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """Cumulative (session) VWAP at each bar: cumsum(typical_price*volume)
    / cumsum(volume). Callers pass only the bars belonging to ONE session
    (see exit_engine/__init__.py::_session_bars) — VWAP is a session-reset
    statistic, not a continuous one like EMA."""
    if df is None or df.empty or not {"high", "low", "close", "volume"}.issubset(df.columns):
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    typical = (high + low + close) / 3.0
    cum_vol = vol.cumsum()
    cum_pv = (typical * vol).cumsum()
    vwap = cum_pv / cum_vol.replace(0, pd.NA)
    return vwap


def compute_vwap(df: Optional[pd.DataFrame]) -> Optional[float]:
    """Latest session VWAP value, or None if it can't be computed."""
    series = compute_vwap_series(df)
    if series is None:
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def ema_series(series: Optional[pd.Series], span: int) -> Optional[pd.Series]:
    if series is None or span <= 0 or len(series) < span:
        return None
    return series.astype(float).ewm(span=span, adjust=False).mean()


def compute_ema(series: Optional[pd.Series], span: int) -> Optional[float]:
    result = ema_series(series, span)
    if result is None or result.empty:
        return None
    return float(result.iloc[-1])


def atr_from_df(df: Optional[pd.DataFrame], period: int = 14) -> Optional[float]:
    """Wilder-style ATR (EWM of True Range) — mirrors engine/regime.py's
    own private _atr() shape without importing it (that one is scoped to
    daily-bar regime detection, this needs to run on intraday bars too)."""
    if df is None or len(df) < period + 1 or not {"high", "low", "close"}.issubset(df.columns):
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else None


def trend_direction(df: Optional[pd.DataFrame], fast: int, slow: int) -> Optional[str]:
    """"UP"/"DOWN"/"FLAT" from fast-vs-slow EMA ordering + fast-EMA slope on
    `close`. None if there isn't enough data for the slow span yet."""
    if df is None or "close" not in df or len(df) < slow + 2:
        return None
    close = df["close"].astype(float)
    fast_series = ema_series(close, fast)
    slow_series = ema_series(close, slow)
    if fast_series is None or slow_series is None or len(fast_series) < 2:
        return None
    fast_now, fast_prev = fast_series.iloc[-1], fast_series.iloc[-2]
    slow_now = slow_series.iloc[-1]
    if pd.isna(fast_now) or pd.isna(fast_prev) or pd.isna(slow_now):
        return None
    if fast_now > slow_now and fast_now >= fast_prev:
        return "UP"
    if fast_now < slow_now and fast_now <= fast_prev:
        return "DOWN"
    return "FLAT"


def avg_volume(df: Optional[pd.DataFrame], lookback: int = 20,
                exclude_last: bool = True) -> Optional[float]:
    """Mean volume over the trailing `lookback` bars, excluding the most
    recent bar by default (so a caller can compare "this bar's volume" vs
    "the recent average, not counting this bar itself")."""
    if df is None or "volume" not in df or df.empty:
        return None
    vol = df["volume"].astype(float)
    window = vol.iloc[:-1] if exclude_last and len(vol) > 1 else vol
    window = window.tail(lookback)
    if window.empty:
        return None
    mean = window.mean()
    return float(mean) if pd.notna(mean) and mean > 0 else None
