"""
engine/indicators.py — shared vectorised technical indicators.

Used by both live sizing (risk/sizing.py via engine/scanner.py) and
backtest_portfolio.py, so momentum-based decisions never drift between the
two paths (same failure mode fixed for regime detection and boll signals —
see engine/regime.py and strategies/boll.py).
"""
import pandas as pd


def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI (EWM approximation, span=period)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, float("nan")))


def rsi_last(close: pd.Series, period: int = 14):
    """Last RSI value as float, or None if unavailable."""
    series = rsi_series(close, period)
    if len(series) == 0:
        return None
    val = series.iloc[-1]
    return None if pd.isna(val) else float(val)
