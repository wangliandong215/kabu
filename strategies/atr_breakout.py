"""
strategies/atr_breakout.py — ATR Donchian Channel Breakout with trailing stop.

Entry:  Close breaks above the 20-day highest high (Donchian upper channel)
Exit:   Price falls below trailing stop = highest_close_since_entry - 2 * ATR
        (trailing stop ratchets up as price rises, never moves down)

This strategy does NOT store position state between bars — it returns the
signal for the CURRENT bar only.  The trailing stop level is returned in
the result dict so the caller / portfolio tracker can maintain it.
"""
import pandas as pd
from .base import Strategy


class ATRBreakoutStrategy(Strategy):
    name = "atr_breakout"

    def __init__(self, channel_period: int = 20, atr_period: int = 14,
                 atr_multiplier: float = 2.0):
        self.channel_period  = channel_period
        self.atr_period      = atr_period
        self.atr_multiplier  = atr_multiplier
        self.required_bars   = max(channel_period, atr_period) + 5

    def compute(self, close: pd.Series) -> dict:
        raise NotImplementedError("ATRBreakoutStrategy needs OHLC — use compute_ohlc()")

    def compute_ohlc(self, df: pd.DataFrame) -> dict:
        """
        df must have columns: open, high, low, close (float).
        Returns signal dict including trailing_stop.
        """
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)

        # ── Donchian channel ──────────────────────────────────────────────────
        # Shift by 1: use yesterday's high/low to avoid look-ahead on current bar
        donchian_high = high.shift(1).rolling(self.channel_period).max()
        donchian_low  = low.shift(1).rolling(self.channel_period).min()

        # ── ATR (Average True Range) ──────────────────────────────────────────
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        price     = self._v(close)
        d_high    = self._v(donchian_high)
        d_low     = self._v(donchian_low)
        atr_val   = self._v(atr)

        # Trailing stop = highest close over recent channel_period - 2*ATR
        recent_high_close = close.rolling(self.channel_period).max()
        v_recent_high     = self._v(recent_high_close)

        trailing_stop = None
        if v_recent_high and atr_val:
            trailing_stop = v_recent_high - self.atr_multiplier * atr_val

        # ── Signal logic ──────────────────────────────────────────────────────
        if None in (price, d_high, d_low, atr_val):
            signal = "HOLD"
        elif price > d_high:
            # Breakout above Donchian upper channel → BUY
            signal = "BUY"
        elif trailing_stop and price < trailing_stop:
            # Price fell below trailing stop → SELL
            signal = "SELL"
        else:
            signal = "HOLD"

        detail = (
            f"Donchian[{self.channel_period}] high={d_high:.2f} low={d_low:.2f}  "
            f"ATR({self.atr_period})={atr_val:.2f}  "
            f"TrailingStop={trailing_stop:.2f}"
        ) if d_high and atr_val and trailing_stop else "insufficient data"

        return {
            "donchian_high":  round(d_high, 4)        if d_high        else None,
            "donchian_low":   round(d_low, 4)         if d_low         else None,
            "atr":            round(atr_val, 4)       if atr_val       else None,
            "trailing_stop":  round(trailing_stop, 4) if trailing_stop else None,
            "current_price":  round(price, 4)         if price         else None,
            "signal":         signal,
            "detail":         detail,
        }

    def full_result(self, df: pd.DataFrame) -> dict:
        """Override base full_result to use compute_ohlc."""
        result = self.compute_ohlc(df)
        sig    = result.get("signal", "HOLD")
        return {
            "signal":          sig,
            "signal_strength": 1.0 if sig != "HOLD" else 0.0,
            "buy_votes":       1 if sig == "BUY"  else 0,
            "sell_votes":      1 if sig == "SELL" else 0,
            "hold_votes":      1 if sig == "HOLD" else 0,
            **result,
        }
