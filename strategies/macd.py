"""
strategies/macd.py — MACD histogram signal.

Signal logic:
  BUY  when histogram > 0  (MACD above signal line)
  SELL when histogram < 0
  Also reports BULLISH / BEARISH crossover if histogram just flipped sign.
"""
import pandas as pd
from .base import Strategy


class MACDStrategy(Strategy):
    name = "macd"
    required_bars = 40

    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9):
        self.fast   = fast
        self.slow   = slow
        self.sig_p  = signal_period
        self.required_bars = slow + signal_period + 5

    def compute(self, close: pd.Series) -> dict:
        ema_f    = close.ewm(span=self.fast, adjust=False).mean()
        ema_s    = close.ewm(span=self.slow, adjust=False).mean()
        macd     = ema_f - ema_s
        sig_line = macd.ewm(span=self.sig_p, adjust=False).mean()
        hist     = macd - sig_line

        vm, vs, vh = self._v(macd), self._v(sig_line), self._v(hist)
        vh_prev    = self._v(hist, -2)

        cross = "NONE"
        if vh_prev is not None and vh is not None:
            if vh_prev <= 0 < vh:  cross = "BULLISH"
            if vh_prev >= 0 > vh:  cross = "BEARISH"

        sig = "HOLD" if vh is None else ("BUY" if vh > 0 else ("SELL" if vh < 0 else "HOLD"))

        detail = (f"MACD={vm:.4f}, Signal={vs:.4f}, Hist={vh:.4f}" if vm else "MACD=N/A")
        if cross != "NONE":
            detail += f" ({cross})"

        return {
            "macd":        round(vm, 4) if vm else None,
            "signal_line": round(vs, 4) if vs else None,
            "histogram":   round(vh, 4) if vh else None,
            "crossover":   cross,
            "signal":      sig,
            "detail":      detail,
        }
