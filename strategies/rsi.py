"""
strategies/rsi.py — Relative Strength Index mean-reversion.

Signal logic:
  BUY  when RSI < 30  (oversold)   — or when RSI rises back above 30
  SELL when RSI > 70  (overbought) — or when RSI falls back below 70
  HOLD otherwise
"""
import pandas as pd
from .base import Strategy


class RSIStrategy(Strategy):
    name = "rsi"
    required_bars = 30

    def __init__(self, period: int = 14):
        self.period = period
        self.required_bars = period + 15

    def compute(self, close: pd.Series) -> dict:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(self.period).mean()
        loss  = (-delta.clip(upper=0)).rolling(self.period).mean()
        rsi   = 100 - 100 / (1 + gain / loss)

        v      = self._v(rsi)
        v_prev = self._v(rsi, -2)

        if v is None:
            return {"value": None, "zone": "unknown", "signal": "HOLD", "detail": "RSI=N/A"}

        zone = "oversold" if v < 30 else ("overbought" if v > 70 else "neutral")
        sig  = "BUY"  if v < 30 else ("SELL" if v > 70 else "HOLD")

        # Exiting an extreme zone reinforces the signal
        if v_prev is not None:
            if v_prev < 30 <= v:  sig = "BUY"
            if v_prev > 70 >= v:  sig = "SELL"

        return {
            "period": self.period,
            "value":  round(v, 2),
            "zone":   zone,
            "signal": sig,
            "detail": f"RSI({self.period})={v:.1f} [{zone}]",
        }
