"""
strategies/ma_cross.py — Moving Average crossover.

Signal logic:
  BUY  when fast MA is above slow MA (uptrend)
  SELL when fast MA is below slow MA (downtrend)
  Also reports GOLDEN_CROSS / DEATH_CROSS if a crossover occurred in last 3 bars.
"""
import pandas as pd
from .base import Strategy


class MACrossStrategy(Strategy):
    name = "ma"
    required_bars = 65   # needs 60 bars for MA60 to warm up

    def __init__(self, fast: int = 5, slow: int = 20):
        self.fast = fast
        self.slow = slow

    def compute(self, close: pd.Series) -> dict:
        maf  = close.rolling(self.fast).mean()
        mas  = close.rolling(self.slow).mean()
        ma60 = close.rolling(60).mean()

        vf, vs, v60 = self._v(maf), self._v(mas), self._v(ma60)
        cross = self._crossover(maf, mas)

        if vf is not None and vs is not None:
            sig = "BUY" if vf > vs else ("SELL" if vf < vs else "HOLD")
        else:
            sig = "HOLD"

        parts = []
        if vf:   parts.append(f"MA{self.fast}={vf:.2f}")
        if vs:   parts.append(f"MA{self.slow}={vs:.2f}")
        if v60:  parts.append(f"MA60={v60:.2f}")
        if cross != "NONE":
            parts.append(f"({cross})")

        return {
            f"ma{self.fast}": round(vf, 4) if vf else None,
            f"ma{self.slow}": round(vs, 4) if vs else None,
            "ma60":           round(v60, 4) if v60 else None,
            "crossover":      cross,
            "signal":         sig,
            "detail":         ", ".join(parts),
        }
