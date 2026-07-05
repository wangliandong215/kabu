"""
strategies/base.py — abstract Strategy interface.
All concrete strategies must inherit from Strategy and implement compute().
"""
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class Strategy(ABC):
    name: str = "base"
    required_bars: int = 60   # minimum bars needed for reliable signals

    @abstractmethod
    def compute(self, close: pd.Series) -> dict:
        """
        Compute indicators from the close price series.
        Must return a dict that always contains:
          signal (str):  "BUY" | "SELL" | "HOLD"
          detail (str):  human-readable summary of indicator values
        """

    def signal(self, df: pd.DataFrame) -> str:
        """Convenience: return just the signal string."""
        return self.full_result(df).get("signal", "HOLD")

    def full_result(self, df: pd.DataFrame) -> dict:
        """Return the full indicator dict for df."""
        close = df["close"].astype(float) if "close" in df.columns else df.astype(float)
        return self.compute(close)

    # ── Shared helper methods ─────────────────────────────────────────────────

    @staticmethod
    def _v(series: pd.Series, idx: int = -1) -> Optional[float]:
        """Safely read series[idx] as float; return None if NaN or out of range."""
        if len(series) == 0 or abs(idx) > len(series):
            return None
        val = series.iloc[idx]
        return None if pd.isna(val) else float(val)

    @staticmethod
    def _crossover(fast: pd.Series, slow: pd.Series) -> str:
        """
        Detect whether fast crossed above (GOLDEN_CROSS) or below (DEATH_CROSS)
        slow within the most recent 3 bars.  Returns 'NONE' otherwise.
        """
        for lag in range(1, 4):
            f0 = Strategy._v(fast, -lag);     f1 = Strategy._v(fast, -lag - 1)
            s0 = Strategy._v(slow, -lag);     s1 = Strategy._v(slow, -lag - 1)
            if None in (f0, f1, s0, s1):
                continue
            if f1 <= s1 and f0 > s0:
                return "GOLDEN_CROSS"
            if f1 >= s1 and f0 < s0:
                return "DEATH_CROSS"
        return "NONE"
