"""
strategies/ma_rsi.py — EMA crossover + RSI trend filter.

Uses EMA (not SMA) for faster response to price changes.

Entry:  EMA fast crosses above EMA slow (golden cross) AND RSI > 50
        (trend confirmed, not overbought)
Exit:   EMA fast crosses below EMA slow (death cross) OR RSI < 40
        (momentum fading — exit proactively)
"""
import pandas as pd
from strategies.base import Strategy


class MARSIStrategy(Strategy):
    name = "ma_rsi"

    def __init__(self, fast: int = 5, slow: int = 20,
                 rsi_period: int = 14,
                 rsi_buy_min: float = 50.0,   # RSI must be above this to buy
                 rsi_sell: float = 40.0):      # RSI below this → exit
        self.fast        = fast
        self.slow        = slow
        self.rsi_period  = rsi_period
        self.rsi_buy_min = rsi_buy_min
        self.rsi_sell    = rsi_sell
        self.required_bars = slow + rsi_period + 5

    def compute(self, close: pd.Series) -> dict:
        # EMA instead of SMA
        emaf = close.ewm(span=self.fast,  adjust=False).mean()
        emas = close.ewm(span=self.slow, adjust=False).mean()

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss  = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rsi   = 100 - 100 / (1 + gain / loss)

        vf   = self._v(emaf)
        vs   = self._v(emas)
        vrsi = self._v(rsi)
        cross = self._crossover(emaf, emas)

        if vf is None or vs is None or vrsi is None:
            signal = "HOLD"
        elif (cross == "GOLDEN_CROSS" or vf > vs) and vrsi > self.rsi_buy_min:
            # Uptrend confirmed by RSI momentum
            signal = "BUY"
        elif cross == "DEATH_CROSS" or vrsi < self.rsi_sell:
            # Trend reversing or momentum fading — exit
            signal = "SELL"
        else:
            signal = "HOLD"

        detail = (
            f"EMA{self.fast}={vf:.2f}, EMA{self.slow}={vs:.2f}, "
            f"RSI({self.rsi_period})={vrsi:.1f}"
            + (f" ({cross})" if cross != "NONE" else "")
        ) if vf and vs and vrsi else "insufficient data"

        return {
            f"ema{self.fast}": round(vf, 4)   if vf   else None,
            f"ema{self.slow}": round(vs, 4)   if vs   else None,
            "rsi":             round(vrsi, 2) if vrsi else None,
            "crossover":       cross,
            "signal":          signal,
            "detail":          detail,
        }
