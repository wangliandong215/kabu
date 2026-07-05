"""
strategies/combined.py — majority-vote across MA / RSI / MACD / Bollinger.

Signal logic:
  Tallies BUY (+1) / HOLD (0) / SELL (-1) from each sub-strategy.
  Overall signal follows the majority; strength = winning_votes / total_votes.

The list of sub-strategies can be customised at construction time, making it
easy to add new strategies or tune weights in the future.
"""
from typing import List

import pandas as pd

from .base import Strategy
from .ma_cross import MACrossStrategy
from .rsi import RSIStrategy
from .macd import MACDStrategy
from .boll import BollStrategy


class CombinedStrategy(Strategy):
    name = "combined"

    def __init__(self, strategies: List[Strategy] = None):
        if strategies is None:
            strategies = [
                MACrossStrategy(),
                RSIStrategy(),
                MACDStrategy(),
                BollStrategy(),
            ]
        self.strategies  = strategies
        self.required_bars = max(s.required_bars for s in strategies)

    def compute(self, close: pd.Series) -> dict:
        sub: dict = {}
        for s in self.strategies:
            sub[s.name] = s.compute(close)

        score_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
        votes = [score_map.get(r.get("signal", "HOLD"), 0) for r in sub.values()]
        n      = len(votes)
        buy_n  = sum(1 for v in votes if v > 0)
        sell_n = sum(1 for v in votes if v < 0)
        hold_n = n - buy_n - sell_n
        score  = sum(votes) / n if n else 0

        if score > 0:
            overall, strength = "BUY",  buy_n  / n
        elif score < 0:
            overall, strength = "SELL", sell_n / n
        else:
            overall, strength = "HOLD", 0.0

        return {
            "signal":          overall,
            "signal_strength": round(strength, 2),
            "buy_votes":       buy_n,
            "sell_votes":      sell_n,
            "hold_votes":      hold_n,
            "detail":          f"BUY={buy_n} SELL={sell_n} HOLD={hold_n} → {overall}",
            "sub_indicators":  sub,
        }
