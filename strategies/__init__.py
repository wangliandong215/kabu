from .base import Strategy
from .ma_cross import MACrossStrategy
from .rsi import RSIStrategy
from .macd import MACDStrategy
from .boll import BollStrategy
from .combined import CombinedStrategy
from .ma_rsi import MARSIStrategy
from .atr_breakout import ATRBreakoutStrategy

REGISTRY: dict = {
    "ma":          MACrossStrategy,
    "rsi":         RSIStrategy,
    "macd":        MACDStrategy,
    "boll":        BollStrategy,
    "combined":    CombinedStrategy,
    "ma_rsi":      MARSIStrategy,
    "atr_breakout": ATRBreakoutStrategy,
    "atr_breakout_early": ATRBreakoutStrategy,  # TRENDING_EARLY route — same
                                                 # signal logic, sized down via
                                                 # config.STRATEGY_MAX_SIZE_PCT
}


def get_strategy(name: str, **kwargs) -> Strategy:
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Available: {sorted(REGISTRY)}")
    return cls(**kwargs)
