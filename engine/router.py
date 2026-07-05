"""
engine/router.py — Dynamic Strategy Routing based on market regime.

Maps each detected regime to the most appropriate strategy name.
The mapping is defined in config.REGIME_STRATEGY_MAP so users can
override it without touching code.

Routing logic:
  TRENDING_UP   → atr_breakout  ride the breakout momentum
  TRENDING_DOWN → (skip)        no long entries into a downtrend
  RANGING       → boll          mean reversion between bands, 60%+ win rate
  VOLATILE      → (skip)        high-churn noise, no edge — save the commissions
"""
import config
from engine.regime import detect, regime_label
import notify.alert as alert


def route(df, code: str = "") -> str | None:
    """
    Detect the market regime for df and return the strategy name to use.
    Returns None if the regime calls for skipping this stock entirely.

    Parameters
    ----------
    df   : OHLC DataFrame
    code : stock code string (for logging only)
    """
    regime = detect(df)
    strategy_name = config.REGIME_STRATEGY_MAP.get(regime)

    label = regime_label(regime)
    if strategy_name:
        alert.info(f"router: {code:20s}  regime={label:12s}  -> {strategy_name}")
    else:
        alert.info(f"router: {code:20s}  regime={label:12s}  -> SKIP")

    return strategy_name
