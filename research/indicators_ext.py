"""
research/indicators_ext.py — technical indicators used by the Mean Reversion
research layer that don't already exist elsewhere in the codebase (RSI lives
in engine/indicators.py, ATR/volume-ratio in engine/regime_features.py — both
reused as-is, not duplicated here).

All functions are vectorised (Series in, Series out) and causal — every
point only depends on bars up to and including it. Pure/no side effects, same
discipline as engine/regime_features.py. Observation-only: nothing here is
imported by engine/runner.py or any decision-logic module.
"""
import pandas as pd


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Classic (slow) Stochastic Oscillator. Returns a DataFrame with
    columns %k, %d — %k = 100*(close - lowest_low)/(highest_high - lowest_low)
    over k_period, %d = simple moving average of %k over d_period."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    span = (highest_high - lowest_low).replace(0, float("nan"))
    k = 100 * (close - lowest_low) / span
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    """Williams %R, range [-100, 0]. -100 = at the period low, 0 = at the
    period high (i.e. deeply negative = oversold)."""
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    span = (highest_high - lowest_low).replace(0, float("nan"))
    return -100 * (highest_high - close) / span


def low_to_close_position(high: pd.Series, low: pd.Series,
                           close: pd.Series) -> pd.Series:
    """Where the close sits within the day's [low, high] range, 0-1 scaled
    (0 = closed at the low, 1 = closed at the high)."""
    span = (high - low).replace(0, float("nan"))
    return (close - low) / span


def close_location_value(open_: pd.Series, high: pd.Series, low: pd.Series,
                          close: pd.Series) -> pd.Series:
    """Standard Close Location Value: ((close-low) - (high-close)) /
    (high-low), range [-1, 1]. +1 = closed at the high, -1 = closed at the
    low. (Distinct from low_to_close_position's 0-1 scale — both requested
    separately in the spec.)"""
    span = (high - low).replace(0, float("nan"))
    return ((close - low) - (high - close)) / span


def lower_wick_pct(open_: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series) -> pd.Series:
    """Lower wick as a fraction of the day's [low, high] range."""
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    span = (high - low).replace(0, float("nan"))
    return (body_low - low) / span


def upper_wick_pct(open_: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series) -> pd.Series:
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    span = (high - low).replace(0, float("nan"))
    return (high - body_high) / span


def body_pct(open_: pd.Series, high: pd.Series, low: pd.Series,
             close: pd.Series) -> pd.Series:
    """Candle body as a fraction of the day's [low, high] range."""
    span = (high - low).replace(0, float("nan"))
    return (close - open_).abs() / span


_HAMMER_MAX_BODY_PCT = 0.35     # body must be small relative to the range
_HAMMER_MIN_LOWER_WICK_PCT = 0.45   # long lower wick (bought up from a low)
_HAMMER_MAX_UPPER_WICK_PCT = 0.15   # little/no upper wick


def is_hammer(open_: pd.Series, high: pd.Series, low: pd.Series,
              close: pd.Series) -> pd.Series:
    """Bool Series: classic hammer/reversal candle shape (small body near the
    top of the range, long lower wick, little upper wick) — a heuristic
    shape check, not a claim about what happens next (see
    research/forward_outcomes.py for the actual forward-looking
    confirmation fields)."""
    bp = body_pct(open_, high, low, close)
    lw = lower_wick_pct(open_, high, low, close)
    uw = upper_wick_pct(open_, high, low, close)
    return (bp <= _HAMMER_MAX_BODY_PCT) & (lw >= _HAMMER_MIN_LOWER_WICK_PCT) \
        & (uw <= _HAMMER_MAX_UPPER_WICK_PCT)


def recovered_prior_low(low: pd.Series, close: pd.Series) -> pd.Series:
    """Bool Series: did today's close recover back above yesterday's low
    (i.e. the selloff didn't follow through intraday)."""
    return close > low.shift(1)


def last_value(series: pd.Series):
    """Last value as a plain float, or None if empty/NaN — convenience for
    building snapshot dicts without repeating the same guard everywhere."""
    if series is None or len(series) == 0:
        return None
    val = series.iloc[-1]
    return None if pd.isna(val) else float(val)


def last_bool(series: pd.Series):
    if series is None or len(series) == 0:
        return None
    val = series.iloc[-1]
    return None if pd.isna(val) else bool(val)
