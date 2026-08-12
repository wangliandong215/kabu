"""
engine/entry_quality.py — Entry Quality Tracking (v2.9.x, observation-only).

Motivation: ATR + Donchian Channel Breakout (strategies/atr_breakout.py)
entries are sometimes filled well above the actual breakout level (recent
MSFT/PLTR cases) — "chasing the breakout". This module computes descriptive
snapshot metrics about *how extended* an entry was, and (separately, later)
how price actually behaved afterward, so that data can be analyzed offline
(see analyze_entry_quality.py at repo root) before any Confidence Score /
Dynamic Position Sizing / entry filter is designed.

Hard constraint (per product decision): nothing in this module may feed back
into a trading decision. Every function here is read-only / pure — no
function takes `portfolio` or places an order, and every function is safe to
call from engine/runner.py's BUY hook wrapped in try/except (never raises;
degrades to None fields on missing/insufficient data instead).

Two independent computations:
  build_entry_quality_snapshot() — everything computable AT entry time from
      already-known values (entry_price, donchian_breakout_price, ATR) plus
      a plain OHLCV history up to and including the entry day. Called once,
      synchronously, from engine/runner.py right after a new position opens.
  compute_forward_returns()      — everything that requires bars AFTER entry
      (5/10/20/60-day forward return, forward MFE/MAE). Can only be computed
      once enough calendar time has passed, so this is called later by
      analyze_entry_quality.py's backfill step, not from the live runner.
"""
from typing import Optional

import pandas as pd

from engine.regime_features import atr_series, volume_ratio_series

_LOOKBACK = 20                  # matches strategies/atr_breakout.py's default channel_period
_ATR_EXPANSION_LOOKBACK = 60    # "recent ATR" vs. this long a trailing average


def build_entry_quality_snapshot(
    df: Optional[pd.DataFrame],
    entry_price: Optional[float],
    donchian_breakout_price: Optional[float] = None,
    atr_at_entry: Optional[float] = None,
    rule_score: Optional[float] = None,
    confidence_score: Optional[float] = None,
) -> Optional[dict]:
    """Compute the at-entry Entry Quality fields.

    df : OHLCV history up to and including the entry day (high/low/close
        required, volume optional). The same shape fetch_kline() returns.
        Should NOT include an unconfirmed/live intraday bar — pass the same
        confirmed-bars df the strategy itself signalled off of, if available,
        otherwise a plain fetch_kline() call is an acceptable approximation
        (this is analytics, not the signal itself).
    entry_price : the actual fill price (not the signal-time price).
    donchian_breakout_price : the Donchian upper-channel value the entry
        strategy computed (strategies/atr_breakout.py's `donchian_high`) —
        reused as-is, not recomputed here, so this module never has two
        implementations of the same breakout level. None for entries opened
        by a non-Donchian strategy (e.g. plain "combined"/auto_route routing
        into a different strategy) — every derived distance field involving
        it is then also None, not fabricated.
    atr_at_entry : reused from the strategy result's `atr` field (same value
        already stored on trades.atr_entry) — not recomputed here either.
    rule_score / confidence_score : passed straight through from the caller
        (pipeline.Candidate.total_score / .confidence) into the return dict
        for convenience — this function does not need them for anything it
        computes, they just ride along so the caller only needs one dict.

    Never raises. Returns None only when there isn't enough to record at all
    (df missing/empty or entry_price missing) — otherwise returns a dict with
    whatever subset of fields could be computed, remaining keys None.
    """
    try:
        if df is None or len(df) == 0 or not entry_price:
            return None

        high = df["high"].astype(float)
        close = df["close"].astype(float)

        high_20d = float(high.tail(_LOOKBACK).max()) if len(high) >= _LOOKBACK else None
        ema_20d = (float(close.ewm(span=_LOOKBACK, adjust=False).mean().iloc[-1])
                   if len(close) >= _LOOKBACK else None)
        prev_close = float(close.iloc[-1]) if len(close) else None

        distance_from_breakout_atr = None
        if donchian_breakout_price and atr_at_entry:
            distance_from_breakout_atr = (
                (entry_price - donchian_breakout_price) / atr_at_entry)

        distance_from_20d_high = None
        if high_20d:
            distance_from_20d_high = (entry_price - high_20d) / high_20d

        distance_from_20d_ema = None
        if ema_20d:
            distance_from_20d_ema = (entry_price - ema_20d) / ema_20d

        entry_day_return = None
        if prev_close:
            entry_day_return = (entry_price - prev_close) / prev_close

        entry_volume_ratio = None
        if "volume" in df.columns and len(df) >= _LOOKBACK:
            vr = volume_ratio_series(df["volume"].astype(float))
            last = vr.iloc[-1]
            if pd.notna(last):
                entry_volume_ratio = float(last)

        atr_expansion_ratio = None
        if atr_at_entry and len(df) >= _ATR_EXPANSION_LOOKBACK and {"high", "low"} <= set(df.columns):
            atr_hist = atr_series(df["high"].astype(float), df["low"].astype(float), close)
            baseline = float(atr_hist.tail(_ATR_EXPANSION_LOOKBACK).mean())
            if baseline:
                atr_expansion_ratio = atr_at_entry / baseline

        return {
            "donchian_breakout_price": donchian_breakout_price,
            "distance_from_breakout_atr": distance_from_breakout_atr,
            "distance_from_20d_high": distance_from_20d_high,
            "distance_from_20d_ema": distance_from_20d_ema,
            "entry_day_return": entry_day_return,
            "entry_volume_ratio": entry_volume_ratio,
            "atr_expansion_ratio": atr_expansion_ratio,
            "rule_score": rule_score,
            "confidence_score": confidence_score,
        }
    except Exception:
        return None


def compute_forward_returns(df_after_entry: Optional[pd.DataFrame],
                             entry_price: Optional[float]) -> Optional[dict]:
    """Compute forward-looking outcome fields for one trade, given a plain
    OHLCV df covering trading days STRICTLY AFTER entry (row 0 = the first
    post-entry bar; the entry-day bar itself must NOT be included, so a
    same-day round-trip doesn't count as its own "return").

    Returns return_5d/10d/20d/60d (None for a horizon not yet reached — a
    trade opened 8 days ago gets return_5d filled and return_10d/20d/60d
    left None, NOT a stale/partial number), max_favorable_excursion /
    max_adverse_excursion (running high/low over whatever bars ARE
    available, capped at 60), and bars_available (so callers/analysis can
    tell "not enough time has passed yet" apart from "no data at all").

    Never raises. Returns None only when there is truly nothing to compute
    from (df missing/empty or entry_price missing).
    """
    try:
        if df_after_entry is None or len(df_after_entry) == 0 or not entry_price:
            return None

        close = df_after_entry["close"].astype(float)
        high = df_after_entry["high"].astype(float) if "high" in df_after_entry.columns else close
        low = df_after_entry["low"].astype(float) if "low" in df_after_entry.columns else close

        n = len(close)
        window = min(n, 60)
        close_w = close.iloc[:window]
        high_w = high.iloc[:window]
        low_w = low.iloc[:window]

        def _return_at(horizon: int) -> Optional[float]:
            if n < horizon:
                return None
            return float(close.iloc[horizon - 1] / entry_price - 1.0)

        return {
            "return_5d": _return_at(5),
            "return_10d": _return_at(10),
            "return_20d": _return_at(20),
            "return_60d": _return_at(60),
            "max_favorable_excursion": float(high_w.max() / entry_price - 1.0),
            "max_adverse_excursion": float(low_w.min() / entry_price - 1.0),
            "forward_bars_available": window,
        }
    except Exception:
        return None
