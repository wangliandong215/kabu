"""
research/forward_outcomes.py — Category 9 (Forward Return) + forward candle
confirmation fields, and the SUCCESS/FAILURE/NEUTRAL/PENDING outcome label
used to distinguish case groups C (failed) / D (succeeded) per the spec.

Same pattern as engine/entry_quality.py's compute_forward_returns() (T+5/10/
20/60 + MFE/MAE), extended with T+1/T+3 and the forward-looking candle
confirmation booleans the spec also asks for (does it keep making new lows,
does a higher low form). Computed AFTER the fact, once enough calendar time
has passed — never influences detection/trigger logic.
"""
from typing import Optional

import pandas as pd

import config

_HORIZONS = (1, 3, 5, 10, 20, 60)


def compute_forward_outcomes(df_after_entry: Optional[pd.DataFrame],
                              entry_price: Optional[float],
                              event_day_low: Optional[float] = None) -> Optional[dict]:
    """df_after_entry: plain OHLCV covering trading days STRICTLY AFTER the
    event day (row 0 = first post-event bar; the event day itself must NOT
    be included). event_day_low: the event day's own low, needed to judge
    "did price make a new low after the event" — passed separately since
    it isn't part of df_after_entry.

    Never raises. Returns None only when there's truly nothing to compute
    from (df missing/empty or entry_price missing)."""
    try:
        if df_after_entry is None or len(df_after_entry) == 0 or not entry_price:
            return None

        close = df_after_entry["close"].astype(float)
        high = df_after_entry["high"].astype(float) if "high" in df_after_entry.columns else close
        low = df_after_entry["low"].astype(float) if "low" in df_after_entry.columns else close

        n = len(close)
        window = min(n, 60)
        high_w = high.iloc[:window]
        low_w = low.iloc[:window]

        def _return_at(horizon: int) -> Optional[float]:
            if n < horizon:
                return None
            return float(close.iloc[horizon - 1] / entry_price - 1.0)

        result = {f"return_fwd_t{h}": _return_at(h) for h in _HORIZONS}
        result["max_favorable_excursion"] = float(high_w.max() / entry_price - 1.0)
        result["max_adverse_excursion"] = float(low_w.min() / entry_price - 1.0)
        result["forward_bars_available"] = window

        made_new_low_t1 = made_new_low_t2 = higher_low_within_3d = None
        if event_day_low:
            if n >= 1:
                made_new_low_t1 = bool(float(low.iloc[0]) < event_day_low)
            if n >= 2:
                made_new_low_t2 = bool(float(low.iloc[1]) < event_day_low)
            lows = [event_day_low] + [float(v) for v in low.iloc[:3].tolist()]
            higher_low_within_3d = any(lows[i] > lows[i - 1] for i in range(1, len(lows)))
        result["made_new_low_t1"] = made_new_low_t1
        result["made_new_low_t2"] = made_new_low_t2
        result["higher_low_within_3d"] = higher_low_within_3d

        return result
    except Exception:
        return None


def derive_outcome_label(forward: Optional[dict]) -> str:
    """SUCCESS / FAILURE / NEUTRAL once T+10 is reachable, else PENDING.
    Purely descriptive — computed after the fact, never influences
    detection/trigger logic. Never raises."""
    if not forward:
        return "PENDING"
    if (forward.get("forward_bars_available") or 0) < 10:
        return "PENDING"
    r10 = forward.get("return_fwd_t10")
    if r10 is None:
        return "PENDING"
    if r10 >= config.MR_OUTCOME_SUCCESS_RETURN_10D:
        return "SUCCESS"
    if r10 <= config.MR_OUTCOME_FAILURE_RETURN_10D:
        return "FAILURE"
    return "NEUTRAL"
