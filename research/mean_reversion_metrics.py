"""
research/mean_reversion_metrics.py — core Mean Reversion observation-layer
metrics (categories 1-4, 6, 8 of the project spec; category 5/HMM is passed
in by the caller from engine/regime_store.py, category 7/fundamentals comes
from research/fundamentals.py, category 9/forward-returns from
research/forward_outcomes.py — kept separate since they have different data
requirements/timing).

Same discipline as engine/entry_quality.py: every function here is pure and
read-only, never raises (degrades to None fields on missing/insufficient
data), and nothing computed here feeds back into a trading decision.
evaluate_trigger() decides only whether a row gets RECORDED — it is never
checked by strategies/*, risk/*, or engine/runner.py.
"""
from typing import Optional

import pandas as pd

import config
from engine.regime_features import atr_series, volume_ratio_series
from engine.indicators import rsi_last
from research.indicators_ext import (
    stochastic, williams_r, low_to_close_position, close_location_value,
    lower_wick_pct, body_pct, is_hammer, last_value, last_bool,
)
from research.sector_etf import sector_etf_for

_VOLUME_SPIKE_RATIO = 1.5


def _return_n(close: pd.Series, n: int) -> Optional[float]:
    if len(close) <= n:
        return None
    prev = close.iloc[-1 - n]
    if not prev:
        return None
    return float(close.iloc[-1] / prev - 1.0)


def evaluate_trigger(df: Optional[pd.DataFrame]) -> Optional[str]:
    """Recording filter only (see module docstring) — returns a
    human-readable reason string if ANY configured extreme-oversold
    threshold fires on the LAST bar of `df`, else None. Never raises."""
    try:
        if df is None or len(df) < 6:
            return None
        close = df["close"].astype(float)
        reasons = []

        r3 = _return_n(close, 3)
        if r3 is not None and r3 <= config.MR_TRIGGER_RETURN_3D:
            reasons.append(f"3D_RETURN<={config.MR_TRIGGER_RETURN_3D:.0%}")

        r5 = _return_n(close, 5)
        if r5 is not None and r5 <= config.MR_TRIGGER_RETURN_5D:
            reasons.append(f"5D_RETURN<={config.MR_TRIGGER_RETURN_5D:.0%}")

        rsi14 = rsi_last(close, 14)
        if rsi14 is not None and rsi14 <= config.MR_TRIGGER_RSI14:
            reasons.append(f"RSI14<={config.MR_TRIGGER_RSI14:.0f}")

        if "open" in df.columns and len(close) >= 2:
            prev_close = float(close.iloc[-2])
            today_open = float(df["open"].astype(float).iloc[-1])
            if prev_close:
                gap = today_open / prev_close - 1.0
                if gap <= config.MR_TRIGGER_GAP_DOWN:
                    reasons.append(f"GAP_DOWN<={config.MR_TRIGGER_GAP_DOWN:.0%}")

        return ",".join(reasons) if reasons else None
    except Exception:
        return None


def _classify_crash_reason(stock_return_3d: Optional[float],
                            spy_return_3d: Optional[float],
                            sector_return_3d: Optional[float]) -> str:
    """Rule-based only — never distinguishes EARNINGS/GUIDANCE/NEWS (no data
    source for those; see research/annotate_event.py for manual override)."""
    if spy_return_3d is not None and spy_return_3d <= config.MR_CRASH_MARKET_SELLOFF_3D:
        return "MARKET_SELL_OFF"
    if sector_return_3d is not None and sector_return_3d <= config.MR_CRASH_SECTOR_SELLOFF_3D:
        return "SECTOR_SELL_OFF"
    if stock_return_3d is not None and stock_return_3d <= config.MR_TRIGGER_RETURN_3D:
        return "STOCK_SPECIFIC"
    return "UNKNOWN"


def build_observation_snapshot(
    code: str,
    df: Optional[pd.DataFrame],
    spy_df: Optional[pd.DataFrame] = None,
    qqq_df: Optional[pd.DataFrame] = None,
    sector_etf_df: Optional[pd.DataFrame] = None,
    vix_close: Optional[float] = None,
    hmm_interface: Optional[dict] = None,
    trigger_reason: Optional[str] = None,
) -> Optional[dict]:
    """Build one event row's categories 1-4, 5 (pass-through), 6, 8.

    df : OHLCV history up to and including the event day (same shape
        fetch_kline() returns; open/high/low/close required, volume
        optional).
    spy_df / qqq_df / sector_etf_df : same-shape OHLCV for the comparison
        series, already fetched by the caller (no fetching happens here).
    hmm_interface : dict from engine.regime_store.get_regime_interface(code)
        (plus 'prev_regime'/'transition_*' the caller derives before/after
        calling hmm_shadow) — passed straight through, never recomputed
        here.

    Never raises. Returns None only when there's nothing to record at all
    (df missing/too short) — otherwise a dict with whatever subset of
    fields could be computed, remaining keys None.
    """
    try:
        if df is None or len(df) < 6:
            return None

        open_ = df["open"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else None

        # ── Category 1: price extremity ──────────────────────────────────
        return_1d = _return_n(close, 1)
        return_3d = _return_n(close, 3)
        return_5d = _return_n(close, 5)
        return_10d = _return_n(close, 10)

        high_20d = float(high.tail(20).max()) if len(high) >= 20 else None
        dist_from_20d_high = (float(close.iloc[-1]) / high_20d - 1.0) if high_20d else None
        high_60d = float(high.tail(60).max()) if len(high) >= 60 else None
        dist_from_60d_high = (float(close.iloc[-1]) / high_60d - 1.0) if high_60d else None

        atr_val = last_value(atr_series(high, low, close))
        daily_return_over_atr = None
        if atr_val and len(close) >= 2:
            daily_return_over_atr = float(close.iloc[-1] - close.iloc[-2]) / atr_val

        gap_down_pct = None
        if len(close) >= 2 and float(close.iloc[-2]):
            gap_down_pct = float(open_.iloc[-1]) / float(close.iloc[-2]) - 1.0

        # ── Category 2: oversold ─────────────────────────────────────────
        rsi14 = rsi_last(close, 14)
        rsi5 = rsi_last(close, 5)
        stoch = stochastic(high, low, close)
        stoch_k = last_value(stoch["k"])
        stoch_d = last_value(stoch["d"])
        wr = last_value(williams_r(high, low, close))

        ma20 = close.rolling(20).mean()
        dist_from_ma20 = None
        if len(close) >= 20 and pd.notna(ma20.iloc[-1]) and ma20.iloc[-1]:
            dist_from_ma20 = float(close.iloc[-1] / ma20.iloc[-1] - 1.0)
        ma50 = close.rolling(50).mean()
        dist_from_ma50 = None
        if len(close) >= 50 and pd.notna(ma50.iloc[-1]) and ma50.iloc[-1]:
            dist_from_ma50 = float(close.iloc[-1] / ma50.iloc[-1] - 1.0)

        # ── Category 3: volume / panic ───────────────────────────────────
        volume_last = volume_ratio_20d = volume_zscore = down_day_volume = None
        volume_spike = None
        if volume is not None and len(volume) >= 2:
            volume_last = float(volume.iloc[-1])
            volume_ratio_20d = last_value(volume_ratio_series(volume))
            if len(volume) >= 20:
                tail = volume.tail(20)
                std = float(tail.std())
                if std:
                    volume_zscore = float((volume.iloc[-1] - tail.mean()) / std)
            if float(close.iloc[-1]) < float(close.iloc[-2]):
                down_day_volume = volume_last
            if volume_ratio_20d is not None:
                volume_spike = bool(volume_ratio_20d >= _VOLUME_SPIKE_RATIO)

        # ── Category 4: candle stabilization (at-event-day only) ────────
        low_close_pos = last_value(low_to_close_position(high, low, close))
        clv = last_value(close_location_value(open_, high, low, close))
        lw_pct = last_value(lower_wick_pct(open_, high, low, close))
        b_pct = last_value(body_pct(open_, high, low, close))
        hammer = last_bool(is_hammer(open_, high, low, close))
        recovered_prior_low = None
        if len(low) >= 2:
            recovered_prior_low = bool(float(close.iloc[-1]) > float(low.iloc[-2]))

        # ── Category 6: market environment ───────────────────────────────
        spy_close = spy_df["close"].astype(float) if spy_df is not None and len(spy_df) else None
        qqq_close = qqq_df["close"].astype(float) if qqq_df is not None and len(qqq_df) else None
        sector_close = (sector_etf_df["close"].astype(float)
                         if sector_etf_df is not None and len(sector_etf_df) else None)

        spy_return_1d = _return_n(spy_close, 1) if spy_close is not None else None
        spy_return_5d = _return_n(spy_close, 5) if spy_close is not None else None
        spy_return_3d = _return_n(spy_close, 3) if spy_close is not None else None
        qqq_return_1d = _return_n(qqq_close, 1) if qqq_close is not None else None
        qqq_return_5d = _return_n(qqq_close, 5) if qqq_close is not None else None

        sector = config.SECTOR_MAP.get(code)
        sector_etf = sector_etf_for(sector) if sector else None
        sector_return_1d = _return_n(sector_close, 1) if sector_close is not None else None
        sector_return_5d = _return_n(sector_close, 5) if sector_close is not None else None
        sector_return_3d = _return_n(sector_close, 3) if sector_close is not None else None

        stock_vs_sector_return_5d = None
        if return_5d is not None and sector_return_5d is not None:
            stock_vs_sector_return_5d = return_5d - sector_return_5d

        # ── Category 8: crash reason (auto heuristic) ────────────────────
        crash_reason = _classify_crash_reason(return_3d, spy_return_3d, sector_return_3d)

        hmm_interface = hmm_interface or {}

        return {
            "code": code,
            # Category 1
            "return_1d": return_1d, "return_3d": return_3d,
            "return_5d": return_5d, "return_10d": return_10d,
            "dist_from_20d_high": dist_from_20d_high,
            "dist_from_60d_high": dist_from_60d_high,
            "atr": atr_val,
            "daily_return_over_atr": daily_return_over_atr,
            "gap_down_pct": gap_down_pct,
            # Category 2
            "rsi14": rsi14, "rsi5": rsi5,
            "stoch_k": stoch_k, "stoch_d": stoch_d,
            "williams_r": wr,
            "dist_from_ma20": dist_from_ma20, "dist_from_ma50": dist_from_ma50,
            # Category 3
            "volume": volume_last, "volume_ratio_20d": volume_ratio_20d,
            "volume_zscore": volume_zscore, "down_day_volume": down_day_volume,
            "volume_spike": volume_spike,
            # Category 4
            "low_to_close_position": low_close_pos,
            "close_location_value": clv,
            "lower_wick_pct": lw_pct, "body_pct": b_pct,
            "is_hammer": hammer, "recovered_prior_low": recovered_prior_low,
            # Category 5 (pass-through from caller)
            "hmm_regime": hmm_interface.get("current_regime"),
            "hmm_confidence": hmm_interface.get("regime_confidence"),
            "hmm_duration_days": hmm_interface.get("regime_duration"),
            "hmm_changed_today": hmm_interface.get("regime_changed_today"),
            "hmm_prev_regime": hmm_interface.get("prev_regime"),
            "transition_bear_to_sideways": hmm_interface.get("transition_bear_to_sideways"),
            "transition_bear_to_bull": hmm_interface.get("transition_bear_to_bull"),
            # Category 6
            "spy_return_1d": spy_return_1d, "spy_return_5d": spy_return_5d,
            "qqq_return_1d": qqq_return_1d, "qqq_return_5d": qqq_return_5d,
            "vix_close": vix_close,
            "sector": sector, "sector_etf": sector_etf,
            "sector_etf_return_1d": sector_return_1d,
            "sector_etf_return_5d": sector_return_5d,
            "stock_vs_sector_return_5d": stock_vs_sector_return_5d,
            # Category 8
            "crash_reason": crash_reason,
            "crash_reason_source": "AUTO_HEURISTIC",
            # trigger
            "trigger_reason": trigger_reason,
        }
    except Exception:
        return None
