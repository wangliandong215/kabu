"""
engine/market_regime.py — Market Regime Detection (MRD), v2.5.

MRD answers exactly one question: "what state is the market (or a given
symbol) currently in?" It is the upstream, purely-descriptive layer of the
pipeline:

    Market Data -> Feature Extraction (engine/regime_features.py)
                -> Market Regime Detection (this module, v2.5)
                -> Risk Engine (future v2.6)
                -> Strategy Engine (v2.3)
                -> Position Sizing (future v2.8)
                -> Exit Engine (future v3.0)

MRD deliberately does NOT decide anything — it never returns allow_buy,
allow_sell, risk_multiplier, position_size, or any other decision field.
Those belong to downstream modules (Risk Engine / Strategy Engine). This
module only outputs facts: which regime, and how confident the classification
is. It is NOT wired into engine/runner.py or backtest_portfolio.py — the two
existing regime-like modules (engine/regime.py for per-stock strategy
routing, engine/market_weather.py for the macro risk multiplier) are
unaffected by this module's existence.

classify()/classify_series() are data-source-agnostic: pass a QQQ OHLCV
DataFrame to get a "market" regime, or a single stock's OHLCV DataFrame to
get a "per-symbol" regime — same logic either way.

Regime is a four-state rule-based classifier for stage 1. RegimeClassifier is
an abstract interface so a stage-2 unsupervised model (GMM/HMM) can be added
later as another implementation without changing any caller.
"""
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd

import config
from engine.regime_features import extract_features


MRD_VERSION = "1.0"   # this module's own classification-logic version, for
                      # audit trails (e.g. engine/trade_tracker.py) that need
                      # to record which MRD version produced a given regime
                      # fact — bump only when classify_series()'s rules change.


class Regime(Enum):
    """Extensible — new members (e.g. BULL_TREND, CRASH) can be added later
    without breaking existing consumers of the four stage-1 states.

    HMM_* members (10-13) are stage-2's namespace (engine/hmm_regime.py,
    V2.7 Stage 3) — a numerically separate range from the stage-1 rule-based
    states (0-3) so the two axes (ADX/ATR-threshold regime vs. learned HMM
    regime) are never confusable at a glance, even though both are legal
    Regime values wherever a RegimeSnapshot is consumed."""
    UNKNOWN            = -1   # insufficient warm-up data — no fact to report
    LOW_VOL_SIDEWAYS   = 0
    HIGH_VOL_SIDEWAYS  = 1
    LOW_VOL_TREND      = 2
    HIGH_VOL_TREND     = 3
    HMM_BEAR           = 10
    HMM_CORRECTION     = 11
    HMM_SIDEWAYS       = 12
    HMM_BULL           = 13


REGIME_NAMES = {
    Regime.UNKNOWN:           "Unknown",
    Regime.LOW_VOL_SIDEWAYS:  "Low Volatility Sideways",
    Regime.HIGH_VOL_SIDEWAYS: "High Volatility Sideways",
    Regime.LOW_VOL_TREND:     "Low Volatility Trend",
    Regime.HIGH_VOL_TREND:    "High Volatility Trend",
    Regime.HMM_BEAR:          "Bear",
    Regime.HMM_CORRECTION:    "Correction",
    Regime.HMM_SIDEWAYS:      "Sideways",
    Regime.HMM_BULL:          "Bull",
}


class RegimeSnapshot:
    """Immutable result of classifying one bar. Plain fields, not a
    NamedTuple, because Regime (an Enum) isn't hashable-friendly with some
    NamedTuple equality checks used in tests — behaves the same either way."""
    __slots__ = ("timestamp", "regime", "regime_name", "adx", "atr",
                 "atr_pct", "bb_width", "volume_ratio", "confidence")

    def __init__(self, timestamp, regime: Regime, adx, atr, atr_pct,
                 bb_width, volume_ratio, confidence: float):
        self.timestamp = timestamp
        self.regime = regime
        self.regime_name = REGIME_NAMES.get(regime, str(regime))
        self.adx = adx
        self.atr = atr
        self.atr_pct = atr_pct
        self.bb_width = bb_width
        self.volume_ratio = volume_ratio
        self.confidence = confidence

    def __repr__(self):
        return (f"RegimeSnapshot(timestamp={self.timestamp!r}, "
                f"regime={self.regime_name!r}, confidence={self.confidence:.1f})")


class RegimeClassifier(ABC):
    """Stable interface for regime classifiers. Stage 1 ships
    RuleBasedRegimeClassifier; a stage-2 MLRegimeClassifier (GMM/HMM) can
    implement this same interface later without any caller-side changes."""

    @abstractmethod
    def classify_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame indexed like df with columns: regime (Regime),
        regime_name (str), adx, atr, atr_pct, bb_width, volume_ratio,
        confidence (float, 0-100).
        """
        raise NotImplementedError


def _axis_confidence(value: pd.Series, threshold: float, scale: float) -> pd.Series:
    """Symmetric distance-based confidence: the further `value` sits from
    `threshold` (in either direction), the more confident the classification
    on that axis is — 50 exactly at the boundary (maximally uncertain),
    saturating at 100 once the distance reaches `scale`."""
    return ((value - threshold).abs() / scale * 50 + 50).clip(upper=100)


class RuleBasedRegimeClassifier(RegimeClassifier):
    """
    Stage 1 implementation. Two independent boolean axes decide the regime:

        is_trending = ADX >= config.MRD_ADX_TREND_MIN
        is_high_vol = ATR% >= config.MRD_ATR_PCT_HIGH

        not trending, not high vol -> LOW_VOL_SIDEWAYS
        not trending, high vol     -> HIGH_VOL_SIDEWAYS
        trending,     not high vol -> LOW_VOL_TREND
        trending,     high vol     -> HIGH_VOL_TREND

    Volume Ratio and BB Width are NOT part of the primary split (keeping the
    classification boundary explainable from two thresholds) — Volume Ratio
    instead corroborates confidence for trending regimes (a trend backed by
    above-average volume is more credible than one that isn't; see PRD
    example: "ATR rising but volume not confirming -> lower confidence").
    BB Width is reported as a raw fact but not yet used in confidence scoring
    (left for a future compression/squeeze refinement, out of scope here).
    """

    def classify_series(self, df: pd.DataFrame) -> pd.DataFrame:
        features = extract_features(df)

        adx = features["adx"]
        atr_pct = features["atr_pct"]
        volume_ratio = features["volume_ratio"]

        is_trending = adx >= config.MRD_ADX_TREND_MIN
        is_high_vol = atr_pct >= config.MRD_ATR_PCT_HIGH

        regime = pd.Series(Regime.LOW_VOL_SIDEWAYS, index=df.index, dtype=object)
        regime[~is_trending & is_high_vol] = Regime.HIGH_VOL_SIDEWAYS
        regime[is_trending & ~is_high_vol] = Regime.LOW_VOL_TREND
        regime[is_trending & is_high_vol]  = Regime.HIGH_VOL_TREND

        # Wilder's ewm-based ADX/ATR are numerically non-NaN from the second
        # bar onward, but not statistically reliable that early (same
        # rationale as engine/regime.py's own min_bars gate — smoothing needs
        # ~2-3x the period to settle). Treat both an explicit minimum history
        # AND any residual NaN as "not enough data to report a fact yet".
        min_bars = (max(config.MRD_ADX_PERIOD, config.MRD_ATR_PERIOD) * 3
                    + max(config.MRD_BB_PERIOD, config.MRD_VOLUME_MA_PERIOD))
        not_enough_history = pd.Series(np.arange(len(df)) < min_bars, index=df.index)
        warmup = not_enough_history | adx.isna() | atr_pct.isna()
        regime[warmup] = Regime.UNKNOWN

        confidence = self._confidence(adx, atr_pct, volume_ratio, regime)
        confidence[warmup] = 0.0

        out = pd.DataFrame(index=df.index)
        out["regime"] = regime
        out["regime_name"] = regime.map(lambda r: REGIME_NAMES.get(r, str(r)))
        out["adx"] = adx
        out["atr"] = features["atr"]
        out["atr_pct"] = atr_pct
        out["bb_width"] = features["bb_width"]
        out["volume_ratio"] = volume_ratio
        out["confidence"] = confidence
        return out

    @staticmethod
    def _confidence(adx: pd.Series, atr_pct: pd.Series, volume_ratio: pd.Series,
                     regime: pd.Series) -> pd.Series:
        trend_conf = _axis_confidence(adx, config.MRD_ADX_TREND_MIN,
                                       config.MRD_ADX_CONFIDENCE_SCALE)
        vol_conf = _axis_confidence(atr_pct, config.MRD_ATR_PCT_HIGH,
                                     config.MRD_ATR_PCT_CONFIDENCE_SCALE)
        base = (trend_conf + vol_conf) / 2

        is_trend_regime = regime.isin([Regime.LOW_VOL_TREND, Regime.HIGH_VOL_TREND])
        confirmed = is_trend_regime & (volume_ratio >= config.MRD_VOLUME_RATIO_CONFIRM)
        weak = is_trend_regime & (volume_ratio < config.MRD_VOLUME_RATIO_WEAK)

        base = base.where(~confirmed, base + config.MRD_CONFIDENCE_VOLUME_BONUS)
        base = base.where(~weak, base - config.MRD_CONFIDENCE_VOLUME_PENALTY)

        return base.clip(lower=0, upper=100)


_default_classifier = RuleBasedRegimeClassifier()


def classify_series(df: pd.DataFrame, classifier: RegimeClassifier = None) -> pd.DataFrame:
    """Per-bar regime classification for the whole df. Pass any OHLCV
    DataFrame — QQQ for a market-wide regime, a single stock for a
    per-symbol regime. Defaults to the stage-1 rule-based classifier."""
    classifier = classifier or _default_classifier
    return classifier.classify_series(df)


def classify(df: pd.DataFrame, classifier: RegimeClassifier = None) -> RegimeSnapshot:
    """Convenience wrapper: classify only the last bar of df."""
    series = classify_series(df, classifier)
    if len(series) == 0:
        return RegimeSnapshot(None, Regime.UNKNOWN, np.nan, np.nan, np.nan,
                               np.nan, np.nan, 0.0)
    last = series.iloc[-1]
    return RegimeSnapshot(
        timestamp=series.index[-1],
        regime=last["regime"],
        adx=last["adx"],
        atr=last["atr"],
        atr_pct=last["atr_pct"],
        bb_width=last["bb_width"],
        volume_ratio=last["volume_ratio"],
        confidence=last["confidence"],
    )


def to_regime_ctx(snapshot: RegimeSnapshot) -> dict:
    """Serialize a RegimeSnapshot into a JSON-ready dict for external
    consumers that want to persist a regime fact without depending on the
    Regime enum (e.g. engine/trade_tracker.py's log_entry/log_exit
    `regime_ctx` argument). `regime_label` uses the enum member's own name
    (e.g. "LOW_VOL_SIDEWAYS") rather than REGIME_NAMES' prose string, since
    that's the stable, machine-groupable identifier downstream attribution
    reports key off of."""
    def _clean(v):
        return None if (isinstance(v, float) and np.isnan(v)) else v

    return {
        "regime": snapshot.regime.value,
        "regime_label": snapshot.regime.name,
        "confidence": _clean(snapshot.confidence),
        "features": {
            "adx":          _clean(snapshot.adx),
            "atr":          _clean(snapshot.atr),
            "atr_pct":      _clean(snapshot.atr_pct),
            "bb_width":     _clean(snapshot.bb_width),
            "volume_ratio": _clean(snapshot.volume_ratio),
        },
        "timestamp": str(snapshot.timestamp) if snapshot.timestamp is not None else None,
    }
