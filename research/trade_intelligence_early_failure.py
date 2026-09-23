"""
research/trade_intelligence_early_failure.py — V3.6-A Early Failure pattern
discovery (spec section 七/十三).

Extends (does not duplicate) the 3 hardcoded MFE/MAE candidate signals
research/early_failure_monitor.py already tracks for long-term precision/
recall drift, by testing a small grid of additional univariate flags
(confidence, entry quality, position size, entry regime) against the same
exit_category == 'EARLY_FAILURE' ground truth. Pure function, no I/O — see
research/trade_intelligence_data.py for the input DataFrame and
research/trade_intelligence_runner.py for orchestration.

Every finding is gated through
research.trade_intelligence_patterns.classify_state() and hedged
(Observation/Candidate Pattern only) — a flagged rate above baseline is
reported as an association, never a proven cause.
"""
from typing import List, Optional

import pandas as pd

from research.trade_intelligence_patterns import (
    PatternCandidate, binary_flag_stats, classify_state,
)

ANALYSIS_TYPE = "EARLY_FAILURE"
_POSITIVE_OUTCOMES = {"EARLY_FAILURE"}

# Fixed heuristic thresholds — deliberately not "the best parameters",
# mirroring early_failure_monitor.py's own precedent of keeping thresholds
# fixed so results are comparable across refreshes.
_HIGH_ENTRY_VOLUME_RATIO = 2.0
_LOW_CONFIDENCE_BAR = 40.0
_LARGE_POSITION_PCT = 0.15
_MFE_D2_TIGHT = 0.01
_MFE_D2_LOOSE = 0.02
_MAE_D5_BAR = -0.04


def _candidate_from_flag(df: pd.DataFrame, flag: pd.Series, pattern_key: str,
                          slice_definition: dict) -> Optional[PatternCandidate]:
    if flag.isna().all():
        return None
    tmp = df[["trade_id", "exit_category"]].copy()
    tmp["_flag"] = flag.reindex(df.index).fillna(0).astype(int)
    if "exit_category" not in tmp or tmp["exit_category"].dropna().empty:
        return None
    stats = binary_flag_stats(tmp, "_flag", "exit_category", _POSITIVE_OUTCOMES)
    n_flagged = stats["n_flagged"]
    if n_flagged == 0:
        return None
    n_total = len(tmp)
    baseline_rate = (stats["n_positive_total"] / n_total) if n_total else None
    effect_size = ((stats["precision"] - baseline_rate)
                    if (stats["precision"] is not None and baseline_rate is not None) else None)
    state = classify_state(n_sample=n_flagged, effect_size=effect_size, precision=stats["precision"])
    if state is None:
        return None

    precision_pct = f"{stats['precision']:.0%}" if stats["precision"] is not None else "n/a"
    baseline_pct = f"{baseline_rate:.0%}" if baseline_rate is not None else "n/a"
    description = (
        f"[{state}] Trades matching '{pattern_key}' showed an EARLY_FAILURE "
        f"rate of {precision_pct} (n={n_flagged}) vs a sample baseline of "
        f"{baseline_pct} (n={n_total}). Association only, not shown to be causal."
    )
    return PatternCandidate(
        analysis_type=ANALYSIS_TYPE, pattern_key=pattern_key, state=state,
        description=description, metric_name="early_failure_rate",
        metric_value=stats["precision"], comparison_metric_value=baseline_rate,
        effect_size=effect_size, n_sample=n_flagged, n_baseline=n_total,
        precision=stats["precision"], recall=stats["recall"],
        false_positive_count=stats["false_positive_count"],
        slice_definition=slice_definition,
        source_trade_ids=list(tmp.loc[tmp["_flag"] == 1, "trade_id"]),
    )


def analyze(df: pd.DataFrame) -> List[PatternCandidate]:
    if df is None or df.empty or "exit_category" not in df.columns:
        return []
    out: List[PatternCandidate] = []

    def _add(flag, key, slice_def):
        cand = _candidate_from_flag(df, flag, key, slice_def)
        if cand is not None:
            out.append(cand)

    if "MFE_at_day_2" in df.columns:
        _add(df["MFE_at_day_2"] < _MFE_D2_TIGHT, "mfe_day2_lt_1pct",
             {"metric": "MFE_at_day_2", "threshold": _MFE_D2_TIGHT})
        _add(df["MFE_at_day_2"] < _MFE_D2_LOOSE, "mfe_day2_lt_2pct",
             {"metric": "MFE_at_day_2", "threshold": _MFE_D2_LOOSE})
    if "MAE_at_day_5" in df.columns:
        _add(df["MAE_at_day_5"] <= _MAE_D5_BAR, "mae_day5_lte_neg4pct",
             {"metric": "MAE_at_day_5", "threshold": _MAE_D5_BAR})
    if "confidence_score" in df.columns:
        _add(df["confidence_score"] < _LOW_CONFIDENCE_BAR, "low_confidence_score",
             {"metric": "confidence_score", "threshold": _LOW_CONFIDENCE_BAR})
    if "entry_volume_ratio" in df.columns:
        _add(df["entry_volume_ratio"] > _HIGH_ENTRY_VOLUME_RATIO, "high_entry_volume_ratio",
             {"metric": "entry_volume_ratio", "threshold": _HIGH_ENTRY_VOLUME_RATIO})
    if "position_pct" in df.columns:
        _add(df["position_pct"] > _LARGE_POSITION_PCT, "large_position_pct",
             {"metric": "position_pct", "threshold": _LARGE_POSITION_PCT})
    if "entry_regime_label" in df.columns:
        for label in df["entry_regime_label"].dropna().unique():
            _add(df["entry_regime_label"] == label, f"entry_regime_{label}",
                 {"metric": "entry_regime_label", "value": str(label)})

    return out
