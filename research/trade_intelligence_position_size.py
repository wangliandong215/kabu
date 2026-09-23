"""
research/trade_intelligence_position_size.py — V3.6-A Position Size pattern
discovery (spec section 十).

Bands position size (final_position if V3.0-A Dynamic Sizing populated it,
else position_pct) via config.TRADE_INTELLIGENCE_POSITION_PCT_BUCKETS and
compares win rate / avg pnl_pct / avg mae per band against the sample
baseline — a read-only check of whether larger allocations from the
existing sizing formula (risk/dynamic_sizing.py, never touched by this
module) actually correlate with better realized outcomes. skip_reason
frequency is reported as a purely descriptive OBSERVATION (no comparison
group, so it can never promote to CANDIDATE_PATTERN).
"""
from typing import List, Optional

import pandas as pd

import config
from research.trade_intelligence_patterns import (
    PatternCandidate, bucket_compare, bucket_label, classify_state,
)

ANALYSIS_TYPE = "POSITION_SIZE"


def _size_col(df: pd.DataFrame) -> Optional[str]:
    if "final_position" in df.columns and df["final_position"].notna().any():
        return "final_position"
    if "position_pct" in df.columns and df["position_pct"].notna().any():
        return "position_pct"
    return None


def _band_candidates(df: pd.DataFrame, size_col: str, metric_col: str,
                      metric_label: str) -> List[PatternCandidate]:
    sub = df[[size_col, metric_col]].dropna()
    if sub.empty:
        return []
    edges = config.TRADE_INTELLIGENCE_POSITION_PCT_BUCKETS
    baseline = float(sub[metric_col].mean())
    sub = sub.copy()
    sub["_band"] = sub[size_col].apply(lambda v: bucket_label(v, edges))

    out = []
    table = bucket_compare(sub, "_band", metric_col)
    for _, row in table.iterrows():
        band, avg, n = row["_band"], float(row["value"]), int(row["n"])
        denom = abs(baseline) if baseline else None
        effect_size = ((avg - baseline) / denom) if denom else None
        state = classify_state(n_sample=n, effect_size=effect_size, precision=None)
        if state is None:
            continue
        description = (
            f"[{state}] Trades with {size_col} in band '{band}' averaged "
            f"{metric_label}={avg:.3f} (n={n}) vs a sample baseline of "
            f"{baseline:.3f}. Association only, not shown to be causal."
        )
        out.append(PatternCandidate(
            analysis_type=ANALYSIS_TYPE,
            pattern_key=f"{size_col}_band_{band}_{metric_label}",
            state=state, description=description, metric_name=metric_label,
            metric_value=avg, comparison_metric_value=baseline,
            effect_size=effect_size, n_sample=n, n_baseline=len(sub),
            slice_definition={"size_col": size_col, "band": str(band), "edges": list(edges)},
        ))
    return out


def _win_rate_band_candidates(df: pd.DataFrame, size_col: str) -> List[PatternCandidate]:
    if "pnl_pct" not in df.columns:
        return []
    sub = df[[size_col, "pnl_pct"]].dropna().copy()
    if sub.empty:
        return []
    sub["_is_win"] = (sub["pnl_pct"] > 0).astype(float)
    return _band_candidates(sub[[size_col, "_is_win"]], size_col, "_is_win", "win_rate")


def _skip_reason_observation(df: pd.DataFrame) -> List[PatternCandidate]:
    if "skip_reason" not in df.columns:
        return []
    sub = df["skip_reason"].dropna()
    if sub.empty:
        return []
    out = []
    for reason, count in sub.value_counts().items():
        if count < config.TRADE_INTELLIGENCE_MIN_OBSERVATION_N:
            continue
        description = (
            f"[OBSERVATION] skip_reason '{reason}' appeared {count} times in "
            f"the observed sample (n_total={len(df)}). Purely descriptive — "
            f"no comparison group, never promotable to CANDIDATE_PATTERN."
        )
        out.append(PatternCandidate(
            analysis_type=ANALYSIS_TYPE, pattern_key=f"skip_reason_{reason}",
            state="OBSERVATION", description=description, metric_name="count",
            metric_value=float(count), n_sample=int(count), n_baseline=len(df),
            slice_definition={"skip_reason": str(reason)},
        ))
    return out


def analyze(df: pd.DataFrame) -> List[PatternCandidate]:
    if df is None or df.empty:
        return []
    out: List[PatternCandidate] = []
    size_col = _size_col(df)
    if size_col:
        if "pnl_pct" in df.columns:
            out.extend(_band_candidates(df, size_col, "pnl_pct", "avg_pnl_pct"))
            out.extend(_win_rate_band_candidates(df, size_col))
        if "mae" in df.columns:
            out.extend(_band_candidates(df, size_col, "mae", "avg_mae"))
    out.extend(_skip_reason_observation(df))
    return out
