"""
research/trade_intelligence_signal_confidence.py — V3.6-A Signal/Confidence
pattern discovery (spec section 九).

Bands confidence_score and its 0-100-scale sub-components (hmm_component,
market_component, volatility_component) via
config.TRADE_INTELLIGENCE_CONFIDENCE_BUCKETS and compares win rate per band
against the sample baseline via bucket_compare — this is the only path that
can reach CANDIDATE_PATTERN. A plain Pearson correlation between each score
and pnl_pct is also reported, but ALWAYS as an OBSERVATION: a bare
correlation coefficient is never, on its own, enough to promote a finding
in this design (see research.trade_intelligence_patterns.classify_state) —
keeps the promotion path exclusively through the interpretable, sample-
size-gated bucket comparison.
"""
from typing import List, Optional

import pandas as pd

import config
from research.trade_intelligence_patterns import (
    PatternCandidate, bucket_compare, bucket_label, classify_state,
)

ANALYSIS_TYPE = "SIGNAL_CONFIDENCE"

# 0-100 scale scores (see engine/confidence_score.py) — banded via the
# shared confidence buckets and win-rate-compared.
_BAND_SCORE_COLUMNS = [
    ("confidence_score", "confidence_score"),
    ("hmm_component", "hmm_component"),
    ("market_component", "market_component"),
    ("volatility_component", "volatility_component"),
]
# Correlation-only (descriptive) — includes historical_win_rate, which is a
# 0..1 fraction and not on the same 0-100 scale as the bucket columns above,
# so it is only ever reported via correlation, never banded.
_CORRELATION_COLUMNS = [c for c, _ in _BAND_SCORE_COLUMNS] + ["historical_win_rate"]


def _band_candidates(df: pd.DataFrame, score_col: str, pattern_prefix: str) -> List[PatternCandidate]:
    if score_col not in df.columns or "pnl_pct" not in df.columns:
        return []
    sub = df[[score_col, "pnl_pct"]].dropna()
    if sub.empty:
        return []
    sub = sub.copy()
    sub["_is_win"] = (sub["pnl_pct"] > 0).astype(float)
    baseline_win_rate = float(sub["_is_win"].mean())
    edges = config.TRADE_INTELLIGENCE_CONFIDENCE_BUCKETS
    sub["_band"] = sub[score_col].apply(lambda v: bucket_label(v, edges))

    out = []
    table = bucket_compare(sub, "_band", "_is_win")
    for _, row in table.iterrows():
        band, win_rate, n = row["_band"], float(row["value"]), int(row["n"])
        effect_size = win_rate - baseline_win_rate
        state = classify_state(n_sample=n, effect_size=effect_size, precision=None)
        if state is None:
            continue
        description = (
            f"[{state}] Trades with {score_col} in band '{band}' had a win "
            f"rate of {win_rate:.0%} (n={n}) vs a sample baseline of "
            f"{baseline_win_rate:.0%}. Association only, not shown to be causal."
        )
        out.append(PatternCandidate(
            analysis_type=ANALYSIS_TYPE, pattern_key=f"{pattern_prefix}_band_{band}",
            state=state, description=description, metric_name="win_rate",
            metric_value=win_rate, comparison_metric_value=baseline_win_rate,
            effect_size=effect_size, n_sample=n, n_baseline=len(sub),
            slice_definition={"score_col": score_col, "band": str(band), "edges": list(edges)},
        ))
    return out


def _correlation_observation(df: pd.DataFrame, score_col: str,
                              pattern_prefix: str) -> Optional[PatternCandidate]:
    if score_col not in df.columns or "pnl_pct" not in df.columns:
        return None
    sub = df[[score_col, "pnl_pct"]].dropna()
    n = len(sub)
    if n < config.TRADE_INTELLIGENCE_MIN_OBSERVATION_N:
        return None
    corr = sub[score_col].corr(sub["pnl_pct"])
    if pd.isna(corr):
        return None
    description = (
        f"[OBSERVATION] {score_col} and pnl_pct show a sample correlation of "
        f"{corr:.2f} (n={n}). Descriptive only — a correlation coefficient "
        f"never promotes to CANDIDATE_PATTERN on its own in this design."
    )
    return PatternCandidate(
        analysis_type=ANALYSIS_TYPE, pattern_key=f"{pattern_prefix}_correlation",
        state="OBSERVATION", description=description, metric_name="pearson_r",
        metric_value=float(corr), n_sample=n, n_baseline=n,
        slice_definition={"score_col": score_col},
    )


def analyze(df: pd.DataFrame) -> List[PatternCandidate]:
    if df is None or df.empty or "pnl_pct" not in df.columns:
        return []
    out: List[PatternCandidate] = []
    for score_col, prefix in _BAND_SCORE_COLUMNS:
        out.extend(_band_candidates(df, score_col, prefix))
    for score_col in _CORRELATION_COLUMNS:
        corr = _correlation_observation(df, score_col, score_col)
        if corr is not None:
            out.append(corr)
    return out
