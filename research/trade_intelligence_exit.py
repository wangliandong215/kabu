"""
research/trade_intelligence_exit.py — V3.6-A Exit pattern discovery (spec
section 八).

Buckets giveback_pct / mfe_capture / pnl_pct by exit_reason_code, buckets
holding_days via config.TRADE_INTELLIGENCE_HOLDING_DAYS_BUCKETS against win
rate, and cross-tabs strategy_name x EARLY_FAILURE rate (only when >=2
strategies are present) — looking for exit behavior that under/over-
performs the sample baseline (a candidate signal that a given exit reason
or holding-time band tends to exit too early or too late). Descriptive/
associative only — see research.trade_intelligence_patterns.classify_state.
"""
from typing import List

import pandas as pd

import config
from research.trade_intelligence_patterns import (
    PatternCandidate, bucket_compare, bucket_label, classify_state,
)

ANALYSIS_TYPE = "EXIT"


def _numeric_bucket_candidates(df: pd.DataFrame, bucket_col: str, metric_col: str,
                                pattern_prefix: str) -> List[PatternCandidate]:
    if bucket_col not in df.columns or metric_col not in df.columns:
        return []
    baseline_series = df[metric_col].dropna()
    if baseline_series.empty:
        return []
    baseline = float(baseline_series.mean())
    table = bucket_compare(df, bucket_col, metric_col)
    out = []
    for _, row in table.iterrows():
        bucket, avg, n = row[bucket_col], float(row["value"]), int(row["n"])
        denom = abs(baseline) if baseline else None
        effect_size = ((avg - baseline) / denom) if denom else None
        state = classify_state(n_sample=n, effect_size=effect_size, precision=None)
        if state is None:
            continue
        description = (
            f"[{state}] Trades exiting via '{bucket}' averaged {metric_col}="
            f"{avg:.3f} (n={n}) vs a sample baseline of {baseline:.3f}. "
            f"Association only, not shown to be causal."
        )
        out.append(PatternCandidate(
            analysis_type=ANALYSIS_TYPE, pattern_key=f"{pattern_prefix}_{bucket}",
            state=state, description=description, metric_name=metric_col,
            metric_value=avg, comparison_metric_value=baseline,
            effect_size=effect_size, n_sample=n, n_baseline=len(baseline_series),
            slice_definition={bucket_col: str(bucket)},
        ))
    return out


def _holding_days_bucket_candidates(df: pd.DataFrame) -> List[PatternCandidate]:
    if "holding_days" not in df.columns or "pnl_pct" not in df.columns:
        return []
    sub = df[["holding_days", "pnl_pct"]].dropna()
    if sub.empty:
        return []
    edges = config.TRADE_INTELLIGENCE_HOLDING_DAYS_BUCKETS
    sub = sub.copy()
    sub["_is_win"] = (sub["pnl_pct"] > 0).astype(float)
    baseline_win_rate = float(sub["_is_win"].mean())
    sub["_band"] = sub["holding_days"].apply(lambda v: bucket_label(v, edges, suffix="d"))

    out = []
    table = bucket_compare(sub, "_band", "_is_win")
    for _, row in table.iterrows():
        band, win_rate, n = row["_band"], float(row["value"]), int(row["n"])
        effect_size = win_rate - baseline_win_rate
        state = classify_state(n_sample=n, effect_size=effect_size, precision=None)
        if state is None:
            continue
        description = (
            f"[{state}] Trades held '{band}' had a win rate of {win_rate:.0%} "
            f"(n={n}) vs a sample baseline of {baseline_win_rate:.0%}. "
            f"Association only, not shown to be causal."
        )
        out.append(PatternCandidate(
            analysis_type=ANALYSIS_TYPE, pattern_key=f"holding_days_{band}",
            state=state, description=description, metric_name="win_rate",
            metric_value=win_rate, comparison_metric_value=baseline_win_rate,
            effect_size=effect_size, n_sample=n, n_baseline=len(sub),
            slice_definition={"holding_days_band": str(band), "edges": list(edges)},
        ))
    return out


def _strategy_early_failure_candidates(df: pd.DataFrame) -> List[PatternCandidate]:
    if "strategy_name" not in df.columns or "exit_category" not in df.columns:
        return []
    sub = df.dropna(subset=["strategy_name", "exit_category"])
    strategies = sub["strategy_name"].unique()
    if len(strategies) < 2:
        return []
    baseline_ef_rate = float((sub["exit_category"] == "EARLY_FAILURE").mean())
    out = []
    for strat in strategies:
        strat_df = sub[sub["strategy_name"] == strat]
        n = len(strat_df)
        ef_rate = float((strat_df["exit_category"] == "EARLY_FAILURE").mean())
        effect_size = ef_rate - baseline_ef_rate
        state = classify_state(n_sample=n, effect_size=effect_size, precision=None)
        if state is None:
            continue
        description = (
            f"[{state}] Strategy '{strat}' showed an EARLY_FAILURE exit rate "
            f"of {ef_rate:.0%} (n={n}) vs a sample baseline of "
            f"{baseline_ef_rate:.0%}. Association only, not shown to be causal."
        )
        out.append(PatternCandidate(
            analysis_type=ANALYSIS_TYPE, pattern_key=f"strategy_{strat}_early_failure_rate",
            state=state, description=description, metric_name="early_failure_rate",
            metric_value=ef_rate, comparison_metric_value=baseline_ef_rate,
            effect_size=effect_size, n_sample=n, n_baseline=len(sub),
            slice_definition={"strategy_name": str(strat)},
        ))
    return out


def analyze(df: pd.DataFrame) -> List[PatternCandidate]:
    if df is None or df.empty:
        return []
    out: List[PatternCandidate] = []
    out.extend(_numeric_bucket_candidates(df, "exit_reason_code", "giveback_pct",
                                           "giveback_by_exit_reason"))
    out.extend(_numeric_bucket_candidates(df, "exit_reason_code", "mfe_capture",
                                           "mfe_capture_by_exit_reason"))
    out.extend(_numeric_bucket_candidates(df, "exit_reason_code", "pnl_pct",
                                           "pnlpct_by_exit_reason"))
    out.extend(_holding_days_bucket_candidates(df))
    out.extend(_strategy_early_failure_candidates(df))
    return out
