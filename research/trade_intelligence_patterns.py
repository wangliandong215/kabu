"""
research/trade_intelligence_patterns.py — V3.6-A shared statistics +
Observation/Candidate Pattern gating logic (spec sections 七~十三).

Every research/trade_intelligence_*.py analysis module builds its findings
on top of these primitives instead of reimplementing precision/recall or
bucket-comparison math independently, and every finding is gated through
classify_state() before it is ever recorded or reported — there is no code
path in this package that can produce a "Validated Pattern" or "Production
Rule" state; those require Backtest + Out-of-Sample + Paper Trading, none
of which V3.6-A performs.

binary_flag_stats() generalizes research/early_failure_monitor.py's
_signal_stats() to an arbitrary boolean flag vs. an arbitrary positive-
outcome set. Reimplemented here (not imported) so early_failure_monitor.py
stays byte-for-byte untouched.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd

import config

_FORBIDDEN_CAUSAL_WORDS = (
    "causes", "cause", "leads to", "results in", "proves", "proven",
    "confirmed", "guarantees", "guarantee",
)

_VALID_STATES = ("OBSERVATION", "CANDIDATE_PATTERN")


def _assert_hedged_language(description: str) -> None:
    """Structural guard against the spec's "never asserted as proven
    causation" requirement — raises rather than merely documenting the
    rule, so a description can never silently drift into a causal claim."""
    lowered = description.lower()
    for word in _FORBIDDEN_CAUSAL_WORDS:
        if word in lowered:
            raise ValueError(
                f"trade_intelligence_patterns: description contains forbidden "
                f"causal language {word!r} — findings must stay hedged "
                f"(Observation/Candidate Pattern, never proven causation): "
                f"{description!r}")


@dataclass
class PatternCandidate:
    analysis_type: str          # EARLY_FAILURE | EXIT | SIGNAL_CONFIDENCE | POSITION_SIZE
    pattern_key: str            # stable machine key, unique within analysis_type
    state: str                  # OBSERVATION | CANDIDATE_PATTERN — see classify_state()
    description: str            # hedged language only — checked at construction
    metric_name: str
    metric_value: Optional[float]
    comparison_metric_value: Optional[float] = None
    effect_size: Optional[float] = None
    n_sample: int = 0
    n_baseline: Optional[int] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    false_positive_count: Optional[int] = None
    slice_definition: dict = field(default_factory=dict)
    source_trade_ids: list = field(default_factory=list)

    def __post_init__(self):
        if self.state not in _VALID_STATES:
            raise ValueError(
                f"PatternCandidate.state must be one of {_VALID_STATES}, "
                f"got {self.state!r}")
        _assert_hedged_language(self.description)


def binary_flag_stats(df: pd.DataFrame, flag_col: str, outcome_col: str,
                       positive_outcomes: Iterable[str]) -> dict:
    """precision/recall/false-positive-count for one 0/1 flag column against
    outcome_col, restricted to rows where the flag is 1. Generalizes
    research/early_failure_monitor.py::_signal_stats()."""
    positive_outcomes = set(positive_outcomes)
    flagged = df[df[flag_col] == 1]
    n_flagged = len(flagged)
    tp = int(flagged[outcome_col].isin(positive_outcomes).sum())
    fp = n_flagged - tp
    n_positive_total = int(df[outcome_col].isin(positive_outcomes).sum())
    precision = (tp / n_flagged) if n_flagged else None
    recall = (tp / n_positive_total) if n_positive_total else None
    return dict(n_flagged=n_flagged, precision=precision, recall=recall,
                false_positive_count=fp, n_positive_total=n_positive_total)


def bucket_compare(df: pd.DataFrame, bucket_col: str, metric_col: str,
                    agg: str = "mean") -> pd.DataFrame:
    """groupby(bucket_col)[metric_col].agg(agg) + count, dropping rows with
    a missing bucket or metric value. Returns columns [bucket_col, "value",
    "n"]. The bucketed win-rate/avg-outcome comparison shared by all four
    analysis modules."""
    sub = df[[bucket_col, metric_col]].dropna(subset=[bucket_col, metric_col])
    if sub.empty:
        return pd.DataFrame(columns=[bucket_col, "value", "n"])
    grouped = sub.groupby(bucket_col)[metric_col]
    out = grouped.agg(agg).rename("value").to_frame()
    out["n"] = grouped.count().astype(int)
    return out.reset_index()


def bucket_label(value, edges: Iterable[float], suffix: str = "") -> Optional[str]:
    """Multi-edge, non-overlapping bucket label: "<e0", "e0-e1", ...,
    ">=e_last". `edges` must be ascending. Shared by every analysis module
    that bands a continuous metric (confidence score, position size,
    holding days) instead of each reimplementing its own boundary logic."""
    if value is None or pd.isna(value):
        return None
    prev = None
    for e in edges:
        if value < e:
            return f"<{e:g}{suffix}" if prev is None else f"{prev:g}-{e:g}{suffix}"
        prev = e
    return f">={prev:g}{suffix}"


def classify_state(n_sample: int, effect_size: Optional[float] = None,
                    precision: Optional[float] = None) -> Optional[str]:
    """Returns None (do not record — too few samples), "OBSERVATION", or
    "CANDIDATE_PATTERN". Structurally incapable of returning anything else.

    n_sample < TRADE_INTELLIGENCE_MIN_OBSERVATION_N -> None (not recorded).
    n_sample < TRADE_INTELLIGENCE_MIN_CANDIDATE_N   -> "OBSERVATION".
    else CANDIDATE_PATTERN iff a large-enough effect_size or precision
    clears its bar; otherwise stays "OBSERVATION" even at large n — a big
    sample with a weak effect is still just an observation, not a pattern
    worth acting on later."""
    if n_sample < config.TRADE_INTELLIGENCE_MIN_OBSERVATION_N:
        return None
    if n_sample < config.TRADE_INTELLIGENCE_MIN_CANDIDATE_N:
        return "OBSERVATION"
    if effect_size is not None and abs(effect_size) >= config.TRADE_INTELLIGENCE_MIN_CANDIDATE_EFFECT_PCT:
        return "CANDIDATE_PATTERN"
    if precision is not None and precision >= config.TRADE_INTELLIGENCE_MIN_CANDIDATE_PRECISION:
        return "CANDIDATE_PATTERN"
    return "OBSERVATION"
