"""
engine/pipeline.py — Two-Phase Execution: Candidate Pool / Global Filters /
Ranking Engine stages for engine/runner.py's "open new positions" step.

Full pipeline (Scanner and Signal Generator already exist elsewhere):

    Scanner (engine.scanner.scan/smart_scan)
        -> Signal Generator (the `results` dict scan() returns)
        -> Candidate Pool     )
        -> Global Filters     )  build_candidate_pool()  <- this module
        -> Ranking Engine     )  rank_candidates()        <- this module
        -> Portfolio Construction  )
        -> Order Executor          )  stay in engine/runner.py (stateful,
                                        mutate `portfolio` sequentially —
                                        Active Replacement / Risk Guard /
                                        Capacity Rules are unchanged)

Why this module exists: engine/runner.py used to rank BUY signals by raw
signal_strength and buy greedily as it walked that order, computing each
candidate's full total_score (trend+fundamental+news+weather) lazily,
one-at-a-time, only for however far the loop got before capacity filled and
it `break`-ed. A later-ranked candidate with an equal or better total_score
than an earlier one never got a chance to be compared — it lost purely to
scan/signal_strength order (see the 2026-07-29 US.BKNG vs US.ABNB case, both
scored 100/FULL, where BKNG only won because it happened to rank higher on
signal_strength and reached the last free slot first).

This module fixes that by computing total_score for every surviving BUY
candidate up front (Candidate Pool), then sorting on that (Ranking Engine)
before engine/runner.py's execution loop ever runs. No trading strategy
(ATR Breakout, Donchian, Fixed Stop, ATR Trail, Active Replacement, Risk
Guard, Capacity Rules) is changed — only when/how candidates are compared.

v1 tie-break is score -> signal_strength only. breakout_age doesn't exist
anywhere in the codebase yet and ATR-direction tie-break needs a product
decision, so both are left for a future ranking upgrade — the whole point of
splitting rank_candidates() out is that a v2.9/v3.0 upgrade (confidence
score, historical expectancy, HMM regime, breakout_age, ATR) only touches
this one function's sort key, not Candidate Pool or engine/runner.py.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from engine import fundamental, news_filter, scoring
from engine.scanner import rank_signals
from risk.earnings import is_earnings_blackout
import notify.alert as alert


@dataclass
class Candidate:
    """A BUY signal that survived Global Filters and has a full composite
    score attached. `raw` keeps the original results[code] dict around so
    engine/runner.py's execution loop (sizing, order placement, TradeTracker
    logging) can pull fields (atr, stop_loss_pct, strategy_used, rsi14, ...)
    without this module having to re-declare every one of them."""
    code: str
    strategy: str
    signal_strength: float
    current_price: float
    total_score: float
    score_label: str
    score_components: dict
    confidence: Optional[float] = None   # v2.9 hook — always None in v1
    reasons: List[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def build_candidate_pool(
    results: Dict[str, dict],
    portfolio,
    macro_block,
    score_env: str,
    weather_code: int,
    min_strength: float = None,
) -> List[Candidate]:
    """Global Filters stage: collect every BUY signal that survives the
    cheap, order-independent checks, score it, and return the survivors —
    with NO capacity/exposure/sector check yet (those stay in
    engine/runner.py's execution loop since they're inherently sequential/
    stateful, see module docstring).

    Mirrors engine/runner.py's old inline loop 1:1 in filter order:
    already-held -> cooldown -> min-strength -> macro_block -> earnings
    blackout -> total_score -> SKIP label. Only the ORDER of operations
    relative to "when is total_score computed" changed: it's now computed
    for every survivor here, not lazily one-at-a-time until capacity ran
    out.
    """
    if macro_block:
        return []

    min_strength = config.MIN_ENTRY_STRENGTH if min_strength is None else min_strength
    candidates: List[Candidate] = []

    for sig in rank_signals(results, signal="BUY"):
        code = sig["code"]

        if portfolio.get_position(code):
            alert.log(f"runner: {code} already held — skip")
            continue

        if portfolio.is_cooldown(code):
            alert.log(f"runner: {code} in TRENDING_EARLY stop-out cooldown — skip")
            continue

        strength = sig.get("signal_strength", 0.0)
        if strength < min_strength:
            alert.log(f"runner: {code} strength {strength:.0%} "
                      f"< {min_strength:.0%} threshold — skip")
            continue

        if is_earnings_blackout(code):
            alert.warn_skip(code, f"{code} 处于财报窗口期，跳过开仓")
            continue

        strength = sig.get("signal_strength", 0.5)
        news_result = news_filter.classify_code(code)
        fund_result = fundamental.score(code, env=score_env)
        total = scoring.compute_total_score(
            trend_strength=strength,
            weather_code=weather_code,
            fundamental_score=fund_result["score"],
            news_score=news_result["score"],
        )

        def _fmt(v):
            return f"{v:5.1f}" if v is not None else "  N/A"

        alert.log(
            f"SCORE {code:8s} trend={_fmt(total.trend_score)} "
            f"fund={_fmt(total.fundamental_score)} news={_fmt(total.news_score)} "
            f"weather={_fmt(total.weather_score)} total={_fmt(total.total)} "
            f"-> {total.label}"
            + (f"  news_tier={news_result['tier']}({news_result.get('matched_keyword')})"
               if news_result["tier"] != 3 else "")
            + (f"  fund_tier={fund_result['tier']}({fund_result['reason']})"
               if fund_result["tier"] != 3 else "")
        )

        if total.label == scoring.LABEL_SKIP:
            continue

        candidates.append(Candidate(
            code=code,
            strategy=sig.get("strategy_used", ""),
            signal_strength=strength,
            current_price=sig.get("current_price", 0.0),
            total_score=total.total,
            score_label=total.label,
            score_components={
                "trend": total.trend_score,
                "fundamental": total.fundamental_score,
                "news": total.news_score,
                "weather": total.weather_score,
                "position_scale": total.position_scale,
            },
            raw=sig,
        ))

    return candidates


def rank_candidates(candidates: List[Candidate]) -> List[Candidate]:
    """Ranking Engine stage: sort by total_score, then signal_strength, both
    descending. v1 deliberately stops at these two keys — no breakout_age
    (doesn't exist anywhere in the codebase today) or ATR-direction tie-break
    (would require a product decision on which direction is "better"), so
    this doesn't smuggle in a new business rule to resolve a handful of
    ties. That's left for a real v2.9 design once Confidence Score exists —
    this function is the sole place that upgrade needs to change; Candidate
    Pool and engine/runner.py's execution loop don't need to know how
    candidates got ordered.

    Determinism: `sorted()` is Python's Timsort, which is guaranteed stable —
    when two candidates tie on (total_score, signal_strength) exactly, they
    keep their relative order from the input list, they are never reordered
    by hash/iteration-order noise. `candidates` must stay a plain list
    end-to-end (never routed through a dict keyed by score or a set) for
    that guarantee to mean anything; Candidate Pool only ever appends to a
    list (see build_candidate_pool), so a full tie resolves to Candidate
    Pool's original order — same input, same output, every run."""
    return sorted(candidates, key=lambda c: (c.total_score, c.signal_strength), reverse=True)
