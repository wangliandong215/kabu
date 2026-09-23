# -*- coding: utf-8 -*-
"""
engine/confidence_score.py — V2.9 Confidence Score (信号质量评分), Phase 1:
Observe / Record / Validate ONLY.

Hard constraint (per product decision, V2.9 spec section 三/十二): this module
is a pure research/observation layer. Nothing it computes may gate, scale, or
otherwise influence a BUY/SELL decision, position sizing, exposure budget, or
any existing risk gate. engine/pipeline.py::build_candidate_pool() attaches
this module's output to Candidate.confidence / Candidate.confidence_detail
purely for logging — rank_candidates() (the only consumer that decides
execution order) sorts on (total_score, signal_strength) alone and never
reads .confidence. See that module's docstring for the ON/OFF equivalence
this relies on.

Two independent halves, kept separate so the scoring math stays pure/
testable and the only side-effecting part (reading trade_history.db for
historical stats) is isolated to one small function:

  compute_confidence_score()  — pure, deterministic, side-effect-free.
      Takes already-known-at-signal-time numbers in, returns a
      ConfidenceScoreResult. No I/O, no imports of engine.trade_tracker/
      engine.regime_store/engine.market_context. Safe to unit test with
      plain floats.

  gather_and_score()          — the one orchestration function that actually
      touches the filesystem/DB (engine.regime_store's local JSON,
      engine.market_context's local SQLite read, TradeTracker's local SQLite
      read for historical win-rate/expectancy). Every I/O call is
      independently try/except-guarded so one missing/broken source degrades
      only its own component to None, never aborts the whole score (same
      discipline as engine/market_context.py's per-source _resolve()).
      Called once per BUY candidate, at signal time — see
      engine/pipeline.py::build_candidate_pool().

No look-ahead / no data leakage (V2.9 spec section 七): historical_win_rate
and historical_expectancy_pct in gather_and_score() are computed ONLY from
trades whose exit_time is already in the past relative to `as_of` (defaults
to "now" — this module is never used to replay a backtest, only live/paper
signal time, so "already closed" and "already known" are the same thing
here). BOOTSTRAP_TRADE_IDS below are excluded from that calculation — see
their docstring.

Formula (v2.9.0, first cut — fixed BEFORE any sample is collected, per spec
section 八/九: weights and thresholds below are principled, documented
defaults, NOT fitted to the 33 historical trades already in trade_history.db.
They may be recalibrated once a real sample (500+ trades or 1-3 months) lets
the V2.9.x validation phase judge them statistically, but that recalibration
is out of scope for Phase 1 and requires a separate product decision):

    confidence_score = 100 * weighted_average(
        rule_based_component   (weight 0.30, always present),
        hmm_component           (weight 0.15),
        win_rate_component      (weight 0.15),
        expectancy_component    (weight 0.15),
        market_component        (weight 0.10),
        volatility_component    (weight 0.10),
        volume_component        (weight 0.05, always None in Phase 1),
    )

Any component that is None (missing input, e.g. no trained HMM model for
this code yet, or fewer than MIN_HISTORICAL_SAMPLE closed trades so far) is
dropped from both the numerator and the weight total before dividing — same
"exclude and renormalize, never fabricate a placeholder value" rule
engine/scoring.py already uses for fundamental_score/news_score. Every
component is independently clipped to [0, 100] before weighting, so the
final confidence_score is always in [0, 100] by construction.
"""
import json
from datetime import datetime
from typing import NamedTuple, Optional

# ── Formula version — bump this string any time a weight/threshold/mapping
# below changes, so trade_confidence_score rows can be grouped by which
# formula produced them (see engine/trade_tracker.py's log_confidence_score).
FORMULA_VERSION = "v2.9.0"

# ── Weights (sum to 1.0 across ALL seven components; missing ones are
# excluded and the remainder renormalized — see module docstring).
W_RULE        = 0.30
W_HMM         = 0.15
W_WIN_RATE    = 0.15
W_EXPECTANCY  = 0.15
W_MARKET      = 0.10
W_VOLATILITY  = 0.10
W_VOLUME      = 0.05

# ── HMM State component ───────────────────────────────────────────────────
# Base "favorability" score per regime label (engine.trade_tracker's
# build_regime_ctx_preferring_hmm() regime_label values), blended toward a
# neutral 50 by (1 - regime_confidence) — a low-confidence regime read pulls
# the component toward "no opinion" rather than fully trusting a shaky label.
_HMM_BASE_SCORE = {
    "HMM_BULL":       90.0,
    "HMM_SIDEWAYS":   50.0,
    "HMM_CORRECTION": 30.0,
    "HMM_BEAR":       10.0,
}

# ── Historical Expectancy component ───────────────────────────────────────
# historical_expectancy_pct is mean(pnl_pct) of past closed trades (a
# fraction, e.g. 0.03 = +3%). Clamped to +-5% and linearly mapped to
# [0, 100] — 5% is a generous multiple of this system's typical per-trade
# swing (see risk/sizing.py's RISK_PER_TRADE_PCT), chosen to keep the mapping
# from saturating on ordinary results while still bounding pathological ones.
EXPECTANCY_CLAMP_PCT = 0.05

# ── Historical sample-size gate ───────────────────────────────────────────
# Below this many qualifying closed trades, historical_win_rate/expectancy
# are reported as None (excluded from the weighted average) rather than as a
# noisy statistic from a handful of trades — this is exactly the "current
# sample is far too small to validate anything" state the V2.9 spec (section
# ten) says the system currently is in. Deliberately small (5, not e.g. 30)
# so a real signal starts contributing to the weighted average as soon as
# there is at least a token amount of history, while single-digit-trade
# noise (1-2 trades, all-or-nothing win rate) still stays excluded.
MIN_HISTORICAL_SAMPLE = 5

# ── Market Statistics component ───────────────────────────────────────────
# VIX fallback (used only when CNN Fear & Greed is unavailable — see
# engine/market_context.py's fetch_cnn_fear_greed/fetch_vix): VIX 10-40
# linearly inverted to 100-0. 10/40 bracket VIX's typical calm-to-stressed
# range without being a genuine historical extreme (VIX has spiked well
# above 40 in real crises) — chosen as a sane default band, not fitted to
# any trade outcome.
VIX_LO, VIX_HI = 10.0, 40.0

# ── Volatility component ──────────────────────────────────────────────────
# atr_pct = ATR / current_price at signal time. Full score (100) inside the
# "normal, tradeable" band [1.5%, 4%]; linearly tapering to 0 outside the
# wider [0.5%, 8%] band. This is a plain sanity-band heuristic (neither
# "higher is better" nor "lower is better" — both a dead-quiet stock and a
# blown-out one are penalized) — not derived from this system's own
# win/loss history, per spec section 八's "don't reverse-engineer from past
# results" constraint.
ATR_OUTER_LO, ATR_BAND_LO = 0.005, 0.015
ATR_BAND_HI, ATR_OUTER_HI = 0.04, 0.08

# ── Bootstrap exclusion (V2.9 spec section 九) ────────────────────────────
# The one-time 2026-07-06/07 initial-position batch (QQQ core position +
# VRSK/ADP/PAYX/TTWO/AMGN/GILD/VRTX/REGN/FAST) was NOT produced by a
# strategy signal — it must never contribute to historical_win_rate/
# historical_expectancy_pct. Includes both the 23:32 orphan-duplicate batch
# (never got an exit_time — see project memory) and the 00:28 batch that
# actually carries real exit/pnl data; both are bootstrap, not signal
# trades, regardless of whether they later closed. Hardcoded exact trade_ids
# (not a ticker/date-range filter) because every one of these tickers was
# later legitimately re-bought by a real strategy signal (e.g. TTWO on
# 2026-08-12) — a ticker or date-range filter would incorrectly exclude
# those real trades too. Frozen list: this batch happened exactly once and
# will never recur, so this constant never needs to grow.
BOOTSTRAP_TRADE_IDS = frozenset({
    "US.QQQ_2026-07-06T22:46:06.692798",
    "US.VRSK_2026-07-06T23:32:32.698460",
    "US.ADP_2026-07-06T23:32:35.537358",
    "US.PAYX_2026-07-06T23:32:38.346525",
    "US.TTWO_2026-07-06T23:32:41.210274",
    "US.AMGN_2026-07-06T23:32:43.971932",
    "US.GILD_2026-07-06T23:32:46.931143",
    "US.VRTX_2026-07-06T23:32:49.745043",
    "US.REGN_2026-07-06T23:32:52.640823",
    "US.FAST_2026-07-06T23:32:55.399854",
    "US.VRSK_2026-07-07T00:28:32.071088",
    "US.ADP_2026-07-07T00:28:34.937572",
    "US.PAYX_2026-07-07T00:28:38.051837",
    "US.TTWO_2026-07-07T00:28:41.008868",
    "US.AMGN_2026-07-07T00:28:43.977331",
    "US.GILD_2026-07-07T00:28:46.964432",
    "US.VRTX_2026-07-07T00:28:49.825197",
    "US.REGN_2026-07-07T00:28:52.899565",
    "US.FAST_2026-07-07T00:28:55.810827",
})


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


# ── Per-component mappers — each Optional[float] in, Optional[float] in
# [0, 100] out. None in -> None out (missing data is never fabricated). ────

def _hmm_component(hmm_state: Optional[str], hmm_confidence: Optional[float]) -> Optional[float]:
    if hmm_state is None or hmm_confidence is None:
        return None
    base = _HMM_BASE_SCORE.get(hmm_state)
    if base is None:
        return None
    weight = _clip(hmm_confidence, 0.0, 100.0) / 100.0
    return _clip(50.0 + (base - 50.0) * weight)


def _win_rate_component(historical_win_rate: Optional[float],
                         historical_win_rate_n: int) -> Optional[float]:
    if historical_win_rate is None or historical_win_rate_n < MIN_HISTORICAL_SAMPLE:
        return None
    return _clip(historical_win_rate * 100.0)


def _expectancy_component(historical_expectancy_pct: Optional[float],
                           historical_win_rate_n: int) -> Optional[float]:
    if historical_expectancy_pct is None or historical_win_rate_n < MIN_HISTORICAL_SAMPLE:
        return None
    clamped = max(-EXPECTANCY_CLAMP_PCT, min(EXPECTANCY_CLAMP_PCT, historical_expectancy_pct))
    return _clip((clamped + EXPECTANCY_CLAMP_PCT) / (2 * EXPECTANCY_CLAMP_PCT) * 100.0)


def _market_component(market_cnn_fear_greed: Optional[float],
                       market_vix_close: Optional[float]) -> Optional[float]:
    if market_cnn_fear_greed is not None:
        return _clip(market_cnn_fear_greed)
    if market_vix_close is not None:
        return _clip((VIX_HI - market_vix_close) / (VIX_HI - VIX_LO) * 100.0)
    return None


def _volatility_component(volatility_atr_pct: Optional[float]) -> Optional[float]:
    if volatility_atr_pct is None or volatility_atr_pct < 0:
        return None
    v = volatility_atr_pct
    if ATR_BAND_LO <= v <= ATR_BAND_HI:
        return 100.0
    if v < ATR_BAND_LO:
        if v <= ATR_OUTER_LO:
            return 0.0
        return _clip((v - ATR_OUTER_LO) / (ATR_BAND_LO - ATR_OUTER_LO) * 100.0)
    if v >= ATR_OUTER_HI:
        return 0.0
    return _clip((ATR_OUTER_HI - v) / (ATR_OUTER_HI - ATR_BAND_HI) * 100.0)


def _volume_component(volume_feature: Optional[float]) -> Optional[float]:
    # Phase 1: no volume feature is wired up yet (V2.9 spec section 四) — this
    # always returns None so the component is excluded/renormalized, never
    # fabricated. Kept as a real branch (not inlined at the call site) so
    # wiring up a real volume_feature later is a one-line change here.
    if volume_feature is None:
        return None
    return _clip(volume_feature)


class ConfidenceScoreResult(NamedTuple):
    confidence_score: float
    rule_component: float
    hmm_component: Optional[float]
    win_rate_component: Optional[float]
    expectancy_component: Optional[float]
    market_component: Optional[float]
    volatility_component: Optional[float]
    volume_component: Optional[float]
    weights_used: dict


def compute_confidence_score(
    rule_based_score: float,
    hmm_state: Optional[str] = None,
    hmm_confidence: Optional[float] = None,
    historical_win_rate: Optional[float] = None,
    historical_win_rate_n: int = 0,
    historical_expectancy_pct: Optional[float] = None,
    market_cnn_fear_greed: Optional[float] = None,
    market_vix_close: Optional[float] = None,
    volatility_atr_pct: Optional[float] = None,
    volume_feature: Optional[float] = None,
) -> ConfidenceScoreResult:
    """Pure, deterministic, side-effect-free. See module docstring for the
    formula. rule_based_score is required (this is engine.scoring.
    compute_total_score()'s `total`, already 0-100, always available for any
    signal that reached Candidate Pool). Every other input is optional —
    None means "not available for this signal", excluded from the weighted
    average rather than defaulted to a placeholder value.

    Never raises on well-typed input; out-of-range inputs are clipped, not
    rejected, so a caller passing e.g. historical_win_rate=1.2 by mistake
    gets a clipped 100 rather than a corrupted downstream average."""
    rule_comp = _clip(rule_based_score)
    hmm_comp = _hmm_component(hmm_state, hmm_confidence)
    win_comp = _win_rate_component(historical_win_rate, historical_win_rate_n)
    exp_comp = _expectancy_component(historical_expectancy_pct, historical_win_rate_n)
    mkt_comp = _market_component(market_cnn_fear_greed, market_vix_close)
    vol_comp = _volatility_component(volatility_atr_pct)
    volu_comp = _volume_component(volume_feature)

    weighted = [
        (rule_comp, W_RULE, "rule"),
        (hmm_comp, W_HMM, "hmm"),
        (win_comp, W_WIN_RATE, "win_rate"),
        (exp_comp, W_EXPECTANCY, "expectancy"),
        (mkt_comp, W_MARKET, "market"),
        (vol_comp, W_VOLATILITY, "volatility"),
        (volu_comp, W_VOLUME, "volume"),
    ]

    numerator = sum(comp * w for comp, w, _ in weighted if comp is not None)
    total_weight = sum(w for comp, w, _ in weighted if comp is not None)
    weights_used = {name: w for comp, w, name in weighted if comp is not None}

    # total_weight can never be 0 — rule_comp is always present (required
    # arg), so this branch is unreachable in practice; kept as a hard
    # fallback (not a fabricated score) rather than dividing by zero.
    score = _clip(numerator / total_weight) if total_weight > 0 else rule_comp

    return ConfidenceScoreResult(
        confidence_score=round(score, 2),
        rule_component=rule_comp,
        hmm_component=hmm_comp,
        win_rate_component=win_comp,
        expectancy_component=exp_comp,
        market_component=mkt_comp,
        volatility_component=vol_comp,
        volume_component=volu_comp,
        weights_used=weights_used,
    )


def historical_performance(tracker, as_of: Optional[str] = None,
                            execution: str = "REAL"):
    """Reads trade_history.db (via an already-open TradeTracker) for
    historical win rate / expectancy, computed ONLY from trades that were
    already closed strictly before `as_of` (defaults to now — see module
    docstring's no-look-ahead note) and excluding BOOTSTRAP_TRADE_IDS.

    execution="REAL" matches trades.execution='REAL' OR NULL (legacy rows
    predating the v2.9 execution column, which are real US trades that
    simply predate that column — see project trade_history.db audit; NOT
    matching them would silently shrink an already-small sample).
    execution="PAPER" matches trades.execution='PAPER' only (the JP paper
    pass's own virtual trades), kept fully separate so paper fills never
    contaminate the real-money historical stats or vice versa.

    Returns (win_rate: Optional[float] in [0,1], expectancy_pct: Optional[float],
    n: int). (None, None, 0) on any failure (no tracker, empty table, query
    error) or when n < MIN_HISTORICAL_SAMPLE — never raises."""
    if tracker is None:
        return None, None, 0
    try:
        as_of = as_of or datetime.now().isoformat()
        df = tracker.query_trades(execution=None)   # filter ourselves, see docstring
        if df.empty:
            return None, None, 0

        closed = df[df["exit_time"].notna() & (df["exit_time"] < as_of)]
        closed = closed[~closed["trade_id"].isin(BOOTSTRAP_TRADE_IDS)]
        if execution == "PAPER":
            closed = closed[closed["execution"] == "PAPER"]
        else:
            closed = closed[closed["execution"] != "PAPER"]

        n = len(closed)
        if n < MIN_HISTORICAL_SAMPLE:
            return None, None, n

        pnl_pct = closed["pnl_pct"].dropna()
        if pnl_pct.empty:
            return None, None, n

        win_rate = float((pnl_pct > 0).mean())
        expectancy_pct = float(pnl_pct.mean())
        return win_rate, expectancy_pct, n
    except Exception:
        return None, None, 0


def gather_and_score(code: str, rule_based_score: float, sig: dict,
                      tracker=None, execution: str = "REAL",
                      as_of: Optional[str] = None) -> ConfidenceScoreResult:
    """Orchestration: gather every input compute_confidence_score() needs
    from its live source, then call it. Every source is independently
    try/except-guarded (mirrors engine/market_context.py's per-source
    degrade-to-None discipline) so one unavailable source never blanks out
    the others. Never raises — callers (engine/pipeline.py) still wrap this
    in their own try/except as defense in depth, matching every other
    observation hook in this codebase.

    `sig` is the raw signal dict (Candidate.raw / the strategy result), used
    only for atr/current_price (volatility) — already computed by the
    strategy for this exact signal, no extra fetch."""
    hmm_state = hmm_confidence = None
    try:
        from engine import regime_store
        interface = regime_store.get_regime_interface(code)
        label = interface.get("current_regime")
        if label:
            hmm_state = f"HMM_{label.upper()}"
            hmm_confidence = interface.get("regime_confidence")
    except Exception:
        pass

    win_rate, expectancy_pct, n = historical_performance(tracker, as_of=as_of, execution=execution)

    cnn_fear_greed = vix_close = None
    try:
        from engine import market_context
        ctx = market_context.get_latest_context()
        if ctx:
            cnn_fear_greed = ctx.get("cnn_fear_greed")
            vix_close = ctx.get("vix_close")
    except Exception:
        pass

    atr_pct = None
    try:
        atr = sig.get("atr")
        price = sig.get("current_price")
        if atr and price:
            atr_pct = float(atr) / float(price)
    except Exception:
        pass

    result = compute_confidence_score(
        rule_based_score=rule_based_score,
        hmm_state=hmm_state,
        hmm_confidence=hmm_confidence,
        historical_win_rate=win_rate,
        historical_win_rate_n=n,
        historical_expectancy_pct=expectancy_pct,
        market_cnn_fear_greed=cnn_fear_greed,
        market_vix_close=vix_close,
        volatility_atr_pct=atr_pct,
        volume_feature=None,   # Phase 1 — always unavailable, see module docstring
    )

    detail = {
        "formula_version": FORMULA_VERSION,
        "rule_based_score": rule_based_score,
        "hmm_state": hmm_state,
        "hmm_confidence": hmm_confidence,
        "hmm_component": result.hmm_component,
        "historical_win_rate": win_rate,
        "historical_win_rate_n": n,
        "historical_expectancy_pct": expectancy_pct,
        "market_cnn_fear_greed": cnn_fear_greed,
        "market_vix_close": vix_close,
        "market_component": result.market_component,
        "volatility_atr_pct": atr_pct,
        "volatility_component": result.volatility_component,
        "volume_feature": None,
        "confidence_score": result.confidence_score,
        "weights_json": json.dumps(result.weights_used),
    }
    return result, detail
