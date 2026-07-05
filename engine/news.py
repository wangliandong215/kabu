"""
engine/news.py — News sentiment filter using moomoo get_search_news().

Two-stage pipeline:
  Stage 1 (local, zero-cost): source trust filtering + circuit-breaker keywords
  Stage 2 (local, zero-cost): keyword sentiment scoring (or FinBERT if installed)

News type routing:
  NOTICE  — Company announcements: highest urgency, hard circuit-breaker
  RATING  — Analyst upgrades/downgrades: medium urgency, adjust signal weight
  NEWS    — Flash news: medium urgency, keyword scored
  (Community/Forum posts are discarded by source filter)

Usage:
  from engine.news import apply_filter, pre_market_scores
  result = apply_filter("US.NVDA", buy_result)   # returns None if blocked
  scores = pre_market_scores(["US.NVDA", "US.CEG"])  # pre-market batch
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import notify.alert as alert

# ── FinBERT availability (checked once at import) ──────────────────────────────
try:
    import transformers  # noqa: F401
    import torch          # noqa: F401
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

alert.info(
    "news: sentiment engine = "
    + ("FinBERT (ProsusAI/finbert)" if FINBERT_AVAILABLE else "keyword fallback (pip install transformers torch to upgrade)")
)

# ── Tuneable ──────────────────────────────────────────────────────────────────
CACHE_TTL         = 4 * 3600   # 4 h — realtime cache
PRE_MKT_CACHE_TTL = 8 * 3600   # 8 h — pre-market batch cache
LOOKBACK_DAYS     = 3
MAX_NEWS          = 20          # headlines per fetch

FILTER_THRESHOLD  = -0.30       # block BUY below this
BOOST_THRESHOLD   = +0.30       # boost signal_strength above this
BOOST_FACTOR      = 1.20

# ── Source trust tiers ────────────────────────────────────────────────────────
# Discard community/forum posts entirely — high noise, often contrarian indicator
_DISCARD_SOURCES = {
    "moomoo community", "moomoo user", "community", "forum",
    "user post", "feed", "social",
}

# High-credibility professional sources — score counted at full weight
_HIGH_TRUST_SOURCES = {
    "reuters", "bloomberg", "associated press", "ap", "dow jones",
    "wall street journal", "wsj", "ft", "financial times",
    "sec", "edgar", "company announcement", "press release",
    "globenewswire", "businesswire", "prnewswire",
    "benzinga", "marketwatch", "barrons", "seeking alpha pro",
    "s&p", "moody", "fitch",
}

# ── Circuit-breaker keywords (immediate hard block regardless of score) ───────
# These indicate fundamental impairment — never buy into these events
_CIRCUIT_BREAKERS = {
    "sec investigation", "sec charges", "securities fraud",
    "class action", "doj investigation", "department of justice",
    "accounting irregularities", "restatement", "going concern",
    "chapter 11", "bankruptcy", "delisted", "delisting",
    "ceo arrested", "cfo arrested", "executive arrested",
    "fda rejection", "fda clinical hold", "complete response letter",
    "product recall", "safety recall",
    "material weakness", "internal controls failure",
}

# ── High-signal keyword pre-filter (only score news containing these) ─────────
_SIGNAL_KEYWORDS = {
    # Earnings
    "earnings", "eps", "revenue", "guidance", "forecast", "outlook",
    "beat", "miss", "raised", "lowered", "cut",
    # Ratings
    "upgrade", "downgrade", "price target", "buy", "sell", "hold",
    "overweight", "underweight", "outperform", "underperform",
    # Corporate events
    "acquisition", "merger", "buyback", "dividend", "spin-off",
    "ipo", "secondary offering", "dilution",
    # Risk events
    "investigation", "lawsuit", "recall", "fine", "penalty",
    "layoff", "restructuring", "bankruptcy",
    # FDA / regulatory
    "fda", "approval", "approved", "rejection", "rejected",
    "clinical trial", "phase", "regulatory",
}

# ── Sentiment keywords ────────────────────────────────────────────────────────
_POS_WORDS = {
    "beat", "beats", "exceeded", "surpassed", "record", "topped",
    "raised guidance", "upgraded", "upgrade", "strong buy", "overweight",
    "outperform", "surge", "rally", "approved", "approval",
    "acquisition", "buyback", "dividend increase", "profit", "growth",
    "partnership", "contract win", "positive", "bullish",
}
_NEG_WORDS = {
    "miss", "missed", "below expectations", "lowered guidance",
    "cut guidance", "downgraded", "downgrade", "underperform", "underweight",
    "sell", "plunge", "decline", "dropped", "loss", "losses",
    "investigation", "lawsuit", "recall", "fine", "fined", "penalty",
    "layoffs", "restructuring", "bankruptcy", "negative", "bearish",
    "rejected", "rejection", "failed", "failure", "warning",
}

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: Dict[str, dict] = {}


# ── Public API ────────────────────────────────────────────────────────────────

def macro_circuit_breaker() -> str:
    """
    Check macro-level news (QQQ / SPY) for systemic black-swan events.
    Returns triggering phrase if market-wide halt is warranted, else "".
    Should be called ONCE per pass (not per stock) and result cached.
    """
    for index in ("QQQ", "SPY"):
        headlines = _fetch(f"US.{index}")
        reason = check_circuit_breaker(headlines)
        if reason:
            return f"MACRO:{index} — {reason}"
    return ""


def apply_filter(code: str, result: dict) -> Optional[dict]:
    """
    Apply news filter to a BUY signal.

    Returns:
      None          — hard block (negative news / circuit breaker)
      result dict   — with signal_strength adjusted and news_score added
    """
    score, blocked_reason = _get_score_and_block(code)
    ticker = _ticker(code)

    if blocked_reason:
        alert.warn(f"news: {ticker:8s}  CIRCUIT BREAKER — {blocked_reason}")
        return None

    if score < FILTER_THRESHOLD:
        alert.warn(f"news: {ticker:8s}  score={score:+.2f}  BLOCKED (negative sentiment)")
        return None

    if score > BOOST_THRESHOLD:
        old = result.get("signal_strength", 0.0)
        result["signal_strength"] = min(1.0, old * BOOST_FACTOR)
        alert.info(
            f"news: {ticker:8s}  score={score:+.2f}  BOOST "
            f"{old:.0%} -> {result['signal_strength']:.0%}"
        )
    else:
        alert.info(f"news: {ticker:8s}  score={score:+.2f}  neutral — proceed")

    result["news_score"] = round(score, 3)
    return result


def pre_market_scores(codes: List[str]) -> Dict[str, float]:
    """
    Batch pre-market news scoring (run once before session open).
    Returns {code: score} for all codes.
    Useful for adjusting watchlist priority before the trading day starts.
    """
    scores = {}
    for code in codes:
        score, _ = _get_score_and_block(code, ttl=PRE_MKT_CACHE_TTL)
        scores[code] = score
        time.sleep(3.5)   # moomoo limit: 10 req / 30 s
    return scores


# ── Internal ──────────────────────────────────────────────────────────────────

def _ticker(code: str) -> str:
    return code.split(".")[-1] if "." in code else code


def _get_score_and_block(
    code: str, ttl: int = CACHE_TTL
) -> Tuple[float, str]:
    """Return (sentiment_score, circuit_breaker_reason_or_empty)."""
    now = time.time()
    cached = _cache.get(code)
    if cached and now - cached["ts"] < ttl:
        return cached["score"], cached.get("blocked", "")

    # _fetch returns ALL source-filtered headlines (no keyword pre-filter).
    # Circuit breaker must see every headline — a "chapter 11" title would
    # otherwise be silently dropped by the keyword filter before we could catch it.
    all_headlines = _fetch(code)
    blocked       = check_circuit_breaker(all_headlines)

    if blocked:
        score = 0.0
    else:
        # Score only the subset that contains high-signal keywords.
        signal_headlines = filter_signal_keywords(all_headlines)
        score = score_headlines(signal_headlines)

    _cache[code] = {"score": score, "blocked": blocked, "ts": now}
    return score, blocked


def _fetch(code: str) -> List[dict]:
    """
    Fetch recent news from moomoo OpenD.
    Returns ALL source-filtered, in-recency headlines as
    {"title": str, "source": str, "sub_type": str}.

    NOTE: keyword pre-filtering is intentionally NOT done here.
    Circuit breaker must see every headline before keywords narrow the set.
    Keyword filtering for scoring is done separately in filter_signal_keywords().
    """
    import moomoo as ft
    ticker  = _ticker(code)
    cutoff  = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    results = []

    try:
        ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)
        try:
            ret, data = ctx.get_search_news(
                ticker, MAX_NEWS, news_sub_type=ft.NewsSubType.ALL
            )
            if ret != ft.RET_OK or data is None or data.empty:
                return []

            for _, row in data.iterrows():
                source   = str(row.get("source", "")).lower().strip()
                sub_type = str(row.get("news_sub_type", "")).lower()
                title    = str(row.get("title", "")).lower().strip()
                pub      = str(row.get("publish_time", ""))

                # Discard community/forum posts — high noise, no signal value
                if any(d in source for d in _DISCARD_SOURCES):
                    continue

                # Recency filter: skip headlines older than LOOKBACK_DAYS
                try:
                    pub_dt = datetime.strptime(pub[:19], "%Y-%m-%d %H:%M:%S")
                    if pub_dt < cutoff:
                        continue
                except ValueError:
                    pass

                results.append({
                    "title":    title,
                    "source":   source,
                    "sub_type": sub_type,
                })
        finally:
            ctx.close()
    except Exception as exc:
        alert.warn(f"news: fetch error for {_ticker(code)} — {exc}")

    return results


def filter_signal_keywords(headlines: List[dict]) -> List[dict]:
    """Keep only headlines that contain at least one high-signal keyword."""
    return [h for h in headlines if any(kw in h["title"] for kw in _SIGNAL_KEYWORDS)]


def check_circuit_breaker(headlines: List[dict]) -> str:
    """
    Scan all headlines (including NOTICE type) for circuit-breaker phrases.
    Returns the triggering phrase if found, else empty string.
    """
    for h in headlines:
        title = h["title"]
        for phrase in _CIRCUIT_BREAKERS:
            if phrase in title:
                return phrase
    return ""


def score_headlines(headlines: List[dict]) -> float:
    """
    Two-stage scoring:
      1. Try FinBERT if transformers is installed (accurate, local, fast).
      2. Fall back to keyword scoring (zero dependencies).

    Source weight: high-trust source counts 2×, unknown source counts 1×.
    """
    if not headlines:
        return 0.0

    # Try FinBERT first (optional upgrade — pip install transformers torch)
    finbert_score = _try_finbert([h["title"] for h in headlines])
    if finbert_score is not None:
        return finbert_score

    # Keyword scoring with source weighting
    pos = neg = 0.0
    for h in headlines:
        title  = h["title"]
        source = h["source"]
        weight = 2.0 if any(t in source for t in _HIGH_TRUST_SOURCES) else 1.0

        for w in _POS_WORDS:
            if w in title:
                pos += weight
        for w in _NEG_WORDS:
            if w in title:
                neg += weight

    total = pos + neg
    return 0.0 if total == 0 else max(-1.0, min(1.0, (pos - neg) / total))


def _try_finbert(texts: List[str]) -> Optional[float]:
    """
    Run FinBERT locally if transformers is installed.
    Returns average sentiment score in [-1, +1], or None if unavailable.

    To enable: pip install transformers torch
    Model downloads automatically on first use (~400 MB).
    """
    try:
        from transformers import pipeline   # type: ignore
        import engine._finbert_cache as _fb_cache
    except (ImportError, ModuleNotFoundError):
        return None

    try:
        pipe  = _fb_cache.get_pipeline()
        total = 0.0
        for text in texts[:10]:   # cap at 10 to limit latency
            out   = pipe(text[:512])[0]
            label = out["label"].lower()
            conf  = out["score"]
            if label == "positive":
                total += conf
            elif label == "negative":
                total -= conf
        return max(-1.0, min(1.0, total / max(len(texts), 1)))
    except Exception:
        return None
