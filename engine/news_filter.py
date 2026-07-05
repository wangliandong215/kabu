"""
engine/news_filter.py — Tiered news/event risk grading (0 / 50 / 100).

Per spec (v2.1 多因子总分模型): a hard-coded three-tier classifier, distinct
from the continuous -1..+1 sentiment score in engine/news.py. This is a
stricter, rule-based risk grade that feeds the "新闻评分(20%)" component of
engine/scoring.py::compute_total_score(), not a standalone chain filter:

  Tier 1 (score=0)   — 归零黑天鹅: financial fraud, delisting risk, short-seller
                        report, SEC investigation, major lawsuit, chairman/CEO
                        arrested or abrupt resignation. Hard veto — no new
                        position regardless of technical signal.
  Tier 2 (score=50)  — 预期差恶化: earnings miss, analyst downgrade, guidance
                        lowered. Technical signal still allowed to fire, but
                        contributes only 50/100 to the news component of the
                        total score (see engine/scoring.py for how that
                        rolls up into the final position_scale).
  Tier 3 (score=100) — 正常/正面: product launch, normal partnership, earnings
                        beat, in-line results, routine media coverage. No
                        interference — full news score.

NOTE ON DATA / BACKTEST vs PAPER/LIVE: this module only classifies headline
text you hand it. It does NOT fetch or source news itself. There is no real
historical news feed covering 2015-2026 for the 142-stock pool (moomoo's
get_search_news() has no date-range parameter — see
engine/news_backtest.py's docstring for the full explanation), so in
env="backtest" the news component is fixed at a neutral 70 (see
engine/scoring.py), never derived from this module. In env="paper"/"live",
engine/runner.py feeds this module the real headlines already fetched by
engine/news.py's _fetch() (no duplicate API call) via classify_headlines().
"""
import time
from typing import Dict, List, Optional, Tuple

TIER1_SCORE = 0
TIER2_SCORE = 50
TIER3_SCORE = 100

ACTION_BLOCK    = "BLOCK"
ACTION_REDUCE   = "REDUCE_50"
ACTION_NORMAL   = "NORMAL"

TIER2_POSITION_SCALE = 0.5

# ── Tier 1 — 归零黑天鹅 (hard veto, no open) ───────────────────────────────────
_TIER1_KEYWORDS = {
    # English
    "accounting fraud", "financial fraud", "cooking the books",
    "delisting risk", "risk of delisting", "delisted", "delisting",
    "short seller report", "short-seller report", "short report",
    "sec investigation", "sec probe", "securities fraud",
    "class action lawsuit", "class-action lawsuit", "major lawsuit",
    "ceo arrested", "cfo arrested", "chairman arrested", "executive arrested",
    "ceo resigns abruptly", "ceo abruptly resigns", "sudden resignation of ceo",
    "chairman resigns abruptly", "chairman abruptly resigns",
    "ceo steps down abruptly", "sudden ceo departure", "abrupt resignation",
    # Chinese
    "财务造假", "退市风险", "做空报告", "sec调查", "重大诉讼",
    "董事长被捕", "ceo被捕", "总裁被捕", "突然辞职", "闪电辞职",
}

# ── Tier 2 — 预期差恶化 (allowed, forced 50% size cut) ─────────────────────────
_TIER2_KEYWORDS = {
    # English
    "earnings miss", "missed earnings", "misses estimates",
    "below expectations", "fell short of expectations",
    "analyst downgrade", "downgraded", "downgrades",
    "guidance lowered", "lowered guidance", "cuts guidance",
    "guidance cut", "trims guidance", "cuts outlook", "lowered outlook",
    # Chinese
    "财报低于预期", "低于预期", "分析师下调评级", "下调评级",
    "盈利指引下修", "下调指引", "下修指引",
}

# ── Tier 3 — 正常/正面 (no interference) ───────────────────────────────────────
# Informational only — not matched against; anything that isn't Tier 1/2 is
# Tier 3 by default (covers "new product launch / normal partnership /
# earnings beat / in line with expectations / routine media coverage" and
# everything else not flagged as risk).
_TIER3_KEYWORDS = {
    "new product launch", "product launch", "unveils", "announces partnership",
    "earnings beat", "beats estimates", "tops estimates",
    "in line with expectations", "meets expectations",
    "新产品发布", "正常合作", "财报超预期", "符合预期", "一般媒体报道",
}


def classify_headline(title: str) -> int:
    """Classify a single headline. Priority: Tier 1 > Tier 2 > Tier 3 (default)."""
    t = title.lower()
    if any(kw in t for kw in _TIER1_KEYWORDS):
        return 1
    if any(kw in t for kw in _TIER2_KEYWORDS):
        return 2
    return 3


def classify_headlines(titles: List[str]) -> Tuple[int, Optional[str]]:
    """
    Classify a batch of headlines (e.g. every headline in the lookback window
    for one stock as of one trading day). Worst tier wins — a single Tier-1
    headline vetoes the whole batch even if nine other headlines are Tier-3.

    Returns (worst_tier, matched_keyword_or_None). Empty input -> (3, None).
    """
    if not titles:
        return 3, None

    worst_tier: int = 3
    worst_match: Optional[str] = None
    for title in titles:
        t = title.lower()
        for kw in _TIER1_KEYWORDS:
            if kw in t:
                return 1, kw   # nothing can outrank Tier 1 — short-circuit
        if worst_tier > 2:
            for kw in _TIER2_KEYWORDS:
                if kw in t:
                    worst_tier, worst_match = 2, kw
                    break
    return worst_tier, worst_match


def grade(titles: List[str]) -> dict:
    """
    Top-level entry point: given the headlines relevant to a stock as of a
    given day, return the risk-grade decision.

    Returns:
      {"tier": 1|2|3, "score": 0|50|100, "action": "BLOCK"|"REDUCE_50"|"NORMAL",
       "position_scale": 0.0|0.5|1.0, "matched_keyword": str|None}
    """
    tier, matched = classify_headlines(titles)
    if tier == 1:
        return {"tier": 1, "score": TIER1_SCORE, "action": ACTION_BLOCK,
                 "position_scale": 0.0, "matched_keyword": matched}
    if tier == 2:
        return {"tier": 2, "score": TIER2_SCORE, "action": ACTION_REDUCE,
                 "position_scale": TIER2_POSITION_SCALE, "matched_keyword": matched}
    return {"tier": 3, "score": TIER3_SCORE, "action": ACTION_NORMAL,
             "position_scale": 1.0, "matched_keyword": None}


# ── env="paper"/"live" 专用：拉真实新闻 + 分类 + 缓存 ───────────────────────────
_CACHE_TTL = 4 * 3600   # 跟 engine.news 自己的情感缓存 TTL 保持一致
_cache: Dict[str, dict] = {}


def classify_code(code: str) -> dict:
    """
    拉这只票最近的真实 headlines（复用 engine.news._fetch，跟实盘连续情感
    打分共享同一次抓取窗口/来源过滤，不重复发一次API请求）并跑三级分类。
    4小时缓存，避免 engine/runner.py 的评分流水线把每次 pass 的新闻API
    请求量翻倍（moomoo 新闻接口本身有限流）。

    只应该在 env="paper"/"live" 时调用——env="backtest" 应直接用固定中性分
    （见 engine/scoring.py 调用方），不应该走到这里。
    """
    now = time.time()
    cached = _cache.get(code)
    if cached and now - cached["ts"] < _CACHE_TTL:
        return cached["result"]

    from engine.news import _fetch   # 延迟导入，避免模块加载期的循环依赖

    headlines = _fetch(code)
    titles = [h["title"] for h in headlines]
    result = grade(titles)
    _cache[code] = {"result": result, "ts": now}
    return result
