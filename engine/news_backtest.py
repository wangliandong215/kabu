"""
engine/news_backtest.py — Point-in-time news sentiment for backtesting.

WHY THIS EXISTS
  moomoo's get_search_news() has no date-range parameter — it only returns
  the *current* most-recent headlines (see engine/news.py::_fetch). There is
  no way to ask "what was the news for NVDA on 2024-03-15" through moomoo's
  API. Historical, point-in-time news is therefore NOT available from moomoo
  at all — plugging live news into a backtest over past dates would silently
  leak future information (today's headlines applied to yesterday's prices).

  This module does NOT fabricate that data. It loads headlines you supply
  from an external source (a news API export, a scrape, a vendor feed) and
  applies the exact same scoring logic as live trading (engine.news.
  score_headlines / check_circuit_breaker), restricted to headlines whose
  publish date is on or before the simulated trading day. If you don't have
  a historical news file, --news-file is simply omitted and the backtest
  runs exactly as before (no news filter applied).

CSV FORMAT (required columns, header row required):
  date,code,title,source
  2024-03-15,US.NVDA,"nvidia beats earnings estimates",reuters
  2024-03-15,US.NVDA,"analyst raises price target",benzinga

  - date   : headline publish date, YYYY-MM-DD (interpreted as that day's close)
  - code   : ticker in moomoo format, e.g. US.NVDA
  - title  : headline text (lowercased automatically)
  - source : publisher name (used for trust-weighting in keyword fallback)

Usage:
  from engine.news_backtest import HistoricalNewsFeed
  feed = HistoricalNewsFeed.load("news_history.csv")
  score, blocked = feed.score_asof("US.NVDA", pd.Timestamp("2024-03-15"))
"""
import csv
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from engine.news import (
    check_circuit_breaker,
    filter_signal_keywords,
    score_headlines,
    LOOKBACK_DAYS,
)


class HistoricalNewsFeed:
    """In-memory index of {code: [(date, headline_dict), ...]} for backtest lookups."""

    def __init__(self, headlines_by_code: Dict[str, List[Tuple[pd.Timestamp, dict]]]):
        self._by_code = headlines_by_code

    @classmethod
    def load(cls, path: str) -> "HistoricalNewsFeed":
        by_code: Dict[str, List[Tuple[pd.Timestamp, dict]]] = defaultdict(list)
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"date", "code", "title", "source"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"news-file missing required columns {required}, "
                    f"got {reader.fieldnames}"
                )
            n = 0
            for row in reader:
                try:
                    dt = pd.Timestamp(row["date"])
                except (ValueError, TypeError):
                    continue
                code = row["code"].strip()
                by_code[code].append((dt, {
                    "title":  row["title"].strip().lower(),
                    "source": row["source"].strip().lower(),
                    "sub_type": "news",
                }))
                n += 1
        for code in by_code:
            by_code[code].sort(key=lambda x: x[0])
        print(f"news_backtest: loaded {n} headlines across {len(by_code)} tickers from {path}")
        return cls(by_code)

    def score_asof(
        self, code: str, as_of: pd.Timestamp, lookback_days: int = LOOKBACK_DAYS
    ) -> Tuple[float, str]:
        """
        Return (sentiment_score, circuit_breaker_reason) using only headlines
        with date in (as_of - lookback_days, as_of] — no look-ahead.
        """
        entries = self._by_code.get(code)
        if not entries:
            return 0.0, ""

        cutoff = as_of - timedelta(days=lookback_days)
        window = [h for dt, h in entries if cutoff < dt <= as_of]
        if not window:
            return 0.0, ""

        blocked = check_circuit_breaker(window)
        if blocked:
            return 0.0, blocked

        signal_headlines = filter_signal_keywords(window)
        score = score_headlines(signal_headlines)
        return score, ""

    def headlines_asof(
        self, code: str, as_of: pd.Timestamp, lookback_days: int = LOOKBACK_DAYS
    ) -> List[str]:
        """
        Raw headline titles in (as_of - lookback_days, as_of] — no look-ahead.
        Used by engine.news_filter.classify_headlines() to derive a v2.1
        tier score (0/50/100) from a real user-supplied historical feed,
        the same classifier used in paper/live (see engine/scoring.py).
        """
        entries = self._by_code.get(code)
        if not entries:
            return []
        cutoff = as_of - timedelta(days=lookback_days)
        return [h["title"] for dt, h in entries if cutoff < dt <= as_of]
