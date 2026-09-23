"""
news/news_item.py -- V3.4 raw ingested news object (spec section 4). One
NewsItem per distinct wire story; the same story re-appearing across
sources/refreshes is deduplicated downstream by
news/scoring/aggregator.py using normalize_headline()/dedup_key(), never
here -- ingestion should not silently drop a second copy that might carry
a corroborating source.
"""
import re
from dataclasses import dataclass
from datetime import datetime

from news.news_sources import Source

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_headline(headline: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace -- just enough
    normalization that the same story with a slightly different source
    byline/punctuation still dedups (V3.4 spec Test 4)."""
    text = _PUNCT_RE.sub("", headline.lower())
    return _WS_RE.sub(" ", text).strip()


@dataclass
class NewsItem:
    id: str
    ticker: str
    source: Source
    timestamp: datetime
    headline: str
    body: str = ""
    url: str = ""
    language: str = "en"

    def dedup_key(self) -> str:
        return f"{self.ticker}:{normalize_headline(self.headline)}"
