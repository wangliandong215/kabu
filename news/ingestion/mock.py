"""
news/ingestion/mock.py -- synthetic/historical NewsSource for tests and for
running the full V3.4 pipeline with zero real API dependency (spec section
2/24: Mock News, Synthetic Event, Historical Event).
"""
from typing import List, Optional

from news.ingestion.base import NewsSource
from news.news_item import NewsItem


class MockNewsSource(NewsSource):
    def __init__(self, items: Optional[List[NewsItem]] = None):
        self._items: List[NewsItem] = list(items or [])

    def add(self, item: NewsItem) -> None:
        self._items.append(item)

    def fetch(self, tickers: List[str]) -> List[NewsItem]:
        wanted = set(tickers)
        return [item for item in self._items if item.ticker in wanted]
