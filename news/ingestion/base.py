"""
news/ingestion/base.py -- V3.4 News Source abstraction (spec section 2/27).
Phase 1 only ships news/ingestion/mock.py -- no real Reuters/Bloomberg/SEC/
Social API. A real source implements this ABC and nothing downstream
(detection/validation/scoring/risk adapter) needs to change.
"""
from abc import ABC, abstractmethod
from typing import List

from news.news_item import NewsItem


class NewsSource(ABC):
    @abstractmethod
    def fetch(self, tickers: List[str]) -> List[NewsItem]:
        raise NotImplementedError
