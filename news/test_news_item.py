"""
Run:  python -m unittest news.test_news_item -v
"""
import unittest
from datetime import datetime

from news.news_item import NewsItem, normalize_headline
from news.news_sources import Source


class TestNormalizeHeadline(unittest.TestCase):

    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(normalize_headline("NVDA Beats Estimates!!"), "nvda beats estimates")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_headline("NVDA   beats   estimates"), "nvda beats estimates")

    def test_equivalent_headlines_normalize_identically(self):
        a = normalize_headline("Company X Files for Bankruptcy.")
        b = normalize_headline("company x files for bankruptcy")
        self.assertEqual(a, b)


class TestNewsItemDedupKey(unittest.TestCase):

    def test_same_ticker_and_headline_same_key(self):
        a = NewsItem(id="1", ticker="US.NVDA", source=Source.REUTERS,
                     timestamp=datetime.now(), headline="NVDA files for bankruptcy")
        b = NewsItem(id="2", ticker="US.NVDA", source=Source.BLOOMBERG,
                     timestamp=datetime.now(), headline="nvda Files For Bankruptcy!")
        self.assertEqual(a.dedup_key(), b.dedup_key())

    def test_different_ticker_different_key(self):
        a = NewsItem(id="1", ticker="US.NVDA", source=Source.REUTERS,
                     timestamp=datetime.now(), headline="earnings beat")
        b = NewsItem(id="2", ticker="US.AMD", source=Source.REUTERS,
                     timestamp=datetime.now(), headline="earnings beat")
        self.assertNotEqual(a.dedup_key(), b.dedup_key())


if __name__ == "__main__":
    unittest.main()
