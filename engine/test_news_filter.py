"""
Unit tests for engine/news_filter.py's tiered risk grading (0/50/100).

Run:  python -m unittest engine.test_news_filter -v
"""
import time
import unittest
from unittest.mock import patch

from engine.news_filter import (
    ACTION_BLOCK,
    ACTION_NORMAL,
    ACTION_REDUCE,
    TIER1_SCORE,
    TIER2_SCORE,
    TIER3_SCORE,
    classify_code,
    classify_headline,
    classify_headlines,
    grade,
)
import engine.news_filter as news_filter


class TestClassifyHeadline(unittest.TestCase):

    def test_tier1_keywords(self):
        cases = [
            "Company admits to accounting fraud in internal audit",
            "Analyst short seller report alleges financial fraud",
            "Stock faces delisting risk after exchange notice",
            "SEC investigation opened into revenue recognition",
            "Shareholders file major lawsuit against the company",
            "CEO arrested on fraud charges",
            "Chairman resigns abruptly amid boardroom dispute",
            "公司被曝财务造假",
            "交易所警示退市风险",
            "做空机构发布做空报告",
            "公司遭SEC调查",
            "股东提起重大诉讼",
            "董事长被捕",
        ]
        for title in cases:
            with self.subTest(title=title):
                self.assertEqual(classify_headline(title), 1)

    def test_tier2_keywords(self):
        cases = [
            "Company reports earnings miss for Q2",
            "Results came in below expectations",
            "Analyst downgrade sends shares lower",
            "Firm lowered guidance for fiscal year",
            "Company cuts outlook citing weak demand",
            "财报低于预期，股价承压",
            "分析师下调评级至中性",
            "公司下调指引",
        ]
        for title in cases:
            with self.subTest(title=title):
                self.assertEqual(classify_headline(title), 2)

    def test_tier3_keywords_and_default(self):
        cases = [
            "Company unveils new product launch at annual event",
            "Firm announces partnership with regional distributor",
            "Company beats estimates on strong quarterly earnings",
            "Results in line with expectations",
            "新产品发布会顺利举行",
            "公司与合作伙伴达成正常合作",
            "一般媒体报道称行业整体平稳",
            "Some totally unrelated headline about weather forecasts",
        ]
        for title in cases:
            with self.subTest(title=title):
                self.assertEqual(classify_headline(title), 3)

    def test_case_insensitive(self):
        self.assertEqual(classify_headline("SEC INVESTIGATION LAUNCHED"), 1)
        self.assertEqual(classify_headline("Earnings MISS reported"), 2)


class TestClassifyHeadlines(unittest.TestCase):

    def test_empty_batch_defaults_tier3(self):
        tier, matched = classify_headlines([])
        self.assertEqual(tier, 3)
        self.assertIsNone(matched)

    def test_tier1_vetoes_batch_even_with_positive_news(self):
        titles = [
            "Company beats estimates on strong quarterly earnings",
            "New product launch well received by analysts",
            "CEO arrested on fraud charges",
        ]
        tier, matched = classify_headlines(titles)
        self.assertEqual(tier, 1)
        self.assertEqual(matched, "ceo arrested")

    def test_tier2_beats_tier3_but_not_tier1(self):
        titles = [
            "Company unveils new product launch",
            "Analyst downgrade issued after weak quarter",
        ]
        tier, matched = classify_headlines(titles)
        self.assertEqual(tier, 2)
        self.assertEqual(matched, "analyst downgrade")

    def test_all_tier3_stays_tier3(self):
        titles = [
            "Company announces partnership with regional distributor",
            "Results in line with expectations",
        ]
        tier, matched = classify_headlines(titles)
        self.assertEqual(tier, 3)
        self.assertIsNone(matched)


class TestGrade(unittest.TestCase):

    def test_tier1_grade(self):
        result = grade(["SEC investigation opened into the company"])
        self.assertEqual(result["tier"], 1)
        self.assertEqual(result["score"], TIER1_SCORE)
        self.assertEqual(result["action"], ACTION_BLOCK)
        self.assertEqual(result["position_scale"], 0.0)
        self.assertEqual(result["matched_keyword"], "sec investigation")

    def test_tier2_grade(self):
        result = grade(["Firm lowered guidance for fiscal year"])
        self.assertEqual(result["tier"], 2)
        self.assertEqual(result["score"], TIER2_SCORE)
        self.assertEqual(result["action"], ACTION_REDUCE)
        self.assertEqual(result["position_scale"], 0.5)

    def test_tier3_grade(self):
        result = grade(["Company beats estimates on strong quarterly earnings"])
        self.assertEqual(result["tier"], 3)
        self.assertEqual(result["score"], TIER3_SCORE)
        self.assertEqual(result["action"], ACTION_NORMAL)
        self.assertEqual(result["position_scale"], 1.0)
        self.assertIsNone(result["matched_keyword"])

    def test_empty_titles_grade_normal(self):
        result = grade([])
        self.assertEqual(result["tier"], 3)
        self.assertEqual(result["action"], ACTION_NORMAL)


class TestClassifyCode(unittest.TestCase):

    def setUp(self):
        news_filter._cache.clear()

    def test_fetches_and_classifies_via_engine_news(self):
        headlines = [{"title": "sec investigation opened", "source": "reuters", "sub_type": "news"}]
        with patch("engine.news._fetch", return_value=headlines) as mock_fetch:
            result = classify_code("US.NVDA")
        mock_fetch.assert_called_once_with("US.NVDA")
        self.assertEqual(result["tier"], 1)

    def test_result_is_cached_within_ttl(self):
        with patch("engine.news._fetch", return_value=[]) as mock_fetch:
            classify_code("US.NVDA")
            classify_code("US.NVDA")
        mock_fetch.assert_called_once()   # second call served from cache

    def test_cache_expires_after_ttl(self):
        with patch("engine.news._fetch", return_value=[]) as mock_fetch:
            classify_code("US.NVDA")
            news_filter._cache["US.NVDA"]["ts"] = time.time() - news_filter._CACHE_TTL - 1
            classify_code("US.NVDA")
        self.assertEqual(mock_fetch.call_count, 2)


if __name__ == "__main__":
    unittest.main()
