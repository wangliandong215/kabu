"""
Unit tests for engine/fundamental.py (mocked moomoo API -- no live OpenD
connection needed to run these). Run:  python -m unittest engine.test_fundamental -v
"""
import time
import unittest
from unittest.mock import MagicMock, patch

import moomoo as ft

from engine.fundamental import TIER2_SCORE, TIER3_SCORE, score


def _inst(uid, name, ratings_newest_first):
    """ratings_newest_first: list of (rating, days_ago) tuples."""
    now = time.time()
    return {
        "institution_info": {"institution_uid": uid, "institution_name": name,
                              "institution_en_name": name},
        "rating_item_list": [
            {"rating": r, "recommendation_date": now - days * 86400, "target_price": 100.0}
            for r, days in ratings_newest_first
        ],
    }


class TestFundamentalScore(unittest.TestCase):

    def test_backtest_env_returns_none_no_api_call(self):
        with patch("moomoo.OpenQuoteContext") as mock_ctx_cls:
            result = score("US.NVDA", env="backtest")
            mock_ctx_cls.assert_not_called()
        self.assertIsNone(result["score"])
        self.assertIsNone(result["tier"])

    def test_recent_downgrade_scores_tier2(self):
        data = {"inst_rating_summary_list": [
            _inst("u1", "UBS", [(2, 5), (3, 10)]),   # 3 -> 2, downgrade, within window
        ]}
        mock_ctx = MagicMock()
        mock_ctx.get_research_rating_summary.return_value = (ft.RET_OK, data)
        with patch("moomoo.OpenQuoteContext", return_value=mock_ctx):
            result = score("US.NVDA", env="paper")
        self.assertEqual(result["tier"], 2)
        self.assertEqual(result["score"], TIER2_SCORE)
        self.assertIn("UBS", result["reason"])

    def test_stable_rating_scores_tier3(self):
        data = {"inst_rating_summary_list": [
            _inst("u1", "UBS", [(3, 5), (3, 10)]),   # unchanged
        ]}
        mock_ctx = MagicMock()
        mock_ctx.get_research_rating_summary.return_value = (ft.RET_OK, data)
        with patch("moomoo.OpenQuoteContext", return_value=mock_ctx):
            result = score("US.NVDA", env="paper")
        self.assertEqual(result["tier"], 3)
        self.assertEqual(result["score"], TIER3_SCORE)

    def test_upgrade_scores_tier3(self):
        data = {"inst_rating_summary_list": [
            _inst("u1", "UBS", [(4, 5), (3, 10)]),   # upgrade
        ]}
        mock_ctx = MagicMock()
        mock_ctx.get_research_rating_summary.return_value = (ft.RET_OK, data)
        with patch("moomoo.OpenQuoteContext", return_value=mock_ctx):
            result = score("US.NVDA", env="paper")
        self.assertEqual(result["tier"], 3)
        self.assertEqual(result["score"], TIER3_SCORE)

    def test_downgrade_outside_lookback_window_ignored(self):
        data = {"inst_rating_summary_list": [
            _inst("u1", "UBS", [(2, 400), (3, 410)]),   # downgrade, but 400 days ago
        ]}
        mock_ctx = MagicMock()
        mock_ctx.get_research_rating_summary.return_value = (ft.RET_OK, data)
        with patch("moomoo.OpenQuoteContext", return_value=mock_ctx):
            result = score("US.NVDA", env="paper")
        self.assertEqual(result["tier"], 3)

    def test_single_rating_no_history_defaults_neutral_tier3(self):
        data = {"inst_rating_summary_list": [
            _inst("u1", "UBS", [(3, 5)]),   # only one data point, can't detect a trend
        ]}
        mock_ctx = MagicMock()
        mock_ctx.get_research_rating_summary.return_value = (ft.RET_OK, data)
        with patch("moomoo.OpenQuoteContext", return_value=mock_ctx):
            result = score("US.NVDA", env="paper")
        self.assertEqual(result["tier"], 3)

    def test_api_error_returns_none_not_raised(self):
        mock_ctx = MagicMock()
        mock_ctx.get_research_rating_summary.return_value = (-1, "some error")
        with patch("moomoo.OpenQuoteContext", return_value=mock_ctx):
            result = score("US.NVDA", env="paper")
        self.assertIsNone(result["score"])
        self.assertEqual(result["reason"], "no_data_or_error")

    def test_empty_data_returns_none(self):
        mock_ctx = MagicMock()
        mock_ctx.get_research_rating_summary.return_value = (ft.RET_OK, {"inst_rating_summary_list": []})
        with patch("moomoo.OpenQuoteContext", return_value=mock_ctx):
            result = score("US.NVDA", env="paper")
        self.assertIsNone(result["score"])

    def test_exception_returns_none_not_raised(self):
        with patch("moomoo.OpenQuoteContext", side_effect=RuntimeError("connection refused")):
            result = score("US.NVDA", env="paper")
        self.assertIsNone(result["score"])
        self.assertEqual(result["reason"], "no_data_or_error")

    def test_multiple_institutions_worst_wins(self):
        data = {"inst_rating_summary_list": [
            _inst("u1", "UBS", [(3, 5), (3, 10)]),          # stable
            _inst("u2", "Bernstein", [(1, 3), (3, 8)]),     # downgrade 3->1
        ]}
        mock_ctx = MagicMock()
        mock_ctx.get_research_rating_summary.return_value = (ft.RET_OK, data)
        with patch("moomoo.OpenQuoteContext", return_value=mock_ctx):
            result = score("US.NVDA", env="paper")
        self.assertEqual(result["tier"], 2)
        self.assertIn("Bernstein", result["reason"])


if __name__ == "__main__":
    unittest.main()
