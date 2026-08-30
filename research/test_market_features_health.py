"""
Unit tests for research/market_features_health.py — category and overall
status rollup from collection_log rows. Uses a throwaway on-disk store,
never touches config.RESEARCH_MARKET_FEATURES_DB_PATH or real OpenD.

Run:  python -m unittest research.test_market_features_health -v
"""
import tempfile
import unittest
from pathlib import Path

from research.market_features_health import check_daily_market_features
from research.market_features_store import MarketFeaturesStore

_MD = "2026-08-28"


class MarketFeaturesHealthTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = MarketFeaturesStore(db_path=Path(self._tmpdir.name) / "test.db")

    def tearDown(self):
        self.store.close()
        self._tmpdir.cleanup()

    def test_all_success_category_and_overall(self):
        self.store.log_collection(_MD, "news", "US.AAPL", "SUCCESS")
        self.store.log_collection(_MD, "news", "US.MSFT", "SUCCESS")

        report = check_daily_market_features(_MD, store=self.store)

        self.assertEqual(report.categories["news"].status, "SUCCESS")
        self.assertEqual(report.categories["news"].checked, 2)

    def test_mixed_success_and_failed_is_partial(self):
        self.store.log_collection(_MD, "news", "US.AAPL", "SUCCESS")
        self.store.log_collection(_MD, "news", "US.BAD", "FAILED", detail="boom")

        report = check_daily_market_features(_MD, store=self.store)

        cat = report.categories["news"]
        self.assertEqual(cat.status, "PARTIAL")
        self.assertEqual(cat.failed, 1)
        self.assertEqual(len(cat.details), 1)
        self.assertEqual(cat.details[0]["target"], "US.BAD")

    def test_all_failed_is_failed(self):
        self.store.log_collection(_MD, "options", "US.AAPL", "FAILED")
        self.store.log_collection(_MD, "options", "US.MSFT", "FAILED")

        report = check_daily_market_features(_MD, store=self.store)

        self.assertEqual(report.categories["options"].status, "FAILED")

    def test_all_no_data_is_no_data_not_success(self):
        self.store.log_collection(_MD, "macro", "CPI", "NO_DATA")

        report = check_daily_market_features(_MD, store=self.store)

        self.assertEqual(report.categories["macro"].status, "NO_DATA")

    def test_no_rows_at_all_is_skipped(self):
        report = check_daily_market_features(_MD, store=self.store)

        self.assertEqual(report.categories["fedwatch"].status, "SKIPPED")

    def test_explicit_skipped_rows_are_skipped(self):
        self.store.log_collection(_MD, "macro", "ALL", "SKIPPED",
                                   detail="RESEARCH_MACRO_INDICATOR_IDS is empty")

        report = check_daily_market_features(_MD, store=self.store)

        self.assertEqual(report.categories["macro"].status, "SKIPPED")

    def test_overall_takes_worst_of_non_skipped_categories(self):
        self.store.log_collection(_MD, "news", "US.AAPL", "SUCCESS")
        self.store.log_collection(_MD, "options", "US.AAPL", "FAILED")
        self.store.log_collection(_MD, "options", "US.MSFT", "SUCCESS")
        self.store.log_collection(_MD, "fedwatch", "TARGET_RATE", "SUCCESS")
        # macro: no rows at all -> SKIPPED, must not drag overall down to SKIPPED

        report = check_daily_market_features(_MD, store=self.store)

        self.assertEqual(report.categories["macro"].status, "SKIPPED")
        self.assertEqual(report.categories["options"].status, "PARTIAL")
        self.assertEqual(report.overall, "PARTIAL",
                          "overall must reflect the worst real category, ignoring SKIPPED")

    def test_overall_is_skipped_only_when_everything_is_skipped(self):
        report = check_daily_market_features(_MD, store=self.store)
        self.assertEqual(report.overall, "SKIPPED")

    def test_default_market_date_matches_market_date_for(self):
        import engine.market_hours as market_hours
        expected = market_hours.market_date_for("US").isoformat()
        report = check_daily_market_features(store=self.store)
        self.assertEqual(report.market_date, expected)

    def test_category_does_not_see_rows_from_other_market_dates(self):
        self.store.log_collection("2026-08-27", "news", "US.AAPL", "SUCCESS")
        self.store.log_collection(_MD, "news", "US.AAPL", "FAILED")

        report = check_daily_market_features(_MD, store=self.store)

        self.assertEqual(report.categories["news"].status, "FAILED")
        self.assertEqual(report.categories["news"].checked, 1)


if __name__ == "__main__":
    unittest.main()
