"""
Unit tests for research/collect_market_features.py — every OpenD-hitting
call (engine.news.research_snapshot, data.options.*, data.macro.*) is
monkeypatched, so this suite never touches OpenD.

Run:  python -m unittest research.test_collect_market_features -v
"""
import tempfile
import unittest
from pathlib import Path

import config
import engine.market_hours as market_hours
import research.collect_market_features as collector
from research.market_features_store import MarketFeaturesStore


def market_hours_market_date() -> str:
    """Same market_date the collector itself computes — used by tests to
    look up the right collection_log market_date bucket without hardcoding
    a date that would go stale."""
    return market_hours.market_date_for("US").isoformat()


class CollectMarketFeaturesTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = MarketFeaturesStore(db_path=Path(self._tmpdir.name) / "test.db")

        self._orig = {
            "research_snapshot": collector.news_module.research_snapshot,
            "get_underlying_overview": collector.options_data.get_underlying_overview,
            "get_market_put_call_ratio": collector.options_data.get_market_put_call_ratio,
            "get_macro_indicator_list": collector.macro_data.get_macro_indicator_list,
            "get_macro_indicator_history": collector.macro_data.get_macro_indicator_history,
            "get_fed_watch_target_rate": collector.macro_data.get_fed_watch_target_rate,
            "get_fed_watch_dot_plot": collector.macro_data.get_fed_watch_dot_plot,
            "RESEARCH_MACRO_INDICATOR_IDS": config.RESEARCH_MACRO_INDICATOR_IDS,
        }

    def tearDown(self):
        collector.news_module.research_snapshot = self._orig["research_snapshot"]
        collector.options_data.get_underlying_overview = self._orig["get_underlying_overview"]
        collector.options_data.get_market_put_call_ratio = self._orig["get_market_put_call_ratio"]
        collector.macro_data.get_macro_indicator_list = self._orig["get_macro_indicator_list"]
        collector.macro_data.get_macro_indicator_history = self._orig["get_macro_indicator_history"]
        collector.macro_data.get_fed_watch_target_rate = self._orig["get_fed_watch_target_rate"]
        collector.macro_data.get_fed_watch_dot_plot = self._orig["get_fed_watch_dot_plot"]
        config.RESEARCH_MACRO_INDICATOR_IDS = self._orig["RESEARCH_MACRO_INDICATOR_IDS"]
        self.store.close()
        self._tmpdir.cleanup()

    def test_news_pass_records_one_row_per_code(self):
        collector.news_module.research_snapshot = lambda code: {
            "news_count_24h": 3, "major_news_flag": False, "announcement_flag": True}

        n = collector.run_news_pass(["US.AAPL", "US.MSFT"], self.store)

        self.assertEqual(n, 2)
        rows = self.store.all_rows("news_snapshots")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["announcement_flag"], 1)
        self.assertIsNotNone(rows[0]["observed_at"])
        self.assertIsNotNone(rows[0]["market_date"])

    def test_news_pass_zero_headlines_is_success_not_failed(self):
        collector.news_module.research_snapshot = lambda code: {
            "news_count_24h": 0, "major_news_flag": False, "announcement_flag": False}

        collector.run_news_pass(["US.AAPL"], self.store)

        log_rows = self.store.collection_log_rows(
            market_hours_market_date(), "news")
        self.assertEqual(log_rows[0]["status"], "SUCCESS",
                          "zero headlines is a legitimate result, not a failure")

    def test_news_pass_survives_one_code_failing(self):
        def _snap(code):
            if code == "US.BAD":
                raise RuntimeError("news fetch failed")
            return {"news_count_24h": 1, "major_news_flag": False, "announcement_flag": False}
        collector.news_module.research_snapshot = _snap

        n = collector.run_news_pass(["US.BAD", "US.AAPL"], self.store)

        self.assertEqual(n, 1, "a single failing code must not abort the whole pass")
        log_rows = {r["target"]: r["status"] for r in
                    self.store.collection_log_rows(market_hours_market_date(), "news")}
        self.assertEqual(log_rows["US.BAD"], "FAILED")
        self.assertEqual(log_rows["US.AAPL"], "SUCCESS")

    def test_options_pass_computes_put_call_ratio(self):
        collector.options_data.get_underlying_overview = lambda codes: [
            {"code": "US.AAPL", "call_volume": 1000, "put_volume": 400,
             "iv": 28.0, "iv_rank": 40.0, "iv_percentile": 55.0, "hv_30d": 25.0},
        ]
        collector.options_data.get_market_put_call_ratio = lambda market: [
            {"time": "2026-08-30", "timestamp": 1.0, "call_value": 100,
             "put_value": 50, "total_value": 150, "ratio": 0.5},
        ]

        n = collector.run_options_pass(["US.AAPL"], self.store)

        self.assertEqual(n, 1)
        rows = self.store.all_rows("options_snapshots")
        self.assertAlmostEqual(rows[0]["put_call_ratio"], 0.4)
        self.assertEqual(len(self.store.all_rows("market_pcr_snapshots")), 1)

    def test_options_pass_handles_zero_call_volume(self):
        collector.options_data.get_underlying_overview = lambda codes: [
            {"code": "US.AAPL", "call_volume": 0, "put_volume": 400,
             "iv": 28.0, "iv_rank": 40.0, "iv_percentile": 55.0, "hv_30d": 25.0},
        ]
        collector.options_data.get_market_put_call_ratio = lambda market: []

        collector.run_options_pass(["US.AAPL"], self.store)

        rows = self.store.all_rows("options_snapshots")
        self.assertIsNone(rows[0]["put_call_ratio"],
                           "call_volume=0 must yield None ratio, not a ZeroDivisionError")

    def test_options_pass_logs_no_data_for_code_missing_from_overview(self):
        collector.options_data.get_underlying_overview = lambda codes: [
            {"code": "US.AAPL", "call_volume": 1000, "put_volume": 400,
             "iv": 28.0, "iv_rank": 40.0, "iv_percentile": 55.0, "hv_30d": 25.0},
        ]
        collector.options_data.get_market_put_call_ratio = lambda market: []

        collector.run_options_pass(["US.AAPL", "US.NODATA"], self.store)

        log_rows = {r["target"]: r["status"] for r in
                    self.store.collection_log_rows(market_hours_market_date(), "options")}
        self.assertEqual(log_rows["US.AAPL"], "SUCCESS")
        self.assertEqual(log_rows["US.NODATA"], "NO_DATA",
                          "a code absent from the overview response is NO_DATA, not FAILED")

    def test_options_pass_batch_failure_marks_every_code_failed(self):
        def _raise(codes):
            raise RuntimeError("OpenD unreachable")
        collector.options_data.get_underlying_overview = _raise
        collector.options_data.get_market_put_call_ratio = lambda market: []

        collector.run_options_pass(["US.AAPL", "US.MSFT"], self.store)

        log_rows = {r["target"]: r["status"] for r in
                    self.store.collection_log_rows(market_hours_market_date(), "options")}
        self.assertEqual(log_rows["US.AAPL"], "FAILED")
        self.assertEqual(log_rows["US.MSFT"], "FAILED")

    def test_macro_pass_skipped_when_no_indicator_ids_configured(self):
        config.RESEARCH_MACRO_INDICATOR_IDS = []

        n = collector.run_macro_pass(self.store)

        self.assertEqual(n, 0)
        log_rows = self.store.collection_log_rows(market_hours_market_date(), "macro")
        self.assertEqual(len(log_rows), 1)
        self.assertEqual(log_rows[0]["status"], "SKIPPED")

    def test_macro_pass_logs_no_data_for_empty_history(self):
        config.RESEARCH_MACRO_INDICATOR_IDS = ["CPI"]
        collector.macro_data.get_macro_indicator_list = lambda region: []
        collector.macro_data.get_macro_indicator_history = lambda indicator_id, time=None, max_count=None: []

        collector.run_macro_pass(self.store)

        log_rows = {r["target"]: r["status"] for r in
                    self.store.collection_log_rows(market_hours_market_date(), "macro")}
        self.assertEqual(log_rows["CPI"], "NO_DATA")

    def test_macro_pass_only_fetches_configured_indicator_ids(self):
        config.RESEARCH_MACRO_INDICATOR_IDS = ["CPI"]
        collector.macro_data.get_macro_indicator_list = lambda region: [
            {"indicator_id": "CPI", "category_name": "Inflation", "name": "CPI YoY"},
            {"indicator_id": "GDP", "category_name": "Growth", "name": "GDP"},
        ]
        calls = []

        def _history(indicator_id, time=None, max_count=None):
            calls.append(indicator_id)
            return [{"data_time": "2026-08-01", "release_time": "2026-08-13",
                      "value": 3.1, "predict_value": 3.0, "previous_value": 3.2,
                      "unit_type": "%"}]
        collector.macro_data.get_macro_indicator_history = _history

        n = collector.run_macro_pass(self.store)

        self.assertEqual(calls, ["CPI"], "must not fetch history for indicators not in the config list")
        self.assertEqual(n, 1)

    def test_fedwatch_pass_records_both_tables(self):
        collector.macro_data.get_fed_watch_target_rate = lambda: [
            {"meeting_date": "2026-09-17", "target_range": "4.00-4.25", "probability": 0.65},
        ]
        collector.macro_data.get_fed_watch_dot_plot = lambda: [
            {"year": 2027, "rate": 3.5, "vote_count": 5, "is_median": True,
             "median_rate": 3.5, "current_rate": 4.0},
        ]

        n = collector.run_fedwatch_pass(self.store)

        self.assertEqual(n, 1)
        self.assertEqual(len(self.store.all_rows("fedwatch_dot_plot_snapshots")), 1)
        log_rows = {r["target"]: r["status"] for r in
                    self.store.collection_log_rows(market_hours_market_date(), "fedwatch")}
        self.assertEqual(log_rows["TARGET_RATE"], "SUCCESS")
        self.assertEqual(log_rows["DOT_PLOT"], "SUCCESS")

    def test_fedwatch_pass_logs_failed_on_exception(self):
        def _raise():
            raise RuntimeError("OpenD unreachable")
        collector.macro_data.get_fed_watch_target_rate = _raise
        collector.macro_data.get_fed_watch_dot_plot = lambda: []

        collector.run_fedwatch_pass(self.store)

        log_rows = {r["target"]: r["status"] for r in
                    self.store.collection_log_rows(market_hours_market_date(), "fedwatch")}
        self.assertEqual(log_rows["TARGET_RATE"], "FAILED")
        self.assertEqual(log_rows["DOT_PLOT"], "NO_DATA")


if __name__ == "__main__":
    unittest.main()
