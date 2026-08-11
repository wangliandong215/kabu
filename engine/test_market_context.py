"""
Unit tests for engine/market_context.py.
Run:  python -m unittest engine.test_market_context -v
"""
import json
import sqlite3
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

import pytz

from engine import market_context
from engine.market_context import (
    MarketContextStore, update_market_context,
    DataIntegrityError, DataStaleError,
    _parse_vix_csv, _parse_cnn_payload, _parse_naaim_chart_json,
    _check_freshness, _validate_numeric,
    fetch_vix, fetch_cnn_fear_greed, fetch_naaim,
)


_ET = pytz.timezone("America/New_York")


class MarketContextStoreTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "test_trade_history.db"
        self.store = MarketContextStore(self.db_path)

    def tearDown(self):
        self.store.close()
        self._tmpdir.cleanup()


class TestSaveAndFetch(MarketContextStoreTestCase):

    def test_save_then_get_latest(self):
        self.store.save({
            "observation_date": "2026-08-07",
            "vix_close": 18.7, "vix_date": "2026-08-06", "vix_source": "CBOE",
            "data_status": {"vix": "OK"},
        })
        row = self.store.get_latest_context()
        self.assertEqual(row["observation_date"], "2026-08-07")
        self.assertEqual(row["vix_close"], 18.7)
        self.assertEqual(row["vix_date"], "2026-08-06")

    def test_get_latest_returns_most_recent_date(self):
        self.store.save({"observation_date": "2026-08-05", "vix_close": 15.0})
        self.store.save({"observation_date": "2026-08-07", "vix_close": 17.0})
        self.store.save({"observation_date": "2026-08-06", "vix_close": 16.0})
        row = self.store.get_latest_context()
        self.assertEqual(row["observation_date"], "2026-08-07")
        self.assertEqual(row["vix_close"], 17.0)

    def test_get_context_for_date_missing_returns_none(self):
        self.assertIsNone(self.store.get_context_for_date("2099-01-01"))

    def test_save_is_idempotent_upsert(self):
        self.store.save({"observation_date": "2026-08-07", "vix_close": 15.0})
        self.store.save({"observation_date": "2026-08-07", "vix_close": 16.0})
        rows = self._raw("SELECT * FROM market_context")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["vix_close"], 16.0)

    def test_no_context_yet_returns_none(self):
        self.assertIsNone(self.store.get_latest_context())

    def _raw(self, sql, params=()):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()


class TestUpdateMarketContext(MarketContextStoreTestCase):
    """update_market_context() creates its own MarketContextStore() using
    the module-level default db path, so these tests patch _DEFAULT_DB_PATH
    (via MarketContextStore's own default lookup) by patching the
    MarketContextStore class construction inside update_market_context to
    point at our temp DB instead."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch("engine.market_context.MarketContextStore",
                              lambda: MarketContextStore(self.db_path))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_all_sources_ok(self):
        with mock.patch("engine.market_context.fetch_vix",
                         return_value={"vix_close": 18.7, "vix_date": "2026-08-06",
                                       "vix_source": "CBOE"}), \
             mock.patch("engine.market_context.fetch_cnn_fear_greed",
                         return_value={"cnn_fear_greed": 63.0,
                                       "cnn_fear_greed_label": "greed",
                                       "cnn_fear_greed_date": "2026-08-07"}), \
             mock.patch("engine.market_context.fetch_naaim",
                         return_value={"naaim_exposure": 82.5,
                                       "naaim_date": "2026-08-05"}):
            record = update_market_context(now=_ET.localize(datetime(2026, 8, 7, 8, 30)))

        self.assertEqual(record["observation_date"], "2026-08-07")
        self.assertEqual(record["vix_close"], 18.7)
        self.assertEqual(record["cnn_fear_greed"], 63.0)
        self.assertEqual(record["naaim_exposure"], 82.5)
        self.assertEqual(record["data_status"]["vix"], "OK")
        self.assertEqual(record["data_status"]["cnn"], "OK")
        self.assertEqual(record["data_status"]["naaim"], "OK")
        self.assertEqual(record["data_status"]["aaii"], "NOT_IMPLEMENTED")
        self.assertEqual(record["data_status"]["put_call"], "NOT_IMPLEMENTED")
        self.assertIsNone(record["aaii_bullish"])
        self.assertIsNone(record["put_call_ratio"])

    def test_fetch_failure_carries_forward_previous_value(self):
        self.store.save({
            "observation_date": "2026-08-06",
            "vix_close": 15.9, "vix_date": "2026-08-06", "vix_source": "CBOE",
        })
        with mock.patch("engine.market_context.fetch_vix", return_value=None), \
             mock.patch("engine.market_context.fetch_cnn_fear_greed", return_value=None), \
             mock.patch("engine.market_context.fetch_naaim", return_value=None):
            record = update_market_context(now=_ET.localize(datetime(2026, 8, 7, 8, 30)))

        self.assertEqual(record["vix_close"], 15.9)
        self.assertEqual(record["vix_date"], "2026-08-06")
        self.assertEqual(record["data_status"]["vix"], "CARRIED_FORWARD")

    def test_fetch_failure_with_no_history_is_null_and_unavailable(self):
        with mock.patch("engine.market_context.fetch_vix", return_value=None), \
             mock.patch("engine.market_context.fetch_cnn_fear_greed", return_value=None), \
             mock.patch("engine.market_context.fetch_naaim", return_value=None):
            record = update_market_context(now=_ET.localize(datetime(2026, 8, 7, 8, 30)))

        self.assertIsNone(record["vix_close"])
        self.assertEqual(record["data_status"]["vix"], "UNAVAILABLE")


class TestShouldRunDailyUpdate(unittest.TestCase):

    def setUp(self):
        market_context._last_update_date = None

    def tearDown(self):
        market_context._last_update_date = None

    def test_fires_after_threshold_on_trading_day(self):
        # 2026-08-07 is a Friday, not a US holiday.
        now = _ET.localize(datetime(2026, 8, 7, 8, 30))
        self.assertTrue(market_context.should_run_daily_update(now))

    def test_does_not_fire_before_threshold(self):
        now = _ET.localize(datetime(2026, 8, 7, 8, 0))
        self.assertFalse(market_context.should_run_daily_update(now))

    def test_does_not_fire_twice_same_day(self):
        now1 = _ET.localize(datetime(2026, 8, 7, 8, 30))
        now2 = _ET.localize(datetime(2026, 8, 7, 12, 0))
        self.assertTrue(market_context.should_run_daily_update(now1))
        self.assertFalse(market_context.should_run_daily_update(now2))

    def test_fires_again_next_trading_day(self):
        now1 = _ET.localize(datetime(2026, 8, 7, 8, 30))
        now2 = _ET.localize(datetime(2026, 8, 10, 8, 30))   # following Monday
        self.assertTrue(market_context.should_run_daily_update(now1))
        self.assertTrue(market_context.should_run_daily_update(now2))

    def test_does_not_fire_on_weekend(self):
        # 2026-08-08 is a Saturday.
        now = _ET.localize(datetime(2026, 8, 8, 9, 0))
        self.assertFalse(market_context.should_run_daily_update(now))

    def test_does_not_fire_on_us_holiday(self):
        # 2026-09-07 is Labor Day (see engine/market_hours.py _US_HOLIDAYS).
        now = _ET.localize(datetime(2026, 9, 7, 9, 0))
        self.assertFalse(market_context.should_run_daily_update(now))


class TestParseVixCsv(unittest.TestCase):
    """Regression coverage for the 2026-08-08 incident: an LLM web-fetch
    tool reading CBOE's VIX_History.csv silently reported an 1997-10-03
    row as "the latest" instead of the true 2026-08-07 row. _parse_vix_csv
    must never trust file position (header/first/last line) — only
    max(date) among all successfully-parsed rows."""

    def test_normal_ascending_csv_picks_last_row(self):
        csv = "DATE,OPEN,HIGH,LOW,CLOSE\n08/05/2026,16.15,18.43,15.48,15.81\n08/06/2026,15.83,16.03,15.11,15.15\n08/07/2026,15.30,15.36,14.77,14.90\n"
        result = _parse_vix_csv(csv)
        self.assertEqual(result["vix_date"], date(2026, 8, 7))
        self.assertEqual(result["vix_close"], 14.90)

    def test_shuffled_csv_still_picks_max_date(self):
        """CSV rows out of chronological order must not fool the parser
        into picking a wrong "last line" — it must scan all rows and pick
        max(date) regardless of file order."""
        csv = ("DATE,OPEN,HIGH,LOW,CLOSE\n"
               "08/05/2026,16.15,18.43,15.48,15.81\n"
               "08/07/2026,15.30,15.36,14.77,14.90\n"   # true latest, in the middle
               "08/06/2026,15.83,16.03,15.11,15.15\n")
        result = _parse_vix_csv(csv)
        self.assertEqual(result["vix_date"], date(2026, 8, 7))
        self.assertEqual(result["vix_close"], 14.90)

    def test_1997_row_mixed_with_2026_row_picks_2026(self):
        """The actual shape of the real VIX_History.csv: decades of old
        rows plus a handful of recent ones. Must pick the recent one, not
        the historical row an LLM summarizer once mistook for "latest"."""
        csv = ("DATE,OPEN,HIGH,LOW,CLOSE\n"
               "10/03/1997,20.75,20.75,20.75,20.75\n"
               "08/07/2026,15.30,15.36,14.77,14.90\n")
        result = _parse_vix_csv(csv)
        self.assertEqual(result["vix_date"], date(2026, 8, 7))
        self.assertEqual(result["vix_close"], 14.90)

    def test_header_blank_and_malformed_lines_are_skipped(self):
        csv = ("DATE,OPEN,HIGH,LOW,CLOSE\n"
               "\n"
               "not,a,valid,row\n"
               "08/07/2026,15.30,15.36,14.77,14.90\n"
               "   \n")
        result = _parse_vix_csv(csv)
        self.assertEqual(result["vix_date"], date(2026, 8, 7))
        self.assertEqual(result["vix_close"], 14.90)

    def test_no_valid_rows_raises_integrity_error(self):
        csv = "DATE,OPEN,HIGH,LOW,CLOSE\nnot,a,valid,row\n"
        with self.assertRaises(DataIntegrityError):
            _parse_vix_csv(csv)

    def test_empty_string_raises_integrity_error(self):
        with self.assertRaises(DataIntegrityError):
            _parse_vix_csv("")

    def test_conflicting_close_for_same_max_date_raises(self):
        csv = ("DATE,OPEN,HIGH,LOW,CLOSE\n"
               "08/07/2026,15.30,15.36,14.77,14.90\n"
               "08/07/2026,15.30,15.36,14.77,99.99\n")
        with self.assertRaises(DataIntegrityError):
            _parse_vix_csv(csv)

    def test_duplicate_identical_rows_do_not_raise(self):
        """Same date AND same value twice is a harmless duplicate, not a
        conflict — must not be confused with the conflicting-values case."""
        csv = ("DATE,OPEN,HIGH,LOW,CLOSE\n"
               "08/07/2026,15.30,15.36,14.77,14.90\n"
               "08/07/2026,15.30,15.36,14.77,14.90\n")
        result = _parse_vix_csv(csv)
        self.assertEqual(result["vix_close"], 14.90)


class TestCheckFreshness(unittest.TestCase):

    def test_within_lag_does_not_raise(self):
        _check_freshness("TEST", date(2026, 8, 7), date(2026, 8, 8), max_lag_days=7)

    def test_weekend_gap_within_lag_does_not_raise(self):
        # Friday close, checked the following Monday — 3 calendar days.
        _check_freshness("TEST", date(2026, 8, 7), date(2026, 8, 10), max_lag_days=7)

    def test_too_old_raises_stale_error(self):
        with self.assertRaises(DataStaleError):
            _check_freshness("TEST", date(1997, 10, 3), date(2026, 8, 8), max_lag_days=7)

    def test_future_date_raises_stale_error(self):
        with self.assertRaises(DataStaleError):
            _check_freshness("TEST", date(2026, 8, 9), date(2026, 8, 8), max_lag_days=7)

    def test_exactly_at_lag_boundary_does_not_raise(self):
        _check_freshness("TEST", date(2026, 8, 1), date(2026, 8, 8), max_lag_days=7)

    def test_one_day_past_lag_boundary_raises(self):
        with self.assertRaises(DataStaleError):
            _check_freshness("TEST", date(2026, 7, 31), date(2026, 8, 8), max_lag_days=7)


class TestValidateNumeric(unittest.TestCase):

    def test_normal_value_does_not_raise(self):
        _validate_numeric("TEST", 14.9, 0.0, 200.0)

    def test_nan_raises(self):
        with self.assertRaises(DataIntegrityError):
            _validate_numeric("TEST", float("nan"), 0.0, 200.0)

    def test_inf_raises(self):
        with self.assertRaises(DataIntegrityError):
            _validate_numeric("TEST", float("inf"), 0.0, 200.0)

    def test_non_numeric_raises(self):
        with self.assertRaises(DataIntegrityError):
            _validate_numeric("TEST", "14.9", 0.0, 200.0)

    def test_out_of_range_raises(self):
        with self.assertRaises(DataIntegrityError):
            _validate_numeric("TEST", 999.0, 0.0, 200.0)


class TestParseCnnPayload(unittest.TestCase):

    def test_normal_payload(self):
        payload = {"fear_and_greed": {"score": 63.6857, "rating": "greed",
                                      "timestamp": "2026-08-07T23:59:47+00:00"}}
        result = _parse_cnn_payload(payload)
        self.assertEqual(result["cnn_fear_greed"], 63.69)
        self.assertEqual(result["cnn_fear_greed_label"], "greed")
        self.assertEqual(result["cnn_fear_greed_date"], date(2026, 8, 7))

    def test_missing_key_raises(self):
        with self.assertRaises(DataIntegrityError):
            _parse_cnn_payload({"fear_and_greed": {"score": 63.0}})

    def test_unparseable_timestamp_raises(self):
        payload = {"fear_and_greed": {"score": 63.0, "rating": "greed",
                                      "timestamp": "not-a-date"}}
        with self.assertRaises(DataIntegrityError):
            _parse_cnn_payload(payload)


class TestParseNaaimChartJson(unittest.TestCase):

    def _chart(self, labels, data):
        return {"data": {"labels": labels,
                         "datasets": [{"label": "NAAIM Number", "data": data},
                                      {"label": "S&P 500", "data": [1, 2]}]}}

    def test_picks_max_date_regardless_of_order(self):
        chart = self._chart(["2026-04-29", "2026-08-05", "2026-07-01"],
                            ["93.79", "82.50", "70.00"])
        result = _parse_naaim_chart_json(chart)
        self.assertEqual(result["naaim_date"], date(2026, 8, 5))
        self.assertEqual(result["naaim_exposure"], 82.50)

    def test_missing_dataset_raises(self):
        chart = {"data": {"labels": ["2026-08-05"],
                          "datasets": [{"label": "S&P 500", "data": [1]}]}}
        with self.assertRaises(DataIntegrityError):
            _parse_naaim_chart_json(chart)

    def test_length_mismatch_raises(self):
        chart = self._chart(["2026-08-05", "2026-08-12"], ["82.50"])
        with self.assertRaises(DataIntegrityError):
            _parse_naaim_chart_json(chart)

    def test_no_valid_pairs_raises(self):
        chart = self._chart(["not-a-date"], ["not-a-number"])
        with self.assertRaises(DataIntegrityError):
            _parse_naaim_chart_json(chart)


class TestFetchVixEndToEnd(unittest.TestCase):
    """fetch_vix() itself (not just _parse_vix_csv) must reject the stale
    row end-to-end — this is the exact regression test the incident asked
    for: a mocked HTTP response shaped like the real bug must come out as
    None, never as VIX=20.75."""

    def _mock_response(self, text):
        resp = mock.Mock()
        resp.text = text
        resp.raise_for_status = mock.Mock()
        return resp

    def test_normal_response_returns_value(self):
        csv = "DATE,OPEN,HIGH,LOW,CLOSE\n08/07/2026,15.30,15.36,14.77,14.90\n"
        with mock.patch("requests.get", return_value=self._mock_response(csv)):
            result = fetch_vix(now=date(2026, 8, 8))
        self.assertEqual(result["vix_close"], 14.90)
        self.assertEqual(result["vix_date"], "2026-08-07")

    def test_vix_stale_historical_row_is_rejected(self):
        """The literal 2026-08-08 incident, reproduced: a response whose
        only/most-recent-looking row is the 1997-10-03/20.75 pair must be
        rejected as stale (DataStaleError, caught internally) — fetch_vix()
        must return None, never {"vix_close": 20.75, ...}."""
        csv = "DATE,OPEN,HIGH,LOW,CLOSE\n10/03/1997,20.75,20.75,20.75,20.75\n"
        with mock.patch("requests.get", return_value=self._mock_response(csv)):
            result = fetch_vix(now=date(2026, 8, 8))
        self.assertIsNone(result)

    def test_mixed_history_still_returns_the_fresh_row_not_none(self):
        """Full realistic shape: decades of old rows AND a fresh one in
        the same file — must return the fresh 2026 value, not reject the
        whole file just because most of it is old."""
        csv = ("DATE,OPEN,HIGH,LOW,CLOSE\n"
               "10/03/1997,20.75,20.75,20.75,20.75\n"
               "08/07/2026,15.30,15.36,14.77,14.90\n")
        with mock.patch("requests.get", return_value=self._mock_response(csv)):
            result = fetch_vix(now=date(2026, 8, 8))
        self.assertEqual(result["vix_close"], 14.90)

    def test_out_of_range_value_is_rejected(self):
        csv = "DATE,OPEN,HIGH,LOW,CLOSE\n08/07/2026,15.30,15.36,14.77,9999.0\n"
        with mock.patch("requests.get", return_value=self._mock_response(csv)):
            result = fetch_vix(now=date(2026, 8, 8))
        self.assertIsNone(result)

    def test_http_error_returns_none(self):
        resp = mock.Mock()
        resp.raise_for_status.side_effect = Exception("HTTP 500")
        with mock.patch("requests.get", return_value=resp):
            result = fetch_vix(now=date(2026, 8, 8))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
