"""
Unit tests for engine/market_hours.py::market_date_for() (v2.11.1 Research
time-model) — the "which US trading day does this observation belong to"
question, distinct from is_open()'s live go/no-go check.

Run:  python -m unittest engine.test_market_date_for -v
"""
import unittest
from datetime import date, datetime

import pytz

from engine.market_hours import market_date_for

_JST = pytz.timezone("Asia/Tokyo")


def _jst(y, m, d, hh, mm=0):
    return _JST.localize(datetime(y, m, d, hh, mm))


class MarketDateForTestCase(unittest.TestCase):

    def test_live_evening_session_summer(self):
        # Monday 2026-08-17 23:00 JST — summer session just opened (22:30 JST).
        got = market_date_for("US.AAPL", now=_jst(2026, 8, 17, 23, 0))
        self.assertEqual(got, date(2026, 8, 17))

    def test_midnight_crossing_tail_summer(self):
        # Tuesday 2026-08-18 03:00 JST — tail of Monday's still-open session.
        got = market_date_for("US.AAPL", now=_jst(2026, 8, 18, 3, 0))
        self.assertEqual(got, date(2026, 8, 17))

    def test_daytime_gap_after_close_before_next_open(self):
        # Tuesday 2026-08-18 07:15 JST — the actual daily collector run time,
        # squarely in the gap after Monday's session closed and before
        # Tuesday evening's session opens.
        got = market_date_for("US.AAPL", now=_jst(2026, 8, 18, 7, 15))
        self.assertEqual(got, date(2026, 8, 17),
                          "collector-time snapshot must attribute to the "
                          "just-completed session, not today's JST calendar day")

    def test_weekend_walks_back_to_friday(self):
        # Saturday 2026-08-22 07:15 JST.
        got = market_date_for("US.AAPL", now=_jst(2026, 8, 22, 7, 15))
        self.assertEqual(got, date(2026, 8, 21))

    def test_sunday_walks_back_to_friday(self):
        # Sunday 2026-08-30 12:00 JST (today, per the actual current session).
        got = market_date_for("US.AAPL", now=_jst(2026, 8, 30, 12, 0))
        self.assertEqual(got, date(2026, 8, 28))

    def test_holiday_monday_walks_back_to_prior_friday(self):
        # Tuesday 2026-09-08 07:15 JST — Labor Day (Mon 2026-09-07) skipped.
        got = market_date_for("US.AAPL", now=_jst(2026, 9, 8, 7, 15))
        self.assertEqual(got, date(2026, 9, 4))

    def test_winter_session_boundary(self):
        # Monday 2026-01-12 23:59 JST — winter session opens 23:30 JST.
        got = market_date_for("US.AAPL", now=_jst(2026, 1, 12, 23, 59))
        self.assertEqual(got, date(2026, 1, 12))

    def test_winter_daytime_gap(self):
        # Tuesday 2026-01-13 07:15 JST — winter collector run.
        got = market_date_for("US.AAPL", now=_jst(2026, 1, 13, 7, 15))
        self.assertEqual(got, date(2026, 1, 12))

    def test_jp_market_same_day_when_trading_day(self):
        got = market_date_for("JP.7203", now=_jst(2026, 8, 18, 7, 15))
        self.assertEqual(got, date(2026, 8, 18))

    def test_jp_market_walks_back_over_weekend(self):
        got = market_date_for("JP.7203", now=_jst(2026, 8, 22, 7, 15))
        self.assertEqual(got, date(2026, 8, 21))

    def test_defaults_to_now_when_omitted(self):
        # Smoke test only: must not raise, must return a date instance.
        got = market_date_for("US.AAPL")
        self.assertIsInstance(got, date)


if __name__ == "__main__":
    unittest.main()
