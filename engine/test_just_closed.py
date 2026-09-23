"""
Unit tests for engine/market_hours.py::just_closed() — the one-time
"美股收盘" push must fire shortly after close, survive a delayed scan pass,
and stay silent on a restart hours later (2026-09-23 bogus push at 09:15 ET).

Run:  python -m unittest engine.test_just_closed -v
"""
import unittest
from datetime import datetime

import pytz

from engine.market_hours import just_closed

_JST = pytz.timezone("Asia/Tokyo")


def _jst(y, m, d, hh, mm=0):
    return _JST.localize(datetime(y, m, d, hh, mm))


class JustClosedTestCase(unittest.TestCase):

    # Wed 2026-09-23 JST morning = close of Tue 2026-09-22 US session (summer, 05:00 JST).

    def test_before_notify_start(self):
        self.assertFalse(just_closed(now=_jst(2026, 9, 23, 5, 3)))

    def test_fires_after_close(self):
        self.assertTrue(just_closed(now=_jst(2026, 9, 23, 5, 5)))

    def test_delayed_pass_still_fires(self):
        # 07-24 case: a stuck OpenD delayed the pass ~20min.
        self.assertTrue(just_closed(now=_jst(2026, 9, 23, 5, 30)))

    def test_silent_on_premarket_restart(self):
        # 2026-09-23 22:15 JST (09:15 ET) restart — the reported bug.
        self.assertFalse(just_closed(now=_jst(2026, 9, 23, 22, 15)))

    def test_silent_on_afternoon_restart(self):
        self.assertFalse(just_closed(now=_jst(2026, 9, 23, 16, 9)))

    def test_silent_mid_session(self):
        # 2026-09-22 bug: restart during a live session.
        self.assertFalse(just_closed(now=_jst(2026, 9, 23, 23, 20)))

    def test_silent_after_non_trading_day(self):
        # Sun 2026-09-20 JST morning follows Sat 09-19 — no session.
        self.assertFalse(just_closed(now=_jst(2026, 9, 20, 5, 10)))

    def test_winter_close(self):
        # Wed 2026-12-02 JST: winter close 06:00 JST.
        self.assertFalse(just_closed(now=_jst(2026, 12, 2, 5, 30)))
        self.assertTrue(just_closed(now=_jst(2026, 12, 2, 6, 10)))
        self.assertFalse(just_closed(now=_jst(2026, 12, 2, 12, 0)))


if __name__ == "__main__":
    unittest.main()
