"""
Unit tests for engine/event_risk.py (v2.11 Event Risk Layer — Earnings).

Pure unit tests: data.earnings.get_earnings_risk() is monkeypatched so
nothing here touches OpenD. Covers the fail-open contract, the config-driven
block-window boundary, and BMO/AMC session parsing.

Run:  python -m unittest engine.test_event_risk -v
"""
import unittest
from datetime import datetime, timedelta

import config
import data.earnings as earnings_data
import engine.event_risk as event_risk


class EventRiskTestCase(unittest.TestCase):

    def setUp(self):
        self._orig_get_earnings_risk = earnings_data.get_earnings_risk
        self._orig_enabled = config.EVENT_RISK_ENABLED
        self._orig_block_days = config.EVENT_RISK_EARNINGS_BLOCK_DAYS
        config.EVENT_RISK_ENABLED = True
        config.EVENT_RISK_EARNINGS_BLOCK_DAYS = 1

    def tearDown(self):
        earnings_data.get_earnings_risk = self._orig_get_earnings_risk
        config.EVENT_RISK_ENABLED = self._orig_enabled
        config.EVENT_RISK_EARNINGS_BLOCK_DAYS = self._orig_block_days

    def test_has_earnings_risk_true_when_earnings_is_today(self):
        today = datetime.now().date().strftime("%Y-%m-%d")
        earnings_data.get_earnings_risk = lambda code, trade_date=None: {
            "security": "US.TEST", "earnings_date": today, "pub_type": "AfterMarket"}

        self.assertTrue(event_risk.has_earnings_risk("US.TEST"))
        self.assertEqual(event_risk.days_to_earnings("US.TEST"), 0)

    def test_has_earnings_risk_false_when_far_outside_window(self):
        far = (datetime.now().date() + timedelta(days=60)).strftime("%Y-%m-%d")
        earnings_data.get_earnings_risk = lambda code, trade_date=None: {
            "security": "US.TEST", "earnings_date": far, "pub_type": "BeforeMarket"}

        self.assertFalse(event_risk.has_earnings_risk("US.TEST"))
        d = event_risk.days_to_earnings("US.TEST")
        self.assertGreater(abs(d), config.EVENT_RISK_EARNINGS_BLOCK_DAYS)

    def test_has_earnings_risk_false_when_no_earnings_data(self):
        earnings_data.get_earnings_risk = lambda code, trade_date=None: None

        self.assertFalse(event_risk.has_earnings_risk("US.TEST"))
        self.assertIsNone(event_risk.days_to_earnings("US.TEST"))
        self.assertIsNone(event_risk.earnings_date("US.TEST"))

    def test_fail_open_on_lookup_error(self):
        def _raise(code, trade_date=None):
            raise RuntimeError("OpenD unreachable")
        earnings_data.get_earnings_risk = _raise

        self.assertFalse(event_risk.has_earnings_risk("US.TEST"),
                          "a data-lookup failure must fail open, not block trading")
        self.assertIsNone(event_risk.earnings_date("US.TEST"))

    def test_disabled_via_config_short_circuits(self):
        config.EVENT_RISK_ENABLED = False
        today = datetime.now().date().strftime("%Y-%m-%d")
        earnings_data.get_earnings_risk = lambda code, trade_date=None: {
            "security": "US.TEST", "earnings_date": today, "pub_type": "AfterMarket"}

        self.assertFalse(event_risk.has_earnings_risk("US.TEST"),
                          "EVENT_RISK_ENABLED=False must disable the gate entirely")

    def test_earnings_session_after_market(self):
        earnings_data.get_earnings_risk = lambda code, trade_date=None: {
            "security": "US.TEST", "earnings_date": "2027-01-01", "pub_type": "AfterMarket"}
        self.assertEqual(event_risk.earnings_session("US.TEST"), "AMC")

    def test_earnings_session_before_market(self):
        earnings_data.get_earnings_risk = lambda code, trade_date=None: {
            "security": "US.TEST", "earnings_date": "2027-01-01", "pub_type": "BeforeMarket"}
        self.assertEqual(event_risk.earnings_session("US.TEST"), "BMO")

    def test_earnings_session_unknown(self):
        earnings_data.get_earnings_risk = lambda code, trade_date=None: {
            "security": "US.TEST", "earnings_date": "2027-01-01", "pub_type": "Unspecified"}
        self.assertIsNone(event_risk.earnings_session("US.TEST"))


if __name__ == "__main__":
    unittest.main()
