"""
Unit tests for engine/market_preflight.py (v2.11 OpenD Preflight check).

Pure unit tests: market_hours.is_open() and the module's own
_fetch_global_state() (the @common.retry()-wrapped OpenD call) are
monkeypatched so nothing here touches OpenD or sleeps through a real retry
backoff.

Run:  python -m unittest engine.test_market_preflight -v
"""
import unittest

import config
import engine.market_preflight as market_preflight


class MarketPreflightTestCase(unittest.TestCase):

    def setUp(self):
        self._orig_is_open = market_preflight.market_hours.is_open
        self._orig_fetch = market_preflight._fetch_global_state
        self._orig_blocked_states = config.PREFLIGHT_BLOCKED_MARKET_STATES
        market_preflight.market_hours.is_open = lambda code: True

    def tearDown(self):
        market_preflight.market_hours.is_open = self._orig_is_open
        market_preflight._fetch_global_state = self._orig_fetch
        config.PREFLIGHT_BLOCKED_MARKET_STATES = self._orig_blocked_states

    def _ok_state(self, market_us="MORNING"):
        return {"qot_logined": True, "trd_logined": True, "market_us": market_us}

    def test_ok_when_everything_normal(self):
        market_preflight._fetch_global_state = lambda: self._ok_state()
        result = market_preflight.check("US.AAPL")
        self.assertTrue(result.ok)

    def test_fails_when_market_hours_closed(self):
        market_preflight.market_hours.is_open = lambda code: False
        market_preflight._fetch_global_state = lambda: self._ok_state()
        result = market_preflight.check("US.AAPL")
        self.assertFalse(result.ok)
        self.assertIn("market_hours", result.reason)

    def test_fails_when_global_state_call_fails(self):
        market_preflight._fetch_global_state = lambda: None
        result = market_preflight.check("US.AAPL")
        self.assertFalse(result.ok)

    def test_fails_when_global_state_raises(self):
        def _raise():
            raise RuntimeError("OpenD down")
        market_preflight._fetch_global_state = _raise
        result = market_preflight.check("US.AAPL")
        self.assertFalse(result.ok)
        self.assertIn("exception", result.reason)

    def test_fails_when_quote_not_logged_in(self):
        market_preflight._fetch_global_state = lambda: {
            "qot_logined": False, "trd_logined": True, "market_us": "MORNING"}
        result = market_preflight.check("US.AAPL")
        self.assertFalse(result.ok)
        self.assertIn("quote", result.reason)

    def test_fails_when_trade_not_logged_in(self):
        market_preflight._fetch_global_state = lambda: {
            "qot_logined": True, "trd_logined": False, "market_us": "MORNING"}
        result = market_preflight.check("US.AAPL")
        self.assertFalse(result.ok)
        self.assertIn("trade", result.reason)

    def test_accepts_string_1_0_form_defensively(self):
        """The SDK's own doc/skill-script describes qot_logined/trd_logined
        as '1'/'0' strings, even though a live OpenD call returns native
        bools (confirmed 2026-08-30) — accept both so a future SDK/OpenD
        version that reverts to strings doesn't silently fail-closed."""
        market_preflight._fetch_global_state = lambda: {
            "qot_logined": "1", "trd_logined": "1", "market_us": "MORNING"}
        result = market_preflight.check("US.AAPL")
        self.assertTrue(result.ok)

    def test_fails_when_market_state_blocked(self):
        config.PREFLIGHT_BLOCKED_MARKET_STATES = {"CLOSED", "NONE", "REST"}
        market_preflight._fetch_global_state = lambda: self._ok_state(market_us="CLOSED")
        result = market_preflight.check("US.AAPL")
        self.assertFalse(result.ok)
        self.assertIn("market_us", result.reason)

    def test_passes_for_market_state_not_in_blocklist(self):
        market_preflight._fetch_global_state = lambda: self._ok_state(market_us="AUCTION")
        result = market_preflight.check("US.AAPL")
        self.assertTrue(result.ok)


if __name__ == "__main__":
    unittest.main()
