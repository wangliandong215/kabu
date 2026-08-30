"""
test_execute_emergency_rebalance.py -- v2.10.1 market-hours pre-check gate.

Verifies the 2026-08-29 fix (see execute_emergency_rebalance.py module
docstring, prompted by a real test run that submitted a QQQ DAY LIMIT order
while the US market was closed -- it correctly failed-stopped with zero
state change, but shouldn't have been attempted at all):
  1. Closed market -> exits immediately, before even fetching broker state
     (let alone placing an order) -- proves "no orders submitted" at the
     cheapest possible point, not just "orders get cancelled afterward".
  2. Open market + not EMERGENCY -> unaffected, existing behavior intact.
  3. Open market + EMERGENCY -> unaffected, still reaches the preview.

Run:  python -m unittest test_execute_emergency_rebalance -v
"""
import sys
import unittest
from unittest.mock import patch

import config
import execute_emergency_rebalance as script
from portfolio.broker_state import BrokerPosition, BrokerState


class TestMarketHoursGate(unittest.TestCase):

    def setUp(self):
        self._orig = {
            "is_open": script.is_open,
            "fetch_broker_state": script.broker_state_mod.fetch_broker_state,
            "Portfolio": script.Portfolio,
            "scan": script.scan,
            "plan_rebalance": script.prm.plan_rebalance,
            "_run_emergency_rebalance": script._run_emergency_rebalance,
            "argv": sys.argv,
        }
        sys.argv = ["execute_emergency_rebalance.py"]
        self.fetch_calls = 0
        self.rebalance_calls = 0

        def _counting_fetch(trd_env):
            self.fetch_calls += 1
            return BrokerState(positions={}, cash=-80_000.0,
                               total_assets=1_000_000.0, long_mv=1_100_000.0)  # 110% EMERGENCY
        script.broker_state_mod.fetch_broker_state = _counting_fetch

        class _FakePortfolio:
            data = {"positions": {}}
        script.Portfolio = lambda: _FakePortfolio()
        script.scan = lambda codes, **kw: {}

        def _counting_rebalance(*a, **kw):
            self.rebalance_calls += 1
            return []
        script._run_emergency_rebalance = _counting_rebalance

    def tearDown(self):
        script.is_open = self._orig["is_open"]
        script.broker_state_mod.fetch_broker_state = self._orig["fetch_broker_state"]
        script.Portfolio = self._orig["Portfolio"]
        script.scan = self._orig["scan"]
        script.prm.plan_rebalance = self._orig["plan_rebalance"]
        script._run_emergency_rebalance = self._orig["_run_emergency_rebalance"]
        sys.argv = self._orig["argv"]

    def test_closed_market_exits_before_any_broker_call(self):
        script.is_open = lambda code: False

        rc = script.main()

        self.assertEqual(rc, 0, "closed market must be a clean/normal exit, not an error")
        self.assertEqual(self.fetch_calls, 0,
                         "must not even query broker state when the market is closed")
        self.assertEqual(self.rebalance_calls, 0)

    def test_open_market_not_emergency_reaches_broker_check(self):
        script.is_open = lambda code: True
        script.broker_state_mod.fetch_broker_state = lambda trd_env: BrokerState(
            positions={}, cash=100_000.0, total_assets=1_000_000.0, long_mv=800_000.0)  # NORMAL

        rc = script.main()

        self.assertEqual(rc, 0)
        self.assertEqual(self.rebalance_calls, 0, "NORMAL tier has nothing to rebalance")

    def test_open_market_emergency_reaches_preview(self):
        script.is_open = lambda code: True
        qqq = config.QQQ_CORE_CODE

        def _fetch_with_qqq(trd_env):
            self.fetch_calls += 1
            return BrokerState(
                positions={qqq: BrokerPosition(qqq, 473, 716.0, 342_087.79, 723.23)},
                cash=-50_000.0, total_assets=280_000.0, long_mv=342_087.79,  # ~122% EMERGENCY
            )
        script.broker_state_mod.fetch_broker_state = _fetch_with_qqq

        class _FakePortfolioWithQqq:
            data = {"positions": {qqq: {"entry_price": 716.0, "qty": 473, "strategy": "core_etf"}}}
        script.Portfolio = lambda: _FakePortfolioWithQqq()

        with patch("builtins.input", return_value="not EXECUTE") as mock_input:
            rc = script.main()

        self.assertEqual(rc, 0)
        self.assertEqual(self.fetch_calls, 1, "open market must proceed to a real broker check")
        mock_input.assert_called_once()   # actually reached the preview + confirmation prompt
        self.assertEqual(self.rebalance_calls, 0,
                         "declining the confirmation prompt must never call the executor")


if __name__ == "__main__":
    unittest.main()
