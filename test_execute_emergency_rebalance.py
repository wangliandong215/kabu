"""
test_execute_emergency_rebalance.py -- v2.10.1 market-hours pre-check gate
+ v2.10.1 (2026-08-31) audit-chain completion.

Verifies two fixes to the manual approval entrypoint (see module docstring
in execute_emergency_rebalance.py for the incidents that prompted each):

  Market-hours gate (2026-08-29, a real test run submitted a QQQ DAY LIMIT
  order while the US market was closed -- it correctly failed-stopped with
  zero state change, but shouldn't have been attempted at all):
    1. Closed market -> exits immediately, before even fetching broker state
       (let alone placing an order).
    2. Open market + not EMERGENCY -> unaffected, existing behavior intact.
    3. Open market + EMERGENCY -> unaffected, still reaches the preview.

  Audit-chain completion (2026-08-31, the first real EXECUTE left no entry
  in risk_state_log.jsonl/reconciliation_diffs.jsonl because this script
  never called reconcile()/log_diff()/log_snapshot() the way run_once()
  does):
    4. reconcile()+log_diff() run on every successful broker fetch,
       regardless of tier.
    5. log_snapshot() runs exactly once per run, on every exit path past the
       broker fetch (NORMAL, EMERGENCY-declined, EMERGENCY-executed) -- with
       the real executed-orders list when a rebalance actually happened.

Run:  python -m unittest test_execute_emergency_rebalance -v
"""
import sys
import unittest
from unittest.mock import patch

import config
import execute_emergency_rebalance as script
import test_support
from portfolio.broker_state import BrokerPosition, BrokerState


class TestExecuteEmergencyRebalance(unittest.TestCase):

    def setUp(self):
        self._alert_handlers = test_support.mute_alert_file_logging()
        self._orig = {
            "is_open": script.is_open,
            "fetch_broker_state": script.broker_state_mod.fetch_broker_state,
            "Portfolio": script.Portfolio,
            "scan": script.scan,
            "plan_rebalance": script.prm.plan_rebalance,
            "_run_emergency_rebalance": script._run_emergency_rebalance,
            "log_diff": script.prm.log_diff,
            "log_snapshot": script.prm.log_snapshot,
            "check_and_update_diff_growth": script.prm.check_and_update_diff_growth,
            "argv": sys.argv,
        }
        sys.argv = ["execute_emergency_rebalance.py"]
        self.fetch_calls = 0
        self.rebalance_calls = 0
        self.logged_diffs = []
        self.snapshot_calls = []

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

        # log_diff/log_snapshot/check_and_update_diff_growth write real
        # JSONL/JSON files under C:\KabuData\portfolio\ -- mocked here (same
        # as engine/test_portfolio_risk_manager_gate.py already does) both to
        # keep the suite hermetic and to let assertions inspect exactly what
        # would have been recorded.
        script.prm.log_diff = lambda d: self.logged_diffs.append(d)
        script.prm.log_snapshot = lambda assessment, diffs, executed: \
            self.snapshot_calls.append((assessment, diffs, executed))
        script.prm.check_and_update_diff_growth = lambda diffs: {}

    def tearDown(self):
        script.is_open = self._orig["is_open"]
        script.broker_state_mod.fetch_broker_state = self._orig["fetch_broker_state"]
        script.Portfolio = self._orig["Portfolio"]
        script.scan = self._orig["scan"]
        script.prm.plan_rebalance = self._orig["plan_rebalance"]
        script._run_emergency_rebalance = self._orig["_run_emergency_rebalance"]
        script.prm.log_diff = self._orig["log_diff"]
        script.prm.log_snapshot = self._orig["log_snapshot"]
        script.prm.check_and_update_diff_growth = self._orig["check_and_update_diff_growth"]
        sys.argv = self._orig["argv"]
        test_support.unmute_alert_file_logging(self._alert_handlers)

    def _fetch_with_qqq(self, qqq_tracker_qty=473):
        qqq = config.QQQ_CORE_CODE

        def _fetch(trd_env):
            self.fetch_calls += 1
            return BrokerState(
                positions={qqq: BrokerPosition(qqq, 473, 716.0, 342_087.79, 723.23)},
                cash=-50_000.0, total_assets=280_000.0, long_mv=342_087.79,  # ~122% EMERGENCY
            )
        script.broker_state_mod.fetch_broker_state = _fetch

        class _FakePortfolioWithQqq:
            data = {"positions": {qqq: {"entry_price": 716.0, "qty": qqq_tracker_qty,
                                        "strategy": "core_etf"}}}
        script.Portfolio = lambda: _FakePortfolioWithQqq()
        return qqq

    # ── market-hours gate ────────────────────────────────────────────────────

    def test_closed_market_exits_before_any_broker_call(self):
        script.is_open = lambda code: False

        rc = script.main()

        self.assertEqual(rc, 0, "closed market must be a clean/normal exit, not an error")
        self.assertEqual(self.fetch_calls, 0,
                         "must not even query broker state when the market is closed")
        self.assertEqual(self.rebalance_calls, 0)
        self.assertEqual(self.snapshot_calls, [],
                         "no broker data was ever fetched, so there's nothing to snapshot")

    def test_open_market_not_emergency_reaches_broker_check(self):
        script.is_open = lambda code: True
        script.broker_state_mod.fetch_broker_state = lambda trd_env: BrokerState(
            positions={}, cash=100_000.0, total_assets=1_000_000.0, long_mv=800_000.0)  # NORMAL

        rc = script.main()

        self.assertEqual(rc, 0)
        self.assertEqual(self.rebalance_calls, 0, "NORMAL tier has nothing to rebalance")

    def test_open_market_emergency_reaches_preview(self):
        script.is_open = lambda code: True
        self._fetch_with_qqq()

        with patch("builtins.input", return_value="not EXECUTE") as mock_input:
            rc = script.main()

        self.assertEqual(rc, 0)
        self.assertEqual(self.fetch_calls, 1, "open market must proceed to a real broker check")
        mock_input.assert_called_once()   # actually reached the preview + confirmation prompt
        self.assertEqual(self.rebalance_calls, 0,
                         "declining the confirmation prompt must never call the executor")

    # ── audit-chain completion (2026-08-31) ─────────────────────────────────

    def test_reconcile_and_log_diff_run_even_on_normal_tier(self):
        """reconcile()/log_diff() must fire on every successful broker fetch,
        not only when EMERGENCY -- matches run_once()'s automatic path."""
        script.is_open = lambda code: True
        qqq = config.QQQ_CORE_CODE
        script.broker_state_mod.fetch_broker_state = lambda trd_env: BrokerState(
            positions={qqq: BrokerPosition(qqq, 473, 716.0, 342_087.79, 723.23)},
            cash=600_000.0, total_assets=1_000_000.0, long_mv=342_087.79)  # NORMAL, but QQQ mismatched

        class _FakePortfolioMismatchedQqq:
            data = {"positions": {qqq: {"entry_price": 716.0, "qty": 349, "strategy": "core_etf"}}}
        script.Portfolio = lambda: _FakePortfolioMismatchedQqq()

        rc = script.main()

        self.assertEqual(rc, 0)
        self.assertEqual(len(self.logged_diffs), 1)
        self.assertEqual(self.logged_diffs[0].code, qqq)
        self.assertEqual(self.logged_diffs[0].diff_qty, 473 - 349)

    def test_log_snapshot_called_once_when_not_emergency(self):
        script.is_open = lambda code: True
        script.broker_state_mod.fetch_broker_state = lambda trd_env: BrokerState(
            positions={}, cash=100_000.0, total_assets=1_000_000.0, long_mv=800_000.0)  # NORMAL

        rc = script.main()

        self.assertEqual(rc, 0)
        self.assertEqual(len(self.snapshot_calls), 1)
        assessment, diffs, executed = self.snapshot_calls[0]
        self.assertEqual(assessment.tier, script.prm.TIER_NORMAL)
        self.assertEqual(executed, [], "no rebalance happened -- must snapshot an empty executed list")

    def test_log_snapshot_called_once_when_emergency_declined(self):
        script.is_open = lambda code: True
        self._fetch_with_qqq()

        with patch("builtins.input", return_value="not EXECUTE"):
            rc = script.main()

        self.assertEqual(rc, 0)
        self.assertEqual(len(self.snapshot_calls), 1)
        assessment, diffs, executed = self.snapshot_calls[0]
        self.assertEqual(assessment.tier, script.prm.TIER_EMERGENCY)
        self.assertEqual(executed, [], "declined confirmation -- must snapshot an empty executed list")

    def test_log_snapshot_called_once_with_real_executed_orders(self):
        script.is_open = lambda code: True
        self._fetch_with_qqq()
        fake_order = {"code": config.QQQ_CORE_CODE, "sell_qty": 118, "price": 716.43,
                      "reason": "QQQ_EXCESS_TRIM"}

        def _counting_rebalance(*a, **kw):
            self.rebalance_calls += 1
            return [fake_order]
        script._run_emergency_rebalance = _counting_rebalance

        with patch("builtins.input", return_value="EXECUTE"):
            rc = script.main()

        self.assertEqual(rc, 0)
        self.assertEqual(self.rebalance_calls, 1)
        self.assertEqual(len(self.snapshot_calls), 1)
        _, _, executed = self.snapshot_calls[0]
        self.assertEqual(executed, [fake_order],
                         "the snapshot must carry the REAL executed orders, not an empty/planned list")


if __name__ == "__main__":
    unittest.main()
