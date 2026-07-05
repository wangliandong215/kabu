"""
Unit tests for engine/runner.py::_attempt_active_replacement() -- v2.3
Portfolio Capacity Manager migrated from backtest_portfolio.py into the
live/paper-trading pipeline.

Rather than mocking the entire run_once() pipeline (500+ lines, depends on
live quotes/news/fundamentals), these tests exercise the extracted
_attempt_active_replacement() function directly against a real Portfolio
backed by a temp JSON file, with only _place_order/get_price monkeypatched
(matching the module-level `from data.fetcher import get_price` /
module-local `_place_order` functions runner.py actually calls).

Run:  python -m unittest engine.test_runner_replacement -v
"""
import tempfile
import unittest
from pathlib import Path

import config
import engine.runner as runner
from engine.scoring import LABEL_FULL, LABEL_OBSERVATION
from portfolio.tracker import Portfolio


class TestAttemptActiveReplacement(unittest.TestCase):

    def setUp(self):
        self._orig_enable = config.ENABLE_ACTIVE_REPLACEMENT
        self._orig_advantage = config.REPLACEMENT_MARGIN
        config.ENABLE_ACTIVE_REPLACEMENT = True
        config.REPLACEMENT_MARGIN = 10.0

        self._tmpdir = tempfile.TemporaryDirectory()
        self.portfolio = Portfolio(path=Path(self._tmpdir.name) / "positions.json")

        self._orig_place_order = runner._place_order
        self._orig_get_price = runner.get_price
        self.placed_orders = []

    def tearDown(self):
        config.ENABLE_ACTIVE_REPLACEMENT = self._orig_enable
        config.REPLACEMENT_MARGIN = self._orig_advantage
        runner._place_order = self._orig_place_order
        runner.get_price = self._orig_get_price
        self._tmpdir.cleanup()

    def _mock_place_order(self, order_id_to_return="FAKE123"):
        def _mock(code, side, qty, price, trd_env, env_label, confirmed):
            self.placed_orders.append({"code": code, "side": side, "qty": qty, "price": price})
            return order_id_to_return if confirmed else ""
        runner._place_order = _mock

    def _seed_position(self, code, score_label, total_score, qty=10):
        self.portfolio.data["positions"][code] = {
            "side": "BUY", "entry_price": 10.0, "avg_cost": 10.0, "qty": qty,
            "signal_strength": 0.3, "strategy": "atr_breakout", "entry_atr": 0.1,
            "atr_mult": config.ATR_MULT_BASE, "breakeven_locked": False,
            "trail_stop": None, "entry_time": "2020-01-01T00:00:00",
            "score_label": score_label, "total_score": total_score,
        }
        self.portfolio._save()

    def test_replacement_succeeds_and_closes_victim(self):
        self._seed_position("US.OBS", LABEL_OBSERVATION, 45.0)
        self._mock_place_order()
        runner.get_price = lambda code: 12.0

        victim = runner._attempt_active_replacement(
            self.portfolio, incoming_score=90.0,
            trd_env=None, env_label="SIMULATE", confirmed=True, results={},
        )
        self.assertEqual(victim, "US.OBS")
        self.assertIsNone(self.portfolio.get_position("US.OBS"))
        self.assertEqual(len(self.placed_orders), 1)
        self.assertEqual(self.placed_orders[0], {"code": "US.OBS", "side": "SELL",
                                                   "qty": 10, "price": 12.0})

    def test_no_observation_held_falls_back_to_none(self):
        self._seed_position("US.FULL_HELD", LABEL_FULL, 95.0)
        self._mock_place_order()

        victim = runner._attempt_active_replacement(
            self.portfolio, incoming_score=90.0,
            trd_env=None, env_label="SIMULATE", confirmed=True, results={},
        )
        self.assertIsNone(victim)
        self.assertEqual(len(self.placed_orders), 0)
        self.assertIsNotNone(self.portfolio.get_position("US.FULL_HELD"))

    def test_advantage_insufficient_returns_none(self):
        self._seed_position("US.OBS", LABEL_OBSERVATION, 85.0)   # 90-85=5 < 10
        self._mock_place_order()

        victim = runner._attempt_active_replacement(
            self.portfolio, incoming_score=90.0,
            trd_env=None, env_label="SIMULATE", confirmed=True, results={},
        )
        self.assertIsNone(victim)
        self.assertEqual(len(self.placed_orders), 0)
        self.assertIsNotNone(self.portfolio.get_position("US.OBS"))

    def test_sell_order_failure_leaves_position_untouched(self):
        # Simulates a broker rejection (_place_order returns "" even though
        # confirmed=True) -- this is the stricter order_id gate this feature
        # adds on top of the existing exit-loop's weaker "confirmed alone"
        # gate (see engine/runner.py::_attempt_active_replacement docstring).
        self._seed_position("US.OBS", LABEL_OBSERVATION, 45.0)
        self._mock_place_order(order_id_to_return="")
        runner.get_price = lambda code: 12.0

        victim = runner._attempt_active_replacement(
            self.portfolio, incoming_score=90.0,
            trd_env=None, env_label="SIMULATE", confirmed=True, results={},
        )
        self.assertIsNone(victim)
        self.assertIsNotNone(self.portfolio.get_position("US.OBS"))   # untouched

    def test_dry_run_reports_victim_without_mutating_portfolio(self):
        self._seed_position("US.OBS", LABEL_OBSERVATION, 45.0)
        self._mock_place_order()
        runner.get_price = lambda code: 12.0

        victim = runner._attempt_active_replacement(
            self.portfolio, incoming_score=90.0,
            trd_env=None, env_label="SIMULATE", confirmed=False, results={},
        )
        self.assertEqual(victim, "US.OBS")
        # dry run: no real order call succeeds, so no state mutation --
        # matches the rest of runner.py's "confirmed=False never touches
        # broker or local state" convention.
        self.assertIsNotNone(self.portfolio.get_position("US.OBS"))

    def test_victim_price_from_results_dict_preferred_over_get_price(self):
        self._seed_position("US.OBS", LABEL_OBSERVATION, 45.0)
        self._mock_place_order()
        runner.get_price = lambda code: (_ for _ in ()).throw(
            AssertionError("get_price should not be called when results has current_price"))

        victim = runner._attempt_active_replacement(
            self.portfolio, incoming_score=90.0,
            trd_env=None, env_label="SIMULATE", confirmed=True,
            results={"US.OBS": {"current_price": 13.5}},
        )
        self.assertEqual(victim, "US.OBS")
        self.assertEqual(self.placed_orders[0]["price"], 13.5)

    def test_disabled_flag_is_enforced_by_caller_not_this_function(self):
        # ENABLE_ACTIVE_REPLACEMENT is checked in run_once() before calling
        # _attempt_active_replacement, not inside the function itself --
        # this test documents that division of responsibility so it doesn't
        # get "fixed" into a duplicate check by accident later.
        self._seed_position("US.OBS", LABEL_OBSERVATION, 45.0)
        self._mock_place_order()
        runner.get_price = lambda code: 12.0
        config.ENABLE_ACTIVE_REPLACEMENT = False

        victim = runner._attempt_active_replacement(
            self.portfolio, incoming_score=90.0,
            trd_env=None, env_label="SIMULATE", confirmed=True, results={},
        )
        self.assertEqual(victim, "US.OBS")


if __name__ == "__main__":
    unittest.main()
