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
from datetime import datetime, timedelta
from pathlib import Path

import config
import engine.runner as runner
import test_support
from engine.scoring import LABEL_FULL, LABEL_OBSERVATION
from portfolio import capacity_manager
from portfolio.tracker import Portfolio


class TestAttemptActiveReplacement(unittest.TestCase):

    def setUp(self):
        self._alert_handlers = test_support.mute_alert_file_logging()
        self._orig_enable = config.ENABLE_ACTIVE_REPLACEMENT
        self._orig_advantage = config.REPLACEMENT_MARGIN
        self._orig_min_score = config.REPLACEMENT_MIN_NEW_SCORE
        self._orig_max_pool = config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE
        self._orig_block_sector = config.REPLACEMENT_BLOCK_SAME_SECTOR
        config.ENABLE_ACTIVE_REPLACEMENT = True
        config.REPLACEMENT_MARGIN = 10.0
        # 这个文件测的是_attempt_active_replacement()自身的下单/回滚/
        # dry-run行为，不是v2.4阶段三转正的三个消融门（那三个门的行为由
        # portfolio/test_capacity_manager.py和下面的
        # TestBacktestRunnerConsistency覆盖）——显式关闭，避免默认值
        # 95分/同板块拦截干扰这组测试用的90分虚构信号。
        config.REPLACEMENT_MIN_NEW_SCORE = None
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = None
        config.REPLACEMENT_BLOCK_SAME_SECTOR = False

        self._tmpdir = tempfile.TemporaryDirectory()
        self.portfolio = Portfolio(path=Path(self._tmpdir.name) / "positions.json")

        self._orig_place_order = runner._place_order
        self._orig_get_price = runner.get_price
        self._orig_trade_sell = runner.alert.trade_sell
        runner.alert.trade_sell = lambda *a, **kw: None
        self.placed_orders = []

    def tearDown(self):
        config.ENABLE_ACTIVE_REPLACEMENT = self._orig_enable
        config.REPLACEMENT_MARGIN = self._orig_advantage
        config.REPLACEMENT_MIN_NEW_SCORE = self._orig_min_score
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = self._orig_max_pool
        config.REPLACEMENT_BLOCK_SAME_SECTOR = self._orig_block_sector
        runner._place_order = self._orig_place_order
        runner.get_price = self._orig_get_price
        runner.alert.trade_sell = self._orig_trade_sell
        self._tmpdir.cleanup()
        test_support.unmute_alert_file_logging(self._alert_handlers)

    def _mock_place_order(self, dealt_qty_override=None):
        def _mock(code, side, qty, price, trd_env, env_label, confirmed):
            self.placed_orders.append({"code": code, "side": side, "qty": qty, "price": price})
            dealt = qty if dealt_qty_override is None else dealt_qty_override
            if not confirmed:
                dealt = 0
            return {"order_id": "FAKE123" if dealt > 0 else "", "dealt_qty": float(dealt),
                    "dealt_avg_price": price if dealt > 0 else 0.0,
                    "status": "FILLED_ALL" if dealt > 0 else "CANCELLED_ALL"}
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
            self.portfolio, incoming_code="US.NEW", incoming_score=90.0,
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
            self.portfolio, incoming_code="US.NEW", incoming_score=90.0,
            trd_env=None, env_label="SIMULATE", confirmed=True, results={},
        )
        self.assertIsNone(victim)
        self.assertEqual(len(self.placed_orders), 0)
        self.assertIsNotNone(self.portfolio.get_position("US.FULL_HELD"))

    def test_advantage_insufficient_returns_none(self):
        self._seed_position("US.OBS", LABEL_OBSERVATION, 85.0)   # 90-85=5 < 10
        self._mock_place_order()

        victim = runner._attempt_active_replacement(
            self.portfolio, incoming_code="US.NEW", incoming_score=90.0,
            trd_env=None, env_label="SIMULATE", confirmed=True, results={},
        )
        self.assertIsNone(victim)
        self.assertEqual(len(self.placed_orders), 0)
        self.assertIsNotNone(self.portfolio.get_position("US.OBS"))

    def test_sell_order_failure_leaves_position_untouched(self):
        # Simulates an order that never filled (dealt_qty=0 even though
        # confirmed=True and order_id came back) -- the dealt_qty gate this
        # feature uses on top of the existing exit-loop's same gate (see
        # engine/runner.py::_attempt_active_replacement docstring).
        self._seed_position("US.OBS", LABEL_OBSERVATION, 45.0)
        self._mock_place_order(dealt_qty_override=0)
        runner.get_price = lambda code: 12.0

        victim = runner._attempt_active_replacement(
            self.portfolio, incoming_code="US.NEW", incoming_score=90.0,
            trd_env=None, env_label="SIMULATE", confirmed=True, results={},
        )
        self.assertIsNone(victim)
        self.assertIsNotNone(self.portfolio.get_position("US.OBS"))   # untouched

    def test_dry_run_reports_victim_without_mutating_portfolio(self):
        self._seed_position("US.OBS", LABEL_OBSERVATION, 45.0)
        self._mock_place_order()
        runner.get_price = lambda code: 12.0

        victim = runner._attempt_active_replacement(
            self.portfolio, incoming_code="US.NEW", incoming_score=90.0,
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
            self.portfolio, incoming_code="US.NEW", incoming_score=90.0,
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
            self.portfolio, incoming_code="US.NEW", incoming_score=90.0,
            trd_env=None, env_label="SIMULATE", confirmed=True, results={},
        )
        self.assertEqual(victim, "US.OBS")


class TestBacktestRunnerConsistency(unittest.TestCase):
    """2026-07-06: 证明 runner.py 的 _attempt_active_replacement() 和
    backtest_portfolio.py 非RSL分支用的是**同一个**
    portfolio.capacity_manager.evaluate_replacement() 决策入口——同一批
    持仓+同一个incoming信号，两条路径必须给出完全一致的Replace决策，
    否则就是"回测和实盘各自维护一套规则"又漂移回来了。runner一侧走
    _attempt_active_replacement()（真实Portfolio对象+mock下单）；
    "backtest一侧"直接调用 evaluate_replacement()（不需要真的跑一遍
    backtest_portfolio.py的逐日循环，因为它的非RSL分支就是原样调用这个
    函数，见backtest_portfolio.py该分支的注释）——两侧用完全相同的
    held_positions/incoming_code/incoming_score/current_day_idx构造，
    唯一允许不同的是调用方式本身。"""

    def setUp(self):
        self._alert_handlers = test_support.mute_alert_file_logging()
        self._orig_margin = config.REPLACEMENT_MARGIN
        self._orig_min_score = config.REPLACEMENT_MIN_NEW_SCORE
        self._orig_max_pool = config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE
        self._orig_block_sector = config.REPLACEMENT_BLOCK_SAME_SECTOR
        self._orig_sector_map = dict(config.SECTOR_MAP)
        self._orig_enable = config.ENABLE_ACTIVE_REPLACEMENT
        config.REPLACEMENT_MARGIN = 10.0
        config.ENABLE_ACTIVE_REPLACEMENT = True
        # 基线关闭三个消融门，每个测试按需自己打开要验证的那一个——这样
        # incoming_score=90的测试信号不会被生产默认值95分门槛挡住。
        config.REPLACEMENT_MIN_NEW_SCORE = None
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = None
        config.REPLACEMENT_BLOCK_SAME_SECTOR = False
        config.SECTOR_MAP["US.NEW"] = "tech"
        config.SECTOR_MAP["US.OBS_TECH"] = "tech"
        config.SECTOR_MAP["US.OBS_ENERGY"] = "energy"

        self._tmpdir = tempfile.TemporaryDirectory()
        self.portfolio = Portfolio(path=Path(self._tmpdir.name) / "positions.json")
        self._orig_place_order = runner._place_order
        self._orig_get_price = runner.get_price
        self._orig_trade_sell = runner.alert.trade_sell
        runner._place_order = lambda code, side, qty, price, trd_env, env_label, confirmed: {
            "order_id": "FAKE", "dealt_qty": float(qty), "dealt_avg_price": price,
            "status": "FILLED_ALL"}
        runner.get_price = lambda code: 12.0
        runner.alert.trade_sell = lambda *a, **kw: None

    def tearDown(self):
        config.REPLACEMENT_MARGIN = self._orig_margin
        config.REPLACEMENT_MIN_NEW_SCORE = self._orig_min_score
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = self._orig_max_pool
        config.REPLACEMENT_BLOCK_SAME_SECTOR = self._orig_block_sector
        config.ENABLE_ACTIVE_REPLACEMENT = self._orig_enable
        config.SECTOR_MAP.clear()
        config.SECTOR_MAP.update(self._orig_sector_map)
        runner._place_order = self._orig_place_order
        runner.get_price = self._orig_get_price
        runner.alert.trade_sell = self._orig_trade_sell
        self._tmpdir.cleanup()
        test_support.unmute_alert_file_logging(self._alert_handlers)

    def _seed(self, code, score_label, total_score, held_days_ago=5):
        entry_time = (datetime.now() - timedelta(days=held_days_ago)).isoformat()
        self.portfolio.data["positions"][code] = {
            "side": "BUY", "entry_price": 10.0, "avg_cost": 10.0, "qty": 10,
            "signal_strength": 0.3, "strategy": "atr_breakout", "entry_atr": 0.1,
            "atr_mult": config.ATR_MULT_BASE, "breakeven_locked": False,
            "trail_stop": None, "entry_time": entry_time,
            "score_label": score_label, "total_score": total_score,
        }
        self.portfolio._save()

    def _direct_evaluate_decision(self, incoming_code, incoming_score):
        """跟_attempt_active_replacement()内部完全同样的held_positions
        构造方式（排除core_etf、补entry_day_idx），直接调用
        evaluate_replacement()——代表"backtest一侧"会得到的决策。"""
        held = {c: p for c, p in self.portfolio.data["positions"].items()
                if p.get("strategy") != "core_etf"}
        today_ord = datetime.now().toordinal()
        held_for_review = {
            c: {**p, "entry_day_idx": runner._day_ordinal(p.get("entry_time"))}
            for c, p in held.items()
        }
        evaluation = capacity_manager.evaluate_replacement(
            incoming_code=incoming_code, incoming_score=incoming_score,
            held_positions=held_for_review, current_day_idx=today_ord)
        return evaluation.victim_code if evaluation.decision == "REPLACE" else None

    def _runner_decision(self, incoming_code, incoming_score):
        return runner._attempt_active_replacement(
            self.portfolio, incoming_code=incoming_code, incoming_score=incoming_score,
            trd_env=None, env_label="SIMULATE", confirmed=False, results={})

    def test_agree_on_successful_replacement(self):
        self._seed("US.OBS_ENERGY", LABEL_OBSERVATION, 40.0)
        expected = self._direct_evaluate_decision("US.NEW", 90.0)
        actual = self._runner_decision("US.NEW", 90.0)
        self.assertEqual(expected, "US.OBS_ENERGY")
        self.assertEqual(actual, expected)

    def test_agree_on_no_candidate(self):
        self._seed("US.FULL_HELD", LABEL_FULL, 95.0)
        expected = self._direct_evaluate_decision("US.NEW", 90.0)
        actual = self._runner_decision("US.NEW", 90.0)
        self.assertIsNone(expected)
        self.assertEqual(actual, expected)

    def test_agree_on_margin_insufficient(self):
        self._seed("US.OBS_ENERGY", LABEL_OBSERVATION, 85.0)   # 90-85=5 < margin(10)
        expected = self._direct_evaluate_decision("US.NEW", 90.0)
        actual = self._runner_decision("US.NEW", 90.0)
        self.assertIsNone(expected)
        self.assertEqual(actual, expected)

    def test_agree_on_min_new_score_gate(self):
        config.REPLACEMENT_MIN_NEW_SCORE = 95.0
        self._seed("US.OBS_ENERGY", LABEL_OBSERVATION, 40.0)
        expected = self._direct_evaluate_decision("US.NEW", 90.0)   # 90 < 95
        actual = self._runner_decision("US.NEW", 90.0)
        self.assertIsNone(expected)
        self.assertEqual(actual, expected)

    def test_agree_on_block_same_sector_gate(self):
        config.REPLACEMENT_BLOCK_SAME_SECTOR = True
        self._seed("US.OBS_TECH", LABEL_OBSERVATION, 40.0)   # same sector as US.NEW
        expected = self._direct_evaluate_decision("US.NEW", 90.0)
        actual = self._runner_decision("US.NEW", 90.0)
        self.assertIsNone(expected)
        self.assertEqual(actual, expected)

    def test_agree_on_block_same_sector_falls_through_to_other_sector(self):
        config.REPLACEMENT_BLOCK_SAME_SECTOR = True
        self._seed("US.OBS_TECH", LABEL_OBSERVATION, 20.0)     # same sector, lower score
        self._seed("US.OBS_ENERGY", LABEL_OBSERVATION, 40.0)   # different sector
        expected = self._direct_evaluate_decision("US.NEW", 90.0)
        actual = self._runner_decision("US.NEW", 90.0)
        self.assertEqual(expected, "US.OBS_ENERGY")
        self.assertEqual(actual, expected)

    def test_agree_on_max_pool_size_gate(self):
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = 1
        self._seed("US.OBS_TECH", LABEL_OBSERVATION, 20.0)
        self._seed("US.OBS_ENERGY", LABEL_OBSERVATION, 40.0)   # pool size 2 > 1
        expected = self._direct_evaluate_decision("US.NEW", 90.0)
        actual = self._runner_decision("US.NEW", 90.0)
        self.assertIsNone(expected)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
