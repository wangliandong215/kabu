"""
Unit tests for engine/runner.py's v2.13 QQQ Core Recovery / Reclaim
integration (Phase 2.1, Observation Mode) — the 2c NEW_ENTRY resource
priority hook that lets QQQ's existing 2d top-up (untouched) reclaim its
target Core weight when normal-strategy buys have been starving it for
several passes. See risk/qqq_core_recovery.py module docstring.

Same hermetic monkeypatch harness as engine/test_portfolio_position_manager_gate.py
— every external dependency run_once() calls is patched, including
portfolio.broker_state.fetch_broker_state, plus the recovery module's own
state I/O (load_state/save_state/log_recovery_pass/log_level3_plan) so these
tests never touch C:\\KabuData.

Run:  python -m unittest engine.test_qqq_core_recovery_gate -v
"""
import tempfile
import unittest
from pathlib import Path

import config
import data.fetcher as fetcher_mod
import engine.runner as runner
import position_manager
import test_support
from portfolio.broker_state import BrokerState, BrokerPosition
from portfolio.tracker import Portfolio
from risk import qqq_core_recovery


class _FakeTracker:
    def __init__(self, *a, **kw): pass
    def update_position_metrics(self, *a, **kw): pass
    def log_exit(self, *a, **kw): pass
    def log_entry(self, *a, **kw): pass
    def log_market_context(self, *a, **kw): pass


class QQQCoreRecoveryGateTestCase(unittest.TestCase):

    def setUp(self):
        self._alert_handlers = test_support.mute_alert_file_logging()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.portfolio_path = Path(self._tmpdir.name) / "positions.json"
        self.placed_orders = []
        self.load_state_calls = []
        self.save_state_calls = []
        self.log_recovery_calls = []
        self.log_level3_calls = []

        self._orig = {
            "Portfolio": runner.Portfolio,
            "_place_order": runner._place_order,
            "filter_open": runner.filter_open,
            "scan": runner.scan,
            "smart_scan": runner.smart_scan,
            "build_candidate_pool": runner.pipeline.build_candidate_pool,
            "check_max_drawdown": runner.guard.check_max_drawdown,
            "macro_circuit_breaker": runner.news_sentiment.macro_circuit_breaker,
            "qqq_macro_halt": runner.regime.qqq_macro_halt,
            "market_weather": runner.market_weather.market_weather,
            "trade_sell": runner.alert.trade_sell,
            "trade_buy": runner.alert.trade_buy,
            "TradeTracker": runner.TradeTracker,
            "fetch_kline": fetcher_mod.fetch_kline,
            "fetch_broker_state": runner.broker_state_mod.fetch_broker_state,
            "log_snapshot": runner.portfolio_risk_manager.log_snapshot,
            "log_diff": runner.portfolio_risk_manager.log_diff,
            "log_qqq_snapshot": runner.portfolio_risk_manager.log_qqq_snapshot,
            "pm_log_attempt": runner.portfolio_position_manager.log_attempt,
            "fundamental_score": runner.pipeline.fundamental.score,
            "classify_code": runner.pipeline.news_filter.classify_code,
            "is_earnings_blackout": runner.pipeline.is_earnings_blackout,
            "_qqq_above_ma": runner._qqq_above_ma,
            "PM_ENABLED": config.PORTFOLIO_POSITION_MANAGER_ENABLED,
            "RECOVERY_ENABLED": config.QQQ_CORE_RECOVERY_ENABLED,
            "RECOVERY_MAX_RESERVE_PCT": config.QQQ_CORE_RECOVERY_MAX_RESERVE_PCT,
            "RECOVERY_TRIGGER_PASSES": config.QQQ_CORE_RECOVERY_TRIGGER_PASSES,
            "has_earnings_risk": runner.event_risk.has_earnings_risk,
            "build_trade_research_snapshot": runner.research_snapshot.build_trade_research_snapshot,
            "get_price": runner.get_price,
            "qcr_load_state": qqq_core_recovery.load_state,
            "qcr_save_state": qqq_core_recovery.save_state,
            "qcr_log_recovery_pass": qqq_core_recovery.log_recovery_pass,
            "qcr_log_level3_plan": qqq_core_recovery.log_level3_plan,
            "regime_evaluate_and_log": runner.regime_evaluate_and_log,
            "POSITION_MANAGER_V31_MODE": config.POSITION_MANAGER_V31_MODE,
        }
        # Regime Observation Layer (2026-09-23) — pure/no-op stub so tests
        # never write to the real C:\KabuData\portfolio\regime_log.jsonl.
        runner.regime_evaluate_and_log = lambda *a, **kw: None
        # V3.1-A Position Manager (2026-09-23) — this file isn't testing it;
        # OFF makes engine/runner.py's new 2b.5 block a complete no-op so
        # US.WEAK (opened via portfolio.open_position() below) doesn't pick
        # up an unrelated side effect (real confidence_score/regime_store
        # I/O, real position_manager_state.json/position_manager_log.jsonl
        # writes).
        config.POSITION_MANAGER_V31_MODE = position_manager.MODE_OFF
        # _qqq_above_ma() hits live quotes -- stub True so the QQQ Beta floor's
        # own MA200 exit/top-up logic never confounds these tests.
        runner._qqq_above_ma = lambda: True
        runner.get_price = lambda code: 700.0
        runner.event_risk.has_earnings_risk = lambda code, trade_date=None: False
        runner.research_snapshot.build_trade_research_snapshot = lambda *a, **kw: {}

        runner.Portfolio = lambda: Portfolio(path=self.portfolio_path)
        runner.filter_open = lambda codes: list(codes)
        runner.regime.qqq_macro_halt = lambda df: False
        runner.market_weather.market_weather = lambda df: 1
        runner.alert.trade_sell = lambda *a, **kw: None
        runner.alert.trade_buy = lambda *a, **kw: None
        runner.guard.check_max_drawdown = lambda portfolio: True
        runner.news_sentiment.macro_circuit_breaker = lambda: ""
        runner.TradeTracker = _FakeTracker
        fetcher_mod.fetch_kline = lambda *a, **kw: None
        runner.pipeline.fundamental.score = lambda code, env: {
            "score": None, "tier": None, "reason": "test_stub"}
        runner.pipeline.news_filter.classify_code = lambda code: {
            "score": None, "tier": 3, "matched_keyword": None}
        runner.pipeline.is_earnings_blackout = lambda code: False
        runner.portfolio_risk_manager.log_snapshot = lambda *a, **kw: None
        runner.portfolio_risk_manager.log_diff = lambda *a, **kw: None
        runner.portfolio_risk_manager.log_qqq_snapshot = lambda *a, **kw: None
        runner.portfolio_position_manager.log_attempt = lambda *a, **kw: None

        # Default: no active Recovery episode -- individual tests override.
        def _default_load_state():
            self.load_state_calls.append(True)
            return qqq_core_recovery.RecoveryState()
        qqq_core_recovery.load_state = _default_load_state
        qqq_core_recovery.save_state = lambda state: self.save_state_calls.append(state)
        qqq_core_recovery.log_recovery_pass = lambda *a, **kw: self.log_recovery_calls.append((a, kw))
        qqq_core_recovery.log_level3_plan = lambda *a, **kw: self.log_level3_calls.append((a, kw))

        def _mock_place_order(code, side, qty, price, trd_env, env_label, confirmed, **_kwargs):
            self.placed_orders.append({"code": code, "side": side, "qty": qty, "price": price})
            dealt = float(qty) if confirmed else 0.0
            return {"order_id": "FAKE123" if confirmed else "", "dealt_qty": dealt,
                    "dealt_avg_price": price if confirmed else 0.0,
                    "status": "FILLED_ALL" if confirmed else "DRY_RUN"}
        runner._place_order = _mock_place_order

    def tearDown(self):
        runner.Portfolio = self._orig["Portfolio"]
        runner._place_order = self._orig["_place_order"]
        runner.filter_open = self._orig["filter_open"]
        runner.scan = self._orig["scan"]
        runner.smart_scan = self._orig["smart_scan"]
        runner.pipeline.build_candidate_pool = self._orig["build_candidate_pool"]
        runner.guard.check_max_drawdown = self._orig["check_max_drawdown"]
        runner.news_sentiment.macro_circuit_breaker = self._orig["macro_circuit_breaker"]
        runner.regime.qqq_macro_halt = self._orig["qqq_macro_halt"]
        runner.market_weather.market_weather = self._orig["market_weather"]
        runner.alert.trade_sell = self._orig["trade_sell"]
        runner.alert.trade_buy = self._orig["trade_buy"]
        runner.TradeTracker = self._orig["TradeTracker"]
        fetcher_mod.fetch_kline = self._orig["fetch_kline"]
        runner.broker_state_mod.fetch_broker_state = self._orig["fetch_broker_state"]
        runner.portfolio_risk_manager.log_snapshot = self._orig["log_snapshot"]
        runner.portfolio_risk_manager.log_diff = self._orig["log_diff"]
        runner.portfolio_risk_manager.log_qqq_snapshot = self._orig["log_qqq_snapshot"]
        runner.portfolio_position_manager.log_attempt = self._orig["pm_log_attempt"]
        runner.pipeline.fundamental.score = self._orig["fundamental_score"]
        runner.pipeline.news_filter.classify_code = self._orig["classify_code"]
        runner.pipeline.is_earnings_blackout = self._orig["is_earnings_blackout"]
        runner._qqq_above_ma = self._orig["_qqq_above_ma"]
        runner.event_risk.has_earnings_risk = self._orig["has_earnings_risk"]
        runner.research_snapshot.build_trade_research_snapshot = self._orig["build_trade_research_snapshot"]
        config.PORTFOLIO_POSITION_MANAGER_ENABLED = self._orig["PM_ENABLED"]
        config.QQQ_CORE_RECOVERY_ENABLED = self._orig["RECOVERY_ENABLED"]
        config.QQQ_CORE_RECOVERY_MAX_RESERVE_PCT = self._orig["RECOVERY_MAX_RESERVE_PCT"]
        config.QQQ_CORE_RECOVERY_TRIGGER_PASSES = self._orig["RECOVERY_TRIGGER_PASSES"]
        runner.get_price = self._orig["get_price"]
        qqq_core_recovery.load_state = self._orig["qcr_load_state"]
        qqq_core_recovery.save_state = self._orig["qcr_save_state"]
        qqq_core_recovery.log_recovery_pass = self._orig["qcr_log_recovery_pass"]
        qqq_core_recovery.log_level3_plan = self._orig["qcr_log_level3_plan"]
        runner.regime_evaluate_and_log = self._orig["regime_evaluate_and_log"]
        config.POSITION_MANAGER_V31_MODE = self._orig["POSITION_MANAGER_V31_MODE"]
        self._tmpdir.cleanup()
        test_support.unmute_alert_file_logging(self._alert_handlers)

    def _seed_fresh_buy_candidate(self, code="US.TEST", price=100.0):
        runner.scan = lambda codes, **kw: {
            code: {"current_price": price, "signal": "BUY",
                   "signal_strength": 0.9, "strategy_used": "atr_breakout"},
        }
        runner.smart_scan = runner.scan

    def _broker_state(self, cash, total_assets, long_mv, qqq_market_val=0.0):
        positions = {}
        if qqq_market_val > 0:
            positions[config.QQQ_CORE_CODE] = BrokerPosition(
                code=config.QQQ_CORE_CODE, qty=1.0, cost_price=qqq_market_val,
                market_val=qqq_market_val, current_price=qqq_market_val)
        return BrokerState(positions=positions, cash=cash,
                            total_assets=total_assets, long_mv=long_mv)

    def _reserve_active_state(self):
        return qqq_core_recovery.RecoveryState(
            active=True, last_shortfall=250_000.0, consecutive_no_progress_passes=5,
            reserve_active=True, level3_eligible=False,
            episode_started_at="2026-01-01T00:00:00")

    # ── A. Observation Mode must never change real order sizes ─────────────

    def test_observation_mode_does_not_change_buy_qty(self):
        """QQQ_CORE_RECOVERY_ENABLED=False (shipped default) must leave the
        2c BUY qty untouched even though a mocked Recovery state has
        reserve_active=True AND level3_eligible=True -- proving Observation
        Mode really only computes/logs, never clamps."""
        config.QQQ_CORE_RECOVERY_ENABLED = False
        qqq_core_recovery.load_state = lambda: qqq_core_recovery.RecoveryState(
            active=True, last_shortfall=250_000.0, consecutive_no_progress_passes=99,
            reserve_active=True, level3_eligible=True, episode_started_at="2026-01-01T00:00:00")
        # Cash tight enough that a live reserve WOULD block the buy if it applied
        # (total exposure kept low/NORMAL so macro_block itself isn't what's
        # blocking the buy -- this test is isolating the Recovery gate only).
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: self._broker_state(
            cash=150.0, total_assets=1_000_000.0, long_mv=100_000.0)
        Portfolio(path=self.portfolio_path)._save()
        self._seed_fresh_buy_candidate(price=100.0)

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        buys = [o for o in self.placed_orders if o["side"] == "BUY" and o["code"] == "US.TEST"]
        self.assertEqual(len(buys), 1,
                          "Observation Mode must not block the BUY regardless of the mocked "
                          "reserve/level3 state")
        self.assertEqual(buys[0]["qty"], 1, "$150 cash @ $100/share affords exactly 1 share "
                                             "-- unaffected by any reserve math")

    # ── B. ENABLED=True must actually reduce 2c's available cash ───────────

    def test_enabled_mode_reduces_2c_qty_by_reserve(self):
        config.QQQ_CORE_RECOVERY_ENABLED = True
        config.QQQ_CORE_RECOVERY_MAX_RESERVE_PCT = 0.05   # 5% of 1,000,000 = $50,000
        qqq_core_recovery.load_state = self._reserve_active_state
        # cash=$50,300 -> without reserve: 503 shares @ $100. shortfall (250,000)
        # and total_assets*5% (50,000) both exceed cash, so cash itself is the
        # binding reserve constraint: reserve = min(shortfall, cap, cash) = cash.
        # That would reserve the ENTIRE cash pool -- use a bigger cash pile so
        # the 5% cap (not the cash itself) is the binding constraint instead.
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: self._broker_state(
            cash=500_000.0, total_assets=1_000_000.0, long_mv=500_000.0)
        Portfolio(path=self.portfolio_path)._save()
        self._seed_fresh_buy_candidate(price=100.0)

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        buys = [o for o in self.placed_orders if o["side"] == "BUY" and o["code"] == "US.TEST"]
        self.assertEqual(len(buys), 1)
        # Available for 2c = 500,000 - 50,000 (5% reserve cap) = 450,000 -> <=4,500 shares.
        self.assertLessEqual(buys[0]["qty"], 4_500,
                              "ENABLED=True must clamp 2c's affordable qty to "
                              "(cash - reserve), not the full cash pool")

    # ── C. Reserve larger than the pool must floor at zero, never negative ──

    def test_reserve_larger_than_cash_blocks_without_going_negative(self):
        config.QQQ_CORE_RECOVERY_ENABLED = True
        qqq_core_recovery.load_state = self._reserve_active_state
        # cash=$80 -> reserve = min(shortfall 250,000; 5%*1,000,000=50,000; pool 80) = 80
        # (total exposure kept low/NORMAL so macro_block itself isn't what's
        # blocking the buy -- this test is isolating the Recovery gate only).
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: self._broker_state(
            cash=80.0, total_assets=1_000_000.0, long_mv=50_000.0)
        Portfolio(path=self.portfolio_path)._save()
        self._seed_fresh_buy_candidate(price=100.0)

        # Must not raise (no negative max_affordable / int() blow-up).
        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        buys = [o for o in self.placed_orders if o["side"] == "BUY" and o["code"] == "US.TEST"]
        self.assertEqual(len(buys), 0, "cash == reserve leaves nothing for 2c -> blocked, not negative")

    # ── D. macro_block=True must skip Recovery state entirely ──────────────

    def test_macro_block_skips_recovery_state_advance(self):
        config.QQQ_CORE_RECOVERY_ENABLED = True
        runner.news_sentiment.macro_circuit_breaker = lambda: "TEST_HALT"
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: self._broker_state(
            cash=500_000.0, total_assets=1_000_000.0, long_mv=500_000.0)
        Portfolio(path=self.portfolio_path)._save()
        self._seed_fresh_buy_candidate(price=100.0)

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        self.assertEqual(self.load_state_calls, [],
                          "macro_block=True must skip Recovery state load/advance/save entirely")
        self.assertEqual(self.save_state_calls, [])
        self.assertEqual(self.log_recovery_calls, [])

    # ── E. Level 3 must only plan/log, never place an order ─────────────────

    def test_level3_eligible_never_places_an_order(self):
        config.QQQ_CORE_RECOVERY_ENABLED = True
        qqq_core_recovery.load_state = lambda: qqq_core_recovery.RecoveryState(
            active=True, last_shortfall=250_000.0, consecutive_no_progress_passes=20,
            reserve_active=True, level3_eligible=True, episode_started_at="2026-01-01T00:00:00")
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: self._broker_state(
            cash=500_000.0, total_assets=1_000_000.0, long_mv=500_000.0)
        portfolio = Portfolio(path=self.portfolio_path)
        # An existing satellite position that Level3's plan would target --
        # proves the plan is computed but never sold.
        portfolio.open_position("US.WEAK", "BUY", 50.0, 100, 0.9, "atr_breakout")
        portfolio._save()
        self._seed_fresh_buy_candidate(price=100.0)

        runner.run_once(codes=["US.TEST", "US.WEAK"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        sells = [o for o in self.placed_orders if o["side"] == "SELL"]
        self.assertEqual(sells, [], "Level 3 must never place a SELL order in this phase")
        self.assertEqual(len(self.log_level3_calls), 1, "Level 3 plan must be logged exactly once")

    # ── F. Exception safety: a crash inside 2c must not leak/corrupt state ──

    def test_exception_in_2c_does_not_leak_reserve_to_next_pass(self):
        config.QQQ_CORE_RECOVERY_ENABLED = True
        config.QQQ_CORE_RECOVERY_MAX_RESERVE_PCT = 0.05
        qqq_core_recovery.load_state = self._reserve_active_state
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: self._broker_state(
            cash=500_000.0, total_assets=1_000_000.0, long_mv=500_000.0)
        Portfolio(path=self.portfolio_path)._save()
        self._seed_fresh_buy_candidate(price=100.0)

        orig_build_pool = runner.pipeline.build_candidate_pool
        call_count = {"n": 0}

        def _flaky_build_pool(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated 2c failure")
            return orig_build_pool(*a, **kw)
        runner.pipeline.build_candidate_pool = _flaky_build_pool

        with self.assertRaises(RuntimeError):
            runner.run_once(codes=["US.TEST"], confirmed=True,
                             auto_route=False, strategy_name="atr_breakout")
        self.assertEqual(self.placed_orders, [], "the crashed pass must not have placed anything")

        # Retry with the real build_candidate_pool restored -- must behave
        # exactly as an uncrashed pass would (no leaked/stuck reservation).
        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")
        buys = [o for o in self.placed_orders if o["side"] == "BUY" and o["code"] == "US.TEST"]
        self.assertEqual(len(buys), 1)
        self.assertLessEqual(buys[0]["qty"], 4_500,
                              "retry after a crashed pass must still respect the same "
                              "(cash - reserve) clamp -- nothing was corrupted by the crash")


if __name__ == "__main__":
    unittest.main()
