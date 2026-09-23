# -*- coding: utf-8 -*-
"""
Integration tests for V3.2-A Minute Exit Engine wiring in
engine/runner.py's new "2b.6" block inside run_once(). Mirrors
engine/test_runner_position_manager_gate.py's harness.

  ENABLE_V3_2=False — the block is a complete no-op: no fetch, no
                       exit_engine log written, no order placed, portfolio
                       untouched.
  ENABLE_V3_2=True  — computes + logs a diagnostic row for every open
                       non-core_etf position, but NEVER places an order or
                       changes portfolio qty (this package cannot execute
                       trades even in principle — see exit_engine/
                       __init__.py's module docstring).

Run:  python -m unittest engine.test_runner_exit_engine_gate -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import config
import data.fetcher as fetcher_mod
import engine.runner as runner
import exit_engine
import test_support
from engine.confidence_score import ConfidenceScoreResult
from exit_engine import state_store
from exit_engine._fixtures import make_bars
from portfolio.broker_state import BrokerState
from portfolio.tracker import Portfolio

_NO_RISK_BROKER_STATE = BrokerState(positions={}, cash=1_000_000.0,
                                     total_assets=1_000_000.0, long_mv=0.0)

_DOWN_BARS = make_bars([130.0 - i for i in range(30)])


class _FakeTracker:
    def __init__(self, *a, **kw): pass
    def update_position_metrics(self, *a, **kw): pass
    def log_exit(self, *a, **kw): pass
    def log_entry(self, *a, **kw): pass


def _stub_gather_and_score(code, rule_based_score, sig, tracker=None,
                            execution="REAL", as_of=None):
    result = ConfidenceScoreResult(
        confidence_score=80.0, rule_component=80.0, hmm_component=None,
        win_rate_component=None, expectancy_component=None, market_component=None,
        volatility_component=None, volume_component=None, weights_used={"rule": 0.30},
    )
    detail = {"formula_version": "test", "confidence_score": 80.0, "hmm_state": None}
    return result, detail


class ExitEngineGateTestCase(unittest.TestCase):

    def setUp(self):
        self._alert_handlers = test_support.mute_alert_file_logging()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.portfolio_path = Path(self._tmpdir.name) / "positions.json"
        self.placed_orders = []

        self._orig = {
            "Portfolio": runner.Portfolio,
            "_place_order": runner._place_order,
            "filter_open": runner.filter_open,
            "scan": runner.scan,
            "smart_scan": runner.smart_scan,
            "check_max_drawdown": runner.guard.check_max_drawdown,
            "macro_circuit_breaker": runner.news_sentiment.macro_circuit_breaker,
            "qqq_macro_halt": runner.regime.qqq_macro_halt,
            "market_weather": runner.market_weather.market_weather,
            "trade_sell": runner.alert.trade_sell,
            "trade_buy": runner.alert.trade_buy,
            "TradeTracker": runner.TradeTracker,
            "fetch_kline": fetcher_mod.fetch_kline,
            "fetch_broker_state": runner.broker_state_mod.fetch_broker_state,
            "has_earnings_risk": runner.event_risk.has_earnings_risk,
            "build_trade_research_snapshot": runner.research_snapshot.build_trade_research_snapshot,
            "regime_evaluate_and_log": runner.regime_evaluate_and_log,
            "gather_and_score": runner.confidence_score.gather_and_score,
            "ENABLE_V3_2": config.ENABLE_V3_2,
            "ENABLE_ATR_EXIT": config.ENABLE_ATR_EXIT,
            "POSITION_MANAGER_V31_MODE": config.POSITION_MANAGER_V31_MODE,
            "EE_STORE_PATH": state_store._STORE_PATH,
            "EE_LOG_PATH": exit_engine.EXIT_ENGINE_LOG_PATH,
        }

        runner.regime_evaluate_and_log = lambda *a, **kw: None
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: _NO_RISK_BROKER_STATE
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
        # Minute/5m/15m bars all resolve to the same synthetic downtrend
        # regardless of ktype/bars — this test only cares whether the
        # block runs and logs, not the exact signal values (those are
        # exit_engine/'s own unit tests' job).
        fetcher_mod.fetch_kline = lambda code, ktype=None, bars=None: _DOWN_BARS
        runner.confidence_score.gather_and_score = _stub_gather_and_score

        runner.scan = lambda codes, **kw: {
            "US.TEST": {"current_price": 95.5, "signal": "HOLD",
                        "signal_strength": 0.3, "strategy_used": "atr_breakout",
                        "atr": 1.0},
            "US.QQQ": {"current_price": 100.0, "signal": "HOLD",
                       "signal_strength": 0.3, "strategy_used": "core_etf",
                       "atr": 1.0},
        }
        runner.smart_scan = runner.scan

        state_store._STORE_PATH = Path(self._tmpdir.name) / "ee_state.json"
        exit_engine.EXIT_ENGINE_LOG_PATH = Path(self._tmpdir.name) / "ee_log.jsonl"
        config.POSITION_MANAGER_V31_MODE = "OFF"   # isolate this test to 2b.6 only

        def _mock_place_order(code, side, qty, price, trd_env, env_label, confirmed, **_kwargs):
            self.placed_orders.append({"code": code, "side": side, "qty": qty})
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
        runner.guard.check_max_drawdown = self._orig["check_max_drawdown"]
        runner.news_sentiment.macro_circuit_breaker = self._orig["macro_circuit_breaker"]
        runner.regime.qqq_macro_halt = self._orig["qqq_macro_halt"]
        runner.market_weather.market_weather = self._orig["market_weather"]
        runner.alert.trade_sell = self._orig["trade_sell"]
        runner.alert.trade_buy = self._orig["trade_buy"]
        runner.TradeTracker = self._orig["TradeTracker"]
        fetcher_mod.fetch_kline = self._orig["fetch_kline"]
        runner.broker_state_mod.fetch_broker_state = self._orig["fetch_broker_state"]
        runner.event_risk.has_earnings_risk = self._orig["has_earnings_risk"]
        runner.research_snapshot.build_trade_research_snapshot = self._orig["build_trade_research_snapshot"]
        runner.regime_evaluate_and_log = self._orig["regime_evaluate_and_log"]
        runner.confidence_score.gather_and_score = self._orig["gather_and_score"]
        config.ENABLE_V3_2 = self._orig["ENABLE_V3_2"]
        config.ENABLE_ATR_EXIT = self._orig["ENABLE_ATR_EXIT"]
        config.POSITION_MANAGER_V31_MODE = self._orig["POSITION_MANAGER_V31_MODE"]
        state_store._STORE_PATH = self._orig["EE_STORE_PATH"]
        exit_engine.EXIT_ENGINE_LOG_PATH = self._orig["EE_LOG_PATH"]
        self._tmpdir.cleanup()
        test_support.unmute_alert_file_logging(self._alert_handlers)

    def _seed_position(self, code="US.TEST", entry_price=100.0, qty=1000,
                        entry_time="2020-01-01T00:00:00", strategy="atr_breakout"):
        portfolio = Portfolio(path=self.portfolio_path)
        portfolio.data["positions"][code] = {
            "side": "BUY", "entry_price": entry_price, "avg_cost": entry_price, "qty": qty,
            "signal_strength": 0.9, "strategy": strategy, "entry_atr": 1.0,
            "atr_mult": config.ATR_MULT_BASE, "breakeven_locked": False,
            "trail_stop": None, "entry_time": entry_time,
            "score_label": "FULL", "total_score": 90.0,
        }
        portfolio._save()

    def _read_log(self):
        path = exit_engine.EXIT_ENGINE_LOG_PATH
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _run(self, code="US.TEST"):
        runner.run_once(codes=[code], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")


class TestDisabledIsNoOp(ExitEngineGateTestCase):
    def test_disabled_no_log_no_order_no_qty_change(self):
        config.ENABLE_V3_2 = False
        self._seed_position()
        self._run()

        self.assertEqual(self._read_log(), [])
        self.assertFalse(state_store._STORE_PATH.exists())
        self.assertEqual([o for o in self.placed_orders if o["side"] == "SELL"], [])
        reloaded = Portfolio(path=self.portfolio_path)
        self.assertEqual(reloaded.data["positions"]["US.TEST"]["qty"], 1000)


class TestEnabledLogsWithoutExecuting(ExitEngineGateTestCase):
    def test_enabled_logs_diagnostics_never_places_order(self):
        config.ENABLE_V3_2 = True
        config.ENABLE_ATR_EXIT = True
        self._seed_position(entry_price=100.0, qty=1000)
        self._run()

        rows = self._read_log()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "US.TEST")
        self.assertIn("pressure_score", rows[0])
        self.assertIn("pressure_tier", rows[0])
        self.assertIn("signals", rows[0])

        self.assertEqual([o for o in self.placed_orders if o["side"] == "SELL"], [],
                          "exit_engine must never place a SELL order")
        reloaded = Portfolio(path=self.portfolio_path)
        self.assertEqual(reloaded.data["positions"]["US.TEST"]["qty"], 1000,
                          "exit_engine must never change actual qty")


class TestCoreEtfNeverEvaluated(ExitEngineGateTestCase):
    def test_core_etf_position_skipped(self):
        self._orig["_qqq_above_ma"] = runner._qqq_above_ma
        runner._qqq_above_ma = lambda: True
        self.addCleanup(lambda: setattr(runner, "_qqq_above_ma", self._orig["_qqq_above_ma"]))

        config.ENABLE_V3_2 = True
        config.ENABLE_ATR_EXIT = True
        self._seed_position(code="US.QQQ", entry_price=100.0, qty=1000, strategy="core_etf")
        self._run(code="US.QQQ")

        self.assertEqual(self._read_log(), [],
                          "core_etf (QQQ Beta floor) must never be evaluated by exit_engine")


class TestPerSymbolFailureIsolation(ExitEngineGateTestCase):
    def test_one_symbol_raising_does_not_stop_the_pass(self):
        config.ENABLE_V3_2 = True
        config.ENABLE_ATR_EXIT = True
        self._seed_position(code="US.TEST", entry_price=100.0, qty=1000)

        orig_build_context = exit_engine.build_context

        def _raising_build_context(*a, **kw):
            raise RuntimeError("boom")
        exit_engine.build_context = _raising_build_context
        self.addCleanup(lambda: setattr(exit_engine, "build_context", orig_build_context))

        # Must not raise out of run_once() despite the injected failure.
        self._run()
        self.assertEqual(self._read_log(), [])


if __name__ == "__main__":
    unittest.main()
