# -*- coding: utf-8 -*-
"""
Integration tests for V3.1-A Position Manager wiring in engine/runner.py's
new "2b.5" block inside run_once(). Verifies the three
config.POSITION_MANAGER_V31_MODE states end-to-end:

  OFF          — the block is a complete no-op: byte-for-byte pre-V3.1
                 behavior (no context built, no log written, no order).
  OBSERVATION  — computes + logs a REDUCE decision but NEVER places an
                 order or changes portfolio qty (Decision vs Execution
                 separation — see position_manager/__init__.py::log_decision).
  ACTIVE       — a REDUCE decision places a real SELL via the existing
                 _place_order()/portfolio.reduce_position() path, exactly
                 like v2.10's rebalance trims.

Same monkeypatch harness as engine/test_runner_pyramid_gate.py /
test_runner_drawdown_halt.py. The seeded US.TEST position is priced 4.5%
below entry — enough to clear position_manager's own DRAWDOWN_TIER1_PCT
(3%) but comfortably short of config.STOP_LOSS_PCT (5%), so the pre-existing
hard stop-loss in the exit-check loop never fires and closes the position
out from under this test before the new 2b.5 block even runs.

Run:  python -m unittest engine.test_runner_position_manager_gate -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import config
import data.fetcher as fetcher_mod
import engine.runner as runner
import position_manager
import test_support
from engine.confidence_score import ConfidenceScoreResult
from portfolio.broker_state import BrokerState
from portfolio.tracker import Portfolio

_NO_RISK_BROKER_STATE = BrokerState(positions={}, cash=1_000_000.0,
                                     total_assets=1_000_000.0, long_mv=0.0)


class _FakeTracker:
    def __init__(self, *a, **kw): pass
    def update_position_metrics(self, *a, **kw): pass
    def log_exit(self, *a, **kw): pass
    def log_entry(self, *a, **kw): pass


def _stub_gather_and_score(code, rule_based_score, sig, tracker=None,
                            execution="REAL", as_of=None):
    """Fixed confidence, no HMM opinion — isolates these tests to the
    Drawdown Reduction module only (see module docstring's price choice),
    so REDUCE/HOLD here is never accidentally driven by confidence/HMM."""
    result = ConfidenceScoreResult(
        confidence_score=80.0, rule_component=80.0, hmm_component=None,
        win_rate_component=None, expectancy_component=None, market_component=None,
        volatility_component=None, volume_component=None, weights_used={"rule": 0.30},
    )
    detail = {"formula_version": "test", "confidence_score": 80.0, "hmm_state": None}
    return result, detail


class PositionManagerGateTestCase(unittest.TestCase):

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
            "POSITION_MANAGER_V31_MODE": config.POSITION_MANAGER_V31_MODE,
            "PM_STORE_PATH": position_manager.state_store._STORE_PATH,
            "PM_LOG_PATH": position_manager.POSITION_MANAGER_LOG_PATH,
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
        fetcher_mod.fetch_kline = lambda *a, **kw: None
        runner.confidence_score.gather_and_score = _stub_gather_and_score

        # No BUY signal this pass — isolates the new 2b.5 existing-position
        # block from the pre-existing 2c new-entry path entirely. -4.5% vs
        # entry (see module docstring) so only Drawdown Reduction triggers.
        runner.scan = lambda codes, **kw: {
            "US.TEST": {"current_price": 95.5, "signal": "HOLD",
                        "signal_strength": 0.3, "strategy_used": "atr_breakout",
                        "atr": 1.0},
            "US.QQQ": {"current_price": 100.0, "signal": "HOLD",
                       "signal_strength": 0.3, "strategy_used": "core_etf",
                       "atr": 1.0},
        }
        runner.smart_scan = runner.scan

        position_manager.state_store._STORE_PATH = Path(self._tmpdir.name) / "pm_state.json"
        position_manager.POSITION_MANAGER_LOG_PATH = Path(self._tmpdir.name) / "pm_log.jsonl"

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
        config.POSITION_MANAGER_V31_MODE = self._orig["POSITION_MANAGER_V31_MODE"]
        position_manager.state_store._STORE_PATH = self._orig["PM_STORE_PATH"]
        position_manager.POSITION_MANAGER_LOG_PATH = self._orig["PM_LOG_PATH"]
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
        path = position_manager.POSITION_MANAGER_LOG_PATH
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _run(self, code="US.TEST"):
        runner.run_once(codes=[code], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")


class TestOffModeIsNoOp(PositionManagerGateTestCase):
    def test_off_mode_no_log_no_order_no_qty_change(self):
        config.POSITION_MANAGER_V31_MODE = "OFF"
        self._seed_position()
        self._run()

        self.assertEqual(self._read_log(), [])
        self.assertFalse(position_manager.state_store._STORE_PATH.exists())
        self.assertEqual([o for o in self.placed_orders if o["side"] == "SELL"], [])
        reloaded = Portfolio(path=self.portfolio_path)
        self.assertEqual(reloaded.data["positions"]["US.TEST"]["qty"], 1000)


class TestObservationModeLogsWithoutExecuting(PositionManagerGateTestCase):
    def test_observation_mode_reduce_decision_never_places_order(self):
        config.POSITION_MANAGER_V31_MODE = "OBSERVATION"
        self._seed_position()
        self._run()

        rows = self._read_log()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "US.TEST")
        self.assertEqual(rows[0]["final_decision"], "REDUCE")
        self.assertEqual(rows[0]["execution"], "SKIPPED_OBSERVATION_MODE")
        self.assertGreater(rows[0]["reduction_amount"], 0)

        self.assertEqual([o for o in self.placed_orders if o["side"] == "SELL"], [],
                          "OBSERVATION mode must never place a SELL order")
        reloaded = Portfolio(path=self.portfolio_path)
        self.assertEqual(reloaded.data["positions"]["US.TEST"]["qty"], 1000,
                          "OBSERVATION mode must never change actual qty")


class TestActiveModeExecutesReduce(PositionManagerGateTestCase):
    def test_active_mode_reduce_decision_places_sell_and_updates_qty(self):
        config.POSITION_MANAGER_V31_MODE = "ACTIVE"
        self._seed_position()
        self._run()

        rows = self._read_log()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["final_decision"], "REDUCE")
        self.assertEqual(rows[0]["execution"], "EXECUTED")

        sells = [o for o in self.placed_orders if o["side"] == "SELL"]
        self.assertEqual(len(sells), 1)
        self.assertGreater(sells[0]["qty"], 0)

        reloaded = Portfolio(path=self.portfolio_path)
        self.assertLess(reloaded.data["positions"]["US.TEST"]["qty"], 1000)
        self.assertEqual(reloaded.data["positions"]["US.TEST"]["qty"],
                          1000 - sells[0]["qty"])


class TestCoreEtfNeverManaged(PositionManagerGateTestCase):
    def test_core_etf_position_skipped_even_in_active_mode(self):
        # Keep the pre-existing MA200 exit (engine/runner.py's exit-check
        # loop, untouched by V3.1) from closing US.QQQ out first — that
        # would make the log empty for the wrong reason. Forcing
        # _qqq_above_ma()=True lets the position survive into the new
        # 2b.5 block, so this test actually exercises Position Manager's
        # OWN core_etf skip guard, not the unrelated MA200 exit.
        self._orig["_qqq_above_ma"] = runner._qqq_above_ma
        runner._qqq_above_ma = lambda: True
        self.addCleanup(lambda: setattr(runner, "_qqq_above_ma", self._orig["_qqq_above_ma"]))

        config.POSITION_MANAGER_V31_MODE = "ACTIVE"
        self._seed_position(code="US.QQQ", entry_price=100.0, qty=1000,
                             strategy="core_etf")
        self._run(code="US.QQQ")

        self.assertEqual(self._read_log(), [],
                          "core_etf (QQQ Beta floor) must never be touched by Position Manager")
        # Note: US.QQQ's qty may still change this pass via the pre-existing,
        # unrelated "2d QQQ core top-up to 25% target" logic (untouched by
        # V3.1) — that's expected and not what this test is checking; the
        # log assertion above is what proves Position Manager itself never
        # looked at this position.


if __name__ == "__main__":
    unittest.main()
