"""
Unit tests for engine/runner.py's v2.11 OpenD Preflight integration — the
check inside _place_order() (the single order-placement choke point) that
runs market_preflight.check(code) right before delegating to the broker
whenever confirmed=True.

Unlike the sibling test_runner_*.py files, this suite does NOT replace
runner._place_order() itself (that would skip the exact code path under
test) — instead it replaces runner.get_broker() with a fake broker so the
real _place_order()/preflight logic executes, while nothing actually
touches OpenD.

Run:  python -m unittest engine.test_runner_preflight_gate -v
"""
import tempfile
import unittest
from pathlib import Path

import config
import data.fetcher as fetcher_mod
import engine.runner as runner
from engine.market_preflight import PreflightResult
from portfolio.broker_state import BrokerState
from portfolio.tracker import Portfolio

_NO_RISK_BROKER_STATE = BrokerState(positions={}, cash=1_000_000.0,
                                     total_assets=1_000_000.0, long_mv=0.0)


class _FakeTracker:
    def __init__(self, *a, **kw): pass
    def update_position_metrics(self, *a, **kw): pass
    def log_exit(self, *a, **kw): pass
    def log_entry(self, *a, **kw): pass
    def log_market_context(self, *a, **kw): pass


class _FakeBroker:
    calls = []

    @staticmethod
    def place_order(code, side, qty, price, trd_env, env_label, confirmed):
        _FakeBroker.calls.append({"code": code, "side": side, "qty": qty})
        dealt = float(qty) if confirmed else 0.0
        return {"order_id": "FAKE123" if confirmed else "", "dealt_qty": dealt,
                "dealt_avg_price": price if confirmed else 0.0,
                "status": "FILLED_ALL" if confirmed else "DRY_RUN"}


class PreflightGateTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.portfolio_path = Path(self._tmpdir.name) / "positions.json"
        _FakeBroker.calls = []

        self._orig = {
            "Portfolio": runner.Portfolio,
            "get_broker": runner.get_broker,
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
            "fundamental_score": runner.pipeline.fundamental.score,
            "classify_code": runner.pipeline.news_filter.classify_code,
            "is_earnings_blackout": runner.pipeline.is_earnings_blackout,
            "has_earnings_risk": runner.event_risk.has_earnings_risk,
            "preflight_check": runner.market_preflight.check,
            "PREFLIGHT_ENABLED": config.PREFLIGHT_ENABLED,
            "build_trade_research_snapshot": runner.research_snapshot.build_trade_research_snapshot,
        }

        runner.broker_state_mod.fetch_broker_state = lambda trd_env: _NO_RISK_BROKER_STATE
        runner.event_risk.has_earnings_risk = lambda code, trade_date=None: False
        runner.research_snapshot.build_trade_research_snapshot = lambda *a, **kw: {}
        runner.Portfolio = lambda: Portfolio(path=self.portfolio_path)
        runner.get_broker = lambda code: _FakeBroker
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
        config.PREFLIGHT_ENABLED = True

        runner.scan = lambda codes, **kw: {
            "US.TEST": {"current_price": 100.0, "signal": "BUY",
                        "signal_strength": 0.9, "strategy_used": "atr_breakout"},
        }
        runner.smart_scan = runner.scan

    def tearDown(self):
        runner.Portfolio = self._orig["Portfolio"]
        runner.get_broker = self._orig["get_broker"]
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
        runner.pipeline.fundamental.score = self._orig["fundamental_score"]
        runner.pipeline.news_filter.classify_code = self._orig["classify_code"]
        runner.pipeline.is_earnings_blackout = self._orig["is_earnings_blackout"]
        runner.event_risk.has_earnings_risk = self._orig["has_earnings_risk"]
        runner.market_preflight.check = self._orig["preflight_check"]
        runner.research_snapshot.build_trade_research_snapshot = self._orig["build_trade_research_snapshot"]
        config.PREFLIGHT_ENABLED = self._orig["PREFLIGHT_ENABLED"]
        self._tmpdir.cleanup()

    def test_buy_proceeds_when_preflight_passes(self):
        runner.market_preflight.check = lambda code: PreflightResult(True, "ok")
        Portfolio(path=self.portfolio_path)._save()

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        self.assertEqual(len(_FakeBroker.calls), 1,
                          "order must reach the broker when preflight passes")

    def test_buy_blocked_when_preflight_fails(self):
        runner.market_preflight.check = lambda code: PreflightResult(False, "quote not logged in")
        Portfolio(path=self.portfolio_path)._save()

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        self.assertEqual(len(_FakeBroker.calls), 0,
                          "order must never reach the broker when preflight fails")
        reloaded = Portfolio(path=self.portfolio_path)
        self.assertNotIn("US.TEST", reloaded.data["positions"],
                          "portfolio state must be untouched when preflight blocks the order")

    def test_preflight_not_invoked_when_disabled(self):
        config.PREFLIGHT_ENABLED = False

        def _should_not_be_called(code):
            raise AssertionError("market_preflight.check must not run when PREFLIGHT_ENABLED=False")
        runner.market_preflight.check = _should_not_be_called
        Portfolio(path=self.portfolio_path)._save()

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        self.assertEqual(len(_FakeBroker.calls), 1)

    def test_preflight_not_invoked_on_dry_run(self):
        def _should_not_be_called(code):
            raise AssertionError("market_preflight.check must not run when confirmed=False")
        runner.market_preflight.check = _should_not_be_called
        Portfolio(path=self.portfolio_path)._save()

        runner.run_once(codes=["US.TEST"], confirmed=False,
                         auto_route=False, strategy_name="atr_breakout")


if __name__ == "__main__":
    unittest.main()
