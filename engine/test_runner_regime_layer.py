"""
Unit tests for engine/runner.py's Regime Observation Layer call site
(added 2026-09-23, see regime/ package docstring).

Unlike the other engine/test_runner_*.py files, this file deliberately
does NOT stub out runner.regime_evaluate_and_log -- it lets the real
regime.evaluate_and_log()/RuleRegimeProvider run (pointed at a temp log
path) so the wiring itself is under test: one JSON line per pass, and no
influence whatsoever on BUY/SELL/order-count behavior regardless of what
regime the layer computes. Every other run_once() dependency is stubbed
exactly like engine/test_runner_drawdown_halt.py.

Run:  python -m unittest engine.test_runner_regime_layer -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import config
import data.fetcher as fetcher_mod
import engine.runner as runner
import regime as regime_pkg
import test_support
from portfolio.broker_state import BrokerState
from portfolio.tracker import Portfolio

_NO_RISK_BROKER_STATE = BrokerState(positions={}, cash=1_000_000.0,
                                     total_assets=1_000_000.0, long_mv=0.0)


class _FakeTracker:
    def __init__(self, *a, **kw): pass
    def update_position_metrics(self, *a, **kw): pass
    def log_exit(self, *a, **kw): pass
    def log_entry(self, *a, **kw): pass


class TestRegimeObservationLayerWiring(unittest.TestCase):

    def setUp(self):
        self._alert_handlers = test_support.mute_alert_file_logging()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.portfolio_path = Path(self._tmpdir.name) / "positions.json"
        self.placed_orders = []

        self._orig_regime_log_path = regime_pkg.REGIME_LOG_PATH
        regime_pkg.REGIME_LOG_PATH = Path(self._tmpdir.name) / "regime_log.jsonl"

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
            "_qqq_above_ma": runner._qqq_above_ma,
            "fundamental_score": runner.pipeline.fundamental.score,
            "classify_code": runner.pipeline.news_filter.classify_code,
            "is_earnings_blackout": runner.pipeline.is_earnings_blackout,
            "log_snapshot": runner.portfolio_risk_manager.log_snapshot,
            "log_diff": runner.portfolio_risk_manager.log_diff,
            "log_qqq_snapshot": runner.portfolio_risk_manager.log_qqq_snapshot,
            "pm_log_attempt": runner.portfolio_position_manager.log_attempt,
        }

        runner.broker_state_mod.fetch_broker_state = lambda trd_env: _NO_RISK_BROKER_STATE
        runner.event_risk.has_earnings_risk = lambda code, trade_date=None: False
        runner.research_snapshot.build_trade_research_snapshot = lambda *a, **kw: {}
        runner.Portfolio = lambda: Portfolio(path=self.portfolio_path)
        runner.filter_open = lambda codes: list(codes)
        runner.news_sentiment.macro_circuit_breaker = lambda: ""
        runner.regime.qqq_macro_halt = lambda df: False
        runner.alert.trade_sell = lambda *a, **kw: None
        runner.alert.trade_buy = lambda *a, **kw: None
        runner.TradeTracker = _FakeTracker
        fetcher_mod.fetch_kline = lambda *a, **kw: None
        runner.guard.check_max_drawdown = lambda portfolio: True   # no halt
        runner.pipeline.fundamental.score = lambda code, env: {
            "score": None, "tier": None, "reason": "test_stub"}
        runner.pipeline.news_filter.classify_code = lambda code: {
            "score": None, "tier": 3, "matched_keyword": None}
        runner.pipeline.is_earnings_blackout = lambda code: False
        runner.portfolio_risk_manager.log_snapshot = lambda *a, **kw: None
        runner.portfolio_risk_manager.log_diff = lambda *a, **kw: None
        runner.portfolio_risk_manager.log_qqq_snapshot = lambda *a, **kw: None
        runner.portfolio_position_manager.log_attempt = lambda *a, **kw: None

        runner.scan = lambda codes, **kw: {
            "US.TEST": {"current_price": 100.0, "signal": "BUY",
                        "signal_strength": 0.9, "strategy_used": "atr_breakout",
                        "atr": 2.0, "donchian_high": 98.0},
        }
        runner.smart_scan = runner.scan

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
        runner._qqq_above_ma = self._orig["_qqq_above_ma"]
        runner.pipeline.fundamental.score = self._orig["fundamental_score"]
        runner.pipeline.news_filter.classify_code = self._orig["classify_code"]
        runner.pipeline.is_earnings_blackout = self._orig["is_earnings_blackout"]
        runner.portfolio_risk_manager.log_snapshot = self._orig["log_snapshot"]
        runner.portfolio_risk_manager.log_diff = self._orig["log_diff"]
        runner.portfolio_risk_manager.log_qqq_snapshot = self._orig["log_qqq_snapshot"]
        runner.portfolio_position_manager.log_attempt = self._orig["pm_log_attempt"]
        regime_pkg.REGIME_LOG_PATH = self._orig_regime_log_path
        self._tmpdir.cleanup()
        test_support.unmute_alert_file_logging(self._alert_handlers)

    def _run(self):
        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

    def _read_regime_log(self):
        if not regime_pkg.REGIME_LOG_PATH.exists():
            return []
        lines = regime_pkg.REGIME_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(l) for l in lines]

    def test_one_pass_writes_exactly_one_regime_log_line(self):
        runner.market_weather.market_weather = lambda df: 2
        runner._qqq_above_ma = lambda: True

        self._run()

        records = self._read_regime_log()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["regime"], "BULL")

    def test_bear_classification_does_not_block_the_buy(self):
        # weather_code=1 (chop) -> RuleRegimeProvider classifies BEAR, but
        # weather_code==1 alone does NOT trip runner.py's macro_block (only
        # weather_code==0 does) -- this isolates "does the new Observation
        # Layer itself gate anything" from the pre-existing market-weather
        # circuit breaker, which is out of scope here.
        runner.market_weather.market_weather = lambda df: 1
        runner._qqq_above_ma = lambda: True

        self._run()

        records = self._read_regime_log()
        self.assertEqual(records[0]["regime"], "BEAR")
        # qqq_above_ma=True here also fires the pre-existing, unrelated QQQ
        # Beta-floor top-up (section 2d) -- filter to the candidate under
        # test so that separate buy path doesn't confound this assertion.
        buys = [o for o in self.placed_orders if o["side"] == "BUY" and o["code"] == "US.TEST"]
        self.assertEqual(len(buys), 1,
                          "BEAR regime classification must not gate the BUY -- "
                          "the Observation Layer is read-only")

    def test_crash_classification_via_below_ma200_does_not_block_the_buy(self):
        # qqq_above_ma=False alone (with weather_code=2, safe) makes the new
        # layer classify BEAR (not CRASH -- only drawdown_halt or
        # weather_code==0 reach CRASH); use drawdown_halt=False/weather=2 to
        # confirm _qqq_above_ma()=False by itself still lets BUY proceed
        # (macro_block is untouched by this layer either way).
        runner.market_weather.market_weather = lambda df: 2
        runner._qqq_above_ma = lambda: False

        self._run()

        records = self._read_regime_log()
        self.assertEqual(records[0]["regime"], "BEAR")
        buys = [o for o in self.placed_orders if o["side"] == "BUY" and o["code"] == "US.TEST"]
        self.assertEqual(len(buys), 1,
                          "Observation Layer regime must never gate a BUY")

    def test_regime_log_record_shape(self):
        runner.market_weather.market_weather = lambda df: 2
        runner._qqq_above_ma = lambda: True

        self._run()

        record = self._read_regime_log()[0]
        for key in ("timestamp", "regime", "confidence", "risk_level",
                    "reason_codes", "source", "weather_code", "qqq_above_ma",
                    "drawdown_halt"):
            self.assertIn(key, record)
        self.assertEqual(record["source"], "rules")


if __name__ == "__main__":
    unittest.main()
