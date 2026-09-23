"""
Unit tests for engine/runner.py's v2.12 Portfolio Position Manager integration
(Phase 2, Observation Mode) — the account-level exposure ceiling layered on
top of (never replacing) the v2.10 Layer2 / v2.11 Layer1 gates, threaded
through the 2a/2b/2c/2d buy paths via the shared `pm_state`/
`remaining_budget` pattern (see risk/portfolio_position_manager.py and
engine/runner.py's _PMPassState/_pm_plan_buy/_pm_record_fill).

Same hermetic monkeypatch harness as engine/test_portfolio_risk_manager_gate.py
— every external dependency run_once() calls is patched, including
portfolio.broker_state.fetch_broker_state.

Run:  python -m unittest engine.test_portfolio_position_manager_gate -v
"""
import tempfile
import unittest
from pathlib import Path

import config
import data.fetcher as fetcher_mod
import engine.runner as runner
import test_support
from portfolio.broker_state import BrokerState
from portfolio.tracker import Portfolio


class _FakeTracker:
    def __init__(self, *a, **kw): pass
    def update_position_metrics(self, *a, **kw): pass
    def log_exit(self, *a, **kw): pass
    def log_entry(self, *a, **kw): pass
    def log_market_context(self, *a, **kw): pass


class PortfolioPositionManagerGateTestCase(unittest.TestCase):

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
            "log_snapshot": runner.portfolio_risk_manager.log_snapshot,
            "log_diff": runner.portfolio_risk_manager.log_diff,
            "log_qqq_snapshot": runner.portfolio_risk_manager.log_qqq_snapshot,
            "pm_log_attempt": runner.portfolio_position_manager.log_attempt,
            "fundamental_score": runner.pipeline.fundamental.score,
            "classify_code": runner.pipeline.news_filter.classify_code,
            "is_earnings_blackout": runner.pipeline.is_earnings_blackout,
            "_qqq_above_ma": runner._qqq_above_ma,
            "PM_ENABLED": config.PORTFOLIO_POSITION_MANAGER_ENABLED,
            "has_earnings_risk": runner.event_risk.has_earnings_risk,
            "build_trade_research_snapshot": runner.research_snapshot.build_trade_research_snapshot,
            "get_price": runner.get_price,
            "regime_evaluate_and_log": runner.regime_evaluate_and_log,
        }
        # Regime Observation Layer (2026-09-23) — pure/no-op stub so tests
        # never write to the real C:\KabuData\portfolio\regime_log.jsonl.
        runner.regime_evaluate_and_log = lambda *a, **kw: None
        # _qqq_above_ma() hits live quotes -- stub True so the QQQ Beta floor's
        # own MA200 exit/top-up logic never confounds these tests, and so the
        # Position Manager regime classifier (which also calls this) is
        # deterministic.
        runner._qqq_above_ma = lambda: True
        runner.get_price = lambda code: 700.0
        runner.event_risk.has_earnings_risk = lambda code, trade_date=None: False
        runner.research_snapshot.build_trade_research_snapshot = lambda *a, **kw: {}

        runner.Portfolio = lambda: Portfolio(path=self.portfolio_path)
        runner.filter_open = lambda codes: list(codes)
        runner.regime.qqq_macro_halt = lambda df: False
        # weather_code=1 (chop/caution) -> classify_market_regime() below
        # resolves to CAUTION (75% placeholder cap, see config.py
        # PORTFOLIO_REGIME_MAX_EXPOSURE) without also tripping macro_block
        # (only weather_code==0 does that) -- lets these tests isolate the
        # Position Manager gate from the pre-existing crisis circuit breaker.
        runner.market_weather.market_weather = lambda df: 1
        runner.alert.trade_sell = lambda *a, **kw: None
        runner.alert.trade_buy = lambda *a, **kw: None
        runner.guard.check_max_drawdown = lambda portfolio: True   # healthy -> drawdown_halt=False
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
        runner.get_price = self._orig["get_price"]
        runner.regime_evaluate_and_log = self._orig["regime_evaluate_and_log"]
        self._tmpdir.cleanup()
        test_support.unmute_alert_file_logging(self._alert_handlers)

    def _seed_fresh_buy_candidate(self, code="US.TEST", price=100.0):
        runner.scan = lambda codes, **kw: {
            code: {"current_price": price, "signal": "BUY",
                   "signal_strength": 0.9, "strategy_used": "atr_breakout"},
        }
        runner.smart_scan = runner.scan

    # ── Observation Mode must never change real order sizes ────────────────

    def test_observation_mode_does_not_change_buy_qty(self):
        """config.PORTFOLIO_POSITION_MANAGER_ENABLED=False (the shipped
        default) must leave the BUY untouched even though the remaining PM
        budget ($50, under CAUTION's 75% placeholder cap) is far below one
        $100 share -- proving Observation Mode really only logs, never
        clamps."""
        config.PORTFOLIO_POSITION_MANAGER_ENABLED = False
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: BrokerState(
            positions={}, cash=1_000_000.0, total_assets=1_000_000.0, long_mv=749_950.0)
        Portfolio(path=self.portfolio_path)._save()
        self._seed_fresh_buy_candidate()

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        buys = [o for o in self.placed_orders if o["side"] == "BUY" and o["code"] == "US.TEST"]
        self.assertEqual(len(buys), 1,
                          "Observation Mode must not block a BUY regardless of PM budget")
        self.assertGreater(buys[0]["qty"], 2,
                            "Observation Mode must not shrink the BUY qty below what unconstrained "
                            "sizing would produce")

    # ── ENABLED=True must actually clamp ────────────────────────────────────

    def test_enabled_mode_blocks_buy_when_budget_below_one_share(self):
        config.PORTFOLIO_POSITION_MANAGER_ENABLED = True
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: BrokerState(
            positions={}, cash=1_000_000.0, total_assets=1_000_000.0, long_mv=749_950.0)  # remaining budget=$50
        Portfolio(path=self.portfolio_path)._save()
        self._seed_fresh_buy_candidate()

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        buys = [o for o in self.placed_orders if o["side"] == "BUY" and o["code"] == "US.TEST"]
        self.assertEqual(len(buys), 0,
                          "ENABLED=True must block a BUY when remaining budget ($50) is below "
                          "one $100 share")

    def test_enabled_mode_reduces_qty_to_remaining_budget(self):
        config.PORTFOLIO_POSITION_MANAGER_ENABLED = True
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: BrokerState(
            positions={}, cash=1_000_000.0, total_assets=1_000_000.0, long_mv=749_800.0)  # remaining budget=$200 -> 2 shares @ $100
        Portfolio(path=self.portfolio_path)._save()
        self._seed_fresh_buy_candidate()

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        buys = [o for o in self.placed_orders if o["side"] == "BUY" and o["code"] == "US.TEST"]
        self.assertEqual(len(buys), 1)
        self.assertLessEqual(buys[0]["qty"], 2,
                              "ENABLED=True must clamp qty to the $200 remaining budget (2 shares @ $100)")

    # ── Shared budget across multiple candidates in one pass (the batch/
    #    concurrency risk flagged in Phase 1) ───────────────────────────────

    def test_enabled_mode_shares_budget_sequentially_across_candidates(self):
        config.PORTFOLIO_POSITION_MANAGER_ENABLED = True
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: BrokerState(
            positions={}, cash=1_000_000.0, total_assets=1_000_000.0, long_mv=749_875.0)  # remaining budget=$125 -> 1 share @ $100 total, shared
        Portfolio(path=self.portfolio_path)._save()
        runner.scan = lambda codes, **kw: {
            "US.AAA": {"current_price": 100.0, "signal": "BUY",
                       "signal_strength": 0.9, "strategy_used": "atr_breakout"},
            "US.BBB": {"current_price": 100.0, "signal": "BUY",
                       "signal_strength": 0.9, "strategy_used": "atr_breakout"},
        }
        runner.smart_scan = runner.scan

        runner.run_once(codes=["US.AAA", "US.BBB"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        buys = [o for o in self.placed_orders if o["side"] == "BUY"]
        total_qty = sum(o["qty"] for o in buys)
        self.assertLessEqual(total_qty, 1,
                              "Combined BUY qty across BOTH candidates in the same pass must not "
                              "exceed the shared $125 PM budget (1 share @ $100), even though each "
                              "candidate independently qualifies on its own")


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
