"""
Unit tests for engine/runner.py::run_once() — 2026-07-11 fix verifying the
Pyramid scale-in path (config.PYRAMID_ENABLED) is gated by the same
macro_block used for BUY/PROMOTE/Replacement, instead of bypassing every
circuit breaker (news/QQQ-technical/market-weather/MAX_DRAWDOWN_HALT).

Before the fix: `if config.PYRAMID_ENABLED:` had no macro_block check at
all, so a pyramid add-on (an increase in portfolio risk exposure, same
category as a new BUY) could still fire while every other risk-increasing
path was correctly blocked.

Same monkeypatch approach as engine/test_runner_drawdown_halt.py — every
external dependency run_once() calls is patched so only the control flow
around macro_block/pyramid is under test.

Run:  python -m unittest engine.test_runner_pyramid_gate -v
"""
import tempfile
import unittest
from pathlib import Path

import config
import data.fetcher as fetcher_mod
import engine.runner as runner
from portfolio.tracker import Portfolio


class _FakeTracker:
    def __init__(self, *a, **kw): pass
    def update_position_metrics(self, *a, **kw): pass
    def log_exit(self, *a, **kw): pass
    def log_entry(self, *a, **kw): pass


class TestPyramidGatedByMacroBlock(unittest.TestCase):

    def setUp(self):
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
            "PYRAMID_ENABLED": config.PYRAMID_ENABLED,
        }

        runner.Portfolio = lambda: Portfolio(path=self.portfolio_path)
        runner.filter_open = lambda codes: list(codes)
        runner.regime.qqq_macro_halt = lambda df: False
        runner.market_weather.market_weather = lambda df: 1  # neutral, not crisis
        runner.alert.trade_sell = lambda *a, **kw: None
        runner.alert.trade_buy = lambda *a, **kw: None
        runner.TradeTracker = _FakeTracker
        fetcher_mod.fetch_kline = lambda *a, **kw: None
        config.PYRAMID_ENABLED = True

        # Default: no breaker active, no drawdown breach — overridden per test.
        runner.guard.check_max_drawdown = lambda portfolio: True
        runner.news_sentiment.macro_circuit_breaker = lambda: ""

        def _mock_place_order(code, side, qty, price, trd_env, env_label, confirmed):
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
        config.PYRAMID_ENABLED = self._orig["PYRAMID_ENABLED"]
        self._tmpdir.cleanup()

    def _seed_pyramid_candidate(self, portfolio):
        """A held position whose new signal is meaningfully stronger than
        when it was opened, priced within the pyramid's above-cost cap —
        i.e. exactly the shape that should trigger a scale-in add."""
        portfolio.data["positions"]["US.TEST"] = {
            "side": "BUY", "entry_price": 100.0, "avg_cost": 100.0, "qty": 10,
            "signal_strength": 0.3, "strategy": "atr_breakout", "entry_atr": 1.0,
            "atr_mult": config.ATR_MULT_BASE, "breakeven_locked": False,
            "trail_stop": None, "entry_time": "2020-01-01T00:00:00",
            "score_label": "FULL", "total_score": 90.0,
        }
        portfolio._save()
        runner.scan = lambda codes, **kw: {
            "US.TEST": {"current_price": 102.0, "signal": "BUY",
                        "signal_strength": 0.6, "strategy_used": "atr_breakout"},
        }
        runner.smart_scan = runner.scan

    def test_pyramid_blocked_when_macro_block_active(self):
        runner.news_sentiment.macro_circuit_breaker = lambda: "NEWS_CIRCUIT_BREAKER: test"
        portfolio = Portfolio(path=self.portfolio_path)
        self._seed_pyramid_candidate(portfolio)

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        self.assertEqual(len(self.placed_orders), 0,
                          "pyramid add must not fire while macro_block (news) is active")

    def test_pyramid_blocked_during_max_drawdown_halt(self):
        runner.guard.check_max_drawdown = lambda portfolio: False  # force breach
        portfolio = Portfolio(path=self.portfolio_path)
        self._seed_pyramid_candidate(portfolio)

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        self.assertEqual(len(self.placed_orders), 0,
                          "pyramid add must not fire during a MAX_DRAWDOWN_HALT")

    def test_pyramid_fires_when_no_breaker_active(self):
        portfolio = Portfolio(path=self.portfolio_path)
        self._seed_pyramid_candidate(portfolio)

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        buys = [o for o in self.placed_orders if o["side"] == "BUY"]
        self.assertEqual(len(buys), 1,
                          "pyramid add should fire normally with no breaker active")
        self.assertEqual(buys[0]["code"], "US.TEST")

    def test_exit_still_runs_while_pyramid_blocked(self):
        """Confirm the pyramid fix didn't collaterally touch EXIT handling —
        a losing position on the same code must still stop out even while
        macro_block also blocks the pyramid add that would otherwise fire
        for a still-BUY signal."""
        runner.news_sentiment.macro_circuit_breaker = lambda: "NEWS_CIRCUIT_BREAKER: test"
        portfolio = Portfolio(path=self.portfolio_path)
        portfolio.data["positions"]["US.TEST"] = {
            "side": "BUY", "entry_price": 100.0, "avg_cost": 100.0, "qty": 10,
            "signal_strength": 0.3, "strategy": "atr_breakout", "entry_atr": 1.0,
            "atr_mult": config.ATR_MULT_BASE, "breakeven_locked": False,
            "trail_stop": None, "entry_time": "2020-01-01T00:00:00",
            "score_label": "FULL", "total_score": 90.0,
        }
        portfolio._save()
        # -10%: hits hard stop-loss regardless of the BUY signal below
        runner.scan = lambda codes, **kw: {
            "US.TEST": {"current_price": 90.0, "signal": "HOLD",
                        "signal_strength": 0.0, "strategy_used": "atr_breakout"},
        }
        runner.smart_scan = runner.scan

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        sells = [o for o in self.placed_orders if o["side"] == "SELL"]
        self.assertEqual(len(sells), 1,
                          "EXIT must still fire even while macro_block blocks pyramid")

    def test_normal_buy_and_replacement_gate_unchanged(self):
        """Regression guard: a fresh BUY signal (no existing position) is
        still blocked by macro_block exactly as before — the pyramid fix
        must not have altered the pre-existing BUY-loop gate."""
        runner.news_sentiment.macro_circuit_breaker = lambda: "NEWS_CIRCUIT_BREAKER: test"
        portfolio = Portfolio(path=self.portfolio_path)
        portfolio._save()  # empty — no positions, only a fresh BUY candidate

        runner.scan = lambda codes, **kw: {
            "US.TEST": {"current_price": 100.0, "signal": "BUY",
                        "signal_strength": 0.9, "strategy_used": "atr_breakout"},
        }
        runner.smart_scan = runner.scan

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        self.assertEqual(len(self.placed_orders), 0,
                          "fresh BUY must still be blocked by macro_block, unchanged")


if __name__ == "__main__":
    unittest.main()
