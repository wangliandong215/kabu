"""
Unit tests for engine/runner.py::run_once() — 2026-07-11 fix verifying a
MAX_DRAWDOWN circuit breach no longer skips the exit-check loop.

Before the fix, `if not guard.check_max_drawdown(portfolio): return` sat
before scan()/the exit-check loop, so a drawdown breach silently disabled
hard stop-loss/ATR trailing/take-profit checks for every open position for
that whole pass — exactly when they matter most. The fix folds the halt
into `macro_block` (computed after the exit-check loop runs), so it blocks
new BUY/PROMOTE/Replacement the same way the pre-existing news/QQQ-technical
/market-weather breakers already do, without ever gating SELL/EXIT.

Rather than driving the full pipeline against live quotes/news/fundamentals,
every external dependency run_once() calls is monkeypatched so only the
control flow around drawdown_halt/macro_block/exit-check is under test —
same approach as engine/test_runner_replacement.py.

Run:  python -m unittest engine.test_runner_drawdown_halt -v
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

# v2.10 Portfolio Risk Manager queries the real broker every run_once() pass
# — stub it to a comfortably-under-95%-exposure state so this pre-existing
# drawdown_halt/macro_block test stays hermetic and unaffected by the new
# feature.
_NO_RISK_BROKER_STATE = BrokerState(positions={}, cash=1_000_000.0,
                                     total_assets=1_000_000.0, long_mv=0.0)


class _FakeTracker:
    def __init__(self, *a, **kw): pass
    def update_position_metrics(self, *a, **kw): pass
    def log_exit(self, *a, **kw): pass
    def log_entry(self, *a, **kw): pass


class TestDrawdownHaltStillRunsExitCheck(unittest.TestCase):

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
        }

        runner.broker_state_mod.fetch_broker_state = lambda trd_env: _NO_RISK_BROKER_STATE
        runner.event_risk.has_earnings_risk = lambda code, trade_date=None: False
        runner.research_snapshot.build_trade_research_snapshot = lambda *a, **kw: {}
        runner.Portfolio = lambda: Portfolio(path=self.portfolio_path)
        runner.filter_open = lambda codes: list(codes)  # pretend everything's open
        runner.news_sentiment.macro_circuit_breaker = lambda: ""  # no news breaker
        runner.regime.qqq_macro_halt = lambda df: False
        runner.market_weather.market_weather = lambda df: 1  # neutral, not crisis
        runner.alert.trade_sell = lambda *a, **kw: None
        runner.alert.trade_buy = lambda *a, **kw: None
        runner.TradeTracker = _FakeTracker
        fetcher_mod.fetch_kline = lambda *a, **kw: None

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
        runner.broker_state_mod.fetch_broker_state = self._orig["fetch_broker_state"]
        runner.event_risk.has_earnings_risk = self._orig["has_earnings_risk"]
        runner.research_snapshot.build_trade_research_snapshot = self._orig["build_trade_research_snapshot"]
        self._tmpdir.cleanup()
        test_support.unmute_alert_file_logging(self._alert_handlers)

    def _seed_losing_position(self, portfolio):
        portfolio.data["positions"]["US.TEST"] = {
            "side": "BUY", "entry_price": 100.0, "avg_cost": 100.0, "qty": 10,
            "signal_strength": 0.5, "strategy": "atr_breakout", "entry_atr": 1.0,
            "atr_mult": config.ATR_MULT_BASE, "breakeven_locked": False,
            "trail_stop": None, "entry_time": "2020-01-01T00:00:00",
            "score_label": "FULL", "total_score": 90.0,
        }
        portfolio._save()

    def test_exit_check_runs_and_sell_placed_during_drawdown_halt(self):
        runner.guard.check_max_drawdown = lambda portfolio: False  # force breach

        portfolio = Portfolio(path=self.portfolio_path)
        self._seed_losing_position(portfolio)

        # -10% from entry — well past the 5% hard stop-loss, independent of
        # strategy_signal/ATR trail, so this alone proves the exit-check
        # loop ran (it wouldn't have, pre-fix, once check_max_drawdown()
        # returned False).
        runner.scan = lambda codes, **kw: {
            "US.TEST": {"current_price": 90.0, "signal": "HOLD",
                        "signal_strength": 0.0, "strategy_used": "atr_breakout"},
        }
        runner.smart_scan = runner.scan

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        sells = [o for o in self.placed_orders if o["side"] == "SELL"]
        buys = [o for o in self.placed_orders if o["side"] == "BUY"]
        self.assertEqual(len(sells), 1,
                          "hard stop-loss SELL must still fire during a drawdown halt")
        self.assertEqual(sells[0]["code"], "US.TEST")
        self.assertEqual(len(buys), 0,
                          "no BUY should happen while drawdown_halt is active")

    def test_exit_check_did_not_run_before_fix(self):
        """Reproduces the pre-fix bug directly against the old code path, to
        prove this test suite would have caught it: manually re-inserting
        the old early `return` (patched onto the live check_max_drawdown
        call site is not possible without editing source, so instead this
        calls the guard exactly as the old code did and asserts *that* the
        old short-circuit condition is what the fix removed) — i.e. this
        documents the regression rather than re-testing removed code."""
        # The old code was: `if not guard.check_max_drawdown(portfolio): return`
        # at the very top, before scan()/exit-check. Confirm today's run_once
        # no longer contains that early return by asserting behavior above
        # (test_exit_check_runs_and_sell_placed_during_drawdown_halt) passes
        # even when check_max_drawdown() returns False — already covered.
        # This second test just pins the *pre-fix* control-flow assumption
        # via source inspection so a future refactor can't silently
        # reintroduce it without this test failing.
        import inspect
        source = inspect.getsource(runner.run_once)
        guard_check_index = source.index("check_max_drawdown")
        watchlist_index = source.index("watchlist = codes or select_watchlist()")
        exit_loop_index = source.index('for code, pos in list(portfolio.data["positions"].items())')
        self.assertLess(
            guard_check_index, exit_loop_index,
            "check_max_drawdown must be evaluated before the exit-check loop",
        )
        # Only the drawdown-guard block itself (up to where the watchlist is
        # resolved) must be free of a bare `return` — the *separate*,
        # legitimate "no markets open" early return further down (between
        # here and the exit-check loop) is out of scope for this fix and
        # expected to still be there.
        drawdown_guard_block = source[guard_check_index:watchlist_index]
        self.assertNotIn(
            "return", drawdown_guard_block,
            "an unconditional `return` right after the drawdown check would "
            "skip scan()/exit-checking again — this is the exact bug this "
            "fix removed",
        )

    def test_buy_blocked_during_drawdown_halt(self):
        runner.guard.check_max_drawdown = lambda portfolio: False  # force breach

        portfolio = Portfolio(path=self.portfolio_path)
        portfolio._save()  # empty portfolio — nothing to exit, only a BUY candidate

        runner.scan = lambda codes, **kw: {
            "US.TEST": {"current_price": 100.0, "signal": "BUY",
                        "signal_strength": 0.9, "strategy_used": "atr_breakout"},
        }
        runner.smart_scan = runner.scan

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        self.assertEqual(
            len(self.placed_orders), 0,
            "no BUY should be placed while drawdown_halt blocks new entries",
        )

    def test_no_halt_does_not_block_exit_either(self):
        """Control case: with check_max_drawdown() True (no breach), the
        same losing position still exits normally — proves the fix didn't
        accidentally change exit behavior in the non-halt case."""
        runner.guard.check_max_drawdown = lambda portfolio: True  # no breach

        portfolio = Portfolio(path=self.portfolio_path)
        self._seed_losing_position(portfolio)

        runner.scan = lambda codes, **kw: {
            "US.TEST": {"current_price": 90.0, "signal": "HOLD",
                        "signal_strength": 0.0, "strategy_used": "atr_breakout"},
        }
        runner.smart_scan = runner.scan

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        sells = [o for o in self.placed_orders if o["side"] == "SELL"]
        self.assertEqual(len(sells), 1, "exit-check must still work with no drawdown halt")


if __name__ == "__main__":
    unittest.main()
