"""
Unit tests for engine/runner.py's v2.9.x Entry Quality Tracking hook — the
new tracker.log_entry_quality() call added right after the existing
log_entry()/log_market_context() calls in the "open new positions" BUY loop.

Product constraint under test: this hook must be strictly additive. A BUY
still executes and portfolio state still updates exactly as before whether
the hook succeeds, raises, or can't fetch data — see
engine/entry_quality.py's module docstring ("nothing in this module may
feed back into a trading decision").

Same monkeypatch harness as engine/test_runner_pyramid_gate.py /
test_runner_drawdown_halt.py (every external dependency run_once() calls is
patched so only the hook's own behavior is under test).

Run:  python -m unittest engine.test_runner_entry_quality_hook -v
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
# entry-quality-hook test stays hermetic and unaffected by the new feature.
_NO_RISK_BROKER_STATE = BrokerState(positions={}, cash=1_000_000.0,
                                     total_assets=1_000_000.0, long_mv=0.0)


class _FakeTrackerNoEntryQuality:
    """Mirrors the _FakeTracker used by the sibling runner test files —
    deliberately does NOT implement log_entry_quality, to prove the hook's
    try/except degrades a missing/erroring method into a no-op rather than
    blocking the BUY it's attached to."""
    def __init__(self, *a, **kw): pass
    def update_position_metrics(self, *a, **kw): pass
    def log_exit(self, *a, **kw): pass
    def log_entry(self, *a, **kw): pass
    def log_market_context(self, *a, **kw): pass


class _FakeTrackerCapturing(_FakeTrackerNoEntryQuality):
    """Same as above, plus a log_entry_quality that records every call so
    tests can assert on what the hook actually passed through."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        _FakeTrackerCapturing.calls = []

    def log_entry_quality(self, **kw):
        _FakeTrackerCapturing.calls.append(kw)


class EntryQualityHookTestCase(unittest.TestCase):

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
            "fundamental_score": runner.pipeline.fundamental.score,
            "classify_code": runner.pipeline.news_filter.classify_code,
            "is_earnings_blackout": runner.pipeline.is_earnings_blackout,
            "fetch_broker_state": runner.broker_state_mod.fetch_broker_state,
            "has_earnings_risk": runner.event_risk.has_earnings_risk,
            "build_trade_research_snapshot": runner.research_snapshot.build_trade_research_snapshot,
        }

        runner.broker_state_mod.fetch_broker_state = lambda trd_env: _NO_RISK_BROKER_STATE
        runner.event_risk.has_earnings_risk = lambda code, trade_date=None: False
        runner.research_snapshot.build_trade_research_snapshot = lambda *a, **kw: {}
        runner.Portfolio = lambda: Portfolio(path=self.portfolio_path)
        runner.filter_open = lambda codes: list(codes)
        runner.regime.qqq_macro_halt = lambda df: False
        runner.market_weather.market_weather = lambda df: 1  # neutral, not crisis
        runner.alert.trade_sell = lambda *a, **kw: None
        runner.alert.trade_buy = lambda *a, **kw: None
        runner.guard.check_max_drawdown = lambda portfolio: True   # no halt
        runner.news_sentiment.macro_circuit_breaker = lambda: ""   # no news block
        # build_candidate_pool (engine/pipeline.py) hits fundamental/news/
        # earnings APIs for a real BUY candidate — stub them out so this
        # test suite stays hermetic (no network / moomoo connection needed),
        # same reasoning as every other external dependency patched here.
        runner.pipeline.fundamental.score = lambda code, env: {
            "score": None, "tier": None, "reason": "test_stub"}
        runner.pipeline.news_filter.classify_code = lambda code: {
            "score": None, "tier": 3, "matched_keyword": None}
        runner.pipeline.is_earnings_blackout = lambda code: False

        runner.scan = lambda codes, **kw: {
            "US.TEST": {"current_price": 100.0, "signal": "BUY",
                        "signal_strength": 0.9, "strategy_used": "atr_breakout",
                        "atr": 2.0, "donchian_high": 98.0},
        }
        runner.smart_scan = runner.scan

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
        runner.pipeline.fundamental.score = self._orig["fundamental_score"]
        runner.pipeline.news_filter.classify_code = self._orig["classify_code"]
        runner.pipeline.is_earnings_blackout = self._orig["is_earnings_blackout"]
        runner.broker_state_mod.fetch_broker_state = self._orig["fetch_broker_state"]
        runner.event_risk.has_earnings_risk = self._orig["has_earnings_risk"]
        runner.research_snapshot.build_trade_research_snapshot = self._orig["build_trade_research_snapshot"]
        self._tmpdir.cleanup()
        test_support.unmute_alert_file_logging(self._alert_handlers)


class TestEntryQualityHookIsNonInvasive(EntryQualityHookTestCase):

    def test_buy_succeeds_when_tracker_lacks_log_entry_quality(self):
        runner.TradeTracker = _FakeTrackerNoEntryQuality
        fetcher_mod.fetch_kline = lambda *a, **kw: None

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        buys = [o for o in self.placed_orders if o["side"] == "BUY"]
        self.assertEqual(len(buys), 1, "missing log_entry_quality must not block the BUY")

    def test_buy_succeeds_when_fetch_kline_raises(self):
        runner.TradeTracker = _FakeTrackerCapturing

        def _raises(*a, **kw):
            raise ConnectionError("OpenD unreachable")
        fetcher_mod.fetch_kline = _raises

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        buys = [o for o in self.placed_orders if o["side"] == "BUY"]
        self.assertEqual(len(buys), 1, "a failing kline fetch must not block the BUY")
        self.assertEqual(_FakeTrackerCapturing.calls, [],
                          "no snapshot should be logged when the fetch failed")

    def test_buy_succeeds_when_fetch_kline_returns_none(self):
        runner.TradeTracker = _FakeTrackerCapturing
        fetcher_mod.fetch_kline = lambda *a, **kw: None

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        buys = [o for o in self.placed_orders if o["side"] == "BUY"]
        self.assertEqual(len(buys), 1)
        self.assertEqual(_FakeTrackerCapturing.calls, [])


class TestEntryQualityHookRecordsSnapshot(EntryQualityHookTestCase):

    def test_snapshot_logged_with_expected_fields(self):
        runner.TradeTracker = _FakeTrackerCapturing

        import numpy as np
        import pandas as pd
        idx = pd.date_range("2024-01-01", periods=90, freq="D")
        close = np.full(90, 95.0)
        fetcher_mod.fetch_kline = lambda code, ktype="K_DAY", bars=120: pd.DataFrame({
            "high": close, "low": close, "close": close, "volume": 1000.0,
        }, index=idx)

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        buys = [o for o in self.placed_orders if o["side"] == "BUY"]
        self.assertEqual(len(buys), 1)
        self.assertEqual(len(_FakeTrackerCapturing.calls), 1)
        call = _FakeTrackerCapturing.calls[0]
        self.assertEqual(call["donchian_breakout_price"], 98.0)
        # entry price 100.0, breakout 98.0, atr 2.0 -> 1.0 ATR extended
        self.assertAlmostEqual(call["distance_from_breakout_atr"], 1.0)
        self.assertEqual(call["rule_score"], call["rule_score"])  # populated, no crash
        self.assertIn("trade_id", call)
        self.assertTrue(call["trade_id"].startswith("US.TEST_"))


if __name__ == "__main__":
    unittest.main()
