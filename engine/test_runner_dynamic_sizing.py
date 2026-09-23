"""
Integration tests for V3.0-A Dynamic Position Sizing wiring in
engine/runner.py's "open new positions" BUY loop (risk/dynamic_sizing.py
supplies the pure confidence->multiplier mapping; this file is about the
end-to-end wiring: does the multiplier actually change what run_once()
buys, does it stay OFF outside score_env=="paper", and does the existing
config.DYNAMIC_POSITION_SIZING_ENABLED kill switch actually disable it).

Same monkeypatch harness as engine/test_runner_entry_quality_hook.py /
test_runner_pyramid_gate.py — every external dependency run_once() calls is
patched so only the sizing behavior under test is exercised. Confidence
itself is injected by monkeypatching engine.pipeline's confidence_score
module directly (bypassing gather_and_score's real I/O), so each test can
target an exact tier boundary from V3.0 spec section 八 without depending on
what engine/confidence_score.py's real formula happens to produce for a
synthetic test signal.

Run:  python -m unittest engine.test_runner_dynamic_sizing -v
"""
import tempfile
import unittest
from pathlib import Path

import config
import data.fetcher as fetcher_mod
import engine.runner as runner
import test_support
from engine.confidence_score import ConfidenceScoreResult
from portfolio.broker_state import BrokerState
from portfolio.tracker import Portfolio

_NO_RISK_BROKER_STATE = BrokerState(positions={}, cash=1_000_000.0,
                                     total_assets=1_000_000.0, long_mv=0.0)


class _FakeTracker:
    def __init__(self, *a, **kw):
        self.confidence_calls = []

    def update_position_metrics(self, *a, **kw): pass
    def log_exit(self, *a, **kw): pass
    def log_entry(self, *a, **kw): pass
    def log_market_context(self, *a, **kw): pass
    def log_entry_quality(self, *a, **kw): pass
    def log_research_snapshot(self, *a, **kw): pass

    def log_confidence_score(self, **kw):
        _FakeTracker.calls.append(kw)


def _stub_gather_and_score(confidence_value):
    """Matches engine.confidence_score.gather_and_score()'s (result, detail)
    return shape closely enough for engine/pipeline.py's build_candidate_pool
    to attach it to Candidate.confidence/.confidence_detail unchanged."""
    def _fn(code, rule_based_score, sig, tracker=None, execution="REAL", as_of=None):
        result = ConfidenceScoreResult(
            confidence_score=confidence_value, rule_component=rule_based_score,
            hmm_component=None, win_rate_component=None, expectancy_component=None,
            market_component=None, volatility_component=None, volume_component=None,
            weights_used={"rule": 0.30},
        )
        detail = {"formula_version": "test", "confidence_score": confidence_value}
        return result, detail
    return _fn


class DynamicSizingTestCase(unittest.TestCase):

    def setUp(self):
        self._alert_handlers = test_support.mute_alert_file_logging()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.portfolio_path = Path(self._tmpdir.name) / "positions.json"
        self.placed_orders = []
        _FakeTracker.calls = []

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
            "regime_evaluate_and_log": runner.regime_evaluate_and_log,
            "gather_and_score": runner.pipeline.confidence_score.gather_and_score,
            "CONFIDENCE_SCORE_ENABLED": config.CONFIDENCE_SCORE_ENABLED,
            "DYNAMIC_POSITION_SIZING_ENABLED": config.DYNAMIC_POSITION_SIZING_ENABLED,
            "TRD_ENV": config.TRD_ENV,
            "parse_trd_env": runner.parse_trd_env,
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
        runner.pipeline.fundamental.score = lambda code, env: {
            "score": None, "tier": None, "reason": "test_stub"}
        runner.pipeline.news_filter.classify_code = lambda code: {
            "score": None, "tier": 3, "matched_keyword": None}
        runner.pipeline.is_earnings_blackout = lambda code: False
        runner.TradeTracker = _FakeTracker
        fetcher_mod.fetch_kline = lambda *a, **kw: None
        config.CONFIDENCE_SCORE_ENABLED = True

        # Strong FULL-label signal, tight stop, plenty of open slots — makes
        # the "signal_based" strong/dynamic tier the binding constraint,
        # not something incidental like a per-strategy cap, so the
        # confidence multiplier is the only thing that can move qty between
        # test cases below.
        runner.scan = lambda codes, **kw: {
            "US.TEST": {"current_price": 100.0, "signal": "BUY",
                        "signal_strength": 0.9, "strategy_used": "atr_breakout",
                        "atr": 2.0, "stop_loss_pct": 0.05},
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
        runner.pipeline.fundamental.score = self._orig["fundamental_score"]
        runner.pipeline.news_filter.classify_code = self._orig["classify_code"]
        runner.pipeline.is_earnings_blackout = self._orig["is_earnings_blackout"]
        runner.broker_state_mod.fetch_broker_state = self._orig["fetch_broker_state"]
        runner.event_risk.has_earnings_risk = self._orig["has_earnings_risk"]
        runner.research_snapshot.build_trade_research_snapshot = self._orig["build_trade_research_snapshot"]
        runner.regime_evaluate_and_log = self._orig["regime_evaluate_and_log"]
        runner.pipeline.confidence_score.gather_and_score = self._orig["gather_and_score"]
        config.CONFIDENCE_SCORE_ENABLED = self._orig["CONFIDENCE_SCORE_ENABLED"]
        config.DYNAMIC_POSITION_SIZING_ENABLED = self._orig["DYNAMIC_POSITION_SIZING_ENABLED"]
        config.TRD_ENV = self._orig["TRD_ENV"]
        runner.parse_trd_env = self._orig["parse_trd_env"]
        self._tmpdir.cleanup()
        test_support.unmute_alert_file_logging(self._alert_handlers)

    def _fresh_portfolio(self):
        # Each call gets its own empty positions file — run_once() persists
        # fills to disk, so reusing one path across two calls in the same
        # test (baseline vs dynamic-sizing) would make the second call see
        # US.TEST "already held" and skip instead of re-buying.
        import uuid
        path = Path(self._tmpdir.name) / f"positions_{uuid.uuid4().hex}.json"
        runner.Portfolio = lambda: Portfolio(path=path)

    def _run_paper(self, confidence):
        self._fresh_portfolio()
        runner.pipeline.confidence_score.gather_and_score = _stub_gather_and_score(confidence)
        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

    def _buy_qty(self):
        buys = [o for o in self.placed_orders if o["side"] == "BUY"]
        return buys[0]["qty"] if buys else None


class TestConfidenceMultiplierChangesQty(DynamicSizingTestCase):

    def test_confidence_100_matches_baseline(self):
        config.DYNAMIC_POSITION_SIZING_ENABLED = False
        self._run_paper(confidence=100)
        baseline_qty = self._buy_qty()

        self.placed_orders.clear()
        config.DYNAMIC_POSITION_SIZING_ENABLED = True
        self._run_paper(confidence=100)
        full_conf_qty = self._buy_qty()

        self.assertEqual(baseline_qty, full_conf_qty)

    def _expected_qty(self, multiplier):
        # Oracle: call risk/sizing.py::calculate() directly with the exact
        # same arguments engine/runner.py's BUY loop passes for this test's
        # scan() stub (100% cash/price/score_label FULL, no other position
        # open yet) — this deliberately does NOT assume "final = base x
        # multiplier" holds by simple arithmetic, because the existing
        # MARKET_WEATHER_MAX_POSITION_PCT single-position hard cap sits
        # AFTER position_scale inside calculate() and does not itself scale
        # with confidence (see risk/sizing.py) — at multiplier=1.0 that cap
        # is the binding constraint for this scan() stub's strong-tier
        # signal, so a naive base*multiplier check would be wrong here even
        # though the wiring is correct. calculate() is the one source of
        # truth for "what does this position_scale actually buy".
        from risk.sizing import calculate as sizing_calculate
        fresh = Portfolio(path=Path(self._tmpdir.name) / "oracle.json")
        return sizing_calculate(
            fresh.available_cash(), 100.0, 0.9,
            total_capital=fresh.total_capital(), stop_loss_pct=0.05,
            kelly_factor=1.0, strategy="atr_breakout", open_positions=0,
            rsi_val=None, position_scale=multiplier,
            market_weather_code=1, score_label="FULL",
        )

    def test_confidence_88_matches_08x_multiplier(self):
        config.DYNAMIC_POSITION_SIZING_ENABLED = True
        self._run_paper(confidence=88)
        self.assertEqual(self._buy_qty(), self._expected_qty(0.8))

    def test_confidence_61_matches_04x_multiplier(self):
        config.DYNAMIC_POSITION_SIZING_ENABLED = True
        self._run_paper(confidence=61)
        self.assertEqual(self._buy_qty(), self._expected_qty(0.4))

    def test_lower_confidence_never_buys_more_shares(self):
        # Non-strict: at the top two tiers this scan() stub's strong-tier
        # signal is bound by the pre-existing MARKET_WEATHER_MAX_POSITION_PCT
        # single-position cap (not by the confidence multiplier itself), so
        # 100 and 88 can legitimately tie — see _expected_qty()'s docstring.
        # The property that must always hold is monotonic non-increase.
        config.DYNAMIC_POSITION_SIZING_ENABLED = True
        qtys = {}
        for confidence in (100, 88, 61):
            self.placed_orders.clear()
            self._run_paper(confidence=confidence)
            qtys[confidence] = self._buy_qty()
        self.assertGreaterEqual(qtys[100], qtys[88])
        self.assertGreater(qtys[88], qtys[61])

    def test_confidence_48_skips_the_trade_entirely(self):
        config.DYNAMIC_POSITION_SIZING_ENABLED = True
        self._run_paper(confidence=48)
        self.assertIsNone(self._buy_qty(), "confidence<55 must skip, not just shrink")


class TestKillSwitchDisablesFeature(DynamicSizingTestCase):

    def test_flag_off_ignores_low_confidence(self):
        # Same confidence=48 that skips the trade in the test above — with
        # the feature flag OFF, V3.0-A must have zero effect (V2.x
        # behavior): the trade goes through at full baseline size.
        config.DYNAMIC_POSITION_SIZING_ENABLED = False
        self._run_paper(confidence=48)
        self.assertIsNotNone(self._buy_qty(),
                              "flag OFF must fully restore V2.x behavior")


class TestPaperOnlyHardGate(DynamicSizingTestCase):

    def test_live_env_ignores_confidence_even_with_flag_on(self):
        # score_env=="live" (non-SIMULATE/JP-PAPER trd_env) must force the
        # multiplier to 1.0 regardless of config.DYNAMIC_POSITION_SIZING_ENABLED
        # -- V3.0 spec section 七: Phase A must never touch real orders.
        # parse_trd_env is stubbed here purely to dodge this codebase's
        # unrelated, pre-existing parse_trd_env(config.TRD_ENV) call-arity
        # bug (common.py::parse_trd_env takes no arguments) so this test
        # exercises only the score_env gate, not that separate bug.
        config.DYNAMIC_POSITION_SIZING_ENABLED = True
        config.TRD_ENV = "REAL"
        runner.parse_trd_env = lambda env_str: env_str

        runner.pipeline.confidence_score.gather_and_score = _stub_gather_and_score(48)
        runner.run_once(codes=["US.TEST"], confirmed=True, use_real=True,
                         auto_route=False, strategy_name="atr_breakout")

        self.assertIsNotNone(self._buy_qty(),
                              "live env must never skip/shrink on confidence")


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
