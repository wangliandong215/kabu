"""
Unit tests for engine/runner.py's v2.10 Portfolio Risk Manager integration —
the new step (right after the exit-check loop) that fetches real broker
state, reconciles it against tracker.json, classifies an exposure tier, folds
PAUSE/WARNING/EMERGENCY into the existing macro_block circuit breaker, and
executes an emergency rebalance plan when exposure > 105%.

Same hermetic monkeypatch harness as engine/test_runner_pyramid_gate.py /
test_runner_drawdown_halt.py / test_runner_entry_quality_hook.py — every
external dependency run_once() calls is patched, including (unlike those
sibling files) portfolio.broker_state.fetch_broker_state itself, since
that's exactly what's under test here.

Run:  python -m unittest engine.test_portfolio_risk_manager_gate -v
"""
import tempfile
import unittest
from pathlib import Path

import config
import data.fetcher as fetcher_mod
import engine.runner as runner
import test_support
from portfolio.broker_state import BrokerPosition, BrokerState
from portfolio.tracker import Portfolio
from risk import portfolio_risk_manager as prm


class _FakeTracker:
    def __init__(self, *a, **kw): pass
    def update_position_metrics(self, *a, **kw): pass
    def log_exit(self, *a, **kw): pass
    def log_entry(self, *a, **kw): pass
    def log_market_context(self, *a, **kw): pass


def _bpos(code, qty, cost_price, market_val, current_price):
    return BrokerPosition(code=code, qty=qty, cost_price=cost_price,
                           market_val=market_val, current_price=current_price)


def _mutate_broker_state(broker_state, code, side, dealt_qty, price):
    """Test helper mimicking what a real fill does to broker state: update
    the position's qty/market_val AND the account-level long_mv/cash that
    evaluate()/plan_rebalance() actually read (BrokerState.long_mv is a
    separate field, not derived from summing .positions — a mock that only
    mutates the position and forgets long_mv leaves exposure_pct frozen
    forever, which is exactly the bug this helper exists to avoid repeating
    in every test)."""
    pos = broker_state.positions[code]
    sign = 1 if side == "SELL" else -1
    pos.qty -= sign * dealt_qty
    pos.market_val = pos.qty * pos.current_price
    trade_value = dealt_qty * price
    broker_state.long_mv -= sign * trade_value
    broker_state.cash += sign * trade_value


class PortfolioRiskManagerGateTestCase(unittest.TestCase):

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
            "fundamental_score": runner.pipeline.fundamental.score,
            "classify_code": runner.pipeline.news_filter.classify_code,
            "is_earnings_blackout": runner.pipeline.is_earnings_blackout,
            "_qqq_above_ma": runner._qqq_above_ma,
            "AUTO_EXECUTE": config.PORTFOLIO_RISK_EMERGENCY_AUTO_EXECUTE,
            "has_earnings_risk": runner.event_risk.has_earnings_risk,
            "build_trade_research_snapshot": runner.research_snapshot.build_trade_research_snapshot,
        }
        # _qqq_above_ma() hits live quotes (OpenQuoteContext) to check QQQ vs
        # its MA200 -- stub it True (no MA200-break exit in flight) so these
        # tests isolate the Portfolio Risk Manager path from the pre-existing,
        # separately-tested MA200 "死仓" exit rule.
        runner._qqq_above_ma = lambda: True
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

        def _mock_place_order(code, side, qty, price, trd_env, env_label, confirmed):
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
        runner.pipeline.fundamental.score = self._orig["fundamental_score"]
        runner.pipeline.news_filter.classify_code = self._orig["classify_code"]
        runner.pipeline.is_earnings_blackout = self._orig["is_earnings_blackout"]
        runner._qqq_above_ma = self._orig["_qqq_above_ma"]
        runner.event_risk.has_earnings_risk = self._orig["has_earnings_risk"]
        runner.research_snapshot.build_trade_research_snapshot = self._orig["build_trade_research_snapshot"]
        config.PORTFOLIO_RISK_EMERGENCY_AUTO_EXECUTE = self._orig["AUTO_EXECUTE"]
        prm._LOCK_PATH.unlink(missing_ok=True)
        self._tmpdir.cleanup()
        test_support.unmute_alert_file_logging(self._alert_handlers)

    def _seed_fresh_buy_candidate(self):
        runner.scan = lambda codes, **kw: {
            "US.TEST": {"current_price": 100.0, "signal": "BUY",
                        "signal_strength": 0.9, "strategy_used": "atr_breakout"},
        }
        runner.smart_scan = runner.scan

    def test_normal_tier_does_not_block_new_buy(self):
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: BrokerState(
            positions={}, cash=1_000_000.0, total_assets=1_000_000.0, long_mv=0.0)
        Portfolio(path=self.portfolio_path)._save()
        self._seed_fresh_buy_candidate()

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        # Filtered to US.TEST: NORMAL tier also leaves the pre-existing QQQ
        # core top-up step unblocked (expected — it's not gated by this
        # feature at all), which would otherwise confound this assertion.
        buys = [o for o in self.placed_orders if o["side"] == "BUY" and o["code"] == "US.TEST"]
        self.assertEqual(len(buys), 1, "NORMAL tier must not block a fresh BUY")

    def test_pause_tier_blocks_new_buy_without_selling(self):
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: BrokerState(
            positions={}, cash=2_000.0, total_assets=1_000_000.0, long_mv=980_000.0)  # 98%
        Portfolio(path=self.portfolio_path)._save()
        self._seed_fresh_buy_candidate()

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        self.assertEqual(len(self.placed_orders), 0,
                          "PAUSE tier (95%-100%) must block new buys and not force-sell anything")

    def _seed_qqq_only_emergency(self):
        """Shared fixture for the EMERGENCY-tier tests below: QQQ-only
        portfolio at ~122% exposure (QQQ trim alone covers the shortfall,
        so exactly one rebalance leg is expected under normal conditions).
        Installs a _place_order mock that mutates `broker_state` in place on
        a confirmed fill (mirrors a real broker) so the iterative re-fetch
        loop actually converges instead of seeing a frozen snapshot forever
        -- tests that need different fill behavior (e.g. a partial fill)
        should override runner._place_order again afterward."""
        qqq = config.QQQ_CORE_CODE
        broker_state = BrokerState(
            positions={qqq: _bpos(qqq, 473, 716.0, 342_087.79, 723.23)},
            cash=-50_000.0, total_assets=280_000.0, long_mv=342_087.79,  # ~122%
        )
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: broker_state

        def _mock_place_order_mutating(code, side, qty, price, trd_env, env_label, confirmed):
            self.placed_orders.append({"code": code, "side": side, "qty": qty, "price": price})
            dealt = float(qty) if confirmed else 0.0
            if confirmed and dealt > 0 and code in broker_state.positions:
                _mutate_broker_state(broker_state, code, side, dealt, price)
            return {"order_id": "FAKE123" if confirmed else "", "dealt_qty": dealt,
                    "dealt_avg_price": price if confirmed else 0.0,
                    "status": "FILLED_ALL" if confirmed else "DRY_RUN"}
        runner._place_order = _mock_place_order_mutating

        portfolio = Portfolio(path=self.portfolio_path)
        portfolio.data["positions"][qqq] = {
            "side": "BUY", "entry_price": 716.0, "avg_cost": 716.0, "qty": 473,
            "signal_strength": 1.0, "strategy": "core_etf", "entry_atr": 0.0,
            "atr_mult": config.ATR_MULT_BASE, "breakeven_locked": False,
            "trail_stop": None, "entry_time": "2020-01-01T00:00:00",
            "score_label": None, "total_score": None,
        }
        portfolio._save()
        runner.scan = lambda codes, **kw: {}
        runner.smart_scan = runner.scan
        return qqq, broker_state

    def test_emergency_tier_stays_preview_by_default_even_when_confirmed(self):
        """v2.10.1: EMERGENCY rebalance must NOT place real orders in the
        automatic loop until config.PORTFOLIO_RISK_EMERGENCY_AUTO_EXECUTE is
        explicitly turned on — even a fully confirmed=True pass stays
        preview-only for rebalance legs. Every other order type is
        unaffected (covered by the other tests in this file)."""
        self.assertFalse(config.PORTFOLIO_RISK_EMERGENCY_AUTO_EXECUTE,
                         "default must be False for this test to be meaningful")
        qqq, _ = self._seed_qqq_only_emergency()

        runner.run_once(codes=[qqq], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        sells = [o for o in self.placed_orders if o["side"] == "SELL" and o["code"] == qqq]
        self.assertEqual(len(sells), 1, "the plan should still be previewed (logged as a would-be order)")

        reloaded = Portfolio(path=self.portfolio_path)
        self.assertEqual(reloaded.data["positions"][qqq]["qty"], 473,
                         "default (auto-execute off) must never actually touch tracker state or place a real order")

    def test_emergency_tier_executes_rebalance_sell_when_auto_execute_enabled(self):
        config.PORTFOLIO_RISK_EMERGENCY_AUTO_EXECUTE = True
        qqq, _ = self._seed_qqq_only_emergency()

        runner.run_once(codes=[qqq], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        sells = [o for o in self.placed_orders if o["side"] == "SELL" and o["code"] == qqq]
        self.assertEqual(len(sells), 1, "EMERGENCY tier (>105%) must trigger a QQQ excess-trim sell")
        self.assertLess(sells[0]["qty"], 473, "rebalance must trim, not fully liquidate, the QQQ core position")

        reloaded = Portfolio(path=self.portfolio_path)
        self.assertIn(qqq, reloaded.data["positions"], "QQQ core position must survive a partial rebalance trim")
        self.assertEqual(reloaded.data["positions"][qqq]["qty"], 473 - sells[0]["qty"])

    def test_partial_fill_is_re_planned_from_fresh_broker_state(self):
        """The first SELL only partially fills -- the iterative loop must
        re-fetch broker state and re-derive the remaining amount to sell
        from the ACTUAL post-fill position, not from the original static
        plan (this is exactly the Q6/Q7 gap the 2026-08-29 rebalance preview
        report identified in the original v2.10 batch-execute design)."""
        config.PORTFOLIO_RISK_EMERGENCY_AUTO_EXECUTE = True
        qqq, broker_state = self._seed_qqq_only_emergency()

        fill_calls = {"n": 0}

        def _partial_then_full(code, side, qty, price, trd_env, env_label, confirmed):
            fill_calls["n"] += 1
            self.placed_orders.append({"code": code, "side": side, "qty": qty, "price": price})
            if fill_calls["n"] == 1:
                dealt = qty // 2  # first order only half-fills
            else:
                dealt = qty
            if confirmed and dealt > 0:
                _mutate_broker_state(broker_state, code, side, dealt, price)
            return {"order_id": "FAKE", "dealt_qty": float(dealt),
                    "dealt_avg_price": price, "status": "FILLED_ALL"}
        runner._place_order = _partial_then_full

        runner.run_once(codes=[qqq], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        sells = [o for o in self.placed_orders if o["side"] == "SELL" and o["code"] == qqq]
        self.assertEqual(len(sells), 2,
                         "a half-filled first leg must trigger a second, smaller leg for the remainder")
        self.assertLess(sells[1]["qty"], sells[0]["qty"],
                        "the second leg should only cover what's still needed, not repeat the original plan")

        reloaded = Portfolio(path=self.portfolio_path)
        total_sold = (473 - reloaded.data["positions"][qqq]["qty"])
        self.assertEqual(total_sold, sells[0]["qty"] // 2 + sells[1]["qty"],
                         "tracker qty must reflect the real cumulative dealt_qty across both legs")

    def test_rebalance_skipped_when_lock_already_held(self):
        """Re-entrancy guard: if a rebalance execution is already marked in
        progress (lock file present and fresh), a new EMERGENCY pass must
        not place any real order — avoids two overlapping executions (e.g.
        the automatic loop and a manually-run execute_emergency_rebalance.py)."""
        config.PORTFOLIO_RISK_EMERGENCY_AUTO_EXECUTE = True
        qqq, _ = self._seed_qqq_only_emergency()
        self.assertEqual(prm.try_acquire_rebalance_lock(), prm.LOCK_ACQUIRED)

        runner.run_once(codes=[qqq], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        sells = [o for o in self.placed_orders if o["side"] == "SELL" and o["code"] == qqq]
        self.assertEqual(len(sells), 0, "a held lock must block execution before any order is placed")

        reloaded = Portfolio(path=self.portfolio_path)
        self.assertEqual(reloaded.data["positions"][qqq]["qty"], 473, "tracker state must be untouched")

    def test_broker_fetch_failure_blocks_buys_without_blind_selling(self):
        def _raise(trd_env):
            raise RuntimeError("OpenD unreachable")
        runner.broker_state_mod.fetch_broker_state = _raise
        Portfolio(path=self.portfolio_path)._save()
        self._seed_fresh_buy_candidate()

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        self.assertEqual(len(self.placed_orders), 0,
                          "a broker data fetch failure must block new buys and never trigger a blind sell")

    def test_cash_guard_caps_buy_qty_to_real_available_cash(self):
        # Real broker cash only covers 5 shares at $100, but sizing.calculate()
        # would otherwise size a much larger position off tracker's own
        # (unreconciled) capital figure.
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: BrokerState(
            positions={}, cash=500.0, total_assets=1_000_000.0, long_mv=0.0)
        Portfolio(path=self.portfolio_path)._save()
        self._seed_fresh_buy_candidate()

        runner.run_once(codes=["US.TEST"], confirmed=True,
                         auto_route=False, strategy_name="atr_breakout")

        buys = [o for o in self.placed_orders if o["side"] == "BUY"]
        self.assertEqual(len(buys), 1)
        self.assertLessEqual(buys[0]["qty"] * buys[0]["price"], 500.0,
                              "buy order must never exceed real broker cash")


if __name__ == "__main__":
    unittest.main()
