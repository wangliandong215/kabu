"""
Tests for V3.6 P5 Execution Audit: engine/order_audit.py, the record-only
hooks in engine/runner.py::_place_order() and engine/broker.py's
MoomooBroker. The real MoomooBroker.place_order() runs against a fake
`moomoo` SDK + fake trade context (never OpenD, never a real order); the
audit DB is a temp file (test_support.isolate_live_state).

Also contains the end-to-end reproduction of the known P7 gap: broker
accepts a BUY, polling then fails, MoomooBroker returns dealt_qty=0, local
state records nothing, and the NEXT pass submits the same BUY again even
though the broker already holds the position.

Run:  python -m unittest engine.test_order_audit -v
"""
import json
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd

import config
import data.fetcher as fetcher_mod
import engine.broker as broker_mod
import engine.order_audit as oa
import engine.runner as runner
import test_support
from engine.market_preflight import PreflightResult
from portfolio.broker_state import BrokerPosition, BrokerState
from portfolio.tracker import Portfolio

RET_OK, RET_ERROR = 0, -1


def _fake_ft():
    ft = types.ModuleType("moomoo")
    ft.RET_OK = RET_OK
    ft.OrderType = types.SimpleNamespace(NORMAL="NORMAL")
    ft.TrdSide = types.SimpleNamespace(BUY="BUY", SELL="SELL")
    ft.ModifyOrderOp = types.SimpleNamespace(CANCEL="CANCEL")
    ft.TrdEnv = types.SimpleNamespace(SIMULATE="SIMULATE", REAL="REAL")
    return ft


class FakeTradeCtx:
    """Scriptable stand-in for moomoo's OpenSecTradeContext.
    place: ("ok", order_id) | ("reject", msg) | ("raise", exc)
    polls: list of ("row", status, dealt, avg) | ("error",) | ("raise", exc);
           the last entry repeats.
    recheck: same shape, used for the post-cancel query."""

    def __init__(self, place=("ok", "OID1"), polls=(("row", "FILLED_ALL", 10, 101.0),),
                 recheck=None, cancel=None):
        self.place, self.polls, self.recheck, self.cancel = place, list(polls), recheck, cancel
        self.poll_i = 0
        self.cancelled = False
        self.place_calls = 0

    def place_order(self, **kw):
        self.place_calls += 1
        kind, val = self.place
        if kind == "raise":
            raise val
        if kind == "reject":
            return RET_ERROR, val
        return RET_OK, pd.DataFrame({"order_id": [val]})

    def _answer(self, spec):
        if spec[0] == "raise":
            raise spec[1]
        if spec[0] == "error":
            return RET_ERROR, "query failed"
        _, status, dealt, avg = spec
        return RET_OK, pd.DataFrame([{"order_status": status, "dealt_qty": dealt,
                                      "dealt_avg_price": avg,
                                      "create_time": "2026-09-25 10:00:00",
                                      "updated_time": "2026-09-25 10:00:01"}])

    def order_list_query(self, order_id, trd_env):
        if self.cancelled and self.recheck is not None:
            return self._answer(self.recheck)
        spec = self.polls[min(self.poll_i, len(self.polls) - 1)]
        self.poll_i += 1
        return self._answer(spec)

    def modify_order(self, **kw):
        if self.cancel is not None:
            raise self.cancel
        self.cancelled = True


class _AuditCase(unittest.TestCase):
    def setUp(self):
        self._alert = test_support.mute_alert_file_logging()
        self._orig = {"moomoo": sys.modules.get("moomoo"),
                      "make_trade_ctx": broker_mod.make_trade_ctx,
                      "safe_close": broker_mod.safe_close,
                      "sleep": broker_mod.time.sleep,
                      "get_broker": runner.get_broker,
                      "PREFLIGHT_ENABLED": config.PREFLIGHT_ENABLED,
                      "RISK_ENGINE_MODE": config.RISK_ENGINE_MODE,
                      "preflight": runner.market_preflight.check,
                      "risk_eval": runner.risk_engine_evaluate}
        sys.modules["moomoo"] = _fake_ft()
        broker_mod.safe_close = lambda ctx: None
        broker_mod.time.sleep = lambda s: None
        runner.get_broker = lambda code: broker_mod.MoomooBroker
        config.PREFLIGHT_ENABLED = False
        self.ctx = FakeTradeCtx()
        broker_mod.make_trade_ctx = lambda market=None: self.ctx
        self.run_id = f"run-{self.id()}"          # isolates rows per test in the shared DB
        oa.start_pass(run_id=self.run_id, env_label="SIMULATE")

    def tearDown(self):
        if self._orig["moomoo"] is not None:
            sys.modules["moomoo"] = self._orig["moomoo"]
        else:
            sys.modules.pop("moomoo", None)
        broker_mod.make_trade_ctx = self._orig["make_trade_ctx"]
        broker_mod.safe_close = self._orig["safe_close"]
        broker_mod.time.sleep = self._orig["sleep"]
        runner.get_broker = self._orig["get_broker"]
        config.PREFLIGHT_ENABLED = self._orig["PREFLIGHT_ENABLED"]
        config.RISK_ENGINE_MODE = self._orig["RISK_ENGINE_MODE"]
        runner.market_preflight.check = self._orig["preflight"]
        runner.risk_engine_evaluate = self._orig["risk_eval"]
        test_support.unmute_alert_file_logging(self._alert)

    def _order(self, side="BUY", qty=10, price=100.0, confirmed=True, intent="ENTRY", **kw):
        return runner._place_order(code="US.TEST", side=side, qty=qty, price=price,
                                   trd_env="SIMULATE_ENUM", env_label="SIMULATE",
                                   confirmed=confirmed, audit_intent=intent, **kw)

    def _rows(self):
        con = sqlite3.connect(str(oa._DB_PATH))
        con.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in con.execute(
                "SELECT * FROM order_audit WHERE run_id=? ORDER BY decision_at", (self.run_id,))]
        finally:
            con.close()

    def _last(self):
        return self._rows()[-1]

    def _stages(self, audit_id):
        con = sqlite3.connect(str(oa._DB_PATH))
        try:
            return [r[0] for r in con.execute(
                "SELECT stage FROM order_audit_events WHERE audit_id=? ORDER BY event_id",
                (audit_id,))]
        finally:
            con.close()


class LifecycleTest(_AuditCase):
    def test_normal_fill_full_chain_and_slippage(self):
        fill = self._order(price=100.0)
        self.assertEqual(fill, {"order_id": "OID1", "dealt_qty": 10.0,
                                "dealt_avg_price": 101.0, "status": "FILLED_ALL"})
        row = self._last()
        self.assertEqual(row["outcome"], oa.FILLED)
        self.assertEqual((row["fill_confirmed"], row["needs_reconcile"], row["partial_fill"]),
                         (1, 0, 0))
        self.assertEqual(row["order_id"], "OID1")
        self.assertEqual(row["intent"], "ENTRY")
        self.assertEqual((row["env_label"], row["trd_env"], row["broker"]),
                         ("SIMULATE", "SIMULATE_ENUM", "MoomooBroker"))
        self.assertEqual(row["run_id"], self.run_id)
        self.assertAlmostEqual(row["slippage_pct"], 0.01)       # BUY 1% above expected = worse
        self.assertAlmostEqual(row["slippage_usd"], 10.0)
        self.assertEqual(row["commission_status"], "NOT_QUERIED")
        self.assertIsNone(row["commission"])
        for col in ("decision_at", "submit_sent_at", "ack_at", "first_fill_seen_at",
                    "finished_at", "broker_updated_time"):
            self.assertTrue(row[col], col)
        self.assertEqual(self._stages(row["audit_id"]),
                         ["DECISION", "RISK_EVALUATED", "PREFLIGHT", "BROKER_SELECTED",
                          "SUBMIT_SENT", "ACK", "POLL", "FINISH"])

    def test_sell_slippage_sign(self):
        self.ctx.polls = [("row", "FILLED_ALL", 10, 99.0)]
        self._order(side="SELL", price=100.0, intent="EXIT:STOP_LOSS")
        row = self._last()
        self.assertAlmostEqual(row["slippage_pct"], 0.01)       # SELL 1% below = worse
        self.assertEqual(row["intent"], "EXIT:STOP_LOSS")

    def test_partial_fill_then_remainder_cancelled(self):
        self.ctx.polls = [("row", "SUBMITTED", 4, 100.0)]
        self.ctx.recheck = ("row", "CANCELLED_PART", 4, 100.0)
        fill = self._order()
        self.assertEqual(fill["dealt_qty"], 4.0)
        row = self._last()
        # CANCELLED_PART is not in the broker's terminal set -> not confirmed
        self.assertEqual(row["outcome"], oa.UNCONFIRMED_AFTER_ACK)
        self.ctx = FakeTradeCtx(polls=[("row", "SUBMITTED", 4, 100.0)],
                                recheck=("row", "FILL_CANCELLED", 4, 100.0))
        fill = self._order()
        row = self._last()
        self.assertEqual(row["outcome"], oa.PARTIAL_FILLED)
        self.assertEqual((row["partial_fill"], row["dealt_qty"], row["requested_qty"]),
                         (1, 4.0, 10.0))
        self.assertTrue(row["cancel_sent_at"])

    def test_unfilled_order_cancelled(self):
        self.ctx.polls = [("row", "CANCELLED_ALL", 0, 0.0)]
        self._order()
        row = self._last()
        self.assertEqual(row["outcome"], oa.NOT_FILLED)
        self.assertEqual(row["fill_confirmed"], 1)

    def test_submit_rejected_by_broker(self):
        self.ctx.place = ("reject", "insufficient buying power")
        fill = self._order()
        self.assertEqual((fill["order_id"], fill["dealt_qty"]), ("", 0.0))
        row = self._last()
        self.assertEqual(row["outcome"], oa.SUBMIT_REJECTED)
        self.assertIn("insufficient", row["submit_error"])

    def test_dry_run(self):
        fill = self._order(confirmed=False)
        self.assertEqual(fill["status"], "DRY_RUN")
        self.assertEqual(self._last()["outcome"], oa.DRY_RUN)
        self.assertEqual(self.ctx.place_calls, 0)

    def test_preflight_failed(self):
        config.PREFLIGHT_ENABLED = True
        runner.market_preflight.check = lambda code: PreflightResult(False, "market CLOSED")
        fill = self._order()
        self.assertEqual(fill["status"], "PREFLIGHT_FAILED")
        row = self._last()
        self.assertEqual((row["outcome"], row["preflight_status"], row["preflight_reason"]),
                         (oa.PREFLIGHT_FAILED, "FAIL", "market CLOSED"))
        self.assertEqual(self.ctx.place_calls, 0)

    def test_risk_blocked_in_enforce_mode(self):
        config.RISK_ENGINE_MODE = "ENFORCE"
        decision = types.SimpleNamespace(status=types.SimpleNamespace(value="BLOCK"),
                                         reason="sector cap")
        runner.risk_engine_evaluate = lambda **kw: decision
        fill = self._order(portfolio_state=object())
        self.assertEqual(fill["status"], "RISK_BLOCKED")
        row = self._last()
        self.assertEqual((row["outcome"], row["risk_status"], row["risk_reason"], row["risk_mode"]),
                         (oa.RISK_BLOCKED, "BLOCK", "sector cap", "ENFORCE"))

    def test_risk_audit_mode_is_recorded_but_does_not_block(self):
        config.RISK_ENGINE_MODE = "AUDIT"
        decision = types.SimpleNamespace(status=types.SimpleNamespace(value="BLOCK"), reason="x")
        runner.risk_engine_evaluate = lambda **kw: decision
        fill = self._order(portfolio_state=object())
        self.assertEqual(fill["status"], "FILLED_ALL")
        row = self._last()
        self.assertEqual((row["risk_status"], row["risk_mode"], row["outcome"]),
                         ("BLOCK", "AUDIT", oa.FILLED))

    def test_pass_id_shared_within_pass(self):
        self._order()
        self._order()
        oa.start_pass(run_id=self.run_id, env_label="SIMULATE")
        self._order()
        pass_ids = [r["pass_id"] for r in self._rows()]
        self.assertEqual(pass_ids[0], pass_ids[1])
        self.assertNotEqual(pass_ids[1], pass_ids[2])


class UnconfirmedStateTest(_AuditCase):
    """The CRITICAL case: broker has the order, local code can't tell."""

    def test_poll_exception_after_ack_returns_zero_and_is_flagged(self):
        self.ctx.polls = [("raise", ConnectionError("OpenD dropped"))]
        fill = self._order()
        # Current (unfixed) behavior the P7 fix must address: the broker
        # accepted OID1 but the caller is told nothing filled.
        self.assertEqual((fill["order_id"], fill["dealt_qty"]), ("OID1", 0.0))
        row = self._last()
        self.assertEqual(row["outcome"], oa.UNCONFIRMED_AFTER_ACK)
        self.assertEqual((row["fill_confirmed"], row["needs_reconcile"]), (0, 1))
        self.assertIn("OpenD dropped", row["exception"])
        self.assertIn("EXCEPTION", self._stages(row["audit_id"]))

    def test_all_polls_fail_and_cancel_fails(self):
        self.ctx.polls = [("error",)]
        self.ctx.recheck = ("error",)
        self.ctx.cancel = RuntimeError("cancel refused")
        fill = self._order()
        self.assertEqual(fill["dealt_qty"], 0.0)
        row = self._last()
        self.assertEqual(row["outcome"], oa.UNCONFIRMED_AFTER_ACK)
        self.assertEqual(row["poll_failures"], 7)            # 6 polls + post-cancel recheck
        self.assertIn("cancel refused", row["cancel_error"])

    def test_place_order_raises_outcome_unknown(self):
        self.ctx.place = ("raise", TimeoutError("response lost"))
        fill = self._order()
        self.assertEqual(fill["dealt_qty"], 0.0)
        row = self._last()
        self.assertEqual(row["outcome"], oa.SUBMIT_OUTCOME_UNKNOWN)
        self.assertEqual(row["needs_reconcile"], 1)

    def test_exception_before_submit_is_recorded_and_reraised(self):
        def _boom(market=None):
            raise OSError("OpenD port refused")
        broker_mod.make_trade_ctx = _boom
        with self.assertRaises(OSError):
            self._order()
        row = self._last()
        self.assertEqual(row["outcome"], oa.NOT_SUBMITTED_ERROR)
        self.assertIn("OpenD port refused", row["exception"])


class AuditNeverBlocksTest(_AuditCase):
    def test_audit_db_unwritable_order_flow_identical(self):
        sentinel = {"order_id": "X", "dealt_qty": 3.0, "dealt_avg_price": 9.0,
                    "status": "FILLED_ALL"}

        class _Broker:
            @staticmethod
            def place_order(code, side, qty, price, trd_env, env_label, confirmed):
                return sentinel

        runner.get_broker = lambda code: _Broker
        orig = oa._DB_PATH
        with tempfile.TemporaryDirectory() as d:
            oa._DB_PATH = Path(d)          # a directory: every sqlite open fails
            try:
                result = self._order()
            finally:
                oa._DB_PATH = orig
        self.assertIs(result, sentinel)
        self.assertEqual(sentinel, {"order_id": "X", "dealt_qty": 3.0,
                                    "dealt_avg_price": 9.0, "status": "FILLED_ALL"})

    def test_fake_broker_without_events_still_classified(self):
        class _Paper:
            @staticmethod
            def place_order(code, side, qty, price, trd_env, env_label, confirmed):
                return {"order_id": "PAPER-1", "dealt_qty": float(qty),
                        "dealt_avg_price": price, "status": "FILLED_ALL"}
        runner.get_broker = lambda code: _Paper
        self._order()
        row = self._last()
        self.assertEqual((row["outcome"], row["broker"]), (oa.FILLED, "_Paper"))

    def test_audit_calls_without_context_are_noops(self):
        oa._current_order.set(None)
        oa.event("POLL", status="FILLED_ALL")
        oa.stage("PREFLIGHT", preflight_status="PASS")
        oa.finish({"status": "FILLED_ALL"})

    def test_terminal_set_matches_broker(self):
        self.assertEqual(set(oa.TERMINAL_STATUSES), set(broker_mod.MoomooBroker._TERMINAL))


# ── end-to-end: the duplicate-BUY gap (P7) ────────────────────────────────────

class _FakeTracker:
    def __init__(self, *a, **kw): pass
    def __getattr__(self, name):
        return lambda *a, **kw: None


class DuplicateBuyReproductionTest(_AuditCase):
    """Two real run_once() passes, real _place_order + real MoomooBroker
    against a fake SDK. Pass 1: broker accepts the BUY, polling raises ->
    dealt_qty=0 returned, local portfolio records nothing. Pass 2: the broker
    now reports the position (it did fill), local still doesn't -> the same
    BUY is submitted again. Documents the CURRENT behavior; when P7 lands,
    this test must be flipped to assert exactly one submission."""

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.pf_path = Path(self._tmp.name) / "positions.json"
        self._r = {name: getattr(runner, name) for name in (
            "Portfolio", "filter_open", "scan", "smart_scan", "TradeTracker",
            "regime_evaluate_and_log")}
        self._r2 = {
            "check_max_drawdown": runner.guard.check_max_drawdown,
            "macro": runner.news_sentiment.macro_circuit_breaker,
            "qqq_halt": runner.regime.qqq_macro_halt,
            "weather": runner.market_weather.market_weather,
            "trade_sell": runner.alert.trade_sell, "trade_buy": runner.alert.trade_buy,
            "fetch_kline": fetcher_mod.fetch_kline,
            "fetch_broker_state": runner.broker_state_mod.fetch_broker_state,
            "fund": runner.pipeline.fundamental.score,
            "news_cls": runner.pipeline.news_filter.classify_code,
            "blackout": runner.pipeline.is_earnings_blackout,
            "earn": runner.event_risk.has_earnings_risk,
            "snapshot": runner.research_snapshot.build_trade_research_snapshot,
        }
        runner.regime_evaluate_and_log = lambda *a, **kw: None
        runner.Portfolio = lambda: Portfolio(path=self.pf_path)
        runner.filter_open = lambda codes: list(codes)
        runner.TradeTracker = _FakeTracker
        runner.scan = lambda codes, **kw: {"US.TEST": {
            "current_price": 100.0, "signal": "BUY", "signal_strength": 0.9,
            "strategy_used": "atr_breakout"}}
        runner.smart_scan = runner.scan
        runner.guard.check_max_drawdown = lambda p: True
        runner.news_sentiment.macro_circuit_breaker = lambda: ""
        runner.regime.qqq_macro_halt = lambda df: False
        runner.market_weather.market_weather = lambda df: 1
        runner.alert.trade_sell = lambda *a, **kw: None
        runner.alert.trade_buy = lambda *a, **kw: None
        fetcher_mod.fetch_kline = lambda *a, **kw: None
        runner.pipeline.fundamental.score = lambda code, env: {
            "score": None, "tier": None, "reason": "test_stub"}
        runner.pipeline.news_filter.classify_code = lambda code: {
            "score": None, "tier": 3, "matched_keyword": None}
        runner.pipeline.is_earnings_blackout = lambda code: False
        runner.event_risk.has_earnings_risk = lambda code, trade_date=None: False
        runner.research_snapshot.build_trade_research_snapshot = lambda *a, **kw: {}
        self.broker_positions = {}
        runner.broker_state_mod.fetch_broker_state = lambda trd_env: BrokerState(
            positions=dict(self.broker_positions), cash=1_000_000.0,
            total_assets=1_000_000.0 + 0.5, long_mv=0.0)

    def tearDown(self):
        for name, v in self._r.items():
            setattr(runner, name, v)
        runner.guard.check_max_drawdown = self._r2["check_max_drawdown"]
        runner.news_sentiment.macro_circuit_breaker = self._r2["macro"]
        runner.regime.qqq_macro_halt = self._r2["qqq_halt"]
        runner.market_weather.market_weather = self._r2["weather"]
        runner.alert.trade_sell = self._r2["trade_sell"]
        runner.alert.trade_buy = self._r2["trade_buy"]
        fetcher_mod.fetch_kline = self._r2["fetch_kline"]
        runner.broker_state_mod.fetch_broker_state = self._r2["fetch_broker_state"]
        runner.pipeline.fundamental.score = self._r2["fund"]
        runner.pipeline.news_filter.classify_code = self._r2["news_cls"]
        runner.pipeline.is_earnings_blackout = self._r2["blackout"]
        runner.event_risk.has_earnings_risk = self._r2["earn"]
        runner.research_snapshot.build_trade_research_snapshot = self._r2["snapshot"]
        self._tmp.cleanup()
        super().tearDown()

    def test_known_p7_gap_duplicate_buy_after_unconfirmed_fill(self):
        Portfolio(path=self.pf_path)._save()
        self.ctx.polls = [("raise", ConnectionError("OpenD dropped after ACK"))]

        runner.run_once(codes=["US.TEST"], confirmed=True, auto_route=False,
                        strategy_name="atr_breakout", run_id=self.run_id)
        first_qty = None
        rows = [r for r in self._rows() if r["intent"] == "ENTRY"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome"], oa.UNCONFIRMED_AFTER_ACK)
        first_qty = rows[0]["requested_qty"]
        self.assertNotIn("US.TEST", Portfolio(path=self.pf_path).data["positions"])

        # Between passes the broker actually filled the order.
        self.broker_positions = {"US.TEST": BrokerPosition(
            code="US.TEST", qty=first_qty, cost_price=100.0,
            market_val=100.0 * first_qty, current_price=100.0)}
        self.ctx = FakeTradeCtx(place=("ok", "OID2"),
                                polls=[("row", "FILLED_ALL", first_qty, 100.0)])

        runner.run_once(codes=["US.TEST"], confirmed=True, auto_route=False,
                        strategy_name="atr_breakout", run_id=self.run_id)
        rows = [r for r in self._rows() if r["intent"] == "ENTRY"]
        # KNOWN GAP (P7): a second BUY reaches the broker although the broker
        # already holds US.TEST — local-only "already held" check.
        self.assertEqual(len(rows), 2)
        self.assertEqual(self.ctx.place_calls, 1)
        self.assertEqual(rows[1]["outcome"], oa.FILLED)


def setUpModule():
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
