"""
Tests for the V3.6 P7 duplicate-BUY guard in trade.py (manual CLI that places
orders directly, not via engine/runner._place_order). Fake `moomoo` SDK +
fake trade context (never OpenD, never a real order), temp order_audit DB,
trade._audit captured (the real trade_audit.jsonl is never written).

Run:  python -m unittest test_trade_guard -v
"""
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

import common
import engine.order_audit as oa
import test_support
import trade


class _Ctx:
    def __init__(self):
        self.place_kwargs = []

    def place_order(self, **kw):
        self.place_kwargs.append(kw)
        return 0, pd.DataFrame({"order_id": ["OID9"]})


class TradeCliGuardTest(unittest.TestCase):
    def setUp(self):
        self._alert = test_support.mute_alert_file_logging()
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._orig = {"db": oa._DB_PATH, "mem": oa._unresolved_mem,
                      "moomoo": sys.modules.get("moomoo"),
                      "mtc": common.make_trade_ctx, "close": common.safe_close,
                      "audit": trade._audit, "strat": trade.get_strategy,
                      "kline": trade.fetch_kline, "price": trade.get_price}
        oa._DB_PATH = Path(self._tmp.name) / "order_audit.db"
        oa._unresolved_mem = {}
        ft = types.ModuleType("moomoo")
        ft.RET_OK = 0
        ft.TrdSide = types.SimpleNamespace(BUY="BUY", SELL="SELL")
        ft.OrderType = types.SimpleNamespace(NORMAL="NORMAL", MARKET="MARKET")
        ft.TrdEnv = types.SimpleNamespace(SIMULATE="SIMULATE", REAL="REAL")
        sys.modules["moomoo"] = ft
        self.ctx = _Ctx()
        common.make_trade_ctx = lambda market=None: self.ctx
        common.safe_close = lambda ctx: None
        self.audits = []
        trade._audit = self.audits.append
        strat = types.SimpleNamespace(full_result=lambda df: {
            "signal": "HOLD", "signal_strength": 0.5, "buy_votes": 0, "sell_votes": 0})
        trade.get_strategy = lambda name: strat
        trade.fetch_kline = lambda *a, **kw: pd.DataFrame({"close": [100.0]})
        trade.get_price = lambda code: 100.0

    def tearDown(self):
        oa._DB_PATH, oa._unresolved_mem = self._orig["db"], self._orig["mem"]
        if self._orig["moomoo"] is not None:
            sys.modules["moomoo"] = self._orig["moomoo"]
        else:
            sys.modules.pop("moomoo", None)
        common.make_trade_ctx, common.safe_close = self._orig["mtc"], self._orig["close"]
        trade._audit, trade.get_strategy = self._orig["audit"], self._orig["strat"]
        trade.fetch_kline, trade.get_price = self._orig["kline"], self._orig["price"]
        self._tmp.cleanup()
        test_support.unmute_alert_file_logging(self._alert)

    def _unresolved_buy(self, code="US.TEST", env_label="SIMULATE"):
        """An earlier BUY whose submit outcome is unknown (P7-protected)."""
        oa.start_pass(run_id="t", env_label=env_label)
        oa.begin(code, "BUY", 5, 100.0, env_label, env_label, True, intent="ENTRY")
        oa.event("SUBMIT_SENT", price=100.0, qty=5, side="BUY")
        oa.finish(exc=TimeoutError("response lost"))

    def _run(self, code="US.TEST", side="BUY", real=False, confirmed=True, output_json=False):
        out = io.StringIO()
        with redirect_stdout(out):
            trade.run_trade(code, side=side, qty=5, use_real=real, confirmed=confirmed,
                            output_json=output_json)
        return out.getvalue()

    def test_blocked_buy_not_placed(self):
        self._unresolved_buy()
        text = self._run()
        self.assertEqual(self.ctx.place_kwargs, [])
        self.assertIn("EXECUTION_GUARD_BLOCKED", text)
        self.assertIn("UNRESOLVED_ORDER", text)
        self.assertEqual(self.audits[-1]["result"], "EXECUTION_GUARD_BLOCKED")
        self.assertIn(oa.SUBMIT_OUTCOME_UNKNOWN, self.audits[-1]["guard_reason"])

    def test_blocked_buy_json_output(self):
        self._unresolved_buy()
        out = json.loads(self._run(output_json=True))
        self.assertEqual(out["status"], "EXECUTION_GUARD_BLOCKED")
        self.assertIn("UNRESOLVED_ORDER", out["guard_reason"])
        self.assertEqual(self.ctx.place_kwargs, [])

    def test_blocked_real_buy(self):
        self._unresolved_buy(env_label="REAL")
        self._run(real=True)
        self.assertEqual(self.ctx.place_kwargs, [])

    def test_clean_buy_placed_unchanged(self):
        text = self._run()
        self.assertEqual(self.ctx.place_kwargs, [{
            "price": 100.0, "qty": 5, "code": "US.TEST", "trd_side": "BUY",
            "order_type": "NORMAL", "trd_env": "SIMULATE"}])
        self.assertIn("Order submitted successfully", text)
        self.assertEqual(self.audits[-1]["result"], "success")

    def test_other_code_and_other_env_not_blocked(self):
        self._unresolved_buy(code="US.OTHER")
        self._unresolved_buy(code="US.TEST", env_label="REAL")
        self._run()                                     # US.TEST, SIMULATE
        self.assertEqual(len(self.ctx.place_kwargs), 1)

    def test_sell_never_guarded(self):
        self._unresolved_buy()
        self._run(side="SELL")
        self.assertEqual(len(self.ctx.place_kwargs), 1)
        self.assertEqual(self.ctx.place_kwargs[0]["trd_side"], "SELL")

    def test_dry_run_preview_unchanged(self):
        self._unresolved_buy()
        text = self._run(confirmed=False)
        self.assertIn("Dry run", text)
        self.assertNotIn("EXECUTION_GUARD_BLOCKED", text)
        self.assertEqual(self.ctx.place_kwargs, [])


def setUpModule():
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
