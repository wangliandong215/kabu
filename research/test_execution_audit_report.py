"""
Tests for research/execution_audit_report.py (read-only order lifecycle
report). Temp DBs/logs only.

Run:  python -m unittest research.test_execution_audit_report -v
"""
import ast
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import engine.order_audit as oa
from engine.trade_tracker import TradeTracker
from research import execution_audit_report as rep
import test_support

_REPO_ROOT = Path(__file__).resolve().parent.parent


class LifecycleReportTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = Path(self._tmp.name)
        self.audit_db, self.trade_db = d / "order_audit.db", d / "trade_history.db"
        self.risk_log, self.diff_log = d / "risk_state_log.jsonl", d / "reconciliation_diffs.jsonl"
        con = sqlite3.connect(self.audit_db)
        con.executescript(oa._SCHEMA)
        con.commit()
        con.close()
        self.tracker = TradeTracker(self.trade_db)

    def tearDown(self):
        self.tracker.close()
        self._tmp.cleanup()

    def _order(self, audit_id, code="US.AAA", side="BUY", intent="ENTRY", outcome="FILLED",
               qty=10, dealt=10, at="2026-09-25T22:36:00+09:00", fin="2026-09-25T22:36:08+09:00",
               confirmed=1):
        con = sqlite3.connect(self.audit_db)
        con.execute(
            "INSERT INTO order_audit (audit_id, decision_at, finished_at, code, side, intent, "
            "requested_qty, dealt_qty, outcome, confirmed, env_label, slippage_pct, "
            "needs_reconcile) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (audit_id, at, fin, code, side, intent, qty, dealt, outcome, confirmed, "SIMULATE",
             0.001 if outcome == "FILLED" else None,
             1 if outcome.startswith(("UNCONFIRMED", "SUBMIT_OUTCOME")) else 0))
        con.commit()
        con.close()

    def _open_trade(self, code="US.AAA", shares=10, ts="2026-09-25T22:36:09"):
        self.tracker.log_entry(trade_id=f"{code}_{ts}", ticker=code, strategy_name="atr_breakout",
                               strategy_version="2.8", direction="LONG", price=100.0,
                               shares=shares, position_value=1000.0, position_pct=0.1,
                               cash=1.0, equity=1.0, timestamp=ts)

    def _logs(self, snapshots=(), diffs=()):
        self.risk_log.write_text("".join(json.dumps({"timestamp": t, "total_assets": 1.5}) + "\n"
                                         for t in snapshots), encoding="utf-8")
        self.diff_log.write_text("".join(json.dumps(x) + "\n" for x in diffs), encoding="utf-8")

    def _build(self):
        return rep.build_lifecycle(self.audit_db, self.trade_db, self.risk_log,
                                   self.diff_log).set_index("audit_id")

    def test_local_record_classification(self):
        self._order("matched")
        self._open_trade()
        self._order("qty", code="US.BBB")
        self._open_trade(code="US.BBB", shares=7)
        self._order("missing", code="US.CCC")
        self._order("unknown", code="US.DDD", outcome="UNCONFIRMED_AFTER_ACK", dealt=0)
        self._order("pyramid", intent="PYRAMID_ADD")
        self._order("nofill", code="US.EEE", outcome="NOT_FILLED", dealt=0)
        self._order("dry", code="US.FFF", outcome="DRY_RUN", dealt=0, confirmed=0)
        self._logs()
        lc = self._build()
        self.assertEqual(lc.loc["matched", "local_record"], "MATCHED")
        self.assertEqual(lc.loc["qty", "local_record"], "QTY_MISMATCH")
        self.assertEqual(lc.loc["missing", "local_record"], "MISSING")
        self.assertEqual(lc.loc["unknown", "local_record"], "UNKNOWN_FILL")
        self.assertEqual(lc.loc["pyramid", "local_record"], "NOT_TRACKED")
        self.assertEqual(lc.loc["nofill", "local_record"], "NO_FILL")
        self.assertEqual(lc.loc["dry", "reconcile"], "NOT_APPLICABLE")

    def test_exit_matches_on_exit_time(self):
        self._open_trade(ts="2026-09-20T22:00:00")
        self.tracker.log_exit(trade_id="US.AAA_2026-09-20T22:00:00", price=99.0, cash=1.0,
                              equity=1.0, timestamp="2026-09-25T22:36:09", exit_reason="STOP_LOSS")
        self._order("exit", side="SELL", intent="EXIT:STOP_LOSS")
        self._logs()
        self.assertEqual(self._build().loc["exit", "local_record"], "MATCHED")

    def test_reconcile_next_pass(self):
        self._order("ok")
        self._order("bad", code="US.BBB", at="2026-09-25T22:37:00+09:00",
                    fin="2026-09-25T22:37:05+09:00")
        self._order("late", code="US.CCC", at="2026-09-25T23:59:00+09:00",
                    fin="2026-09-25T23:59:05+09:00")
        self._logs(snapshots=["2026-09-25T22:45:10"],
                   diffs=[{"timestamp": "2026-09-25T22:45:05", "code": "US.BBB",
                           "broker_qty": 10, "tracker_qty": 0, "diff_qty": 10}])
        lc = self._build()
        self.assertEqual(lc.loc["ok", "reconcile"], "CONSISTENT")
        self.assertEqual(lc.loc["bad", "reconcile"], "MISMATCH")
        self.assertEqual(lc.loc["bad", "reconcile_diff_qty"], 10)
        self.assertEqual(lc.loc["late", "reconcile"], "NOT_YET_RECONCILED")

    def test_summary_and_markdown(self):
        self._order("a")
        self._order("b", outcome="UNCONFIRMED_AFTER_ACK", dealt=0)
        self._logs()
        lc = rep.build_lifecycle(self.audit_db, self.trade_db, self.risk_log, self.diff_log)
        s = rep.summarize(lc)
        self.assertEqual((s["orders"], s["needs_reconcile"], s["commission_recorded"]), (2, 1, 0))
        self.assertIn("UNCONFIRMED_AFTER_ACK", rep.render_markdown(lc))

    def test_missing_sources_are_safe(self):
        d = Path(self._tmp.name)
        self.assertTrue(rep.build_lifecycle(d / "none.db", d / "none2.db",
                                            d / "x.jsonl", d / "y.jsonl").empty)
        self.assertEqual(rep.summarize(rep.build_lifecycle(d / "none.db")), {"orders": 0})
        rep.render_markdown(rep.build_lifecycle(d / "none.db"))
        self._order("solo")
        lc = rep.build_lifecycle(self.audit_db, d / "none2.db", d / "x.jsonl", d / "y.jsonl")
        self.assertEqual(lc.iloc[0]["local_record"], "MISSING")
        self.assertEqual(lc.iloc[0]["reconcile"], "NOT_YET_RECONCILED")

    def test_report_is_read_only_and_imports_no_trading_code(self):
        tree = ast.parse((_REPO_ROOT / "research" / "execution_audit_report.py")
                         .read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for n in names:
                for bad in ("engine", "risk", "strategies", "portfolio", "position_manager",
                            "exit_engine", "ai_decision"):
                    self.assertFalse(n == bad or n.startswith(bad + "."), n)


def setUpModule():
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
