"""
Tests for V3.6 P7 Execution Safety: engine/execution_guard.py (duplicate-BUY
protection + reconciliation), the CANCELLED_PART terminal fix in
engine/broker.py, and the guard hook in engine/runner.py::_place_order().

Real _place_order + real MoomooBroker against the fake `moomoo` SDK / fake
trade context from engine/test_order_audit.py (never OpenD, never a real
order); a fresh temp audit DB per test. The run_once() cases use the same
harness as the end-to-end duplicate-BUY test there.

Run:  python -m unittest engine.test_execution_guard -v
"""
import io
import unittest
from contextlib import redirect_stdout

import engine.execution_guard as eg
import engine.order_audit as oa
import engine.runner as runner
import engine.test_order_audit as base
import test_support
from portfolio.tracker import Portfolio
from risk.portfolio_risk_manager import ReconciliationDiff

FakeTradeCtx = base.FakeTradeCtx
ERR = ("raise", ConnectionError("OpenD dropped after ACK"))


class _GuardCase(base._AuditCase):
    def _buy(self, code="US.TEST", qty=10, **kw):
        return runner._place_order(code=code, side="BUY", qty=qty, price=100.0,
                                   trd_env="SIMULATE_ENUM", env_label="SIMULATE",
                                   confirmed=kw.pop("confirmed", True),
                                   audit_intent="ENTRY", **kw)

    def _fresh_ctx(self, **kw):
        self.ctx = FakeTradeCtx(**kw)
        return self.ctx

    def _reconcile(self, confirmed=True):
        return eg.reconcile_pending(trd_env="SIMULATE_ENUM", env_label="SIMULATE",
                                    confirmed=confirmed)

    def _row(self, audit_id):
        return next(r for r in self._rows() if r["audit_id"] == audit_id)

    def _events(self, audit_id):
        import json
        import sqlite3
        con = sqlite3.connect(str(oa._DB_PATH))
        try:
            return [(st, json.loads(d)) for st, d in con.execute(
                "SELECT stage, detail_json FROM order_audit_events WHERE audit_id=? "
                "ORDER BY event_id", (audit_id,))]
        finally:
            con.close()

    def _make_unknown(self, kind="after_ack"):
        """First BUY ends in an unconfirmed state; returns its audit row."""
        if kind == "after_ack":
            self.ctx.polls = [ERR]
        elif kind == "submit":
            self.ctx.place = ("raise", TimeoutError("response lost"))
        elif kind == "poll_and_cancel_fail":
            self.ctx.polls = [("error",)]
            self.ctx.recheck = ("error",)
            self.ctx.cancel = RuntimeError("cancel refused")
        self._buy()
        row = self._last()
        self.assertIn(row["outcome"], oa.UNCONFIRMED)
        return row

    def assertBlocked(self, fill, *needles):
        self.assertEqual((fill["status"], fill["dealt_qty"], fill["order_id"]),
                         (oa.EXECUTION_GUARD_BLOCKED, 0.0, ""))
        row = self._last()
        self.assertEqual((row["outcome"], row["guard_status"]),
                         (oa.EXECUTION_GUARD_BLOCKED, "BLOCK"))
        for n in needles:
            self.assertIn(n, row["guard_reason"])
        self.assertIsNone(row["submit_sent_at"])          # never reached the broker

    def assertNotBlocked(self, fill):
        self.assertNotEqual(fill["status"], oa.EXECUTION_GUARD_BLOCKED)
        self.assertEqual(self._last()["guard_status"], "PASS")


# ── P7.5: normal terminal states never block ─────────────────────────────────

class NormalTerminalStatesTest(_GuardCase):
    def _second_buy_goes_through(self):
        ctx = self._fresh_ctx(place=("ok", "OID2"))
        fill = self._buy()
        self.assertNotBlocked(fill)
        self.assertEqual(ctx.place_calls, 1)

    def test_filled_does_not_block(self):                                   # 1
        self._buy()
        self.assertEqual(self._last()["outcome"], oa.FILLED)
        self._second_buy_goes_through()

    def test_not_filled_does_not_block(self):                               # 2
        self.ctx.polls = [("row", "SUBMITTED", 0, 0.0)]
        self.ctx.recheck = ("row", "CANCELLED_ALL", 0, 0.0)
        self._buy()
        self.assertEqual(self._last()["outcome"], oa.NOT_FILLED)
        self._second_buy_goes_through()

    def test_cancelled_does_not_block(self):                                # 3
        self.ctx.polls = [("row", "CANCELLED_ALL", 0, 0.0)]
        self._buy()
        self.assertEqual(self._last()["outcome"], oa.NOT_FILLED)
        self._second_buy_goes_through()

    def test_normal_path_makes_no_reconcile_broker_call(self):
        self._buy()
        calls = []
        class _Spy:
            @staticmethod
            def query_orders(*a, **kw):
                calls.append(a)
                return True, []
        eg.reconcile_pending("SIMULATE_ENUM", "SIMULATE", True, broker=_Spy)
        self.assertEqual(calls, [])

    def test_dry_run_and_rejection_do_not_block(self):
        self._buy(confirmed=False)
        self.ctx.place = ("reject", "insufficient buying power")
        self._buy()
        self.assertEqual(self._last()["outcome"], oa.SUBMIT_REJECTED)
        self._second_buy_goes_through()

    def test_sell_is_never_guarded(self):
        self._make_unknown()
        self._fresh_ctx(place=("ok", "OID2"))
        fill = runner._place_order(code="US.TEST", side="SELL", qty=10, price=100.0,
                                   trd_env="SIMULATE_ENUM", env_label="SIMULATE",
                                   confirmed=True, audit_intent="EXIT:STOP_LOSS")
        self.assertEqual(fill["status"], "FILLED_ALL")
        self.assertIsNone(self._last()["guard_status"])


# ── P7.3: CANCELLED_PART ─────────────────────────────────────────────────────

class CancelledPartTest(_GuardCase):
    def test_partial_then_cancelled_part_seen_while_polling(self):          # 4
        self.ctx.polls = [("row", "FILLED_PART", 4, 100.0), ("row", "CANCELLED_PART", 4, 100.0)]
        fill = self._buy()
        self.assertEqual((fill["status"], fill["dealt_qty"]), ("CANCELLED_PART", 4.0))
        self.assertEqual(self.ctx.cancel_calls, 0)        # terminal -> no second cancel
        self.assertEqual(self.ctx.poll_i, 2)              # stopped polling, no timeout
        row = self._last()
        self.assertEqual((row["outcome"], row["partial_fill"], row["dealt_qty"],
                          row["requested_qty"], row["needs_reconcile"], row["status_confirmed"]),
                         (oa.PARTIAL_FILLED, 1, 4.0, 10.0, 0, 1))
        finish = [d for st, d in self._events(row["audit_id"]) if st == "FINISH"][0]
        self.assertTrue(finish["remainder_cancelled"])
        self._fresh_ctx(place=("ok", "OID2"))
        self.assertNotBlocked(self._buy())

    def test_cancelled_part_after_our_cancel(self):
        self.ctx.polls = [("row", "FILLED_PART", 4, 100.0)]
        self.ctx.recheck = ("row", "CANCELLED_PART", 4, 100.0)
        fill = self._buy()
        self.assertEqual(fill["dealt_qty"], 4.0)
        self.assertEqual(self.ctx.cancel_calls, 1)
        self.assertEqual(self._last()["outcome"], oa.PARTIAL_FILLED)

    def test_cancelling_in_progress_is_still_unknown(self):
        # Transitional state after cancel: remainder may still fill -> protected.
        self.ctx.polls = [("row", "FILLED_PART", 4, 100.0)]
        self.ctx.recheck = ("row", "CANCELLING_PART", 4, 100.0)
        self._buy()
        row = self._last()
        self.assertEqual((row["outcome"], row["unknown_reason"]),
                         (oa.UNCONFIRMED_AFTER_ACK, "NON_TERMINAL_AFTER_CANCEL:CANCELLING_PART"))


# ── P7.1 / P7.4 / P7.5: unknown states block the next BUY ────────────────────

class UnknownStatesBlockTest(_GuardCase):
    def test_submit_outcome_unknown_blocks(self):                           # 5
        self._make_unknown("submit")
        ctx = self._fresh_ctx(place=("ok", "OID2"))
        self.assertBlocked(self._buy(), "UNRESOLVED_ORDER", oa.SUBMIT_OUTCOME_UNKNOWN,
                           "SUBMIT_EXCEPTION_AFTER_SEND")
        self.assertEqual(ctx.place_calls, 0)

    def test_unconfirmed_after_ack_blocks(self):                            # 6
        self._make_unknown("after_ack")
        ctx = self._fresh_ctx(place=("ok", "OID2"))
        self.assertBlocked(self._buy(), oa.UNCONFIRMED_AFTER_ACK, "order_id=OID1")
        self.assertEqual(ctx.place_calls, 0)

    def test_all_polls_fail_and_cancel_fails_blocks(self):                  # 7
        row = self._make_unknown("poll_and_cancel_fail")
        self.assertEqual((row["order_id"], row["unknown_reason"], row["needs_reconcile"],
                          row["fill_confirmed"]),
                         ("OID1", "ALL_POLLS_FAILED_AND_CANCEL_FAILED", 1, 0))
        stages = self._stages(row["audit_id"])
        for st in ("ACK", "POLL_FAILED", "CANCEL_SENT", "CANCEL_FAILED", "FINISH", "UNKNOWN"):
            self.assertIn(st, stages)
        ctx = self._fresh_ctx(place=("ok", "OID2"))
        self.assertBlocked(self._buy(), "ALL_POLLS_FAILED_AND_CANCEL_FAILED")
        self.assertEqual(ctx.place_calls, 0)

    def test_same_symbol_only(self):                                        # 12
        self._make_unknown("after_ack")
        self._fresh_ctx(place=("ok", "OID2"))
        self.assertBlocked(self._buy(code="US.TEST"))
        self.assertNotBlocked(self._buy(code="US.OTHER"))

    def test_other_env_not_blocked(self):
        self._make_unknown("after_ack")
        self._fresh_ctx(place=("ok", "OID2"))
        fill = runner._place_order(code="US.TEST", side="BUY", qty=10, price=100.0,
                                   trd_env="REAL_ENUM", env_label="REAL", confirmed=True,
                                   audit_intent="ENTRY")
        self.assertNotBlocked(fill)

    def test_blocks_every_buy_intent_and_dry_run(self):
        self._make_unknown("after_ack")
        for intent in ("PYRAMID_ADD", "PYRAMID_PROMOTE", "QQQ_CORE_BUY"):
            fill = runner._place_order(code="US.TEST", side="BUY", qty=1, price=100.0,
                                       trd_env="SIMULATE_ENUM", env_label="SIMULATE",
                                       confirmed=True, audit_intent=intent)
            self.assertBlocked(fill)
        self.assertBlocked(self._buy(confirmed=False))

    def test_crashed_in_flight_row_blocks(self):
        # Process died between SUBMIT_SENT and FINISH: no outcome at all.
        oa.begin("US.TEST", "BUY", 10, 100.0, "SIMULATE_ENUM", "SIMULATE", True, intent="ENTRY")
        oa.event("SUBMIT_SENT", price=100.0, qty=10, side="BUY")
        oa._current_order.set(None)                       # "crash": finish() never runs
        self.assertBlocked(self._buy(), "IN_FLIGHT")

    def test_db_unreadable_falls_back_to_in_process_record(self):
        self._make_unknown("after_ack")
        from pathlib import Path
        good = oa._DB_PATH
        oa._DB_PATH = Path(self._db_tmp.name)             # a directory: sqlite can't open it
        try:
            fill = self._buy()
        finally:
            oa._DB_PATH = good
        self.assertEqual(fill["status"], oa.EXECUTION_GUARD_BLOCKED)


# ── P7.2: reconciliation ─────────────────────────────────────────────────────

class ReconcileTest(_GuardCase):
    def test_reconcile_not_filled_releases_protection(self):                # 8
        row = self._make_unknown("after_ack")
        self._fresh_ctx(place=("ok", "OID2"), polls=[("row", "CANCELLED_ALL", 0, 0.0)])
        summary = self._reconcile()
        self.assertEqual(summary["resolved"], [row["audit_id"]])
        r = self._row(row["audit_id"])
        self.assertEqual((r["reconcile_status"], r["reconciled_outcome"], r["reconciled_dealt_qty"],
                          r["reconciled_broker_status"], r["reconciled_order_id"], r["needs_manual"]),
                         (oa.RECONCILE_RESOLVED, oa.NOT_FILLED, 0.0, "CANCELLED_ALL", "OID1", 0))
        self.assertEqual(r["outcome"], oa.UNCONFIRMED_AFTER_ACK)   # original outcome kept
        end = [d for st, d in self._events(row["audit_id"]) if st == "RECONCILE_END"][-1]
        self.assertTrue(end["protection_released"])
        self.assertIn("RECONCILE_START", self._stages(row["audit_id"]))
        self.assertNotBlocked(self._buy())
        self.assertEqual(self.ctx.place_calls, 1)

    def test_reconcile_fill_already_recorded_releases(self):
        # Poll saw a partial fill (4, recorded by the caller) then raised;
        # broker final = CANCELLED_PART 4 -> book agrees -> released.
        self.ctx.polls = [("row", "FILLED_PART", 4, 100.0), ERR]
        self._buy()
        row = self._last()
        self.assertEqual((row["outcome"], row["returned_dealt_qty"]), (oa.UNCONFIRMED_AFTER_ACK, 4.0))
        self._fresh_ctx(place=("ok", "OID2"), polls=[("row", "CANCELLED_PART", 4, 100.0)])
        self._reconcile()
        r = self._row(row["audit_id"])
        self.assertEqual((r["reconcile_status"], r["reconciled_outcome"]),
                         (oa.RECONCILE_RESOLVED, oa.PARTIAL_FILLED))
        self.assertNotBlocked(self._buy())

    def test_reconcile_unrecorded_fill_needs_manual_and_keeps_blocking(self):
        row = self._make_unknown("after_ack")
        self._fresh_ctx(place=("ok", "OID2"), polls=[("row", "FILLED_ALL", 10, 100.5)])
        self._reconcile()
        r = self._row(row["audit_id"])
        self.assertEqual((r["reconcile_status"], r["reconciled_outcome"], r["reconciled_dealt_qty"],
                          r["reconciled_fill_price"], r["needs_manual"]),
                         (oa.RECONCILE_NEEDS_MANUAL, oa.FILLED, 10.0, 100.5, 1))
        self.assertIn("local book short by +10", r["manual_reason"])
        self.assertBlocked(self._buy(), oa.RECONCILE_NEEDS_MANUAL)
        # Already final: later passes don't query the broker again for it.
        ctx = self._fresh_ctx(polls=[("error",)])
        self._reconcile()
        self.assertEqual(ctx.poll_i, 0)

    def test_manual_clear_releases(self):
        row = self._make_unknown("after_ack")
        self._fresh_ctx(place=("ok", "OID2"), polls=[("row", "FILLED_ALL", 10, 100.0)])
        self._reconcile()
        self.assertTrue(eg.manual_clear(row["audit_id"], "added 10 to positions.json"))
        self.assertEqual(self._row(row["audit_id"])["reconcile_status"], oa.RECONCILE_MANUAL_CLEARED)
        self.assertIn("MANUAL_CLEAR", self._stages(row["audit_id"]))
        self.assertNotBlocked(self._buy())
        self.assertFalse(eg.manual_clear(row["audit_id"], "again"))

    def test_cli_list_and_clear(self):
        row = self._make_unknown("after_ack")
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(eg._main(["list", "--env", "SIMULATE"]), 0)
        self.assertIn(row["audit_id"], out.getvalue())
        self.assertIn("1 unresolved", out.getvalue())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(eg._main(["clear", row["audit_id"], "--note", "checked broker"]), 0)
            self.assertEqual(eg._main(["clear", row["audit_id"], "--note", "x"]), 1)

    def test_reconcile_query_failure_keeps_protection_then_needs_manual(self):  # 9
        row = self._make_unknown("after_ack")
        for attempt in range(1, eg.MANUAL_AFTER_ATTEMPTS + 1):
            self._fresh_ctx(polls=[("error",)])
            summary = self._reconcile()
            self.assertEqual(summary["unresolved"], [row["audit_id"]])
            r = self._row(row["audit_id"])
            self.assertEqual((r["reconcile_status"], r["reconcile_attempts"]),
                             (oa.RECONCILE_UNRESOLVED, attempt))
            self.assertEqual(r["needs_manual"], int(attempt >= eg.MANUAL_AFTER_ATTEMPTS))
            self.assertBlocked(self._buy(), oa.RECONCILE_UNRESOLVED)
        self.assertEqual(r["manual_reason"], "BROKER_QUERY_FAILED")
        end = [d for st, d in self._events(row["audit_id"]) if st == "RECONCILE_END"][-1]
        self.assertFalse(end["protection_released"])
        # ... and a later successful query still resolves it.
        self._fresh_ctx(polls=[("row", "CANCELLED_ALL", 0, 0.0)])
        self._reconcile()
        self.assertEqual(self._row(row["audit_id"])["reconcile_status"], oa.RECONCILE_RESOLVED)

    def test_reconcile_raising_broker_keeps_protection(self):
        row = self._make_unknown("after_ack")
        self._fresh_ctx(polls=[("raise", OSError("OpenD down"))])
        self._reconcile()
        self.assertEqual(self._row(row["audit_id"])["reconcile_status"], oa.RECONCILE_UNRESOLVED)
        self.assertBlocked(self._buy())

    def test_acked_order_missing_at_broker_is_not_assumed_unfilled(self):
        row = self._make_unknown("after_ack")
        ctx = self._fresh_ctx()
        ctx.order_list_query = lambda **kw: (base.RET_OK, base.pd.DataFrame())
        self._reconcile()
        r = self._row(row["audit_id"])
        self.assertEqual(r["reconcile_status"], oa.RECONCILE_UNRESOLVED)
        self.assertBlocked(self._buy())

    def test_open_order_is_cancelled_then_resolved(self):
        row = self._make_unknown("after_ack")
        ctx = self._fresh_ctx(polls=[("row", "SUBMITTED", 0, 0.0)],
                              recheck=("row", "CANCELLED_ALL", 0, 0.0))
        self._reconcile()
        self.assertEqual(ctx.cancel_calls, 1)
        r = self._row(row["audit_id"])
        self.assertEqual((r["reconcile_status"], r["reconciled_outcome"]),
                         (oa.RECONCILE_RESOLVED, oa.NOT_FILLED))
        self.assertIn("RECONCILE_CANCEL_SENT", self._stages(row["audit_id"]))

    def test_open_order_cancel_fails_stays_unresolved(self):
        row = self._make_unknown("after_ack")
        ctx = self._fresh_ctx(polls=[("row", "SUBMITTED", 0, 0.0)],
                              cancel=RuntimeError("cancel refused"))
        self._reconcile()
        self.assertEqual(ctx.cancel_calls, 1)
        self.assertEqual(self._row(row["audit_id"])["reconcile_status"], oa.RECONCILE_UNRESOLVED)
        self.assertIn("RECONCILE_CANCEL_FAILED", self._stages(row["audit_id"]))
        self.assertBlocked(self._buy())

    def test_dry_run_pass_never_cancels(self):
        row = self._make_unknown("after_ack")
        ctx = self._fresh_ctx(polls=[("row", "SUBMITTED", 0, 0.0)])
        self._reconcile(confirmed=False)
        self.assertEqual(ctx.cancel_calls, 0)
        self.assertEqual(self._row(row["audit_id"])["reconcile_status"], oa.RECONCILE_UNRESOLVED)

    def test_lost_response_matched_by_search(self):
        row = self._make_unknown("submit")
        found = {"order_id": "OIDX", "code": "US.TEST", "trd_side": "BUY", "qty": 10,
                 "price": 100.0, "order_status": "FILLED_ALL", "dealt_qty": 10,
                 "dealt_avg_price": 100.2}
        other = dict(found, order_id="OIDY", qty=3)       # different qty: not ours
        self._fresh_ctx(search=[found, other], polls=[("error",)])
        self._reconcile()
        r = self._row(row["audit_id"])
        self.assertEqual((r["reconcile_status"], r["reconciled_order_id"], r["reconciled_dealt_qty"]),
                         (oa.RECONCILE_NEEDS_MANUAL, "OIDX", 10.0))

    def test_lost_response_not_found_resolves_unfilled(self):
        row = self._make_unknown("submit")
        self._fresh_ctx(search=[], history=[])
        self._reconcile()
        r = self._row(row["audit_id"])
        self.assertEqual((r["reconcile_status"], r["reconciled_outcome"], r["reconciled_broker_status"]),
                         (oa.RECONCILE_RESOLVED, "NOT_FOUND", "NOT_FOUND"))
        self.assertNotBlocked(self._buy())

    def test_lost_response_search_failure_is_not_not_found(self):
        row = self._make_unknown("submit")
        self._fresh_ctx(search=[], history=("error",))
        self._reconcile()
        self.assertEqual(self._row(row["audit_id"])["reconcile_status"], oa.RECONCILE_UNRESOLVED)
        self.assertBlocked(self._buy())

    def test_lost_response_ignores_orders_owned_by_other_rows(self):
        self._buy()                                       # OID1, FILLED, owned by row 1
        row = self._make_unknown("submit")
        owned = {"order_id": "OID1", "code": "US.TEST", "trd_side": "BUY", "qty": 10,
                 "price": 100.0, "order_status": "FILLED_ALL", "dealt_qty": 10,
                 "dealt_avg_price": 100.0}
        self._fresh_ctx(search=[owned])
        self._reconcile()
        self.assertEqual(self._row(row["audit_id"])["reconciled_outcome"], "NOT_FOUND")

    def test_lost_response_ambiguous_needs_manual(self):
        row = self._make_unknown("submit")
        a = {"order_id": "A", "code": "US.TEST", "trd_side": "BUY", "qty": 10, "price": 100.0,
             "order_status": "FILLED_ALL", "dealt_qty": 10, "dealt_avg_price": 100.0}
        self._fresh_ctx(search=[a, dict(a, order_id="B")])
        self._reconcile()
        r = self._row(row["audit_id"])
        self.assertEqual((r["reconcile_status"], r["needs_manual"]), (oa.RECONCILE_UNRESOLVED, 1))
        self.assertIn("AMBIGUOUS", r["manual_reason"])


# ── P7.1: broker position vs local book ──────────────────────────────────────

class PositionMismatchTest(_GuardCase):
    def _note(self, broker_qty, local_qty, code="US.TEST", ignore=()):
        eg.note_position_diffs([ReconciliationDiff(code, broker_qty, local_qty,
                                                   broker_qty - local_qty)], ignore, "SIMULATE")

    def test_broker_holds_local_does_not(self):                             # 10
        self._note(10, 0)
        self.assertBlocked(self._buy(), "POSITION_MISMATCH broker=10 local=0")
        self.assertEqual(self.ctx.place_calls, 0)

    def test_local_and_broker_qty_differ(self):                             # 11
        self._note(15, 10)
        self.assertBlocked(self._buy(), "POSITION_MISMATCH broker=15 local=10")
        self._note(5, 10)
        self.assertBlocked(self._buy(), "POSITION_MISMATCH broker=5 local=10")

    def test_other_code_unaffected(self):
        self._note(10, 0)
        self.assertNotBlocked(self._buy(code="US.OTHER"))

    def test_ignore_codes_exempt(self):
        # config.RECONCILIATION_IGNORE_CODES (US.QQQ +124 / US.0000): known,
        # stable diffs must not newly block the QQQ Beta top-up.
        self._note(473, 349, code="US.QQQ", ignore=["US.0000", "US.QQQ"])
        self.assertNotBlocked(self._buy(code="US.QQQ"))

    def test_mismatch_snapshot_is_pass_scoped(self):
        self._note(10, 0)
        oa.start_pass(run_id=self.run_id, env_label="SIMULATE")   # next pass, no snapshot yet
        self.assertNotBlocked(self._buy())


class RunOnceMismatchTest(base._RunOnceHarness):
    def test_broker_already_holds_code_entry_not_submitted(self):
        Portfolio(path=self.pf_path)._save()
        self._broker_holds(7)
        self._pass()
        rows = self._entries()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome"], oa.EXECUTION_GUARD_BLOCKED)
        self.assertIn("POSITION_MISMATCH broker=7 local=0", rows[0]["guard_reason"])
        self.assertEqual(self.ctx.place_calls, 0)

    def test_clean_state_entry_still_filled(self):
        Portfolio(path=self.pf_path)._save()
        self._pass()
        rows = self._entries()
        self.assertEqual(rows[0]["guard_status"], "PASS")
        self.assertIn(rows[0]["outcome"], (oa.FILLED, oa.PARTIAL_FILLED))
        self.assertEqual(self.ctx.place_calls, 1)
        self.assertIn("US.TEST", Portfolio(path=self.pf_path).data["positions"])

    def test_reconcile_success_lets_next_pass_buy(self):
        Portfolio(path=self.pf_path)._save()
        self.ctx.polls = [ERR]
        self._pass()
        # Broker says OID1 was cancelled unfilled -> released, so pass 2 buys.
        self.ctx = base.FakeTradeCtx(place=("ok", "OID2"),
                                     polls=[("row", "CANCELLED_ALL", 0, 0.0),
                                            ("row", "FILLED_ALL", 10, 100.0)])
        self._pass()
        rows = self._entries()
        self.assertEqual(rows[0]["reconcile_status"], oa.RECONCILE_RESOLVED)
        self.assertEqual(rows[1]["guard_status"], "PASS")
        self.assertEqual(self.ctx.place_calls, 1)


def setUpModule():
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
