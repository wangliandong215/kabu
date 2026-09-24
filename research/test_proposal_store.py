"""
Tests for research/proposal_store.py + research/proposal_cli.py (V3.6
Validation Gate) against throwaway temp SQLite DBs. Never touches the
production trade_intelligence.db, and never writes config.py (asserted).

Run:  python -m unittest research.test_proposal_store -v
"""
import ast
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import config
from research import proposal_cli
from research import proposal_store as ps
from research.proposal_store import (APPROVED, BACKTESTED, DRAFT, GATE_PASSED, LADDER,
                                     OOS_PASSED, PAPER, REJECTED, InvalidTransition,
                                     ProposalStore, ValidationFailed)
from research.test_trade_intelligence_store import _make_candidate
from research.trade_intelligence_store import TradeIntelligenceStore
import test_support

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NEW_MODULES = [_REPO_ROOT / "research" / "proposal_store.py",
                _REPO_ROOT / "research" / "proposal_cli.py"]
_FORBIDDEN_IMPORT_PREFIXES = (
    "engine", "risk", "strategies", "portfolio", "position_manager", "exit_engine",
    "ai_decision", "news", "regime", "notify", "runner", "main", "trade", "common",
)

# A real, existing config parameter; the proposal changes it only in the DB.
_PARAM = "STOP_LOSS_PCT"
_BT = {"period_start": "2022-01-01", "period_end": "2024-12-31", "total_return": 0.18}
_OOS_OK = {"period_start": "2025-01-01", "period_end": "2025-12-31", "passed": True}
_PAPER_OK = {"period_start": "2026-01-01", "period_end": "2026-03-31",
             "passed": True, "n_trades": ps.MIN_PAPER_TRADES}


def _config_py_digest() -> str:
    return hashlib.sha256((_REPO_ROOT / "config.py").read_bytes()).hexdigest()


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "ti.db"
        self.store = ProposalStore(db_path=self.db)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _new(self, value=None):
        value = config.STOP_LOSS_PCT * 0.8 if value is None else value
        return self.store.create("Tighten stop", {_PARAM: value}, created_by="tester",
                                 rationale="unit test")

    def _advance_to(self, target):
        row = self._new()
        pid, v = row["proposal_id"], row["version"]
        steps = {
            BACKTESTED: lambda: self.store.record_backtest(pid, v, _BT, "tester"),
            OOS_PASSED: lambda: self.store.record_oos(pid, v, _OOS_OK, "tester"),
            PAPER: lambda: self.store.start_paper(pid, v, "tester"),
            GATE_PASSED: lambda: (self.store.record_paper_result(pid, v, _PAPER_OK, "tester"),
                                  self.store.pass_gate(pid, v, "tester")),
        }
        for status in LADDER[1:LADDER.index(target) + 1]:
            steps[status]()
        self.assertEqual(self.store.get(pid, v)["status"], target)
        return pid, v


class CreateTest(_StoreCase):
    def test_create_records_draft_with_snapshots_hash_and_commit(self):
        row = self._new(0.04)
        self.assertEqual(row["status"], DRAFT)
        self.assertEqual(row["version"], 1)
        self.assertTrue(row["proposal_id"].startswith("PR-"))
        before = json.loads(row["params_before_json"])
        after = json.loads(row["params_after_json"])
        self.assertEqual(before, {_PARAM: config.STOP_LOSS_PCT})
        self.assertEqual(after, {_PARAM: 0.04})
        self.assertEqual(row["params_before_hash"], ps.params_hash(before))
        self.assertEqual(row["params_after_hash"], ps.params_hash(after))
        self.assertNotEqual(row["params_before_hash"], row["params_after_hash"])
        self.assertTrue(row["git_commit"])
        ev = self.store.events(row["proposal_id"])
        self.assertEqual(list(ev["event_type"]), ["CREATED"])

    def test_create_refuses_unknown_param_noop_change_and_empty(self):
        with self.assertRaises(ValidationFailed):
            self.store.create("x", {"NO_SUCH_PARAM_XYZ": 1}, "tester")
        with self.assertRaises(ValidationFailed):
            self.store.create("x", {_PARAM: config.STOP_LOSS_PCT}, "tester")
        with self.assertRaises(ValidationFailed):
            self.store.create("x", {}, "tester")
        with self.assertRaises(ValidationFailed):
            self.store.create("x", {_PARAM: 0.01}, "  ")

    def test_source_pattern_must_exist(self):
        with self.assertRaises(ValidationFailed):
            self.store.create("x", {_PARAM: 0.01}, "tester", source_pattern_id="EARLY_FAILURE:k1")
        ti = TradeIntelligenceStore(db_path=self.db)
        ti.upsert_pattern(_make_candidate(), run_id="r1")
        ti.close()
        row = self.store.create("x", {_PARAM: 0.01}, "tester",
                                source_pattern_id="EARLY_FAILURE:k1")
        self.assertEqual(row["source_pattern_id"], "EARLY_FAILURE:k1")

    def test_set_valued_config_snapshot_is_order_stable(self):
        snap = ps.snapshot_params(["PREFLIGHT_BLOCKED_MARKET_STATES"])
        self.assertEqual(snap["PREFLIGHT_BLOCKED_MARKET_STATES"],
                         sorted(config.PREFLIGHT_BLOCKED_MARKET_STATES))


class LadderTest(_StoreCase):
    def test_full_happy_path_records_every_artifact_and_event(self):
        pid, v = self._advance_to(GATE_PASSED)
        row = self.store.approve(pid, v, approved_by="wang")
        self.assertEqual(row["status"], APPROVED)
        self.assertEqual(row["approved_by"], "wang")
        self.assertTrue(row["approved_at"])
        for col in ("backtest_result_json", "oos_result_json", "paper_result_json",
                    "gate_result_json"):
            self.assertTrue(row[col], col)
        self.assertEqual(list(self.store.events(pid)["event_type"]),
                         ["CREATED", "BACKTEST_RECORDED", "OOS_RECORDED", "PAPER_STARTED",
                          "PAPER_RESULT_RECORDED", "GATE_PASSED", "APPROVED"])

    def test_no_step_can_be_skipped_via_python_api(self):
        pid, v = self._advance_to(DRAFT)
        with self.assertRaises(InvalidTransition):
            self.store.record_oos(pid, v, _OOS_OK, "tester")
        with self.assertRaises(InvalidTransition):
            self.store.start_paper(pid, v, "tester")
        with self.assertRaises(InvalidTransition):
            self.store.record_paper_result(pid, v, _PAPER_OK, "tester")
        with self.assertRaises(InvalidTransition):
            self.store.pass_gate(pid, v, "tester")
        with self.assertRaises(InvalidTransition):
            self.store.approve(pid, v, approved_by="wang")
        self.assertEqual(self.store.get(pid, v)["status"], DRAFT)

        pid, v = self._advance_to(OOS_PASSED)
        with self.assertRaises(InvalidTransition):
            self.store.pass_gate(pid, v, "tester")
        with self.assertRaises(InvalidTransition):
            self.store.approve(pid, v, approved_by="wang")
        with self.assertRaises(InvalidTransition):
            self.store.record_backtest(pid, v, _BT, "tester")   # no going back either

        pid, v = self._advance_to(PAPER)
        with self.assertRaises(InvalidTransition):
            self.store.approve(pid, v, approved_by="wang")

    def test_internal_transition_refuses_non_adjacent_pairs(self):
        pid, v = self._advance_to(DRAFT)
        for target in (OOS_PASSED, PAPER, GATE_PASSED, APPROVED):
            with self.assertRaises(InvalidTransition):
                self.store._transition(pid, v, DRAFT, target, "tester", "HACK")
        self.assertEqual(self.store.get(pid, v)["status"], DRAFT)

    def test_reject_allowed_from_every_non_terminal_state(self):
        for status in LADDER[:-1]:
            pid, v = self._advance_to(status)
            row = self.store.reject(pid, v, "not convincing", "tester")
            self.assertEqual(row["status"], REJECTED)
            self.assertEqual(row["rejected_from_status"], status)
            self.assertEqual(row["rejection_reason"], "not convincing")

    def test_reject_needs_reason_and_terminal_states_are_final(self):
        pid, v = self._advance_to(BACKTESTED)
        with self.assertRaises(ValidationFailed):
            self.store.reject(pid, v, "  ", "tester")
        self.store.reject(pid, v, "why", "tester")
        with self.assertRaises(InvalidTransition):
            self.store.reject(pid, v, "again", "tester")
        pid, v = self._advance_to(GATE_PASSED)
        self.store.approve(pid, v, approved_by="wang")
        with self.assertRaises(InvalidTransition):
            self.store.reject(pid, v, "late", "tester")

    def test_compare_and_set_second_writer_loses(self):
        pid, v = self._advance_to(DRAFT)
        other = ProposalStore(db_path=self.db)
        try:
            other.record_backtest(pid, v, _BT, "a")
            with self.assertRaises(InvalidTransition):
                self.store.record_backtest(pid, v, _BT, "b")
        finally:
            other.close()
        self.assertEqual(len(self.store.events(pid)), 2)


class ValidationRulesTest(_StoreCase):
    def test_oos_must_not_overlap_backtest(self):
        pid, v = self._advance_to(BACKTESTED)
        overlap = dict(_OOS_OK, period_start="2024-06-01")
        with self.assertRaises(ValidationFailed):
            self.store.record_oos(pid, v, overlap, "tester")
        self.assertEqual(self.store.get(pid, v)["status"], BACKTESTED)

    def test_failed_oos_rejects(self):
        pid, v = self._advance_to(BACKTESTED)
        row = self.store.record_oos(pid, v, dict(_OOS_OK, passed=False), "tester")
        self.assertEqual(row["status"], REJECTED)
        self.assertEqual(row["rejection_reason"], "OOS failed")

    def test_results_need_periods_and_flags(self):
        pid, v = self._advance_to(DRAFT)
        with self.assertRaises(ValidationFailed):
            self.store.record_backtest(pid, v, {"total_return": 0.2}, "tester")
        with self.assertRaises(ValidationFailed):
            self.store.record_backtest(pid, v, {"period_start": "2024-01-01",
                                                "period_end": "2023-01-01"}, "tester")
        self.store.record_backtest(pid, v, _BT, "tester")
        with self.assertRaises(ValidationFailed):
            self.store.record_oos(pid, v, {k: _OOS_OK[k] for k in ("period_start", "period_end")},
                                  "tester")

    def test_gate_failure_keeps_paper_and_logs_event(self):
        for bad in (dict(_PAPER_OK, n_trades=ps.MIN_PAPER_TRADES - 1),
                    dict(_PAPER_OK, passed=False),
                    dict(_PAPER_OK, period_start="2025-06-01")):
            pid, v = self._advance_to(PAPER)
            self.store.record_paper_result(pid, v, bad, "tester")
            with self.assertRaises(ValidationFailed):
                self.store.pass_gate(pid, v, "tester")
            self.assertEqual(self.store.get(pid, v)["status"], PAPER)
            self.assertEqual(self.store.events(pid)["event_type"].iloc[-1], "GATE_CHECK_FAILED")

    def test_gate_without_paper_result_fails(self):
        pid, v = self._advance_to(PAPER)
        with self.assertRaises(ValidationFailed):
            self.store.pass_gate(pid, v, "tester")

    def test_stale_config_blocks_gate_and_approval(self):
        pid, v = self._advance_to(PAPER)
        self.store.record_paper_result(pid, v, _PAPER_OK, "tester")
        original = config.STOP_LOSS_PCT
        try:
            config.STOP_LOSS_PCT = original + 0.01   # in-memory only
            with self.assertRaises(ValidationFailed):
                self.store.pass_gate(pid, v, "tester")
        finally:
            config.STOP_LOSS_PCT = original
        self.store.pass_gate(pid, v, "tester")
        try:
            config.STOP_LOSS_PCT = original + 0.01
            with self.assertRaises(ValidationFailed):
                self.store.approve(pid, v, approved_by="wang")
        finally:
            config.STOP_LOSS_PCT = original
        self.assertEqual(self.store.get(pid, v)["status"], GATE_PASSED)

    def test_approver_must_be_a_named_person(self):
        pid, v = self._advance_to(GATE_PASSED)
        for who in ("", "   ", "AI", "claude", "system", "auto"):
            with self.assertRaises(ValidationFailed):
                self.store.approve(pid, v, approved_by=who)
        self.assertEqual(self.store.get(pid, v)["status"], GATE_PASSED)

    def test_revise_only_from_rejected_keeps_old_version(self):
        pid, v = self._advance_to(BACKTESTED)
        with self.assertRaises(InvalidTransition):
            self.store.revise(pid, {_PARAM: 0.03}, "tester")
        self.store.reject(pid, v, "too aggressive", "tester")
        row2 = self.store.revise(pid, {_PARAM: 0.03}, "tester")
        self.assertEqual((row2["version"], row2["status"]), (2, DRAFT))
        old = self.store.get(pid, 1)
        self.assertEqual((old["status"], old["rejection_reason"]), (REJECTED, "too aggressive"))


class DatabaseTriggerTest(_StoreCase):
    """Defense in depth: raw SQL from outside proposal_store.py is refused too."""

    def _raw(self, sql, params=()):
        con = sqlite3.connect(str(self.db))
        try:
            con.execute(sql, params)
            con.commit()
        finally:
            con.close()

    def test_raw_skip_and_raw_approve_refused(self):
        pid, v = self._advance_to(DRAFT)
        with self.assertRaises(sqlite3.IntegrityError):
            self._raw("UPDATE proposals SET status='APPROVED', approved_by='x' "
                      "WHERE proposal_id=?", (pid,))
        with self.assertRaises(sqlite3.IntegrityError):
            self._raw("UPDATE proposals SET status='GATE_PASSED' WHERE proposal_id=?", (pid,))
        pid, v = self._advance_to(GATE_PASSED)
        with self.assertRaises(sqlite3.IntegrityError):
            self._raw("UPDATE proposals SET status='APPROVED' WHERE proposal_id=?", (pid,))
        with self.assertRaises(sqlite3.IntegrityError):
            self._raw("UPDATE proposals SET status='BOGUS' WHERE proposal_id=?", (pid,))

    def test_raw_insert_must_be_draft(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self._raw("INSERT INTO proposals (proposal_id, version, title, created_by, "
                      "created_at, updated_at, status, params_before_json, params_before_hash, "
                      "params_after_json, params_after_hash) VALUES "
                      "('X',1,'t','u','a','a','APPROVED','{}','h','{}','h')")

    def test_terminal_rows_immutable_and_nothing_deletable(self):
        pid, v = self._advance_to(GATE_PASSED)
        self.store.approve(pid, v, approved_by="wang")
        with self.assertRaises(sqlite3.IntegrityError):
            self._raw("UPDATE proposals SET params_after_json='{}' WHERE proposal_id=?", (pid,))
        with self.assertRaises(sqlite3.IntegrityError):
            self._raw("DELETE FROM proposals WHERE proposal_id=?", (pid,))
        with self.assertRaises(sqlite3.IntegrityError):
            self._raw("UPDATE proposal_events SET actor='x' WHERE proposal_id=?", (pid,))
        with self.assertRaises(sqlite3.IntegrityError):
            self._raw("DELETE FROM proposal_events WHERE proposal_id=?", (pid,))


class CliTest(_StoreCase):
    def _cli(self, argv, typed="", tty=True):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = proposal_cli.main(["--db", str(self.db)] + argv,
                                   input_fn=lambda _p: typed, isatty=lambda: tty)
        return rc, out.getvalue(), err.getvalue()

    def _write(self, name, obj):
        path = Path(self._tmp.name) / name
        path.write_text(json.dumps(obj), encoding="utf-8")
        return str(path)

    def test_cli_walks_full_ladder_and_approves_interactively(self):
        rc, out, _ = self._cli(["create", "--title", "t", "--by", "wang",
                                "--set", f"{_PARAM}=0.04"])
        self.assertEqual(rc, 0)
        pid = out.split()[0]
        for argv in (["backtest", pid, "--version", "1", "--result", self._write("bt.json", _BT)],
                     ["oos", pid, "--version", "1", "--result", self._write("o.json", _OOS_OK)],
                     ["paper-start", pid, "--version", "1"],
                     ["paper-result", pid, "--version", "1",
                      "--result", self._write("p.json", _PAPER_OK)],
                     ["gate", pid, "--version", "1"]):
            rc, _, err = self._cli(argv + ["--by", "wang"])
            self.assertEqual(rc, 0, err)

        rc, _, err = self._cli(["approve", pid, "--version", "1", "--by", "wang"],
                               typed=pid, tty=False)
        self.assertEqual(rc, 2)
        self.assertIn("interactive", err)
        rc, _, err = self._cli(["approve", pid, "--version", "1", "--by", "wang"],
                               typed="wrong", tty=True)
        self.assertEqual(rc, 2)
        self.assertEqual(self.store.get(pid, 1)["status"], GATE_PASSED)

        rc, out, _ = self._cli(["approve", pid, "--version", "1", "--by", "wang"],
                               typed=pid, tty=True)
        self.assertEqual(rc, 0)
        row = self.store.get(pid, 1)
        self.assertEqual((row["status"], row["approved_by"]), (APPROVED, "wang"))

    def test_cli_refuses_skip(self):
        rc, out, _ = self._cli(["create", "--title", "t", "--by", "wang",
                                "--set", f"{_PARAM}=0.04"])
        pid = out.split()[0]
        rc, _, err = self._cli(["gate", pid, "--version", "1", "--by", "wang"])
        self.assertEqual(rc, 2)
        self.assertIn("REFUSED", err)


class NoProductionWriteTest(_StoreCase):
    def test_full_lifecycle_leaves_config_py_and_config_values_untouched(self):
        digest = _config_py_digest()
        value = config.STOP_LOSS_PCT
        pid, v = self._advance_to(GATE_PASSED)
        self.store.approve(pid, v, approved_by="wang")
        self.assertEqual(_config_py_digest(), digest)
        self.assertEqual(config.STOP_LOSS_PCT, value)

    def test_default_db_is_research_db_not_trade_history(self):
        self.assertEqual(str(ps._DEFAULT_DB_PATH), config.TRADE_INTELLIGENCE_DB_PATH)
        self.assertNotIn("trade_history", str(ps._DEFAULT_DB_PATH))


def _tree(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class StructuralGuardTest(unittest.TestCase):
    def test_new_modules_import_no_trading_code(self):
        for path in _NEW_MODULES:
            for node in ast.walk(_tree(path)):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    for bad in _FORBIDDEN_IMPORT_PREFIXES:
                        self.assertFalse(name == bad or name.startswith(bad + "."),
                                         f"{path.name} imports {name!r}")

    def test_new_modules_cannot_write_config(self):
        for path in _NEW_MODULES:
            for node in ast.walk(_tree(path)):
                if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for t in targets:
                        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                            self.assertNotEqual(t.value.id, "config",
                                                f"{path.name} assigns config.{t.attr}")
                if isinstance(node, ast.Call):
                    fn = node.func
                    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                    self.assertNotIn(name, ("setattr", "write_text", "write_bytes"),
                                     f"{path.name} calls {name}")
                    if name == "open":
                        modes = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)]
                        modes += [k.value.value for k in node.keywords
                                  if k.arg == "mode" and isinstance(k.value, ast.Constant)]
                        for m in modes:
                            self.assertFalse(set(str(m)) & set("wax+"),
                                             f"{path.name} opens a file for writing")

    def test_only_cli_calls_approve(self):
        allowed = {_REPO_ROOT / "research" / "proposal_cli.py"}
        for path in _REPO_ROOT.rglob("*.py"):
            rel = path.relative_to(_REPO_ROOT).parts
            if rel[0].startswith(".") or path.name.startswith("test_") or path in allowed:
                continue
            for node in ast.walk(_tree(path)):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "approve"):
                    self.fail(f"{path.relative_to(_REPO_ROOT)}:{node.lineno} calls .approve()")

    def test_live_trading_entry_points_do_not_reference_proposals(self):
        for name in ("engine/runner.py", "main.py"):
            path = _REPO_ROOT / name
            if path.exists():
                self.assertNotIn("proposal", path.read_text(encoding="utf-8"), name)


def setUpModule():
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
