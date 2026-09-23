"""
Unit tests for ai_decision/audit_log.py. Monkeypatches _LOG_PATH to a
tempfile so tests never touch C:\\KabuData.

Run:  python -m unittest ai_decision.test_audit_log -v
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import ai_decision.audit_log as audit_log
from ai_decision.schema import AIAction, AIDecision, AIDecisionContext


class TestLogDecision(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._orig_path = audit_log._LOG_PATH
        audit_log._LOG_PATH = Path(self._tmp_dir.name) / "ai_decision_log.jsonl"

    def tearDown(self):
        audit_log._LOG_PATH = self._orig_path
        self._tmp_dir.cleanup()

    def test_writes_one_jsonl_line_with_expected_fields(self):
        ctx = AIDecisionContext(symbol="US.NVDA", market_regime="NORMAL",
                                 risk_decision_snapshot={"status": "ALLOW"})
        decision = AIDecision(decision=AIAction.HOLD, confidence=0.5,
                               reason_codes=["NO_SIGNAL"], model="mock")
        audit_log.log_decision(ctx, decision)

        self.assertTrue(audit_log._LOG_PATH.exists())
        lines = audit_log._LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["symbol"], "US.NVDA")
        self.assertEqual(record["decision"], "HOLD")
        self.assertEqual(record["input_snapshot"]["market_regime"], "NORMAL")

    def test_never_raises_when_log_path_unwritable(self):
        audit_log._LOG_PATH = Path(self._tmp_dir.name) / "nonexistent_dir" / "sub" / "x.jsonl"
        # make the intended parent path itself a file so mkdir() fails
        blocker = Path(self._tmp_dir.name) / "nonexistent_dir"
        blocker.write_text("not a directory", encoding="utf-8")

        ctx = AIDecisionContext(symbol="US.NVDA")
        decision = AIDecision(decision=AIAction.SKIP, confidence=0.0)
        try:
            audit_log.log_decision(ctx, decision)
        except Exception as exc:
            self.fail(f"log_decision() raised {exc!r}, expected fail-silent")


if __name__ == "__main__":
    unittest.main()
