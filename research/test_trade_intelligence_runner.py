"""
Unit tests for research/trade_intelligence_runner.py — end-to-end refresh()
against throwaway temp DBs on both ends, plus a static guardrail asserting
the V3.6-A package never imports live-trading code and is never imported by
engine/runner.py.

Run:  python -m unittest research.test_trade_intelligence_runner -v
"""
import ast
import tempfile
import unittest
from pathlib import Path

from engine.trade_tracker import TradeTracker
from research.trade_intelligence_runner import refresh

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESEARCH_DIR = _REPO_ROOT / "research"

_FORBIDDEN_IMPORT_PREFIXES = (
    "engine.runner", "strategies", "engine.scoring", "engine.confidence_score",
    "risk.dynamic_sizing", "risk.sizing", "position_manager",
    "risk.portfolio_position_manager", "exit_engine", "risk.portfolio_risk_engine",
    "ai_decision",
)


def _module_imports(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


class TradeIntelligenceRunnerTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.source_db = Path(self._tmpdir.name) / "fake_trade_history.db"
        self.research_db = Path(self._tmpdir.name) / "fake_trade_intelligence.db"
        self.report_dir = Path(self._tmpdir.name) / "reports"
        self.tracker = TradeTracker(self.source_db)

    def tearDown(self):
        self.tracker.close()
        self._tmpdir.cleanup()

    def _make_trade(self, trade_id, ticker, exit_price):
        self.tracker.log_entry(
            trade_id=trade_id, ticker=ticker, strategy_name="atr_breakout",
            strategy_version="test", direction="LONG", price=100.0, shares=10,
            position_value=1000.0, position_pct=0.1, cash=9000.0, equity=10000.0,
            timestamp="2026-01-01T00:00:00", execution="REAL",
        )
        self.tracker.update_position_metrics(
            trade_id=trade_id, date="2026-01-01", close=exit_price,
            entry_price=100.0, shares=10, high=max(exit_price, 101.0), low=95.0)
        self.tracker.log_exit(
            trade_id=trade_id, price=exit_price, cash=1.0, equity=1.0,
            timestamp="2026-01-01T12:00:00", exit_reason="STRATEGY_EXIT(atr_breakout)")
        self.tracker.log_exit_diagnostics(trade_id=trade_id, strategy_exit_triggered=True)

    def test_refresh_end_to_end_completes_and_writes_report(self):
        for i in range(10):
            self._make_trade(f"t{i}", f"US.T{i}", exit_price=90.0 if i % 2 == 0 else 110.0)

        summary = refresh(source_db_path=self.source_db, research_db_path=self.research_db,
                           report_dir=self.report_dir, execution="REAL")

        self.assertEqual(summary["n_trades_analyzed"], 10)
        self.assertTrue((self.report_dir / "trade_intelligence_latest.md").exists())
        self.assertTrue((self.report_dir / "trade_intelligence_latest.json").exists())

        import pandas as pd
        import sqlite3
        con = sqlite3.connect(str(self.research_db))
        runs = pd.read_sql_query("SELECT * FROM research_runs", con)
        con.close()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs.iloc[0]["status"], "OK")

    def test_refresh_on_empty_source_db_does_not_raise(self):
        summary = refresh(source_db_path=self.source_db, research_db_path=self.research_db,
                           report_dir=self.report_dir, execution="REAL")
        self.assertEqual(summary["n_trades_analyzed"], 0)
        self.assertEqual(summary["n_patterns_total"], 0)

    def test_no_trade_intelligence_module_imports_live_trading_code(self):
        for path in _RESEARCH_DIR.glob("trade_intelligence_*.py"):
            imports = _module_imports(path)
            for name in imports:
                for forbidden in _FORBIDDEN_IMPORT_PREFIXES:
                    self.assertFalse(
                        name == forbidden or name.startswith(forbidden + "."),
                        f"{path.name} imports forbidden module {name!r} "
                        f"(matches guardrail prefix {forbidden!r})")

    def test_runner_py_does_not_reference_trade_intelligence(self):
        runner_source = (_REPO_ROOT / "engine" / "runner.py").read_text(encoding="utf-8")
        self.assertNotIn("trade_intelligence", runner_source)
        main_path = _REPO_ROOT / "main.py"
        if main_path.exists():
            self.assertNotIn("trade_intelligence", main_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
