"""
Tests for the V3.6 P2① research-side data layer: research/
trade_intelligence_data.py's full JOIN + population classification and
research/trade_intelligence_quality.py's fill-rate report. Throwaway temp
DBs / logs only; asserts the source DB is never modified.

Run:  python -m unittest research.test_trade_intelligence_population -v
"""
import ast
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from engine.trade_tracker import TradeTracker
from research import trade_intelligence_data as data
from research import trade_intelligence_quality as quality
from research.trade_intelligence_runner import refresh
from research.trade_intelligence_store import TradeIntelligenceStore
import test_support

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FORBIDDEN_IMPORT_PREFIXES = ("engine", "risk", "strategies", "portfolio", "position_manager",
                              "exit_engine", "ai_decision", "news", "regime", "notify")

_CTX = {"regime": 1, "regime_label": "TRENDING_UP", "confidence": 80.0,
        "features": {"adx": 25.0, "atr_pct": 0.02, "bb_width": 0.07, "volume_ratio": 1.3}}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _DbCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db = self.dir / "trade_history.db"
        self.risk_log = self.dir / "risk_state_log.jsonl"
        self.tracker = TradeTracker(self.db)

    def tearDown(self):
        self.tracker.close()
        self._tmp.cleanup()

    def _open(self, trade_id, ticker="US.AAA", execution="REAL", equity=10000.0,
              ts="2026-09-10T22:40:00", run_id=None, regime_ctx=_CTX):
        self.tracker.log_entry(
            trade_id=trade_id, ticker=ticker, strategy_name="atr_breakout",
            strategy_version="2.8", direction="LONG", price=100.0, shares=10,
            position_value=1000.0, position_pct=0.1, cash=9000.0, equity=equity,
            timestamp=ts, regime_ctx=regime_ctx, run_id=run_id, atr_entry=2.0,
            execution=execution)

    def _close(self, trade_id, day1="2026-09-11", day2="2026-09-12"):
        self.tracker.update_position_metrics(trade_id=trade_id, date=day1, close=102.0,
                                             entry_price=100.0, shares=10, high=103.0, low=99.0)
        self.tracker.update_position_metrics(trade_id=trade_id, date=day2, close=101.0,
                                             entry_price=100.0, shares=10, high=104.0, low=100.0)
        self.tracker.log_exit(trade_id=trade_id, price=101.0, cash=1.0, equity=1.0,
                              timestamp=f"{day2}T22:00:00", exit_reason="STOP_LOSS")
        self.tracker.log_exit_diagnostics(trade_id=trade_id, stop_loss_triggered=True)

    def _pop(self, **kw):
        return data.load_population(self.db, risk_log_path=self.risk_log, **kw) \
                   .set_index("trade_id")


class JoinTest(_DbCase):
    def test_every_source_table_is_joined(self):
        run_id = self.tracker.log_run_metadata(run_id="run1", strategy_version="2.8",
                                               market="SIMULATE", parameter_hash="ph1",
                                               git_commit="abc1234")
        self._open("t1", run_id=run_id)
        self.tracker.log_market_context(trade_id="t1", ctx={
            "vix_close": 15.5, "cnn_fear_greed": 55.0, "cnn_fear_greed_label": "neutral",
            "naaim_exposure": 90.0})
        self.tracker.log_research_snapshot(trade_id="t1", snapshot={
            "signal_time": "2026-09-10T22:39:00", "rsi14": 61.0, "signal_strength": 1.0,
            "total_score": 90.0, "news_count_24h": 3})
        self._close("t1")
        row = self._pop().loc["t1"]
        self.assertEqual(row["entry_regime_label"], "TRENDING_UP")
        self.assertAlmostEqual(row["entry_adx"], 25.0)
        self.assertAlmostEqual(row["entry_atr_pct"], 0.02)
        self.assertFalse(pd.isna(row["exit_category"]))
        self.assertAlmostEqual(row["mc_vix_close"], 15.5)
        self.assertAlmostEqual(row["mc_naaim_exposure"], 90.0)
        self.assertTrue(pd.isna(row["mc_aaii_spread"]))            # never recorded -> NULL
        self.assertAlmostEqual(row["rs_rsi14"], 61.0)
        self.assertEqual(row["rs_news_count_24h"], 3)
        self.assertEqual(row["parameter_hash"], "ph1")
        self.assertEqual(row["git_commit"], "abc1234")
        self.assertEqual(row["account_env"], "SIMULATE")
        self.assertEqual(row["daily_n_rows"], 2)
        self.assertTrue(pd.isna(row["daily_realized_vol"]))   # only 1 daily return (< 2)
        self.assertEqual(row["daily_regime_changes"], 0)

    def test_metadata_only_via_run_id(self):
        self.tracker.log_run_metadata(run_id="run1", parameter_hash="ph1", git_commit="c1")
        self._open("with_run", run_id="run1")
        self._open("no_run", run_id=None)
        self._open("dangling_run", run_id="missing")
        pop = self._pop(execution=None)
        self.assertEqual(pop.loc["with_run", "parameter_hash"], "ph1")
        self.assertEqual(pop.loc["with_run", "git_commit"], "c1")
        for tid in ("no_run", "dangling_run"):
            self.assertTrue(pd.isna(pop.loc[tid, "parameter_hash"]))
            self.assertTrue(pd.isna(pop.loc[tid, "git_commit"]))
            self.assertTrue(pd.isna(pop.loc[tid, "account_env"]))

    def test_one_row_per_trade(self):
        for i in range(3):
            self._open(f"t{i}")
            self._close(f"t{i}")
        pop = data.load_population(self.db, risk_log_path=self.risk_log)
        self.assertEqual(len(pop), 3)
        self.assertEqual(pop["trade_id"].nunique(), 3)

    def test_source_db_not_modified(self):
        self._open("t1")
        self._close("t1")
        self.tracker.close()
        before = _digest(self.db)
        data.load_population(self.db, risk_log_path=self.risk_log)
        data.load_dataset(self.db, risk_log_path=self.risk_log)
        self.assertEqual(_digest(self.db), before)
        self.tracker = TradeTracker(self.db)


class MarketAndRiskStateTest(_DbCase):
    def _qqq_day(self, date, close, label="HMM_BULL"):
        self.tracker.update_position_metrics(
            trade_id="QQQ_core", date=date, close=close, entry_price=500.0, shares=1,
            regime_ctx={"regime": 11, "regime_label": label, "confidence": 90.0})

    def test_qqq_state_from_prior_recorded_day_only(self):
        self._open("QQQ_core", ticker="US.QQQ", ts="2026-09-01T22:00:00")
        self._qqq_day("2026-09-09", 600.0)
        self._qqq_day("2026-09-10", 610.0, label="HMM_CORRECTION")  # same day as entry
        self._open("t1", ts="2026-09-10T22:40:00")
        self._open("stale", ts="2026-09-20T22:40:00")          # last QQQ day 10 days back
        pop = self._pop(execution=None)
        self.assertEqual(pop.loc["t1", "qqq_state_date"], "2026-09-09")
        self.assertAlmostEqual(pop.loc["t1", "qqq_prev_close"], 600.0)
        self.assertEqual(pop.loc["t1", "qqq_prev_regime_label"], "HMM_BULL")
        self.assertTrue(pd.isna(pop.loc["stale", "qqq_prev_close"]))
        self.assertTrue(pd.isna(pop.loc["t1", "spy_prev_close"]))   # never held -> NULL

    def test_risk_state_from_snapshot_just_before_entry(self):
        lines = [
            {"timestamp": "2026-09-10T22:20:00.1", "tier": "PAUSE_NEW", "exposure_pct": 1.0,
             "total_assets": 1000.5},
            {"timestamp": "2026-09-10T22:39:50.1", "tier": "NORMAL", "exposure_pct": 0.9,
             "total_assets": 1000.5},
            {"timestamp": "2026-09-10T22:39:55.1", "tier": "EMERGENCY", "exposure_pct": 0.0,
             "total_assets": 1000000},                          # test-pollution row
            {"timestamp": "2026-09-10T22:40:30.1", "tier": "WARNING", "exposure_pct": 1.02,
             "total_assets": 1000.5},                          # after entry
        ]
        self.risk_log.write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n",
                                 encoding="utf-8")
        self._open("t1", ts="2026-09-10T22:40:00")
        self._open("far", ts="2026-09-10T23:30:00")
        pop = self._pop(execution=None)
        self.assertEqual(pop.loc["t1", "entry_risk_tier"], "NORMAL")
        self.assertAlmostEqual(pop.loc["t1", "entry_exposure_pct"], 0.9)
        self.assertTrue(pd.isna(pop.loc["far", "entry_risk_tier"]))   # > 15 min stale

    def test_missing_risk_log_gives_null(self):
        self._open("t1")
        pop = self._pop(execution=None)
        self.assertTrue(pd.isna(pop.loc["t1", "entry_risk_tier"]))


class ClassificationTest(_DbCase):
    def _build(self):
        self._open("real", execution="REAL")
        self._close("real")
        self._open("paper", ticker="JP.7203", execution="PAPER")
        self._close("paper")
        self._open("legacy_us", execution=None, ts="2026-07-07T00:28:32")
        self._close("legacy_us", "2026-07-08", "2026-07-09")
        self._open("null_late", execution=None, ts="2026-08-05T22:00:00")
        self._close("null_late", "2026-08-06", "2026-08-07")
        self._open("legacy_jp", ticker="JP.7203", execution=None, ts="2026-07-07T00:28:32")
        self._close("legacy_jp", "2026-07-08", "2026-07-09")
        self._open("phantom", execution=None, equity=50000.0, ts="2026-07-06T23:32:32")
        self._close("phantom", "2026-07-08", "2026-07-09")
        self._open("open_real", execution="REAL")

    def test_null_execution_is_classified_never_silently_dropped(self):
        self._build()
        pop = self._pop()
        self.assertEqual(len(pop), 7)
        self.assertEqual(pop.loc["legacy_us", "execution_class"], data.INFERRED_REAL)
        self.assertEqual(pop.loc["null_late", "execution_class"], data.UNKNOWN)
        self.assertEqual(pop.loc["legacy_jp", "execution_class"], data.UNKNOWN)
        self.assertTrue(pd.isna(pop.loc["legacy_us", "execution"]))   # raw value untouched
        self.assertEqual(pop.loc["legacy_us", "inclusion_reason"], "INCLUDED")
        self.assertEqual(pop.loc["null_late", "inclusion_reason"], "EXECUTION_UNKNOWN")
        self.assertEqual(pop.loc["legacy_jp", "inclusion_reason"], "EXECUTION_UNKNOWN")
        self.assertEqual(pop.loc["paper", "inclusion_reason"], "EXECUTION_NOT_SELECTED")
        for tid in pop.index:
            self.assertTrue(pop.loc[tid, "execution_class_reason"])

    def test_include_inferred_false_excludes_with_reason(self):
        self._build()
        pop = self._pop(include_inferred=False)
        self.assertEqual(pop.loc["legacy_us", "inclusion_reason"], "INFERRED_NOT_SELECTED")
        self.assertEqual(sorted(data.load_closed_trades(
            self.db, include_inferred=False, risk_log_path=self.risk_log)["trade_id"]), ["real"])
        self.assertEqual(sorted(data.load_closed_trades(
            self.db, risk_log_path=self.risk_log)["trade_id"]), ["legacy_us", "real"])

    def test_phantom_and_open_excluded_before_execution(self):
        self._build()
        pop = self._pop(execution=None)
        self.assertEqual(pop.loc["phantom", "inclusion_reason"], "PHANTOM")
        self.assertTrue(pop.loc["phantom", "is_phantom"])
        self.assertEqual(pop.loc["open_real", "inclusion_reason"], "OPEN")
        self.assertEqual(sorted(data.included(pop.reset_index())["trade_id"]),
                         ["legacy_jp", "legacy_us", "null_late", "paper", "real"])

    def test_paper_selection(self):
        self._build()
        self.assertEqual(list(data.load_closed_trades(
            self.db, execution="PAPER", risk_log_path=self.risk_log)["trade_id"]), ["paper"])

    def test_pre_fill_confirmation_flag(self):
        self._build()
        pop = self._pop(execution=None)
        self.assertTrue(pop.loc["legacy_us", "pre_fill_confirmation_fix"])
        self.assertFalse(pop.loc["real", "pre_fill_confirmation_fix"])


class QualityReportTest(_DbCase):
    def test_population_summary_and_fill_rates(self):
        self.tracker.log_run_metadata(run_id="run1", market="SIMULATE",
                                      parameter_hash="ph", git_commit="c")
        self._open("a", run_id="run1")
        self._close("a")
        self._open("b", execution=None, ts="2026-07-07T00:28:32")
        self._close("b", "2026-07-08", "2026-07-09")
        self._open("open1")
        self._open("ph", equity=50000.0)
        pop = data.load_population(self.db, risk_log_path=self.risk_log)
        rep = quality.build_quality_report(pop, run_id="r")
        p = rep["population"]
        self.assertEqual((p["total"], p["phantom"], p["open"], p["closed_non_phantom"],
                          p["analyzed"]), (4, 1, 1, 2, 2))
        self.assertEqual(p["by_execution_class"], {"REAL": 3, "INFERRED_REAL": 1})
        self.assertEqual(p["raw_execution"], {"REAL": 3, "NULL": 1})
        self.assertEqual(p["inclusion"], {"INCLUDED": 2, "OPEN": 1, "PHANTOM": 1})
        fields = {f["column"]: f for f in rep["fields"]}
        self.assertEqual(fields["parameter_hash"]["all"], {"filled": 1, "total": 4})
        self.assertEqual(fields["parameter_hash"]["analyzed"], {"filled": 1, "total": 2})
        self.assertEqual(fields["entry_price"]["all"], {"filled": 4, "total": 4})
        # exit-scope fields count closed trades only (a, b) — open1/ph are not "missing"
        self.assertEqual(fields["exit_category"]["all"], {"filled": 2, "total": 2})
        self.assertEqual(fields["commission"]["all"], {"filled": 0, "total": 2})
        md = quality.render_markdown(rep)
        self.assertIn("| parameter_hash | entry | 1/4 | 1/2 | 1/2 |", md)

    def test_empty_population_report(self):
        rep = quality.build_quality_report(pd.DataFrame(columns=data.ALL_COLUMNS))
        self.assertEqual(rep["population"]["total"], 0)
        self.assertTrue(all(f["all"]["total"] == 0 for f in rep["fields"]))
        quality.render_markdown(rep)


class MissingDataTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_db_and_missing_trades_table_return_empty(self):
        self.assertTrue(data.load_population(self.dir / "nope.db").empty)
        db = self.dir / "empty.db"
        sqlite3.connect(db).close()
        self.assertTrue(data.load_population(db).empty)
        self.assertTrue(data.load_dataset(db).empty)

    def test_minimal_legacy_schema_and_bad_json(self):
        db = self.dir / "minimal.db"
        con = sqlite3.connect(db)
        con.executescript("""
            CREATE TABLE trades (trade_id TEXT, ticker TEXT, entry_time TEXT,
                                 entry_price REAL, exit_time TEXT, equity_before REAL);
            INSERT INTO trades VALUES ('t1','US.AAA','2026-07-07T00:00:00',10,
                                       '2026-07-09T00:00:00', 1000);
            CREATE TABLE trade_attribution (trade_id TEXT, entry_context_json TEXT);
            INSERT INTO trade_attribution VALUES ('t1', '{not json');
        """)
        con.commit()
        con.close()
        pop = data.load_population(db, risk_log_path=self.dir / "none.jsonl")
        self.assertEqual(len(pop), 1)
        row = pop.iloc[0]
        self.assertTrue(pd.isna(row["execution"]))
        self.assertEqual(row["execution_class"], data.INFERRED_REAL)
        self.assertTrue(pd.isna(row["entry_adx"]))
        self.assertTrue(pd.isna(row["parameter_hash"]))
        self.assertTrue(set(data.ALL_COLUMNS) <= set(pop.columns))
        quality.build_quality_report(pop)


class RunnerIntegrationTest(_DbCase):
    def test_refresh_writes_quality_report_and_history(self):
        self._open("a")
        self._close("a")
        self._open("b", execution=None, ts="2026-07-07T00:28:32")
        self._close("b", "2026-07-08", "2026-07-09")
        research_db = self.dir / "ti.db"
        reports = self.dir / "reports"
        summary = refresh(source_db_path=self.db, research_db_path=research_db,
                          report_dir=reports, risk_log_path=self.risk_log)
        self.assertEqual(summary["n_trades_total"], 2)
        self.assertEqual(summary["n_trades_analyzed"], 2)
        self.assertTrue(Path(summary["data_quality_report_path"]).exists())
        self.assertTrue((reports / "data_quality_latest.json").exists())
        summary2 = refresh(source_db_path=self.db, research_db_path=research_db,
                           report_dir=reports, include_inferred=False,
                           risk_log_path=self.risk_log)
        self.assertEqual(summary2["n_trades_analyzed"], 1)
        self.assertEqual(summary2["inclusion"].get("INFERRED_NOT_SELECTED"), 1)
        store = TradeIntelligenceStore(db_path=research_db)
        try:
            self.assertEqual(len(store.data_quality_history()), 2)
        finally:
            store.close()

    def test_refresh_on_empty_source(self):
        empty = self.dir / "empty.db"
        sqlite3.connect(empty).close()
        summary = refresh(source_db_path=empty, research_db_path=self.dir / "ti.db",
                          report_dir=self.dir / "r", risk_log_path=self.risk_log)
        self.assertEqual((summary["n_trades_total"], summary["n_trades_analyzed"]), (0, 0))


class ImportGuardTest(unittest.TestCase):
    def test_research_data_layer_imports_no_trading_code(self):
        for name in ("trade_intelligence_data.py", "trade_intelligence_quality.py",
                     "trade_intelligence_runner.py", "trade_intelligence_store.py",
                     "early_failure_trajectory.py"):
            path = _REPO_ROOT / "research" / name
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for n in names:
                    for bad in _FORBIDDEN_IMPORT_PREFIXES:
                        self.assertFalse(n == bad or n.startswith(bad + "."),
                                         f"{name} imports {n!r}")


def setUpModule():
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
