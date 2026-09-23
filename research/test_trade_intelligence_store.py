"""
Unit tests for research/trade_intelligence_store.py against a throwaway
temp SQLite DB. Never touches the production trade_intelligence.db.

Run:  python -m unittest research.test_trade_intelligence_store -v
"""
import tempfile
import unittest
from pathlib import Path

from research.trade_intelligence_patterns import PatternCandidate
from research.trade_intelligence_store import TradeIntelligenceStore


def _make_candidate(state="OBSERVATION", metric_value=0.5, n_sample=10,
                     pattern_key="k1"):
    return PatternCandidate(
        analysis_type="EARLY_FAILURE", pattern_key=pattern_key, state=state,
        description="Association only, not shown to be causal.",
        metric_name="rate", metric_value=metric_value, n_sample=n_sample,
        n_baseline=20, precision=0.8, recall=0.6, false_positive_count=1,
        slice_definition={"a": 1}, source_trade_ids=["t1", "t2"],
    )


class TradeIntelligenceStoreTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "fake_trade_intelligence.db"
        self.store = TradeIntelligenceStore(db_path=self.db_path)

    def tearDown(self):
        self.store.close()
        self._tmpdir.cleanup()

    def test_upsert_then_reupsert_updates_row_in_place_not_duplicated(self):
        c1 = _make_candidate(state="OBSERVATION", metric_value=0.5, n_sample=10)
        self.store.upsert_pattern(c1, run_id="run1")
        c2 = _make_candidate(state="CANDIDATE_PATTERN", metric_value=0.9, n_sample=20)
        self.store.upsert_pattern(c2, run_id="run2")

        patterns = self.store.all_patterns()
        self.assertEqual(len(patterns), 1)
        row = patterns.iloc[0]
        self.assertEqual(row["state"], "CANDIDATE_PATTERN")
        self.assertAlmostEqual(row["metric_value"], 0.9)
        self.assertEqual(row["times_observed"], 2)
        self.assertEqual(row["last_refresh_run_id"], "run2")

    def test_pattern_history_gains_one_row_per_refresh(self):
        c1 = _make_candidate()
        self.store.upsert_pattern(c1, run_id="run1")
        self.store.upsert_pattern(c1, run_id="run2")
        self.store.upsert_pattern(c1, run_id="run3")

        hist = self.store.pattern_history("EARLY_FAILURE:k1")
        self.assertEqual(len(hist), 3)
        self.assertListEqual(list(hist["run_id"]), ["run1", "run2", "run3"])

    def test_record_run_persists_summary(self):
        self.store.record_run(
            run_id="run1", started_at="2026-01-01T00:00:00",
            finished_at="2026-01-01T00:01:00", source_db_path="C:/fake.db",
            n_trades_analyzed=42, n_patterns_total=3, n_candidate_patterns=1,
            n_observations=2, report_path="C:/report.md", status="OK",
        )
        import pandas as pd
        df = pd.read_sql_query("SELECT * FROM research_runs", self.store._conn)
        self.assertEqual(len(df), 1)
        self.assertEqual(int(df.iloc[0]["n_trades_analyzed"]), 42)
        self.assertEqual(df.iloc[0]["status"], "OK")

    def test_invalid_state_raises(self):
        candidate = _make_candidate()
        candidate.state = "PRODUCTION_RULE"  # bypass PatternCandidate's own guard
        with self.assertRaises(ValueError):
            self.store.upsert_pattern(candidate, run_id="run1")

    def test_all_patterns_filters_by_state(self):
        self.store.upsert_pattern(_make_candidate(state="OBSERVATION", pattern_key="obs1"), run_id="run1")
        self.store.upsert_pattern(_make_candidate(state="CANDIDATE_PATTERN", pattern_key="cand1"), run_id="run1")
        obs = self.store.all_patterns(state="OBSERVATION")
        cand = self.store.all_patterns(state="CANDIDATE_PATTERN")
        self.assertEqual(len(obs), 1)
        self.assertEqual(len(cand), 1)


if __name__ == "__main__":
    unittest.main()
