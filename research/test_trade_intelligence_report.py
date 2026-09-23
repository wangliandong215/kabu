"""
Unit tests for research/trade_intelligence_report.py — report building and
writing to a throwaway temp directory. Never touches the production
reports dir.

Run:  python -m unittest research.test_trade_intelligence_report -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import test_support
from research.trade_intelligence_narrative import NarrativeProvider, TradeIntelligenceNarrative
from research.trade_intelligence_patterns import PatternCandidate
from research.trade_intelligence_report import build_report, write_report


class _RaisingProvider(NarrativeProvider):
    def generate(self, context):
        raise RuntimeError("boom")


class _StaticProvider(NarrativeProvider):
    def generate(self, context):
        return TradeIntelligenceNarrative(summary="static narrative text", model="test")


def _make_candidate(state="OBSERVATION"):
    return PatternCandidate(
        analysis_type="EARLY_FAILURE", pattern_key="k1", state=state,
        description="Association only, not shown to be causal.",
        metric_name="rate", metric_value=0.5, n_sample=10,
    )


class TradeIntelligenceReportTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.report_dir = Path(self._tmpdir.name) / "reports"
        self._alert_handlers = test_support.mute_alert_file_logging()

    def tearDown(self):
        test_support.unmute_alert_file_logging(self._alert_handlers)
        self._tmpdir.cleanup()

    def _run_summary(self):
        return {"run_id": "run1", "started_at": "t0", "finished_at": "t1",
                "n_trades_analyzed": 30}

    def test_markdown_contains_disclaimer_verbatim(self):
        payload = build_report([_make_candidate()], self._run_summary(), _StaticProvider())
        paths = write_report(payload, report_dir=self.report_dir)
        md_text = paths["run_md"].read_text(encoding="utf-8")
        self.assertIn(payload["disclaimer"], md_text)
        self.assertIn("static narrative text", md_text)

    def test_json_round_trips_same_pattern_count_as_input(self):
        candidates = [_make_candidate(), _make_candidate()]
        candidates[1].pattern_key = "k2"
        payload = build_report(candidates, self._run_summary(), _StaticProvider())
        paths = write_report(payload, report_dir=self.report_dir)
        loaded = json.loads(paths["run_json"].read_text(encoding="utf-8"))
        self.assertEqual(len(loaded["patterns"]), 2)

    def test_files_land_only_under_temp_report_dir(self):
        payload = build_report([_make_candidate()], self._run_summary(), _StaticProvider())
        paths = write_report(payload, report_dir=self.report_dir)
        for p in paths.values():
            self.assertTrue(str(p).startswith(str(self.report_dir)))

    def test_latest_files_overwritten_not_accumulated(self):
        payload1 = build_report([_make_candidate()], self._run_summary(), _StaticProvider())
        write_report(payload1, report_dir=self.report_dir)
        run_summary2 = dict(self._run_summary())
        run_summary2["run_id"] = "run2"
        payload2 = build_report([_make_candidate()], run_summary2, _StaticProvider())
        write_report(payload2, report_dir=self.report_dir)

        md_files = list(self.report_dir.glob("*_latest.md"))
        json_files = list(self.report_dir.glob("*_latest.json"))
        self.assertEqual(len(md_files), 1)
        self.assertEqual(len(json_files), 1)
        latest = json.loads(json_files[0].read_text(encoding="utf-8"))
        self.assertEqual(latest["run_summary"]["run_id"], "run2")

    def test_narrative_failure_is_caught_and_replaced_with_fallback(self):
        payload = build_report([_make_candidate()], self._run_summary(), _RaisingProvider())
        self.assertIn("unavailable", payload["narrative"].lower())
        # never blocks the patterns section
        self.assertEqual(len(payload["patterns"]), 1)


if __name__ == "__main__":
    unittest.main()
