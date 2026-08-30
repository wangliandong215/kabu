"""
Unit tests for engine/research_snapshot.py::build_trade_research_snapshot()
(v2.11.1 Priority 3).

Every dependency (engine.event_risk, research.market_features_store) is
monkeypatched or backed by a throwaway on-disk store — this suite asserts,
among other things, that the function NEVER calls anything that could hit
OpenD (data.news/data.options/data.macro), per its hard "read local DB only"
constraint.

Run:  python -m unittest engine.test_research_snapshot -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import engine.event_risk as event_risk
import engine.research_snapshot as research_snapshot
from research.market_features_store import MarketFeaturesStore


class _FakeCandidate:
    def __init__(self, current_price=100.0, signal_strength=0.9,
                 total_score=88.0, confidence=0.7):
        self.current_price = current_price
        self.signal_strength = signal_strength
        self.total_score = total_score
        self.confidence = confidence


class BuildTradeResearchSnapshotTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "test.db"

        self._orig = {
            "has_earnings_risk": event_risk.has_earnings_risk,
            "days_to_earnings": event_risk.days_to_earnings,
            "earnings_date": event_risk.earnings_date,
            "earnings_session": event_risk.earnings_session,
        }
        event_risk.has_earnings_risk = lambda code, trade_date=None: False
        event_risk.days_to_earnings = lambda code, trade_date=None: None
        event_risk.earnings_date = lambda code, trade_date=None: None
        event_risk.earnings_session = lambda code: None

        self._orig_store_path = research_snapshot.MarketFeaturesStore
        research_snapshot.MarketFeaturesStore = lambda: MarketFeaturesStore(db_path=self.db_path)

    def tearDown(self):
        event_risk.has_earnings_risk = self._orig["has_earnings_risk"]
        event_risk.days_to_earnings = self._orig["days_to_earnings"]
        event_risk.earnings_date = self._orig["earnings_date"]
        event_risk.earnings_session = self._orig["earnings_session"]
        research_snapshot.MarketFeaturesStore = self._orig_store_path
        self._tmpdir.cleanup()

    def test_pulls_strategy_indicators_from_result_and_cand(self):
        result = {"rsi14": 28.5, "atr": 3.5, "current_price": 100.0}
        cand = _FakeCandidate()

        snap = research_snapshot.build_trade_research_snapshot(
            "US.AAPL", result, cand, signal_time="2026-01-01T09:30:00")

        self.assertEqual(snap["signal_time"], "2026-01-01T09:30:00")
        self.assertAlmostEqual(snap["rsi14"], 28.5)
        self.assertAlmostEqual(snap["atr"], 3.5)
        self.assertAlmostEqual(snap["total_score"], 88.0)
        self.assertAlmostEqual(snap["confidence_score"], 0.7)
        self.assertIsNotNone(snap["observed_at"])
        self.assertIsNotNone(snap["market_date"])

    def test_event_risk_fields_populated(self):
        event_risk.has_earnings_risk = lambda code, trade_date=None: True
        event_risk.days_to_earnings = lambda code, trade_date=None: 1
        import datetime
        event_risk.earnings_date = lambda code, trade_date=None: datetime.date(2026, 1, 5)
        event_risk.earnings_session = lambda code: "AMC"

        snap = research_snapshot.build_trade_research_snapshot(
            "US.AAPL", {}, _FakeCandidate(), signal_time="2026-01-04T09:30:00")

        self.assertEqual(snap["has_earnings_risk"], 1)
        self.assertEqual(snap["days_to_earnings"], 1)
        self.assertEqual(snap["earnings_date"], "2026-01-05")
        self.assertEqual(snap["earnings_session"], "AMC")

    def test_event_risk_failure_leaves_fields_unset_not_raises(self):
        def _raise(code, trade_date=None):
            raise RuntimeError("boom")
        event_risk.has_earnings_risk = _raise

        snap = research_snapshot.build_trade_research_snapshot(
            "US.AAPL", {}, _FakeCandidate(), signal_time="2026-01-04T09:30:00")

        self.assertNotIn("has_earnings_risk", snap)

    def test_research_features_pulled_when_present(self):
        store = MarketFeaturesStore(db_path=self.db_path)
        store.upsert("news_snapshots", {
            "code": "US.AAPL", "snapshot_date": "2026-01-03",
            "news_count_24h": 4, "major_news_flag": 0, "announcement_flag": 0,
            "created_at": "x", "observed_at": "2026-01-04T00:00:00+00:00",
            "market_date": "2026-01-03",
        })
        store.upsert("options_snapshots", {
            "code": "US.AAPL", "snapshot_date": "2026-01-03",
            "iv": 30.0, "hv_30d": 25.0, "put_call_ratio": 0.7,
            "created_at": "x", "observed_at": "2026-01-04T00:00:00+00:00",
            "market_date": "2026-01-03",
        })
        store.upsert("fedwatch_target_rate_snapshots", {
            "snapshot_date": "2026-01-03", "meeting_date": "2026-02-01",
            "target_range": "4.00-4.25", "probability": 0.6,
            "created_at": "x", "observed_at": "2026-01-04T00:00:00+00:00",
            "market_date": "2026-01-03",
        })
        store.close()

        with mock.patch.object(research_snapshot.common, "utc_now_iso",
                                return_value="2026-01-04T12:00:00+00:00"):
            snap = research_snapshot.build_trade_research_snapshot(
                "US.AAPL", {}, _FakeCandidate(), signal_time="2026-01-04T09:30:00")

        self.assertEqual(snap["news_count_24h"], 4)
        self.assertAlmostEqual(snap["iv"], 30.0)
        self.assertAlmostEqual(snap["put_call_ratio"], 0.7)
        self.assertEqual(snap["fedwatch_target_range"], "4.00-4.25")

    def test_research_features_absent_when_no_row_exists(self):
        snap = research_snapshot.build_trade_research_snapshot(
            "US.NEVER_COLLECTED", {}, _FakeCandidate(), signal_time="2026-01-04T09:30:00")

        self.assertNotIn("news_count_24h", snap)
        self.assertNotIn("iv", snap)

    def test_research_features_never_look_ahead_past_signal_time(self):
        """A market_features row observed AFTER this snapshot's own
        observed_at must never be picked up."""
        store = MarketFeaturesStore(db_path=self.db_path)
        store.upsert("news_snapshots", {
            "code": "US.AAPL", "snapshot_date": "2026-01-05",
            "news_count_24h": 99, "major_news_flag": 1, "announcement_flag": 0,
            "created_at": "x", "observed_at": "2026-01-05T00:00:00+00:00",
            "market_date": "2026-01-04",
        })
        store.close()

        with mock.patch.object(research_snapshot.common, "utc_now_iso",
                                return_value="2026-01-04T12:00:00+00:00"):
            snap = research_snapshot.build_trade_research_snapshot(
                "US.AAPL", {}, _FakeCandidate(), signal_time="2026-01-04T09:30:00")

        self.assertNotIn("news_count_24h", snap,
                          "must not see a research row observed after this snapshot's own time")

    def test_module_never_references_opend_facing_fetchers(self):
        """Structural check (same source-inspection style as
        engine/test_runner_drawdown_halt.py's regression pin): this module
        must never directly import/call data.news, data.options, data.macro,
        or an OpenD connection primitive — it may only go through
        engine.event_risk's already-defined interface (whose own cache
        reuse is that module's responsibility, not this one's) and
        research.market_features_store's local-DB reads. This is what
        actually keeps this function from becoming a fresh OpenD call site
        as more Research Features get added later."""
        import inspect
        source = inspect.getsource(research_snapshot)
        for forbidden in ("data.news", "data.options", "data.macro",
                          "make_quote_ctx", "OpenQuoteContext", "import moomoo"):
            self.assertNotIn(forbidden, source,
                              f"{forbidden!r} must not appear in engine/research_snapshot.py")


if __name__ == "__main__":
    unittest.main()
