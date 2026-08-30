"""
Unit tests for research/market_features_store.py — schema creation and
upsert round-trip for each table, against a throwaway on-disk DB (never
touches config.RESEARCH_MARKET_FEATURES_DB_PATH). Also covers the v2.11.1
time model: observed_at/market_date backfill migration, the macro_snapshots
primary-key rebuild (revision-safe versioning), latest_before()'s
Look-ahead guard, and the collection_log health table.

Run:  python -m unittest research.test_market_features_store -v
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from research.market_features_store import MarketFeaturesStore


class MarketFeaturesStoreTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "market_features_test.db"
        self.store = MarketFeaturesStore(db_path=self.db_path)

    def tearDown(self):
        self.store.close()
        self._tmpdir.cleanup()

    def test_news_snapshot_roundtrip(self):
        self.store.upsert("news_snapshots", {
            "code": "US.NVDA", "snapshot_date": "2026-08-30",
            "news_count_24h": 5, "major_news_flag": 1, "announcement_flag": 0,
            "created_at": "2026-08-30T00:00:00",
        })
        rows = self.store.all_rows("news_snapshots")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["code"], "US.NVDA")
        self.assertEqual(rows[0]["news_count_24h"], 5)

    def test_upsert_replaces_same_primary_key(self):
        base = {"code": "US.NVDA", "snapshot_date": "2026-08-30",
                "news_count_24h": 5, "major_news_flag": 0, "announcement_flag": 0,
                "created_at": "2026-08-30T00:00:00"}
        self.store.upsert("news_snapshots", base)
        self.store.upsert("news_snapshots", {**base, "news_count_24h": 9})
        rows = self.store.all_rows("news_snapshots")
        self.assertEqual(len(rows), 1, "same primary key must replace, not duplicate")
        self.assertEqual(rows[0]["news_count_24h"], 9)

    def test_macro_snapshot_roundtrip(self):
        self.store.upsert("macro_snapshots", {
            "region": "US", "indicator_id": "CPI", "category_name": "Inflation",
            "name": "CPI YoY", "data_time": "2026-08-01", "release_time": "2026-08-13",
            "value": 3.1, "predict_value": 3.0, "previous_value": 3.2,
            "unit_type": "%", "snapshot_date": "2026-08-30",
            "created_at": "2026-08-30T00:00:00",
        })
        rows = self.store.all_rows("macro_snapshots")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["value"], 3.1)

    def test_options_snapshot_roundtrip(self):
        self.store.upsert("options_snapshots", {
            "code": "US.AAPL", "snapshot_date": "2026-08-30",
            "iv": 28.5, "iv_rank": 40.0, "iv_percentile": 55.0, "hv_30d": 25.0,
            "call_volume": 1000, "put_volume": 800, "put_call_ratio": 0.8,
            "created_at": "2026-08-30T00:00:00",
        })
        rows = self.store.all_rows("options_snapshots")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["put_call_ratio"], 0.8)

    def test_fedwatch_and_indicator_shadow_tables_exist(self):
        self.store.upsert("fedwatch_target_rate_snapshots", {
            "snapshot_date": "2026-08-30", "meeting_date": "2026-09-17",
            "target_range": "4.00-4.25", "probability": 0.65,
            "created_at": "2026-08-30T00:00:00",
        })
        self.store.upsert("indicator_shadow_log", {
            "code": "US.AAPL", "snapshot_date": "2026-08-30", "indicator": "RSI14",
            "our_bucket": "OVERSOLD", "moomoo_bucket": "OVERSOLD", "match": 1,
            "created_at": "2026-08-30T00:00:00",
        })
        self.assertEqual(len(self.store.all_rows("fedwatch_target_rate_snapshots")), 1)
        self.assertEqual(len(self.store.all_rows("indicator_shadow_log")), 1)

    # ── v2.11.1 time model ──────────────────────────────────────────────────

    def test_new_rows_carry_observed_at_and_market_date(self):
        self.store.upsert("news_snapshots", {
            "code": "US.NVDA", "snapshot_date": "2026-08-30",
            "news_count_24h": 0, "major_news_flag": 0, "announcement_flag": 0,
            "created_at": "2026-08-30T07:15:00",
            "observed_at": "2026-08-29T22:15:00+00:00", "market_date": "2026-08-28",
        })
        row = self.store.all_rows("news_snapshots")[0]
        self.assertEqual(row["observed_at"], "2026-08-29T22:15:00+00:00")
        self.assertEqual(row["market_date"], "2026-08-28")

    def test_latest_before_excludes_rows_observed_after_cutoff(self):
        self.store.upsert("news_snapshots", {
            "code": "US.NVDA", "snapshot_date": "2026-08-17",
            "news_count_24h": 1, "major_news_flag": 0, "announcement_flag": 0,
            "created_at": "x", "observed_at": "2026-08-17T22:00:00+00:00",
            "market_date": "2026-08-17",
        })
        self.store.upsert("news_snapshots", {
            "code": "US.NVDA", "snapshot_date": "2026-08-18",
            "news_count_24h": 9, "major_news_flag": 1, "announcement_flag": 0,
            "created_at": "x", "observed_at": "2026-08-18T22:00:00+00:00",
            "market_date": "2026-08-18",
        })

        # A signal fired mid-day on 08-18, BEFORE that evening's snapshot
        # was collected — must only ever see the 08-17 row.
        as_of = "2026-08-18T05:00:00+00:00"
        row = self.store.latest_before("news_snapshots", {"code": "US.NVDA"}, as_of)
        self.assertIsNotNone(row)
        self.assertEqual(row["news_count_24h"], 1,
                          "must not see a snapshot observed after the as-of cutoff")

    def test_latest_before_returns_none_when_nothing_qualifies(self):
        self.store.upsert("news_snapshots", {
            "code": "US.NVDA", "snapshot_date": "2026-08-18",
            "news_count_24h": 1, "major_news_flag": 0, "announcement_flag": 0,
            "created_at": "x", "observed_at": "2026-08-18T22:00:00+00:00",
            "market_date": "2026-08-18",
        })
        row = self.store.latest_before(
            "news_snapshots", {"code": "US.NVDA"}, "2026-08-01T00:00:00+00:00")
        self.assertIsNone(row)

    def test_macro_snapshot_revision_does_not_overwrite_earlier_vintage(self):
        base = {"region": "US", "indicator_id": "CPI", "category_name": "Inflation",
                "name": "CPI YoY", "data_time": "2026-07-01", "release_time": "2026-08-13",
                "predict_value": 3.0, "previous_value": 3.2, "unit_type": "%",
                "snapshot_date": "2026-08-13", "created_at": "x"}
        self.store.upsert("macro_snapshots", {
            **base, "value": 3.1, "observed_at": "2026-08-13T00:00:00+00:00",
            "market_date": "2026-08-12",
        })
        # Same economic period (data_time), revised value, fetched later.
        self.store.upsert("macro_snapshots", {
            **base, "value": 3.4, "observed_at": "2026-09-15T00:00:00+00:00",
            "market_date": "2026-09-14",
        })

        rows = self.store.all_rows("macro_snapshots")
        self.assertEqual(len(rows), 2,
                          "each observed_at must be its own permanent vintage row")
        values = sorted(r["value"] for r in rows)
        self.assertEqual(values, [3.1, 3.4])

        # A trade that happened before the revision must still see 3.1, not 3.4.
        row = self.store.latest_before(
            "macro_snapshots",
            {"region": "US", "indicator_id": "CPI", "data_time": "2026-07-01"},
            "2026-08-20T00:00:00+00:00",
        )
        self.assertAlmostEqual(row["value"], 3.1)

    def test_migration_backfills_legacy_rows_from_created_at(self):
        """Simulate a pre-v2.11.1 DB (old macro_snapshots key, no
        observed_at/market_date columns anywhere) and confirm opening it
        with the current MarketFeaturesStore migrates in place without
        losing rows."""
        self.store.close()
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("DROP TABLE IF EXISTS macro_snapshots")
        conn.execute("""
            CREATE TABLE news_snapshots_legacy_test (x INTEGER)
        """)   # no-op marker table, just to prove this is the "old" file
        conn.execute("""
            CREATE TABLE macro_snapshots (
                region TEXT, indicator_id TEXT, category_name TEXT, name TEXT,
                data_time TEXT, release_time TEXT, value REAL, predict_value REAL,
                previous_value REAL, unit_type TEXT, snapshot_date TEXT,
                created_at TEXT,
                PRIMARY KEY (region, indicator_id, data_time)
            )
        """)
        # JST-naive created_at, as every pre-migration row actually looked.
        conn.execute(
            "INSERT INTO macro_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("US", "CPI", "Inflation", "CPI YoY", "2026-07-01", "2026-08-13",
             3.1, 3.0, 3.2, "%", "2026-08-18", "2026-08-18T07:15:00"),
        )
        conn.commit()
        conn.close()

        migrated = MarketFeaturesStore(db_path=self.db_path)
        try:
            rows = migrated.all_rows("macro_snapshots")
            self.assertEqual(len(rows), 1, "migration must not lose the old row")
            self.assertAlmostEqual(rows[0]["value"], 3.1)
            self.assertIsNotNone(rows[0]["observed_at"],
                                  "legacy row must be backfilled from created_at")
            self.assertEqual(rows[0]["market_date"], "2026-08-17",
                              "07:15 JST on 08-18 backfills to the prior session's date")
        finally:
            migrated.close()

    def test_migration_is_idempotent(self):
        """Opening an already-migrated DB a second time must not re-run the
        macro_snapshots PK rebuild or duplicate rows."""
        self.store.upsert("macro_snapshots", {
            "region": "US", "indicator_id": "CPI", "data_time": "2026-07-01",
            "observed_at": "2026-08-13T00:00:00+00:00", "value": 3.1,
            "market_date": "2026-08-12", "created_at": "x",
        })
        self.store.close()

        reopened = MarketFeaturesStore(db_path=self.db_path)
        try:
            self.assertEqual(len(reopened.all_rows("macro_snapshots")), 1)
        finally:
            reopened.close()

    # ── v2.11.1 Priority 2 — collection_log ──────────────────────────────────

    def test_log_collection_and_query(self):
        self.store.log_collection("2026-08-28", "news", "US.AAPL", "SUCCESS")
        self.store.log_collection("2026-08-28", "news", "US.BAD", "FAILED", detail="boom")
        self.store.log_collection("2026-08-28", "options", "US.AAPL", "SUCCESS")

        news_rows = self.store.collection_log_rows("2026-08-28", "news")
        self.assertEqual(len(news_rows), 2)
        statuses = {r["target"]: r["status"] for r in news_rows}
        self.assertEqual(statuses["US.AAPL"], "SUCCESS")
        self.assertEqual(statuses["US.BAD"], "FAILED")

        all_rows = self.store.collection_log_rows("2026-08-28")
        self.assertEqual(len(all_rows), 3)


if __name__ == "__main__":
    unittest.main()
