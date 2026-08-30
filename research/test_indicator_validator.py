"""
Unit tests for research/indicator_validator.py — our_rsi_bucket() and
moomoo_rsi_buckets() are monkeypatched, so this suite never touches OpenD.

Run:  python -m unittest research.test_indicator_validator -v
"""
import tempfile
import unittest
from pathlib import Path

import research.indicator_validator as validator
from research.market_features_store import MarketFeaturesStore


class IndicatorValidatorTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = MarketFeaturesStore(db_path=Path(self._tmpdir.name) / "test.db")
        self._orig_our = validator.our_rsi_bucket
        self._orig_moomoo = validator.moomoo_rsi_buckets

    def tearDown(self):
        validator.our_rsi_bucket = self._orig_our
        validator.moomoo_rsi_buckets = self._orig_moomoo
        self.store.close()
        self._tmpdir.cleanup()

    def test_bucket_boundaries(self):
        self.assertEqual(validator._bucket(20.0), "OVERSOLD")
        self.assertEqual(validator._bucket(80.0), "OVERBOUGHT")
        self.assertEqual(validator._bucket(50.0), "NEUTRAL")
        self.assertEqual(validator._bucket(None), "UNKNOWN")

    def test_match_recorded_when_buckets_agree(self):
        validator.our_rsi_bucket = lambda code: ("OVERSOLD", 25.0)
        validator.moomoo_rsi_buckets = lambda market: ({"US.AAPL"}, set())

        stats = validator.run_validation(["US.AAPL"], self.store)

        self.assertEqual(stats["match"], 1)
        self.assertEqual(stats["mismatch"], 0)
        rows = self.store.all_rows("indicator_shadow_log")
        self.assertEqual(rows[0]["match"], 1)
        self.assertEqual(rows[0]["moomoo_bucket"], "OVERSOLD")

    def test_mismatch_recorded_when_buckets_disagree(self):
        validator.our_rsi_bucket = lambda code: ("OVERSOLD", 25.0)
        validator.moomoo_rsi_buckets = lambda market: (set(), set())   # moomoo says NEUTRAL

        stats = validator.run_validation(["US.AAPL"], self.store)

        self.assertEqual(stats["mismatch"], 1)
        self.assertEqual(stats["match"], 0)

    def test_unknown_skips_without_recording(self):
        validator.our_rsi_bucket = lambda code: ("UNKNOWN", None)
        validator.moomoo_rsi_buckets = lambda market: (set(), set())

        stats = validator.run_validation(["US.AAPL"], self.store)

        self.assertEqual(stats["unknown"], 1)
        self.assertEqual(len(self.store.all_rows("indicator_shadow_log")), 0)

    def test_a_failing_code_does_not_abort_the_batch(self):
        def _our(code):
            if code == "US.BAD":
                raise RuntimeError("kline fetch failed")
            return "NEUTRAL", 50.0
        validator.our_rsi_bucket = _our
        validator.moomoo_rsi_buckets = lambda market: (set(), set())

        stats = validator.run_validation(["US.BAD", "US.AAPL"], self.store)

        self.assertEqual(stats["match"], 1)


if __name__ == "__main__":
    unittest.main()
