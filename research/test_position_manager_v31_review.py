# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

from research import position_manager_v31_review as review
import test_support


class TestLoadLog(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        self.assertEqual(review.load_log(Path("Z:\\does\\not\\exist.jsonl")), [])

    def test_corrupt_line_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log.jsonl"
            path.write_text(
                '{"symbol": "US.A", "final_decision": "HOLD"}\n'
                'not valid json\n'
                '{"symbol": "US.B", "final_decision": "REDUCE"}\n',
                encoding="utf-8")
            rows = review.load_log(path)
            self.assertEqual(len(rows), 2)

    def test_rows_sorted_by_timestamp(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log.jsonl"
            path.write_text(
                '{"timestamp": "2026-09-23T10:00:00", "symbol": "US.B"}\n'
                '{"timestamp": "2026-09-23T09:00:00", "symbol": "US.A"}\n',
                encoding="utf-8")
            rows = review.load_log(path)
            self.assertEqual([r["symbol"] for r in rows], ["US.A", "US.B"])


def _row(symbol="US.TEST", decision="HOLD", modules=None, current=100, reduce_amt=0,
         target=None):
    if target is None:
        target = current - reduce_amt
    return {
        "timestamp": "2026-09-23T10:00:00", "symbol": symbol,
        "final_decision": decision, "triggered_modules": modules or [],
        "current_position": current, "reduction_amount": reduce_amt,
        "target_position": target,
    }


class TestSummarize(unittest.TestCase):
    def test_empty_rows(self):
        stats = review.summarize([])
        self.assertEqual(stats["total_evaluations"], 0)
        self.assertIsNone(stats["avg_reduction_pct"])

    def test_hold_reduce_counts(self):
        rows = [_row(decision="HOLD"), _row(decision="REDUCE", modules=["drawdown"], reduce_amt=10)]
        stats = review.summarize(rows)
        self.assertEqual(stats["hold"], 1)
        self.assertEqual(stats["reduce"], 1)
        self.assertEqual(stats["drawdown_triggered"], 1)

    def test_conflicting_signals_counted_when_two_or_more_modules(self):
        rows = [_row(decision="REDUCE", modules=["confidence", "hmm"], reduce_amt=5)]
        stats = review.summarize(rows)
        self.assertEqual(stats["conflicting_signals"], 1)
        self.assertEqual(stats["confidence_triggered"], 1)
        self.assertEqual(stats["hmm_triggered"], 1)

    def test_repeated_reduce_detected_for_consecutive_same_symbol(self):
        rows = [
            _row(symbol="US.A", decision="REDUCE", modules=["drawdown"], reduce_amt=5),
            _row(symbol="US.A", decision="REDUCE", modules=["drawdown"], reduce_amt=5),
            _row(symbol="US.B", decision="REDUCE", modules=["drawdown"], reduce_amt=5),
        ]
        stats = review.summarize(rows)
        # US.A's second REDUCE counts as repeated; US.B's first REDUCE does not.
        self.assertEqual(stats["repeated_reduce"], 1)

    def test_hold_between_reduces_breaks_the_streak(self):
        rows = [
            _row(symbol="US.A", decision="REDUCE", modules=["drawdown"], reduce_amt=5),
            _row(symbol="US.A", decision="HOLD"),
            _row(symbol="US.A", decision="REDUCE", modules=["drawdown"], reduce_amt=5),
        ]
        stats = review.summarize(rows)
        self.assertEqual(stats["repeated_reduce"], 0)

    def test_reduction_pct_stats(self):
        rows = [
            _row(decision="REDUCE", modules=["drawdown"], current=100, reduce_amt=10),
            _row(decision="REDUCE", modules=["drawdown"], current=100, reduce_amt=20),
        ]
        stats = review.summarize(rows)
        self.assertAlmostEqual(stats["avg_reduction_pct"], 15.0)
        self.assertAlmostEqual(stats["max_reduction_pct"], 20.0)

    def test_invariant_violations_flagged(self):
        rows = [
            _row(decision="REDUCE", current=100, reduce_amt=10, target=-5),
            _row(decision="REDUCE", current=100, reduce_amt=0, target=150),
        ]
        stats = review.summarize(rows)
        self.assertEqual(stats["target_below_zero_count"], 1)
        self.assertEqual(stats["target_exceeds_current_count"], 1)


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
