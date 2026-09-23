# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

from position_manager import state_store
import test_support


class TestStateStore(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_path = state_store._STORE_PATH
        state_store._STORE_PATH = Path(self._tmpdir.name) / "position_manager_state.json"

    def tearDown(self):
        state_store._STORE_PATH = self._orig_path
        self._tmpdir.cleanup()

    def test_get_or_init_creates_baseline(self):
        entry = state_store.get_or_init("US.TEST", "2026-01-01T00:00:00", 100.0, 50,
                                         105.0, 80.0, "HMM_BULL")
        self.assertEqual(entry["baseline_qty"], 50)
        self.assertEqual(entry["peak_price"], 105.0)
        self.assertEqual(entry["confidence_baseline"], 80.0)
        self.assertEqual(entry["hmm_state_baseline"], "HMM_BULL")

    def test_get_or_init_is_stable_across_calls_same_entry_time(self):
        state_store.get_or_init("US.TEST", "t1", 100.0, 50, 105.0, 80.0, "HMM_BULL")
        second = state_store.get_or_init("US.TEST", "t1", 999.0, 999, 999.0, 1.0, "HMM_BEAR")
        # Same entry_time -> existing baseline reused, NOT overwritten by
        # the (deliberately wildly different) second call's arguments.
        self.assertEqual(second["baseline_qty"], 50)
        self.assertEqual(second["confidence_baseline"], 80.0)

    def test_new_entry_time_resets_baseline(self):
        state_store.get_or_init("US.TEST", "t1", 100.0, 50, 105.0, 80.0, "HMM_BULL")
        state_store.update("US.TEST", peak_price=120.0, baseline_qty=70)
        # A new entry_time means the previous holding period closed and this
        # is a fresh one — must NOT inherit the stale peak_price/baseline_qty.
        fresh = state_store.get_or_init("US.TEST", "t2", 50.0, 20, 50.0, 60.0, "HMM_SIDEWAYS")
        self.assertEqual(fresh["baseline_qty"], 20)
        self.assertEqual(fresh["peak_price"], 50.0)

    def test_update_is_monotonic_max(self):
        state_store.get_or_init("US.TEST", "t1", 100.0, 50, 100.0, 80.0, "HMM_BULL")
        state_store.update("US.TEST", peak_price=110.0, baseline_qty=70)
        entry = state_store.update("US.TEST", peak_price=90.0, baseline_qty=60)
        # Neither field ever shrinks — a price dip or a PM-driven reduction
        # must not un-grow peak_price/baseline_qty.
        self.assertEqual(entry["peak_price"], 110.0)
        self.assertEqual(entry["baseline_qty"], 70)

    def test_update_on_unknown_symbol_is_noop(self):
        result = state_store.update("US.UNKNOWN", peak_price=100.0)
        self.assertEqual(result, {})


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
