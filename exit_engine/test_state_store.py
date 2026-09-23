# -*- coding: utf-8 -*-
"""Unit tests for exit_engine/state_store.py — peak-price/peak-date
persistence. Mirrors position_manager/test_state_store.py's structure.

Run:  python -m unittest exit_engine.test_state_store -v
"""
import tempfile
import unittest
from pathlib import Path

from exit_engine import state_store


class StateStoreTestCase(unittest.TestCase):
    def setUp(self):
        self._orig_path = state_store._STORE_PATH
        self._tmpdir = tempfile.TemporaryDirectory()
        state_store._STORE_PATH = Path(self._tmpdir.name) / "exit_engine_state.json"

    def tearDown(self):
        state_store._STORE_PATH = self._orig_path
        self._tmpdir.cleanup()

    def test_get_or_init_creates_new_entry(self):
        entry = state_store.get_or_init("US.TEST", "2026-01-01T00:00:00", 100.0, 105.0, "2026-01-01")
        self.assertEqual(entry["peak_price"], 105.0)
        self.assertEqual(entry["peak_date"], "2026-01-01")

    def test_get_or_init_uses_entry_price_when_higher_than_current(self):
        entry = state_store.get_or_init("US.TEST", "2026-01-01T00:00:00", 110.0, 105.0, "2026-01-01")
        self.assertEqual(entry["peak_price"], 110.0)

    def test_update_raises_peak_price_and_peak_date_on_new_high(self):
        state_store.get_or_init("US.TEST", "2026-01-01T00:00:00", 100.0, 100.0, "2026-01-01")
        entry = state_store.update("US.TEST", price=120.0, today_iso="2026-01-03")
        self.assertEqual(entry["peak_price"], 120.0)
        self.assertEqual(entry["peak_date"], "2026-01-03")

    def test_update_does_not_lower_peak_on_price_dip(self):
        state_store.get_or_init("US.TEST", "2026-01-01T00:00:00", 100.0, 120.0, "2026-01-01")
        entry = state_store.update("US.TEST", price=90.0, today_iso="2026-01-05")
        self.assertEqual(entry["peak_price"], 120.0)
        self.assertEqual(entry["peak_date"], "2026-01-01")   # unchanged — no new high made

    def test_update_no_op_when_symbol_never_initialized(self):
        entry = state_store.update("US.NEVER_SEEN", price=100.0, today_iso="2026-01-01")
        self.assertEqual(entry, {})

    def test_get_or_init_resets_on_new_holding_period(self):
        state_store.get_or_init("US.TEST", "2026-01-01T00:00:00", 100.0, 150.0, "2026-01-01")
        # A different entry_time means the previous position closed and a
        # fresh one opened — peak must reset, not inherit the old high.
        entry = state_store.get_or_init("US.TEST", "2026-02-01T00:00:00", 50.0, 50.0, "2026-02-01")
        self.assertEqual(entry["peak_price"], 50.0)
        self.assertEqual(entry["peak_date"], "2026-02-01")

    def test_persists_across_reload(self):
        state_store.get_or_init("US.TEST", "2026-01-01T00:00:00", 100.0, 100.0, "2026-01-01")
        state_store.update("US.TEST", price=130.0, today_iso="2026-01-02")
        reloaded = state_store._load()
        self.assertEqual(reloaded["symbols"]["US.TEST"]["peak_price"], 130.0)

    def test_missing_store_file_returns_empty_symbols(self):
        data = state_store._load()
        self.assertEqual(data, {"symbols": {}})

    def test_corrupt_store_file_degrades_to_empty(self):
        state_store._STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state_store._STORE_PATH.write_text("not valid json{{{", encoding="utf-8")
        data = state_store._load()
        self.assertEqual(data, {"symbols": {}})


if __name__ == "__main__":
    unittest.main()
