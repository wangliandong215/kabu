# -*- coding: utf-8 -*-
"""Integration tests for exit_engine/__init__.py — build_context ->
evaluate -> log_diagnostics round trip, missing data, market-open/close
adjacent timestamps, a multi-pass position-lifecycle scenario, and a
volume-anomaly fixture.

Run:  python -m unittest exit_engine.test_exit_engine -v
"""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import config
import exit_engine
from exit_engine import state_store
from exit_engine._fixtures import make_bars
import test_support


class ExitEngineTestCase(unittest.TestCase):
    def setUp(self):
        self._orig_store_path = state_store._STORE_PATH
        self._orig_log_path = exit_engine.EXIT_ENGINE_LOG_PATH
        self._tmpdir = tempfile.TemporaryDirectory()
        state_store._STORE_PATH = Path(self._tmpdir.name) / "exit_engine_state.json"
        exit_engine.EXIT_ENGINE_LOG_PATH = Path(self._tmpdir.name) / "exit_engine_v32_log.jsonl"

        self._orig_flags = {name: getattr(config, name) for name in (
            "ENABLE_VWAP_EXIT", "ENABLE_EMA_EXIT", "ENABLE_1M_TREND_EXIT",
            "ENABLE_5M_TREND_EXIT", "ENABLE_15M_TREND_EXIT",
            "ENABLE_VOLUME_EXHAUSTION", "ENABLE_HEAVY_SELLING",
            "ENABLE_OPENING_WEAKNESS", "ENABLE_AFTERNOON_FADE",
            "ENABLE_ATR_EXIT", "ENABLE_TRAILING_STOP", "ENABLE_BREAK_EVEN",
            "ENABLE_TIME_EXIT",
        )}

    def tearDown(self):
        state_store._STORE_PATH = self._orig_store_path
        exit_engine.EXIT_ENGINE_LOG_PATH = self._orig_log_path
        for name, value in self._orig_flags.items():
            setattr(config, name, value)
        self._tmpdir.cleanup()

    def _set_all_flags(self, value: bool):
        for name in self._orig_flags:
            setattr(config, name, value)

    def _pos(self, entry_price=100.0, qty=100, entry_time="2026-01-01T00:00:00", entry_atr=1.0):
        return {"entry_price": entry_price, "qty": qty, "entry_time": entry_time,
                "entry_atr": entry_atr, "strategy": "atr_breakout"}

    def _read_log(self):
        if not exit_engine.EXIT_ENGINE_LOG_PATH.exists():
            return []
        with open(exit_engine.EXIT_ENGINE_LOG_PATH, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # ── build_context ─────────────────────────────────────────────────
    def test_build_context_returns_none_on_nonpositive_price(self):
        ctx = exit_engine.build_context(
            code="US.TEST", pos=self._pos(), current_price=0.0,
            bars_1m=None, bars_5m=None, bars_15m=None, current_atr=1.0,
            trade_id="US.TEST_x", market_regime=1)
        self.assertIsNone(ctx)

    def test_build_context_computes_holding_days(self):
        now = datetime(2026, 1, 6, 0, 0, 0)
        ctx = exit_engine.build_context(
            code="US.TEST", pos=self._pos(entry_time="2026-01-01T00:00:00"),
            current_price=100.0, bars_1m=None, bars_5m=None, bars_15m=None,
            current_atr=1.0, trade_id="US.TEST_x", market_regime=1, now=now)
        self.assertAlmostEqual(ctx.holding_days, 5.0)

    def test_build_context_filters_session_bars_to_last_session_only(self):
        # Two days of 1m bars concatenated — session_bars_1m must keep only
        # the last day's rows.
        day1 = make_bars([100.0] * 5, start=datetime(2026, 1, 1, 9, 30))
        day2 = make_bars([101.0] * 3, start=datetime(2026, 1, 2, 9, 30))
        import pandas as pd
        bars = pd.concat([day1, day2], ignore_index=True)
        ctx = exit_engine.build_context(
            code="US.TEST", pos=self._pos(), current_price=101.0,
            bars_1m=bars, bars_5m=None, bars_15m=None, current_atr=1.0,
            trade_id="US.TEST_x", market_regime=1)
        self.assertEqual(len(ctx.session_bars_1m), 3)

    # ── full round trip ──────────────────────────────────────────────
    def test_full_round_trip_all_signals_off_yields_low_zero_score(self):
        self._set_all_flags(False)
        ctx = exit_engine.build_context(
            code="US.TEST", pos=self._pos(), current_price=99.0,
            bars_1m=make_bars([100.0] * 30), bars_5m=make_bars([100.0] * 30),
            bars_15m=make_bars([100.0] * 30), current_atr=1.0,
            trade_id="US.TEST_x", market_regime=1)
        result = exit_engine.evaluate(ctx)
        self.assertEqual(result.pressure_score, 0.0)
        self.assertEqual(result.pressure_tier, "LOW")
        self.assertIsNone(result.potential_exit_reason)
        exit_engine.log_diagnostics(ctx, result)
        rows = self._read_log()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "US.TEST")
        self.assertEqual(rows[0]["pressure_tier"], "LOW")

    def test_full_round_trip_all_signals_on_triggers_and_logs(self):
        self._set_all_flags(True)
        down = [130.0 - i for i in range(60)]
        session = make_bars(down[:30], start=datetime(2026, 1, 2, 9, 30))
        ctx = exit_engine.build_context(
            code="US.TEST", pos=self._pos(entry_price=140.0, entry_atr=0.5),
            current_price=down[-1],
            bars_1m=session, bars_5m=make_bars(down), bars_15m=make_bars(down),
            current_atr=3.0, trade_id="US.TEST_x", market_regime=1)
        result = exit_engine.evaluate(ctx)
        self.assertGreater(result.pressure_score, 0.0)
        self.assertIsNotNone(result.potential_exit_reason)
        exit_engine.log_diagnostics(ctx, result)
        rows = self._read_log()
        self.assertEqual(len(rows), 1)
        self.assertIn("signals", rows[0])
        self.assertEqual(set(rows[0]["signals"].keys()),
                          {"vwap", "ema", "trend", "volume", "time", "atr",
                           "trailing_stop", "break_even"})

    def test_one_signal_module_exception_does_not_break_evaluate(self):
        self._set_all_flags(True)
        ctx = exit_engine.build_context(
            code="US.TEST", pos=self._pos(), current_price=100.0,
            bars_1m="not-a-dataframe", bars_5m="not-a-dataframe",
            bars_15m="not-a-dataframe", current_atr=1.0,
            trade_id="US.TEST_x", market_regime=1)
        # Should not raise despite malformed bars input.
        result = exit_engine.evaluate(ctx)
        self.assertIsInstance(result.pressure_score, float)

    def test_log_diagnostics_failure_does_not_raise(self):
        # Point the log path at something that can never be created (a file
        # masquerading as a directory) to force a write failure.
        blocking_file = Path(self._tmpdir.name) / "blocking_file"
        blocking_file.write_text("x", encoding="utf-8")
        exit_engine.EXIT_ENGINE_LOG_PATH = blocking_file / "exit_engine_v32_log.jsonl"
        ctx = exit_engine.build_context(
            code="US.TEST", pos=self._pos(), current_price=100.0,
            bars_1m=None, bars_5m=None, bars_15m=None, current_atr=1.0,
            trade_id="US.TEST_x", market_regime=1)
        result = exit_engine.evaluate(ctx)
        exit_engine.log_diagnostics(ctx, result)   # must not raise

    # ── position lifecycle (multi-pass) ─────────────────────────────
    def test_position_lifecycle_logs_one_row_per_pass_without_mutating_pos(self):
        self._set_all_flags(False)
        config.ENABLE_ATR_EXIT = True
        pos = self._pos(entry_price=100.0, qty=100, entry_atr=1.0)
        pos_snapshot_before = dict(pos)

        prices = [101.0, 103.0, 99.0]   # open -> rally -> pull back
        for price in prices:
            ctx = exit_engine.build_context(
                code="US.TEST", pos=pos, current_price=price,
                bars_1m=None, bars_5m=None, bars_15m=None, current_atr=1.0,
                trade_id="US.TEST_x", market_regime=1)
            result = exit_engine.evaluate(ctx)
            exit_engine.log_diagnostics(ctx, result)

        rows = self._read_log()
        self.assertEqual(len(rows), len(prices))
        # peak_price in the log must reflect the running high-water mark,
        # not just the last pass's price.
        self.assertEqual(rows[-1]["peak_price"], 103.0)
        # This package must never mutate the caller's position dict.
        self.assertEqual(pos, pos_snapshot_before)

    # ── volume anomaly ───────────────────────────────────────────────
    def test_zero_volume_bars_do_not_crash_evaluate(self):
        self._set_all_flags(True)
        zero_vol = make_bars([100.0] * 30, volumes=[0.0] * 30)
        ctx = exit_engine.build_context(
            code="US.TEST", pos=self._pos(), current_price=99.0,
            bars_1m=zero_vol, bars_5m=zero_vol, bars_15m=zero_vol,
            current_atr=1.0, trade_id="US.TEST_x", market_regime=1)
        result = exit_engine.evaluate(ctx)
        self.assertIsInstance(result.pressure_score, float)


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
