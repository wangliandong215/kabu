"""
Unit tests for engine/trade_tracker.py.
Run:  python -m unittest engine.test_trade_tracker -v
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from engine.trade_tracker import (TradeTracker, build_regime_ctx,
                                   ingest_backtest_trade_log)

N = 90  # comfortably past MRD's min-bars warmup gate (see test_market_regime.py)


def _trend_df(n=N, start=100.0, daily_return=0.01, vol_pct=0.002, volume=1000.0):
    close = start * (1 + daily_return) ** np.arange(n)
    half_range = close * vol_pct
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "high": close + half_range,
        "low": close - half_range,
        "close": close,
        "volume": volume,
    }, index=idx)


class TradeTrackerTestCase(unittest.TestCase):
    """Common setUp/tearDown: a fresh temp-file SQLite DB per test, matching
    the isolation style test_capacity_manager.py uses for config globals —
    here applied to a real file instead, since TradeTracker has no in-memory
    mode (sqlite3.connect(":memory:") would work too, but a real file
    exercises the same path production code takes)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "test_trade_history.db"
        self.tracker = TradeTracker(self.db_path)

    def tearDown(self):
        self.tracker.close()
        self._tmpdir.cleanup()

    def _raw(self, sql, params=()):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()


class TestLogEntry(TradeTrackerTestCase):

    def test_writes_trade_row_and_open_event(self):
        self.tracker.log_entry(
            trade_id="US.AAPL_2026-01-01T00:00:00", ticker="US.AAPL",
            strategy_name="atr_breakout", strategy_version="2.6",
            direction="LONG", price=100.0, shares=10,
            position_value=1000.0, position_pct=0.1,
            cash=9000.0, equity=10000.0, timestamp="2026-01-01T00:00:00",
            regime_ctx={"regime": 2, "regime_label": "LOW_VOL_TREND",
                       "confidence": 80.0, "features": {"adx": 30.0}},
        )
        rows = self._raw("SELECT * FROM trades")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["ticker"], "US.AAPL")
        self.assertEqual(row["entry_price"], 100.0)
        self.assertIsNone(row["exit_time"])   # not closed yet

        events = self._raw("SELECT * FROM trade_events WHERE event_type='OPEN'")
        self.assertEqual(len(events), 1)

        attr = self._raw("SELECT * FROM trade_attribution")[0]
        self.assertEqual(attr["entry_regime"], 2)
        self.assertEqual(attr["entry_regime_label"], "LOW_VOL_TREND")
        self.assertEqual(attr["entry_confidence"], 80.0)
        self.assertIn("adx", attr["entry_context_json"])

    def test_regime_ctx_none_leaves_attribution_fields_null(self):
        self.tracker.log_entry(
            trade_id="t1", ticker="US.AAPL", strategy_name="boll",
            strategy_version="2.6", direction="LONG", price=100.0, shares=1,
            position_value=100.0, position_pct=0.01, cash=1.0, equity=1.0,
            timestamp="2026-01-01T00:00:00", regime_ctx=None,
        )
        attr = self._raw("SELECT * FROM trade_attribution")[0]
        self.assertIsNone(attr["entry_regime"])
        self.assertIsNone(attr["entry_context_json"])

    def test_same_trade_id_overwrites_not_crashes(self):
        """Re-running the same backtest window should be idempotent, not
        raise a PRIMARY KEY violation."""
        for _ in range(2):
            self.tracker.log_entry(
                trade_id="dup", ticker="US.AAPL", strategy_name="boll",
                strategy_version="2.6", direction="LONG", price=100.0,
                shares=1, position_value=100.0, position_pct=0.01,
                cash=1.0, equity=1.0, timestamp="2026-01-01T00:00:00",
            )
        self.assertEqual(len(self._raw("SELECT * FROM trades")), 1)


class TestLogExit(TradeTrackerTestCase):

    def _open(self, direction="LONG", entry_price=100.0, shares=10,
              entry_time="2026-01-01T00:00:00", regime_ctx=None):
        self.tracker.log_entry(
            trade_id="t1", ticker="US.AAPL", strategy_name="atr_breakout",
            strategy_version="2.6", direction=direction, price=entry_price,
            shares=shares, position_value=entry_price * shares,
            position_pct=0.1, cash=9000.0, equity=10000.0,
            timestamp=entry_time, regime_ctx=regime_ctx,
        )

    def test_long_pnl_and_holding(self):
        self._open(entry_price=100.0, shares=10, entry_time="2026-01-01T00:00:00")
        self.tracker.log_exit(
            trade_id="t1", price=110.0, cash=10100.0, equity=11100.0,
            timestamp="2026-01-03T12:00:00", exit_reason="TAKE_PROFIT",
        )
        row = self._raw("SELECT * FROM trades WHERE trade_id='t1'")[0]
        self.assertAlmostEqual(row["pnl"], 100.0)          # (110-100)*10
        self.assertAlmostEqual(row["pnl_pct"], 0.10)
        self.assertAlmostEqual(row["holding_hours"], 60.0)  # 2.5 days
        self.assertAlmostEqual(row["holding_days"], 2.5)
        self.assertEqual(row["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(row["exit_price"], 110.0)

        events = self._raw("SELECT * FROM trade_events WHERE event_type='EXIT'")
        self.assertEqual(len(events), 1)

    def test_short_pnl_sign_flips(self):
        self._open(direction="SHORT", entry_price=100.0, shares=10)
        self.tracker.log_exit(
            trade_id="t1", price=90.0, cash=1.0, equity=1.0,
            timestamp="2026-01-02T00:00:00", exit_reason="TAKE_PROFIT",
        )
        row = self._raw("SELECT * FROM trades WHERE trade_id='t1'")[0]
        self.assertAlmostEqual(row["pnl"], 100.0)   # price fell 10 -> short profits

    def test_unknown_trade_id_raises(self):
        with self.assertRaises(ValueError):
            self.tracker.log_exit(
                trade_id="does-not-exist", price=1.0, cash=1.0, equity=1.0,
                timestamp="2026-01-01T00:00:00", exit_reason="STOP_LOSS",
            )

    def test_regime_drift_detected(self):
        self._open(regime_ctx={"regime": 2, "regime_label": "LOW_VOL_TREND",
                               "confidence": 80.0, "features": {}})
        self.tracker.log_exit(
            trade_id="t1", price=90.0, cash=1.0, equity=1.0,
            timestamp="2026-01-02T00:00:00", exit_reason="STOP_LOSS",
            regime_ctx={"regime": 1, "regime_label": "HIGH_VOL_SIDEWAYS",
                       "confidence": 60.0, "features": {}},
        )
        attr = self._raw("SELECT * FROM trade_attribution WHERE trade_id='t1'")[0]
        self.assertEqual(attr["regime_drifted"], 1)
        self.assertEqual(attr["exit_regime"], 1)
        self.assertEqual(attr["exit_regime_label"], "HIGH_VOL_SIDEWAYS")

    def test_no_drift_when_regime_unchanged(self):
        ctx = {"regime": 2, "regime_label": "LOW_VOL_TREND",
               "confidence": 80.0, "features": {}}
        self._open(regime_ctx=ctx)
        self.tracker.log_exit(
            trade_id="t1", price=105.0, cash=1.0, equity=1.0,
            timestamp="2026-01-02T00:00:00", exit_reason="TAKE_PROFIT",
            regime_ctx=ctx,
        )
        attr = self._raw("SELECT * FROM trade_attribution WHERE trade_id='t1'")[0]
        self.assertEqual(attr["regime_drifted"], 0)

    def test_drift_null_when_regime_unknown_on_either_side(self):
        self._open(regime_ctx=None)   # no entry regime captured
        self.tracker.log_exit(
            trade_id="t1", price=105.0, cash=1.0, equity=1.0,
            timestamp="2026-01-02T00:00:00", exit_reason="TAKE_PROFIT",
            regime_ctx={"regime": 1, "regime_label": "HIGH_VOL_SIDEWAYS",
                       "confidence": 60.0, "features": {}},
        )
        attr = self._raw("SELECT * FROM trade_attribution WHERE trade_id='t1'")[0]
        self.assertIsNone(attr["regime_drifted"])


class TestBuildRegimeCtx(unittest.TestCase):

    def test_valid_df_returns_dict_with_enum_name_label(self):
        ctx = build_regime_ctx(_trend_df(daily_return=0.02, vol_pct=0.001))
        self.assertIsNotNone(ctx)
        self.assertIn(ctx["regime_label"],
                      ("LOW_VOL_TREND", "HIGH_VOL_TREND"))   # a trending df
        self.assertIn("adx", ctx["features"])

    def test_none_df_returns_none(self):
        self.assertIsNone(build_regime_ctx(None))

    def test_empty_df_returns_none(self):
        self.assertIsNone(build_regime_ctx(pd.DataFrame()))


class TestIngestBacktestTradeLog(TradeTrackerTestCase):

    def test_pairs_buy_and_sell_ignores_promote(self):
        trade_log = [
            {"date": pd.Timestamp("2026-01-01"), "code": "US.AAPL", "side": "BUY",
             "qty": 10, "price": 100.0, "pnl": None, "reason": "SIGNAL",
             "strategy": "atr_breakout"},
            {"date": pd.Timestamp("2026-01-02"), "code": "US.AAPL", "side": "BUY",
             "qty": 4, "price": 101.0, "pnl": None, "reason": "PROMOTE",
             "strategy": "atr_breakout"},
            {"date": pd.Timestamp("2026-01-05"), "code": "US.AAPL", "side": "SELL",
             "qty": 14, "price": 110.0, "pnl": 100.0, "reason": "TAKE_PROFIT",
             "strategy": "atr_breakout"},
        ]
        all_data = {"US.AAPL": _trend_df()}
        n = ingest_backtest_trade_log(self.tracker, trade_log, all_data)
        self.assertEqual(n, 1)
        rows = self._raw("SELECT * FROM trades")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        # PROMOTE's qty must NOT be folded into the recorded trade's shares —
        # only the original SIGNAL open (qty=10) defines the round-trip.
        self.assertEqual(row["shares"], 10)
        self.assertEqual(row["exit_reason"], "TAKE_PROFIT")
        self.assertIsNotNone(row["exit_time"])

    def test_dangling_open_position_gets_log_entry_only(self):
        trade_log = [
            {"date": pd.Timestamp("2026-01-01"), "code": "US.AAPL", "side": "BUY",
             "qty": 10, "price": 100.0, "pnl": None, "reason": "SIGNAL",
             "strategy": "atr_breakout"},
        ]
        all_data = {"US.AAPL": _trend_df()}
        n = ingest_backtest_trade_log(self.tracker, trade_log, all_data)
        self.assertEqual(n, 1)
        row = self._raw("SELECT * FROM trades")[0]
        self.assertIsNone(row["exit_time"])

    def test_multiple_codes_independent_pairing(self):
        trade_log = [
            {"date": pd.Timestamp("2026-01-01"), "code": "US.AAPL", "side": "BUY",
             "qty": 10, "price": 100.0, "pnl": None, "reason": "SIGNAL",
             "strategy": "boll"},
            {"date": pd.Timestamp("2026-01-01"), "code": "US.QQQ", "side": "BUY",
             "qty": 5, "price": 400.0, "pnl": None, "reason": "QQQ_BETA_FLOOR",
             "strategy": "core_etf"},
            {"date": pd.Timestamp("2026-01-03"), "code": "US.AAPL", "side": "SELL",
             "qty": 10, "price": 105.0, "pnl": 50.0, "reason": "STRAT_EXIT",
             "strategy": "boll"},
            {"date": pd.Timestamp("2026-01-10"), "code": "US.QQQ", "side": "SELL",
             "qty": 5, "price": 380.0, "pnl": -100.0, "reason": "MA200_BREAK",
             "strategy": "core_etf"},
        ]
        all_data = {"US.AAPL": _trend_df(), "US.QQQ": _trend_df(start=400.0)}
        n = ingest_backtest_trade_log(self.tracker, trade_log, all_data)
        self.assertEqual(n, 2)
        tickers = {r["ticker"] for r in self._raw("SELECT * FROM trades")}
        self.assertEqual(tickers, {"US.AAPL", "US.QQQ"})


if __name__ == "__main__":
    unittest.main()
