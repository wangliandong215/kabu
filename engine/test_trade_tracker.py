"""
Unit tests for engine/trade_tracker.py.
Run:  python -m unittest engine.test_trade_tracker -v
"""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from engine import regime_store
from engine.trade_tracker import (TradeTracker, build_regime_ctx,
                                   build_regime_ctx_preferring_hmm,
                                   ingest_backtest_trade_log,
                                   normalize_exit_reason,
                                   EXIT_STOP_LOSS, EXIT_TRAILING,
                                   EXIT_TAKE_PROFIT, EXIT_STRATEGY_SIGNAL,
                                   EXIT_REGIME_BREAK, EXIT_REPLACED,
                                   EXIT_UNKNOWN)

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


class TestLogMarketContext(TradeTrackerTestCase):
    """v2.9 Market Context Logging snapshot table — see
    engine/market_context.py for the fetch/store side; this only covers
    trade_tracker.py's own log_market_context()."""

    def _open_trade(self, trade_id="US.AAPL_2026-08-07T09:30:00"):
        self.tracker.log_entry(
            trade_id=trade_id, ticker="US.AAPL", strategy_name="atr_breakout",
            strategy_version="2.9", direction="LONG", price=100.0, shares=10,
            position_value=1000.0, position_pct=0.1,
            cash=9000.0, equity=10000.0, timestamp="2026-08-07T09:30:00",
        )
        return trade_id

    def test_writes_full_snapshot(self):
        trade_id = self._open_trade()
        self.tracker.log_market_context(trade_id, {
            "observation_date": "2026-08-07",
            "vix_close": 18.7, "vix_date": "2026-08-06", "vix_source": "CBOE",
            "cnn_fear_greed": 63.0, "cnn_fear_greed_label": "greed",
            "cnn_fear_greed_date": "2026-08-07",
            "naaim_exposure": 82.5, "naaim_date": "2026-08-05",
            "aaii_bullish": None, "put_call_ratio": None,
            "data_status": {"vix": "OK", "cnn": "OK", "naaim": "OK",
                            "aaii": "NOT_IMPLEMENTED", "put_call": "NOT_IMPLEMENTED"},
        })
        row = self._raw("SELECT * FROM trade_market_context WHERE trade_id=?",
                         (trade_id,))[0]
        self.assertEqual(row["observation_date"], "2026-08-07")
        self.assertEqual(row["vix_close"], 18.7)
        self.assertEqual(row["cnn_fear_greed"], 63.0)
        self.assertEqual(row["naaim_exposure"], 82.5)
        self.assertIsNone(row["aaii_bullish"])
        self.assertIn("NOT_IMPLEMENTED", row["data_status"])

    def test_none_ctx_writes_all_null_row(self):
        trade_id = self._open_trade()
        self.tracker.log_market_context(trade_id, None)
        row = self._raw("SELECT * FROM trade_market_context WHERE trade_id=?",
                         (trade_id,))[0]
        self.assertIsNone(row["vix_close"])
        self.assertIsNone(row["observation_date"])

    def test_same_trade_id_overwrites_not_crashes(self):
        trade_id = self._open_trade()
        self.tracker.log_market_context(trade_id, {"vix_close": 15.0})
        self.tracker.log_market_context(trade_id, {"vix_close": 16.0})
        rows = self._raw("SELECT * FROM trade_market_context WHERE trade_id=?",
                          (trade_id,))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["vix_close"], 16.0)


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

    def test_daily_position_log_populates_trade_daily_and_trade_mfe_mae(self):
        trade_log = [
            {"date": pd.Timestamp("2026-01-01"), "code": "US.AAPL", "side": "BUY",
             "qty": 10, "price": 100.0, "pnl": None, "reason": "SIGNAL",
             "strategy": "atr_breakout"},
            {"date": pd.Timestamp("2026-01-03"), "code": "US.AAPL", "side": "SELL",
             "qty": 10, "price": 105.0, "pnl": 50.0, "reason": "TAKE_PROFIT",
             "strategy": "atr_breakout", "fee": 2.05},
        ]
        all_data = {"US.AAPL": _trend_df()}
        daily_position_log = [
            {"code": "US.AAPL", "date": pd.Timestamp("2026-01-01"), "close": 100.0,
             "high": 101.0, "low": 99.0, "atr": 2.0, "regime": None, "confidence": None,
             "entry_date": pd.Timestamp("2026-01-01"), "entry_price": 100.0, "shares": 10},
            {"code": "US.AAPL", "date": pd.Timestamp("2026-01-02"), "close": 108.0,
             "high": 112.0, "low": 104.0, "atr": 2.1, "regime": None, "confidence": None,
             "entry_date": pd.Timestamp("2026-01-01"), "entry_price": 100.0, "shares": 10},
        ]
        n = ingest_backtest_trade_log(self.tracker, trade_log, all_data,
                                       daily_position_log=daily_position_log)
        self.assertEqual(n, 1)
        daily_rows = self._raw("SELECT * FROM trade_daily ORDER BY date")
        self.assertEqual(len(daily_rows), 2)
        # day 2's high=112 is the best favorable excursion: (112-100)*10=120
        row = self._raw("SELECT * FROM trades")[0]
        self.assertAlmostEqual(row["mfe"], 120.0)
        self.assertAlmostEqual(row["mae"], -10.0)   # day1 low=99 -> (99-100)*10
        self.assertAlmostEqual(row["commission"], 2.05)

    def test_prefers_hmm_regime_fields_over_rule_based(self):
        trade_log = [
            {"date": pd.Timestamp("2026-01-01"), "code": "US.AAPL", "side": "BUY",
             "qty": 10, "price": 100.0, "pnl": None, "reason": "SIGNAL",
             "strategy": "atr_breakout", "regime_at_entry": "Bull",
             "confidence_at_entry": 85.0, "duration_at_entry": 12.0},
            {"date": pd.Timestamp("2026-01-03"), "code": "US.AAPL", "side": "SELL",
             "qty": 10, "price": 105.0, "pnl": 50.0, "reason": "TAKE_PROFIT",
             "strategy": "atr_breakout", "regime_at_exit": "Correction",
             "confidence_at_exit": 55.0, "duration_at_exit": 1.0},
        ]
        all_data = {"US.AAPL": _trend_df()}
        ingest_backtest_trade_log(self.tracker, trade_log, all_data)
        attr = self._raw("SELECT * FROM trade_attribution")[0]
        self.assertEqual(attr["entry_regime_label"], "HMM_BULL")
        self.assertEqual(attr["entry_confidence"], 85.0)
        self.assertIn('"source": "HMM"', attr["entry_context_json"])
        self.assertEqual(attr["exit_regime_label"], "HMM_CORRECTION")

    def test_run_id_threaded_into_ingested_trades(self):
        trade_log = [
            {"date": pd.Timestamp("2026-01-01"), "code": "US.AAPL", "side": "BUY",
             "qty": 10, "price": 100.0, "pnl": None, "reason": "SIGNAL",
             "strategy": "atr_breakout"},
        ]
        all_data = {"US.AAPL": _trend_df()}
        ingest_backtest_trade_log(self.tracker, trade_log, all_data, run_id="run-xyz")
        row = self._raw("SELECT * FROM trades")[0]
        self.assertEqual(row["run_id"], "run-xyz")


class TestNormalizeExitReason(unittest.TestCase):

    def test_known_reasons(self):
        self.assertEqual(normalize_exit_reason("STOP_LOSS"), EXIT_STOP_LOSS)
        self.assertEqual(normalize_exit_reason("ATR_TRAIL"), EXIT_TRAILING)
        self.assertEqual(normalize_exit_reason("TAKE_PROFIT"), EXIT_TAKE_PROFIT)
        self.assertEqual(normalize_exit_reason("STRAT_EXIT"), EXIT_STRATEGY_SIGNAL)
        self.assertEqual(normalize_exit_reason("STRATEGY_EXIT(boll)"),
                          EXIT_STRATEGY_SIGNAL)
        self.assertEqual(normalize_exit_reason("MA200_BREAK"), EXIT_REGIME_BREAK)
        self.assertEqual(normalize_exit_reason("ACTIVE_REPLACEMENT"), EXIT_REPLACED)

    def test_unknown_and_empty_fall_back_to_unknown(self):
        self.assertEqual(normalize_exit_reason("SOMETHING_NEW"), EXIT_UNKNOWN)
        self.assertEqual(normalize_exit_reason(""), EXIT_UNKNOWN)
        self.assertEqual(normalize_exit_reason(None), EXIT_UNKNOWN)


class TestLogExitDerivedFields(TradeTrackerTestCase):
    """exit_reason_code + mfe/mae are computed inside log_exit() itself,
    not passed by the caller — covered here rather than in TestLogExit
    above to keep that class focused on pnl/holding/regime-drift."""

    def _open(self):
        self.tracker.log_entry(
            trade_id="t1", ticker="US.AAPL", strategy_name="atr_breakout",
            strategy_version="2.6", direction="LONG", price=100.0, shares=10,
            position_value=1000.0, position_pct=0.1, cash=9000.0, equity=10000.0,
            timestamp="2026-01-01T00:00:00",
        )

    def test_exit_reason_code_derived_from_raw_reason(self):
        self._open()
        self.tracker.log_exit(
            trade_id="t1", price=95.0, cash=1.0, equity=1.0,
            timestamp="2026-01-02T00:00:00", exit_reason="STOP_LOSS",
        )
        row = self._raw("SELECT * FROM trades WHERE trade_id='t1'")[0]
        self.assertEqual(row["exit_reason_code"], EXIT_STOP_LOSS)

    def test_mfe_mae_null_when_no_trade_daily_rows(self):
        self._open()
        self.tracker.log_exit(
            trade_id="t1", price=105.0, cash=1.0, equity=1.0,
            timestamp="2026-01-02T00:00:00", exit_reason="TAKE_PROFIT",
        )
        row = self._raw("SELECT * FROM trades WHERE trade_id='t1'")[0]
        self.assertIsNone(row["mfe"])
        self.assertIsNone(row["mae"])

    def test_mfe_mae_aggregated_from_trade_daily(self):
        self._open()
        self.tracker.update_position_metrics(
            trade_id="t1", date="2026-01-01", close=103.0,
            entry_price=100.0, shares=10, high=106.0, low=98.0,
        )
        self.tracker.log_exit(
            trade_id="t1", price=104.0, cash=1.0, equity=1.0,
            timestamp="2026-01-02T00:00:00", exit_reason="TAKE_PROFIT",
        )
        row = self._raw("SELECT * FROM trades WHERE trade_id='t1'")[0]
        self.assertAlmostEqual(row["mfe"], 60.0)    # (106-100)*10
        self.assertAlmostEqual(row["mae"], -20.0)   # (98-100)*10

    def test_commission_stored(self):
        self._open()
        self.tracker.log_exit(
            trade_id="t1", price=105.0, cash=1.0, equity=1.0,
            timestamp="2026-01-02T00:00:00", exit_reason="TAKE_PROFIT",
            commission=3.14,
        )
        row = self._raw("SELECT * FROM trades WHERE trade_id='t1'")[0]
        self.assertAlmostEqual(row["commission"], 3.14)


class TestUpdatePositionMetrics(TradeTrackerTestCase):

    def _open(self, trade_id="t1", entry_price=100.0, shares=10):
        self.tracker.log_entry(
            trade_id=trade_id, ticker="US.AAPL", strategy_name="atr_breakout",
            strategy_version="2.6", direction="LONG", price=entry_price,
            shares=shares, position_value=entry_price * shares,
            position_pct=0.1, cash=9000.0, equity=10000.0,
            timestamp="2026-01-01T00:00:00",
        )

    def test_inserts_one_row_per_date(self):
        self._open()
        self.tracker.update_position_metrics(
            trade_id="t1", date="2026-01-01", close=101.0,
            entry_price=100.0, shares=10, high=102.0, low=99.0, atr=1.5,
        )
        rows = self._raw("SELECT * FROM trade_daily")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["close"], 101.0)
        self.assertAlmostEqual(row["floating_pnl"], 10.0)
        self.assertAlmostEqual(row["floating_pnl_pct"], 0.01)
        self.assertAlmostEqual(row["mfe"], 20.0)   # (102-100)*10
        self.assertAlmostEqual(row["mae"], -10.0)  # (99-100)*10

    def test_repeated_call_same_day_upserts_and_accumulates_extremes(self):
        self._open()
        self.tracker.update_position_metrics(
            trade_id="t1", date="2026-01-01", close=101.0,
            entry_price=100.0, shares=10, high=103.0, low=99.0,
        )
        # A second, later pass the same day: narrower high/low range, but a
        # different close — mfe/mae must stay at the first pass's extremes
        # (running high-water-mark), while close/floating_pnl reflect the
        # latest snapshot.
        self.tracker.update_position_metrics(
            trade_id="t1", date="2026-01-01", close=100.5,
            entry_price=100.0, shares=10, high=101.0, low=100.0,
        )
        rows = self._raw("SELECT * FROM trade_daily")
        self.assertEqual(len(rows), 1)   # still one row, not two
        row = rows[0]
        self.assertAlmostEqual(row["close"], 100.5)   # overwritten to latest
        self.assertAlmostEqual(row["mfe"], 30.0)      # (103-100)*10, preserved
        self.assertAlmostEqual(row["mae"], -10.0)     # (99-100)*10, preserved

    def test_different_dates_are_separate_rows(self):
        self._open()
        self.tracker.update_position_metrics(
            trade_id="t1", date="2026-01-01", close=101.0,
            entry_price=100.0, shares=10)
        self.tracker.update_position_metrics(
            trade_id="t1", date="2026-01-02", close=103.0,
            entry_price=100.0, shares=10)
        rows = self._raw("SELECT * FROM trade_daily ORDER BY date")
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["date"] for r in rows], ["2026-01-01", "2026-01-02"])

    def test_short_direction_sign_flips_mfe_mae(self):
        self._open()
        self.tracker.update_position_metrics(
            trade_id="t1", date="2026-01-01", close=95.0,
            entry_price=100.0, shares=10, direction="SHORT",
            high=102.0, low=90.0,
        )
        row = self._raw("SELECT * FROM trade_daily")[0]
        # SHORT: favorable direction is price falling -> use low for mfe
        self.assertAlmostEqual(row["mfe"], 100.0)   # (100-90)*10
        self.assertAlmostEqual(row["mae"], -20.0)   # (100-102)*10

    def test_unknown_trade_id_does_not_raise(self):
        # No log_entry() call first — must not raise (unlike log_exit).
        self.tracker.update_position_metrics(
            trade_id="never-opened", date="2026-01-01", close=10.0,
            entry_price=10.0, shares=1,
        )
        rows = self._raw("SELECT * FROM trade_daily")
        self.assertEqual(len(rows), 1)


class TestLogRunMetadata(TradeTrackerTestCase):

    def test_auto_generates_run_id_and_created_at(self):
        run_id = self.tracker.log_run_metadata(strategy_version="2.8")
        self.assertTrue(run_id)
        row = self._raw("SELECT * FROM metadata WHERE run_id=?", (run_id,))[0]
        self.assertEqual(row["strategy_version"], "2.8")
        self.assertIsNotNone(row["created_at"])

    def test_explicit_run_id_and_fields_stored(self):
        with mock.patch("engine.trade_tracker._detect_git_commit",
                         return_value="abc1234"):
            run_id = self.tracker.log_run_metadata(
                run_id="my-run", strategy_version="2.8", market="US",
                start_date="2026-01-01", end_date="2026-06-30",
                hmm_version="hmm-v1", parameter_hash="deadbeef",
            )
        self.assertEqual(run_id, "my-run")
        row = self._raw("SELECT * FROM metadata WHERE run_id='my-run'")[0]
        self.assertEqual(row["market"], "US")
        self.assertEqual(row["start_date"], "2026-01-01")
        self.assertEqual(row["hmm_version"], "hmm-v1")
        self.assertEqual(row["parameter_hash"], "deadbeef")
        self.assertEqual(row["git_commit"], "abc1234")

    def test_git_commit_detection_never_raises_on_failure(self):
        with mock.patch("engine.trade_tracker._detect_git_commit",
                         return_value=None):
            run_id = self.tracker.log_run_metadata()
        row = self._raw("SELECT * FROM metadata WHERE run_id=?", (run_id,))[0]
        self.assertIsNone(row["git_commit"])


class TestExportCsv(TradeTrackerTestCase):

    def test_rejects_unknown_table(self):
        with self.assertRaises(ValueError):
            self.tracker.export_csv("sqlite_master", Path(self._tmpdir.name) / "x.csv")

    def test_exports_trades_table_to_csv(self):
        self.tracker.log_entry(
            trade_id="t1", ticker="US.AAPL", strategy_name="boll",
            strategy_version="2.6", direction="LONG", price=100.0, shares=1,
            position_value=100.0, position_pct=0.01, cash=1.0, equity=1.0,
            timestamp="2026-01-01T00:00:00",
        )
        out_path = Path(self._tmpdir.name) / "trades.csv"
        self.tracker.export_csv("trades", out_path)
        with open(out_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "US.AAPL")


class TestBuildRegimeCtxPreferringHmm(unittest.TestCase):

    def test_uses_hmm_store_when_available(self):
        with mock.patch.object(regime_store, "get_regime_interface",
                                return_value={
                                    "current_regime": "Bull",
                                    "regime_confidence": 88.0,
                                    "regime_duration": 5,
                                    "regime_changed_today": False,
                                    "hmm_version": "hmm-v1",
                                }):
            ctx = build_regime_ctx_preferring_hmm("US.AAPL")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["regime_label"], "HMM_BULL")
        self.assertEqual(ctx["confidence"], 88.0)
        self.assertEqual(ctx["source"], "HMM")

    def test_falls_back_to_rule_based_when_store_empty(self):
        with mock.patch.object(regime_store, "get_regime_interface",
                                return_value={
                                    "current_regime": None, "regime_confidence": None,
                                    "regime_duration": 0, "regime_changed_today": False,
                                    "hmm_version": None,
                                }):
            ctx = build_regime_ctx_preferring_hmm(
                "US.AAPL", df=_trend_df(daily_return=0.02, vol_pct=0.001))
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["source"], "RULE")

    def test_returns_none_when_store_empty_and_no_df(self):
        with mock.patch.object(regime_store, "get_regime_interface",
                                return_value={
                                    "current_regime": None, "regime_confidence": None,
                                    "regime_duration": 0, "regime_changed_today": False,
                                    "hmm_version": None,
                                }):
            ctx = build_regime_ctx_preferring_hmm("US.AAPL", df=None)
        self.assertIsNone(ctx)


if __name__ == "__main__":
    unittest.main()
