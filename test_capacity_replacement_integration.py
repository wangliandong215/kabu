"""
test_capacity_replacement_integration.py -- v2.3 PRD Test 1 / Test 2:
integration-level verification of Active Replacement against the real
backtest_portfolio.py::simulate_from_prepared() execution path (not just the
pure portfolio/capacity_manager.py decision function -- see
portfolio/test_capacity_manager.py for that).

Builds a minimal hand-rolled `prepared` fixture (2 fake tickers, 2 trading
days, benchmark_df=None so QQQ-beta/macro-halt/weather all take their safe
no-op defaults) instead of fetching real market data, and monkeypatches
`bp.MAX_POSITIONS = 1` so a single held position is enough to trigger a
capacity review on day 2.

v2.4 note: every test here runs with `config.ENABLE_REPLACEMENT_STABILIZATION
= False` (set in setUp) -- this file's purpose is specifically pinning down
v2.3's original OBSERVATION-only behavior, so RSL's new LOW_PARTIAL/WEAK_FULL
tiers must stay out of the way here. See
test_replacement_stabilizer_integration.py for the v2.4-specific end-to-end
scenarios (LOW_PARTIAL eviction, cooldown, budget, new trade_log fields)
with RSL enabled.

Run:  python -m unittest test_capacity_replacement_integration -v
"""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd

import backtest_portfolio as bp
import config


def _simulate_quietly(prepared, **kwargs):
    """simulate_from_prepared() prints a full report to stdout unconditionally
    -- swallow it so test output stays readable alongside the other 48 tests."""
    with contextlib.redirect_stdout(io.StringIO()):
        return bp.simulate_from_prepared(prepared, **kwargs)

DAY0 = pd.Timestamp("2020-01-02")
DAY1 = pd.Timestamp("2020-01-03")


def _price_df(price: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [price, price], "high": [price, price],
         "low": [price, price], "close": [price, price], "volume": [1_000, 1_000]},
        index=[DAY0, DAY1],
    )


def _signal_df(day0_signal: str, day0_strength: float,
               day1_signal: str, day1_strength: float,
               strategy: str = "boll_mr") -> pd.DataFrame:
    return pd.DataFrame({
        "signal":   [day0_signal, day1_signal],
        "strength": [day0_strength, day1_strength],
        "atr":      [0.0, 0.0],
        "stop_pct": [0.05, 0.05],
        "strategy": [strategy, strategy],
        "regime":   ["RANGING", "RANGING"],
        "rsi14":    [50.0, 50.0],
    }, index=[DAY0, DAY1])


def _make_prepared(all_data, signals):
    return {
        "stocks": list(all_data.keys()), "start": str(DAY0.date()), "end": str(DAY1.date()),
        "all_data": all_data, "signals": signals,
        "benchmark_df": None,
        "qqq_above_ma200": pd.Series(dtype=bool),
        "qqq_macro_halt":  pd.Series(dtype=bool),
        "qqq_vol20":       pd.Series(dtype=float),
        "qqq_weather":     pd.Series(dtype=int),
        "qqq_vol_pct":     pd.Series(dtype=float),
        "all_dates": [DAY0, DAY1],
    }


class TestActiveReplacementIntegration(unittest.TestCase):

    def setUp(self):
        self._orig_max_positions = bp.MAX_POSITIONS
        self._orig_enable = config.ENABLE_ACTIVE_REPLACEMENT
        self._orig_advantage = config.REPLACEMENT_MARGIN
        self._orig_rsl = config.ENABLE_REPLACEMENT_STABILIZATION
        bp.MAX_POSITIONS = 1
        config.ENABLE_ACTIVE_REPLACEMENT = True
        config.REPLACEMENT_MARGIN = 10.0
        # v2.4 RSL默认True，但这个文件的测试目的是验证v2.3原始行为
        # （只允许换OBSERVATION）——每个测试按需自行决定是否开启RSL，
        # test_2需要显式关闭，见该测试内的注释。
        config.ENABLE_REPLACEMENT_STABILIZATION = False

    def tearDown(self):
        bp.MAX_POSITIONS = self._orig_max_positions
        config.ENABLE_ACTIVE_REPLACEMENT = self._orig_enable
        config.REPLACEMENT_MARGIN = self._orig_advantage
        config.ENABLE_REPLACEMENT_STABILIZATION = self._orig_rsl

    def test_1_observation_held_full_signal_triggers_active_replacement(self):
        # Day0: US.OBS gets a weak BUY (strength=0.2 -> total~46.7, OBSERVATION),
        #       the sole MAX_POSITIONS=1 slot is now taken.
        # Day1: US.FULL gets a strong BUY (strength=1.0 -> total=100, FULL).
        #       Capacity is full -> Active Replacement should close US.OBS and
        #       open US.FULL within the same tick.
        all_data = {"US.OBS": _price_df(10.0), "US.FULL": _price_df(20.0)}
        signals = {
            "US.OBS":  _signal_df("BUY", 0.2, "HOLD", 0.0),
            "US.FULL": _signal_df("HOLD", 0.0, "BUY", 1.0),
        }
        prepared = _make_prepared(all_data, signals)
        r = _simulate_quietly(prepared, cash=100_000.0, currency="$")

        sells = [t for t in r["trade_log"] if t["side"] == "SELL"]
        buys  = [t for t in r["trade_log"] if t["side"] == "BUY"]

        replacement_sells = [t for t in sells if t["reason"] == "ACTIVE_REPLACEMENT"]
        self.assertEqual(len(replacement_sells), 1, r["trade_log"])
        self.assertEqual(replacement_sells[0]["code"], "US.OBS")
        self.assertEqual(replacement_sells[0]["replacement_to"], "US.FULL")
        self.assertIsNotNone(replacement_sells[0]["replacement_score_delta"])
        self.assertGreater(replacement_sells[0]["replacement_score_delta"], 0)

        full_buys = [t for t in buys if t["code"] == "US.FULL"]
        self.assertEqual(len(full_buys), 1, r["trade_log"])
        self.assertEqual(full_buys[0]["date"], DAY1)
        self.assertEqual(full_buys[0]["replacement_from"], "US.OBS")

        self.assertEqual(len(r["replacements"]), 1)
        self.assertEqual(r["replacements"][0]["replacement_from"], "US.OBS")
        self.assertEqual(r["replacements"][0]["replacement_to"], "US.FULL")

    def test_2_no_observation_held_falls_back_to_capacity_block(self):
        # Day0: US.PARTIAL gets a mid-strength BUY (strength=0.5 -> total~66.7,
        #       PARTIAL, NOT OBSERVATION) -- takes the sole slot.
        # Day1: US.FULL gets a strong BUY (strength=1.0 -> FULL). No
        #       OBSERVATION exists to replace -> must fall back to the
        #       original Capacity Block behavior (no swap, no ACTIVE_REPLACEMENT).
        all_data = {"US.PARTIAL": _price_df(10.0), "US.FULL": _price_df(20.0)}
        signals = {
            "US.PARTIAL": _signal_df("BUY", 0.5, "HOLD", 0.0),
            "US.FULL":    _signal_df("HOLD", 0.0, "BUY", 1.0),
        }
        prepared = _make_prepared(all_data, signals)
        r = _simulate_quietly(prepared, cash=100_000.0, currency="$")

        reasons = [t["reason"] for t in r["trade_log"] if t["side"] == "SELL"]
        self.assertNotIn("ACTIVE_REPLACEMENT", reasons)
        self.assertEqual(len(r["replacements"]), 0)

        full_buys = [t for t in r["trade_log"] if t["side"] == "BUY" and t["code"] == "US.FULL"]
        self.assertEqual(len(full_buys), 0, r["trade_log"])   # FULL never got in

        full_blocks = [b for b in r["capacity_blocks"]
                       if any(c["code"] == "US.FULL" for c in b["blocked_candidates"])]
        self.assertGreaterEqual(len(full_blocks), 1)

        partial_buys = [t for t in r["trade_log"] if t["side"] == "BUY" and t["code"] == "US.PARTIAL"]
        self.assertEqual(len(partial_buys), 1)   # original position untouched

    def test_disabled_flag_falls_back_to_capacity_block_even_with_observation(self):
        config.ENABLE_ACTIVE_REPLACEMENT = False
        all_data = {"US.OBS": _price_df(10.0), "US.FULL": _price_df(20.0)}
        signals = {
            "US.OBS":  _signal_df("BUY", 0.2, "HOLD", 0.0),
            "US.FULL": _signal_df("HOLD", 0.0, "BUY", 1.0),
        }
        prepared = _make_prepared(all_data, signals)
        r = _simulate_quietly(prepared, cash=100_000.0, currency="$")

        self.assertEqual(len(r["replacements"]), 0)
        full_buys = [t for t in r["trade_log"] if t["side"] == "BUY" and t["code"] == "US.FULL"]
        self.assertEqual(len(full_buys), 0)


if __name__ == "__main__":
    unittest.main()
