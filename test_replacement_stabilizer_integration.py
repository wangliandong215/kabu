"""
test_replacement_stabilizer_integration.py -- v2.4 RSL end-to-end verification
against the real backtest_portfolio.py::simulate_from_prepared() execution
path (mirrors test_capacity_replacement_integration.py's fixture style, but
with config.ENABLE_REPLACEMENT_STABILIZATION left ON -- that file pins down
pure v2.3 behavior with RSL off, this one covers what v2.4 adds on top:
LOW_PARTIAL eviction, the master on/off switch, and Replacement Budget).

Cooldown / same-day-chain / Stability-Score-protection / adaptive-threshold
logic itself is exhaustively covered at the pure-function level in
portfolio/test_replacement_stabilizer.py -- these integration tests only
check that backtest_portfolio.py wires RSL's decisions through into
trade_log/replacements/capacity_blocks correctly end-to-end.

Run:  python -m unittest test_replacement_stabilizer_integration -v
"""
import contextlib
import io
import unittest

import pandas as pd

import backtest_portfolio as bp
import config
from portfolio.replacement_stabilizer import REPL_LOW_PARTIAL_EVICT, REPL_OBSERVATION_EVICT


def _simulate_quietly(prepared, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return bp.simulate_from_prepared(prepared, **kwargs)


DAY0 = pd.Timestamp("2020-01-02")
DAY1 = pd.Timestamp("2020-01-03")
DAY2 = pd.Timestamp("2020-01-06")


def _price_df(price: float, n_days: int) -> pd.DataFrame:
    idx = [DAY0, DAY1, DAY2][:n_days]
    return pd.DataFrame(
        {"open": [price] * n_days, "high": [price] * n_days,
         "low": [price] * n_days, "close": [price] * n_days,
         "volume": [1_000] * n_days},
        index=idx,
    )


def _signal_df(signals_strengths, strategy: str = "boll_mr") -> pd.DataFrame:
    """signals_strengths: list of (signal, strength) tuples, one per day."""
    n = len(signals_strengths)
    idx = [DAY0, DAY1, DAY2][:n]
    return pd.DataFrame({
        "signal":   [s for s, _ in signals_strengths],
        "strength": [v for _, v in signals_strengths],
        "atr":      [0.0] * n,
        "stop_pct": [0.05] * n,
        "strategy": [strategy] * n,
        "regime":   ["RANGING"] * n,
        "rsi14":    [50.0] * n,
    }, index=idx)


def _make_prepared(all_data, signals, n_days):
    dates = [DAY0, DAY1, DAY2][:n_days]
    return {
        "stocks": list(all_data.keys()), "start": str(dates[0].date()), "end": str(dates[-1].date()),
        "all_data": all_data, "signals": signals,
        "benchmark_df": None,
        "qqq_above_ma200": pd.Series(dtype=bool),
        "qqq_macro_halt":  pd.Series(dtype=bool),
        "qqq_vol20":       pd.Series(dtype=float),
        "qqq_weather":     pd.Series(dtype=int),
        "qqq_vol_pct":     pd.Series(dtype=float),
        "all_dates": dates,
    }


class TestReplacementStabilizerIntegration(unittest.TestCase):

    def setUp(self):
        self._orig_max_positions = bp.MAX_POSITIONS
        self._orig_enable = config.ENABLE_ACTIVE_REPLACEMENT
        self._orig_advantage = config.REPLACEMENT_MARGIN
        self._orig_rsl = config.ENABLE_REPLACEMENT_STABILIZATION
        self._orig_low_partial = config.REPLACEMENT_LOW_PARTIAL_THRESHOLD
        self._orig_budget = config.REPLACEMENT_BUDGET_PER_100_DAYS
        bp.MAX_POSITIONS = 1
        config.ENABLE_ACTIVE_REPLACEMENT = True
        config.REPLACEMENT_MARGIN = 10.0
        config.ENABLE_REPLACEMENT_STABILIZATION = True
        config.REPLACEMENT_LOW_PARTIAL_THRESHOLD = 70.0

    def tearDown(self):
        bp.MAX_POSITIONS = self._orig_max_positions
        config.ENABLE_ACTIVE_REPLACEMENT = self._orig_enable
        config.REPLACEMENT_MARGIN = self._orig_advantage
        config.ENABLE_REPLACEMENT_STABILIZATION = self._orig_rsl
        config.REPLACEMENT_LOW_PARTIAL_THRESHOLD = self._orig_low_partial
        config.REPLACEMENT_BUDGET_PER_100_DAYS = self._orig_budget

    def test_low_partial_evicted_when_no_observation_held(self):
        # Day0: US.LP gets a mid BUY (strength=0.5 -> total~66.7, PARTIAL,
        #       below REPLACEMENT_LOW_PARTIAL_THRESHOLD=70) -- takes the slot.
        # Day1: US.FULL gets a strong BUY (strength=1.0 -> FULL). No
        #       OBSERVATION exists, but v2.4's new LOW_PARTIAL tier should
        #       evict the weak PARTIAL instead of falling back to a block.
        all_data = {"US.LP": _price_df(10.0, 2), "US.FULL": _price_df(20.0, 2)}
        signals = {
            "US.LP":   _signal_df([("BUY", 0.5), ("HOLD", 0.0)]),
            "US.FULL": _signal_df([("HOLD", 0.0), ("BUY", 1.0)]),
        }
        prepared = _make_prepared(all_data, signals, 2)
        r = _simulate_quietly(prepared, cash=100_000.0, currency="$")

        sells = [t for t in r["trade_log"] if t["side"] == "SELL"]
        replacement_sells = [t for t in sells if t["reason"] == "ACTIVE_REPLACEMENT"]
        self.assertEqual(len(replacement_sells), 1, r["trade_log"])
        self.assertEqual(replacement_sells[0]["code"], "US.LP")
        self.assertEqual(replacement_sells[0]["replacement_type"], REPL_LOW_PARTIAL_EVICT)
        self.assertIsNotNone(replacement_sells[0]["replacement_stability_score"])

        self.assertEqual(len(r["replacements"]), 1)
        self.assertEqual(r["replacements"][0]["replacement_type"], REPL_LOW_PARTIAL_EVICT)

        full_buys = [t for t in r["trade_log"] if t["side"] == "BUY" and t["code"] == "US.FULL"]
        self.assertEqual(len(full_buys), 1, r["trade_log"])
        self.assertEqual(full_buys[0]["replacement_from"], "US.LP")

    def test_high_score_partial_not_evicted_falls_back_to_block(self):
        # strength=0.625 -> total=75.0, still PARTIAL but >= threshold(70) ->
        # not a LOW_PARTIAL candidate -> no eligible victim anywhere -> block.
        all_data = {"US.MIDPARTIAL": _price_df(10.0, 2), "US.FULL": _price_df(20.0, 2)}
        signals = {
            "US.MIDPARTIAL": _signal_df([("BUY", 0.625), ("HOLD", 0.0)]),
            "US.FULL":       _signal_df([("HOLD", 0.0), ("BUY", 1.0)]),
        }
        prepared = _make_prepared(all_data, signals, 2)
        r = _simulate_quietly(prepared, cash=100_000.0, currency="$")

        self.assertEqual(len(r["replacements"]), 0)
        full_buys = [t for t in r["trade_log"] if t["side"] == "BUY" and t["code"] == "US.FULL"]
        self.assertEqual(len(full_buys), 0, r["trade_log"])
        full_blocks = [b for b in r["capacity_blocks"]
                       if any(c["code"] == "US.FULL" for c in b["blocked_candidates"])]
        self.assertGreaterEqual(len(full_blocks), 1)

    def test_master_switch_off_reproduces_v23_even_with_low_partial_present(self):
        config.ENABLE_REPLACEMENT_STABILIZATION = False
        all_data = {"US.LP": _price_df(10.0, 2), "US.FULL": _price_df(20.0, 2)}
        signals = {
            "US.LP":   _signal_df([("BUY", 0.5), ("HOLD", 0.0)]),
            "US.FULL": _signal_df([("HOLD", 0.0), ("BUY", 1.0)]),
        }
        prepared = _make_prepared(all_data, signals, 2)
        r = _simulate_quietly(prepared, cash=100_000.0, currency="$")

        # With RSL off, only v2.3's OBSERVATION-only rule applies -- a PARTIAL
        # holding must NOT be evicted, exactly like pre-v2.4 behavior.
        self.assertEqual(len(r["replacements"]), 0)
        full_buys = [t for t in r["trade_log"] if t["side"] == "BUY" and t["code"] == "US.FULL"]
        self.assertEqual(len(full_buys), 0, r["trade_log"])

    def test_replacement_budget_blocks_further_eviction_within_window(self):
        # Two OBSERVATION slots filled on Day0 (MAX_POSITIONS=2). Day1: a FULL
        # signal replaces the weaker OBSERVATION (1st replacement -> budget
        # counter now at 1). Day2: another FULL signal targets the remaining
        # OBSERVATION holding -- with REPLACEMENT_BUDGET_PER_100_DAYS pinned
        # to 1, this second, otherwise-perfectly-eligible replacement must be
        # refused purely on budget grounds.
        bp.MAX_POSITIONS = 2
        config.REPLACEMENT_BUDGET_PER_100_DAYS = 1
        all_data = {
            "US.OBS_A":  _price_df(10.0, 3),
            "US.OBS_B":  _price_df(11.0, 3),
            "US.FULL_1": _price_df(20.0, 3),
            "US.FULL_2": _price_df(21.0, 3),
        }
        signals = {
            "US.OBS_A":  _signal_df([("BUY", 0.20), ("HOLD", 0.0), ("HOLD", 0.0)]),
            "US.OBS_B":  _signal_df([("BUY", 0.25), ("HOLD", 0.0), ("HOLD", 0.0)]),
            "US.FULL_1": _signal_df([("HOLD", 0.0), ("BUY", 1.0), ("HOLD", 0.0)]),
            "US.FULL_2": _signal_df([("HOLD", 0.0), ("HOLD", 0.0), ("BUY", 1.0)]),
        }
        prepared = _make_prepared(all_data, signals, 3)
        r = _simulate_quietly(prepared, cash=200_000.0, currency="$")

        self.assertEqual(len(r["replacements"]), 1, r["replacements"])
        self.assertEqual(r["replacements"][0]["replacement_from"], "US.OBS_A")  # lower score evicted first

        full2_buys = [t for t in r["trade_log"] if t["side"] == "BUY" and t["code"] == "US.FULL_2"]
        self.assertEqual(len(full2_buys), 0, r["trade_log"])  # blocked by budget, not bought

        blocked = [b for b in r["capacity_blocks"]
                   if any(c["code"] == "US.FULL_2" for c in b["blocked_candidates"])]
        self.assertGreaterEqual(len(blocked), 1)
        self.assertTrue(any(b.get("budget_blocked") for b in blocked))

        # US.OBS_B must have survived untouched (never replaced).
        obs_b_sells = [t for t in r["trade_log"]
                       if t["code"] == "US.OBS_B" and t["reason"] == "ACTIVE_REPLACEMENT"]
        self.assertEqual(len(obs_b_sells), 0)

    def test_observation_evict_logs_type_and_null_stability_score(self):
        all_data = {"US.OBS": _price_df(10.0, 2), "US.FULL": _price_df(20.0, 2)}
        signals = {
            "US.OBS":  _signal_df([("BUY", 0.2), ("HOLD", 0.0)]),
            "US.FULL": _signal_df([("HOLD", 0.0), ("BUY", 1.0)]),
        }
        prepared = _make_prepared(all_data, signals, 2)
        r = _simulate_quietly(prepared, cash=100_000.0, currency="$")

        sells = [t for t in r["trade_log"] if t["reason"] == "ACTIVE_REPLACEMENT"]
        self.assertEqual(len(sells), 1, r["trade_log"])
        self.assertEqual(sells[0]["replacement_type"], REPL_OBSERVATION_EVICT)
        # Tier 1 (OBSERVATION) doesn't use Stability Score protection -- v2.3's
        # original logic is untouched, so this field stays None for this tier.
        self.assertIsNone(sells[0]["replacement_stability_score"])


if __name__ == "__main__":
    unittest.main()
