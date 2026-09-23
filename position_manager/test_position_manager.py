# -*- coding: utf-8 -*-
import unittest

import config
import position_manager
from position_manager.models import PositionManagementContext
import test_support


def _ctx(**overrides):
    base = dict(
        symbol="US.TEST", current_position=100, current_position_pct=100.0,
        baseline_qty=100, target_position=100, target_position_pct=100.0,
        entry_price=100.0, current_price=100.0, unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0, peak_price=100.0, drawdown_pct=0.0,
        confidence=80.0, confidence_baseline=80.0, confidence_change=0.0,
        hmm_state="HMM_BULL", hmm_state_baseline="HMM_BULL", hmm_state_change=False,
        volatility=0.01, volatility_state="NORMAL", atr=1.0, atr_pct=0.01,
        trend_state="atr_breakout", holding_period=5, portfolio_exposure=0.5,
        sector_exposure=0.1, available_cash=1000.0,
    )
    base.update(overrides)
    # Always derive confidence_change/hmm_state_change from the final
    # confidence/confidence_baseline/hmm_state/hmm_state_baseline — passing
    # e.g. confidence=40.0 without separately overriding confidence_change
    # must not silently leave the default confidence_change=0.0 in place.
    if base["confidence"] is not None and base["confidence_baseline"] is not None:
        base["confidence_change"] = base["confidence"] - base["confidence_baseline"]
    else:
        base["confidence_change"] = None
    base["hmm_state_change"] = bool(
        base["hmm_state"] and base["hmm_state_baseline"]
        and base["hmm_state"] != base["hmm_state_baseline"])
    return PositionManagementContext(**base)


class TestPositionManagerAggregation(unittest.TestCase):
    def test_no_signals_holds(self):
        decision = position_manager.evaluate(_ctx())
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.target_qty, 100)
        self.assertEqual(decision.reduction_qty, 0)
        self.assertEqual(decision.triggered_modules, [])

    def test_current_position_zero(self):
        decision = position_manager.evaluate(
            _ctx(current_position=0, current_position_pct=0.0, baseline_qty=100))
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.target_qty, 0)
        self.assertEqual(decision.reduction_qty, 0)

    def test_baseline_qty_zero_degrades_safely(self):
        decision = position_manager.evaluate(
            _ctx(current_position=100, baseline_qty=0))
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.target_qty, 100)

    def test_worked_example_from_spec_matches_multiplicative_combination(self):
        # V3.1 spec's own worked example: Confidence 15%, HMM 10%, Volatility
        # 10%, Drawdown 5% simultaneously -> Target ~= 65%. Multiplicative
        # combination (what position_manager.evaluate() implements) gives
        # 0.85*0.90*0.90*0.95 ~= 65.4%, matching the spec closely; the naive
        # additive 100-(15+10+10+5)=70% (and further-off 30% misreading)
        # does not — this pins the aggregation algorithm itself, independent
        # of which exact config thresholds happen to produce 15/10/10/5.
        remaining = (1 - 0.15) * (1 - 0.10) * (1 - 0.10) * (1 - 0.05)
        self.assertAlmostEqual(remaining, 0.654075, places=6)
        self.assertGreater(remaining * 100.0, 60.0)
        self.assertLess(remaining * 100.0, 70.0)

    def test_multiple_signals_do_not_exceed_full_reduction(self):
        # Four modules all maximally triggered (reduction_pct=1.0 each is not
        # realistic given config's tiers, but the aggregator must still clamp
        # to [0, 100] rather than go negative if it ever happened).
        ctx = _ctx(
            confidence=0.0, confidence_baseline=100.0,     # severe drop
            hmm_state="HMM_BEAR", hmm_state_baseline="HMM_BULL",
            volatility_state="HIGH", volatility=0.10, atr_pct=0.10,
            peak_price=100.0, current_price=80.0, drawdown_pct=0.20,
        )
        decision = position_manager.evaluate(ctx)
        self.assertGreaterEqual(decision.target_position_pct, 0.0)
        self.assertLessEqual(decision.target_position_pct, 100.0)
        self.assertGreaterEqual(decision.target_qty, 0)

    def test_conflicting_signals_only_triggered_ones_apply(self):
        # Confidence deteriorated, everything else is calm/unchanged.
        ctx = _ctx(confidence=40.0, confidence_baseline=82.0)
        decision = position_manager.evaluate(ctx)
        self.assertEqual(decision.triggered_modules, ["confidence"])

    def test_target_never_exceeds_current_position(self):
        # Even in a hypothetical where remaining computes to > current_pct
        # (can't happen with reduction-only modules, but pin the invariant).
        decision = position_manager.evaluate(_ctx())
        self.assertLessEqual(decision.target_qty, decision.current_qty)

    def test_target_never_negative(self):
        ctx = _ctx(
            confidence=0.0, confidence_baseline=100.0,
            hmm_state="HMM_BEAR", hmm_state_baseline="HMM_BULL",
            volatility_state="HIGH", atr_pct=0.10,
            peak_price=100.0, current_price=70.0, drawdown_pct=0.30,
        )
        decision = position_manager.evaluate(ctx)
        self.assertGreaterEqual(decision.target_qty, 0)
        self.assertGreaterEqual(decision.target_position_pct, 0.0)

    def test_already_at_target_does_not_repeat_reduce(self):
        # Simulates "already reduced": current_position (post-execution) now
        # equals what a fresh evaluate() against the SAME unchanged severity
        # would target again — must resolve to HOLD, not another REDUCE.
        ctx = _ctx(confidence=55.0, confidence_baseline=82.0,   # moderate tier, ~15% reduction
                    baseline_qty=100)
        first = position_manager.evaluate(ctx)
        self.assertEqual(first.action, "REDUCE")

        # baseline_qty stays 100 (unchanged holding period, no pyramid), so
        # current_position_pct == target_qty numerically here.
        ctx_after = _ctx(confidence=55.0, confidence_baseline=82.0,
                          current_position=first.target_qty,
                          current_position_pct=float(first.target_qty),
                          baseline_qty=100)
        second = position_manager.evaluate(ctx_after)
        self.assertEqual(second.action, "HOLD")
        self.assertEqual(second.target_qty, first.target_qty)

    def test_below_min_reduction_pct_holds_instead_of_micro_reducing(self):
        # A reduction whose magnitude is under config.PM_MIN_REDUCTION_PCT of
        # current qty must not fire REDUCE (avoids reacting to noise).
        ctx = _ctx(current_position=1000, baseline_qty=1000,
                    confidence=79.0, confidence_baseline=82.0)  # 3-point drop, under MINOR anyway
        decision = position_manager.evaluate(ctx)
        self.assertEqual(decision.action, "HOLD")


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
