# -*- coding: utf-8 -*-
import unittest

import config
from position_manager import hmm_reduction
from position_manager.models import PositionManagementContext
import test_support


def _ctx(hmm_state, hmm_state_baseline):
    return PositionManagementContext(
        symbol="US.TEST", current_position=100, current_position_pct=100.0,
        baseline_qty=100, target_position=100, target_position_pct=100.0,
        entry_price=100.0, current_price=100.0, unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0, peak_price=100.0, drawdown_pct=0.0,
        confidence=None, confidence_baseline=None, confidence_change=None,
        hmm_state=hmm_state, hmm_state_baseline=hmm_state_baseline,
        hmm_state_change=(hmm_state != hmm_state_baseline),
        volatility=None, volatility_state="NORMAL", atr=None, atr_pct=None,
        trend_state="atr_breakout", holding_period=5, portfolio_exposure=0.5,
        sector_exposure=0.1, available_cash=1000.0,
    )


class TestHmmReduction(unittest.TestCase):
    def test_missing_state_no_opinion(self):
        sig = hmm_reduction.evaluate(_ctx(None, None))
        self.assertFalse(sig.triggered)

    def test_unchanged_state_no_trigger(self):
        sig = hmm_reduction.evaluate(_ctx("HMM_BULL", "HMM_BULL"))
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.reduction_pct, 0.0)

    def test_bull_to_neutral_configured_downgrade(self):
        sig = hmm_reduction.evaluate(_ctx("HMM_SIDEWAYS", "HMM_BULL"))
        self.assertTrue(sig.triggered)
        self.assertAlmostEqual(sig.reduction_pct,
                                config.HMM_TRANSITION_REDUCTION[("HMM_BULL", "HMM_SIDEWAYS")])

    def test_bull_to_bear_larger_reduction_than_bull_to_sideways(self):
        sideways = hmm_reduction.evaluate(_ctx("HMM_SIDEWAYS", "HMM_BULL"))
        bear = hmm_reduction.evaluate(_ctx("HMM_BEAR", "HMM_BULL"))
        self.assertGreater(bear.reduction_pct, sideways.reduction_pct)

    def test_never_produces_exit_action(self):
        # This module only ever returns a ModuleSignal (reduction_pct), never
        # an EXIT-shaped decision — enforced structurally by ModuleSignal's
        # own shape, this test just documents/pins that contract.
        sig = hmm_reduction.evaluate(_ctx("HMM_BEAR", "HMM_BULL"))
        self.assertLess(sig.reduction_pct, 1.0)

    def test_recovery_transition_not_configured_no_trigger(self):
        sig = hmm_reduction.evaluate(_ctx("HMM_BULL", "HMM_BEAR"))
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.reduction_pct, 0.0)


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
