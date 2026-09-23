# -*- coding: utf-8 -*-
import unittest

import config
from position_manager import drawdown_reduction
from position_manager.models import PositionManagementContext
import test_support


def _ctx(peak_price, current_price):
    dd = max(0.0, (peak_price - current_price) / peak_price) if peak_price > 0 else 0.0
    return PositionManagementContext(
        symbol="US.TEST", current_position=100, current_position_pct=100.0,
        baseline_qty=100, target_position=100, target_position_pct=100.0,
        entry_price=peak_price, current_price=current_price, unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0, peak_price=peak_price, drawdown_pct=dd,
        confidence=None, confidence_baseline=None, confidence_change=None,
        hmm_state=None, hmm_state_baseline=None, hmm_state_change=False,
        volatility=None, volatility_state="NORMAL", atr=None, atr_pct=None,
        trend_state="atr_breakout", holding_period=5, portfolio_exposure=0.5,
        sector_exposure=0.1, available_cash=1000.0,
    )


class TestDrawdownReduction(unittest.TestCase):
    def test_no_drawdown_no_trigger(self):
        sig = drawdown_reduction.evaluate(_ctx(100.0, 100.0))
        self.assertFalse(sig.triggered)

    def test_below_tier1_no_trigger(self):
        sig = drawdown_reduction.evaluate(_ctx(100.0, 99.0))  # -1%
        self.assertFalse(sig.triggered)

    def test_tier1(self):
        sig = drawdown_reduction.evaluate(_ctx(100.0, 96.5))  # -3.5%
        self.assertTrue(sig.triggered)
        self.assertAlmostEqual(sig.reduction_pct, config.DRAWDOWN_TIER1_REDUCTION)

    def test_tier2_greater_than_tier1(self):
        t1 = drawdown_reduction.evaluate(_ctx(100.0, 96.5))   # -3.5%
        t2 = drawdown_reduction.evaluate(_ctx(100.0, 94.5))   # -5.5%
        self.assertGreater(t2.reduction_pct, t1.reduction_pct)

    def test_tier3_deepest(self):
        sig = drawdown_reduction.evaluate(_ctx(100.0, 90.0))  # -10%
        self.assertAlmostEqual(sig.reduction_pct, config.DRAWDOWN_TIER3_REDUCTION)

    def test_independent_of_stop_loss(self):
        # Confirms this module never returns an EXIT-shaped output even at a
        # deep drawdown — risk/guard.py's stop-loss is a completely separate
        # code path this module has no knowledge of.
        sig = drawdown_reduction.evaluate(_ctx(100.0, 80.0))  # -20%
        self.assertLess(sig.reduction_pct, 1.0)


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
