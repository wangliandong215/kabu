# -*- coding: utf-8 -*-
import unittest

import config
from position_manager import volatility_reduction
from position_manager.models import (PositionManagementContext, VOLATILITY_ELEVATED,
                                      VOLATILITY_HIGH, VOLATILITY_NORMAL)


def _ctx(atr_pct):
    state = volatility_reduction.classify(atr_pct) if atr_pct is not None else VOLATILITY_NORMAL
    return PositionManagementContext(
        symbol="US.TEST", current_position=100, current_position_pct=100.0,
        baseline_qty=100, target_position=100, target_position_pct=100.0,
        entry_price=100.0, current_price=100.0, unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0, peak_price=100.0, drawdown_pct=0.0,
        confidence=None, confidence_baseline=None, confidence_change=None,
        hmm_state=None, hmm_state_baseline=None, hmm_state_change=False,
        volatility=atr_pct, volatility_state=state,
        atr=(atr_pct * 100.0) if atr_pct is not None else None, atr_pct=atr_pct,
        trend_state="atr_breakout", holding_period=5, portfolio_exposure=0.5,
        sector_exposure=0.1, available_cash=1000.0,
    )


class TestVolatilityReduction(unittest.TestCase):
    def test_classify_bands(self):
        self.assertEqual(volatility_reduction.classify(0.01), VOLATILITY_NORMAL)
        self.assertEqual(volatility_reduction.classify(config.VOL_ELEVATED_ATR_PCT), VOLATILITY_ELEVATED)
        self.assertEqual(volatility_reduction.classify(config.VOL_HIGH_ATR_PCT), VOLATILITY_HIGH)

    def test_missing_atr_no_opinion(self):
        sig = volatility_reduction.evaluate(_ctx(None))
        self.assertFalse(sig.triggered)

    def test_normal_no_trigger(self):
        sig = volatility_reduction.evaluate(_ctx(0.01))
        self.assertFalse(sig.triggered)

    def test_elevated_triggers_moderate_reduction(self):
        sig = volatility_reduction.evaluate(_ctx(0.035))
        self.assertTrue(sig.triggered)
        self.assertAlmostEqual(sig.reduction_pct, config.VOL_ELEVATED_REDUCTION)

    def test_high_triggers_larger_reduction(self):
        sig = volatility_reduction.evaluate(_ctx(0.06))
        self.assertTrue(sig.triggered)
        self.assertAlmostEqual(sig.reduction_pct, config.VOL_HIGH_REDUCTION)
        self.assertGreater(sig.reduction_pct, config.VOL_ELEVATED_REDUCTION)

    def test_never_hard_exits_on_high_vol(self):
        sig = volatility_reduction.evaluate(_ctx(0.20))  # extreme
        self.assertLess(sig.reduction_pct, 1.0)


if __name__ == "__main__":
    unittest.main()
