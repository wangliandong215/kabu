# -*- coding: utf-8 -*-
import unittest

import config
from position_manager import confidence_reduction
from position_manager.models import PositionManagementContext


def _ctx(confidence, confidence_baseline):
    change = (confidence - confidence_baseline
              if confidence is not None and confidence_baseline is not None else None)
    return PositionManagementContext(
        symbol="US.TEST", current_position=100, current_position_pct=100.0,
        baseline_qty=100, target_position=100, target_position_pct=100.0,
        entry_price=100.0, current_price=100.0, unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0, peak_price=100.0, drawdown_pct=0.0,
        confidence=confidence, confidence_baseline=confidence_baseline,
        confidence_change=change, hmm_state=None, hmm_state_baseline=None,
        hmm_state_change=False, volatility=None, volatility_state="NORMAL",
        atr=None, atr_pct=None, trend_state="atr_breakout", holding_period=5,
        portfolio_exposure=0.5, sector_exposure=0.1, available_cash=1000.0,
    )


class TestConfidenceReduction(unittest.TestCase):
    def test_missing_baseline_no_opinion(self):
        sig = confidence_reduction.evaluate(_ctx(80.0, None))
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.reduction_pct, 0.0)

    def test_small_drift_no_trigger(self):
        # 0.82 -> 0.78 in spec's own 0-1 language == 82 -> 78 here (4 points, well under MINOR)
        sig = confidence_reduction.evaluate(_ctx(78.0, 82.0))
        self.assertFalse(sig.triggered)

    def test_sharp_deterioration_triggers_moderate(self):
        # 0.82 -> 0.55 == a 27-point drop: >= CONF_DROP_MODERATE(20), < CONF_DROP_SEVERE(30).
        sig = confidence_reduction.evaluate(_ctx(55.0, 82.0))
        self.assertTrue(sig.triggered)
        self.assertAlmostEqual(sig.reduction_pct, config.CONF_DROP_MODERATE_REDUCTION)

    def test_severe_tier(self):
        sig = confidence_reduction.evaluate(_ctx(40.0, 82.0))  # 42-point drop
        self.assertTrue(sig.triggered)
        self.assertAlmostEqual(sig.reduction_pct, config.CONF_DROP_SEVERE_REDUCTION)

    def test_improvement_never_triggers(self):
        sig = confidence_reduction.evaluate(_ctx(95.0, 82.0))
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.reduction_pct, 0.0)

    def test_mild_tier_between_minor_and_moderate(self):
        # drop of 15 (between MINOR=10 and MODERATE=20) -> partial reduction, less than MODERATE's full cut
        sig = confidence_reduction.evaluate(_ctx(67.0, 82.0))
        self.assertTrue(sig.triggered)
        self.assertLess(sig.reduction_pct, config.CONF_DROP_MODERATE_REDUCTION)
        self.assertGreater(sig.reduction_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
