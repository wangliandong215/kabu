"""
Unit tests for engine/exit_diagnostics.py.
Run:  python -m unittest engine.test_exit_diagnostics -v
"""
import unittest

from engine.exit_diagnostics import (
    classify_exit, EARLY_FAILURE, TREND_REVERSAL, EXTREME_GIVEBACK,
    PROFIT_GIVEBACK, OTHER,
)


class TestClassifyExit(unittest.TestCase):

    def test_missing_mfe_returns_none(self):
        self.assertIsNone(classify_exit(mfe=None, mae=-10.0, pnl=5.0, position_value=1000.0))

    def test_missing_pnl_returns_none(self):
        self.assertIsNone(classify_exit(mfe=50.0, mae=-10.0, pnl=None, position_value=1000.0))

    def test_missing_position_value_returns_none(self):
        self.assertIsNone(classify_exit(mfe=50.0, mae=-10.0, pnl=5.0, position_value=None))
        self.assertIsNone(classify_exit(mfe=50.0, mae=-10.0, pnl=5.0, position_value=0.0))

    def test_early_failure_small_peak_ends_negative(self):
        # mfe_pct 1% < 5% threshold, final pnl negative -> EARLY_FAILURE
        result = classify_exit(mfe=10.0, mae=-300.0, pnl=-296.0, position_value=1000.0)
        self.assertEqual(result["exit_category"], EARLY_FAILURE)
        self.assertAlmostEqual(result["mfe_pct"], 0.01)

    def test_other_small_peak_ends_nonnegative(self):
        # mfe_pct 1% < 5% threshold but pnl >= 0 -> OTHER, not EARLY_FAILURE
        result = classify_exit(mfe=10.0, mae=-5.0, pnl=2.0, position_value=1000.0)
        self.assertEqual(result["exit_category"], OTHER)

    def test_trend_reversal_real_move_then_net_loss(self):
        # mfe_pct 8% >= 5% threshold, but closed net negative -> TREND_REVERSAL
        result = classify_exit(mfe=80.0, mae=-50.0, pnl=-10.0, position_value=1000.0)
        self.assertEqual(result["exit_category"], TREND_REVERSAL)
        self.assertGreater(result["giveback_pct"], 1.0)   # gave back more than 100% of MFE

    def test_profit_giveback_normal_trend_shape(self):
        # PAYX-like: mfe_pct 20.9%, final +11.7% -> giveback ~44%, below 75% extreme bar
        result = classify_exit(mfe=36896.16, mae=-300.24, pnl=20724.90, position_value=176457.72,
                                holding_days=63.9)
        self.assertEqual(result["exit_category"], PROFIT_GIVEBACK)
        self.assertAlmostEqual(result["giveback_pct"], 0.438, places=3)
        self.assertAlmostEqual(result["mfe_capture"], 0.562, places=3)

    def test_extreme_giveback_nflx_like(self):
        # NFLX-like: mfe_pct 10.85%, final +1.96% -> giveback ~82%, >= 75% extreme bar
        result = classify_exit(mfe=9951.15, mae=-1794.87, pnl=1800.975, position_value=91721.52)
        self.assertEqual(result["exit_category"], EXTREME_GIVEBACK)
        self.assertGreater(result["giveback_pct"], 0.75)

    def test_extreme_giveback_requires_both_mfe_and_giveback_thresholds(self):
        # High giveback (80%) but mfe_pct only 6% (below the 10% extreme-tier bar)
        # -> stays PROFIT_GIVEBACK, not EXTREME_GIVEBACK.
        result = classify_exit(mfe=60.0, mae=-5.0, pnl=12.0, position_value=1000.0)
        self.assertEqual(result["exit_category"], PROFIT_GIVEBACK)

    def test_mfe_capture_full_when_no_giveback(self):
        result = classify_exit(mfe=100.0, mae=-5.0, pnl=100.0, position_value=1000.0)
        self.assertAlmostEqual(result["giveback_pct"], 0.0)
        self.assertAlmostEqual(result["mfe_capture"], 1.0)

    def test_negative_mfe_leaves_giveback_and_capture_none(self):
        # Price never went favorable at all -> giveback_pct/mfe_capture undefined.
        result = classify_exit(mfe=-20.0, mae=-300.0, pnl=-296.0, position_value=1000.0)
        self.assertIsNone(result["giveback_pct"])
        self.assertIsNone(result["mfe_capture"])
        self.assertEqual(result["exit_category"], EARLY_FAILURE)

    def test_mae_pct_computed_when_mae_present(self):
        result = classify_exit(mfe=10.0, mae=-50.0, pnl=-5.0, position_value=1000.0)
        self.assertAlmostEqual(result["mae_pct"], -0.05)

    def test_mae_none_leaves_mae_pct_none(self):
        result = classify_exit(mfe=10.0, mae=None, pnl=-5.0, position_value=1000.0)
        self.assertIsNone(result["mae_pct"])

    def test_flags_pass_through(self):
        result = classify_exit(mfe=10.0, mae=-5.0, pnl=-8.0, position_value=1000.0,
                                stop_loss_triggered=True, strategy_exit_triggered=False)
        self.assertTrue(result["stop_loss_triggered"])
        self.assertFalse(result["strategy_exit_triggered"])

    def test_holding_days_passthrough(self):
        result = classify_exit(mfe=10.0, mae=-5.0, pnl=2.0, position_value=1000.0,
                                holding_days=12.5)
        self.assertEqual(result["holding_days"], 12.5)

    def test_never_raises_on_bad_types(self):
        self.assertIsNone(classify_exit(mfe="not a number", mae=-5.0, pnl=2.0,
                                         position_value=1000.0))


if __name__ == "__main__":
    unittest.main()
