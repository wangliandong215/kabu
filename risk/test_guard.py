"""
Unit tests for risk/guard.py -- focused on breakeven_lock_floor(), the
shared helper extracted 2026-07-06 so backtest_portfolio.py and
update_trailing_stop() (live/simulated) compute the breakeven lock the
same way instead of backtest silently missing it.

Run:  python -m unittest risk.test_guard -v
"""
import unittest

import config
from risk.guard import breakeven_lock_floor, update_trailing_stop


class TestBreakevenLockFloor(unittest.TestCase):

    def setUp(self):
        self._orig_trigger = config.ATR_BREAKEVEN_TRIGGER
        config.ATR_BREAKEVEN_TRIGGER = 1.0

    def tearDown(self):
        config.ATR_BREAKEVEN_TRIGGER = self._orig_trigger

    def test_not_locked_below_trigger(self):
        # entry=100, entry_atr=2 -> trigger at price > 102
        locked, floor = breakeven_lock_floor(entry=100.0, entry_atr=2.0,
                                              current_price=101.9, was_locked=False)
        self.assertFalse(locked)
        self.assertIsNone(floor)

    def test_locks_once_price_clears_trigger(self):
        locked, floor = breakeven_lock_floor(entry=100.0, entry_atr=2.0,
                                              current_price=102.1, was_locked=False)
        self.assertTrue(locked)
        self.assertEqual(floor, 100.0)

    def test_boundary_is_strict_greater_than(self):
        locked, floor = breakeven_lock_floor(entry=100.0, entry_atr=2.0,
                                              current_price=102.0, was_locked=False)
        self.assertFalse(locked)
        self.assertIsNone(floor)

    def test_stays_locked_even_if_price_falls_back(self):
        # Once locked, never unlocks -- price dropping back below the
        # trigger must not reset was_locked.
        locked, floor = breakeven_lock_floor(entry=100.0, entry_atr=2.0,
                                              current_price=100.5, was_locked=True)
        self.assertTrue(locked)
        self.assertEqual(floor, 100.0)


class TestUpdateTrailingStopBreakeven(unittest.TestCase):
    """确认 update_trailing_stop()（实盘/模拟盘路径）内部真的在用
    breakeven_lock_floor()，行为跟独立调用一致。"""

    def setUp(self):
        self._orig = {
            "ATR_MULT_BASE": config.ATR_MULT_BASE,
            "ATR_MULT_MID": config.ATR_MULT_MID,
            "ATR_MULT_TIGHT": config.ATR_MULT_TIGHT,
            "ATR_BREAKEVEN_TRIGGER": config.ATR_BREAKEVEN_TRIGGER,
        }
        config.ATR_MULT_BASE = 4.0
        config.ATR_MULT_MID = 4.0
        config.ATR_MULT_TIGHT = 4.0
        config.ATR_BREAKEVEN_TRIGGER = 1.0

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(config, k, v)

    def test_trail_floors_at_breakeven_once_triggered(self):
        pos = {"avg_cost": 100.0, "entry_atr": 2.0, "trail_stop": None,
               "breakeven_locked": False}
        # price=103 -> cleared entry+1*ATR(102); raw trail = 103-4*2=95,
        # but breakeven floor (100) should win.
        update_trailing_stop(pos, current_price=103.0, current_atr=2.0)
        self.assertTrue(pos["breakeven_locked"])
        self.assertEqual(pos["trail_stop"], 100.0)

    def test_trail_not_floored_before_trigger(self):
        pos = {"avg_cost": 100.0, "entry_atr": 2.0, "trail_stop": None,
               "breakeven_locked": False}
        update_trailing_stop(pos, current_price=101.0, current_atr=2.0)
        self.assertFalse(pos["breakeven_locked"])
        self.assertEqual(pos["trail_stop"], 101.0 - 4.0 * 2.0)


if __name__ == "__main__":
    unittest.main()
