"""
Unit tests for risk/qqq_core_recovery.py -- v2.13 QQQ Core Recovery / Reclaim
(Phase 2.1, Observation Mode). Pure-function tests only; engine/runner.py's
integration (2c NEW_ENTRY gate, macro_block skip, exception behavior) is
covered separately by engine/test_qqq_core_recovery_gate.py.

Run:  python -m unittest risk.test_qqq_core_recovery -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import config
from risk import qqq_core_recovery as qcr
from risk.portfolio_risk_manager import RebalanceOrder


class ComputeShortfallTestCase(unittest.TestCase):

    def setUp(self):
        self._orig_target = config.QQQ_CORE_TARGET_PCT
        config.QQQ_CORE_TARGET_PCT = 0.25

    def tearDown(self):
        config.QQQ_CORE_TARGET_PCT = self._orig_target

    def test_normal_shortfall(self):
        # target=25% of 1,000,000 = 250,000; held 100,000 -> shortfall 150,000
        self.assertEqual(qcr.compute_shortfall(100_000.0, 1_000_000.0), 150_000.0)

    def test_zero_when_at_target(self):
        self.assertEqual(qcr.compute_shortfall(250_000.0, 1_000_000.0), 0.0)

    def test_floors_at_zero_when_over_target(self):
        self.assertEqual(qcr.compute_shortfall(400_000.0, 1_000_000.0), 0.0)

    def test_none_total_assets(self):
        self.assertEqual(qcr.compute_shortfall(100_000.0, None), 0.0)

    def test_none_qqq_value_treated_as_zero_held(self):
        self.assertEqual(qcr.compute_shortfall(None, 1_000_000.0), 250_000.0)


class AdvanceStateTestCase(unittest.TestCase):

    def setUp(self):
        self._orig = (config.PORTFOLIO_REBALANCE_MIN_TRADE_USD,
                      config.QQQ_CORE_RECOVERY_TRIGGER_PASSES,
                      config.QQQ_CORE_RECOVERY_LEVEL3_TRIGGER_PASSES,
                      config.QQQ_CORE_RECOVERY_LEVEL3_ORDINARY_EXPOSURE_MIN_PCT)
        config.PORTFOLIO_REBALANCE_MIN_TRADE_USD = 1000.0
        config.QQQ_CORE_RECOVERY_TRIGGER_PASSES = 3
        config.QQQ_CORE_RECOVERY_LEVEL3_TRIGGER_PASSES = 5
        config.QQQ_CORE_RECOVERY_LEVEL3_ORDINARY_EXPOSURE_MIN_PCT = 0.60

    def tearDown(self):
        (config.PORTFOLIO_REBALANCE_MIN_TRADE_USD,
         config.QQQ_CORE_RECOVERY_TRIGGER_PASSES,
         config.QQQ_CORE_RECOVERY_LEVEL3_TRIGGER_PASSES,
         config.QQQ_CORE_RECOVERY_LEVEL3_ORDINARY_EXPOSURE_MIN_PCT) = self._orig

    def test_below_ma_clears_to_inactive(self):
        prev = qcr.RecoveryState(active=True, reserve_active=True, level3_eligible=True,
                                  consecutive_no_progress_passes=99)
        state = qcr.advance_state(prev, qqq_above_ma=False, shortfall=50_000.0,
                                   ordinary_exposure_pct=0.9)
        self.assertEqual(state, qcr.RecoveryState())

    def test_shortfall_zero_clears_to_inactive(self):
        prev = qcr.RecoveryState(active=True, reserve_active=True, consecutive_no_progress_passes=10)
        state = qcr.advance_state(prev, qqq_above_ma=True, shortfall=0.0,
                                   ordinary_exposure_pct=0.9)
        self.assertEqual(state, qcr.RecoveryState())

    def test_new_episode_starts_at_zero(self):
        prev = qcr.RecoveryState()   # inactive
        state = qcr.advance_state(prev, qqq_above_ma=True, shortfall=50_000.0,
                                   ordinary_exposure_pct=0.5)
        self.assertTrue(state.active)
        self.assertEqual(state.consecutive_no_progress_passes, 0)
        self.assertFalse(state.reserve_active)
        self.assertFalse(state.level3_eligible)
        self.assertEqual(state.last_shortfall, 50_000.0)
        self.assertIsNotNone(state.episode_started_at)

    def test_meaningful_progress_resets_counter(self):
        prev = qcr.RecoveryState(active=True, last_shortfall=50_000.0,
                                  consecutive_no_progress_passes=2,
                                  episode_started_at="2026-01-01T00:00:00")
        # shrunk by 5,000 > 1,000 min-trade noise floor -> real progress
        state = qcr.advance_state(prev, qqq_above_ma=True, shortfall=45_000.0,
                                   ordinary_exposure_pct=0.5)
        self.assertEqual(state.consecutive_no_progress_passes, 0)

    def test_noise_level_progress_still_counts_as_no_progress(self):
        prev = qcr.RecoveryState(active=True, last_shortfall=50_000.0,
                                  consecutive_no_progress_passes=1,
                                  episode_started_at="2026-01-01T00:00:00")
        # shrunk by only 500 < 1,000 min-trade noise floor
        state = qcr.advance_state(prev, qqq_above_ma=True, shortfall=49_500.0,
                                   ordinary_exposure_pct=0.5)
        self.assertEqual(state.consecutive_no_progress_passes, 2)

    def test_reserve_triggers_at_trigger_passes(self):
        prev = qcr.RecoveryState(active=True, last_shortfall=50_000.0,
                                  consecutive_no_progress_passes=2,
                                  episode_started_at="2026-01-01T00:00:00")
        state = qcr.advance_state(prev, qqq_above_ma=True, shortfall=50_000.0,
                                   ordinary_exposure_pct=0.5)
        self.assertEqual(state.consecutive_no_progress_passes, 3)
        self.assertTrue(state.reserve_active)

    def test_reserve_active_does_not_revert_on_later_progress(self):
        prev = qcr.RecoveryState(active=True, last_shortfall=50_000.0,
                                  consecutive_no_progress_passes=3, reserve_active=True,
                                  episode_started_at="2026-01-01T00:00:00")
        # big progress this pass -> counter resets, but reserve_active stays True
        state = qcr.advance_state(prev, qqq_above_ma=True, shortfall=10_000.0,
                                   ordinary_exposure_pct=0.5)
        self.assertEqual(state.consecutive_no_progress_passes, 0)
        self.assertTrue(state.reserve_active)

    def test_level3_eligible_requires_both_passes_and_exposure(self):
        prev = qcr.RecoveryState(active=True, last_shortfall=50_000.0,
                                  consecutive_no_progress_passes=4, reserve_active=True,
                                  episode_started_at="2026-01-01T00:00:00")
        state = qcr.advance_state(prev, qqq_above_ma=True, shortfall=50_000.0,
                                   ordinary_exposure_pct=0.60)
        self.assertEqual(state.consecutive_no_progress_passes, 5)
        self.assertTrue(state.level3_eligible)

    def test_level3_not_eligible_when_ordinary_exposure_below_threshold(self):
        prev = qcr.RecoveryState(active=True, last_shortfall=50_000.0,
                                  consecutive_no_progress_passes=4, reserve_active=True,
                                  episode_started_at="2026-01-01T00:00:00")
        state = qcr.advance_state(prev, qqq_above_ma=True, shortfall=50_000.0,
                                   ordinary_exposure_pct=0.30)
        self.assertEqual(state.consecutive_no_progress_passes, 5)
        self.assertFalse(state.level3_eligible)

    def test_level3_not_eligible_when_exposure_unknown(self):
        prev = qcr.RecoveryState(active=True, last_shortfall=50_000.0,
                                  consecutive_no_progress_passes=4, reserve_active=True,
                                  episode_started_at="2026-01-01T00:00:00")
        state = qcr.advance_state(prev, qqq_above_ma=True, shortfall=50_000.0,
                                   ordinary_exposure_pct=None)
        self.assertFalse(state.level3_eligible)

    def test_episode_started_at_preserved_across_passes(self):
        prev = qcr.RecoveryState(active=True, last_shortfall=50_000.0,
                                  consecutive_no_progress_passes=1,
                                  episode_started_at="2026-01-01T00:00:00")
        state = qcr.advance_state(prev, qqq_above_ma=True, shortfall=50_000.0,
                                   ordinary_exposure_pct=0.5)
        self.assertEqual(state.episode_started_at, "2026-01-01T00:00:00")


class ReserveAmountTestCase(unittest.TestCase):

    def setUp(self):
        self._orig_pct = config.QQQ_CORE_RECOVERY_MAX_RESERVE_PCT
        config.QQQ_CORE_RECOVERY_MAX_RESERVE_PCT = 0.05

    def tearDown(self):
        config.QQQ_CORE_RECOVERY_MAX_RESERVE_PCT = self._orig_pct

    def test_zero_when_reserve_not_active(self):
        state = qcr.RecoveryState(active=True, reserve_active=False)
        self.assertEqual(qcr.reserve_amount(state, 100_000.0, 1_000_000.0, 500_000.0), 0.0)

    def test_zero_when_shortfall_not_positive(self):
        state = qcr.RecoveryState(active=True, reserve_active=True)
        self.assertEqual(qcr.reserve_amount(state, 0.0, 1_000_000.0, 500_000.0), 0.0)

    def test_zero_when_pool_none(self):
        state = qcr.RecoveryState(active=True, reserve_active=True)
        self.assertEqual(qcr.reserve_amount(state, 100_000.0, 1_000_000.0, None), 0.0)

    def test_zero_when_total_assets_none(self):
        state = qcr.RecoveryState(active=True, reserve_active=True)
        self.assertEqual(qcr.reserve_amount(state, 100_000.0, None, 500_000.0), 0.0)

    def test_shortfall_is_the_binding_constraint(self):
        # shortfall (2,000) < 5% of 1,000,000 (50,000) < pool (500,000)
        state = qcr.RecoveryState(active=True, reserve_active=True)
        self.assertEqual(qcr.reserve_amount(state, 2_000.0, 1_000_000.0, 500_000.0), 2_000.0)

    def test_max_reserve_pct_is_the_binding_constraint(self):
        # 5% of 1,000,000 = 50,000 < shortfall (200,000) < pool (500,000)
        state = qcr.RecoveryState(active=True, reserve_active=True)
        self.assertEqual(qcr.reserve_amount(state, 200_000.0, 1_000_000.0, 500_000.0), 50_000.0)

    def test_available_pool_is_the_binding_constraint(self):
        # pool (10,000) < 5% of 1,000,000 (50,000) < shortfall (200,000)
        state = qcr.RecoveryState(active=True, reserve_active=True)
        self.assertEqual(qcr.reserve_amount(state, 200_000.0, 1_000_000.0, 10_000.0), 10_000.0)


class PlanLevel3ReleaseTestCase(unittest.TestCase):

    def setUp(self):
        self._orig_min_trade = config.PORTFOLIO_REBALANCE_MIN_TRADE_USD
        config.PORTFOLIO_REBALANCE_MIN_TRADE_USD = 1000.0

    def tearDown(self):
        config.PORTFOLIO_REBALANCE_MIN_TRADE_USD = self._orig_min_trade

    def test_empty_when_shortfall_not_positive(self):
        self.assertEqual(qcr.plan_level3_release({}, {}, 0.0), [])

    def test_reuses_weakness_ranking_order(self):
        # US.WEAK has an active SELL signal (weakest), US.STRONG is a plain BUY hold
        positions = {
            "US.WEAK":   {"strategy": "atr_breakout", "qty": 100, "entry_time": "2026-01-01"},
            "US.STRONG": {"strategy": "atr_breakout", "qty": 100, "entry_time": "2026-01-01"},
        }
        results = {
            "US.WEAK":   {"signal": "SELL", "signal_strength": 0.9, "current_price": 50.0},
            "US.STRONG": {"signal": "HOLD", "signal_strength": 0.0, "current_price": 50.0},
        }
        plan = qcr.plan_level3_release(positions, results, shortfall_remaining=3_000.0)
        self.assertEqual(plan[0].code, "US.WEAK")

    def test_accumulates_until_shortfall_covered(self):
        positions = {
            "US.A": {"strategy": "atr_breakout", "qty": 100, "entry_time": "2026-01-01"},
            "US.B": {"strategy": "atr_breakout", "qty": 100, "entry_time": "2026-01-01"},
        }
        results = {
            "US.A": {"signal": "SELL", "signal_strength": 0.9, "current_price": 50.0},   # $5,000 held
            "US.B": {"signal": "HOLD", "signal_strength": 0.0, "current_price": 50.0},   # $5,000 held
        }
        # shortfall 6,000 -> fully sell US.A ($5,000) then partially trim US.B ($1,000)
        plan = qcr.plan_level3_release(positions, results, shortfall_remaining=6_000.0)
        self.assertEqual(len(plan), 2)
        total_value = sum(o.sell_qty * o.price for o in plan)
        self.assertLessEqual(total_value, 6_000.0)
        self.assertGreater(total_value, 5_000.0)

    def test_skips_fragments_below_min_trade(self):
        positions = {"US.A": {"strategy": "atr_breakout", "qty": 10, "entry_time": "2026-01-01"}}
        results = {"US.A": {"signal": "SELL", "signal_strength": 0.9, "current_price": 50.0}}
        # US.A only worth $500 total, shortfall tiny remainder would be below min trade
        plan = qcr.plan_level3_release(positions, results, shortfall_remaining=10.0)
        self.assertEqual(plan, [])

    def test_returns_rebalance_order_instances_only(self):
        positions = {"US.A": {"strategy": "atr_breakout", "qty": 100, "entry_time": "2026-01-01"}}
        results = {"US.A": {"signal": "SELL", "signal_strength": 0.9, "current_price": 50.0}}
        plan = qcr.plan_level3_release(positions, results, shortfall_remaining=3_000.0)
        self.assertTrue(all(isinstance(o, RebalanceOrder) for o in plan))


class StateIOTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_state_path = qcr._STATE_PATH
        qcr._STATE_PATH = Path(self._tmpdir.name) / "qqq_recovery_state.json"

    def tearDown(self):
        qcr._STATE_PATH = self._orig_state_path
        self._tmpdir.cleanup()

    def test_missing_file_returns_inactive_default(self):
        self.assertEqual(qcr.load_state(), qcr.RecoveryState())

    def test_corrupt_json_returns_inactive_default(self):
        qcr._STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        qcr._STATE_PATH.write_text("{not valid json", encoding="utf-8")
        self.assertEqual(qcr.load_state(), qcr.RecoveryState())

    def test_round_trip(self):
        state = qcr.RecoveryState(active=True, last_shortfall=1234.5,
                                   consecutive_no_progress_passes=2, reserve_active=True,
                                   level3_eligible=False, episode_started_at="2026-01-01T00:00:00")
        qcr.save_state(state)
        self.assertEqual(qcr.load_state(), state)

    def test_save_never_raises_on_unwritable_path(self):
        qcr._STATE_PATH = Path("Z:\\definitely\\does\\not\\exist\\state.json")
        try:
            qcr.save_state(qcr.RecoveryState())
        except Exception as exc:
            self.fail(f"save_state() must never raise, got {exc!r}")

    def test_load_never_raises_on_unwritable_path(self):
        qcr._STATE_PATH = Path("Z:\\definitely\\does\\not\\exist\\state.json")
        try:
            qcr.load_state()
        except Exception as exc:
            self.fail(f"load_state() must never raise, got {exc!r}")


if __name__ == "__main__":
    unittest.main()
