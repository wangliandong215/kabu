"""
Unit tests for portfolio/replacement_stabilizer.py (v2.4 RSL).
Run:  python -m unittest portfolio.test_replacement_stabilizer -v
"""
import json
import unittest

import config
from engine.scoring import LABEL_FULL, LABEL_OBSERVATION, LABEL_PARTIAL
from portfolio.replacement_stabilizer import (
    REPL_FULL_DOWNGRADE_REPLACE, REPL_LOW_PARTIAL_EVICT, REPL_OBSERVATION_EVICT,
    adaptive_threshold, decide, explain, explain_capital_flow, stability_score,
)


class TestReplacementStabilizer(unittest.TestCase):

    def setUp(self):
        self._orig = {
            "REPLACEMENT_MARGIN":              config.REPLACEMENT_MARGIN,
            "REPLACEMENT_BUDGET_PER_100_DAYS":         config.REPLACEMENT_BUDGET_PER_100_DAYS,
            "REPLACEMENT_LOW_PARTIAL_THRESHOLD":       config.REPLACEMENT_LOW_PARTIAL_THRESHOLD,
            "REPLACEMENT_WEAK_FULL_PERCENTILE":        config.REPLACEMENT_WEAK_FULL_PERCENTILE,
            "REPLACEMENT_WEAK_FULL_MIN_HISTORY":       config.REPLACEMENT_WEAK_FULL_MIN_HISTORY,
            "REPLACEMENT_STABILITY_HOLDING_NORM_DAYS": config.REPLACEMENT_STABILITY_HOLDING_NORM_DAYS,
            "REPLACEMENT_STABILITY_VOL_NORM_ATR_PCT":  config.REPLACEMENT_STABILITY_VOL_NORM_ATR_PCT,
            "REPLACEMENT_STABILITY_PROTECT_THRESHOLD": config.REPLACEMENT_STABILITY_PROTECT_THRESHOLD,
            "REPLACEMENT_EXTREME_ADVANTAGE_OVERRIDE":  config.REPLACEMENT_EXTREME_ADVANTAGE_OVERRIDE,
            "REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS": config.REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS,
            "REPLACEMENT_WEAK_FULL_ENABLED":           config.REPLACEMENT_WEAK_FULL_ENABLED,
            "REPLACEMENT_OBSERVATION_TIER_ENABLED":    config.REPLACEMENT_OBSERVATION_TIER_ENABLED,
            "REPLACEMENT_LOW_PARTIAL_TIER_ENABLED":    config.REPLACEMENT_LOW_PARTIAL_TIER_ENABLED,
            # 2026-07-06: find_replaceable_position()（Tier1用）现在是
            # evaluate_replacement()的薄封装，会经过v2.4阶段三转正的这三个
            # 消融过滤门——这里显式关闭，这组RSL测试用的是虚构代码，不该被
            # 生产环境的new_score/same_sector默认值干扰。
            "REPLACEMENT_MIN_NEW_SCORE":               config.REPLACEMENT_MIN_NEW_SCORE,
            "REPLACEMENT_MAX_OBSERVATION_POOL_SIZE":   config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE,
            "REPLACEMENT_BLOCK_SAME_SECTOR":           config.REPLACEMENT_BLOCK_SAME_SECTOR,
        }
        config.REPLACEMENT_MARGIN              = 10.0
        config.REPLACEMENT_BUDGET_PER_100_DAYS         = 15
        config.REPLACEMENT_LOW_PARTIAL_THRESHOLD       = 70.0
        config.REPLACEMENT_WEAK_FULL_PERCENTILE        = 70.0
        config.REPLACEMENT_WEAK_FULL_MIN_HISTORY       = 20
        config.REPLACEMENT_STABILITY_HOLDING_NORM_DAYS = 20
        config.REPLACEMENT_STABILITY_VOL_NORM_ATR_PCT  = 0.08
        config.REPLACEMENT_STABILITY_PROTECT_THRESHOLD = 0.5
        config.REPLACEMENT_EXTREME_ADVANTAGE_OVERRIDE  = 30.0
        config.REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS = 8
        config.REPLACEMENT_WEAK_FULL_ENABLED            = True
        config.REPLACEMENT_OBSERVATION_TIER_ENABLED     = True
        config.REPLACEMENT_LOW_PARTIAL_TIER_ENABLED     = True
        config.REPLACEMENT_MIN_NEW_SCORE                = None
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE    = None
        config.REPLACEMENT_BLOCK_SAME_SECTOR            = False

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(config, k, v)

    def _decide(self, incoming_score, held, day_idx=20, vol=0.0, crowd=0.0,
                history=None, cooldown=None, replaced_today=None, recent_count=0,
                new_tier_count=0):
        return decide(
            incoming_code="US.NEW", incoming_score=incoming_score, held_positions=held,
            current_day_idx=day_idx, volatility_factor=vol, crowding_factor=crowd,
            full_score_history=history or [], cooldown_until=cooldown or {},
            replaced_today=replaced_today or set(), recent_replacement_count=recent_count,
            recent_new_tier_replacement_count=new_tier_count,
        )

    # ── adaptive_threshold ────────────────────────────────────────────────

    def test_adaptive_threshold_never_below_flat_advantage(self):
        self.assertEqual(adaptive_threshold(0.0, 0.0), 10.0)
        self.assertEqual(adaptive_threshold(5.0, 3.0), 18.0)

    # ── Tier 1: OBSERVATION mirrors v2.3, just gated stricter ──────────────

    def test_tier1_observation_evict_when_advantage_sufficient(self):
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                            "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1}}
        d = self._decide(90.0, held)
        self.assertEqual(d.victim_code, "US.OBS")
        self.assertEqual(d.replacement_type, REPL_OBSERVATION_EVICT)

    def test_tier1_refused_when_volatility_crowding_push_threshold_past_delta(self):
        # 90 - 45 = 45 advantage. Flat 10pt gate would pass, but a large
        # adaptive buffer (vol+crowding) can still refuse it.
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                            "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1}}
        d = self._decide(90.0, held, vol=10.0, crowd=10.0)  # threshold = 10+10+10=30 < 45, still passes
        self.assertEqual(d.victim_code, "US.OBS")
        d2 = self._decide(60.0, held, vol=10.0, crowd=10.0)  # delta=15 < threshold(30) -> refused
        self.assertIsNone(d2.victim_code)

    # ── Tier 2: LOW PARTIAL only kicks in when Tier 1 has no candidate ─────

    def test_tier2_used_only_when_no_observation(self):
        held = {
            "US.LP": {"score_label": LABEL_PARTIAL, "total_score": 62.0,
                      "entry_day_idx": 0, "avg_cost": 10.0, "entry_atr": 1.0},  # low stability (fresh, weak vol penalty)
        }
        d = self._decide(90.0, held, day_idx=1)
        self.assertEqual(d.victim_code, "US.LP")
        self.assertEqual(d.replacement_type, REPL_LOW_PARTIAL_EVICT)

    def test_tier1_takes_priority_over_tier2_when_both_present(self):
        held = {
            "US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                       "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1},
            "US.LP":  {"score_label": LABEL_PARTIAL, "total_score": 20.0,
                       "entry_day_idx": 0, "avg_cost": 10.0, "entry_atr": 1.0},
        }
        d = self._decide(90.0, held)
        self.assertEqual(d.victim_code, "US.OBS")
        self.assertEqual(d.replacement_type, REPL_OBSERVATION_EVICT)

    def test_partial_at_or_above_threshold_is_not_low_partial(self):
        held = {"US.MIDPARTIAL": {"score_label": LABEL_PARTIAL, "total_score": 75.0,
                                   "entry_day_idx": 0, "avg_cost": 10.0, "entry_atr": 1.0}}
        d = self._decide(90.0, held)
        self.assertIsNone(d.victim_code)

    # ── Stability Score protection (Tier 2/3 only) ─────────────────────────

    def test_stability_score_formula(self):
        # holding 20d/norm20 -> 1.0 ; score 80/100 -> 0.8 ; atr_pct 0.04/0.08 -> vol_penalty 0.5
        pos = {"_holding_days": 20, "total_score": 80.0, "avg_cost": 100.0, "entry_atr": 4.0}
        self.assertAlmostEqual(stability_score(pos), 1.0 * 0.8 * 0.5, places=6)

    def test_high_stability_low_partial_protected_unless_extreme_delta(self):
        # Long-held, high score, low own-volatility -> high stability score.
        held = {"US.STABLE": {"score_label": LABEL_PARTIAL, "total_score": 69.0,
                               "entry_day_idx": 0, "avg_cost": 100.0, "entry_atr": 1.0}}
        # delta = 95 - 69 = 26 < EXTREME_ADVANTAGE_OVERRIDE(30) -> protected, refused
        d = self._decide(95.0, held, day_idx=25)
        self.assertIsNone(d.victim_code)
        # delta = 100 - 69 = 31 >= 30 -> extreme override, allowed
        d2 = self._decide(100.0, held, day_idx=25)
        self.assertEqual(d2.victim_code, "US.STABLE")
        self.assertEqual(d2.replacement_type, REPL_LOW_PARTIAL_EVICT)

    # ── Tier 3: WEAK FULL requires min history and percentile cutoff ───────

    def test_tier3_skipped_when_history_too_short(self):
        held = {"US.WEAKFULL": {"score_label": LABEL_FULL, "total_score": 80.0,
                                 "entry_day_idx": 0, "avg_cost": 10.0, "entry_atr": 1.0}}
        d = self._decide(95.0, held, day_idx=1, history=[80.0] * 5)  # < min history 20
        self.assertIsNone(d.victim_code)

    def test_tier3_evicts_below_percentile_cutoff(self):
        history = [70.0] * 15 + [95.0] * 15   # 70th percentile sits around 85-90
        held = {"US.WEAKFULL": {"score_label": LABEL_FULL, "total_score": 80.0,
                                 "entry_day_idx": 0, "avg_cost": 10.0, "entry_atr": 1.0}}
        d = self._decide(96.0, held, day_idx=1, history=history)
        self.assertEqual(d.victim_code, "US.WEAKFULL")
        self.assertEqual(d.replacement_type, REPL_FULL_DOWNGRADE_REPLACE)

    def test_tier3_not_used_when_full_score_above_cutoff(self):
        history = [70.0] * 15 + [95.0] * 15
        held = {"US.STRONGFULL": {"score_label": LABEL_FULL, "total_score": 96.0,
                                   "entry_day_idx": 0, "avg_cost": 10.0, "entry_atr": 1.0}}
        d = self._decide(99.0, held, day_idx=1, history=history)
        self.assertIsNone(d.victim_code)

    # ── Tier 3 shadow mode (REPLACEMENT_WEAK_FULL_ENABLED=False) ───────────

    def test_weak_full_shadow_mode_does_not_execute_but_logs_would_be_victim(self):
        config.REPLACEMENT_WEAK_FULL_ENABLED = False
        try:
            history = [70.0] * 15 + [95.0] * 15
            held = {"US.WEAKFULL": {"score_label": LABEL_FULL, "total_score": 80.0,
                                     "entry_day_idx": 0, "avg_cost": 10.0, "entry_atr": 1.0}}
            d = self._decide(96.0, held, day_idx=1, history=history)
            # Not executed -- the portfolio must behave exactly as if Tier3
            # didn't exist (falls back to the normal capacity-block outcome).
            self.assertIsNone(d.victim_code)
            self.assertIsNone(d.replacement_type)
            # But the shadow log must record what WOULD have happened.
            self.assertEqual(d.shadow_weak_full_victim, "US.WEAKFULL")
            self.assertIsNotNone(d.shadow_weak_full_stability_score)
        finally:
            config.REPLACEMENT_WEAK_FULL_ENABLED = True

    def test_weak_full_shadow_mode_silent_when_no_tier3_candidate(self):
        config.REPLACEMENT_WEAK_FULL_ENABLED = False
        try:
            held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                                "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1}}
            # Tier1 fires normally -- shadow mode only concerns Tier3 and must
            # not interfere with Tier1/Tier2 at all.
            d = self._decide(90.0, held)
            self.assertEqual(d.victim_code, "US.OBS")
            self.assertIsNone(d.shadow_weak_full_victim)
        finally:
            config.REPLACEMENT_WEAK_FULL_ENABLED = True

    # ── Tier 1/2 shadow mode (generalizes the Tier3 pattern) ────────────────

    def test_observation_shadow_mode_falls_through_to_low_partial(self):
        config.REPLACEMENT_OBSERVATION_TIER_ENABLED = False
        try:
            held = {
                "US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                           "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1},
                "US.LP":  {"score_label": LABEL_PARTIAL, "total_score": 62.0,
                           "entry_day_idx": 0, "avg_cost": 10.0, "entry_atr": 1.0},
            }
            d = self._decide(90.0, held, day_idx=1)
            # Tier1 doesn't execute -- Tier2 wins instead (the slot behaves
            # as if OBSERVATION eviction wasn't available this review).
            self.assertEqual(d.victim_code, "US.LP")
            self.assertEqual(d.replacement_type, REPL_LOW_PARTIAL_EVICT)
            self.assertEqual(d.shadow_observation_victim, "US.OBS")
        finally:
            config.REPLACEMENT_OBSERVATION_TIER_ENABLED = True

    def test_observation_shadow_mode_falls_back_to_block_when_nothing_else_eligible(self):
        config.REPLACEMENT_OBSERVATION_TIER_ENABLED = False
        try:
            held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                                "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1}}
            d = self._decide(90.0, held, day_idx=12)
            self.assertIsNone(d.victim_code)
            self.assertEqual(d.shadow_observation_victim, "US.OBS")
        finally:
            config.REPLACEMENT_OBSERVATION_TIER_ENABLED = True

    def test_low_partial_shadow_mode_does_not_execute_but_logs(self):
        config.REPLACEMENT_LOW_PARTIAL_TIER_ENABLED = False
        try:
            held = {"US.LP": {"score_label": LABEL_PARTIAL, "total_score": 62.0,
                               "entry_day_idx": 0, "avg_cost": 10.0, "entry_atr": 1.0}}
            d = self._decide(90.0, held, day_idx=1)
            self.assertIsNone(d.victim_code)
            self.assertEqual(d.shadow_low_partial_victim, "US.LP")
            self.assertIsNotNone(d.shadow_low_partial_stability_score)
        finally:
            config.REPLACEMENT_LOW_PARTIAL_TIER_ENABLED = True

    # ── Cooldown ────────────────────────────────────────────────────────────

    def test_cooldown_excludes_code_from_candidacy(self):
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                            "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1}}
        d = self._decide(90.0, held, day_idx=12, cooldown={"US.OBS": 15})
        self.assertIsNone(d.victim_code)
        self.assertTrue(d.cooldown_blocked)
        # once cooldown has expired (day_idx >= cooldown_until) it's eligible again
        d2 = self._decide(90.0, held, day_idx=15, cooldown={"US.OBS": 15})
        self.assertEqual(d2.victim_code, "US.OBS")

    # ── Same-day chain-replacement guard ────────────────────────────────────

    def test_replaced_today_blocks_same_day_chain(self):
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                            "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1}}
        d = self._decide(90.0, held, day_idx=12, replaced_today={"US.OBS"})
        self.assertIsNone(d.victim_code)
        self.assertTrue(d.cooldown_blocked)

    def test_incoming_beneficiary_blocked_when_itself_on_cooldown(self):
        # US.NEW (the incoming candidate) was itself a party to a replacement
        # 2 days ago and is still cooling down -- it must not be allowed to
        # become the beneficiary of a brand new replacement either, even
        # though the victim side (US.OBS) is perfectly eligible.
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                            "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1}}
        d = self._decide(90.0, held, day_idx=12, cooldown={"US.NEW": 15})
        self.assertIsNone(d.victim_code)
        self.assertTrue(d.cooldown_blocked)

    def test_incoming_beneficiary_blocked_when_replaced_earlier_today(self):
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                            "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1}}
        d = self._decide(90.0, held, day_idx=12, replaced_today={"US.NEW"})
        self.assertIsNone(d.victim_code)
        self.assertTrue(d.cooldown_blocked)

    # ── Budget ───────────────────────────────────────────────────────────────

    def test_budget_exhausted_blocks_any_tier(self):
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                            "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1}}
        d = self._decide(90.0, held, recent_count=15)
        self.assertIsNone(d.victim_code)
        self.assertTrue(d.budget_blocked)
        d2 = self._decide(90.0, held, recent_count=14)
        self.assertEqual(d2.victim_code, "US.OBS")

    def test_no_eligible_candidates_returns_none_without_cooldown_flag(self):
        held = {}
        d = self._decide(90.0, held)
        self.assertIsNone(d.victim_code)
        self.assertFalse(d.cooldown_blocked)
        self.assertFalse(d.budget_blocked)

    # ── Tier2/3 独立子预算：不与Tier1共享额度 ─────────────────────────────────

    def test_new_tier_budget_exhausted_blocks_tier2_but_not_tier1(self):
        # Tier1(OBSERVATION) must remain unaffected by the Tier2/3 sub-budget
        # -- it's governed only by the shared REPLACEMENT_BUDGET_PER_100_DAYS.
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                            "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1}}
        d = self._decide(90.0, held, new_tier_count=8)
        self.assertEqual(d.victim_code, "US.OBS")
        self.assertEqual(d.replacement_type, REPL_OBSERVATION_EVICT)
        self.assertFalse(d.new_tier_budget_blocked)

    def test_new_tier_budget_exhausted_blocks_low_partial(self):
        held = {"US.LP": {"score_label": LABEL_PARTIAL, "total_score": 62.0,
                           "entry_day_idx": 0, "avg_cost": 10.0, "entry_atr": 1.0}}
        d = self._decide(90.0, held, day_idx=1, new_tier_count=8)
        self.assertIsNone(d.victim_code)
        self.assertTrue(d.new_tier_budget_blocked)
        d2 = self._decide(90.0, held, day_idx=1, new_tier_count=7)
        self.assertEqual(d2.victim_code, "US.LP")
        self.assertEqual(d2.replacement_type, REPL_LOW_PARTIAL_EVICT)

    def test_new_tier_budget_exhausted_blocks_weak_full(self):
        history = [70.0] * 15 + [95.0] * 15
        held = {"US.WEAKFULL": {"score_label": LABEL_FULL, "total_score": 80.0,
                                 "entry_day_idx": 0, "avg_cost": 10.0, "entry_atr": 1.0}}
        d = self._decide(96.0, held, day_idx=1, history=history, new_tier_count=8)
        self.assertIsNone(d.victim_code)
        self.assertTrue(d.new_tier_budget_blocked)


class TestExplainCapitalFlow(unittest.TestCase):
    """explain_capital_flow() 是纯解释层，不参与decide()的判断——这里验证
    它对已知场景算出的三变量能正确"复现"decide()的真实结论，证明这个
    统一解释层是忠实的，不是凭空看起来合理但实际方向不对的装饰。"""

    def setUp(self):
        self._orig = {
            "REPLACEMENT_MARGIN":              config.REPLACEMENT_MARGIN,
            "REPLACEMENT_BUDGET_PER_100_DAYS":         config.REPLACEMENT_BUDGET_PER_100_DAYS,
            "REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS": config.REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS,
            "REPLACEMENT_STABILITY_PROTECT_THRESHOLD": config.REPLACEMENT_STABILITY_PROTECT_THRESHOLD,
            "REPLACEMENT_EXTREME_ADVANTAGE_OVERRIDE":  config.REPLACEMENT_EXTREME_ADVANTAGE_OVERRIDE,
        }
        config.REPLACEMENT_MARGIN               = 10.0
        config.REPLACEMENT_BUDGET_PER_100_DAYS          = 15
        config.REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS = 8
        config.REPLACEMENT_STABILITY_PROTECT_THRESHOLD  = 0.5
        config.REPLACEMENT_EXTREME_ADVANTAGE_OVERRIDE   = 30.0

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(config, k, v)

    def test_matches_tier1_allow(self):
        # Mirrors test_tier1_observation_evict_when_advantage_sufficient.
        v = explain_capital_flow(incoming_score=90.0, victim_score=45.0,
                                  volatility_factor=0.0, crowding_factor=0.0,
                                  recent_replacement_count=0)
        self.assertTrue(v.would_pass_score_gate)
        self.assertGreater(v.capital_velocity_pressure, 0)
        self.assertEqual(v.constraint_state, v.temporal_friction_cost)
        self.assertTrue(v.would_reallocate)

    def test_matches_tier1_refuse_under_adaptive_threshold(self):
        # Mirrors test_tier1_refused_when_volatility_crowding_push_threshold_past_delta.
        v = explain_capital_flow(incoming_score=60.0, victim_score=45.0,
                                  volatility_factor=10.0, crowding_factor=10.0,
                                  recent_replacement_count=0)
        self.assertFalse(v.would_pass_score_gate)   # delta=15 < threshold=30
        self.assertFalse(v.would_reallocate)

    def test_matches_budget_exhausted(self):
        # Mirrors test_budget_exhausted_blocks_any_tier: score gate would
        # pass, but capital_velocity_pressure hitting 0 is a SEPARATE,
        # independent reason the real decide() blocks it. Under the Latent
        # Constraint Model this collapses into one inequality: constraint_state
        # goes to +inf, so would_reallocate is False even though the raw
        # score gate alone would have passed.
        v = explain_capital_flow(incoming_score=90.0, victim_score=45.0,
                                  volatility_factor=0.0, crowding_factor=0.0,
                                  recent_replacement_count=15)
        self.assertTrue(v.would_pass_score_gate)
        self.assertEqual(v.capital_velocity_pressure, 0.0)
        self.assertEqual(v.constraint_state, float("inf"))
        self.assertFalse(v.would_reallocate)

    def test_matches_stability_protection_step_function(self):
        # Mirrors test_high_stability_low_partial_protected_unless_extreme_delta:
        # stability_score ~0.604 (>=0.5) -> protected -> effective friction
        # jumps to the extreme override (30), not a smooth function of the
        # stability score's exact value.
        pos = {"_holding_days": 25, "total_score": 69.0, "avg_cost": 100.0, "entry_atr": 1.0}
        s = stability_score(pos)
        self.assertGreaterEqual(s, 0.5)

        v_blocked = explain_capital_flow(incoming_score=95.0, victim_score=69.0,
                                          volatility_factor=0.0, crowding_factor=0.0,
                                          recent_replacement_count=0, victim_stability_score=s)
        self.assertEqual(v_blocked.temporal_friction_cost, 30.0)
        self.assertFalse(v_blocked.would_pass_score_gate)   # delta=26 < 30

        v_override = explain_capital_flow(incoming_score=100.0, victim_score=69.0,
                                           volatility_factor=0.0, crowding_factor=0.0,
                                           recent_replacement_count=0, victim_stability_score=s)
        self.assertTrue(v_override.would_pass_score_gate)   # delta=31 >= 30

    def test_new_tier_headroom_is_the_binding_constraint_when_lower(self):
        v = explain_capital_flow(incoming_score=90.0, victim_score=45.0,
                                  volatility_factor=0.0, crowding_factor=0.0,
                                  recent_replacement_count=0,
                                  recent_new_tier_replacement_count=8)
        self.assertEqual(v.capital_velocity_pressure, 0.0)   # sub-budget exhausted


class TestExplainJSON(unittest.TestCase):
    """explain()的"decision"块必须跟同一组输入下decide()的真实输出逐字
    段一致——这是"100% replay decide()"这个要求的可验证保证，不是靠文档
    自我声称。"""

    def setUp(self):
        self._orig = {
            "REPLACEMENT_MARGIN":              config.REPLACEMENT_MARGIN,
            "REPLACEMENT_BUDGET_PER_100_DAYS":         config.REPLACEMENT_BUDGET_PER_100_DAYS,
            "REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS": config.REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS,
            "REPLACEMENT_LOW_PARTIAL_THRESHOLD":       config.REPLACEMENT_LOW_PARTIAL_THRESHOLD,
            "REPLACEMENT_WEAK_FULL_MIN_HISTORY":       config.REPLACEMENT_WEAK_FULL_MIN_HISTORY,
            "REPLACEMENT_MIN_NEW_SCORE":               config.REPLACEMENT_MIN_NEW_SCORE,
            "REPLACEMENT_MAX_OBSERVATION_POOL_SIZE":   config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE,
            "REPLACEMENT_BLOCK_SAME_SECTOR":           config.REPLACEMENT_BLOCK_SAME_SECTOR,
        }
        config.REPLACEMENT_MARGIN               = 10.0
        config.REPLACEMENT_BUDGET_PER_100_DAYS          = 15
        config.REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS = 8
        config.REPLACEMENT_LOW_PARTIAL_THRESHOLD        = 70.0
        config.REPLACEMENT_WEAK_FULL_MIN_HISTORY        = 20
        config.REPLACEMENT_MIN_NEW_SCORE                = None
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE    = None
        config.REPLACEMENT_BLOCK_SAME_SECTOR            = False

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(config, k, v)

    def _explain(self, incoming_score, held, day_idx=20, vol=0.0, crowd=0.0,
                 history=None, cooldown=None, replaced_today=None, recent_count=0,
                 new_tier_count=0):
        return explain(
            incoming_code="US.NEW", incoming_score=incoming_score, held_positions=held,
            current_day_idx=day_idx, volatility_factor=vol, crowding_factor=crowd,
            full_score_history=history or [], cooldown_until=cooldown or {},
            replaced_today=replaced_today or set(), recent_replacement_count=recent_count,
            recent_new_tier_replacement_count=new_tier_count,
        )

    def _decide(self, incoming_score, held, **kwargs):
        defaults = dict(day_idx=20, vol=0.0, crowd=0.0, history=None,
                         cooldown=None, replaced_today=None, recent_count=0, new_tier_count=0)
        defaults.update(kwargs)
        return decide(
            incoming_code="US.NEW", incoming_score=incoming_score, held_positions=held,
            current_day_idx=defaults["day_idx"], volatility_factor=defaults["vol"],
            crowding_factor=defaults["crowd"], full_score_history=defaults["history"] or [],
            cooldown_until=defaults["cooldown"] or {}, replaced_today=defaults["replaced_today"] or set(),
            recent_replacement_count=defaults["recent_count"],
            recent_new_tier_replacement_count=defaults["new_tier_count"],
        )

    def test_decision_block_matches_real_decide_on_tier1_allow(self):
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                            "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1}}
        d = self._decide(90.0, held, day_idx=20)
        e = self._explain(90.0, held, day_idx=20)
        self.assertEqual(e["decision"]["action"], "REPLACE")
        self.assertEqual(e["decision"]["victim_code"], d.victim_code)
        self.assertEqual(e["decision"]["tier"], d.replacement_type)

    def test_decision_block_matches_real_decide_on_no_candidate(self):
        d = self._decide(90.0, {})
        e = self._explain(90.0, {})
        self.assertEqual(e["decision"]["action"], "NO_ACTION")
        self.assertIsNone(e["decision"]["victim_code"])
        self.assertEqual(e["decision"]["victim_code"], d.victim_code)

    def test_decision_block_matches_real_decide_on_budget_blocked(self):
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                            "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1}}
        d = self._decide(90.0, held, recent_count=15)
        e = self._explain(90.0, held, recent_count=15)
        self.assertEqual(e["decision"]["action"], "NO_ACTION")
        self.assertEqual(e["decision"]["budget_blocked"], d.budget_blocked)
        self.assertTrue(e["decision"]["budget_blocked"])
        self.assertFalse(e["eligibility"]["budget_ok"])

    def test_decision_block_matches_real_decide_on_low_partial(self):
        held = {"US.LP": {"score_label": LABEL_PARTIAL, "total_score": 62.0,
                           "entry_day_idx": 0, "avg_cost": 10.0, "entry_atr": 1.0}}
        d = self._decide(90.0, held, day_idx=1)
        e = self._explain(90.0, held, day_idx=1)
        self.assertEqual(e["decision"]["victim_code"], "US.LP")
        self.assertEqual(e["decision"]["victim_code"], d.victim_code)
        self.assertEqual(e["decision"]["tier"], REPL_LOW_PARTIAL_EVICT)
        # Tier breakdown surfaces the candidate considered, independent of
        # which tier ultimately wins.
        self.assertEqual(e["tiers"][REPL_LOW_PARTIAL_EVICT]["candidate"], "US.LP")
        self.assertIsNone(e["tiers"][REPL_OBSERVATION_EVICT]["candidate"])

    def test_output_is_json_serializable(self):
        held = {
            "US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                       "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1},
            "US.LP":  {"score_label": LABEL_PARTIAL, "total_score": 62.0,
                       "entry_day_idx": 0, "avg_cost": 10.0, "entry_atr": 1.0},
        }
        e = self._explain(90.0, held)
        json.dumps(e)   # raises if anything (e.g. a bare float("inf")) isn't serializable

    def test_constraint_state_serialized_as_string_inf_not_bare_float(self):
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                            "entry_day_idx": 10, "avg_cost": 10.0, "entry_atr": 0.1}}
        e = self._explain(90.0, held, recent_count=15)   # budget exhausted
        self.assertEqual(e["tiers"][REPL_OBSERVATION_EVICT]["constraint_state"], "inf")
        json.dumps(e)


if __name__ == "__main__":
    unittest.main()
