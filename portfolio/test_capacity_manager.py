"""
Unit tests for portfolio/capacity_manager.py::find_replaceable_position() /
find_weak_full_replaceable_position() / evaluate_replacement().
Run:  python -m unittest portfolio.test_capacity_manager -v
"""
import unittest

import config
from engine.scoring import LABEL_FULL, LABEL_OBSERVATION, LABEL_PARTIAL
from portfolio.capacity_manager import (
    REPL_OBSERVATION_EVICT,
    REPL_WEAK_FULL_EVICT,
    evaluate_replacement,
    find_replaceable_position,
    find_weak_full_replaceable_position,
)


class TestFindReplaceablePosition(unittest.TestCase):
    """2026-07-06起 find_replaceable_position() 是 evaluate_replacement()
    的薄封装（见capacity_manager.py），这里只测纯margin/排序行为，跟
    TestEvaluateReplacement一样显式关闭三个消融过滤门，不让它们干扰——
    这组测试用的US.OBS等代码不在SECTOR_MAP里会一起落进"other"桶，且
    不传incoming_code时同样默认"other"，不关闭same_sector门会被误判。"""

    def setUp(self):
        self._orig_advantage = config.REPLACEMENT_MARGIN
        self._orig_min_score = config.REPLACEMENT_MIN_NEW_SCORE
        self._orig_max_pool = config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE
        self._orig_block_sector = config.REPLACEMENT_BLOCK_SAME_SECTOR
        config.REPLACEMENT_MARGIN = 10.0
        config.REPLACEMENT_MIN_NEW_SCORE = None
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = None
        config.REPLACEMENT_BLOCK_SAME_SECTOR = False

    def tearDown(self):
        config.REPLACEMENT_MARGIN = self._orig_advantage
        config.REPLACEMENT_MIN_NEW_SCORE = self._orig_min_score
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = self._orig_max_pool
        config.REPLACEMENT_BLOCK_SAME_SECTOR = self._orig_block_sector

    # ── PRD Test 1 类比：存在一个OBSERVATION，分数优势足够 -> 应该被换出 ──────

    def test_replaces_observation_when_advantage_sufficient(self):
        held = {
            "US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0,
                       "entry_day_idx": 10},
        }
        victim = find_replaceable_position(incoming_score=90.0, held_positions=held,
                                            current_day_idx=15)
        self.assertEqual(victim, "US.OBS")

    # ── PRD Test 2 类比：不存在OBSERVATION -> 必须返回None（回退到Block）────

    def test_returns_none_when_no_observation_held(self):
        held = {
            "US.FULL_HELD":    {"score_label": LABEL_FULL, "total_score": 95.0,
                                 "entry_day_idx": 5},
            "US.PARTIAL_HELD": {"score_label": LABEL_PARTIAL, "total_score": 65.0,
                                 "entry_day_idx": 8},
        }
        victim = find_replaceable_position(incoming_score=90.0, held_positions=held,
                                            current_day_idx=15)
        self.assertIsNone(victim)

    def test_returns_none_when_no_positions_held(self):
        victim = find_replaceable_position(incoming_score=90.0, held_positions={},
                                            current_day_idx=15)
        self.assertIsNone(victim)

    # ── REPLACEMENT_MARGIN 保护：分数差距不够时不允许置换（防止churn）──

    def test_refuses_replacement_when_advantage_insufficient(self):
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 55.0,
                            "entry_day_idx": 10}}
        # 90 - 55 = 35 >= 10 -> allowed
        self.assertEqual(
            find_replaceable_position(90.0, held, 15), "US.OBS")
        # 60 - 55 = 5 < 10 -> refused
        self.assertIsNone(find_replaceable_position(60.0, held, 15))

    def test_advantage_boundary_is_strict_greater_than(self):
        # incoming must be STRICTLY greater than victim_score + advantage,
        # exactly-equal should not trigger a replacement.
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 50.0,
                            "entry_day_idx": 0}}
        self.assertIsNone(find_replaceable_position(60.0, held, 0))          # 60 == 50+10
        self.assertEqual(find_replaceable_position(60.01, held, 0), "US.OBS")

    # ── 候选排序：Fused Score 由低到高优先，打平按 Holding Time 由长到短 ──────

    def test_picks_lowest_score_observation_among_multiple(self):
        held = {
            "US.A": {"score_label": LABEL_OBSERVATION, "total_score": 55.0, "entry_day_idx": 0},
            "US.B": {"score_label": LABEL_OBSERVATION, "total_score": 42.0, "entry_day_idx": 0},
            "US.C": {"score_label": LABEL_OBSERVATION, "total_score": 48.0, "entry_day_idx": 0},
        }
        victim = find_replaceable_position(incoming_score=90.0, held_positions=held,
                                            current_day_idx=10)
        self.assertEqual(victim, "US.B")   # lowest total_score wins

    def test_tiebreak_prefers_longest_held_when_scores_equal(self):
        held = {
            "US.NEW": {"score_label": LABEL_OBSERVATION, "total_score": 45.0, "entry_day_idx": 9},
            "US.OLD": {"score_label": LABEL_OBSERVATION, "total_score": 45.0, "entry_day_idx": 0},
        }
        victim = find_replaceable_position(incoming_score=90.0, held_positions=held,
                                            current_day_idx=10)
        self.assertEqual(victim, "US.OLD")   # held 10 days vs 1 day -> release the stale one

    def test_ignores_non_observation_positions_even_if_lower_score(self):
        # A PARTIAL position scoring lower than an OBSERVATION one must never
        # be selected -- only OBSERVATION is an eligible victim in this version.
        held = {
            "US.PARTIAL": {"score_label": LABEL_PARTIAL, "total_score": 20.0, "entry_day_idx": 0},
            "US.OBS":     {"score_label": LABEL_OBSERVATION, "total_score": 45.0, "entry_day_idx": 0},
        }
        victim = find_replaceable_position(incoming_score=90.0, held_positions=held,
                                            current_day_idx=10)
        self.assertEqual(victim, "US.OBS")

    def test_missing_total_score_is_excluded_defensively(self):
        held = {"US.NOSCORE": {"score_label": LABEL_OBSERVATION, "entry_day_idx": 0}}
        self.assertIsNone(find_replaceable_position(90.0, held, 10))


class TestFindWeakFullReplaceablePosition(unittest.TestCase):
    """独立 WEAK_FULL 通道（不带RSL的cooldown/budget/stability score）。"""

    def setUp(self):
        self._orig_margin = config.REPLACEMENT_MARGIN
        self._orig_percentile = config.REPLACEMENT_WEAK_FULL_PERCENTILE
        self._orig_min_history = config.REPLACEMENT_WEAK_FULL_MIN_HISTORY
        config.REPLACEMENT_MARGIN = 10.0
        config.REPLACEMENT_WEAK_FULL_PERCENTILE = 70.0
        config.REPLACEMENT_WEAK_FULL_MIN_HISTORY = 5

    def tearDown(self):
        config.REPLACEMENT_MARGIN = self._orig_margin
        config.REPLACEMENT_WEAK_FULL_PERCENTILE = self._orig_percentile
        config.REPLACEMENT_WEAK_FULL_MIN_HISTORY = self._orig_min_history

    def test_returns_none_when_history_insufficient(self):
        held = {"US.FULL": {"score_label": LABEL_FULL, "total_score": 20.0, "entry_day_idx": 0}}
        short_history = [50.0, 55.0, 60.0]   # 3 < min_history(5)
        self.assertIsNone(
            find_weak_full_replaceable_position(90.0, held, 10, short_history))

    def test_returns_none_when_history_is_none_or_empty(self):
        held = {"US.FULL": {"score_label": LABEL_FULL, "total_score": 20.0, "entry_day_idx": 0}}
        self.assertIsNone(find_weak_full_replaceable_position(90.0, held, 10, None))
        self.assertIsNone(find_weak_full_replaceable_position(90.0, held, 10, []))

    def test_replaces_full_position_below_percentile_cutoff(self):
        history = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]   # 70th pct = 73.0
        held = {"US.WEAK": {"score_label": LABEL_FULL, "total_score": 30.0, "entry_day_idx": 0}}
        victim = find_weak_full_replaceable_position(90.0, held, 10, history)
        self.assertEqual(victim, "US.WEAK")

    def test_ignores_full_position_at_or_above_percentile_cutoff(self):
        history = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]   # 70th pct = 73.0
        held = {"US.STRONG": {"score_label": LABEL_FULL, "total_score": 85.0, "entry_day_idx": 0}}
        victim = find_weak_full_replaceable_position(90.0, held, 10, history)
        self.assertIsNone(victim)

    def test_ignores_observation_and_partial_positions(self):
        history = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        held = {
            "US.OBS":     {"score_label": LABEL_OBSERVATION, "total_score": 10.0, "entry_day_idx": 0},
            "US.PARTIAL": {"score_label": LABEL_PARTIAL, "total_score": 10.0, "entry_day_idx": 0},
        }
        self.assertIsNone(find_weak_full_replaceable_position(90.0, held, 10, history))

    def test_respects_margin_on_weak_full_candidate(self):
        history = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        held = {"US.WEAK": {"score_label": LABEL_FULL, "total_score": 30.0, "entry_day_idx": 0}}
        # 39 - 30 = 9 < margin(10) -> refused
        self.assertIsNone(find_weak_full_replaceable_position(39.0, held, 10, history))
        # 41 - 30 = 11 > margin(10) -> allowed
        self.assertEqual(find_weak_full_replaceable_position(41.0, held, 10, history), "US.WEAK")


class TestEvaluateReplacement(unittest.TestCase):
    """Explain Layer 入口：完整决策记录（含被margin拦截的KEEP）。这里只
    测margin/WEAK_FULL逻辑本身，2026-07-05转正的三个消融过滤门
    （new_score/same_sector/pool_size）显式关掉，不干扰这组既有测试
    ——那三个门的开/关行为由 TestEvaluateReplacementAblationGates 单独
    覆盖。"""

    def setUp(self):
        self._orig_margin = config.REPLACEMENT_MARGIN
        self._orig_standalone = config.REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED
        self._orig_percentile = config.REPLACEMENT_WEAK_FULL_PERCENTILE
        self._orig_min_history = config.REPLACEMENT_WEAK_FULL_MIN_HISTORY
        self._orig_min_score = config.REPLACEMENT_MIN_NEW_SCORE
        self._orig_max_pool = config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE
        self._orig_block_sector = config.REPLACEMENT_BLOCK_SAME_SECTOR
        config.REPLACEMENT_MARGIN = 10.0
        config.REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED = True
        config.REPLACEMENT_WEAK_FULL_PERCENTILE = 70.0
        config.REPLACEMENT_WEAK_FULL_MIN_HISTORY = 5
        config.REPLACEMENT_MIN_NEW_SCORE = None
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = None
        config.REPLACEMENT_BLOCK_SAME_SECTOR = False

    def tearDown(self):
        config.REPLACEMENT_MARGIN = self._orig_margin
        config.REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED = self._orig_standalone
        config.REPLACEMENT_WEAK_FULL_PERCENTILE = self._orig_percentile
        config.REPLACEMENT_WEAK_FULL_MIN_HISTORY = self._orig_min_history
        config.REPLACEMENT_MIN_NEW_SCORE = self._orig_min_score
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = self._orig_max_pool
        config.REPLACEMENT_BLOCK_SAME_SECTOR = self._orig_block_sector

    def test_no_candidate_at_all(self):
        held = {"US.FULL": {"score_label": LABEL_FULL, "total_score": 95.0, "entry_day_idx": 0}}
        ev = evaluate_replacement("US.NEW", 90.0, held, 10)
        self.assertEqual(ev.decision, "NO_CANDIDATE")
        self.assertIsNone(ev.victim_code)

    def test_observation_replace_decision(self):
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0, "entry_day_idx": 0}}
        ev = evaluate_replacement("US.NEW", 90.0, held, 10)
        self.assertEqual(ev.decision, "REPLACE")
        self.assertEqual(ev.victim_code, "US.OBS")
        self.assertEqual(ev.replacement_type, REPL_OBSERVATION_EVICT)
        self.assertEqual(ev.old_score, 45.0)
        self.assertEqual(ev.new_score, 90.0)
        self.assertEqual(ev.score_difference, 45.0)
        self.assertEqual(ev.margin, 10.0)
        self.assertTrue(ev.margin_satisfied)

    def test_observation_keep_decision_when_margin_insufficient(self):
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 85.0, "entry_day_idx": 0}}
        ev = evaluate_replacement("US.NEW", 90.0, held, 10)
        self.assertEqual(ev.decision, "KEEP")
        self.assertEqual(ev.victim_code, "US.OBS")
        self.assertFalse(ev.margin_satisfied)

    def test_falls_back_to_weak_full_when_no_observation(self):
        history = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        held = {"US.WEAK": {"score_label": LABEL_FULL, "total_score": 30.0, "entry_day_idx": 0}}
        ev = evaluate_replacement("US.NEW", 90.0, held, 10, full_score_history=history)
        self.assertEqual(ev.decision, "REPLACE")
        self.assertEqual(ev.victim_code, "US.WEAK")
        self.assertEqual(ev.replacement_type, REPL_WEAK_FULL_EVICT)

    def test_observation_takes_priority_over_weak_full(self):
        history = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        held = {
            "US.OBS":  {"score_label": LABEL_OBSERVATION, "total_score": 45.0, "entry_day_idx": 0},
            "US.WEAK": {"score_label": LABEL_FULL, "total_score": 5.0, "entry_day_idx": 0},
        }
        ev = evaluate_replacement("US.NEW", 90.0, held, 10, full_score_history=history)
        self.assertEqual(ev.victim_code, "US.OBS")
        self.assertEqual(ev.replacement_type, REPL_OBSERVATION_EVICT)

    def test_weak_full_disabled_by_standalone_flag(self):
        config.REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED = False
        history = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        held = {"US.WEAK": {"score_label": LABEL_FULL, "total_score": 30.0, "entry_day_idx": 0}}
        ev = evaluate_replacement("US.NEW", 90.0, held, 10, full_score_history=history)
        self.assertEqual(ev.decision, "NO_CANDIDATE")


class TestEvaluateReplacementAblationGates(unittest.TestCase):
    """v2.4优化阶段三消融实验：三个可选前置过滤门，默认关闭
    （config.py里None/False），这里验证开启后的行为。"""

    def setUp(self):
        self._orig_margin = config.REPLACEMENT_MARGIN
        self._orig_min_score = config.REPLACEMENT_MIN_NEW_SCORE
        self._orig_max_pool = config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE
        self._orig_block_sector = config.REPLACEMENT_BLOCK_SAME_SECTOR
        self._orig_sector_map = dict(config.SECTOR_MAP)
        config.REPLACEMENT_MARGIN = 10.0
        config.REPLACEMENT_MIN_NEW_SCORE = None
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = None
        config.REPLACEMENT_BLOCK_SAME_SECTOR = False

    def tearDown(self):
        config.REPLACEMENT_MARGIN = self._orig_margin
        config.REPLACEMENT_MIN_NEW_SCORE = self._orig_min_score
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = self._orig_max_pool
        config.REPLACEMENT_BLOCK_SAME_SECTOR = self._orig_block_sector
        config.SECTOR_MAP.clear()
        config.SECTOR_MAP.update(self._orig_sector_map)

    def test_min_new_score_disabled_by_default_has_no_effect(self):
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0, "entry_day_idx": 0}}
        ev = evaluate_replacement("US.NEW", 60.0, held, 10)
        self.assertEqual(ev.decision, "REPLACE")

    def test_min_new_score_blocks_weak_incoming_signal(self):
        config.REPLACEMENT_MIN_NEW_SCORE = 95.0
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0, "entry_day_idx": 0}}
        ev = evaluate_replacement("US.NEW", 90.0, held, 10)   # 90 < 95 threshold
        self.assertEqual(ev.decision, "NO_CANDIDATE")

    def test_min_new_score_allows_strong_incoming_signal(self):
        config.REPLACEMENT_MIN_NEW_SCORE = 95.0
        held = {"US.OBS": {"score_label": LABEL_OBSERVATION, "total_score": 45.0, "entry_day_idx": 0}}
        ev = evaluate_replacement("US.NEW", 96.0, held, 10)
        self.assertEqual(ev.decision, "REPLACE")

    def test_max_observation_pool_size_blocks_when_pool_too_large(self):
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = 2
        held = {
            "US.A": {"score_label": LABEL_OBSERVATION, "total_score": 40.0, "entry_day_idx": 0},
            "US.B": {"score_label": LABEL_OBSERVATION, "total_score": 42.0, "entry_day_idx": 0},
            "US.C": {"score_label": LABEL_OBSERVATION, "total_score": 44.0, "entry_day_idx": 0},
        }
        ev = evaluate_replacement("US.NEW", 90.0, held, 10)   # pool size 3 > 2
        self.assertEqual(ev.decision, "NO_CANDIDATE")

    def test_max_observation_pool_size_allows_when_pool_small_enough(self):
        config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = 2
        held = {
            "US.A": {"score_label": LABEL_OBSERVATION, "total_score": 40.0, "entry_day_idx": 0},
            "US.B": {"score_label": LABEL_OBSERVATION, "total_score": 42.0, "entry_day_idx": 0},
        }
        ev = evaluate_replacement("US.NEW", 90.0, held, 10)   # pool size 2 <= 2
        self.assertEqual(ev.decision, "REPLACE")

    def test_block_same_sector_skips_same_sector_victim_but_tries_another(self):
        config.REPLACEMENT_BLOCK_SAME_SECTOR = True
        config.SECTOR_MAP["US.NEW"] = "tech"
        config.SECTOR_MAP["US.SAMESECTOR"] = "tech"
        config.SECTOR_MAP["US.OTHERSECTOR"] = "energy"
        held = {
            "US.SAMESECTOR":  {"score_label": LABEL_OBSERVATION, "total_score": 20.0, "entry_day_idx": 0},
            "US.OTHERSECTOR": {"score_label": LABEL_OBSERVATION, "total_score": 40.0, "entry_day_idx": 0},
        }
        ev = evaluate_replacement("US.NEW", 90.0, held, 10)
        self.assertEqual(ev.victim_code, "US.OTHERSECTOR")   # 同板块的US.SAMESECTOR被跳过

    def test_block_same_sector_no_candidate_when_only_same_sector_available(self):
        config.REPLACEMENT_BLOCK_SAME_SECTOR = True
        config.SECTOR_MAP["US.NEW"] = "tech"
        config.SECTOR_MAP["US.SAMESECTOR"] = "tech"
        held = {"US.SAMESECTOR": {"score_label": LABEL_OBSERVATION, "total_score": 20.0, "entry_day_idx": 0}}
        ev = evaluate_replacement("US.NEW", 90.0, held, 10)
        self.assertEqual(ev.decision, "NO_CANDIDATE")


if __name__ == "__main__":
    unittest.main()
