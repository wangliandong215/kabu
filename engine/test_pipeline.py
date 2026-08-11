"""
engine/test_pipeline.py -- regression coverage for the Two-Phase Execution
refactor (engine/pipeline.py::build_candidate_pool / rank_candidates).

Pins down the exact bug this refactor fixes: 2026-07-29's US.BKNG vs
US.ABNB case, where both scored 100/FULL but BKNG won purely because it
ranked higher on raw signal_strength and reached the sole free slot before
ABNB's total_score was even computed. rank_candidates() must order by
total_score first regardless of `results` dict iteration / signal_strength
order, and build_candidate_pool() must compute total_score for every
surviving candidate up front (not lazily, not short-circuited by scan
order).
"""
import unittest
from unittest import mock

import config
from engine import news_filter, fundamental
from engine.pipeline import Candidate, build_candidate_pool, rank_candidates


class _FakePortfolio:
    """Minimal stand-in for portfolio.tracker.Portfolio — build_candidate_pool
    only calls get_position()/is_cooldown() (the Global Filters that don't
    mutate state), never anything else."""
    def __init__(self, held=None, cooldowns=None):
        self._held = held or set()
        self._cooldowns = cooldowns or set()

    def get_position(self, code):
        return {"code": code} if code in self._held else None

    def is_cooldown(self, code):
        return code in self._cooldowns


def _signal(code, strength, atr=1.0, price=100.0):
    return {"code": code, "signal": "BUY", "signal_strength": strength,
            "current_price": price, "atr": atr, "strategy_used": "atr_breakout"}


class TestRankCandidatesOrdering(unittest.TestCase):
    """rank_candidates() itself: pure sort, no mocking needed."""

    def _cand(self, code, total_score, signal_strength):
        return Candidate(code=code, strategy="atr_breakout",
                         signal_strength=signal_strength, current_price=100.0,
                         total_score=total_score, score_label="FULL",
                         score_components={"position_scale": 1.0})

    def test_total_score_wins_over_scan_order(self):
        # ABNB has a HIGHER total_score than BKNG but appears second in the
        # input list (mirrors it being later in `results` dict / scan order)
        # -- under the old signal_strength-only ranking this would have lost.
        bkng = self._cand("US.BKNG", total_score=90.0, signal_strength=0.95)
        abnb = self._cand("US.ABNB", total_score=100.0, signal_strength=0.60)
        ranked = rank_candidates([bkng, abnb])
        self.assertEqual([c.code for c in ranked], ["US.ABNB", "US.BKNG"])

    def test_tied_total_score_falls_back_to_signal_strength(self):
        # The literal 2026-07-29 case: both total_score=100/FULL. Tie-break
        # must be deterministic (signal_strength), not input-order-dependent.
        bkng = self._cand("US.BKNG", total_score=100.0, signal_strength=0.95)
        abnb = self._cand("US.ABNB", total_score=100.0, signal_strength=0.60)
        ranked_forward = rank_candidates([bkng, abnb])
        ranked_reversed = rank_candidates([abnb, bkng])
        self.assertEqual([c.code for c in ranked_forward], ["US.BKNG", "US.ABNB"])
        self.assertEqual([c.code for c in ranked_reversed], ["US.BKNG", "US.ABNB"])

    def test_full_tie_is_stable_not_business_rule_broken(self):
        """v1 explicitly does NOT add breakout_age/ATR tie-break. When
        total_score AND signal_strength are both identical, rank_candidates()
        must fall back to a stable sort -- i.e. preserve the input list's
        original order (Candidate Pool order) -- rather than reordering via
        any dict/set hashing. Run the same input twice and from two
        different input orders to prove it's deterministic, not
        incidentally consistent."""
        a = self._cand("US.AAA", total_score=100.0, signal_strength=1.0)
        b = self._cand("US.BBB", total_score=100.0, signal_strength=1.0)
        c = self._cand("US.CCC", total_score=100.0, signal_strength=1.0)

        self.assertEqual([x.code for x in rank_candidates([a, b, c])],
                         ["US.AAA", "US.BBB", "US.CCC"])
        # Different input order -> different (but still input-order-preserving) output.
        self.assertEqual([x.code for x in rank_candidates([c, a, b])],
                         ["US.CCC", "US.AAA", "US.BBB"])
        # Re-running with the same input is byte-for-byte reproducible.
        self.assertEqual([x.code for x in rank_candidates([a, b, c])],
                         [x.code for x in rank_candidates([a, b, c])])


class TestBuildCandidatePool(unittest.TestCase):

    def setUp(self):
        self._orig_min_strength = config.MIN_ENTRY_STRENGTH
        self._orig_classify = news_filter.classify_code
        self._orig_score = fundamental.score
        self._orig_blackout = None

    def tearDown(self):
        config.MIN_ENTRY_STRENGTH = self._orig_min_strength
        news_filter.classify_code = self._orig_classify
        fundamental.score = self._orig_score

    def test_all_survivors_get_scored_even_past_capacity(self):
        """The core fix: build_candidate_pool must NOT stop scoring once some
        notional capacity is full -- it doesn't even know about capacity.
        Both BKNG and ABNB must come back with a real total_score, unlike the
        old greedy loop where ABNB would never have reached the scoring step
        once BKNG had already taken the last slot. Numbers are chosen so both
        land on the same non-ceiling total_score (84) via different
        trend/fundamental/news mixes -- a tie that isn't just "everything
        maxed out", to show the tie-break is real, not degenerate."""
        results = {
            "US.BKNG": _signal("US.BKNG", strength=1.0),   # trend=100
            "US.ABNB": _signal("US.ABNB", strength=0.60),  # trend=60
        }
        portfolio = _FakePortfolio()

        def fake_news(code):
            return {"score": 60.0 if code == "US.BKNG" else 100.0, "tier": 1, "matched_keyword": None}

        def fake_fund(code, env):
            return {"score": 60.0 if code == "US.BKNG" else 100.0, "tier": 1, "reason": "ok"}

        news_filter.classify_code = fake_news
        fundamental.score = fake_fund

        with mock.patch("engine.pipeline.is_earnings_blackout", return_value=False):
            pool = build_candidate_pool(results, portfolio, macro_block=None,
                                        score_env="paper", weather_code=2)

        codes = {c.code for c in pool}
        self.assertEqual(codes, {"US.BKNG", "US.ABNB"})
        scores = {c.code: c.total_score for c in pool}
        self.assertEqual(scores["US.BKNG"], scores["US.ABNB"])
        for c in pool:
            self.assertEqual(c.score_label, "FULL")

        ranked = rank_candidates(pool)
        # Tied on total_score -> falls back to signal_strength -> BKNG (1.0) wins.
        self.assertEqual(ranked[0].code, "US.BKNG")

    def test_higher_total_score_overtakes_lower_signal_strength(self):
        """If ABNB's REAL total_score (once fundamental/news are counted) is
        actually higher than BKNG's, ABNB must rank first even though BKNG has
        the higher raw signal_strength and would have scanned/ranked first
        under the old signal_strength-only ordering."""
        results = {
            "US.BKNG": _signal("US.BKNG", strength=0.95),
            "US.ABNB": _signal("US.ABNB", strength=0.55),
        }
        portfolio = _FakePortfolio()

        def fake_news(code):
            return {"score": 40.0 if code == "US.BKNG" else 100.0, "tier": 1, "matched_keyword": None}

        def fake_fund(code, env):
            return {"score": 40.0 if code == "US.BKNG" else 100.0, "tier": 1, "reason": "x"}

        news_filter.classify_code = fake_news
        fundamental.score = fake_fund

        with mock.patch("engine.pipeline.is_earnings_blackout", return_value=False):
            pool = build_candidate_pool(results, portfolio, macro_block=None,
                                        score_env="paper", weather_code=2)
        ranked = rank_candidates(pool)
        self.assertEqual(ranked[0].code, "US.ABNB")

    def test_already_held_and_cooldown_and_weak_strength_are_filtered(self):
        results = {
            "US.HELD":   _signal("US.HELD", strength=0.9),
            "US.COOL":   _signal("US.COOL", strength=0.9),
            "US.WEAK":   _signal("US.WEAK", strength=0.01),
            "US.OK":     _signal("US.OK", strength=0.9),
        }
        portfolio = _FakePortfolio(held={"US.HELD"}, cooldowns={"US.COOL"})
        news_filter.classify_code = lambda code: {"score": 100.0, "tier": 1, "matched_keyword": None}
        fundamental.score = lambda code, env: {"score": 100.0, "tier": 1, "reason": "ok"}

        with mock.patch("engine.pipeline.is_earnings_blackout", return_value=False):
            pool = build_candidate_pool(results, portfolio, macro_block=None,
                                        score_env="paper", weather_code=2,
                                        min_strength=0.3)
        self.assertEqual({c.code for c in pool}, {"US.OK"})

    def test_macro_block_returns_empty_pool(self):
        results = {"US.OK": _signal("US.OK", strength=0.9)}
        portfolio = _FakePortfolio()
        pool = build_candidate_pool(results, portfolio,
                                    macro_block="MAX_DRAWDOWN_HALT: test",
                                    score_env="paper", weather_code=2)
        self.assertEqual(pool, [])


if __name__ == "__main__":
    unittest.main()
