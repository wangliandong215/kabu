"""
Unit tests for engine/scoring.py::compute_total_score().
Run:  python -m unittest engine.test_scoring -v
"""
import unittest

from engine.scoring import (LABEL_FULL, LABEL_OBSERVATION, LABEL_PARTIAL,
                             LABEL_SKIP, compute_total_score)


class TestComputeTotalScore(unittest.TestCase):

    def test_all_present_and_maxed_gives_full_100(self):
        r = compute_total_score(trend_strength=1.0, weather_code=2,
                                 fundamental_score=100, news_score=100)
        self.assertEqual(r.total, 100.0)
        self.assertEqual(r.position_scale, 1.0)
        self.assertEqual(r.label, LABEL_FULL)

    def test_weather_zero_hard_vetoes_even_with_strong_signal(self):
        r = compute_total_score(trend_strength=1.0, weather_code=0,
                                 fundamental_score=100, news_score=100)
        self.assertEqual(r.position_scale, 0.0)
        self.assertEqual(r.label, LABEL_SKIP)
        self.assertEqual(r.weather_score, 0.0)

    def test_state1_and_state2_score_identically(self):
        r1 = compute_total_score(trend_strength=0.5, weather_code=1,
                                  fundamental_score=70, news_score=70)
        r2 = compute_total_score(trend_strength=0.5, weather_code=2,
                                  fundamental_score=70, news_score=70)
        self.assertEqual(r1.total, r2.total)
        self.assertEqual(r1.position_scale, r2.position_scale)

    # ── 缺失数据剔除 + 重新归一化 ────────────────────────────────────────────

    def test_missing_fundamental_and_news_excluded_not_defaulted(self):
        # trend+weather only (backtest without --news-file): weight 0.4/0.6
        # trend + 0.2/0.6 weather, NOT diluted by any placeholder score.
        r = compute_total_score(trend_strength=0.5, weather_code=2,
                                 fundamental_score=None, news_score=None)
        expected = (50.0 * 0.4 + 100.0 * 0.2) / 0.6
        self.assertAlmostEqual(r.total, expected, places=2)
        self.assertIsNone(r.fundamental_score)
        self.assertIsNone(r.news_score)

    def test_missing_data_reproduces_old_medium_tier_boundary_0_4(self):
        r = compute_total_score(trend_strength=0.4, weather_code=2,
                                 fundamental_score=None, news_score=None)
        self.assertAlmostEqual(r.total, 60.0, places=2)
        self.assertEqual(r.label, LABEL_PARTIAL)   # >=60 is not SKIP

    def test_missing_data_reproduces_old_strong_tier_boundary_0_7(self):
        r = compute_total_score(trend_strength=0.7, weather_code=2,
                                 fundamental_score=None, news_score=None)
        self.assertAlmostEqual(r.total, 80.0, places=2)
        self.assertEqual(r.label, LABEL_FULL)

    def test_missing_data_below_old_weak_boundary_now_observation(self):
        # v2.2: this used to fall straight to SKIP in v2.1 (total<60 hard
        # filter). It's now OBSERVATION (40<=total<60) — restoring the old
        # v2.0 3% observation-position behavior for weak technical signals.
        r = compute_total_score(trend_strength=0.3, weather_code=2,
                                 fundamental_score=None, news_score=None)
        self.assertLess(r.total, 60.0)
        self.assertGreaterEqual(r.total, 40.0)
        self.assertEqual(r.label, LABEL_OBSERVATION)

    def test_trend_strength_clamped_to_0_1(self):
        r_over = compute_total_score(trend_strength=1.5, weather_code=2,
                                      fundamental_score=100, news_score=100)
        r_exact = compute_total_score(trend_strength=1.0, weather_code=2,
                                       fundamental_score=100, news_score=100)
        self.assertEqual(r_over.total, r_exact.total)

    # ── PARTIAL 档折扣只在 fund/news 至少一项真实参与时生效（2026-07-04，
    #    用户二次确认：两项都缺失时应等同旧medium档满仓，不做连续折算）──────

    def test_partial_band_no_discount_when_both_fund_and_news_missing(self):
        # trend=0.5, weather=100, fund/news both None -> total=83.3(FULL)...
        # pick a trend value that actually lands in PARTIAL with both missing.
        r = compute_total_score(trend_strength=0.5, weather_code=2,
                                 fundamental_score=None, news_score=None)
        self.assertEqual(r.label, LABEL_PARTIAL)
        self.assertEqual(r.position_scale, 1.0)   # no discount -- old medium-tier parity

    def test_partial_band_discounts_when_news_present(self):
        # Same trend/weather as above, but news_score is real (e.g. --news-file
        # supplied a real classification) -> discount should apply for real.
        r = compute_total_score(trend_strength=0.5, weather_code=2,
                                 fundamental_score=None, news_score=50)
        self.assertEqual(r.label, LABEL_PARTIAL)
        self.assertAlmostEqual(r.position_scale, r.total / 100.0, places=4)
        self.assertLess(r.position_scale, 1.0)

    def test_partial_band_discounts_when_fundamental_present(self):
        r = compute_total_score(trend_strength=0.5, weather_code=2,
                                 fundamental_score=50, news_score=None)
        self.assertEqual(r.label, LABEL_PARTIAL)
        self.assertAlmostEqual(r.position_scale, r.total / 100.0, places=4)
        self.assertLess(r.position_scale, 1.0)

    def test_full_band_always_scale_1_regardless_of_missing_data(self):
        r = compute_total_score(trend_strength=1.0, weather_code=2,
                                 fundamental_score=None, news_score=None)
        self.assertEqual(r.label, LABEL_FULL)
        self.assertEqual(r.position_scale, 1.0)

    def test_skip_band_scale_0_regardless_of_missing_data(self):
        r = compute_total_score(trend_strength=0.0, weather_code=2,
                                 fundamental_score=None, news_score=None)
        self.assertEqual(r.label, LABEL_SKIP)
        self.assertEqual(r.position_scale, 0.0)

    # ── v2.2 四档状态机：OBSERVATION（40-59分）──────────────────────────────
    # PRD 边界用例：total=50 -> OBSERVATION；total=39 -> SKIP；total=60 ->
    # PARTIAL（见上面 test_missing_data_reproduces_old_medium_tier_boundary_0_4）；
    # total=80 -> FULL（见 test_missing_data_reproduces_old_strong_tier_boundary_0_7）。
    # 注意：OBSERVATION 的具体3%仓位比例由 risk/sizing.py 的
    # config.OBSERVATION_POSITION_PCT 决定，不是这里的 position_scale——
    # Decision Layer 只判断状态，position_scale=1.0 表示不做额外折算。

    def test_total_score_50_is_observation(self):
        # trend_strength=0.25, weather=100, fund/news None -> total=50.0
        r = compute_total_score(trend_strength=0.25, weather_code=2,
                                 fundamental_score=None, news_score=None)
        self.assertAlmostEqual(r.total, 50.0, places=2)
        self.assertEqual(r.label, LABEL_OBSERVATION)
        self.assertEqual(r.position_scale, 1.0)

    def test_total_score_39_is_skip(self):
        # trend_strength=0.085, weather=100, fund/news None -> total=39.0
        r = compute_total_score(trend_strength=0.085, weather_code=2,
                                 fundamental_score=None, news_score=None)
        self.assertAlmostEqual(r.total, 39.0, places=2)
        self.assertEqual(r.label, LABEL_SKIP)
        self.assertEqual(r.position_scale, 0.0)

    def test_observation_lower_boundary_40_inclusive(self):
        # trend_strength=0.10 -> total=40.0 exactly -> OBSERVATION, not SKIP
        r = compute_total_score(trend_strength=0.10, weather_code=2,
                                 fundamental_score=None, news_score=None)
        self.assertAlmostEqual(r.total, 40.0, places=2)
        self.assertEqual(r.label, LABEL_OBSERVATION)

    def test_just_below_observation_boundary_is_skip(self):
        r = compute_total_score(trend_strength=0.099, weather_code=2,
                                 fundamental_score=None, news_score=None)
        self.assertLess(r.total, 40.0)
        self.assertEqual(r.label, LABEL_SKIP)

    def test_weather_zero_vetoes_observation_band_too(self):
        # trend_strength=0.7, fund/news None: with weather contributing its
        # normal 100 this would land at total=(70*0.4+100*0.2)/0.6=80.0 (FULL);
        # forcing weather_code=0 instead makes weather_score=0, so total drops
        # to (70*0.4+0*0.2)/0.6=46.67 -- numerically inside the OBSERVATION
        # band -- but the weather_code==0 hard veto must still force SKIP.
        r = compute_total_score(trend_strength=0.7, weather_code=0,
                                 fundamental_score=None, news_score=None)
        self.assertGreaterEqual(r.total, 40.0)
        self.assertLess(r.total, 60.0)
        self.assertEqual(r.label, LABEL_SKIP)
        self.assertEqual(r.position_scale, 0.0)


if __name__ == "__main__":
    unittest.main()
