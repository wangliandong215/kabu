"""
Unit tests for risk/portfolio_position_manager.py (v2.12 Portfolio Position
Manager, Phase 2) — pure-function level: regime classification and exposure
budget arithmetic. Integration-level tests (Observation Mode leaves real
order sizes untouched, ENABLED=True clamps qty, shared budget across
multiple BUY candidates in one pass) live in
engine/test_portfolio_position_manager_gate.py.

Run:  python -m unittest test_portfolio_position_manager -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import config
from risk import portfolio_position_manager as ppm


class TestClassifyMarketRegime(unittest.TestCase):

    def test_drawdown_halt_forces_risk_off_even_in_good_weather(self):
        regime = ppm.classify_market_regime(weather_code=2, qqq_above_ma=True, drawdown_halt=True)
        self.assertEqual(regime, ppm.REGIME_RISK_OFF)

    def test_crisis_weather_is_risk_off(self):
        regime = ppm.classify_market_regime(weather_code=0, qqq_above_ma=True, drawdown_halt=False)
        self.assertEqual(regime, ppm.REGIME_RISK_OFF)

    def test_chop_weather_is_caution(self):
        regime = ppm.classify_market_regime(weather_code=1, qqq_above_ma=True, drawdown_halt=False)
        self.assertEqual(regime, ppm.REGIME_CAUTION)

    def test_below_ma200_is_caution_even_in_safe_weather(self):
        regime = ppm.classify_market_regime(weather_code=2, qqq_above_ma=False, drawdown_halt=False)
        self.assertEqual(regime, ppm.REGIME_CAUTION)

    def test_safe_weather_above_ma200_is_bull(self):
        regime = ppm.classify_market_regime(weather_code=2, qqq_above_ma=True, drawdown_halt=False)
        self.assertEqual(regime, ppm.REGIME_BULL)

    def test_missing_weather_code_degrades_to_caution_not_bull(self):
        regime = ppm.classify_market_regime(weather_code=None, qqq_above_ma=True, drawdown_halt=False)
        self.assertEqual(regime, ppm.REGIME_CAUTION)

    def test_missing_qqq_above_ma_degrades_to_caution_not_bull(self):
        regime = ppm.classify_market_regime(weather_code=2, qqq_above_ma=None, drawdown_halt=False)
        self.assertEqual(regime, ppm.REGIME_CAUTION)


class TestMaxExposureForRegime(unittest.TestCase):

    def test_known_regimes_match_config_table(self):
        for regime, expected in config.PORTFOLIO_REGIME_MAX_EXPOSURE.items():
            self.assertEqual(ppm.max_exposure_for_regime(regime), expected)

    def test_unknown_regime_falls_back_to_most_conservative_tier(self):
        result = ppm.max_exposure_for_regime("SOME_FUTURE_REGIME_NOT_IN_TABLE")
        self.assertEqual(result, min(config.PORTFOLIO_REGIME_MAX_EXPOSURE.values()))


class TestComputeExposureBudget(unittest.TestCase):

    def test_arithmetic_matches_max_mv_minus_current_mv(self):
        budget = ppm.compute_exposure_budget(ppm.REGIME_CAUTION, long_mv=740_000.0, total_assets=1_000_000.0)
        self.assertAlmostEqual(budget.max_exposure_pct, 0.75)
        self.assertAlmostEqual(budget.current_exposure_pct, 0.74)
        self.assertAlmostEqual(budget.max_exposure_mv, 750_000.0)
        self.assertAlmostEqual(budget.remaining_budget, 10_000.0)

    def test_over_cap_current_exposure_yields_negative_remaining_budget(self):
        budget = ppm.compute_exposure_budget(ppm.REGIME_RISK_OFF, long_mv=900_000.0, total_assets=1_000_000.0)
        self.assertLess(budget.remaining_budget, 0)

    def test_none_total_assets_yields_none_fields_not_a_guess(self):
        budget = ppm.compute_exposure_budget(ppm.REGIME_NORMAL, long_mv=500_000.0, total_assets=None)
        self.assertIsNone(budget.remaining_budget)
        self.assertIsNone(budget.current_exposure_pct)
        self.assertIsNone(budget.max_exposure_mv)

    def test_zero_total_assets_yields_none_fields(self):
        budget = ppm.compute_exposure_budget(ppm.REGIME_NORMAL, long_mv=0.0, total_assets=0.0)
        self.assertIsNone(budget.remaining_budget)

    def test_none_long_mv_yields_none_remaining_budget(self):
        budget = ppm.compute_exposure_budget(ppm.REGIME_NORMAL, long_mv=None, total_assets=1_000_000.0)
        self.assertIsNone(budget.remaining_budget)


class TestLogAttempt(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_path = ppm._PM_LOG_PATH
        ppm._PM_LOG_PATH = Path(self._tmpdir.name) / "position_manager_log.jsonl"

    def tearDown(self):
        ppm._PM_LOG_PATH = self._orig_path
        self._tmpdir.cleanup()

    def test_writes_one_json_line_with_expected_fields(self):
        budget = ppm.compute_exposure_budget(ppm.REGIME_CAUTION, long_mv=740_000.0, total_assets=1_000_000.0)
        ppm.log_attempt(section="2C_NEW_ENTRY", code="US.TEST", regime=ppm.REGIME_CAUTION,
                         budget=budget, remaining_budget=10_000.0, requested_qty=100,
                         allowed_qty=50, applied_qty=50, price=100.0, action="REDUCE", enabled=True)
        lines = ppm._PM_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["section"], "2C_NEW_ENTRY")
        self.assertEqual(record["code"], "US.TEST")
        self.assertEqual(record["action"], "REDUCE")
        self.assertEqual(record["requested_qty"], 100)
        self.assertEqual(record["allowed_qty"], 50)
        self.assertEqual(record["applied_qty"], 50)
        self.assertAlmostEqual(record["blocked_usd"], 5_000.0)
        self.assertTrue(record["enabled"])

    def test_never_raises_when_log_dir_unwritable(self):
        ppm._PM_LOG_PATH = Path("Z:\\definitely\\not\\a\\real\\drive\\position_manager_log.jsonl")
        budget = ppm.compute_exposure_budget(ppm.REGIME_NORMAL, long_mv=0.0, total_assets=1_000_000.0)
        try:
            ppm.log_attempt(section="2C_NEW_ENTRY", code="US.TEST", regime=ppm.REGIME_NORMAL,
                             budget=budget, remaining_budget=900_000.0, requested_qty=1,
                             allowed_qty=1, applied_qty=1, price=100.0, action="ALLOW", enabled=False)
        except Exception as exc:
            self.fail(f"log_attempt() must never raise, got {exc!r}")


if __name__ == "__main__":
    unittest.main()
