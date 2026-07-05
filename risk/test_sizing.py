"""
Unit tests for risk/sizing.py::calculate() -- v2.2 OBSERVATION tier wiring.
Run:  python -m unittest risk.test_sizing -v
"""
import unittest

import config
from engine.scoring import LABEL_FULL, LABEL_OBSERVATION, LABEL_PARTIAL
from risk.sizing import calculate


class TestObservationPositionSizing(unittest.TestCase):
    """
    PRD Test 1: score=50 (OBSERVATION) -> position = 3% (config.OBSERVATION_
    POSITION_PCT, independently configurable, not hardcoded).
    """

    def setUp(self):
        # Snapshot so tests that mutate config.OBSERVATION_POSITION_PCT to
        # prove the wiring (rather than a coincidental match with
        # SIZING_PCT_WEAK) don't leak into other tests.
        self._orig = config.OBSERVATION_POSITION_PCT

    def tearDown(self):
        config.OBSERVATION_POSITION_PCT = self._orig

    def test_observation_label_uses_default_3pct(self):
        qty = calculate(available_cash=100_000.0, price=100.0,
                         total_capital=100_000.0, stop_loss_pct=0.05,
                         strategy="", market_weather_code=2,
                         score_label=LABEL_OBSERVATION)
        self.assertEqual(qty, int(100_000.0 * config.OBSERVATION_POSITION_PCT / 100.0))
        self.assertEqual(qty, 30)   # 3% of 100,000 cash / $100 price

    def test_observation_position_is_independently_configurable(self):
        # Prove calculate() actually reads config.OBSERVATION_POSITION_PCT at
        # call time, not a value coincidentally equal to SIZING_PCT_WEAK.
        config.OBSERVATION_POSITION_PCT = 0.05
        qty = calculate(available_cash=100_000.0, price=100.0,
                         total_capital=100_000.0, stop_loss_pct=0.05,
                         strategy="", market_weather_code=2,
                         score_label=LABEL_OBSERVATION)
        self.assertEqual(qty, 50)   # 5% of 100,000 / $100

    def test_observation_respects_risk_cap_when_binding(self):
        # A very tight stop_loss_pct makes the risk cap smaller than the
        # observation cash tier -- OBSERVATION must still be capped by
        # RISK_PER_TRADE_PCT like every other tier (Position Layer doesn't
        # bypass existing risk caps).
        qty = calculate(available_cash=100_000.0, price=100.0,
                         total_capital=100_000.0, stop_loss_pct=0.9,
                         strategy="", market_weather_code=2,
                         score_label=LABEL_OBSERVATION)
        risk_budget = 100_000.0 * config.RISK_PER_TRADE_PCT
        expected_qty_by_risk = int(risk_budget / (100.0 * 0.9))
        self.assertEqual(qty, min(30, expected_qty_by_risk))
        self.assertLess(qty, 30)

    def test_observation_blocked_by_weather_state0(self):
        qty = calculate(available_cash=100_000.0, price=100.0,
                         total_capital=100_000.0, stop_loss_pct=0.05,
                         strategy="", market_weather_code=0,
                         score_label=LABEL_OBSERVATION)
        self.assertEqual(qty, 0)

    def test_full_and_partial_labels_unaffected_by_observation_change(self):
        # Sanity: adding the OBSERVATION branch must not perturb the existing
        # FULL/PARTIAL branches.
        full_qty = calculate(available_cash=100_000.0, price=100.0,
                              total_capital=100_000.0, stop_loss_pct=0.05,
                              strategy="", market_weather_code=2,
                              score_label=LABEL_FULL)
        partial_qty = calculate(available_cash=100_000.0, price=100.0,
                                 total_capital=100_000.0, stop_loss_pct=0.05,
                                 strategy="", market_weather_code=2,
                                 score_label=LABEL_PARTIAL)
        self.assertGreater(full_qty, partial_qty)
        self.assertGreater(partial_qty, 30)   # both stronger tiers than observation


if __name__ == "__main__":
    unittest.main()
