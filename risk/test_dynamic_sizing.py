"""
Unit tests for risk/dynamic_sizing.py -- V3.0-A Dynamic Position Sizing.
Run:  python -m unittest risk.test_dynamic_sizing -v
"""
import unittest

from risk.dynamic_sizing import confidence_to_position_multiplier
from risk.sizing import calculate


class TestConfidenceToPositionMultiplier(unittest.TestCase):
    """V3.0 spec section 五/八: exact boundary table."""

    def test_boundaries(self):
        cases = [
            (100, 1.0),
            (95, 1.0),
            (94, 0.8),
            (85, 0.8),
            (84, 0.6),
            (70, 0.6),
            (69, 0.4),
            (55, 0.4),
            (54, 0.0),
            (40, 0.0),
            (0, 0.0),
        ]
        for confidence, expected in cases:
            with self.subTest(confidence=confidence):
                self.assertEqual(confidence_to_position_multiplier(confidence), expected)

    def test_none_falls_back_to_1x_not_skip(self):
        # confidence unavailable (compute failed / feature disabled) must
        # never silently shrink or drop a trade -- see module docstring.
        self.assertEqual(confidence_to_position_multiplier(None), 1.0)

    def test_out_of_range_is_clipped_not_rejected(self):
        self.assertEqual(confidence_to_position_multiplier(150), 1.0)
        self.assertEqual(confidence_to_position_multiplier(-10), 0.0)

    def test_fractional_confidence_uses_lower_bound_inclusive(self):
        self.assertEqual(confidence_to_position_multiplier(94.99), 0.8)
        self.assertEqual(confidence_to_position_multiplier(95.0), 1.0)


class TestPositionCalculationWithBasePosition(unittest.TestCase):
    """V3.0 spec section 八's own toy formula: Final Position = Base Position
    x Confidence Multiplier, Base Position = $10,000. This is a direct test
    of that formula in isolation (confidence_to_position_multiplier() times
    a base dollar amount) -- NOT a call through risk/sizing.py::calculate(),
    because position_scale is only honored by calculate()'s "signal_based"
    branch (SIZING_METHOD in this codebase defaults to "signal_based" -- see
    config.py -- and that real end-to-end wiring is covered separately by
    TestDoesNotBypassExistingRiskManager below). fixed_amount/fixed_percent/
    fixed_qty do not read position_scale at all; that is a pre-existing
    limitation of risk/sizing.py, out of scope for V3.0-A to touch."""

    BASE_POSITION = 10_000.0

    def _final_position_value(self, confidence):
        return self.BASE_POSITION * confidence_to_position_multiplier(confidence)

    def test_confidence_96_full_size(self):
        self.assertEqual(self._final_position_value(96), 10_000.0)

    def test_confidence_88_80pct(self):
        self.assertEqual(self._final_position_value(88), 8_000.0)

    def test_confidence_76_60pct(self):
        self.assertEqual(self._final_position_value(76), 6_000.0)

    def test_confidence_61_40pct(self):
        self.assertEqual(self._final_position_value(61), 4_000.0)

    def test_confidence_48_skip(self):
        self.assertEqual(self._final_position_value(48), 0.0)


class TestDoesNotBypassExistingRiskManager(unittest.TestCase):
    """V3.0 spec section 三/四/八: Dynamic Position Sizing must never widen a
    position past what the existing signal_based risk caps already allow --
    it can only shrink (position_scale multiplies AFTER qty_by_tier/
    qty_by_risk/qty_by_strat are min()-ed together in risk/sizing.py)."""

    def setUp(self):
        import config
        self._orig_method = config.SIZING_METHOD
        config.SIZING_METHOD = "signal_based"

    def tearDown(self):
        import config
        config.SIZING_METHOD = self._orig_method

    def _qty(self, confidence, **kwargs):
        multiplier = confidence_to_position_multiplier(confidence)
        return calculate(position_scale=multiplier, **kwargs)

    def test_full_confidence_matches_uncapped_baseline(self):
        # multiplier=1.0 must reproduce exactly what V2.x (no dynamic sizing,
        # position_scale=1.0 implicitly) would have bought -- V3.0-A changes
        # nothing for a top-tier signal.
        kwargs = dict(available_cash=100_000.0, price=50.0,
                      signal_strength=0.9, total_capital=100_000.0,
                      stop_loss_pct=0.05, strategy="atr_breakout",
                      market_weather_code=2)
        baseline_qty = calculate(**kwargs)
        dynamic_qty = self._qty(confidence=97, **kwargs)
        self.assertEqual(baseline_qty, dynamic_qty)

    def test_lower_confidence_only_shrinks_never_grows(self):
        kwargs = dict(available_cash=100_000.0, price=50.0,
                      signal_strength=0.9, total_capital=100_000.0,
                      stop_loss_pct=0.05, strategy="atr_breakout",
                      market_weather_code=2)
        baseline_qty = calculate(**kwargs)
        for confidence in (96, 88, 76, 61, 48, 10):
            with self.subTest(confidence=confidence):
                dynamic_qty = self._qty(confidence=confidence, **kwargs)
                self.assertLessEqual(dynamic_qty, baseline_qty)

    def test_risk_per_trade_cap_still_binds_at_full_multiplier(self):
        # A tight stop makes the existing RISK_PER_TRADE_PCT cap the binding
        # constraint -- confidence=100 (multiplier=1.0) must not let the
        # position exceed that cap. Mirrors risk/test_sizing.py's own
        # equivalent check for the OBSERVATION tier.
        import config
        kwargs = dict(available_cash=1_000_000.0, price=50.0,
                      signal_strength=0.9, total_capital=1_000_000.0,
                      stop_loss_pct=0.30,   # very wide stop -> tiny risk-capped qty
                      strategy="atr_breakout", market_weather_code=2)
        # stop_loss_pct (0.30) > STOP_MAX_STRONG downgrades the cash tier to
        # SIZING_PCT_MEDIUM (see risk/sizing.py's "downgrade: stop too wide"
        # branch) -- that tier alone would still allow more shares than the
        # risk cap below, proving the risk cap (not the cash tier) binds.
        cash_tier_qty = int(1_000_000.0 * config.SIZING_PCT_MEDIUM / 50.0)
        dynamic_qty = self._qty(confidence=100, **kwargs)
        self.assertLess(dynamic_qty, cash_tier_qty)
        expected_risk_qty = int(
            (1_000_000.0 * config.RISK_PER_TRADE_PCT) / (50.0 * 0.30))
        self.assertEqual(dynamic_qty, expected_risk_qty)

    def test_zero_multiplier_yields_zero_regardless_of_caps(self):
        kwargs = dict(available_cash=100_000.0, price=50.0,
                      signal_strength=0.9, total_capital=100_000.0,
                      stop_loss_pct=0.05, strategy="atr_breakout",
                      market_weather_code=2)
        self.assertEqual(self._qty(confidence=10, **kwargs), 0)


if __name__ == "__main__":
    unittest.main()
