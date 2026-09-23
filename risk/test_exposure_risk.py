"""
Unit tests for risk/exposure_risk.py.

Run:  python -m unittest risk.test_exposure_risk -v
"""
import unittest

import config
from risk.exposure_risk import check
from risk.portfolio_state import PortfolioState, PositionSnapshot
from risk.risk_decision import OrderIntent, RiskStatus


def _state(long_mv, total_assets, cash=0.0):
    return PortfolioState(positions={}, cash=cash, total_assets=total_assets,
                           long_mv=long_mv, total_exposure_pct=long_mv / total_assets,
                           position_count=0)


class TestExposureRisk(unittest.TestCase):

    def setUp(self):
        self._orig_limit = config.RISK_ENGINE_MAX_TOTAL_EXPOSURE_PCT
        config.RISK_ENGINE_MAX_TOTAL_EXPOSURE_PCT = 0.95

    def tearDown(self):
        config.RISK_ENGINE_MAX_TOTAL_EXPOSURE_PCT = self._orig_limit

    def test_buy_within_limit_allows(self):
        state = _state(long_mv=500_000, total_assets=1_000_000)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=100, price=100.0)  # +$10k -> 51%
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_buy_pushing_projected_over_limit_blocks(self):
        state = _state(long_mv=940_000, total_assets=1_000_000)   # 94%
        order = OrderIntent(code="US.NVDA", side="BUY", qty=200, price=100.0)  # +$20k -> 96%
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)
        self.assertIn("MAX_TOTAL_EXPOSURE", decision.violations)

    def test_sell_never_blocked_even_far_over_limit(self):
        # Already at 150% exposure (way past limit) -- a SELL must still ALLOW.
        state = _state(long_mv=1_500_000, total_assets=1_000_000)
        order = OrderIntent(code="US.NVDA", side="SELL", qty=1000, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_zero_qty_allows(self):
        state = _state(long_mv=940_000, total_assets=1_000_000)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=0, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_missing_total_assets_is_data_unavailable_not_block(self):
        state = PortfolioState(total_assets=0.0, total_exposure_pct=None)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=100, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)   # fails open, never a guessed BLOCK


if __name__ == "__main__":
    unittest.main()
