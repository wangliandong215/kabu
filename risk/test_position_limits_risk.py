"""
Unit tests for risk/position_limits_risk.py — MAX_POSITIONS and
MAX_POSITION_WEIGHT, both on the PROJECTED portfolio.

Run:  python -m unittest risk.test_position_limits_risk -v
"""
import unittest

import config
from risk.position_limits_risk import check_position_count, check_position_weight
from risk.portfolio_state import PortfolioState, PositionSnapshot
from risk.risk_decision import OrderIntent, RiskStatus


def _pos(code, qty, market_val, total_assets, sector="other", is_core=False):
    return PositionSnapshot(code=code, qty=qty, market_val=market_val,
                             current_price=market_val / qty if qty else 0.0,
                             weight=market_val / total_assets, sector=sector, is_core=is_core)


class TestPositionCount(unittest.TestCase):

    def setUp(self):
        self._orig = config.RISK_ENGINE_MAX_POSITIONS
        config.RISK_ENGINE_MAX_POSITIONS = 2

    def tearDown(self):
        config.RISK_ENGINE_MAX_POSITIONS = self._orig

    def _state(self, codes, total_assets=1_000_000):
        positions = {c: _pos(c, 10, 10_000, total_assets) for c in codes}
        return PortfolioState(positions=positions, total_assets=total_assets,
                               position_count=len(codes))

    def test_new_open_within_limit_allows(self):
        state = self._state(["US.AAA"])   # 1 held, limit 2
        order = OrderIntent(code="US.BBB", side="BUY", qty=10, price=100.0)
        _, decision = check_position_count(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_new_open_at_limit_blocks(self):
        state = self._state(["US.AAA", "US.BBB"])   # 2 held, limit 2
        order = OrderIntent(code="US.CCC", side="BUY", qty=10, price=100.0)
        _, decision = check_position_count(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)
        self.assertIn("MAX_POSITIONS", decision.violations)

    def test_add_to_existing_position_never_blocked_by_count(self):
        state = self._state(["US.AAA", "US.BBB"])   # already at limit
        order = OrderIntent(code="US.AAA", side="BUY", qty=10, price=100.0)   # add, not new
        _, decision = check_position_count(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_qqq_core_excluded_from_count(self):
        total_assets = 1_000_000
        positions = {
            "US.AAA": _pos("US.AAA", 10, 10_000, total_assets),
            "US.BBB": _pos("US.BBB", 10, 10_000, total_assets),
            config.QQQ_CORE_CODE: _pos(config.QQQ_CORE_CODE, 100, 100_000, total_assets, is_core=True),
        }
        state = PortfolioState(positions=positions, total_assets=total_assets, position_count=2)
        order = OrderIntent(code="US.CCC", side="BUY", qty=10, price=100.0)
        _, decision = check_position_count(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)   # 2 non-core already at limit

    def test_full_close_never_blocked(self):
        state = self._state(["US.AAA", "US.BBB"])
        order = OrderIntent(code="US.AAA", side="SELL", qty=10, price=100.0)
        _, decision = check_position_count(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_zero_qty_allows(self):
        state = self._state(["US.AAA", "US.BBB"])
        order = OrderIntent(code="US.CCC", side="BUY", qty=0, price=100.0)
        _, decision = check_position_count(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)


class TestPositionWeight(unittest.TestCase):

    def setUp(self):
        self._orig = config.RISK_ENGINE_MAX_POSITION_WEIGHT_PCT
        config.RISK_ENGINE_MAX_POSITION_WEIGHT_PCT = 0.10

    def tearDown(self):
        config.RISK_ENGINE_MAX_POSITION_WEIGHT_PCT = self._orig

    def test_buy_within_limit_allows(self):
        total_assets = 1_000_000
        state = PortfolioState(positions={"US.NVDA": _pos("US.NVDA", 100, 80_000, total_assets)},
                                total_assets=total_assets)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=100, price=100.0)  # +$10k -> 9%
        _, decision = check_position_weight(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_buy_pushing_projected_weight_over_limit_blocks(self):
        total_assets = 1_000_000
        state = PortfolioState(positions={"US.NVDA": _pos("US.NVDA", 100, 95_000, total_assets)},
                                total_assets=total_assets)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=100, price=100.0)  # +$10k -> 10.5%
        _, decision = check_position_weight(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)
        self.assertIn("MAX_POSITION_WEIGHT", decision.violations)

    def test_new_position_uses_projected_not_current_zero(self):
        # V3.3's whole point vs. guard.check_sector_exposure(): a brand-new
        # position must be evaluated on what it WOULD become, not "0% now".
        total_assets = 1_000_000
        state = PortfolioState(positions={}, total_assets=total_assets)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=2000, price=100.0)  # $200k -> 20%
        _, decision = check_position_weight(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)

    def test_partial_close_never_blocked(self):
        total_assets = 1_000_000
        state = PortfolioState(positions={"US.NVDA": _pos("US.NVDA", 2000, 300_000, total_assets)},
                                total_assets=total_assets)   # 30%, already over any sane limit
        order = OrderIntent(code="US.NVDA", side="SELL", qty=500, price=150.0)
        _, decision = check_position_weight(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_full_close_never_blocked(self):
        total_assets = 1_000_000
        state = PortfolioState(positions={"US.NVDA": _pos("US.NVDA", 2000, 300_000, total_assets)},
                                total_assets=total_assets)
        order = OrderIntent(code="US.NVDA", side="SELL", qty=2000, price=150.0)
        _, decision = check_position_weight(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_missing_total_assets_is_data_unavailable_not_block(self):
        state = PortfolioState(total_assets=0.0)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=100, price=100.0)
        _, decision = check_position_weight(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)


if __name__ == "__main__":
    unittest.main()
