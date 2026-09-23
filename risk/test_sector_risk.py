"""
Unit tests for risk/sector_risk.py — projected (not current) sector
exposure, the gap guard.check_sector_exposure() left open.

Run:  python -m unittest risk.test_sector_risk -v
"""
import unittest

import config
from risk.sector_risk import check
from risk.portfolio_state import PortfolioState, PositionSnapshot
from risk.risk_decision import OrderIntent, RiskStatus


def _pos(code, market_val, total_assets, sector):
    return PositionSnapshot(code=code, qty=10, market_val=market_val,
                             current_price=market_val / 10, weight=market_val / total_assets,
                             sector=sector, is_core=False)


class TestSectorRisk(unittest.TestCase):

    def setUp(self):
        self._orig_limits = dict(config.RISK_ENGINE_MAX_SECTOR_EXPOSURE_PCT)
        self._orig_map = dict(config.SECTOR_MAP)
        self._orig_ratio = config.RISK_ENGINE_SECTOR_WARN_RATIO
        config.RISK_ENGINE_MAX_SECTOR_EXPOSURE_PCT = {"default": 0.40, "semiconductor": 0.25}
        config.SECTOR_MAP["US.NVDA"] = "semiconductor"
        config.SECTOR_MAP["US.AMD"] = "semiconductor"
        config.RISK_ENGINE_SECTOR_WARN_RATIO = 0.90

    def tearDown(self):
        config.RISK_ENGINE_MAX_SECTOR_EXPOSURE_PCT = self._orig_limits
        config.SECTOR_MAP.clear()
        config.SECTOR_MAP.update(self._orig_map)
        config.RISK_ENGINE_SECTOR_WARN_RATIO = self._orig_ratio

    def test_buy_within_sector_limit_allows(self):
        total_assets = 1_000_000
        state = PortfolioState(positions={"US.AMD": _pos("US.AMD", 100_000, total_assets, "semiconductor")},
                                total_assets=total_assets)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=100, price=100.0)  # +$10k -> 11%
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_buy_pushing_projected_sector_over_limit_blocks(self):
        total_assets = 1_000_000
        state = PortfolioState(positions={"US.AMD": _pos("US.AMD", 230_000, total_assets, "semiconductor")},
                                total_assets=total_assets)   # 23%
        order = OrderIntent(code="US.NVDA", side="BUY", qty=300, price=100.0)  # +$30k -> 26%
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)
        self.assertIn("MAX_SECTOR_EXPOSURE", decision.violations)

    def test_near_limit_warns_not_blocks(self):
        total_assets = 1_000_000
        state = PortfolioState(positions={"US.AMD": _pos("US.AMD", 220_000, total_assets, "semiconductor")},
                                total_assets=total_assets)   # 22%
        order = OrderIntent(code="US.NVDA", side="BUY", qty=100, price=100.0)  # +$10k -> 23% (>90% of 25%, <25%)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.WARN)

    def test_uses_projected_not_just_current_state(self):
        # This is exactly the gap vs. guard.check_sector_exposure(): a
        # single large order into an empty sector must be judged on what
        # it becomes, not "0% currently".
        total_assets = 1_000_000
        state = PortfolioState(positions={}, total_assets=total_assets)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=3000, price=100.0)  # $300k -> 30% > 25%
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)

    def test_sell_never_blocked(self):
        total_assets = 1_000_000
        state = PortfolioState(positions={"US.NVDA": _pos("US.NVDA", 400_000, total_assets, "semiconductor")},
                                total_assets=total_assets)   # already 40%, way over 25%
        order = OrderIntent(code="US.NVDA", side="SELL", qty=100, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_unknown_sector_falls_back_to_default(self):
        total_assets = 1_000_000
        state = PortfolioState(positions={}, total_assets=total_assets)
        order = OrderIntent(code="US.NOTMAPPED", side="BUY", qty=3000, price=100.0)  # $300k -> 30%
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)   # under default 40%


if __name__ == "__main__":
    unittest.main()
