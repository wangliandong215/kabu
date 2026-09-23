"""
Unit tests for risk/market_regime_risk.py — thin adapter over
risk/portfolio_position_manager.py::classify_market_regime()'s existing
BULL/NORMAL/CAUTION/RISK_OFF taxonomy.

Run:  python -m unittest risk.test_market_regime_risk -v
"""
import unittest

import config
from risk.market_regime_risk import check
from risk.portfolio_state import PortfolioState, PositionSnapshot
from risk.risk_decision import OrderIntent, RiskStatus
import test_support


def _pos(code, qty=10, market_val=10_000):
    return PositionSnapshot(code=code, qty=qty, market_val=market_val,
                             current_price=market_val / qty, weight=0.01,
                             sector="other", is_core=False)


class TestMarketRegimeRisk(unittest.TestCase):

    def setUp(self):
        self._orig_restrict = config.RISK_ENGINE_REGIME_RESTRICT_NEW_POSITIONS_ON
        self._orig_warn = config.RISK_ENGINE_REGIME_WARN_ON
        config.RISK_ENGINE_REGIME_RESTRICT_NEW_POSITIONS_ON = ("RISK_OFF",)
        config.RISK_ENGINE_REGIME_WARN_ON = ("CAUTION", "UNKNOWN")

    def tearDown(self):
        config.RISK_ENGINE_REGIME_RESTRICT_NEW_POSITIONS_ON = self._orig_restrict
        config.RISK_ENGINE_REGIME_WARN_ON = self._orig_warn

    def test_bull_new_position_allows(self):
        state = PortfolioState(positions={}, market_regime="BULL")
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_risk_off_new_position_blocks(self):
        state = PortfolioState(positions={}, market_regime="RISK_OFF")
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)

    def test_caution_new_position_warns(self):
        state = PortfolioState(positions={}, market_regime="CAUTION")
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.WARN)

    def test_unset_regime_treated_as_unknown_warns(self):
        state = PortfolioState(positions={}, market_regime=None)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.WARN)

    def test_risk_off_add_to_existing_not_blocked(self):
        state = PortfolioState(positions={"US.NVDA": _pos("US.NVDA")}, market_regime="RISK_OFF")
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_risk_off_sell_never_blocked(self):
        state = PortfolioState(positions={"US.NVDA": _pos("US.NVDA")}, market_regime="RISK_OFF")
        order = OrderIntent(code="US.NVDA", side="SELL", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
