"""
Unit tests for risk/beta_risk.py.

Run:  python -m unittest risk.test_beta_risk -v
"""
import unittest

import pandas as pd

import config
from risk.beta_risk import check, symbol_beta
from risk.portfolio_state import PortfolioState, PositionSnapshot
from risk.risk_decision import DATA_UNAVAILABLE, OrderIntent, RiskStatus


def _pos(code, market_val, total_assets):
    return PositionSnapshot(code=code, qty=10, market_val=market_val,
                             current_price=market_val / 10, weight=market_val / total_assets,
                             sector="other", is_core=False)


def _returns_beta_2x(n=100, seed_offset=0):
    """market moves +/-1% alternating; symbol moves exactly 2x that -> beta==2."""
    market = pd.Series([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
    symbol = market * 2.0
    return market, symbol


class TestSymbolBeta(unittest.TestCase):

    def setUp(self):
        self._orig_min_bars = config.RISK_ENGINE_BETA_MIN_BARS
        config.RISK_ENGINE_BETA_MIN_BARS = 10

    def tearDown(self):
        config.RISK_ENGINE_BETA_MIN_BARS = self._orig_min_bars

    def test_computes_expected_beta(self):
        qqq, nvda = _returns_beta_2x()
        returns = {config.QQQ_CORE_CODE: qqq, "US.NVDA": nvda}
        beta = symbol_beta(returns, "US.NVDA")
        self.assertAlmostEqual(beta, 2.0, places=6)

    def test_missing_symbol_returns_none(self):
        returns = {config.QQQ_CORE_CODE: pd.Series([0.01] * 20)}
        self.assertIsNone(symbol_beta(returns, "US.NVDA"))

    def test_missing_market_proxy_returns_none(self):
        returns = {"US.NVDA": pd.Series([0.01] * 20)}
        self.assertIsNone(symbol_beta(returns, "US.NVDA"))

    def test_too_few_bars_returns_none(self):
        config.RISK_ENGINE_BETA_MIN_BARS = 60
        qqq, nvda = _returns_beta_2x(n=20)
        returns = {config.QQQ_CORE_CODE: qqq, "US.NVDA": nvda}
        self.assertIsNone(symbol_beta(returns, "US.NVDA"))


class TestBetaCheck(unittest.TestCase):

    def setUp(self):
        self._orig_limit = config.RISK_ENGINE_MAX_PORTFOLIO_BETA
        self._orig_mode = config.RISK_ENGINE_BETA_DATA_MODE
        self._orig_min_bars = config.RISK_ENGINE_BETA_MIN_BARS
        config.RISK_ENGINE_MAX_PORTFOLIO_BETA = 1.20
        config.RISK_ENGINE_BETA_DATA_MODE = "safe"
        config.RISK_ENGINE_BETA_MIN_BARS = 10

    def tearDown(self):
        config.RISK_ENGINE_MAX_PORTFOLIO_BETA = self._orig_limit
        config.RISK_ENGINE_BETA_DATA_MODE = self._orig_mode
        config.RISK_ENGINE_BETA_MIN_BARS = self._orig_min_bars

    def test_no_price_returns_is_data_unavailable_warn_in_safe_mode(self):
        total_assets = 1_000_000
        state = PortfolioState(positions={"US.NVDA": _pos("US.NVDA", 100_000, total_assets)},
                                total_assets=total_assets, price_returns=None)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.WARN)
        self.assertIn("BETA_DATA_UNAVAILABLE", decision.warnings)

    def test_no_price_returns_blocks_in_strict_mode(self):
        config.RISK_ENGINE_BETA_DATA_MODE = "strict"
        total_assets = 1_000_000
        state = PortfolioState(positions={"US.NVDA": _pos("US.NVDA", 100_000, total_assets)},
                                total_assets=total_assets, price_returns=None)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)

    def test_projected_beta_under_limit_allows(self):
        qqq, low_beta = _returns_beta_2x()
        low_beta = qqq * 0.5   # beta 0.5
        total_assets = 1_000_000
        state = PortfolioState(positions={}, total_assets=total_assets,
                                price_returns={config.QQQ_CORE_CODE: qqq, "US.LOW": low_beta})
        order = OrderIntent(code="US.LOW", side="BUY", qty=100, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_projected_beta_over_limit_blocks(self):
        qqq, high_beta = _returns_beta_2x()   # beta 2.0
        total_assets = 1_000_000
        state = PortfolioState(positions={}, total_assets=total_assets,
                                price_returns={config.QQQ_CORE_CODE: qqq, "US.NVDA": high_beta})
        order = OrderIntent(code="US.NVDA", side="BUY", qty=3000, price=100.0)  # dominates portfolio
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)
        self.assertIn("MAX_PORTFOLIO_BETA", decision.violations)

    def test_order_symbol_missing_data_is_data_unavailable(self):
        qqq, _ = _returns_beta_2x()
        total_assets = 1_000_000
        state = PortfolioState(positions={}, total_assets=total_assets,
                                price_returns={config.QQQ_CORE_CODE: qqq})
        order = OrderIntent(code="US.UNKNOWN", side="BUY", qty=100, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.WARN)


if __name__ == "__main__":
    unittest.main()
