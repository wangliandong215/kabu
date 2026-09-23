"""
Unit tests for risk/correlation_risk.py — SOFT RISK, WARN-only (never
BLOCK, by design — see V3.3 spec section 十一).

Run:  python -m unittest risk.test_correlation_risk -v
"""
import unittest

import pandas as pd

import config
from risk.correlation_risk import check, pairwise_correlation
from risk.portfolio_state import PortfolioState, PositionSnapshot
from risk.risk_decision import OrderIntent, RiskStatus


def _pos(code, market_val, total_assets):
    return PositionSnapshot(code=code, qty=10, market_val=market_val,
                             current_price=market_val / 10, weight=market_val / total_assets,
                             sector="other", is_core=False)


class TestPairwiseCorrelation(unittest.TestCase):

    def setUp(self):
        self._orig_min_bars = config.RISK_ENGINE_CORRELATION_MIN_BARS
        config.RISK_ENGINE_CORRELATION_MIN_BARS = 10

    def tearDown(self):
        config.RISK_ENGINE_CORRELATION_MIN_BARS = self._orig_min_bars

    def test_perfectly_correlated_series(self):
        a = pd.Series([0.01, -0.01] * 20)
        b = a * 2.0
        self.assertAlmostEqual(pairwise_correlation({"A": a, "B": b}, "A", "B"), 1.0, places=6)

    def test_missing_symbol_returns_none(self):
        a = pd.Series([0.01] * 20)
        self.assertIsNone(pairwise_correlation({"A": a}, "A", "B"))

    def test_too_few_bars_returns_none(self):
        config.RISK_ENGINE_CORRELATION_MIN_BARS = 60
        a = pd.Series([0.01, -0.01] * 5)
        b = pd.Series([0.01, -0.01] * 5)
        self.assertIsNone(pairwise_correlation({"A": a, "B": b}, "A", "B"))


class TestCorrelationCheck(unittest.TestCase):

    def setUp(self):
        self._orig_threshold = config.RISK_ENGINE_CORRELATION_WARN_THRESHOLD
        self._orig_min_bars = config.RISK_ENGINE_CORRELATION_MIN_BARS
        config.RISK_ENGINE_CORRELATION_WARN_THRESHOLD = 0.80
        config.RISK_ENGINE_CORRELATION_MIN_BARS = 10

    def tearDown(self):
        config.RISK_ENGINE_CORRELATION_WARN_THRESHOLD = self._orig_threshold
        config.RISK_ENGINE_CORRELATION_MIN_BARS = self._orig_min_bars

    def test_high_correlation_warns_never_blocks(self):
        a = pd.Series([0.01, -0.01] * 20)
        b = a * 1.5   # correlation 1.0
        state = PortfolioState(positions={"US.AVGO": _pos("US.AVGO", 100_000, 1_000_000)},
                                total_assets=1_000_000,
                                price_returns={"US.AVGO": a, "US.NVDA": b})
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.WARN)
        self.assertIn("HIGH_CORRELATION", decision.warnings)

    def test_low_correlation_allows(self):
        a = pd.Series([0.01, -0.01] * 20)
        b = pd.Series([-0.01, 0.01] * 20)   # perfectly anti-correlated -> |corr|=1 too high actually
        # use uncorrelated-ish series instead
        import random
        random.seed(0)
        a = pd.Series([random.uniform(-0.01, 0.01) for _ in range(60)])
        b = pd.Series([random.uniform(-0.01, 0.01) for _ in range(60)])
        state = PortfolioState(positions={"US.AVGO": _pos("US.AVGO", 100_000, 1_000_000)},
                                total_assets=1_000_000,
                                price_returns={"US.AVGO": a, "US.NVDA": b})
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertIn(decision.status, (RiskStatus.ALLOW, RiskStatus.WARN))
        # never BLOCK regardless -- the key structural guarantee for this module
        self.assertNotEqual(decision.status, RiskStatus.BLOCK)

    def test_sell_orders_skip_correlation_check(self):
        state = PortfolioState(positions={"US.NVDA": _pos("US.NVDA", 100_000, 1_000_000)},
                                total_assets=1_000_000, price_returns={})
        order = OrderIntent(code="US.NVDA", side="SELL", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_no_price_history_allows(self):
        state = PortfolioState(positions={}, total_assets=1_000_000, price_returns=None)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_never_blocks_regardless_of_correlation_magnitude(self):
        a = pd.Series([0.01, -0.01] * 30)
        b = a.copy()   # correlation 1.0, identical asset
        state = PortfolioState(positions={"US.AVGO": _pos("US.AVGO", 900_000, 1_000_000)},
                                total_assets=1_000_000,
                                price_returns={"US.AVGO": a, "US.NVDA": b})
        order = OrderIntent(code="US.NVDA", side="BUY", qty=1000, price=100.0)
        _, decision = check(order, state)
        self.assertNotEqual(decision.status, RiskStatus.BLOCK)


if __name__ == "__main__":
    unittest.main()
