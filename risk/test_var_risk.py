"""
Unit tests for risk/var_risk.py.

Run:  python -m unittest risk.test_var_risk -v
"""
import unittest

import pandas as pd

import config
from risk.var_risk import (HistoricalVarCalculator, ParametricVarCalculator, check,
                            _weighted_portfolio_returns)
from risk.portfolio_state import PortfolioState, PositionSnapshot
from risk.risk_decision import DATA_UNAVAILABLE, OrderIntent, RiskStatus


def _pos(code, market_val, total_assets):
    return PositionSnapshot(code=code, qty=10, market_val=market_val,
                             current_price=market_val / 10, weight=market_val / total_assets,
                             sector="other", is_core=False)


class TestHistoricalVarCalculator(unittest.TestCase):
    def test_95pct_var_is_5th_percentile_loss(self):
        # 100 days: worst 10 days are -10%, rest are +0.1% -- the 5th
        # percentile (1 - 0.95 confidence) falls inside that negative tail.
        returns = pd.Series([-0.10] * 10 + [0.001] * 90)
        var = HistoricalVarCalculator().compute(returns, confidence=0.95)
        self.assertGreater(var, 0.0)

    def test_empty_series_raises(self):
        with self.assertRaises(ValueError):
            HistoricalVarCalculator().compute(pd.Series([], dtype=float), confidence=0.95)


class TestParametricVarCalculator(unittest.TestCase):
    def test_zero_vol_gives_zero_var(self):
        returns = pd.Series([0.0] * 50)
        var = ParametricVarCalculator().compute(returns, confidence=0.95)
        self.assertAlmostEqual(var, 0.0, places=6)

    def test_insufficient_data_raises(self):
        with self.assertRaises(ValueError):
            ParametricVarCalculator().compute(pd.Series([0.01]), confidence=0.95)


class TestWeightedPortfolioReturns(unittest.TestCase):
    def test_no_price_returns_is_none(self):
        state = PortfolioState(positions={"US.A": _pos("US.A", 100_000, 1_000_000)},
                                total_assets=1_000_000, price_returns=None)
        self.assertIsNone(_weighted_portfolio_returns(state))

    def test_missing_a_held_symbols_series_is_none(self):
        state = PortfolioState(positions={"US.A": _pos("US.A", 100_000, 1_000_000)},
                                total_assets=1_000_000,
                                price_returns={})   # US.A missing
        self.assertIsNone(_weighted_portfolio_returns(state))

    def test_empty_portfolio_returns_zero_series(self):
        state = PortfolioState(positions={}, total_assets=1_000_000, price_returns={"US.A": pd.Series([0.01])})
        result = _weighted_portfolio_returns(state)
        self.assertIsNotNone(result)


class TestVarCheck(unittest.TestCase):

    def setUp(self):
        self._orig = {
            "warn": config.RISK_ENGINE_VAR_WARN_PCT,
            "block": config.RISK_ENGINE_VAR_BLOCK_PCT,
            "method": config.RISK_ENGINE_VAR_METHOD,
            "confidence": config.RISK_ENGINE_VAR_CONFIDENCE,
        }
        config.RISK_ENGINE_VAR_WARN_PCT = 0.03
        config.RISK_ENGINE_VAR_BLOCK_PCT = None
        config.RISK_ENGINE_VAR_METHOD = "parametric"
        config.RISK_ENGINE_VAR_CONFIDENCE = 0.95

    def tearDown(self):
        config.RISK_ENGINE_VAR_WARN_PCT = self._orig["warn"]
        config.RISK_ENGINE_VAR_BLOCK_PCT = self._orig["block"]
        config.RISK_ENGINE_VAR_METHOD = self._orig["method"]
        config.RISK_ENGINE_VAR_CONFIDENCE = self._orig["confidence"]

    def test_missing_data_is_warn_never_block(self):
        state = PortfolioState(positions={"US.A": _pos("US.A", 100_000, 1_000_000)},
                                total_assets=1_000_000, price_returns=None)
        order = OrderIntent(code="US.A", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.WARN)

    def test_low_vol_allows(self):
        returns = {"US.A": pd.Series([0.0001] * 50)}
        state = PortfolioState(positions={"US.A": _pos("US.A", 100_000, 1_000_000)},
                                total_assets=1_000_000, price_returns=returns)
        order = OrderIntent(code="US.A", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_high_vol_warns_above_threshold(self):
        returns = {"US.A": pd.Series([0.10, -0.10] * 25)}   # 10% daily swings
        state = PortfolioState(positions={"US.A": _pos("US.A", 1_000_000, 1_000_000)},
                                total_assets=1_000_000, price_returns=returns)
        order = OrderIntent(code="US.A", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.WARN)

    def test_block_pct_none_by_default_never_blocks(self):
        # Even extreme vol must never BLOCK while RISK_ENGINE_VAR_BLOCK_PCT is None.
        returns = {"US.A": pd.Series([0.50, -0.50] * 25)}
        state = PortfolioState(positions={"US.A": _pos("US.A", 1_000_000, 1_000_000)},
                                total_assets=1_000_000, price_returns=returns)
        order = OrderIntent(code="US.A", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertNotEqual(decision.status, RiskStatus.BLOCK)

    def test_block_pct_configured_can_block(self):
        config.RISK_ENGINE_VAR_BLOCK_PCT = 0.05
        returns = {"US.A": pd.Series([0.50, -0.50] * 25)}
        state = PortfolioState(positions={"US.A": _pos("US.A", 1_000_000, 1_000_000)},
                                total_assets=1_000_000, price_returns=returns)
        order = OrderIntent(code="US.A", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)


if __name__ == "__main__":
    unittest.main()
