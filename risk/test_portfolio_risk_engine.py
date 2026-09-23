"""
Unit tests for risk/portfolio_risk_engine.py — aggregation (status
precedence, violations/warnings/metrics merge, never-raises), and the V3.3
spec section 二十 requirement that SELL/close orders can never be BLOCKed
by any hard-limit check, exercised end-to-end through the real engine with
every limit configured to be maximally restrictive.

Run:  python -m unittest risk.test_portfolio_risk_engine -v
"""
import unittest
from unittest import mock

import pandas as pd

import config
import risk.portfolio_risk_engine as portfolio_risk_engine
from risk.portfolio_risk_engine import PortfolioRiskEngine, evaluate_order
from risk.portfolio_state import PortfolioState, PositionSnapshot
from risk.risk_decision import OrderIntent, RiskStatus, allow
import test_support


def _pos(code, qty, market_val, total_assets, sector="semiconductor", is_core=False):
    return PositionSnapshot(code=code, qty=qty, market_val=market_val,
                             current_price=market_val / qty, weight=market_val / total_assets,
                             sector=sector, is_core=is_core)


def _flat_returns(n=100):
    """Low, identical volatility for QQQ and any symbol -- keeps beta_risk/
    var_risk fully computable (no DATA_UNAVAILABLE) so a test can assert a
    clean ALLOW across every sub-check."""
    series = pd.Series([0.0005 if i % 2 == 0 else -0.0005 for i in range(n)])
    return {config.QQQ_CORE_CODE: series, "US.NVDA": series.copy()}


class TestAggregation(unittest.TestCase):

    def test_all_allow_yields_allow(self):
        # NVDA's synthetic return series is identical to QQQ's here (see
        # _flat_returns()), which would otherwise trip correlation_risk's
        # WARN on its own -- push the threshold out of reach so this test
        # isolates "every OTHER sub-check is clean ALLOW".
        orig_corr = config.RISK_ENGINE_CORRELATION_WARN_THRESHOLD
        config.RISK_ENGINE_CORRELATION_WARN_THRESHOLD = 1.1
        try:
            state = PortfolioState(positions={}, total_assets=1_000_000, cash=1_000_000,
                                    long_mv=0.0, total_exposure_pct=0.0, position_count=0,
                                    market_regime="BULL", price_returns=_flat_returns())
            order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
            decision = PortfolioRiskEngine().evaluate(order, state)
            self.assertEqual(decision.status, RiskStatus.ALLOW)
            self.assertTrue(decision.allowed)
        finally:
            config.RISK_ENGINE_CORRELATION_WARN_THRESHOLD = orig_corr

    def test_any_block_wins_over_warn(self):
        orig_limit = config.RISK_ENGINE_MAX_TOTAL_EXPOSURE_PCT
        orig_regime = config.RISK_ENGINE_REGIME_WARN_ON
        try:
            config.RISK_ENGINE_MAX_TOTAL_EXPOSURE_PCT = 0.10   # will BLOCK
            state = PortfolioState(positions={}, total_assets=1_000_000, cash=1_000_000,
                                    long_mv=0.0, total_exposure_pct=0.0, position_count=0,
                                    market_regime="CAUTION")   # will WARN
            order = OrderIntent(code="US.NVDA", side="BUY", qty=5000, price=100.0)
            decision = PortfolioRiskEngine().evaluate(order, state)
            self.assertEqual(decision.status, RiskStatus.BLOCK)
            self.assertFalse(decision.allowed)
            self.assertIn("MAX_TOTAL_EXPOSURE", decision.violations)
        finally:
            config.RISK_ENGINE_MAX_TOTAL_EXPOSURE_PCT = orig_limit
            config.RISK_ENGINE_REGIME_WARN_ON = orig_regime

    def test_sub_check_exception_downgrades_to_warn_never_raises(self):
        state = PortfolioState(positions={}, total_assets=1_000_000, cash=1_000_000,
                                long_mv=0.0, total_exposure_pct=0.0, position_count=0,
                                market_regime="BULL")
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)

        def _raiser(order, state):
            raise RuntimeError("boom")

        # _SUB_CHECKS binds each check function by reference at import time,
        # so patching risk.beta_risk.check afterward wouldn't reach it --
        # replace the whole tuple instead.
        patched = tuple((name, _raiser if name == "beta" else fn)
                         for name, fn in portfolio_risk_engine._SUB_CHECKS)
        with mock.patch.object(portfolio_risk_engine, "_SUB_CHECKS", patched):
            decision = PortfolioRiskEngine().evaluate(order, state)   # must not raise
        self.assertIn("BETA_CHECK_FAILED", decision.warnings)

    def test_never_raises_even_if_logging_fails(self):
        state = PortfolioState(positions={}, total_assets=1_000_000, market_regime="BULL")
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        with mock.patch("builtins.open", side_effect=OSError("disk full")):
            decision = PortfolioRiskEngine().evaluate(order, state)   # must not raise
        self.assertIsNotNone(decision)


class TestEvaluateOrderEntryPoint(unittest.TestCase):

    def test_disabled_returns_allow_without_evaluating(self):
        orig = config.RISK_ENGINE_ENABLED
        try:
            config.RISK_ENGINE_ENABLED = False
            decision = evaluate_order("US.NVDA", "BUY", 10, 100.0, PortfolioState())
            self.assertEqual(decision.status, RiskStatus.ALLOW)
        finally:
            config.RISK_ENGINE_ENABLED = orig

    def test_off_mode_returns_allow(self):
        orig = config.RISK_ENGINE_MODE
        try:
            config.RISK_ENGINE_MODE = "OFF"
            decision = evaluate_order("US.NVDA", "BUY", 10, 100.0, PortfolioState())
            self.assertEqual(decision.status, RiskStatus.ALLOW)
        finally:
            config.RISK_ENGINE_MODE = orig


class TestSellNeverBlockedByHardLimits(unittest.TestCase):
    """V3.3 spec section二十: 'SELL和平仓订单不能被最大仓位等规则错误BLOCK'.
    Configures every hard limit to be as restrictive as possible (already
    breached before the order) and confirms a SELL of that same position
    still comes back ALLOW end-to-end through the real engine."""

    def setUp(self):
        self._orig = {
            "max_exposure": config.RISK_ENGINE_MAX_TOTAL_EXPOSURE_PCT,
            "max_positions": config.RISK_ENGINE_MAX_POSITIONS,
            "max_weight": config.RISK_ENGINE_MAX_POSITION_WEIGHT_PCT,
            "sector_map": dict(config.RISK_ENGINE_MAX_SECTOR_EXPOSURE_PCT),
            "max_beta": config.RISK_ENGINE_MAX_PORTFOLIO_BETA,
            "regime_restrict": config.RISK_ENGINE_REGIME_RESTRICT_NEW_POSITIONS_ON,
        }
        # Maximally restrictive -- portfolio is already WAY over every limit.
        config.RISK_ENGINE_MAX_TOTAL_EXPOSURE_PCT = 0.01
        config.RISK_ENGINE_MAX_POSITIONS = 0
        config.RISK_ENGINE_MAX_POSITION_WEIGHT_PCT = 0.01
        config.RISK_ENGINE_MAX_SECTOR_EXPOSURE_PCT = {"default": 0.01}
        config.RISK_ENGINE_MAX_PORTFOLIO_BETA = 0.01
        config.RISK_ENGINE_REGIME_RESTRICT_NEW_POSITIONS_ON = ("BULL", "NORMAL", "CAUTION", "RISK_OFF")

    def tearDown(self):
        config.RISK_ENGINE_MAX_TOTAL_EXPOSURE_PCT = self._orig["max_exposure"]
        config.RISK_ENGINE_MAX_POSITIONS = self._orig["max_positions"]
        config.RISK_ENGINE_MAX_POSITION_WEIGHT_PCT = self._orig["max_weight"]
        config.RISK_ENGINE_MAX_SECTOR_EXPOSURE_PCT = self._orig["sector_map"]
        config.RISK_ENGINE_MAX_PORTFOLIO_BETA = self._orig["max_beta"]
        config.RISK_ENGINE_REGIME_RESTRICT_NEW_POSITIONS_ON = self._orig["regime_restrict"]

    def _state(self):
        total_assets = 1_000_000
        return PortfolioState(
            positions={"US.NVDA": _pos("US.NVDA", 2000, 900_000, total_assets)},
            cash=100_000, total_assets=total_assets, long_mv=900_000,
            total_exposure_pct=0.90, position_count=1, market_regime="RISK_OFF",
        )

    # NOTE: these assert allowed==True and violations==[] rather than a
    # strict status==ALLOW -- _state() sets no price_returns, so beta_risk/
    # var_risk legitimately come back DATA_UNAVAILABLE -> WARN regardless
    # of side (see those modules' docstrings: missing data is never
    # silently treated as fine). That WARN is orthogonal to what this class
    # tests: that no HARD limit (violations) ever fires for a SELL.

    def test_full_close_allowed_despite_every_limit_breached(self):
        order = OrderIntent(code="US.NVDA", side="SELL", qty=2000, price=450.0)
        decision = PortfolioRiskEngine().evaluate(order, self._state())
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.violations, [])

    def test_partial_close_allowed_despite_every_limit_breached(self):
        order = OrderIntent(code="US.NVDA", side="SELL", qty=500, price=450.0)
        decision = PortfolioRiskEngine().evaluate(order, self._state())
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.violations, [])

    def test_zero_qty_order_allowed(self):
        order = OrderIntent(code="US.NVDA", side="SELL", qty=0, price=450.0)
        decision = PortfolioRiskEngine().evaluate(order, self._state())
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.violations, [])

    def test_buy_same_state_is_blocked_control_case(self):
        # Control: the same portfolio state DOES block a BUY -- proves the
        # SELL-always-allowed result above isn't just "checks are broken".
        order = OrderIntent(code="US.NVDA", side="BUY", qty=100, price=450.0)
        decision = PortfolioRiskEngine().evaluate(order, self._state())
        self.assertEqual(decision.status, RiskStatus.BLOCK)
        self.assertTrue(decision.violations)


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
