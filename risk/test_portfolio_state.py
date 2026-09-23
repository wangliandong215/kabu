"""
Unit tests for risk/portfolio_state.py::build_portfolio_state().

Run:  python -m unittest risk.test_portfolio_state -v
"""
import unittest

import config
from portfolio.broker_state import BrokerPosition, BrokerState
from portfolio.tracker import Portfolio
from risk.portfolio_state import build_portfolio_state


class TestBuildPortfolioState(unittest.TestCase):

    def test_none_broker_state_yields_data_unavailable_not_zero(self):
        state = build_portfolio_state(None, tracker=None)
        self.assertIsNone(state.total_exposure_pct)
        self.assertEqual(state.position_count, 0)
        self.assertEqual(state.positions, {})

    def test_zero_total_assets_yields_data_unavailable(self):
        bs = BrokerState(positions={}, cash=0.0, total_assets=0.0, long_mv=0.0)
        state = build_portfolio_state(bs, tracker=None)
        self.assertIsNone(state.total_exposure_pct)

    def test_normal_state_computes_weights_and_exposure(self):
        bs = BrokerState(
            positions={"US.NVDA": BrokerPosition(code="US.NVDA", qty=100, cost_price=90.0,
                                                   market_val=100_000.0, current_price=100.0)},
            cash=900_000.0, total_assets=1_000_000.0, long_mv=100_000.0,
        )
        state = build_portfolio_state(bs, tracker=None)
        self.assertAlmostEqual(state.total_exposure_pct, 0.10)
        self.assertAlmostEqual(state.positions["US.NVDA"].weight, 0.10)
        self.assertEqual(state.position_count, 1)

    def test_zero_qty_position_excluded(self):
        bs = BrokerState(
            positions={"US.DEAD": BrokerPosition(code="US.DEAD", qty=0, cost_price=10.0,
                                                   market_val=0.0, current_price=10.0)},
            cash=1_000_000.0, total_assets=1_000_000.0, long_mv=0.0,
        )
        state = build_portfolio_state(bs, tracker=None)
        self.assertEqual(state.positions, {})
        self.assertEqual(state.position_count, 0)

    def test_qqq_core_excluded_from_position_count(self):
        bs = BrokerState(
            positions={
                config.QQQ_CORE_CODE: BrokerPosition(code=config.QQQ_CORE_CODE, qty=100,
                                                       cost_price=700.0, market_val=250_000.0,
                                                       current_price=750.0),
                "US.NVDA": BrokerPosition(code="US.NVDA", qty=10, cost_price=90.0,
                                           market_val=10_000.0, current_price=100.0),
            },
            cash=740_000.0, total_assets=1_000_000.0, long_mv=260_000.0,
        )
        state = build_portfolio_state(bs, tracker=None)
        self.assertEqual(state.position_count, 1)   # QQQ excluded
        self.assertTrue(state.positions[config.QQQ_CORE_CODE].is_core)
        self.assertFalse(state.positions["US.NVDA"].is_core)

    def test_tracker_strategy_core_etf_also_excludes(self):
        # A non-QQQ code tagged core_etf in the tracker should also be
        # excluded -- same convention as guard.can_open_position().
        bs = BrokerState(
            positions={"US.SPY": BrokerPosition(code="US.SPY", qty=10, cost_price=400.0,
                                                  market_val=4_500.0, current_price=450.0)},
            cash=995_500.0, total_assets=1_000_000.0, long_mv=4_500.0,
        )
        tracker = Portfolio.__new__(Portfolio)
        tracker.data = {"positions": {"US.SPY": {"strategy": "core_etf"}}}
        state = build_portfolio_state(bs, tracker=tracker)
        self.assertEqual(state.position_count, 0)
        self.assertTrue(state.positions["US.SPY"].is_core)


if __name__ == "__main__":
    unittest.main()
