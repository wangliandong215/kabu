"""
Unit tests for risk/news_event_risk.py — the V3.4 sub-check registered in
risk/portfolio_risk_engine.py's _SUB_CHECKS.

Run:  python -m unittest risk.test_news_event_risk -v
"""
import unittest
from datetime import datetime

import config
from news.event_types import Direction, EventType
from news.news_event import NewsEvent
from news.news_sources import Source
from risk.news_event_risk import check
from risk.portfolio_state import PortfolioState, PositionSnapshot
from risk.risk_decision import OrderIntent, RiskStatus


def _pos(code, qty=10, market_val=10_000):
    return PositionSnapshot(code=code, qty=qty, market_val=market_val,
                             current_price=market_val / qty, weight=0.01,
                             sector="other", is_core=False)


def _event(ticker, event_type, severity, confidence, emergency, direction=Direction.NEGATIVE,
           news_score=-0.9):
    return NewsEvent(ticker=ticker, event_type=event_type, direction=direction,
                      severity=severity, confidence=confidence, news_score=news_score,
                      source=Source.SEC, timestamp=datetime.now(), decay_hours=24 * 30,
                      emergency=emergency)


class TestNewsEventRisk(unittest.TestCase):

    def setUp(self):
        self._orig_enabled = config.NEWS_ENGINE_ENABLED
        self._orig_warn = config.NEWS_ENGINE_WARN_SCORE_THRESHOLD
        config.NEWS_ENGINE_ENABLED = True
        config.NEWS_ENGINE_WARN_SCORE_THRESHOLD = -0.40

    def tearDown(self):
        config.NEWS_ENGINE_ENABLED = self._orig_enabled
        config.NEWS_ENGINE_WARN_SCORE_THRESHOLD = self._orig_warn

    def test_disabled_engine_allows_regardless_of_events(self):
        config.NEWS_ENGINE_ENABLED = False
        state = PortfolioState(positions={}, news_events={
            "US.NVDA": [_event("US.NVDA", EventType.BANKRUPTCY, 0.95, 0.95, True)]
        })
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_no_news_events_attached_allows(self):
        state = PortfolioState(positions={}, news_events=None)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_no_events_for_this_code_allows(self):
        state = PortfolioState(positions={}, news_events={"US.NVDA": []})
        order = OrderIntent(code="US.AMD", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_emergency_candidate_blocks_new_buy(self):
        state = PortfolioState(positions={}, news_events={
            "US.NVDA": [_event("US.NVDA", EventType.BANKRUPTCY, 0.95, 0.95, True)]
        })
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)

    def test_emergency_candidate_does_not_block_adding_to_existing_position(self):
        state = PortfolioState(positions={"US.NVDA": _pos("US.NVDA")}, news_events={
            "US.NVDA": [_event("US.NVDA", EventType.BANKRUPTCY, 0.95, 0.95, True)]
        })
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertNotEqual(decision.status, RiskStatus.BLOCK)

    def test_emergency_candidate_never_blocks_sell(self):
        # V3.4 spec Test 8 — direct order guard: news can never produce a SELL block.
        state = PortfolioState(positions={"US.NVDA": _pos("US.NVDA")}, news_events={
            "US.NVDA": [_event("US.NVDA", EventType.BANKRUPTCY, 0.95, 0.95, True)]
        })
        order = OrderIntent(code="US.NVDA", side="SELL", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_moderate_negative_score_warns_not_blocks(self):
        state = PortfolioState(positions={}, news_events={
            "US.NVDA": [_event("US.NVDA", EventType.EARNINGS_MISS, 0.5, 0.8, False, news_score=-0.5)]
        })
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.WARN)

    def test_mild_negative_score_within_threshold_allows(self):
        state = PortfolioState(positions={}, news_events={
            "US.NVDA": [_event("US.NVDA", EventType.PRODUCT_LAUNCH, 0.2, 0.6, False, news_score=-0.05)]
        })
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertEqual(decision.status, RiskStatus.ALLOW)

    def test_low_confidence_high_severity_never_blocks(self):
        # V3.4 spec Test 3, exercised through the sub-check: an event whose
        # confidence was too low to ever set emergency=True (see
        # news/validation/event_validator.py) must never BLOCK a new buy.
        state = PortfolioState(positions={}, news_events={
            "US.NVDA": [_event("US.NVDA", EventType.BANKRUPTCY, 0.95, 0.20, False, news_score=-0.19)]
        })
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        _, decision = check(order, state)
        self.assertNotEqual(decision.status, RiskStatus.BLOCK)


if __name__ == "__main__":
    unittest.main()
