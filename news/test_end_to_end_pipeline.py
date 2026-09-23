"""
V3.4 spec section 26's completion-criteria test: Mock News -> NewsItem ->
RuleBased Event Detector -> NewsEvent -> Validation -> Severity/Confidence/
News Score -> Time Decay -> Event Aggregation -> V3.3 Risk Adapter ->
Portfolio Risk Engine -> RiskDecision, run end-to-end through the REAL
risk/portfolio_risk_engine.py::PortfolioRiskEngine (not a mock), plus the
Mock LLM pipeline, plus V3.4 spec section 25's Test 8 (Direct Order Guard).

Run:  python -m unittest news.test_end_to_end_pipeline -v
"""
import unittest
from datetime import datetime

import config
from news.detection.llm_detector import LLMBackedEventDetector, MODE_SHADOW
from news.ingestion.mock import MockNewsSource
from news.integration.risk_adapter import NewsEngine
from news.news_item import NewsItem
from news.news_sources import Source
from risk.portfolio_risk_engine import PortfolioRiskEngine
from risk.portfolio_state import PortfolioState
from risk.risk_decision import OrderIntent, RiskStatus


class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        self._orig_enabled = config.NEWS_ENGINE_ENABLED
        self._orig_llm_mode = config.NEWS_ENGINE_LLM_MODE
        config.NEWS_ENGINE_ENABLED = True
        config.NEWS_ENGINE_LLM_MODE = "OFF"

    def tearDown(self):
        config.NEWS_ENGINE_ENABLED = self._orig_enabled
        config.NEWS_ENGINE_LLM_MODE = self._orig_llm_mode

    def _news_state(self, llm_detector=None):
        source = MockNewsSource([
            NewsItem(id="1", ticker="US.NVDA", source=Source.SEC, timestamp=datetime.now(),
                     headline="NVDA files for Chapter 11 bankruptcy amid SEC investigation"),
            NewsItem(id="2", ticker="US.AMD", source=Source.EARNINGS, timestamp=datetime.now(),
                     headline="AMD earnings beat estimates, raised guidance"),
        ])
        engine = NewsEngine(source=source, llm_detector=llm_detector)
        return engine.build_news_events_state(["US.NVDA", "US.AMD", "US.MSFT"])

    def test_full_pipeline_blocks_new_buy_on_emergency_ticker(self):
        news_events = self._news_state()
        state = PortfolioState(positions={}, total_assets=1_000_000, cash=1_000_000,
                                market_regime="BULL", news_events=news_events)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        decision = PortfolioRiskEngine().evaluate(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)
        self.assertIn("NEWS_EVENT_EMERGENCY_CANDIDATE", decision.violations)

    def test_full_pipeline_allows_new_buy_on_clean_ticker(self):
        news_events = self._news_state()
        state = PortfolioState(positions={}, total_assets=1_000_000, cash=1_000_000,
                                market_regime="BULL", news_events=news_events)
        order = OrderIntent(code="US.MSFT", side="BUY", qty=10, price=100.0)
        decision = PortfolioRiskEngine().evaluate(order, state)
        self.assertNotEqual(decision.status, RiskStatus.BLOCK)

    def test_positive_event_does_not_block_new_buy(self):
        news_events = self._news_state()
        state = PortfolioState(positions={}, total_assets=1_000_000, cash=1_000_000,
                                market_regime="BULL", news_events=news_events)
        order = OrderIntent(code="US.AMD", side="BUY", qty=10, price=100.0)
        decision = PortfolioRiskEngine().evaluate(order, state)
        self.assertNotEqual(decision.status, RiskStatus.BLOCK)

    def test_sell_on_emergency_ticker_never_blocked_direct_order_guard(self):
        # V3.4 spec Test 8 — Direct Order Guard: NewsEvent -> Risk Engine
        # only; there is no code path from a NewsEvent to a SELL, even for
        # the worst possible event on a ticker already held.
        from risk.portfolio_state import PositionSnapshot
        news_events = self._news_state()
        state = PortfolioState(
            positions={"US.NVDA": PositionSnapshot(code="US.NVDA", qty=10, market_val=1000,
                                                     current_price=100.0, weight=0.001,
                                                     sector="other", is_core=False)},
            total_assets=1_000_000, cash=1_000_000, market_regime="BULL", news_events=news_events,
        )
        order = OrderIntent(code="US.NVDA", side="SELL", qty=10, price=100.0)
        decision = PortfolioRiskEngine().evaluate(order, state)
        self.assertNotEqual(decision.status, RiskStatus.BLOCK)

    def test_mock_llm_pipeline_also_runs_end_to_end(self):
        # spec section 26's second pipeline: Mock News -> LLM Interface ->
        # EventExtractionResult -> Validation -> NewsEvent -> Risk Engine.
        news_events = self._news_state(llm_detector=LLMBackedEventDetector(mode=MODE_SHADOW))
        state = PortfolioState(positions={}, total_assets=1_000_000, cash=1_000_000,
                                market_regime="BULL", news_events=news_events)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        decision = PortfolioRiskEngine().evaluate(order, state)
        self.assertEqual(decision.status, RiskStatus.BLOCK)

    def test_news_engine_disabled_is_fully_inert(self):
        config.NEWS_ENGINE_ENABLED = False
        news_events = self._news_state()
        state = PortfolioState(positions={}, total_assets=1_000_000, cash=1_000_000,
                                market_regime="BULL", news_events=news_events)
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        decision = PortfolioRiskEngine().evaluate(order, state)
        self.assertNotEqual(decision.status, RiskStatus.BLOCK)

    def test_state_never_touched_when_news_events_not_attached(self):
        # Confirms zero behavior change for every existing caller that
        # doesn't populate news_events yet (engine/runner.py today).
        state = PortfolioState(positions={}, total_assets=1_000_000, cash=1_000_000,
                                market_regime="BULL")   # news_events left at default None
        order = OrderIntent(code="US.NVDA", side="BUY", qty=10, price=100.0)
        decision = PortfolioRiskEngine().evaluate(order, state)
        self.assertNotEqual(decision.status, RiskStatus.BLOCK)


if __name__ == "__main__":
    unittest.main()
