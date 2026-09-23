"""
risk/news_event_risk.py -- V3.4 News & Event Engine's Risk Engine sub-check.

"News is information, not an order" (V3.4 spec's final principle): this
module never generates a SELL and never touches SELL orders at all -- an
event-driven exit is not something the News Engine or this check is
allowed to decide (spec section 13: "Event Engine 只能产生 Emergency
Candidate...不能直接产生 SELL ORDER"). It can only make NEW BUYs on an
already-flagged ticker harder or impossible, mirroring the "only ever
gates opening a NEW position" convention risk/market_regime_risk.py
already uses.

state.news_events is attached by whichever caller runs the V3.4 pipeline
(see news/integration/risk_adapter.py::NewsEngine.build_news_events_state())
-- None (not yet wired, or config.NEWS_ENGINE_ENABLED=False) means "no
news data this pass", which is DATA_UNAVAILABLE-safe ALLOW, same
convention as every other sub-check here, never a guessed neutral score.
"""
from dataclasses import dataclass
from typing import List, Optional

import config
from news.news_event import NewsEvent
from news.scoring.aggregator import aggregate
from risk.portfolio_state import PortfolioState
from risk.risk_decision import DATA_UNAVAILABLE, OrderIntent, allow, block, warn

VIOLATION_EMERGENCY = "NEWS_EVENT_EMERGENCY_CANDIDATE"
WARNING_NEGATIVE = "NEWS_EVENT_NEGATIVE"


@dataclass
class NewsEventCheckResult:
    status: str
    effective_score: Optional[float]
    worst_severity: Optional[float]
    confidence: Optional[float]
    emergency: bool


def check(order: OrderIntent, state: PortfolioState):
    if not config.NEWS_ENGINE_ENABLED or state.news_events is None:
        result = NewsEventCheckResult(status=DATA_UNAVAILABLE, effective_score=None,
                                       worst_severity=None, confidence=None, emergency=False)
        return result, allow("news engine disabled or no news data this pass",
                              metrics={"news_event": result})

    events: List[NewsEvent] = state.news_events.get(order.code, [])

    if order.side != "BUY" or not events:
        result = NewsEventCheckResult(status="OK", effective_score=None,
                                       worst_severity=None, confidence=None, emergency=False)
        return result, allow(metrics={"news_event": result})

    signal = aggregate(events)
    result = NewsEventCheckResult(status="OK", effective_score=signal.effective_score,
                                   worst_severity=signal.worst_severity,
                                   confidence=signal.confidence, emergency=signal.emergency)

    pos = state.get(order.code)
    opens_new_position = (pos is None or pos.qty == 0)

    if signal.emergency and opens_new_position:
        result.status = "BLOCK"
        return result, block(
            f"{order.code} has an emergency-candidate news event "
            f"(severity={signal.worst_severity:.2f}, confidence={signal.confidence:.2f}) "
            f"— new position blocked",
            violations=[VIOLATION_EMERGENCY], metrics={"news_event": result},
        )

    if signal.effective_score <= config.NEWS_ENGINE_WARN_SCORE_THRESHOLD:
        result.status = "WARN"
        return result, warn(
            f"{order.code} effective news score {signal.effective_score:+.2f} — elevated caution",
            warnings=[WARNING_NEGATIVE], metrics={"news_event": result},
        )

    return result, allow(metrics={"news_event": result})
