"""
news/event_types.py -- V3.4 EventType/Direction taxonomy (spec sections
5/7) plus emergency eligibility (spec section 13).

Direction is NOT a trade signal (spec section 7's "Earnings Beat ->
Direction=POSITIVE 并不意味着BUY") -- it only describes whether the event
itself reads positive/negative. Whether that ever affects an order is
entirely risk/news_event_risk.py's call, applied only through the
Portfolio Risk Engine.
"""
from enum import Enum

import config


class EventType(str, Enum):
    EARNINGS_MISS = "EARNINGS_MISS"
    EARNINGS_BEAT = "EARNINGS_BEAT"
    CEO_CHANGE = "CEO_CHANGE"
    MANAGEMENT_RISK = "MANAGEMENT_RISK"
    M_AND_A = "M_AND_A"
    REGULATORY_RISK = "REGULATORY_RISK"
    LAWSUIT = "LAWSUIT"
    BANKRUPTCY = "BANKRUPTCY"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    GEOPOLITICAL_RISK = "GEOPOLITICAL_RISK"
    FDA_APPROVAL = "FDA_APPROVAL"
    FDA_REJECTION = "FDA_REJECTION"
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    PRODUCT_RECALL = "PRODUCT_RECALL"
    ACCOUNTING_RISK = "ACCOUNTING_RISK"
    FRAUD_RISK = "FRAUD_RISK"
    TRADING_HALT = "TRADING_HALT"
    GUIDANCE_CHANGE = "GUIDANCE_CHANGE"
    CREDIT_RISK = "CREDIT_RISK"
    INSIDER_SELLING = "INSIDER_SELLING"
    INSIDER_BUYING = "INSIDER_BUYING"
    UNKNOWN = "UNKNOWN"


class Direction(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


# Event types eligible to ever set NewsEvent.emergency=True (spec section
# 13's BANKRUPTCY / TRADING_HALT / MAJOR_FRAUD / MAJOR_ACCOUNTING_RISK /
# SEVERE_REGULATORY_ACTION). Eligibility alone is not enough -- see
# is_emergency_candidate(): severity AND confidence must also clear their
# configured thresholds, so a low-confidence guess never sets it (V3.4
# spec Test 3 -- "不能因为Severity高就直接触发极端交易").
EMERGENCY_ELIGIBLE_TYPES = frozenset({
    EventType.BANKRUPTCY,
    EventType.TRADING_HALT,
    EventType.FRAUD_RISK,
    EventType.ACCOUNTING_RISK,
    EventType.REGULATORY_RISK,
})


def is_emergency_candidate(event_type: EventType, severity: float, confidence: float) -> bool:
    """Emergency CANDIDATE only (spec section 13) -- this never places an
    order; risk/news_event_risk.py only ever uses this to BLOCK a NEW BUY
    on the affected ticker, never to generate a SELL."""
    if event_type not in EMERGENCY_ELIGIBLE_TYPES:
        return False
    return (severity >= config.NEWS_ENGINE_EMERGENCY_SEVERITY_THRESHOLD
            and confidence >= config.NEWS_ENGINE_EMERGENCY_CONFIDENCE_THRESHOLD)
