"""
news/scoring/severity.py -- V3.4 Severity (spec section 8): "0.0~1.0
表示事件本身的严重程度", combining event type + how strongly the source
text matched (keyword_strength). The BASE_SEVERITY table below is a
reference starting point per the spec's own caution ("不要把这些示例直接
硬编码为最终规则") -- expected to be retuned from real data, not treated
as gospel.
"""
from typing import Dict

from news.event_types import EventType

BASE_SEVERITY: Dict[EventType, float] = {
    EventType.BANKRUPTCY: 1.00,
    EventType.TRADING_HALT: 0.90,
    EventType.FRAUD_RISK: 0.90,
    EventType.ACCOUNTING_RISK: 0.80,
    EventType.REGULATORY_RISK: 0.75,
    EventType.FDA_REJECTION: 0.65,
    EventType.CREDIT_RISK: 0.60,
    EventType.LAWSUIT: 0.55,
    EventType.M_AND_A: 0.55,
    EventType.PRODUCT_RECALL: 0.55,
    EventType.MANAGEMENT_RISK: 0.50,
    EventType.GEOPOLITICAL_RISK: 0.50,
    EventType.CEO_CHANGE: 0.45,
    EventType.EARNINGS_MISS: 0.45,
    EventType.SUPPLY_CHAIN: 0.45,
    EventType.GUIDANCE_CHANGE: 0.40,
    EventType.FDA_APPROVAL: 0.40,
    EventType.EARNINGS_BEAT: 0.35,
    EventType.INSIDER_SELLING: 0.30,
    EventType.INSIDER_BUYING: 0.25,
    EventType.PRODUCT_LAUNCH: 0.20,
    EventType.UNKNOWN: 0.20,
}


def compute_severity(event_type: EventType, keyword_strength: float = 1.0) -> float:
    """keyword_strength >1.0 for multiple corroborating keyword hits in the
    same headline, <1.0 for a weak/ambiguous match. Clamped to +/-50% of
    base so a single noisy headline can never push a type's severity out
    of its plausible range."""
    base = BASE_SEVERITY.get(event_type, BASE_SEVERITY[EventType.UNKNOWN])
    scaled = base * max(0.5, min(1.5, keyword_strength))
    return max(0.0, min(1.0, scaled))
