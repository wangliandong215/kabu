"""
news/scoring/decay.py -- V3.4 Time Decay (spec section 11): "新闻影响不能
永久存在". Half-life model: decay_hours IS the half-life -- at
age_hours == decay_hours the effective score is exactly half the raw
score, regardless of event type. Different EventTypes get different
decay_hours (DEFAULT_DECAY_HOURS below), which is what gives "重大
Regulatory Risk 衰减慢 / 普通市场新闻 衰减快" its effect, without needing a
second decay-shape parameter per type (spec: "第一版先建立通用decay
framework，不要过度优化参数").
"""
from datetime import datetime
from typing import Dict, Optional

from news.event_types import EventType

DEFAULT_DECAY_HOURS: Dict[EventType, float] = {
    EventType.BANKRUPTCY: 24 * 30,
    EventType.FRAUD_RISK: 24 * 14,
    EventType.ACCOUNTING_RISK: 24 * 14,
    EventType.M_AND_A: 24 * 10,
    EventType.TRADING_HALT: 24 * 7,
    EventType.REGULATORY_RISK: 24 * 7,
    EventType.CREDIT_RISK: 24 * 7,
    EventType.FDA_REJECTION: 24 * 7,
    EventType.SUPPLY_CHAIN: 24 * 5,
    EventType.PRODUCT_RECALL: 24 * 5,
    EventType.FDA_APPROVAL: 24 * 5,
    EventType.LAWSUIT: 24 * 5,
    EventType.CEO_CHANGE: 24 * 5,
    EventType.MANAGEMENT_RISK: 24 * 5,
    EventType.EARNINGS_MISS: 24 * 3,
    EventType.EARNINGS_BEAT: 24 * 3,
    EventType.GUIDANCE_CHANGE: 24 * 3,
    EventType.GEOPOLITICAL_RISK: 24 * 3,
    EventType.INSIDER_SELLING: 24 * 3,
    EventType.INSIDER_BUYING: 24 * 3,
    EventType.PRODUCT_LAUNCH: 24 * 2,
    EventType.UNKNOWN: 24 * 1,
}


def decayed_score(raw_score: float, event_ts: datetime, decay_hours: float,
                   as_of: Optional[datetime] = None) -> float:
    as_of = as_of or datetime.now()
    age_hours = max(0.0, (as_of - event_ts).total_seconds() / 3600.0)
    if decay_hours <= 0:
        return 0.0
    half_life_factor = 0.5 ** (age_hours / decay_hours)
    return raw_score * half_life_factor
