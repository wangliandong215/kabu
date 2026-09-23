"""
news/detection/rule_based.py -- V3.4 spec Step 4's RuleBased Event
Detector. Deliberately simple/keyword-based (spec section 4: "先用 Rule
Based Event 把 pipeline 跑通"; section 28 forbids over-engineering this
ahead of a real LLM).

Independent lexicon from engine/news_filter.py's 3-tier grader and
engine/news.py's sentiment keywords -- those two already feed the live BUY
signal_strength/scoring path directly today and are out of scope for V3.4
(see risk/news_event_risk.py's module docstring for why this is a
separate, new layer rather than a replacement).
"""
from typing import List, Tuple

from news.event_types import Direction, EventType
from news.news_event import NewsEvent
from news.news_item import NewsItem
from news.scoring import confidence as confidence_mod
from news.scoring import severity as severity_mod
from news.validation.event_validator import validate_event

# (event_type, direction, keywords) -- order matters: first rule that
# matches wins, most-specific/severe event types listed first so e.g. a
# headline mentioning both "lawsuit" and "bankruptcy" classifies as
# BANKRUPTCY, not LAWSUIT.
_RULES: Tuple[Tuple[EventType, Direction, Tuple[str, ...]], ...] = (
    (EventType.BANKRUPTCY, Direction.NEGATIVE,
     ("bankruptcy", "chapter 11", "chapter 7", "insolvency")),
    (EventType.TRADING_HALT, Direction.NEGATIVE,
     ("trading halt", "halted trading", "circuit breaker halt")),
    (EventType.FRAUD_RISK, Direction.NEGATIVE,
     ("securities fraud", "accounting fraud", "cooking the books", "ponzi")),
    (EventType.ACCOUNTING_RISK, Direction.NEGATIVE,
     ("restatement", "material weakness", "going concern", "internal controls failure")),
    (EventType.REGULATORY_RISK, Direction.NEGATIVE,
     ("sec investigation", "sec probe", "doj investigation", "antitrust probe",
      "regulatory action", "consent decree")),
    (EventType.LAWSUIT, Direction.NEGATIVE,
     ("class action", "class-action lawsuit", "sued", "lawsuit filed")),
    (EventType.FDA_REJECTION, Direction.NEGATIVE,
     ("fda rejection", "complete response letter", "fda clinical hold")),
    (EventType.FDA_APPROVAL, Direction.POSITIVE,
     ("fda approval", "fda approved", "fda clears")),
    (EventType.PRODUCT_RECALL, Direction.NEGATIVE,
     ("product recall", "safety recall", "recalls ")),
    (EventType.M_AND_A, Direction.POSITIVE,
     ("to acquire", "acquisition of", "merger agreement", "to be acquired")),
    (EventType.CEO_CHANGE, Direction.NEGATIVE,
     ("ceo resigns", "ceo steps down", "ceo departure", "ceo abruptly resigns",
      "sudden resignation of ceo")),
    (EventType.MANAGEMENT_RISK, Direction.NEGATIVE,
     ("cfo resigns", "executive departure", "management shakeup")),
    (EventType.SUPPLY_CHAIN, Direction.NEGATIVE,
     ("supply chain disruption", "chip shortage", "factory shutdown", "supplier halt")),
    (EventType.GEOPOLITICAL_RISK, Direction.NEGATIVE,
     ("export ban", "sanctions", "tariff", "trade war")),
    (EventType.CREDIT_RISK, Direction.NEGATIVE,
     ("credit downgrade", "debt downgrade", "default risk")),
    (EventType.INSIDER_SELLING, Direction.NEGATIVE,
     ("insider selling", "insider sold", "form 4 sale")),
    (EventType.INSIDER_BUYING, Direction.POSITIVE,
     ("insider buying", "insider bought", "form 4 purchase")),
    (EventType.GUIDANCE_CHANGE, Direction.NEGATIVE,
     ("guidance lowered", "cuts guidance", "lowered outlook", "cuts outlook")),
    (EventType.EARNINGS_MISS, Direction.NEGATIVE,
     ("earnings miss", "missed earnings", "misses estimates", "below expectations")),
    (EventType.EARNINGS_BEAT, Direction.POSITIVE,
     ("earnings beat", "beats estimates", "tops estimates", "raised guidance")),
    (EventType.PRODUCT_LAUNCH, Direction.POSITIVE,
     ("unveils", "product launch", "announces partnership")),
)


class RuleBasedEventDetector:
    def detect(self, item: NewsItem) -> List[NewsEvent]:
        """At most one NewsEvent per headline -- _RULES is ordered most-
        severe-first and the first match wins, so an ambiguous headline
        never produces multiple conflicting events for the same story."""
        text = f"{item.headline} {item.body}".lower()

        for event_type, direction, keywords in _RULES:
            matches = [kw for kw in keywords if kw in text]
            if not matches:
                continue

            severity = severity_mod.compute_severity(
                event_type, keyword_strength=1.0 + 0.1 * (len(matches) - 1))
            confidence = confidence_mod.compute_confidence(
                item.source, keyword_match_count=len(matches))

            candidate = {
                "ticker": item.ticker,
                "event_type": event_type,
                "direction": direction,
                "severity": severity,
                "confidence": confidence,
                "source": item.source,
                "timestamp": item.timestamp,
                "headline": item.headline,
                "summary": item.headline,
                "evidence": matches[0],
                "news_id": item.id,
            }
            event = validate_event(candidate)
            return [event] if event is not None else []

        return []
