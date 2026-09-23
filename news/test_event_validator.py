"""
Unit tests for news/validation/event_validator.py — this is the module that
must make V3.4 spec Test 1 (LLM outputs out-of-range values) and Test 2
(UNKNOWN event_type) safe.

Run:  python -m unittest news.test_event_validator -v
"""
import unittest
from datetime import datetime

import config
from news.event_types import Direction, EventType
from news.news_sources import Source
from news.validation.event_validator import validate_event


class TestEventValidator(unittest.TestCase):

    def setUp(self):
        self._orig_sev = config.NEWS_ENGINE_EMERGENCY_SEVERITY_THRESHOLD
        self._orig_conf = config.NEWS_ENGINE_EMERGENCY_CONFIDENCE_THRESHOLD
        config.NEWS_ENGINE_EMERGENCY_SEVERITY_THRESHOLD = 0.85
        config.NEWS_ENGINE_EMERGENCY_CONFIDENCE_THRESHOLD = 0.85

    def tearDown(self):
        config.NEWS_ENGINE_EMERGENCY_SEVERITY_THRESHOLD = self._orig_sev
        config.NEWS_ENGINE_EMERGENCY_CONFIDENCE_THRESHOLD = self._orig_conf

    def test_missing_ticker_returns_none(self):
        event = validate_event({"event_type": "BANKRUPTCY"})
        self.assertIsNone(event)

    def test_out_of_range_severity_and_confidence_are_clamped(self):
        # V3.4 spec Test 1 — severity=5.7, confidence=-2
        event = validate_event({
            "ticker": "US.NVDA", "event_type": "REGULATORY_RISK", "direction": "NEGATIVE",
            "severity": 5.7, "confidence": -2,
        })
        self.assertIsNotNone(event)
        self.assertEqual(event.severity, 1.0)
        self.assertEqual(event.confidence, 0.0)

    def test_non_numeric_severity_defaults_safely(self):
        event = validate_event({
            "ticker": "US.NVDA", "event_type": "REGULATORY_RISK", "direction": "NEGATIVE",
            "severity": "very bad", "confidence": None,
        })
        self.assertEqual(event.severity, 0.0)
        self.assertEqual(event.confidence, 0.0)

    def test_invalid_event_type_becomes_unknown(self):
        event = validate_event({"ticker": "US.NVDA", "event_type": "TOTALLY_MADE_UP"})
        self.assertEqual(event.event_type, EventType.UNKNOWN)

    def test_invalid_direction_becomes_unknown(self):
        event = validate_event({"ticker": "US.NVDA", "direction": "SIDEWAYS"})
        self.assertEqual(event.direction, Direction.UNKNOWN)

    def test_invalid_source_becomes_other(self):
        event = validate_event({"ticker": "US.NVDA", "source": "TWITTER_RUMOR"})
        self.assertEqual(event.source, Source.OTHER)

    def test_unknown_event_type_never_emergency_even_with_high_severity(self):
        # V3.4 spec Test 2 — event_type=UNKNOWN must not trigger emergency.
        event = validate_event({
            "ticker": "US.NVDA", "event_type": "UNKNOWN", "direction": "NEGATIVE",
            "severity": 0.99, "confidence": 0.99,
        })
        self.assertFalse(event.emergency)

    def test_high_severity_low_confidence_never_emergency(self):
        # V3.4 spec Test 3 — severity=0.95, confidence=0.20 must not trigger emergency.
        event = validate_event({
            "ticker": "US.NVDA", "event_type": "BANKRUPTCY", "direction": "NEGATIVE",
            "severity": 0.95, "confidence": 0.20,
        })
        self.assertFalse(event.emergency)

    def test_high_severity_high_confidence_emergency_eligible_type_is_emergency(self):
        event = validate_event({
            "ticker": "US.NVDA", "event_type": "BANKRUPTCY", "direction": "NEGATIVE",
            "severity": 0.95, "confidence": 0.95,
        })
        self.assertTrue(event.emergency)

    def test_high_severity_high_confidence_non_eligible_type_not_emergency(self):
        event = validate_event({
            "ticker": "US.NVDA", "event_type": "EARNINGS_MISS", "direction": "NEGATIVE",
            "severity": 0.95, "confidence": 0.95,
        })
        self.assertFalse(event.emergency)

    def test_missing_decay_hours_uses_event_type_default(self):
        event = validate_event({"ticker": "US.NVDA", "event_type": "BANKRUPTCY"})
        self.assertGreater(event.decay_hours, 0)

    def test_invalid_decay_hours_falls_back_to_default(self):
        event = validate_event({"ticker": "US.NVDA", "event_type": "EARNINGS_MISS", "decay_hours": -5})
        self.assertGreater(event.decay_hours, 0)

    def test_missing_timestamp_defaults_to_now(self):
        before = datetime.now()
        event = validate_event({"ticker": "US.NVDA"})
        after = datetime.now()
        self.assertTrue(before <= event.timestamp <= after)


if __name__ == "__main__":
    unittest.main()
