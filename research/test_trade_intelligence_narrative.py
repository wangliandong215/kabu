"""
Unit tests for research/trade_intelligence_narrative.py — the Null/Mock
narrative placeholder (real LLM wiring deliberately deferred).

Run:  python -m unittest research.test_trade_intelligence_narrative -v
"""
import unittest

from research.trade_intelligence_narrative import (
    MockNarrativeProvider, NullNarrativeProvider, get_narrative_provider,
)


class NarrativeProviderTestCase(unittest.TestCase):

    def test_default_mode_off_is_null_provider(self):
        self.assertIsInstance(get_narrative_provider("OFF"), NullNarrativeProvider)

    def test_shadow_mode_is_mock_provider(self):
        self.assertIsInstance(get_narrative_provider("SHADOW"), MockNarrativeProvider)

    def test_unknown_mode_falls_back_to_null(self):
        self.assertIsInstance(get_narrative_provider("ACTIVE"), NullNarrativeProvider)

    def test_null_provider_returns_disabled_notice(self):
        result = NullNarrativeProvider().generate({})
        self.assertIn("disabled", result.summary.lower())
        self.assertEqual(result.model, "none")

    def test_mock_provider_only_echoes_context_counts(self):
        context = {
            "n_trades_analyzed": 42, "n_candidate_patterns": 3,
            "n_observations": 7, "analysis_types": ["EARLY_FAILURE", "EXIT"],
        }
        result = MockNarrativeProvider().generate(context)
        self.assertIn("42", result.summary)
        self.assertIn("3 candidate pattern", result.summary)
        self.assertIn("7", result.summary)
        self.assertIn("EARLY_FAILURE", result.summary)
        self.assertIn("EXIT", result.summary)

    def test_mock_provider_handles_empty_context_without_error(self):
        result = MockNarrativeProvider().generate({})
        self.assertIn("0 closed trades", result.summary)


if __name__ == "__main__":
    unittest.main()
