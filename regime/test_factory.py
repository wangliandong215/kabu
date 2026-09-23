"""
Unit tests for regime/factory.py::get_regime_provider() and its LLM
fallback wrapper — covers config.LLM_ENABLED=False (today's only
supported state), plus the fallback-safety contract that must hold once
LLM_ENABLED=True is ever set: any LLM failure mode (not implemented,
crash, timeout, invalid output, schema-invalid MarketRegime) must fall
back to RuleRegimeProvider and never raise out to the caller.

Run:  python -m unittest regime.test_factory -v
"""
import unittest

import config
from regime.factory import get_regime_provider, _FallbackRegimeProvider, _is_valid
from regime.models import MarketContext, MarketRegime, REGIME_BULL, REGIME_CRASH, SOURCE_RULES
from regime.rule_provider import RuleRegimeProvider
from regime.llm_provider import LocalLLMRegimeProvider
from regime.provider import RegimeProvider
import test_support


_CTX = MarketContext(weather_code=2, qqq_above_ma=True, drawdown_halt=False)


class _RaisingProvider(RegimeProvider):
    def __init__(self, exc):
        self._exc = exc

    def evaluate(self, context):
        raise self._exc


class _InvalidOutputProvider(RegimeProvider):
    """Simulates a schema-validation failure: returns something that is
    not a well-formed MarketRegime at all (not just a bad regime label,
    since MarketRegime's own __post_init__ already rejects that at
    construction time -- this simulates a provider bypassing/mocking
    that construction, which is the realistic failure shape for a JSON
    payload parsed by hand)."""
    def evaluate(self, context):
        return {"regime": "BULL", "confidence": 0.9}   # not a MarketRegime instance


class _FixedProvider(RegimeProvider):
    def __init__(self, result):
        self._result = result

    def evaluate(self, context):
        return self._result


class TestGetRegimeProviderDisabled(unittest.TestCase):

    def setUp(self):
        self._orig_llm_enabled = config.LLM_ENABLED

    def tearDown(self):
        config.LLM_ENABLED = self._orig_llm_enabled

    def test_llm_enabled_false_returns_plain_rule_provider(self):
        config.LLM_ENABLED = False
        provider = get_regime_provider()
        self.assertIsInstance(provider, RuleRegimeProvider)
        self.assertNotIsInstance(provider, _FallbackRegimeProvider)

    def test_config_default_is_false(self):
        # Captured pre-mutation in setUp -- pins config.py's actual default
        # (every test that flips LLM_ENABLED restores it in tearDown).
        self.assertFalse(self._orig_llm_enabled)

    def test_evaluate_matches_plain_rule_provider_result(self):
        config.LLM_ENABLED = False
        provider = get_regime_provider()
        direct = RuleRegimeProvider().evaluate(_CTX)
        wrapped = provider.evaluate(_CTX)
        self.assertEqual(wrapped.regime, direct.regime)
        self.assertEqual(wrapped.reason_codes, direct.reason_codes)


class TestGetRegimeProviderEnabled(unittest.TestCase):

    def setUp(self):
        self._orig_llm_enabled = config.LLM_ENABLED
        config.LLM_ENABLED = True

    def tearDown(self):
        config.LLM_ENABLED = self._orig_llm_enabled

    def test_llm_enabled_true_returns_fallback_wrapper(self):
        provider = get_regime_provider()
        self.assertIsInstance(provider, _FallbackRegimeProvider)

    def test_placeholder_llm_provider_not_implemented_falls_back(self):
        provider = get_regime_provider()
        result = provider.evaluate(_CTX)
        self.assertIsInstance(result, MarketRegime)
        self.assertEqual(result.source, SOURCE_RULES)
        self.assertTrue(any(rc.startswith("LLM_FALLBACK:") for rc in result.reason_codes))
        self.assertIn("LLM_FALLBACK:NotImplementedError", result.reason_codes)


class TestFallbackRegimeProviderDirect(unittest.TestCase):
    """Exercises _FallbackRegimeProvider directly with synthetic primaries
    so each failure mode from the user's spec (model not started, crash,
    timeout, invalid JSON/schema failure, out-of-range values) is pinned
    independently of whether a real LLM provider exists yet."""

    def test_generic_exception_falls_back(self):
        provider = _FallbackRegimeProvider(primary=_RaisingProvider(RuntimeError("boom")),
                                           fallback=RuleRegimeProvider())
        result = provider.evaluate(_CTX)
        self.assertIsInstance(result, MarketRegime)
        self.assertIn("LLM_FALLBACK:RuntimeError", result.reason_codes)

    def test_timeout_exception_falls_back(self):
        provider = _FallbackRegimeProvider(primary=_RaisingProvider(TimeoutError("slow")),
                                           fallback=RuleRegimeProvider())
        result = provider.evaluate(_CTX)
        self.assertIn("LLM_FALLBACK:TimeoutError", result.reason_codes)

    def test_process_unavailable_connection_error_falls_back(self):
        provider = _FallbackRegimeProvider(primary=_RaisingProvider(ConnectionError("no model process")),
                                           fallback=RuleRegimeProvider())
        result = provider.evaluate(_CTX)
        self.assertIn("LLM_FALLBACK:ConnectionError", result.reason_codes)

    def test_invalid_json_value_error_falls_back(self):
        provider = _FallbackRegimeProvider(primary=_RaisingProvider(ValueError("invalid json")),
                                           fallback=RuleRegimeProvider())
        result = provider.evaluate(_CTX)
        self.assertIn("LLM_FALLBACK:ValueError", result.reason_codes)

    def test_structurally_invalid_output_falls_back(self):
        provider = _FallbackRegimeProvider(primary=_InvalidOutputProvider(),
                                           fallback=RuleRegimeProvider())
        result = provider.evaluate(_CTX)
        self.assertIsInstance(result, MarketRegime)
        self.assertIn("LLM_FALLBACK:INVALID_OUTPUT", result.reason_codes)

    def test_never_raises_out_to_caller(self):
        provider = _FallbackRegimeProvider(primary=_RaisingProvider(Exception("anything")),
                                           fallback=RuleRegimeProvider())
        try:
            provider.evaluate(_CTX)
        except Exception as exc:   # pragma: no cover -- this failing is the bug under test
            self.fail(f"_FallbackRegimeProvider raised instead of falling back: {exc}")

    def test_fallback_result_is_still_a_valid_rule_classification(self):
        provider = _FallbackRegimeProvider(primary=_RaisingProvider(RuntimeError("boom")),
                                           fallback=RuleRegimeProvider())
        result = provider.evaluate(_CTX)
        # _CTX is weather_code=2/qqq_above_ma=True/drawdown_halt=False -> BULL
        # under RuleRegimeProvider's table -- confirms the fallback actually
        # ran the real rule logic, not a hardcoded stub.
        self.assertEqual(result.regime, REGIME_BULL)

    def test_healthy_primary_result_passes_through_unmodified(self):
        healthy = MarketRegime(regime=REGIME_CRASH, confidence=0.7, risk_level=0.1,
                               reason_codes=["LLM_SAYS_CRASH"], source="llm")
        provider = _FallbackRegimeProvider(primary=_FixedProvider(healthy),
                                           fallback=RuleRegimeProvider())
        result = provider.evaluate(_CTX)
        self.assertIs(result, healthy)


class TestIsValid(unittest.TestCase):

    def test_valid_market_regime_passes(self):
        mr = MarketRegime(regime=REGIME_BULL, confidence=0.5, risk_level=0.5,
                           reason_codes=[], source="llm")
        self.assertTrue(_is_valid(mr))

    def test_non_market_regime_object_fails(self):
        self.assertFalse(_is_valid({"regime": "BULL"}))
        self.assertFalse(_is_valid(None))
        self.assertFalse(_is_valid("BULL"))


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
