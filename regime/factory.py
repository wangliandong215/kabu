"""
regime/factory.py — get_regime_provider(): the single place that decides
whether the Rule-based or (future) Local LLM regime provider runs this
pass. Nothing outside this module should read config.LLM_ENABLED to pick
a provider — see regime/__init__.py module docstring.

config.LLM_ENABLED=False (default, and the only supported value today)
-> RuleRegimeProvider directly, no wrapping, no fallback machinery in the
   call path at all.

config.LLM_ENABLED=True (not supported yet — regime/llm_provider.py's
LocalLLMRegimeProvider is still a placeholder) -> the returned provider
tries the LLM first; ANY exception (NotImplementedError today; in the
future: model not started, process crash, timeout, invalid JSON, schema
validation failure, out-of-range values) or any structurally invalid
MarketRegime is caught, logged via notify.alert, and the call falls back
to RuleRegimeProvider — so this layer NEVER raises out to
engine/runner.py and NEVER blocks a trading pass.
"""
import config
from regime.models import MarketContext, MarketRegime, VALID_REGIMES
from regime.provider import RegimeProvider
from regime.rule_provider import RuleRegimeProvider
from regime.llm_provider import LocalLLMRegimeProvider


class _FallbackRegimeProvider(RegimeProvider):
    """Wraps a primary provider (the future LLM) with a rule-based safety
    net. Only ever constructed when config.LLM_ENABLED=True."""

    def __init__(self, primary: RegimeProvider, fallback: RegimeProvider):
        self._primary = primary
        self._fallback = fallback

    def evaluate(self, context: MarketContext) -> MarketRegime:
        import notify.alert as alert

        try:
            result = self._primary.evaluate(context)
        except Exception as exc:
            alert.log(f"regime: LLM provider failed "
                      f"({exc.__class__.__name__}: {exc}) — falling back to "
                      f"RuleRegimeProvider")
            return self._fallback_with_reason(context, f"LLM_FALLBACK:{exc.__class__.__name__}")

        if not _is_valid(result):
            alert.log("regime: LLM provider returned an invalid MarketRegime "
                      "— falling back to RuleRegimeProvider")
            return self._fallback_with_reason(context, "LLM_FALLBACK:INVALID_OUTPUT")

        return result

    def _fallback_with_reason(self, context: MarketContext, reason_code: str) -> MarketRegime:
        result = self._fallback.evaluate(context)
        result.reason_codes = [reason_code] + list(result.reason_codes)
        return result


def _is_valid(result) -> bool:
    if not isinstance(result, MarketRegime):
        return False
    if result.regime not in VALID_REGIMES:
        return False
    if not (0.0 <= result.confidence <= 1.0):
        return False
    if not (0.0 <= result.risk_level <= 1.0):
        return False
    return True


def get_regime_provider() -> RegimeProvider:
    if not config.LLM_ENABLED:
        return RuleRegimeProvider()
    return _FallbackRegimeProvider(primary=LocalLLMRegimeProvider(),
                                    fallback=RuleRegimeProvider())
