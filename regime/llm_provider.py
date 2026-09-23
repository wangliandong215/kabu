"""
regime/llm_provider.py — LocalLLMRegimeProvider: a deliberate placeholder.

No local LLM is implemented, downloaded, or invoked anywhere in this
codebase in this phase. evaluate() always raises NotImplementedError so
this class can never silently fabricate a MarketRegime. It exists purely
so the RegimeProvider abstraction and regime/factory.py's fallback
wiring never need to change shape when a real local model (Qwen, Llama,
Mistral, Gemma, ...) is wired in later — only this class's body changes.

config.LLM_ENABLED must stay False (the default) while this class is a
placeholder. See regime/factory.py for what happens if LLM_ENABLED=True
is set anyway: evaluate_with_fallback() catches this NotImplementedError
(and any future real failure mode — process crash, timeout, invalid
JSON, schema validation failure) and falls back to RuleRegimeProvider,
so run_once() is never blocked by this class raising.
"""
from regime.models import MarketContext, MarketRegime
from regime.provider import RegimeProvider


class LocalLLMRegimeProvider(RegimeProvider):
    def evaluate(self, context: MarketContext) -> MarketRegime:
        raise NotImplementedError(
            "LocalLLMRegimeProvider is a placeholder — no local LLM is wired "
            "in yet. config.LLM_ENABLED must stay False until a real "
            "implementation replaces this class's evaluate() method."
        )
