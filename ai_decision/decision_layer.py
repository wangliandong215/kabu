"""
ai_decision/decision_layer.py — V3.5 AI Decision Layer (spec 全文).

Authority chain this file locks in (spec 四/五/六):

    Market Data / Strategy / News&Event / Macro / Financials / Trade DB
            |
    Portfolio Risk Engine   (risk/portfolio_risk_engine.py, unchanged)
            |
    RiskDecision   (final — status/allowed/violations already set)
            |
    AIDecisionLayer.decide()   <-- this module, always optional, always AFTER
            |
    AIDecision  (BUY/HOLD/REDUCE/SKIP + confidence + reasoning — advisory)
            |
    Rule Engine / Order Validation / Execution   (unchanged, untouched)

The AI can: analyze, explain, summarize, flag risk, suggest a
BUY/HOLD/REDUCE/SKIP. The AI cannot: bypass the Risk Engine, place an
order, override BLOCK, change a position, or feed back into
RiskDecision.allowed/status — there is no code path anywhere in this
module that lets an AIDecision reach engine/runner.py's order flow.
decide() is called strictly after the RiskDecision is finalized, and this
module is NOT called from engine/runner.py in this phase (V3.5 spec 九:
"第一阶段不下单，不修改持仓，不改变 Rule Engine" — Shadow Mode / Research
Mode only, same two-phase rollout risk/llm_advisor.py (V3.3) and
news/integration/risk_adapter.py (V3.4) already went through before either
was wired into a live call site). Wiring a call site into
engine/runner.py::_place_order() (still advisory-only, still never
affecting the return value) is V3.5 Phase 2.

Fail-open contract mirrors risk/llm_advisor.py::LLMRiskAdvisor.analyze()
exactly: any exception from the provider is caught and logged, decide()
returns None, the caller's real (Rule Engine) decision is always what
governs behavior.

Modes (config.AI_DECISION_MODE): OFF (default, NullAIDecisionProvider,
always returns SKIP/confidence=0.0 trivially) / SHADOW
(MockAIDecisionProvider today — no real LLM backend is implemented yet;
wiring a real ai_decision/llm/*_provider.py in is future work). There is
deliberately no ACTIVE mode, same convention as RISK_ENGINE_LLM_MODE /
NEWS_ENGINE_LLM_MODE / POSITION_LLM_MODE — see those modules' docstrings.
"""
from typing import Any, Dict, Optional

import config
from ai_decision.llm.base import AIDecisionProvider
from ai_decision.llm.mock_provider import MockAIDecisionProvider
from ai_decision.llm.null_provider import NullAIDecisionProvider
from ai_decision.schema import AIDecision, AIDecisionContext

MODE_OFF = "OFF"
MODE_SHADOW = "SHADOW"
VALID_MODES = (MODE_OFF, MODE_SHADOW)


def get_ai_decision_provider(mode: Optional[str] = None) -> AIDecisionProvider:
    mode = mode if mode is not None else config.AI_DECISION_MODE
    if mode == MODE_SHADOW:
        return MockAIDecisionProvider()
    return NullAIDecisionProvider()


def build_context(ctx: AIDecisionContext) -> Dict[str, Any]:
    return {
        "symbol": ctx.symbol,
        "market_regime": ctx.market_regime,
        "portfolio": ctx.portfolio_snapshot,
        "risk_decision": ctx.risk_decision_snapshot,
        "news": ctx.news_snapshot or {},
        "event": ctx.event_snapshot or {},
        "trade_history": ctx.trade_history_snapshot or {},
        "macro": ctx.macro_snapshot or {},
        "financial": ctx.financial_snapshot or {},
        "institution": ctx.institution_snapshot or {},
    }


class AIDecisionLayer:
    def __init__(self, provider: Optional[AIDecisionProvider] = None, mode: Optional[str] = None):
        self._mode = mode if mode is not None else config.AI_DECISION_MODE
        self._provider = provider

    def decide(self, ctx: AIDecisionContext) -> Optional[AIDecision]:
        """Never raises. Returns None if the mode is OFF-equivalent or the
        provider fails for any reason — see module docstring for the full
        fail-open / advisory-only contract. A successful decision is also
        appended to ai_decision/audit_log.py's JSONL log (spec 八); a
        logging failure there is swallowed by log_decision() itself and
        never turns a real decision into None."""
        if self._mode not in VALID_MODES or self._mode == MODE_OFF:
            return None

        provider = self._provider or get_ai_decision_provider(self._mode)
        try:
            context = build_context(ctx)
            decision = provider.decide(context)
        except Exception as exc:
            import notify.alert as alert
            alert.log(f"ai_decision.decision_layer: decide failed for {ctx.symbol} "
                      f"({exc.__class__.__name__}: {exc}) — advisory skipped")
            return None

        from ai_decision.audit_log import log_decision
        log_decision(ctx, decision)
        return decision
