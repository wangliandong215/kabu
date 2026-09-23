"""
ai_decision/llm/mock_provider.py — canned, deterministic AIDecisionProvider
for tests and for exercising SHADOW mode before any real LLM backend
exists. NOT wired to any real model — same "placeholder, not a fake real
answer" contract as risk/llm/mock_provider.py, except this one folds in
more of the input snapshot so ai_decision/decision_layer.py's SHADOW-mode
log output is representative of the real Decision Schema (spec section
三). The aggregation rule below is illustrative only, not a trading
signal — a real backend replacing this class is expected to reason over
the same context dict, not necessarily reproduce this logic.
"""
from typing import Any, Dict, List

from ai_decision.llm.base import AIDecisionProvider
from ai_decision.schema import AIAction, AIDecision


class MockAIDecisionProvider(AIDecisionProvider):
    def decide(self, context: Dict[str, Any]) -> AIDecision:
        risk = context.get("risk_decision") or {}
        news = context.get("news") or {}
        portfolio = context.get("portfolio") or {}
        regime = context.get("market_regime")

        reason_codes: List[str] = []
        risk_flags: List[str] = []
        supporting: List[str] = []
        negative: List[str] = []

        status = risk.get("status", "UNKNOWN")
        if status == "BLOCK":
            reason_codes.append("RISK_ENGINE_BLOCK")
            negative.append(f"risk engine blocked: {risk.get('reason', '')}")
        elif status == "WARN":
            reason_codes.append("RISK_ENGINE_WARN")
            risk_flags.append("RISK_ENGINE_WARN")

        news_score = news.get("effective_score")
        if news_score is not None:
            if news_score > 0.2:
                reason_codes.append("NEWS_POSITIVE")
                supporting.append(f"news score {news_score:.2f}")
            elif news_score < -0.2:
                reason_codes.append("NEWS_NEGATIVE")
                negative.append(f"news score {news_score:.2f}")

        if regime == "RISK_OFF":
            reason_codes.append("REGIME_RISK_OFF")
            risk_flags.append("REGIME_RISK_OFF")
        elif regime:
            reason_codes.append(f"REGIME_{regime}")

        weight = portfolio.get("weight")
        if weight is not None and weight >= 0.08:
            reason_codes.append("POSITION_ALREADY_LARGE")
            risk_flags.append("POSITION_CONCENTRATION")

        if status == "BLOCK":
            decision = AIAction.SKIP
            confidence = 0.75
        elif negative and not supporting:
            decision = AIAction.REDUCE
            confidence = 0.55
        elif supporting and not negative:
            decision = AIAction.BUY
            confidence = 0.6
        else:
            decision = AIAction.HOLD
            confidence = 0.5

        reasoning = (f"[mock] {context.get('symbol', '?')}: risk={status}, "
                     f"news_score={news_score}, regime={regime}, weight={weight}")

        return AIDecision(
            decision=decision, confidence=confidence,
            reason_codes=reason_codes or ["NO_SIGNAL"],
            reasoning=reasoning, risk_flags=risk_flags,
            supporting_factors=supporting, negative_factors=negative,
            model="mock",
        )
