"""
risk/llm_advisor.py — V3.3 LLM Risk Advisor (spec section 十四/十五).

Authority chain this file locks in:

    PortfolioRiskEngine.evaluate()
            |
    RiskDecision  (final — status/allowed/violations already set)
            |
    LLMRiskAdvisor.analyze()   <-- this module, always optional, always AFTER
            |
    (log / dashboard / future UI — advisory only)

The LLM can: analyze, explain, summarize, flag anomalies, suggest. The LLM
cannot: bypass the Risk Engine, place an order, override BLOCK, change a
hard limit, or auto-liquidate — there is no code path anywhere in this
module (or risk/portfolio_risk_engine.py) that lets LLMRiskAnalysis feed
back into RiskDecision.allowed/status. analyze() is called strictly after
the RiskDecision is finalized.

Mirrors position_manager/llm_interface.py::review_position()'s exact
fail-open contract: any exception (including a placeholder/future backend
timing out or erroring) is caught here and logged; the caller always gets
either a real LLMRiskAnalysis or None, never an exception, never affecting
the trading pass.

Modes (config.RISK_ENGINE_LLM_MODE): OFF (default, NullLLMProvider, always
succeeds trivially) / SHADOW (MockLLMProvider today — no real backend is
implemented yet; wiring a real risk/llm/*_provider.py in is future work).
There is deliberately no ACTIVE mode — see module docstring above.
"""
from typing import Any, Dict, Optional

import config
from risk.llm.base import LLMProvider, LLMRiskAnalysis
from risk.llm.null_provider import NullLLMProvider
from risk.llm.mock_provider import MockLLMProvider
from risk.risk_decision import OrderIntent, RiskDecision
from risk.portfolio_state import PortfolioState

MODE_OFF = "OFF"
MODE_SHADOW = "SHADOW"
VALID_MODES = (MODE_OFF, MODE_SHADOW)


def get_llm_provider(mode: Optional[str] = None) -> LLMProvider:
    mode = mode if mode is not None else config.RISK_ENGINE_LLM_MODE
    if mode == MODE_SHADOW:
        return MockLLMProvider()
    return NullLLMProvider()


def build_context(order: OrderIntent, state: PortfolioState, decision: RiskDecision) -> Dict[str, Any]:
    return {
        "order": {"code": order.code, "side": order.side, "qty": order.qty, "price": order.price},
        "portfolio": {"total_assets": state.total_assets, "cash": state.cash,
                      "total_exposure_pct": state.total_exposure_pct,
                      "position_count": state.position_count,
                      "market_regime": state.market_regime},
        "risk_decision": decision.to_log_dict(),
    }


class LLMRiskAdvisor:
    def __init__(self, provider: Optional[LLMProvider] = None, mode: Optional[str] = None):
        self._mode = mode if mode is not None else config.RISK_ENGINE_LLM_MODE
        self._provider = provider

    def analyze(self, order: OrderIntent, state: PortfolioState,
                decision: RiskDecision) -> Optional[LLMRiskAnalysis]:
        """Never raises. Returns None if the mode is OFF-equivalent or the
        provider fails for any reason — the RiskDecision that was already
        finalized before this call is always what governs behavior."""
        if self._mode not in VALID_MODES or self._mode == MODE_OFF:
            return None

        provider = self._provider or get_llm_provider(self._mode)
        try:
            context = build_context(order, state, decision)
            return provider.analyze(context)
        except Exception as exc:
            import notify.alert as alert
            alert.log(f"risk.llm_advisor: analyze failed for {order.code} "
                      f"({exc.__class__.__name__}: {exc}) — advisory skipped, "
                      f"RiskDecision unaffected")
            return None
