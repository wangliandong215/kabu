"""
risk/portfolio_risk_engine.py — V3.3 Portfolio Risk Engine, the unified gate
described in the spec's final principle: "整个交易系统唯一、统一、可审计的
Portfolio Risk Gate". PortfolioRiskEngine.evaluate() is the one function
engine/runner.py::_place_order() calls for every order (BUY and SELL).

Aggregation rule (spec section 十二/十三 — RiskDecision, never a bool):
    any sub-check BLOCK  -> overall BLOCK
    else any sub-check WARN (incl. DATA_UNAVAILABLE mapped to WARN) -> overall WARN
    else                  -> overall ALLOW
violations/warnings/metrics from every sub-check are merged onto the single
returned RiskDecision, so a BLOCK line in the log is self-explanatory
(which rule, current value, limit) without cross-referencing other calls.

Never raises: any sub-check module throwing (bug, unexpected data shape) is
caught individually and downgraded to a WARN with the exception recorded in
metrics, rather than ever taking down _place_order() — same "new/
experimental logic fails open with visibility" convention already used by
position_manager/llm_interface.py::review_position() and
risk/qqq_core_recovery.py's Level 3 plan-only path.

Relationship to existing modules (spec section 二十一/二十二 — NOT a
replacement for either):
    risk/portfolio_risk_manager.py (v2.10)      — QQQ concentration / total
                                                    exposure tiers / Emergency
                                                    Rebalance: untouched, still
                                                    the authority for handling
                                                    ALREADY-excess risk.
    risk/portfolio_position_manager.py (v2.12)  — regime exposure budget:
                                                    untouched; this module
                                                    reuses its
                                                    classify_market_regime()
                                                    via risk/market_regime_risk.py.
    This module (V3.3)                           — BLOCKs NEW risk only, never
                                                    liquidates or rebalances
                                                    anything itself.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import config
from risk import beta_risk, correlation_risk, exposure_risk, market_regime_risk
from risk import position_limits_risk, sector_risk, var_risk
from risk.llm_advisor import LLMRiskAdvisor
from risk.portfolio_state import PortfolioState
from risk.risk_decision import OrderIntent, RiskDecision, RiskStatus, allow

_LOG_PATH = Path(r"C:\KabuData\portfolio\risk_engine_log.jsonl")

# Each entry: (name, check_fn(order, state) -> (sub_result, RiskDecision))
_SUB_CHECKS = (
    ("exposure", exposure_risk.check),
    ("position_count", position_limits_risk.check_position_count),
    ("position_weight", position_limits_risk.check_position_weight),
    ("sector", sector_risk.check),
    ("beta", beta_risk.check),
    ("var", var_risk.check),
    ("market_regime", market_regime_risk.check),
    ("correlation", correlation_risk.check),
)


class PortfolioRiskEngine:
    def __init__(self, llm_advisor: Optional[LLMRiskAdvisor] = None):
        self._llm_advisor = llm_advisor or LLMRiskAdvisor()

    def evaluate(self, order: OrderIntent, state: PortfolioState) -> RiskDecision:
        violations: List[str] = []
        warnings: List[str] = []
        metrics = {}
        worst = RiskStatus.ALLOW
        reasons: List[str] = []

        for name, check_fn in _SUB_CHECKS:
            try:
                _, sub_decision = check_fn(order, state)
            except Exception as exc:
                warnings.append(f"{name.upper()}_CHECK_FAILED")
                metrics[f"{name}_error"] = f"{exc.__class__.__name__}: {exc}"
                if worst == RiskStatus.ALLOW:
                    worst = RiskStatus.WARN
                continue

            metrics.update(sub_decision.metrics)
            violations.extend(sub_decision.violations)
            warnings.extend(sub_decision.warnings)

            if sub_decision.status == RiskStatus.BLOCK:
                worst = RiskStatus.BLOCK
                reasons.append(sub_decision.reason)
            elif sub_decision.status == RiskStatus.WARN and worst != RiskStatus.BLOCK:
                worst = RiskStatus.WARN
                reasons.append(sub_decision.reason)

        if worst == RiskStatus.BLOCK:
            reason = "; ".join(r for r in reasons if r) or "risk limit exceeded"
        elif worst == RiskStatus.WARN:
            reason = "; ".join(r for r in reasons if r) or "risk warning"
        else:
            reason = "within all configured risk limits"

        decision = RiskDecision(
            status=worst, allowed=(worst != RiskStatus.BLOCK), reason=reason,
            violations=violations, warnings=warnings, metrics=metrics,
        )

        self._log(order, state, decision)

        try:
            self._llm_advisor.analyze(order, state, decision)
        except Exception:
            pass   # LLMRiskAdvisor already fails open internally; belt & suspenders.

        return decision

    def _log(self, order: OrderIntent, state: PortfolioState, decision: RiskDecision) -> None:
        try:
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            record = decision.to_log_dict(order)
            record["current_position_qty"] = (state.get(order.code).qty
                                               if state.get(order.code) else 0)
            record["total_exposure_pct"] = state.total_exposure_pct
            record["position_count"] = state.position_count
            record["market_regime"] = state.market_regime
            record["mode"] = config.RISK_ENGINE_MODE
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass


def evaluate_order(code: str, side: str, qty: float, price: float,
                    state: PortfolioState, strategy: str = "",
                    engine: Optional[PortfolioRiskEngine] = None) -> RiskDecision:
    """Convenience entry point for callers that don't want to construct
    OrderIntent/PortfolioRiskEngine themselves — this is what
    engine/runner.py::_place_order() calls."""
    if not config.RISK_ENGINE_ENABLED or config.RISK_ENGINE_MODE == "OFF":
        return allow("risk engine disabled")
    order = OrderIntent(code=code, side=side, qty=qty, price=price, strategy=strategy)
    eng = engine or PortfolioRiskEngine()
    return eng.evaluate(order, state)
