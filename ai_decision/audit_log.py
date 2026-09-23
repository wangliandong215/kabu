"""
ai_decision/audit_log.py — V3.5 AI Decision Audit (spec section 八).

JSONL append, no DB — same convention as
risk/portfolio_risk_engine.py::_LOG_PATH / risk_engine_log.jsonl. One line
per AIDecision so a future review can answer "AI 当时为什么做出这个判断"
and "AI 的 BUY/HOLD/REDUCE/SKIP 哪些情况下有效" (spec 八) without touching
the trading path. Never raises — same fail-silent contract as every other
*_LOG_PATH writer in risk/ (portfolio_risk_engine.py, portfolio_risk_manager
.py, qqq_core_recovery.py, portfolio_position_manager.py all swallow
logging failures with `except Exception: pass`), since a disk-full or
permission error here must never affect ai_decision/decision_layer.py's
caller.
"""
import json
from pathlib import Path

from ai_decision.schema import AIDecision, AIDecisionContext

_LOG_PATH = Path(r"C:\KabuData\ai_decision\ai_decision_log.jsonl")


def log_decision(ctx: AIDecisionContext, decision: AIDecision) -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "input_snapshot": {
                "symbol": ctx.symbol,
                "market_regime": ctx.market_regime,
                "portfolio_snapshot": ctx.portfolio_snapshot,
                "risk_decision_snapshot": ctx.risk_decision_snapshot,
                "news_snapshot": ctx.news_snapshot,
                "event_snapshot": ctx.event_snapshot,
                "trade_history_snapshot": ctx.trade_history_snapshot,
                "macro_snapshot": ctx.macro_snapshot,
                "financial_snapshot": ctx.financial_snapshot,
                "institution_snapshot": ctx.institution_snapshot,
            },
            **decision.to_log_dict(symbol=ctx.symbol),
        }
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
