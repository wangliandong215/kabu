"""
regime/ — LLM-ready Market Regime Observation Layer (added 2026-09-23).

WHAT THIS PACKAGE IS
  A standalone, read-only classification of the overall market environment
  (BULL/NEUTRAL/BEAR/CRASH, see regime/models.py), computed once per
  engine/runner.py::run_once() pass and appended to
  C:\\KabuData\\portfolio\\regime_log.jsonl. Today it is produced by
  RuleRegimeProvider (regime/rule_provider.py) from three signals
  run_once() already computes for other purposes. config.LLM_ENABLED
  (default False) is the only switch that will ever change which
  RegimeProvider runs — see regime/factory.py.

WHY IT EXISTS
  So that plugging in a local LLM later (Qwen/Llama/Mistral/Gemma/...)
  only means: (1) implement regime/llm_provider.py::LocalLLMRegimeProvider
  for real, (2) flip config.LLM_ENABLED to True. No change to
  engine/runner.py's call site, risk/portfolio_position_manager.py,
  risk/portfolio_risk_manager.py, strategies/, or the order-execution
  path is required. See regime/provider.py::RegimeProvider for the
  contract that makes this swap possible.

WHAT THIS PACKAGE IS NOT (as of this phase)
  - It is NOT risk/portfolio_position_manager.py::classify_market_regime().
    That function's BULL/NORMAL/CAUTION/RISK_OFF output, and every
    decision downstream of it (compute_exposure_budget(), the 2a/2b/2c/2d
    buy-path gating in engine/runner.py), is untouched by this package —
    zero lines of risk/portfolio_position_manager.py or
    risk/portfolio_risk_manager.py changed to add this layer.
  - It does NOT gate, size, or block any BUY/SELL. engine/runner.py calls
    evaluate_and_log() purely for its logging side effect; the returned
    MarketRegime is not read by any decision branch. Do not add
    `if market_regime.regime == ...` anywhere in a trading decision path
    without a separate, explicit design/approval step first.
  - It does NOT touch QQQ's MA200 exit, QQQ Core 25% target, QQQ
    concentration control, or Emergency Rebalance — those Hard Rules live
    entirely in engine/runner.py and risk/portfolio_risk_manager.py and
    have no dependency on this package.
  - It does NOT invoke any real local LLM. LocalLLMRegimeProvider
    (regime/llm_provider.py) is a placeholder that always raises
    NotImplementedError; config.LLM_ENABLED defaults to (and must stay)
    False until a real implementation lands.

FAILURE HANDLING
  evaluate_and_log() never raises. A failure computing the regime, or a
  failure writing the log line, is caught, reported via notify.alert, and
  degrades to returning None — engine/runner.py's pass continues exactly
  as if this package were absent.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from regime.factory import get_regime_provider
from regime.models import MarketContext, MarketRegime
from regime.provider import RegimeProvider
from regime.rule_provider import RuleRegimeProvider
from regime.llm_provider import LocalLLMRegimeProvider

REGIME_LOG_PATH = Path(r"C:\KabuData\portfolio\regime_log.jsonl")

__all__ = [
    "MarketContext", "MarketRegime", "RegimeProvider",
    "RuleRegimeProvider", "LocalLLMRegimeProvider",
    "get_regime_provider", "evaluate_and_log", "REGIME_LOG_PATH",
]


def evaluate_and_log(context: MarketContext,
                      provider: Optional[RegimeProvider] = None) -> Optional[MarketRegime]:
    """Compute this pass's MarketRegime and append one JSON line to
    REGIME_LOG_PATH. Observation only — see module docstring above for
    what the caller must NOT do with the return value.

    `provider` lets tests inject a specific RegimeProvider instead of
    going through get_regime_provider()/config.LLM_ENABLED. Never raises;
    returns None (and logs a warning) if evaluation itself fails, so a
    bug in a RegimeProvider can never block engine/runner.py.
    """
    import notify.alert as alert

    try:
        active_provider = provider or get_regime_provider()
        result = active_provider.evaluate(context)
    except Exception as exc:
        alert.log(f"regime: evaluate_and_log failed, skipping this pass's "
                  f"regime observation — {exc}")
        return None

    _log(context, result)
    return result


def _log(context: MarketContext, result: MarketRegime) -> None:
    """Best-effort JSONL append. A logging failure (missing dir, disk
    full, serialization error, ...) must never propagate — same contract
    as risk/portfolio_position_manager.py::log_attempt()."""
    try:
        REGIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "regime": result.regime,
            "confidence": result.confidence,
            "risk_level": result.risk_level,
            "reason_codes": result.reason_codes,
            "source": result.source,
            "weather_code": context.weather_code,
            "qqq_above_ma": context.qqq_above_ma,
            "drawdown_halt": context.drawdown_halt,
        }
        with open(REGIME_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        import notify.alert as alert
        alert.log(f"regime: failed to write regime_log.jsonl — {exc}")
