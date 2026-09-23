"""
regime/rule_provider.py — RuleRegimeProvider: the only RegimeProvider
actually used today (config.LLM_ENABLED=False, see regime/factory.py).

Reuses the exact three signals engine/runner.py already computes every
pass, WITHOUT recomputing them, to avoid any semantic drift from the
already-validated implementations those three functions provide:

  weather_code  — engine/market_weather.py::market_weather()'s 0/1/2
                   crisis/chop/safe scale (int; runner.py defaults it to 1
                   if the underlying kline fetch/compute fails — see
                   engine/runner.py's "weather_code = 1 # safe/cautious
                   default" comment. Never actually None in production,
                   but this provider tolerates None defensively).
  qqq_above_ma  — runner.py::_qqq_above_ma()'s QQQ MA200 trend gate (bool;
                   that function itself catches all exceptions and
                   returns False on failure — never raises, never None in
                   production, tolerated here defensively).
  drawdown_halt — risk/guard.py::check_max_drawdown()'s halt flag, negated
                   (bool; always present, engine/runner.py computes it
                   unconditionally at the top of run_once()).

IMPORTANT — this is a SEPARATE classification from
risk/portfolio_position_manager.py::classify_market_regime()'s existing
BULL/NORMAL/CAUTION/RISK_OFF output. That function, its 3-input decision
table, and every call site that consumes it (compute_exposure_budget(),
the 2a/2b/2c/2d buy-path gating in engine/runner.py) are completely
untouched by this module and by this whole regime/ package. This class
produces a differently-named BULL/NEUTRAL/BEAR/CRASH taxonomy purely for
the LLM-ready Market Regime Observation Layer described in
regime/__init__.py — its output is logged only and is never passed into
classify_market_regime(), compute_exposure_budget(), or any BUY/SELL/
sizing/exposure decision. Do not merge these two classifications; they
exist for different consumers and may diverge in the future (e.g. once a
local LLM or a new signal like VIX/breadth is added here).

The decision table below intentionally mirrors classify_market_regime()'s
structure (same three inputs, same priority order: drawdown/crisis first,
missing data next, chop/below-MA200 next, safe+above-MA200 last) so this
provider's four states map onto that function's four states conceptually
(CRASH~RISK_OFF, BEAR~CAUTION, NEUTRAL~NORMAL, BULL~BULL) — but the two
functions are independent code paths and must be changed independently.
"""
from typing import List, Tuple

from regime.models import (MarketContext, MarketRegime, SOURCE_RULES,
                            REGIME_BULL, REGIME_NEUTRAL, REGIME_BEAR, REGIME_CRASH)
from regime.provider import RegimeProvider


class RuleRegimeProvider(RegimeProvider):
    def evaluate(self, context: MarketContext) -> MarketRegime:
        weather_code = context.weather_code
        qqq_above_ma = context.qqq_above_ma
        drawdown_halt = context.drawdown_halt

        reason_codes = _reason_codes(weather_code, qqq_above_ma, drawdown_halt)
        regime, risk_level = _classify(weather_code, qqq_above_ma, drawdown_halt)
        confidence = 0.9 if (weather_code is not None and qqq_above_ma is not None) else 0.5

        return MarketRegime(regime=regime, confidence=confidence, risk_level=risk_level,
                             reason_codes=reason_codes, source=SOURCE_RULES)


def _reason_codes(weather_code, qqq_above_ma, drawdown_halt: bool) -> List[str]:
    codes = []
    if drawdown_halt:
        codes.append("DRAWDOWN_HALT")
    if weather_code == 0:
        codes.append("WEATHER_CRISIS")
    elif weather_code == 1:
        codes.append("WEATHER_CHOP")
    elif weather_code == 2:
        codes.append("WEATHER_SAFE")
    else:
        codes.append("WEATHER_UNKNOWN")
    if qqq_above_ma is True:
        codes.append("QQQ_ABOVE_MA200")
    elif qqq_above_ma is False:
        codes.append("QQQ_BELOW_MA200")
    else:
        codes.append("QQQ_MA_UNKNOWN")
    return codes


def _classify(weather_code, qqq_above_ma, drawdown_halt: bool) -> Tuple[str, float]:
    if drawdown_halt or weather_code == 0:
        return REGIME_CRASH, 0.0
    if weather_code is None or qqq_above_ma is None:
        return REGIME_BEAR, 0.3   # degraded data -> conservative, mirrors CAUTION's default
    if weather_code == 1 or not qqq_above_ma:
        return REGIME_BEAR, 0.3
    if weather_code == 2 and qqq_above_ma:
        return REGIME_BULL, 1.0
    return REGIME_NEUTRAL, 0.6    # defensive fallback, not reachable with today's 3 inputs
