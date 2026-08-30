"""
risk/earnings.py — Earnings blackout guard.

Forbids opening NEW positions in any stock within
config.EVENT_RISK_EARNINGS_BLOCK_DAYS trading days of its earnings
announcement. Earnings gaps routinely gap through stop-losses, making the
2%-risk-per-trade guarantee meaningless.

v2.11: this is now a thin wrapper around engine/event_risk.py (which in turn
sits on data/earnings.py's cached get_earnings_calendar() calls) — kept as a
separate module purely so engine/pipeline.py's existing call site
(is_earnings_blackout) doesn't need to change.
"""
import engine.event_risk as event_risk


def is_earnings_blackout(code: str) -> bool:
    """
    True if the stock has earnings within config.EVENT_RISK_EARNINGS_BLOCK_DAYS
    trading days from today. Fails open (returns False) on any API error so
    trading is not blocked by a data-availability problem.
    """
    return event_risk.has_earnings_risk(code)
