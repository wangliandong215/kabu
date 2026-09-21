"""
engine/exit_diagnostics.py — Exit Diagnostics (v2.9.x, observation-only).

Motivation: the September 2026 loss-attribution investigation found that
trades exiting via STRATEGY_EXIT(atr_breakout) mix two unrelated shapes
under one label — "the trade never really worked" (MFE<5%, closed negative)
and "a real trend winner gave back part of its gain" (MFE>=10%, closed
positive) — and a naive "STRATEGY_EXIT gives back too much profit" read of
the data conflates them. This module classifies the SHAPE of a closed
trade's MFE/MAE/pnl relationship into a small stable category set, purely
from numbers `trade_tracker.log_exit()` already computed and stored
(mfe/mae/pnl/position_value/holding_days) — it does not fetch new price
data, and it is agnostic to which exit_reason_code actually fired (a
STOP_LOSS or ATR_TRAIL trade gets classified exactly the same way a
STRATEGY_EXIT trade does).

Hard constraint (per product decision, same as engine/entry_quality.py):
nothing in this module may feed back into a trading decision. classify_exit()
is pure and never raises — degrades to category="OTHER" / None fields on
missing/insufficient data instead of throwing, so it is always safe to call
from trade_tracker.log_exit_diagnostics() right after a trade's exit leg is
recorded.

Category definitions (thresholds are an initial, explicit codification of
the 2026-09 investigation's own rules — expected to be revisited once more
samples accumulate, see analyze_exit_diagnostics.py):

  EARLY_FAILURE    mfe_pct < 5%  AND final pnl < 0
      The trade never proved itself — it barely (or never) went favorable
      before drifting to a loss without ever hitting STOP_LOSS. This is a
      *stop-loss/exit-timing gap* question, not a giveback question.

  TREND_REVERSAL   mfe_pct >= 5% AND final pnl < 0
      The trade had a real favorable move (>=5%) at some point but reversed
      past breakeven into a net loss — giveback > 100% of MFE by
      construction. Distinct from EARLY_FAILURE: this one DID work for a
      while.

  EXTREME_GIVEBACK mfe_pct >= 10% AND final pnl >= 0 AND giveback_pct >= 75%
      A genuine double-digit trend move where the exit mechanism let more
      than three-quarters of the peak evaporate before closing. The
      2026-09 investigation's specific watch condition (see engine/
      exit_diagnostics.py's module docstring in the corresponding chat
      record) — 3-5 repeats of this is the threshold for reconsidering
      STRATEGY_EXIT's timing, not one.

  PROFIT_GIVEBACK  mfe_pct >= 5% AND final pnl >= 0, not EXTREME_GIVEBACK
      The normal, expected trend-following shape: gave back some of the
      peak to let the trend run, closed positive. Not evidence of a
      problem on its own.

  OTHER            mfe_pct < 5% AND final pnl >= 0
      Negligible peak, closed flat/marginally positive — no giveback story
      either way.
"""
from typing import Optional

EARLY_FAILURE = "EARLY_FAILURE"
TREND_REVERSAL = "TREND_REVERSAL"
EXTREME_GIVEBACK = "EXTREME_GIVEBACK"
PROFIT_GIVEBACK = "PROFIT_GIVEBACK"
OTHER = "OTHER"

_MFE_EARLY_FAILURE_THRESHOLD = 0.05   # mfe_pct below this = "never really worked"
_MFE_EXTREME_TIER_THRESHOLD = 0.10    # mfe_pct at/above this = "real double-digit move"
_GIVEBACK_EXTREME_THRESHOLD = 0.75    # giveback_pct at/above this = "gave back the bulk of it"


def classify_exit(
    mfe: Optional[float],
    mae: Optional[float],
    pnl: Optional[float],
    position_value: Optional[float],
    holding_days: Optional[float] = None,
    stop_loss_triggered: Optional[bool] = None,
    strategy_exit_triggered: Optional[bool] = None,
) -> Optional[dict]:
    """Classify one closed trade's MFE/MAE/pnl shape.

    mfe/mae : dollar max favorable/adverse excursion, as already stored on
        trades.mfe/trades.mae (aggregated from trade_daily by log_exit()).
    pnl : dollar realized pnl, as already stored on trades.pnl.
    position_value : dollar cost basis, as already stored on
        trades.position_value — the denominator for mfe_pct/mae_pct
        (matches trade_entry_quality's distance_from_20d_high convention of
        expressing everything as a fraction, not a raw dollar figure).
    holding_days : passed through for convenience so callers/analysis don't
        need a second lookup; not used in the classification itself.
    stop_loss_triggered / strategy_exit_triggered : caller-supplied booleans
        derived from exit_reason_code (STOP_LOSS / EXIT_STRATEGY_SIGNAL) —
        passed through as-is, this module does not re-derive them from a
        reason string so it stays agnostic to exit_reason's exact spelling.

    Never raises. Returns None only when there isn't enough to classify at
    all (mfe or pnl or position_value missing/zero) — otherwise returns a
    dict with whatever subset could be computed, remaining keys None.
    """
    try:
        if mfe is None or pnl is None or not position_value:
            return None

        mfe_pct = mfe / position_value
        mae_pct = (mae / position_value) if mae is not None else None
        # giveback_pct: how much of the peak was given back by the close.
        # mfe<=0 means price never went favorably enough to have anything to
        # give back — giveback_pct is undefined (None), not 0 or inf.
        giveback_pct = ((mfe - pnl) / mfe) if mfe > 0 else None
        mfe_capture = (pnl / mfe) if mfe > 0 else None

        if mfe_pct < _MFE_EARLY_FAILURE_THRESHOLD:
            category = EARLY_FAILURE if pnl < 0 else OTHER
        elif pnl < 0:
            category = TREND_REVERSAL
        elif mfe_pct >= _MFE_EXTREME_TIER_THRESHOLD and giveback_pct is not None \
                and giveback_pct >= _GIVEBACK_EXTREME_THRESHOLD:
            category = EXTREME_GIVEBACK
        else:
            category = PROFIT_GIVEBACK

        return {
            "exit_category": category,
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "giveback_pct": giveback_pct,
            "mfe_capture": mfe_capture,
            "holding_days": holding_days,
            "stop_loss_triggered": stop_loss_triggered,
            "strategy_exit_triggered": strategy_exit_triggered,
        }
    except Exception:
        return None
