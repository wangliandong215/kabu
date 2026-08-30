"""
engine/research_snapshot.py — Trade Research Snapshot builder (v2.11.1
Priority 3).

Single responsibility: assemble the dict engine/trade_tracker.py's
log_research_snapshot() persists, answering "what did the system actually
see when this trade fired" — permanently, so future research never has to
(and never can) re-derive it from data that has since moved on.

Hard constraint: this module must NEVER call OpenD. Every value here comes
from either:
  - `result`/`cand` — already computed in-memory by the caller for this
    exact signal, zero extra work;
  - engine.event_risk — reads data/earnings.py's process-local, 6h-TTL
    cache (already populated earlier this same run_once() pass by
    engine/pipeline.py's earnings-blackout filter), not a fresh OpenD call
    in the normal case;
  - research/market_features_store.py's latest_before() — a local SQLite
    read of whatever the daily collector already wrote, filtered to rows
    observed_at <= this snapshot's own observed_at (Look-ahead-safe: never
    picks up a research row written after this moment).
If a Research Features row simply doesn't exist yet for a code (collector
hasn't run today, or never covers this code), the corresponding fields stay
None — this function never blocks on or retries a missing row.
"""
from datetime import date
from typing import Optional

import common
import engine.event_risk as event_risk
import engine.market_hours as market_hours
from research.market_features_store import MarketFeaturesStore


def build_trade_research_snapshot(code: str, result: dict, cand, signal_time: str) -> dict:
    observed_at = common.utc_now_iso()
    market_date = market_hours.market_date_for(code).isoformat()

    snapshot = {
        "signal_time": signal_time,
        "observed_at": observed_at,
        "market_date": market_date,
        "rsi14": result.get("rsi14"),
        "atr": result.get("atr"),
        "current_price": result.get("current_price", getattr(cand, "current_price", None)),
        "signal_strength": getattr(cand, "signal_strength", None) or result.get("signal_strength"),
        "total_score": getattr(cand, "total_score", None),
        "confidence_score": getattr(cand, "confidence", None),
    }

    _add_event_risk_fields(snapshot, code)
    _add_research_feature_fields(snapshot, code, as_of=observed_at)
    return snapshot


def _add_event_risk_fields(snapshot: dict, code: str) -> None:
    try:
        snapshot["has_earnings_risk"] = int(event_risk.has_earnings_risk(code))
        snapshot["days_to_earnings"] = event_risk.days_to_earnings(code)
        edate: Optional[date] = event_risk.earnings_date(code)
        snapshot["earnings_date"] = edate.isoformat() if edate else None
        snapshot["earnings_session"] = event_risk.earnings_session(code)
    except Exception:
        pass   # Event Risk fields stay unset (NULL) — never blocks the snapshot


def _add_research_feature_fields(snapshot: dict, code: str, as_of: str) -> None:
    """Reads ONLY local SQLite (MarketFeaturesStore.latest_before) — no
    OpenD call under any circumstance, per this module's hard constraint."""
    try:
        store = MarketFeaturesStore()
    except Exception:
        return
    try:
        try:
            news_row = store.latest_before("news_snapshots", {"code": code}, as_of)
            if news_row:
                snapshot["news_count_24h"] = news_row.get("news_count_24h")
                snapshot["news_observed_at"] = news_row.get("observed_at")
        except Exception:
            pass

        try:
            opt_row = store.latest_before("options_snapshots", {"code": code}, as_of)
            if opt_row:
                snapshot["iv"] = opt_row.get("iv")
                snapshot["hv_30d"] = opt_row.get("hv_30d")
                snapshot["put_call_ratio"] = opt_row.get("put_call_ratio")
                snapshot["options_observed_at"] = opt_row.get("observed_at")
        except Exception:
            pass

        try:
            # No natural per-code filter for FedWatch (it's a whole-Fed
            # snapshot, not per-symbol) — picks the most recently observed
            # row as of `as_of`; ties among rows sharing one observed_at
            # (the normal case — one run_fedwatch_pass() writes several
            # meeting_date rows together) break arbitrarily, which is fine
            # here since this field is a coarse "what was the Fed-rate
            # backdrop" reference, not a precise per-meeting lookup.
            fw_row = store.latest_before("fedwatch_target_rate_snapshots", {}, as_of)
            if fw_row:
                snapshot["fedwatch_target_range"] = fw_row.get("target_range")
                snapshot["fedwatch_probability"] = fw_row.get("probability")
                snapshot["fedwatch_observed_at"] = fw_row.get("observed_at")
        except Exception:
            pass
    finally:
        store.close()
