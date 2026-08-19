"""
research/scan_mean_reversion_candidates.py — daily standalone Mean Reversion
observation scan (v2.9.x).

Deliberately a standalone script, same isolation discipline as
engine/hmm_shadow.py: does NOT import portfolio/, risk/, strategies/, or
engine/runner.py, so there is no code path by which running it can affect a
trading decision. Meant to run once per day via a separate Scheduled Task
("Kabu Research Scan"), fully independent of main.py's
`--interval 300` live trading loop.

Two passes, both idempotent (safe to re-run the same day):
  1. Detect — for each config.WATCHLIST code, evaluate the loose
     extreme-oversold trigger (research.mean_reversion_metrics.
     evaluate_trigger — a RECORDING filter only, never a trading signal).
     On a fresh trigger (respecting a per-code cooldown so one selloff
     doesn't spawn dozens of overlapping events), build the full snapshot
     (categories 1-8) and upsert one row.
  2. Backfill — for every stored row not yet at 60 forward bars, fetch
     fresh price history and fill in whatever forward-return horizons
     (category 9) have now elapsed, deriving the SUCCESS/FAILURE/NEUTRAL
     outcome label.

No DingTalk/Telegram push from this script — this is a research artifact,
not an actionable trading alert.
"""
import argparse
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

import config
from data.fetcher import fetch_kline
from engine import market_context, regime_store
from engine.hmm_shadow import run_shadow_pass
from research import fundamentals
from research.forward_outcomes import compute_forward_outcomes, derive_outcome_label
from research.mean_reversion_metrics import build_observation_snapshot, evaluate_trigger
from research.sector_etf import sector_etf_for
from research.store import ObservationStore

_MARKET_ENV_CODES = {"US.SPY", "US.QQQ"}


def _bars_after(code: str, after_date_iso: str, bars: int = 300) -> Optional[pd.DataFrame]:
    """Fetch recent kline and slice to bars strictly AFTER after_date_iso —
    same convention as analyze_entry_quality.py's _bars_after_entry()."""
    try:
        df = fetch_kline(code, "1d", bars)
        if df is None or len(df) == 0 or "time_key" not in df.columns:
            return None
        after_date = pd.Timestamp(after_date_iso).normalize()
        times = pd.to_datetime(df["time_key"])
        sliced = df[times.dt.normalize() > after_date].reset_index(drop=True)
        return sliced if len(sliced) else None
    except Exception:
        return None


def _event_date_of(df: pd.DataFrame) -> str:
    return pd.to_datetime(df["time_key"]).iloc[-1].strftime("%Y-%m-%d")


def _cooldown_active(code: str, df: pd.DataFrame, store: ObservationStore) -> bool:
    last_event_date = store.find_latest_event_date(code)
    if last_event_date is None:
        return False
    gap_df = _bars_after(code, last_event_date, bars=len(df))
    trading_days_since = len(gap_df) if gap_df is not None else 0
    return trading_days_since < config.MR_TRIGGER_COOLDOWN_DAYS


def _hmm_snapshot(code: str, as_of: str) -> dict:
    """Read regime_store before/after a shadow-mode update to derive the
    transition edges the spec asks for (Bear->Sideways, Bear->Bull)."""
    prev = regime_store.get_regime_interface(code)
    prev_regime = prev.get("current_regime")
    try:
        run_shadow_pass([code], as_of=as_of)
    except Exception:
        pass
    new = regime_store.get_regime_interface(code)
    new["prev_regime"] = prev_regime
    new["transition_bear_to_sideways"] = bool(
        prev_regime == "Bear" and new.get("current_regime") == "Sideways")
    new["transition_bear_to_bull"] = bool(
        prev_regime == "Bear" and new.get("current_regime") == "Bull")
    return new


def _vix_close() -> Optional[float]:
    try:
        ctx = market_context.get_latest_context()
        return ctx.get("vix_close") if ctx else None
    except Exception:
        return None


def run_detect_pass(codes: list, store: ObservationStore) -> int:
    """Returns number of new/updated candidate events."""
    spy_df = fetch_kline("US.SPY", "1d", 60)
    qqq_df = fetch_kline("US.QQQ", "1d", 60)
    vix_close = _vix_close()
    sector_etf_cache: dict = {}
    n_recorded = 0

    for code in codes:
        if code in _MARKET_ENV_CODES:
            continue   # SPY/QQQ are comparison series, not candidates themselves
        try:
            df = fetch_kline(code, "1d", 300)
            if df is None or len(df) < 6:
                print(f"[skip] {code}: no/insufficient price history")
                continue

            trigger = evaluate_trigger(df)
            if trigger is None:
                continue
            if _cooldown_active(code, df, store):
                print(f"[skip] {code}: triggered ({trigger}) but within "
                      f"{config.MR_TRIGGER_COOLDOWN_DAYS}-trading-day cooldown")
                continue

            event_date = _event_date_of(df)
            hmm_iface = _hmm_snapshot(code, event_date)

            sector = config.SECTOR_MAP.get(code)
            etf = sector_etf_for(sector) if sector else None
            sector_df = None
            if etf:
                if etf not in sector_etf_cache:
                    sector_etf_cache[etf] = fetch_kline(etf, "1d", 60)
                sector_df = sector_etf_cache[etf]

            snapshot = build_observation_snapshot(
                code, df, spy_df=spy_df, qqq_df=qqq_df, sector_etf_df=sector_df,
                vix_close=vix_close, hmm_interface=hmm_iface, trigger_reason=trigger,
            )
            if snapshot is None:
                print(f"[skip] {code}: triggered but snapshot build failed")
                continue

            fnd = fundamentals.fetch_fundamentals(code)
            record = {
                "event_id": f"{code}_{event_date}",
                "event_date": event_date,
                "detected_at": datetime.now().isoformat(),
                **snapshot,
                **fnd,
            }
            store.upsert(record)
            n_recorded += 1
            print(f"[ok] {code}: new candidate event {event_date} ({trigger})")
        except Exception as e:
            print(f"[error] {code}: {e!r}")

    return n_recorded


def run_backfill_pass(store: ObservationStore) -> int:
    """Returns number of rows updated with fresh forward-return data."""
    n_updated = 0
    for row in store.rows_needing_backfill():
        code, event_date = row["code"], row["event_date"]
        try:
            after_df = _bars_after(code, event_date)
            if after_df is None:
                continue
            entry_df = fetch_kline(code, "1d", 300)
            event_day_low = None
            if entry_df is not None and len(entry_df):
                match = entry_df[pd.to_datetime(entry_df["time_key"]).dt.strftime("%Y-%m-%d") == event_date]
                if len(match):
                    event_day_low = float(match.iloc[-1]["low"])

            entry_price = row.get("manual_entry_price")
            if entry_price is None and entry_df is not None and len(entry_df):
                match = entry_df[pd.to_datetime(entry_df["time_key"]).dt.strftime("%Y-%m-%d") == event_date]
                if len(match):
                    entry_price = float(match.iloc[-1]["close"])
            if not entry_price:
                continue

            forward = compute_forward_outcomes(after_df, entry_price, event_day_low)
            if forward is None:
                continue
            forward["outcome_label"] = derive_outcome_label(forward)
            store.upsert({"event_id": row["event_id"], "code": code,
                           "event_date": event_date, **forward})
            n_updated += 1
        except Exception as e:
            print(f"[error] backfill {code}_{event_date}: {e!r}")
    return n_updated


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Mean Reversion research scan — observation only, "
                     "never affects live trading. See module docstring.")
    parser.add_argument("--codes", nargs="+", default=None,
                         help="Override the code list (default: config.WATCHLIST).")
    parser.add_argument("--no-detect", action="store_true",
                         help="Skip the trigger-detection pass (backfill only).")
    parser.add_argument("--no-backfill", action="store_true",
                         help="Skip the forward-return backfill pass.")
    args = parser.parse_args(argv)

    codes = args.codes or config.WATCHLIST
    store = ObservationStore()
    try:
        if not args.no_detect:
            n_new = run_detect_pass(codes, store)
            print(f"\nDetect pass: {n_new} new candidate event(s) recorded.")
        if not args.no_backfill:
            n_bf = run_backfill_pass(store)
            print(f"Backfill pass: {n_bf} row(s) updated with forward returns.")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
