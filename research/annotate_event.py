"""
research/annotate_event.py — manual labeling CLI for the Mean Reversion
observation layer (v2.9.x).

Two uses:
  1. Seed/label a human (off-bot) trade — e.g. the GLW/MU/MSFT manual buys
     that motivated this whole layer. If an auto-detected event already
     exists for that code within a few days of the given date, the manual
     fields are merged into it; otherwise a fresh retroactive snapshot is
     built for that historical date using the EXACT SAME
     build_observation_snapshot() the daily scan uses, so manually-seeded
     and auto-detected rows are never computed by two different code paths.
  2. Manually override category 8's crash_reason for any existing row to
     one of EARNINGS/GUIDANCE/FUNDAMENTAL_DAMAGE/NEWS — values the automatic
     heuristic in research/mean_reversion_metrics.py deliberately never
     produces (no news/earnings-calendar data source exists).

Usage:
    python research/annotate_event.py --code US.GLW --date 2026-07-28 \\
        --bought --price 118.34 --note "manual buy, human discretion"

    python research/annotate_event.py --code US.INTC --date 2026-05-01 \\
        --crash-reason GUIDANCE

Never touches positions.json, trade_history.db, or anything under
portfolio/risk/strategies — writes only to the research observation DB.
"""
import argparse
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

import backtest
import config
from data.fetcher import fetch_kline
from engine import market_context
from engine.hmm_regime import decode_regime_series_causal, load_stock_hmm, prepare_ohlcv
from research import fundamentals
from research.mean_reversion_metrics import build_observation_snapshot
from research.sector_etf import sector_etf_for
from research.store import ObservationStore

_CRASH_REASONS = ("MARKET_SELL_OFF", "SECTOR_SELL_OFF", "STOCK_SPECIFIC",
                   "EARNINGS", "GUIDANCE", "FUNDAMENTAL_DAMAGE", "NEWS", "UNKNOWN")


def _df_as_of(code: str, date_iso: str, bars: int = 400) -> Optional[pd.DataFrame]:
    """History up to and including `date_iso` (a possibly-past date) —
    data/fetcher.fetch_kline always fetches through "today", so this slices
    the returned frame down to rows <= date_iso rather than assuming the
    last row is the target date."""
    df = fetch_kline(code, "1d", bars)
    if df is None or len(df) == 0 or "time_key" not in df.columns:
        return None
    target = pd.Timestamp(date_iso).normalize()
    times = pd.to_datetime(df["time_key"]).dt.normalize()
    sliced = df[times <= target].reset_index(drop=True)
    return sliced if len(sliced) else None


def _historical_hmm_interface(code: str, date_iso: str) -> dict:
    """Regime AS OF a past date — deliberately NOT engine.regime_store (that
    store only ever holds "current" state and has no history; writing to it
    with a past as_of would also risk corrupting its duration/changed_today
    bookkeeping for the live daily job). Uses
    engine.hmm_regime.decode_regime_series_causal(), the function that
    module's own docstring designates for backtesting/analytics use — walk-
    forward, no-look-ahead, and it never touches regime_store.py at all.
    Empty dict (all-None downstream) if no trained model exists yet or
    there isn't enough history before date_iso."""
    payload = load_stock_hmm(code)
    if payload is None:
        return {}
    try:
        raw = backtest.fetch_kline(code, payload["train_start"], date_iso)
        if raw.empty:
            return {}
        work = prepare_ohlcv(raw)
        decoded = decode_regime_series_causal(payload, work)
        if decoded.empty:
            return {}
        last = decoded.iloc[-1]
        if pd.isna(last.get("regime")):
            return {}
        return {
            "current_regime": last["regime"],
            "regime_confidence": float(last["confidence"]),
            "regime_duration": int(last["duration"]) if pd.notna(last["duration"]) else None,
            "regime_changed_today": None,   # not meaningful for a single historical lookup
            "prev_regime": None,
            "transition_bear_to_sideways": None,
            "transition_bear_to_bull": None,
        }
    except Exception:
        return {}


def build_retroactive_snapshot(code: str, date_iso: str) -> Optional[dict]:
    df = _df_as_of(code, date_iso)
    if df is None:
        return None

    spy_df = _df_as_of("US.SPY", date_iso, bars=60)
    qqq_df = _df_as_of("US.QQQ", date_iso, bars=60)
    sector = config.SECTOR_MAP.get(code)
    etf = sector_etf_for(sector) if sector else None
    sector_df = _df_as_of(etf, date_iso, bars=60) if etf else None

    vix_close = None
    try:
        store = market_context.MarketContextStore()
        try:
            ctx = store.get_context_for_date(date_iso)
        finally:
            store.close()
        vix_close = ctx.get("vix_close") if ctx else None
        # market_context's own history only starts 2026-08-08 (see that
        # module's docstring) — a date before that has no stored VIX and
        # correctly stays None rather than substituting today's value.
    except Exception:
        pass

    hmm_iface = _historical_hmm_interface(code, date_iso)

    return build_observation_snapshot(
        code, df, spy_df=spy_df, qqq_df=qqq_df, sector_etf_df=sector_df,
        vix_close=vix_close, hmm_interface=hmm_iface,
        trigger_reason="MANUAL_ANNOTATION",
    )


def annotate(code: str, date_iso: str, store: ObservationStore, *,
             bought: bool = False, price: Optional[float] = None,
             note: Optional[str] = None, crash_reason: Optional[str] = None) -> str:
    """Returns a human-readable outcome string. Never raises to the caller
    (argparse-driven CLI — errors are printed and turned into a non-zero
    exit code by main())."""
    existing = store.find_near(code, date_iso, within_days=3)

    if existing is not None:
        event_id, event_date = existing["event_id"], existing["event_date"]
        record = {"event_id": event_id, "code": code, "event_date": event_date}
        outcome = f"updated existing event {event_id}"
    else:
        snapshot = build_retroactive_snapshot(code, date_iso)
        if snapshot is None:
            raise RuntimeError(
                f"no price history available for {code} as of {date_iso} — "
                f"cannot build a retroactive snapshot")
        fnd = fundamentals.fetch_fundamentals(code)
        event_id = f"{code}_{date_iso}"
        record = {"event_id": event_id, "event_date": date_iso,
                   "detected_at": datetime.now().isoformat(), **snapshot, **fnd}
        outcome = f"created new retroactive event {event_id}"

    if bought:
        record["was_manually_bought"] = 1
        record["manual_entry_price"] = price
        record["manual_entry_date"] = date_iso
        if note:
            record["manual_note"] = note
    if crash_reason:
        record["crash_reason"] = crash_reason
        record["crash_reason_source"] = "MANUAL"

    store.upsert(record)
    return outcome


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Manually label a Mean Reversion observation event — "
                     "observation only, never affects live trading.")
    parser.add_argument("--code", required=True, help="e.g. US.GLW")
    parser.add_argument("--date", required=True,
                         help="Event/trade date, ISO format e.g. 2026-07-28")
    parser.add_argument("--bought", action="store_true",
                         help="Mark this as a human (off-bot) buy.")
    parser.add_argument("--price", type=float, default=None,
                         help="Manual entry price (with --bought).")
    parser.add_argument("--note", default=None, help="Free-text note.")
    parser.add_argument("--crash-reason", choices=_CRASH_REASONS, default=None,
                         help="Manually override category 8's crash reason.")
    args = parser.parse_args(argv)

    if args.bought and args.price is None:
        parser.error("--bought requires --price")

    store = ObservationStore()
    try:
        outcome = annotate(
            args.code, args.date, store,
            bought=args.bought, price=args.price, note=args.note,
            crash_reason=args.crash_reason,
        )
        print(f"[ok] {outcome}")

        # Historical dates usually already have forward bars available —
        # fill in category 9 immediately rather than waiting for tomorrow's
        # scheduled scan.
        from research.scan_mean_reversion_candidates import run_backfill_pass
        n_bf = run_backfill_pass(store)
        print(f"[ok] backfill: {n_bf} row(s) updated with forward returns")
        return 0
    except Exception as e:
        print(f"[error] {e}")
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
