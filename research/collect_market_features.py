"""
research/collect_market_features.py — daily standalone Research Market
Features collector (v2.11 Priority 3; v2.11.1 time model + health logging).

Deliberately a standalone script, same isolation discipline as
research/scan_mean_reversion_candidates.py: does NOT import portfolio/,
risk/, strategies/, or engine/runner.py, so there is no code path by which
running it can affect a trading decision. Meant to run once per day via a
separate Scheduled Task ("Kabu Market Features Scan"), independent of both
main.py's live loop and the existing "Kabu Research Scan" task.

Four independent, individually-skippable passes:
  1. News       — engine.news.research_snapshot() per watchlist code.
  2. Options    — data.options per watchlist code (IV/HV/put-call-ratio) +
                  one market-wide Put/Call ratio pull.
  3. Macro      — data.macro indicator list (+ history for any IDs listed in
                  config.RESEARCH_MACRO_INDICATOR_IDS).
  4. FedWatch   — data.macro target-rate + dot-plot snapshots.

Every row carries `observed_at` (UTC, common.utc_now_iso()) and
`market_date` (engine.market_hours.market_date_for("US")) — see
research/market_features_store.py's module docstring for why these, not the
legacy `snapshot_date`, are the fields future Look-ahead-safe research
queries must use.

v2.11.1 Priority 2: every collection attempt (per code / per indicator / per
FedWatch sub-call) is also logged to `collection_log` via
store.log_collection() — SUCCESS/NO_DATA/FAILED/SKIPPED — so
research/market_features_health.py can tell "ran and found nothing" apart
from "silently failed" apart from "never ran". This module only records
outcomes; it never raises out of a failed item (one bad code must not abort
the rest of the pass) and never touches anything that could affect a
trading decision.
"""
import argparse
from datetime import datetime
from typing import List

import common
import config
import engine.market_hours as market_hours
from data import macro as macro_data
from data import options as options_data
from engine import news as news_module
from research.market_features_store import MarketFeaturesStore


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _now() -> tuple:
    """(observed_at UTC ISO, market_date ISO, legacy snapshot_date) as of
    right now — called fresh per item so a long pass's later items get their
    own accurate observed_at rather than all sharing the pass's start time."""
    observed_at = common.utc_now_iso()
    market_date = market_hours.market_date_for("US").isoformat()
    return observed_at, market_date


def run_news_pass(codes: List[str], store: MarketFeaturesStore) -> int:
    snapshot_date = _today()
    n = 0
    for code in codes:
        observed_at, market_date = _now()
        try:
            snap = news_module.research_snapshot(code)
            store.upsert("news_snapshots", {
                "code": code, "snapshot_date": snapshot_date,
                "news_count_24h": snap["news_count_24h"],
                "major_news_flag": int(snap["major_news_flag"]),
                "announcement_flag": int(snap["announcement_flag"]),
                "created_at": datetime.now().isoformat(),
                "observed_at": observed_at, "market_date": market_date,
            })
            store.log_collection(market_date, "news", code, "SUCCESS",
                                  observed_at=observed_at)
            n += 1
        except Exception as e:
            print(f"[error] news snapshot {code}: {e!r}")
            store.log_collection(market_date, "news", code, "FAILED",
                                  detail=repr(e), observed_at=observed_at)
    return n


def run_options_pass(codes: List[str], store: MarketFeaturesStore) -> int:
    snapshot_date = _today()
    n = 0
    observed_at, market_date = _now()
    try:
        rows = options_data.get_underlying_overview(codes)
        by_code = {row.get("code"): row for row in rows}
        fetch_failed = False
    except Exception as e:
        print(f"[error] get_underlying_overview: {e!r}")
        by_code = {}
        fetch_failed = True

    for code in codes:
        item_observed_at, item_market_date = _now()
        if fetch_failed:
            store.log_collection(item_market_date, "options", code, "FAILED",
                                  detail="get_underlying_overview batch call failed",
                                  observed_at=item_observed_at)
            continue
        row = by_code.get(code)
        if row is None:
            store.log_collection(item_market_date, "options", code, "NO_DATA",
                                  observed_at=item_observed_at)
            continue
        try:
            call_vol = row.get("call_volume") or 0
            put_vol = row.get("put_volume") or 0
            pcr = (put_vol / call_vol) if call_vol else None
            store.upsert("options_snapshots", {
                "code": code, "snapshot_date": snapshot_date,
                "iv": row.get("iv"), "iv_rank": row.get("iv_rank"),
                "iv_percentile": row.get("iv_percentile"),
                "hv_30d": row.get("hv_30d"),
                "call_volume": call_vol, "put_volume": put_vol,
                "put_call_ratio": pcr, "created_at": datetime.now().isoformat(),
                "observed_at": item_observed_at, "market_date": item_market_date,
            })
            store.log_collection(item_market_date, "options", code, "SUCCESS",
                                  observed_at=item_observed_at)
            n += 1
        except Exception as e:
            print(f"[error] options snapshot {code}: {e!r}")
            store.log_collection(item_market_date, "options", code, "FAILED",
                                  detail=repr(e), observed_at=item_observed_at)

    pcr_observed_at, pcr_market_date = _now()
    try:
        pcr_rows = options_data.get_market_put_call_ratio(config.RESEARCH_OPTIONS_MARKET)
        for row in pcr_rows:
            store.upsert("market_pcr_snapshots", {
                "market": config.RESEARCH_OPTIONS_MARKET, "time": row.get("time"),
                "timestamp": row.get("timestamp"), "call_value": row.get("call_value"),
                "put_value": row.get("put_value"), "total_value": row.get("total_value"),
                "ratio": row.get("ratio"), "snapshot_date": snapshot_date,
                "created_at": datetime.now().isoformat(),
                "observed_at": pcr_observed_at, "market_date": pcr_market_date,
            })
        status = "SUCCESS" if pcr_rows else "NO_DATA"
        store.log_collection(pcr_market_date, "options", "MARKET_PCR", status,
                              observed_at=pcr_observed_at)
    except Exception as e:
        print(f"[error] market put/call ratio: {e!r}")
        store.log_collection(pcr_market_date, "options", "MARKET_PCR", "FAILED",
                              detail=repr(e), observed_at=pcr_observed_at)
    return n


def run_macro_pass(store: MarketFeaturesStore) -> int:
    snapshot_date = _today()
    n = 0

    if not config.RESEARCH_MACRO_INDICATOR_IDS:
        observed_at, market_date = _now()
        store.log_collection(market_date, "macro", "ALL", "SKIPPED",
                              detail="RESEARCH_MACRO_INDICATOR_IDS is empty",
                              observed_at=observed_at)
        return 0

    list_observed_at, list_market_date = _now()
    try:
        indicators = macro_data.get_macro_indicator_list(config.RESEARCH_MACRO_REGION)
        store.log_collection(list_market_date, "macro", "INDICATOR_LIST", "SUCCESS",
                              observed_at=list_observed_at)
    except Exception as e:
        print(f"[error] get_macro_indicator_list: {e!r}")
        store.log_collection(list_market_date, "macro", "INDICATOR_LIST", "FAILED",
                              detail=repr(e), observed_at=list_observed_at)
        indicators = []

    by_id = {row.get("indicator_id"): row for row in indicators}
    for indicator_id in config.RESEARCH_MACRO_INDICATOR_IDS:
        meta = by_id.get(indicator_id, {})
        item_observed_at, item_market_date = _now()
        try:
            history = macro_data.get_macro_indicator_history(indicator_id, max_count=1)
        except Exception as e:
            print(f"[error] get_macro_indicator_history {indicator_id}: {e!r}")
            store.log_collection(item_market_date, "macro", str(indicator_id), "FAILED",
                                  detail=repr(e), observed_at=item_observed_at)
            continue

        if not history:
            store.log_collection(item_market_date, "macro", str(indicator_id), "NO_DATA",
                                  observed_at=item_observed_at)
            continue

        item_ok = False
        for row in history:
            try:
                store.upsert("macro_snapshots", {
                    "region": config.RESEARCH_MACRO_REGION, "indicator_id": indicator_id,
                    "category_name": meta.get("category_name"), "name": meta.get("name"),
                    "data_time": row.get("data_time"), "release_time": row.get("release_time"),
                    "value": row.get("value"), "predict_value": row.get("predict_value"),
                    "previous_value": row.get("previous_value"), "unit_type": row.get("unit_type"),
                    "snapshot_date": snapshot_date, "created_at": datetime.now().isoformat(),
                    "observed_at": item_observed_at, "market_date": item_market_date,
                })
                n += 1
                item_ok = True
            except Exception as e:
                print(f"[error] macro snapshot {indicator_id}: {e!r}")
        store.log_collection(item_market_date, "macro", str(indicator_id),
                              "SUCCESS" if item_ok else "FAILED",
                              observed_at=item_observed_at)
    return n


def run_fedwatch_pass(store: MarketFeaturesStore) -> int:
    snapshot_date = _today()
    n = 0

    tr_observed_at, tr_market_date = _now()
    try:
        rows = macro_data.get_fed_watch_target_rate()
        for row in rows:
            store.upsert("fedwatch_target_rate_snapshots", {
                "snapshot_date": snapshot_date, "meeting_date": row.get("meeting_date"),
                "target_range": row.get("target_range"), "probability": row.get("probability"),
                "created_at": datetime.now().isoformat(),
                "observed_at": tr_observed_at, "market_date": tr_market_date,
            })
            n += 1
        store.log_collection(tr_market_date, "fedwatch", "TARGET_RATE",
                              "SUCCESS" if rows else "NO_DATA", observed_at=tr_observed_at)
    except Exception as e:
        print(f"[error] get_fed_watch_target_rate: {e!r}")
        store.log_collection(tr_market_date, "fedwatch", "TARGET_RATE", "FAILED",
                              detail=repr(e), observed_at=tr_observed_at)

    dp_observed_at, dp_market_date = _now()
    try:
        rows = macro_data.get_fed_watch_dot_plot()
        for row in rows:
            store.upsert("fedwatch_dot_plot_snapshots", {
                "snapshot_date": snapshot_date, "year": row.get("year"),
                "rate": row.get("rate"), "vote_count": row.get("vote_count"),
                "is_median": int(bool(row.get("is_median"))),
                "median_rate": row.get("median_rate"), "current_rate": row.get("current_rate"),
                "created_at": datetime.now().isoformat(),
                "observed_at": dp_observed_at, "market_date": dp_market_date,
            })
        store.log_collection(dp_market_date, "fedwatch", "DOT_PLOT",
                              "SUCCESS" if rows else "NO_DATA", observed_at=dp_observed_at)
    except Exception as e:
        print(f"[error] get_fed_watch_dot_plot: {e!r}")
        store.log_collection(dp_market_date, "fedwatch", "DOT_PLOT", "FAILED",
                              detail=repr(e), observed_at=dp_observed_at)
    return n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Research Market Features collector — observation only, "
                     "never affects live trading. See module docstring.")
    parser.add_argument("--codes", nargs="+", default=None,
                         help="Override the code list (default: config.WATCHLIST).")
    parser.add_argument("--no-news", action="store_true")
    parser.add_argument("--no-options", action="store_true")
    parser.add_argument("--no-macro", action="store_true")
    parser.add_argument("--no-fedwatch", action="store_true")
    args = parser.parse_args(argv)

    if not config.RESEARCH_MARKET_FEATURES_ENABLED:
        print("RESEARCH_MARKET_FEATURES_ENABLED=False — nothing to do.")
        store = MarketFeaturesStore()
        try:
            observed_at, market_date = _now()
            for category in ("news", "options", "macro", "fedwatch"):
                store.log_collection(market_date, category, "ALL", "SKIPPED",
                                      detail="RESEARCH_MARKET_FEATURES_ENABLED=False",
                                      observed_at=observed_at)
        finally:
            store.close()
        return 0

    codes = args.codes or config.WATCHLIST
    store = MarketFeaturesStore()
    try:
        if not args.no_news:
            n = run_news_pass(codes, store)
            print(f"News pass: {n} snapshot(s) recorded.")
        else:
            observed_at, market_date = _now()
            store.log_collection(market_date, "news", "ALL", "SKIPPED",
                                  detail="--no-news", observed_at=observed_at)
        if not args.no_options:
            n = run_options_pass(codes, store)
            print(f"Options pass: {n} snapshot(s) recorded.")
        else:
            observed_at, market_date = _now()
            store.log_collection(market_date, "options", "ALL", "SKIPPED",
                                  detail="--no-options", observed_at=observed_at)
        if not args.no_macro:
            n = run_macro_pass(store)
            print(f"Macro pass: {n} snapshot(s) recorded.")
        else:
            observed_at, market_date = _now()
            store.log_collection(market_date, "macro", "ALL", "SKIPPED",
                                  detail="--no-macro", observed_at=observed_at)
        if not args.no_fedwatch:
            n = run_fedwatch_pass(store)
            print(f"FedWatch pass: {n} snapshot(s) recorded.")
        else:
            observed_at, market_date = _now()
            store.log_collection(market_date, "fedwatch", "ALL", "SKIPPED",
                                  detail="--no-fedwatch", observed_at=observed_at)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    import sys
    sys.exit(main())
