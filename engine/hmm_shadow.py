# -*- coding: utf-8 -*-
"""
engine/hmm_shadow.py — V2.7 Stage 3, Step 3: Shadow Mode.

Runs the trained per-stock HMM regime classifiers (engine/hmm_regime.py)
once per day, updates engine/regime_store.py's durable state, and appends
one row per stock to a plain CSV log (date, code, regime, confidence,
changed_today, duration_days) — exactly the fields the task spec asks Shadow
Mode to record.

Deliberately a standalone script, NOT called from engine/runner.py's
run_once()/run_loop() or main.py's live trading loop. This is the actual
safety mechanism behind "v2.6 and v2.7 buy/sell decisions must stay
byte-identical while Shadow Mode runs" — this script doesn't import
portfolio/, risk/, strategies/, or engine/runner.py at all, so there is no
code path by which running it can affect a trading decision. It's meant to
be run manually or via a separate daily cron/scheduled task, independent of
the intraday `python main.py --interval 300` trading loop.

Data source: like engine/hmm_regime.py's training path, this uses
backtest.fetch_kline() (data_cache/ cache), not data/fetcher.py — so Shadow
Mode's daily data pull is also fully decoupled from the live intraday
runner's data path.
"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from data_provider.provider_factory import get_provider
from engine.hmm_regime import MLRegimeClassifier, load_stock_hmm, HMM_VERSION, prepare_ohlcv
from engine.regime_store import update_regime

_provider = get_provider(config.MARKET)

SHADOW_LOG_PATH = Path(__file__).parent / "regime_shadow_log.csv"
SHADOW_LOG_COLUMNS = ["date", "code", "regime", "confidence", "changed_today", "duration_days"]


def run_shadow_pass(codes: list, as_of: str = None) -> pd.DataFrame:
    """
    One Shadow Mode pass over `codes`: for each, load its trained HMM model,
    fetch data up to `as_of` (today by default), decode the latest bar's
    regime, update the durable regime_store, and append a row to the CSV log.

    Every per-stock step is independently try/except-guarded — one stock's
    missing model / stale data / decode failure never aborts the batch or
    raises out of this function, matching engine/trade_tracker.py's
    "never break the caller" discipline. Failures are printed, not silenced,
    so a bad pass is still visible in the console/cron log.
    """
    as_of = as_of or datetime.now().strftime("%Y-%m-%d")
    rows = []

    for code in codes:
        try:
            payload = load_stock_hmm(code)
            if payload is None:
                print(f"[skip] {code}: no trained HMM model yet (run engine/hmm_regime.py --codes {code} first)")
                continue

            raw = _provider.get_history(code, start=payload["train_start"], end=as_of)
            if raw.empty:
                print(f"[skip] {code}: no data returned for [{payload['train_start']}, {as_of}]")
                continue

            work = prepare_ohlcv(raw)
            classifier = MLRegimeClassifier(code, model_payload=payload)
            result = classifier.classify_series(work)

            if result.empty:
                print(f"[skip] {code}: empty classification result")
                continue

            last = result.iloc[-1]
            if last["regime_name"] in (None, "Unknown"):
                print(f"[skip] {code}: not enough fresh bars past feature warmup to decode a regime yet")
                continue

            entry = update_regime(
                code=code,
                as_of=as_of,
                regime_label=last["regime_name"],
                confidence=float(last["confidence"]),
                hmm_version=HMM_VERSION,
            )
            rows.append(
                {
                    "date": as_of,
                    "code": code,
                    "regime": entry["regime"],
                    "confidence": round(entry["confidence"], 1),
                    "changed_today": entry["changed_today"],
                    "duration_days": entry["duration_days"],
                }
            )
            print(
                f"[ok] {code}: {entry['regime']} (confidence={entry['confidence']:.1f}, "
                f"duration={entry['duration_days']}d, changed_today={entry['changed_today']})"
            )
        except Exception as e:
            print(f"[error] {code}: {e!r}")

    out_df = pd.DataFrame(rows, columns=SHADOW_LOG_COLUMNS)
    if not out_df.empty:
        # Idempotent append: skip (date, code) pairs already present, so
        # re-running the same day's pass (retry after a crash, manual re-run)
        # doesn't leave duplicate-looking rows in the audit log — matches
        # regime_store.update_regime()'s idempotency for the same reason.
        to_write = out_df
        if SHADOW_LOG_PATH.exists():
            existing = pd.read_csv(SHADOW_LOG_PATH, usecols=["date", "code"], dtype=str)
            already = set(zip(existing["date"], existing["code"]))
            to_write = out_df[~out_df.apply(lambda r: (r["date"], r["code"]) in already, axis=1)]

        if not to_write.empty:
            header_needed = not SHADOW_LOG_PATH.exists()
            to_write.to_csv(SHADOW_LOG_PATH, mode="a", header=header_needed, index=False)
    return out_df


def _cli_main():
    parser = argparse.ArgumentParser(description="V2.7 HMM Shadow Mode daily pass")
    parser.add_argument("--codes", nargs="+", required=True, help="e.g. US.AAPL US.QQQ")
    parser.add_argument("--as-of", type=str, default=None, help="ISO date, default: today")
    args = parser.parse_args()

    result = run_shadow_pass(args.codes, as_of=args.as_of)
    print(f"\nShadow Mode pass complete: {len(result)}/{len(args.codes)} stocks recorded.")
    print(f"Log: {SHADOW_LOG_PATH}")


if __name__ == "__main__":
    _cli_main()
