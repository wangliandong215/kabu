"""
research/analyze_mean_reversion.py — offline analysis skeleton for the Mean
Reversion observation layer (v2.9.x).

Deliberately thin: the user wants 30-50+ labeled cases accumulated (see
research/scan_mean_reversion_candidates.py + research/annotate_event.py)
before any real statistical analysis is worth doing. This script just loads
everything recorded so far into a DataFrame and prints basic counts —
enough to track accumulation progress and sanity-check the data, not a
model. Read-only with respect to trading behavior (and to the DB itself,
except via research/scan_mean_reversion_candidates.py's own backfill,
optionally triggered here with --backfill).
"""
import argparse
import sys

import pandas as pd

from research.store import ObservationStore


def load_dataframe(store: ObservationStore) -> pd.DataFrame:
    rows = store.all_rows()
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{len(df)} event(s) recorded total.")
    if df.empty:
        print("Nothing recorded yet — run research/scan_mean_reversion_candidates.py "
              "and/or research/annotate_event.py first.")
        return

    print(f"\nBy code:\n{df['code'].value_counts().to_string()}")

    print("\nBy outcome_label (category 9, PENDING = not enough forward bars yet):")
    print(df["outcome_label"].fillna("PENDING").value_counts().to_string())

    bought = df["was_manually_bought"].fillna(0).astype(bool)
    outcome = df["outcome_label"].fillna("PENDING")
    print("\nCase groups (per spec: A=bought&SUCCESS, B=not bought, "
          "C=FAILURE, D=SUCCESS — a row can belong to more than one):")
    print(f"  A (bought & SUCCESS):  {int((bought & (outcome == 'SUCCESS')).sum())}")
    print(f"  B (not bought):        {int((~bought).sum())}")
    print(f"  C (FAILURE):           {int((outcome == 'FAILURE').sum())}")
    print(f"  D (SUCCESS):           {int((outcome == 'SUCCESS').sum())}")

    print("\nBy crash_reason:")
    print(df["crash_reason"].fillna("N/A").value_counts().to_string())

    print("\nBy HMM regime at event time:")
    print(df["hmm_regime"].fillna("NO_MODEL").value_counts().to_string())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Mean Reversion research — accumulation-progress summary "
                     "(observation only, never affects live trading).")
    parser.add_argument("--backfill", action="store_true",
                         help="Run the forward-return backfill pass first "
                              "(research.scan_mean_reversion_candidates.run_backfill_pass).")
    parser.add_argument("--csv", default=None,
                         help="Also export the full dataset to this CSV path.")
    args = parser.parse_args(argv)

    store = ObservationStore()
    try:
        if args.backfill:
            from research.scan_mean_reversion_candidates import run_backfill_pass
            n = run_backfill_pass(store)
            print(f"Backfilled {n} row(s).")

        df = load_dataframe(store)
        print_summary(df)

        if args.csv and not df.empty:
            df.to_csv(args.csv, index=False)
            print(f"\nExported {len(df)} row(s) to {args.csv}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
