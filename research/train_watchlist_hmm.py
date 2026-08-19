"""
research/train_watchlist_hmm.py — one-time (re-runnable) HMM training driver
for the Mean Reversion observation layer's Category 5 (HMM Regime) fields.

Trains engine/hmm_regime.py's 4-state GaussianHMM for every code in
config.WATCHLIST and persists to engine/hmm_models/ (same mechanism
engine/hmm_regime.py's own CLI uses). Purely a one-time infrastructure step
— training itself is already documented as observation-only (see that
module's docstring: "NOT wired into engine/runner.py or
backtest_portfolio.py"). Re-running is safe/idempotent (re-trains and
overwrites); useful after the watchlist changes.

Does not touch any trading state. Safe to run while the live bot
(main.py --interval 300) is running — different process, different files
(engine/hmm_models/*.pkl is only ever read by engine/hmm_shadow.py, which is
itself never called from engine/runner.py).
"""
import sys

import config
from engine.hmm_regime import train_stock_hmm


def train_watchlist(codes: list) -> tuple:
    """Returns (trained_codes, failed_codes_with_reason)."""
    trained, failed = [], []
    for code in codes:
        try:
            payload = train_stock_hmm(code)
            trained.append(code)
            print(f"[ok] {code}: trained on {payload['n_samples']} bars "
                  f"({payload['train_start']} .. {payload['train_end']})")
        except Exception as e:
            failed.append((code, str(e)))
            print(f"[skip] {code}: {e}")
    return trained, failed


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Train HMM regime models for config.WATCHLIST (or "
                     "--codes) — observation-only infrastructure step.")
    parser.add_argument("--codes", nargs="+", default=None,
                         help="Override the code list (default: config.WATCHLIST).")
    args = parser.parse_args(argv)

    codes = args.codes or config.WATCHLIST
    trained, failed = train_watchlist(codes)
    print(f"\n{len(trained)}/{len(codes)} trained, {len(failed)} skipped "
          f"(insufficient history or degenerate fit).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
