"""
analyze_entry_quality.py — offline analysis for Entry Quality Tracking
(v2.9.x, see engine/entry_quality.py and engine/trade_tracker.py's
trade_entry_quality table).

This script is READ-ONLY with respect to trading behavior: it never touches
positions.json, never places an order, and the only writes it performs are
backfilling trade_entry_quality.return_*/max_favorable_excursion/
max_adverse_excursion columns (pure analytics, see
TradeTracker.update_forward_returns()). It is not imported by
engine/runner.py or backtest.py — running it (or not) can never change a
BUY/SELL decision.

Two steps:
  1. Backfill — for entry-quality rows still missing forward returns (or not
     yet complete out to 60 trading days), fetch price history since entry
     and fill in whatever horizons have now elapsed. Skippable with
     --no-backfill (e.g. when OpenD isn't running, or for a quick re-look at
     already-backfilled data).
  2. Analyze — group trades by distance_from_breakout_atr into the buckets
     the product spec calls for (0~0.5 / 0.5~1.0 / 1.0~1.5 / 1.5~2.0 / >2.0
     ATR), and separately by data-driven quantiles of entry_day_return /
     atr_expansion_ratio / distance_from_20d_ema / distance_from_20d_high /
     entry_volume_ratio ("did this breakout look overheated"). No behavior
     changes on any threshold found here — this is purely descriptive
     output for a human to read.

Usage:
    python analyze_entry_quality.py                    # backfill + analyze
    python analyze_entry_quality.py --no-backfill       # analyze only
    python analyze_entry_quality.py --horizon 10        # rank buckets by return_10d
    python analyze_entry_quality.py --min-sample 5       # lower sample-size bar
"""
import argparse
import sys
from typing import Optional

import pandas as pd

from engine.entry_quality import compute_forward_returns
from engine.trade_tracker import TradeTracker

# Bucket edges exactly as specified by the product ask — a report grouping
# choice, not a trading threshold (nothing here feeds back into BUY/SELL).
_ATR_BUCKETS = [
    (0.0, 0.5, "0~0.5 ATR"),
    (0.5, 1.0, "0.5~1.0 ATR"),
    (1.0, 1.5, "1.0~1.5 ATR"),
    (1.5, 2.0, "1.5~2.0 ATR"),
    (2.0, float("inf"), ">2.0 ATR"),
]
_ATR_BUCKET_ORDER = ["< 0 ATR (filled below breakout)"] + [b[2] for b in _ATR_BUCKETS] + ["N/A"]

_OVERHEAT_DIMENSIONS = [
    ("entry_day_return", "Entry-day return"),
    ("atr_expansion_ratio", "ATR expansion (entry ATR / 60d avg ATR)"),
    ("distance_from_20d_ema", "Distance from 20d EMA"),
    ("distance_from_20d_high", "Distance from 20d high"),
    ("entry_volume_ratio", "Entry-day volume ratio"),
]


# ── Step 1: backfill forward returns ──────────────────────────────────────────

def _bars_after_entry(fetch_kline_fn, code: str, entry_time_iso: str,
                       ktype: str = "K_DAY", bars: int = 300) -> Optional[pd.DataFrame]:
    """Fetch recent kline history and slice to bars strictly AFTER
    entry_time_iso's calendar date. Returns None on any failure (no OpenD
    connection, bad code, etc.) — callers treat that as "can't backfill this
    trade right now", not an error."""
    try:
        df = fetch_kline_fn(code, ktype=ktype, bars=bars)
        if df is None or len(df) == 0 or "time_key" not in df.columns:
            return None
        entry_date = pd.Timestamp(entry_time_iso).normalize()
        times = pd.to_datetime(df["time_key"])
        after = df[times.dt.normalize() > entry_date].reset_index(drop=True)
        return after if len(after) else None
    except Exception:
        return None


def backfill_forward_returns(tracker: TradeTracker, fetch_kline_fn=None,
                              force: bool = False) -> int:
    """Fill in return_5d/10d/20d/60d + MFE/MAE for every entry-quality row
    that isn't complete yet (forward_bars_available < 60), using whatever
    price history is available today. Safe to call repeatedly — a trade
    that's only 8 calendar days old today will get return_5d filled now and
    return_10d/20d/60d filled automatically on a later run once more bars
    exist, since forward_bars_available < 60 keeps it eligible until then.

    force=True re-fetches and recomputes even rows already at
    forward_bars_available >= 60 (e.g. after a data correction).

    Returns the number of rows updated. Never raises — a failure on one
    trade (missing fetch_kline_fn, network error, bad code) is skipped, not
    fatal to the batch."""
    if fetch_kline_fn is None:
        from data.fetcher import fetch_kline as fetch_kline_fn

    df = tracker.query_entry_quality()
    if df.empty:
        return 0

    pending = df if force else df[df["forward_bars_available"].fillna(0) < 60]
    updated = 0
    for _, row in pending.iterrows():
        bars_df = _bars_after_entry(fetch_kline_fn, row["symbol"], row["entry_time"])
        if bars_df is None:
            continue
        result = compute_forward_returns(bars_df, entry_price=row["entry_price"])
        if result is None:
            continue
        try:
            tracker.update_forward_returns(trade_id=row["trade_id"], **result)
            updated += 1
        except Exception:
            continue
    return updated


# ── Step 2: bucketed analysis ─────────────────────────────────────────────────

def _atr_bucket_label(x) -> str:
    if pd.isna(x):
        return "N/A"
    if x < 0:
        return "< 0 ATR (filled below breakout)"
    for lo, hi, label in _ATR_BUCKETS:
        if lo <= x < hi:
            return label
    return ">2.0 ATR"


def _bucket_stats(df: pd.DataFrame, bucket_col: str, horizon_col: str,
                   min_sample: int) -> pd.DataFrame:
    """One row per bucket value present in df[bucket_col]. sample_count is
    the number of trades with a non-null horizon_col value (forward return
    not backfilled yet -> excluded from win-rate/return stats but still
    counted in trade_count) — flagged INSUFFICIENT_SAMPLE below min_sample
    so a two-trade bucket is never mistaken for a validated pattern."""
    rows = []
    for label, g in df.groupby(bucket_col, sort=False):
        outcomes = g[horizon_col].dropna()
        n_total = len(g)
        n_sample = len(outcomes)
        wins = int((outcomes > 0).sum())
        gains = outcomes[outcomes > 0].sum()
        losses = -outcomes[outcomes < 0].sum()
        if losses > 0:
            profit_factor = gains / losses
        elif gains > 0:
            profit_factor = float("inf")
        else:
            profit_factor = None
        rows.append({
            "bucket": label,
            "trade_count": n_total,
            "sample_count": n_sample,
            "win_rate": (wins / n_sample) if n_sample else None,
            "avg_return": outcomes.mean() if n_sample else None,
            "median_return": outcomes.median() if n_sample else None,
            "return_5d_avg": g["return_5d"].mean(),
            "return_10d_avg": g["return_10d"].mean(),
            "return_20d_avg": g["return_20d"].mean(),
            "return_60d_avg": g["return_60d"].mean(),
            "profit_factor": profit_factor,
            "mfe_avg": g["max_favorable_excursion"].mean(),
            "mae_avg": g["max_adverse_excursion"].mean(),
            "flag": "INSUFFICIENT_SAMPLE" if n_sample < min_sample else "",
        })
    return pd.DataFrame(rows)


def analyze_by_atr_distance(df: pd.DataFrame, horizon_col: str,
                             min_sample: int) -> pd.DataFrame:
    sub = df.copy()
    sub["bucket"] = sub["distance_from_breakout_atr"].apply(_atr_bucket_label)
    stats = _bucket_stats(sub, "bucket", horizon_col, min_sample)
    order = {label: i for i, label in enumerate(_ATR_BUCKET_ORDER)}
    stats["_order"] = stats["bucket"].map(order).fillna(len(order))
    return stats.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def analyze_overheat_dimension(df: pd.DataFrame, column: str, horizon_col: str,
                                min_sample: int, quantiles: int = 4) -> Optional[pd.DataFrame]:
    """Quantile-bucket one dimension (no hand-picked thresholds — bucket
    edges come from the data's own distribution via pd.qcut). Returns None
    when there isn't even enough non-null data to form `quantiles` distinct
    buckets — the caller should report that as its own INSUFFICIENT_SAMPLE
    case rather than printing a misleading 1-bucket table."""
    sub = df.dropna(subset=[column]).copy()
    if len(sub) < quantiles:
        return None
    try:
        sub["bucket"] = pd.qcut(sub[column], q=quantiles, duplicates="drop")
    except ValueError:
        return None
    sub["bucket"] = sub["bucket"].astype(str)
    stats = _bucket_stats(sub, "bucket", horizon_col, min_sample)
    return stats.sort_values("bucket").reset_index(drop=True)


def analyze_composite_overheat(df: pd.DataFrame, horizon_col: str,
                                min_sample: int, quantiles: int = 4) -> Optional[pd.DataFrame]:
    """"突破 + ATR扩大 + 短期涨幅过大" combined view: rank each trade within
    each available overheat dimension (entry_day_return, atr_expansion_ratio,
    distance_from_20d_high — the three most directly about "how hot was this
    breakout"), average the ranks a trade has data for, then quantile-bucket
    that composite. Purely a descriptive composite — no fixed thresholds,
    still needs >= quantiles trades with at least one dimension populated."""
    cols = ["entry_day_return", "atr_expansion_ratio", "distance_from_20d_high"]
    sub = df.copy()
    rank_cols = []
    for col in cols:
        if col not in sub.columns or sub[col].notna().sum() < quantiles:
            continue
        rank_col = f"_rank_{col}"
        sub[rank_col] = sub[col].rank(pct=True)
        rank_cols.append(rank_col)
    if not rank_cols:
        return None
    sub["composite_overheat_rank"] = sub[rank_cols].mean(axis=1)
    sub = sub.dropna(subset=["composite_overheat_rank"])
    if len(sub) < quantiles:
        return None
    try:
        sub["bucket"] = pd.qcut(sub["composite_overheat_rank"], q=quantiles, duplicates="drop")
    except ValueError:
        return None
    sub["bucket"] = sub["bucket"].astype(str)
    stats = _bucket_stats(sub, "bucket", horizon_col, min_sample)
    return stats.sort_values("bucket").reset_index(drop=True)


def _print_table(title: str, stats: Optional[pd.DataFrame]) -> None:
    print(f"\n== {title} ==")
    if stats is None or stats.empty:
        print("  (insufficient data)")
        return
    with pd.option_context("display.width", 200, "display.max_columns", None,
                            "display.float_format", lambda v: f"{v:.4f}"):
        print(stats.to_string(index=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Entry Quality Tracking — offline bucketed analysis "
                     "(observation only, never affects live trading).")
    parser.add_argument("--db-path", default=None,
                         help="Override the trade_history.db path (default: "
                              "TradeTracker's own default).")
    parser.add_argument("--no-backfill", action="store_true",
                         help="Skip the forward-return backfill step (e.g. "
                              "when OpenD isn't reachable).")
    parser.add_argument("--force-backfill", action="store_true",
                         help="Recompute forward returns even for rows "
                              "already at 60 bars of history.")
    parser.add_argument("--horizon", type=int, default=20, choices=(5, 10, 20, 60),
                         help="Return horizon (days) used for win rate / "
                              "avg / median / profit factor (default: 20).")
    parser.add_argument("--min-sample", type=int, default=10,
                         help="Buckets with fewer trades than this are "
                              "flagged INSUFFICIENT_SAMPLE (default: 10).")
    parser.add_argument("--quantiles", type=int, default=4,
                         help="Number of quantile buckets for the overheat "
                              "dimensions (default: 4 = quartiles).")
    args = parser.parse_args(argv)

    tracker = TradeTracker(args.db_path)
    try:
        if not args.no_backfill:
            n = backfill_forward_returns(tracker, force=args.force_backfill)
            print(f"Backfilled forward returns for {n} trade(s).")

        df = tracker.query_entry_quality()
        print(f"\n{len(df)} entry-quality row(s) total.")
        if df.empty:
            print("Nothing to analyze yet — no BUY has been logged with "
                  "Entry Quality Tracking active.")
            return 0

        horizon_col = f"return_{args.horizon}d"

        _print_table(
            f"By distance from Donchian breakout (ATR units) — outcome = {horizon_col}",
            analyze_by_atr_distance(df, horizon_col, args.min_sample),
        )

        for column, label in _OVERHEAT_DIMENSIONS:
            _print_table(
                f"Overheat check — {label} (quantiles) — outcome = {horizon_col}",
                analyze_overheat_dimension(df, column, horizon_col,
                                           args.min_sample, args.quantiles),
            )

        _print_table(
            f"Overheat check — composite rank (entry-day return + ATR "
            f"expansion + distance from 20d high) — outcome = {horizon_col}",
            analyze_composite_overheat(df, horizon_col, args.min_sample, args.quantiles),
        )
        return 0
    finally:
        tracker.close()


if __name__ == "__main__":
    sys.exit(main())
