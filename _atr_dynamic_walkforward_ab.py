"""
_atr_dynamic_walkforward_ab.py -- nested IS/OOS walk-forward A/B test for the
dynamic (inverse-volatility) ATR trailing-stop multiplier vs two fixed-ATR
benchmarks. This is the out-of-sample check for the idea motivated by
_atr_walkforward_sweep.py (4-window best-ATR sweep, range 3.25~5.75) and
_atr_regime_correlation.py (best-ATR correlated moderately, not strongly,
with volatility measures; ADX showed ~no relationship).

Design (as specified):
  Benchmark A : fixed ATR_TRAIL_MULT = 3.5  (tight/defensive)
  Benchmark B : fixed ATR_TRAIL_MULT = 5.5  (wide/trend-following)
  Treatment   : dynamic ATR, engine.regime.dynamic_atr_multiplier_series()
                driven by QQQ 20-day realized volatility, base/alpha CHOSEN
                PER FOLD by an in-sample grid search (not fixed at 4.5/0.5 --
                that would just be re-testing the same single hyperparameter
                pick already explored above; a real walk-forward re-optimizes
                every fold so the OOS result reflects "run this process going
                forward", not "get lucky with one fixed setting").

  IS  window : 12 months, grid-search (base, alpha) in
               {3.5,4.0,4.5,5.0,5.5} x {0.0,0.25,0.5,0.75,1.0} (25 combos,
               min_mult=3.0/max_mult=6.0 fixed, not swept), pick max Sharpe.
  OOS window : the following 3 months, run the IS-winning (base, alpha)
               untouched -- this is the only number that counts.
  Roll        : advance by 3 months (OOS length) and repeat until OOS end
               reaches the end of history (2026-07-03).

Data is fetched ONCE for the full 2015-01-01..2026-07-03 span (with QQQ's
extra pre-history for MA200/vol-mean warmup already handled inside
_prepare_backtest_data) and every fold/grid-combo simulates against slices
of that one fetch via simulate_from_prepared(sim_start=, sim_end=,
atr_mult_series=) -- see backtest_portfolio.py refactor. Benchmarks A/B use
atr_mult_series=None (bp.ATR_TRAIL_MULT set directly) since they need no
grid search.

Every fold resets cash/positions (fair, independent trial). The OOS daily
%-returns from consecutive folds are compounded together into one stitched
curve per variant -- THAT stitched curve, not any single fold, is what
Sharpe/Calmar/MaxDD are computed from at the end.

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _atr_dynamic_walkforward_ab.py
"""
import time

import numpy as np
import pandas as pd

import backtest_portfolio as bp
import engine.regime as regime_mod

FULL_START = "2015-01-01"
FULL_END   = "2026-07-03"
CASH       = 7_000_000.0
USD_JPY    = 140.0

IS_MONTHS  = 12
OOS_MONTHS = 3

GRID_BASE  = [3.5, 4.0, 4.5, 5.0, 5.5]
GRID_ALPHA = [0.0, 0.25, 0.5, 0.75, 1.0]
MIN_MULT, MAX_MULT = 3.0, 6.0

BENCH_A_ATR = 3.5
BENCH_B_ATR = 5.5

LOG_PATH = "_atr_dynamic_walkforward_ab.log"


def month_add(date_str: str, months: int) -> str:
    ts = pd.Timestamp(date_str) + pd.DateOffset(months=months)
    return ts.strftime("%Y-%m-%d")


def build_folds():
    folds = []
    is_start = FULL_START
    while True:
        is_end   = month_add(is_start, IS_MONTHS)
        oos_start = is_end
        oos_end   = month_add(oos_start, OOS_MONTHS)
        if pd.Timestamp(oos_end) > pd.Timestamp(FULL_END):
            break
        folds.append({
            "is_start": is_start, "is_end": is_end,
            "oos_start": oos_start, "oos_end": oos_end,
        })
        is_start = month_add(is_start, OOS_MONTHS)
    return folds


def run_variant_oos(prepared, oos_start, oos_end, atr_mult_series=None, atr_const=None):
    if atr_const is not None:
        bp.ATR_TRAIL_MULT = atr_const
        r = bp.simulate_from_prepared(prepared, cash=CASH, currency="JPY ",
                                       sim_start=oos_start, sim_end=oos_end,
                                       atr_mult_series=None)
    else:
        r = bp.simulate_from_prepared(prepared, cash=CASH, currency="JPY ",
                                       sim_start=oos_start, sim_end=oos_end,
                                       atr_mult_series=atr_mult_series)
    return r


def daily_returns_from_result(r: dict) -> pd.Series:
    """% daily returns within one fold's simulation, first day measured
    against the fold's starting CASH (not against a carried-over equity)."""
    eq = r["equity_curve"]
    if len(eq) == 0:
        return pd.Series(dtype=float)
    prev = pd.concat([pd.Series([CASH]), eq.iloc[:-1]])
    prev.index = eq.index
    return (eq - prev) / prev


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")
    t0 = time.time()
    print(f"Fetching full history {FULL_START} ~ {FULL_END} once...")
    import contextlib
    with contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(bp.BACKTEST_STOCKS, FULL_START, FULL_END, usd_to_jpy=USD_JPY)
    if prepared is None:
        print("NO DATA -- aborting")
        return
    print(f"Fetch done in {time.time()-t0:.1f}s")

    qqq_close = prepared["benchmark_df"]["close"].astype(float)
    qqq_vol20 = prepared["qqq_vol20"]

    folds = build_folds()
    print(f"{len(folds)} IS/OOS folds scheduled "
          f"({IS_MONTHS}mo IS -> {OOS_MONTHS}mo OOS, rolled by {OOS_MONTHS}mo)")

    fold_records = []
    t_folds_start = time.time()

    for i, f in enumerate(folds):
        t_fold0 = time.time()
        # ── IS grid search ──────────────────────────────────────────────────
        best = None
        with contextlib.redirect_stdout(log_f):
            print(f"\n{'#'*70}\n# FOLD {i}: IS {f['is_start']}~{f['is_end']}"
                  f"  OOS {f['oos_start']}~{f['oos_end']}\n{'#'*70}")
            for base in GRID_BASE:
                for alpha in GRID_ALPHA:
                    dyn = regime_mod.dynamic_atr_multiplier_series(
                        qqq_close, base_multiplier=base, alpha=alpha,
                        min_mult=MIN_MULT, max_mult=MAX_MULT, vol_window=20, hist_window=252)
                    r_is = bp.simulate_from_prepared(
                        prepared, cash=CASH, currency="JPY ",
                        sim_start=f["is_start"], sim_end=f["is_end"],
                        atr_mult_series=dyn)
                    if not r_is:
                        continue
                    sh = r_is["sharpe"]
                    if best is None or sh > best["sharpe"]:
                        best = {"base": base, "alpha": alpha, "sharpe": sh}

            if best is None:
                print(f"[FOLD {i}] IS grid produced no valid result, skipping fold")
                continue

            # ── OOS: Treatment (IS-winning params), Benchmark A, Benchmark B ──
            dyn_best = regime_mod.dynamic_atr_multiplier_series(
                qqq_close, base_multiplier=best["base"], alpha=best["alpha"],
                min_mult=MIN_MULT, max_mult=MAX_MULT, vol_window=20, hist_window=252)
            r_treat = run_variant_oos(prepared, f["oos_start"], f["oos_end"], atr_mult_series=dyn_best)
            r_a     = run_variant_oos(prepared, f["oos_start"], f["oos_end"], atr_const=BENCH_A_ATR)
            r_b     = run_variant_oos(prepared, f["oos_start"], f["oos_end"], atr_const=BENCH_B_ATR)

        fold_oos_vol = qqq_vol20.loc[(qqq_vol20.index >= pd.Timestamp(f["oos_start"]))
                                     & (qqq_vol20.index <= pd.Timestamp(f["oos_end"]))].mean()

        fold_records.append({
            "fold": i, **f,
            "is_best_base": best["base"], "is_best_alpha": best["alpha"], "is_best_sharpe": best["sharpe"],
            "oos_vol": fold_oos_vol,
            "treat": r_treat, "bench_a": r_a, "bench_b": r_b,
        })
        print(f"[FOLD {i}/{len(folds)-1}] IS best=(base={best['base']}, alpha={best['alpha']}, "
              f"Sharpe={best['sharpe']:.2f})  OOS treat={r_treat.get('total_return_pct')}%"
              f"  A={r_a.get('total_return_pct')}%  B={r_b.get('total_return_pct')}%"
              f"  ({time.time()-t_fold0:.1f}s)")

    log_f.close()
    print(f"\nALL {len(fold_records)} FOLDS DONE in {time.time()-t_folds_start:.1f}s "
          f"(total incl. fetch: {time.time()-t0:.1f}s)")

    # ── Stitch OOS daily returns across folds into one continuous curve per variant ──
    stitched = {"treat": [], "bench_a": [], "bench_b": []}
    per_fold_summary = []
    for rec in fold_records:
        for key in ("treat", "bench_a", "bench_b"):
            r = rec[key]
            if not r:
                continue
            rets = daily_returns_from_result(r)
            stitched[key].append(rets)
        per_fold_summary.append(rec)

    print(f"\n{'='*100}")
    print("PER-FOLD OOS SUMMARY")
    print(f"{'='*100}")
    print(f"{'fold':>4} {'OOS window':>23} {'IS(base,a)':>12} {'oos_vol%':>9}"
          f" {'Treat%':>8} {'A(3.5)%':>8} {'B(5.5)%':>8} "
          f"{'Treat_tr':>9} {'A_tr':>6} {'B_tr':>6}")
    for rec in per_fold_summary:
        t, a, b = rec["treat"], rec["bench_a"], rec["bench_b"]
        print(f"{rec['fold']:>4} {rec['oos_start']}~{rec['oos_end']:>10}"
              f" ({rec['is_best_base']:>4},{rec['is_best_alpha']:>4})"
              f" {rec['oos_vol']*100:>8.2f}"
              f" {t.get('total_return_pct', float('nan')):>8.2f}"
              f" {a.get('total_return_pct', float('nan')):>8.2f}"
              f" {b.get('total_return_pct', float('nan')):>8.2f}"
              f" {t.get('num_trades', 0):>9} {a.get('num_trades', 0):>6} {b.get('num_trades', 0):>6}")

    def stitch_and_metrics(rets_list):
        all_rets = pd.concat(rets_list).sort_index()
        equity = (1 + all_rets).cumprod()
        total_return = float(equity.iloc[-1] - 1) * 100
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        ann_return = (equity.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
        sharpe = (all_rets.mean() / all_rets.std()) * np.sqrt(252) if all_rets.std() > 0 else 0
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max
        max_dd = float(dd.min()) * 100
        calmar = (ann_return * 100) / abs(max_dd) if max_dd != 0 else float("nan")
        return {
            "total_return_pct": total_return, "sharpe": float(sharpe),
            "max_drawdown_pct": max_dd, "calmar": calmar,
            "equity": equity, "returns": all_rets,
        }

    print(f"\n{'='*100}")
    print("STITCHED OOS 全期表现 (2016-01 起, 所有OOS折拼接)")
    print(f"{'='*100}")
    metrics = {}
    for key, label in (("treat", "Treatment (dynamic ATR)"), ("bench_a", "Benchmark A (fixed 3.5)"), ("bench_b", "Benchmark B (fixed 5.5)")):
        m = stitch_and_metrics(stitched[key])
        metrics[key] = m
        total_trades = sum(rec[key].get("num_trades", 0) for rec in per_fold_summary)
        total_wins_weighted = sum(rec[key].get("num_trades", 0) * rec[key].get("win_rate_pct", 0) / 100 for rec in per_fold_summary)
        win_rate = (total_wins_weighted / total_trades * 100) if total_trades else float("nan")
        print(f"\n{label}:")
        print(f"  总收益: {m['total_return_pct']:.2f}%   Sharpe: {m['sharpe']:.3f}"
              f"   Calmar: {m['calmar']:.3f}   最大回撤: {m['max_drawdown_pct']:.2f}%")
        print(f"  OOS期间总交易笔数: {total_trades}   加权平均胜率: {win_rate:.1f}%")

    # ── 2022 bear-market-specific slice ──────────────────────────────────────
    print(f"\n{'='*100}")
    print("2022熊市窗口专项对比 (OOS落在2022年内的折)")
    print(f"{'='*100}")
    bear_folds = [rec for rec in per_fold_summary
                  if pd.Timestamp(rec["oos_start"]).year == 2022 or pd.Timestamp(rec["oos_end"]).year == 2022]
    for key, label in (("treat", "Treatment"), ("bench_a", "Bench A (3.5)"), ("bench_b", "Bench B (5.5)")):
        rets_list = []
        total_trades = 0
        for rec in bear_folds:
            r = rec[key]
            if not r:
                continue
            rets_list.append(daily_returns_from_result(r))
            total_trades += r.get("num_trades", 0)
        if not rets_list:
            continue
        m = stitch_and_metrics(rets_list)
        print(f"  {label:<16}: 2022 OOS收益={m['total_return_pct']:.2f}%  "
              f"最大回撤={m['max_drawdown_pct']:.2f}%  交易笔数={total_trades}  (共{len(bear_folds)}折)")

    # ── Low-vol vs high-vol regime whipsaw check ─────────────────────────────
    print(f"\n{'='*100}")
    print("低波动 vs 高波动 regime 下的交易笔数对比 (检查动态ATR是否减少低波动whipsaw)")
    print(f"{'='*100}")
    vols = [rec["oos_vol"] for rec in per_fold_summary if rec["oos_vol"] == rec["oos_vol"]]
    median_vol = float(np.median(vols)) if vols else float("nan")
    print(f"  (按跨折OOS平均波动率中位数 {median_vol*100:.2f}% 分组)")
    for regime_name, cond in (("低波动折", lambda v: v < median_vol), ("高波动折", lambda v: v >= median_vol)):
        print(f"\n  [{regime_name}]")
        for key, label in (("treat", "Treatment"), ("bench_a", "Bench A (3.5)"), ("bench_b", "Bench B (5.5)")):
            trades = sum(rec[key].get("num_trades", 0) for rec in per_fold_summary
                         if rec["oos_vol"] == rec["oos_vol"] and cond(rec["oos_vol"]))
            n_folds = sum(1 for rec in per_fold_summary if rec["oos_vol"] == rec["oos_vol"] and cond(rec["oos_vol"]))
            print(f"    {label:<16}: 交易笔数={trades}  ({n_folds}折,平均{trades/n_folds if n_folds else 0:.1f}笔/折)")


if __name__ == "__main__":
    main()
