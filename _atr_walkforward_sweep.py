"""
_atr_walkforward_sweep.py -- 4-window rolling walk-forward ATR sweep.

Question being answered: is a single fixed ATR_TRAIL_MULT (currently 4.0,
see backtest_portfolio.py line ~93 / config.ATR_MULT_BASE) genuinely optimal
across market regimes, or does the best value drift enough between windows
that a dynamic/regime-dependent ATR multiplier would be worth building?

Method: 4 overlapping rolling windows spanning the full 2015-2026 history
(4-year windows, ~2.5y step), sweep ATR_TRAIL_MULT over 2.5..6.5 step 0.25
(17 values) inside each window, rank by combined Sharpe+total-return rank.

Data fetch + signal precompute (Steps 1-3 of backtest_portfolio.py) do NOT
depend on ATR_TRAIL_MULT, so each window is fetched ONCE and reused across
all 17 ATR variants (see _prepare_backtest_data / simulate_from_prepared
split) -- this is what keeps the full 4x17=68-run sweep to ~10 minutes
instead of ~100 minutes of redundant re-fetching.

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _atr_walkforward_sweep.py
"""
import contextlib
import io
import sys
import time

import backtest_portfolio as bp

WINDOWS = [
    ("W1", "2015-01-01", "2019-01-01"),
    ("W2", "2017-07-01", "2021-07-01"),
    ("W3", "2020-01-01", "2024-01-01"),
    ("W4", "2022-07-01", "2026-07-03"),
]

ATR_GRID = [round(2.5 + 0.25 * i, 2) for i in range(17)]  # 2.5 .. 6.5 step 0.25

CASH = 7_000_000.0
USD_JPY = 140.0

LOG_PATH = "_atr_walkforward_sweep.log"


def main():
    all_results = []  # list of dict: window, atr, total_return_pct, sharpe, max_dd, trades, win_rate
    log_f = open(LOG_PATH, "w", encoding="utf-8")

    t_start_all = time.time()
    for win_name, start, end in WINDOWS:
        print(f"\n[{win_name}] {start} ~ {end}  fetching + precomputing signals...")
        t0 = time.time()
        with contextlib.redirect_stdout(log_f):
            print(f"\n{'#'*70}\n# {win_name}: {start} ~ {end}\n{'#'*70}")
            prepared = bp._prepare_backtest_data(bp.BACKTEST_STOCKS, start, end, usd_to_jpy=USD_JPY)
        t1 = time.time()
        if prepared is None:
            print(f"[{win_name}] NO DATA, skipping")
            continue
        print(f"[{win_name}] fetch done in {t1-t0:.1f}s, sweeping {len(ATR_GRID)} ATR values...")

        for atr in ATR_GRID:
            bp.ATR_TRAIL_MULT = atr
            with contextlib.redirect_stdout(log_f):
                print(f"\n--- {win_name}  ATR_TRAIL_MULT={atr} ---")
                r = bp.simulate_from_prepared(prepared, cash=CASH, currency="JPY ")
            if not r:
                continue
            all_results.append({
                "window": win_name, "start": start, "end": end, "atr": atr,
                "total_return_pct": r["total_return_pct"],
                "sharpe": r["sharpe"],
                "max_drawdown_pct": r["max_drawdown_pct"],
                "num_trades": r["num_trades"],
                "win_rate_pct": r["win_rate_pct"],
            })
        t2 = time.time()
        print(f"[{win_name}] sweep done in {t2-t1:.1f}s (total {t2-t0:.1f}s)")

    log_f.close()
    print(f"\nALL DONE in {time.time()-t_start_all:.1f}s. Full per-trade logs: {LOG_PATH}")

    # ── Rank ATR values within each window by combined Sharpe+return rank ──
    print(f"\n{'='*78}")
    print("四窗口 Walk-Forward ATR 扫描结果")
    print(f"{'='*78}")

    best_per_window = {}
    for win_name, start, end in WINDOWS:
        rows = [r for r in all_results if r["window"] == win_name]
        if not rows:
            continue
        by_sharpe = sorted(rows, key=lambda r: r["sharpe"], reverse=True)
        by_return = sorted(rows, key=lambda r: r["total_return_pct"], reverse=True)
        rank_sharpe = {id(r): i for i, r in enumerate(by_sharpe)}
        rank_return = {id(r): i for i, r in enumerate(by_return)}
        for r in rows:
            r["combined_rank"] = rank_sharpe[id(r)] + rank_return[id(r)]
        rows_sorted = sorted(rows, key=lambda r: r["combined_rank"])
        best = rows_sorted[0]
        best_per_window[win_name] = best

        print(f"\n[{win_name}] {start} ~ {end}")
        print(f"  {'ATR':>6} {'收益%':>8} {'Sharpe':>8} {'回撤%':>8} {'交易数':>6} {'胜率%':>6} {'综合排名':>6}")
        for r in sorted(rows, key=lambda r: r["atr"]):
            marker = " <== BEST" if r is best else ""
            print(f"  {r['atr']:>6} {r['total_return_pct']:>8.2f} {r['sharpe']:>8.3f}"
                  f" {r['max_drawdown_pct']:>8.2f} {r['num_trades']:>6} {r['win_rate_pct']:>6.1f}"
                  f" {r['combined_rank']:>6}{marker}")

    print(f"\n{'='*78}")
    print("各窗口最佳 ATR 汇总")
    print(f"{'='*78}")
    best_atrs = []
    for win_name, _, _ in WINDOWS:
        b = best_per_window.get(win_name)
        if b:
            best_atrs.append(b["atr"])
            print(f"  {win_name}: ATR={b['atr']}  收益={b['total_return_pct']:.2f}%"
                  f"  Sharpe={b['sharpe']:.3f}  回撤={b['max_drawdown_pct']:.2f}%")

    if best_atrs:
        spread = max(best_atrs) - min(best_atrs)
        print(f"\n最佳ATR跨窗口范围: {min(best_atrs)} ~ {max(best_atrs)}  (spread={spread})")
        if spread <= 0.5:
            print("=> 四窗口最佳ATR高度一致，支持固定ATR方案。")
        elif spread <= 1.5:
            print("=> 四窗口最佳ATR有一定漂移，固定ATR仍大致可用，但值得留意。")
        else:
            print("=> 四窗口最佳ATR差异明显，固定ATR证据不足，值得考虑动态/分regime的ATR。")


if __name__ == "__main__":
    main()
