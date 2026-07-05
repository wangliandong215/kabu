"""
_atr55_stress_mc.py -- re-run the commission-stress test and Monte Carlo
order-shuffle robustness check on the NEW locked default (ATR_TRAIL_MULT =
config.ATR_MULT_BASE = 5.5, 2026-07-04), for direct comparison against the
same tests previously run under the old ATR=4.0 default (see
_robust_stress_mc.log, 2026-07-04 00:04, baseline return=+163.09%
sharpe=0.758 maxdd=-23.54% trades=1067; commission 0.001->0.002 stress:
return=+128.97% sharpe=0.648 maxdd=-24.25% trades=1090; Monte Carlo N=1000
order-shuffle: mean maxdd=-22.57%, 5th pct(worst)=-31.96%, worst
single=-49.18%).

Same methodology, same full 2015-01-01~2026-07-03 window, same 142-stock
universe, same $7,000,000 JPY cash -- only ATR_TRAIL_MULT changed (4.0->5.5,
already locked in config.py/backtest_portfolio.py), so this is a clean
apples-to-apples re-test of "does the new baseline still have the same cost-
sensitivity / tail-risk profile, or did moving to 5.5 (which already cuts
trade count 1067->873) also help there for free."

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _atr55_stress_mc.py
"""
import contextlib

import numpy as np
import pandas as pd

import backtest_portfolio as bp

FULL_START = "2015-01-01"
FULL_END   = "2026-07-03"
CASH       = 7_000_000.0
USD_JPY    = 140.0

N_MC = 1000
LOG_PATH = "_atr55_stress_mc.log"


def run(prepared, commission):
    bp.COMMISSION = commission
    r = bp.simulate_from_prepared(prepared, cash=CASH, currency="JPY ")
    return r


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")
    bp.ATR_TRAIL_MULT = 5.5  # the newly locked default -- explicit here so this
                             # script's result doesn't silently depend on
                             # import-time module state

    print(f"Fetching full history {FULL_START} ~ {FULL_END} once "
          f"(ATR_TRAIL_MULT=5.5, shared across baseline + stress runs)...")
    with contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(bp.BACKTEST_STOCKS, FULL_START, FULL_END, usd_to_jpy=USD_JPY)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    print("Running BASELINE (commission=0.001)...")
    with contextlib.redirect_stdout(log_f):
        r_base = run(prepared, 0.001)

    print("Running STRESS (commission=0.002, 2x)...")
    with contextlib.redirect_stdout(log_f):
        r_stress = run(prepared, 0.002)

    log_f.close()

    def pct_change(new, old):
        return (new - old) / abs(old) * 100 if old else float("nan")

    print(f"\n{'='*100}")
    print("手续费/滑点压力测试 -- ATR=5.5 新基线 vs 旧ATR=4.0结果 (2026-07-04 00:04, _robust_stress_mc.log)")
    print(f"{'='*100}")
    print(f"{'':<28}{'总收益%':>10}{'Sharpe':>9}{'最大回撤%':>11}{'交易笔数':>9}")
    print(f"{'ATR=5.5 baseline(0.001)':<28}{r_base['total_return_pct']:>10.2f}"
          f"{r_base['sharpe']:>9.3f}{r_base['max_drawdown_pct']:>11.2f}{r_base['num_trades']:>9}")
    print(f"{'ATR=5.5 stress(0.002,2x)':<28}{r_stress['total_return_pct']:>10.2f}"
          f"{r_stress['sharpe']:>9.3f}{r_stress['max_drawdown_pct']:>11.2f}{r_stress['num_trades']:>9}")
    ret_drop_pp = r_base['total_return_pct'] - r_stress['total_return_pct']
    ret_drop_rel = pct_change(r_stress['total_return_pct'], r_base['total_return_pct'])
    sharpe_drop = r_base['sharpe'] - r_stress['sharpe']
    sharpe_drop_rel = pct_change(r_stress['sharpe'], r_base['sharpe'])
    print(f"\n收益下降: {ret_drop_pp:.2f}pp (相对 {-ret_drop_rel:.1f}%)   "
          f"Sharpe下降: {sharpe_drop:.3f} (相对 {-sharpe_drop_rel:.1f}%)")
    print(f"\n[对照] ATR=4.0旧基线同一测试: 收益下降34.12pp(相对20.9%)  Sharpe下降0.110(相对14.5%)")

    # ── Monte Carlo: reorder baseline daily returns, N times ─────────────────
    print(f"\n{'='*100}")
    print(f"Monte Carlo 顺序重排 (N={N_MC}, 基于ATR=5.5 baseline的真实日收益率序列)")
    print(f"{'='*100}")
    eq = r_base["equity_curve"]
    daily_ret = eq.pct_change().dropna().values
    first_ret = (eq.iloc[0] - CASH) / CASH
    all_rets = np.concatenate([[first_ret], daily_ret])

    rng = np.random.default_rng(42)
    mc_returns = np.empty(N_MC)
    mc_maxdds = np.empty(N_MC)
    for i in range(N_MC):
        shuffled = rng.permutation(all_rets)
        curve = np.cumprod(1 + shuffled)
        mc_returns[i] = (curve[-1] - 1) * 100
        roll_max = np.maximum.accumulate(curve)
        dd = (curve - roll_max) / roll_max
        mc_maxdds[i] = dd.min() * 100

    print(f"实际历史顺序: return={r_base['total_return_pct']:+.2f}%  maxdd={r_base['max_drawdown_pct']:.2f}%")
    print(f"\nMonte Carlo 分布 ({N_MC}次随机重排同一批日收益率):")
    print(f"  收益: mean={mc_returns.mean():+.2f}%  median={np.median(mc_returns):+.2f}%"
          f"  5th pct={np.percentile(mc_returns, 5):+.2f}%  95th pct={np.percentile(mc_returns, 95):+.2f}%")
    print(f"  回撤: mean={mc_maxdds.mean():.2f}%  median={np.median(mc_maxdds):.2f}%"
          f"  5th pct(最差)={np.percentile(mc_maxdds, 5):.2f}%  95th pct(最好)={np.percentile(mc_maxdds, 95):.2f}%")
    print(f"  单次最差模拟回撤: {mc_maxdds.min():.2f}%")
    print(f"\n[对照] ATR=4.0旧基线同一测试: mean maxdd=-22.57%  5th pct(最差)=-31.96%  单次最差=-49.18%")


if __name__ == "__main__":
    main()
