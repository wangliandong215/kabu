"""
_atr55_riskoff_stress_mc.py -- validate the new Macro Risk-Off half-size
circuit breaker (engine.regime.qqq_risk_off_series, wired into
backtest_portfolio.simulate_from_prepared's entry sizing 2026-07-04) against
the pre-riskoff ATR=5.5 baseline (see _atr55_stress_mc.log /
_atr55_stress_mc.py): baseline commission-stress relative return drop was
-35.6% (277.18%->178.44%), trade-level... no, that earlier test used DAILY-
return-order Monte Carlo (mean maxdd=-22.81%, 5th pct=-31.94%, worst
single=-45.13%). This script switches the Monte Carlo to TRADE-level
resequencing (shuffle the order of individual realized trade P&Ls, not daily
portfolio returns) per the new request, and re-runs the same 2x-commission
stress test now that Risk-Off halving is active.

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _atr55_riskoff_stress_mc.py
"""
import contextlib

import numpy as np
import pandas as pd

import backtest_portfolio as bp

FULL_START = "2015-01-01"
FULL_END   = "2026-07-03"
CASH       = 7_000_000.0
USD_JPY    = 140.0
N_MC       = 1000

LOG_PATH = "_atr55_riskoff_stress_mc.log"


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")
    bp.ATR_TRAIL_MULT = 5.5

    print(f"Fetching full history {FULL_START} ~ {FULL_END} once "
          f"(ATR=5.5 + Risk-Off halving now permanently wired in)...")
    with contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(bp.BACKTEST_STOCKS, FULL_START, FULL_END, usd_to_jpy=USD_JPY)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    print("Running BASELINE (commission=0.001, Risk-Off ON)...")
    with contextlib.redirect_stdout(log_f):
        bp.COMMISSION = 0.001
        r_base = bp.simulate_from_prepared(prepared, cash=CASH, currency="JPY ")

    print("Running STRESS (commission=0.002, 2x, Risk-Off ON)...")
    with contextlib.redirect_stdout(log_f):
        bp.COMMISSION = 0.002
        r_stress = bp.simulate_from_prepared(prepared, cash=CASH, currency="JPY ")

    log_f.close()

    print(f"\n{'='*100}")
    print("手续费压力测试 -- Risk-Off半仓 ON vs 之前无Risk-Off的ATR=5.5结果")
    print(f"{'='*100}")
    print(f"{'':<32}{'总收益%':>10}{'Sharpe':>9}{'最大回撤%':>11}{'交易笔数':>9}")
    print(f"{'baseline(0.001, RiskOff ON)':<32}{r_base['total_return_pct']:>10.2f}"
          f"{r_base['sharpe']:>9.3f}{r_base['max_drawdown_pct']:>11.2f}{r_base['num_trades']:>9}")
    print(f"{'stress(0.002,2x, RiskOff ON)':<32}{r_stress['total_return_pct']:>10.2f}"
          f"{r_stress['sharpe']:>9.3f}{r_stress['max_drawdown_pct']:>11.2f}{r_stress['num_trades']:>9}")
    ret_drop_rel = (r_stress['total_return_pct'] - r_base['total_return_pct']) / r_base['total_return_pct'] * 100
    sharpe_drop_rel = (r_stress['sharpe'] - r_base['sharpe']) / r_base['sharpe'] * 100
    print(f"\n收益相对跌幅: {ret_drop_rel:.1f}%   Sharpe相对跌幅: {sharpe_drop_rel:.1f}%")
    print(f"\n[对照] 无Risk-Off, ATR=5.5: baseline 277.18%/0.920/-19.85%/873笔"
          f" -> stress 178.44%/0.757/-20.94%/927笔  (收益相对跌幅-35.6%, Sharpe相对跌幅-17.7%)")
    print(f"[对照] 无Risk-Off, ATR=4.0(旧): baseline 163.09%/0.758/-23.54%/1067笔"
          f" -> stress 128.97%/0.648/-24.25%/1090笔 (收益相对跌幅-20.9%, Sharpe相对跌幅-14.5%)")

    # ── Trade-level Monte Carlo (reshuffle realized trade P&Ls, not daily returns) ──
    print(f"\n{'='*100}")
    print(f"Trade-level Monte Carlo (N={N_MC}, 基于 baseline(0.001, RiskOff ON) 的逐笔已实现P&L重排)")
    print(f"{'='*100}")
    sell_trades = [t for t in r_base["trade_log"] if t["side"] == "SELL"]
    pnls = np.array([t["pnl"] for t in sell_trades], dtype=float)
    print(f"trades used: {len(pnls)}  (sum pnl = {pnls.sum():,.0f}, matches final equity - cash approx)")

    trade_rets = pnls / CASH  # % of initial capital per trade, compounded sequentially

    rng = np.random.default_rng(42)
    mc_returns = np.empty(N_MC)
    mc_maxdds = np.empty(N_MC)
    for i in range(N_MC):
        shuffled = rng.permutation(trade_rets)
        curve = np.cumprod(1 + shuffled)
        mc_returns[i] = (curve[-1] - 1) * 100
        roll_max = np.maximum.accumulate(curve)
        dd = (curve - roll_max) / roll_max
        mc_maxdds[i] = dd.min() * 100

    actual_curve = np.cumprod(1 + trade_rets)
    actual_return = (actual_curve[-1] - 1) * 100
    actual_roll_max = np.maximum.accumulate(actual_curve)
    actual_maxdd = ((actual_curve - actual_roll_max) / actual_roll_max).min() * 100
    print(f"实际交易顺序 (trade-level compounding, 近似值不等于全equity曲线): "
          f"return={actual_return:+.2f}%  maxdd={actual_maxdd:.2f}%")

    print(f"\nTrade-level MC 分布 ({N_MC}次随机重排同一批交易P&L):")
    print(f"  收益: mean={mc_returns.mean():+.2f}%  median={np.median(mc_returns):+.2f}%"
          f"  5th pct={np.percentile(mc_returns, 5):+.2f}%  95th pct={np.percentile(mc_returns, 95):+.2f}%")
    print(f"  回撤: mean={mc_maxdds.mean():.2f}%  median={np.median(mc_maxdds):.2f}%"
          f"  5th pct(最差)={np.percentile(mc_maxdds, 5):.2f}%  95th pct(最好)={np.percentile(mc_maxdds, 95):.2f}%")
    print(f"  单次最差模拟回撤: {mc_maxdds.min():.2f}%")
    print(f"\n[对照] 之前daily-return-order MC (无trade-level对照数据，方法不同不能直接比较数值，"
          f"仅供参考): ATR=5.5无RiskOff时 mean maxdd=-22.81%  5th pct(最差)=-31.94%  单次最差=-45.13%")


if __name__ == "__main__":
    main()
