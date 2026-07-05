"""
_atr_regime_correlation.py -- characterize each walk-forward window's market
regime and correlate against the best ATR_TRAIL_MULT found for that window
(see _atr_walkforward_sweep.py results: W1=5.0, W2=3.25, W3=3.75, W4=5.75).

Goal: figure out WHY the best ATR drifts before designing any dynamic-ATR
rule -- if a regime variable (ADX, trend strength, volatility) correlates
strongly with best-ATR across the 4 windows, that's the theoretical basis
for conditioning ATR_TRAIL_MULT on that variable. If nothing correlates,
dynamic ATR isn't justified by this evidence and the drift may just be
window-specific noise/overfitting.

Caveat (must be reported, not hidden): n=4 windows means any correlation
coefficient is extremely fragile -- a "clean" r=0.8+ can easily arise by
chance with only 4 points. Treat this as a first-pass signal, not proof.

Variables computed per window (all from data already used in the sweep):
  1. avg_adx_breadth   -- cross-sectional mean ADX(14) across the ~90-stock
                          backtest universe (breadth: how "trendy" is the
                          whole traded universe, not just the index)
  2. qqq_adx           -- ADX(14) of QQQ itself (benchmark's own trend
                          strength/cleanliness)
  3. qqq_realized_vol  -- QQQ annualized realized volatility (stdev of daily
                          returns x sqrt(252)). NOTE: true historical VIX
                          index level is NOT fetchable via moomoo/OpenD
                          (US.VIX errors "unknown stock"; only VXX/UVXY
                          futures ETNs are listed, and those decay/reverse-
                          split so they don't track absolute VIX *level*
                          over multi-year windows) -- this realized-vol
                          proxy is the closest available substitute and is
                          reported as such, not as real VIX.
  4. avg_atr_pct       -- cross-sectional mean ATR(14)/close across the
                          stock universe (average daily range as % of
                          price -- the volatility unit ATR_TRAIL_MULT
                          actually multiplies against)

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _atr_regime_correlation.py
"""
import numpy as np
import pandas as pd

import backtest_portfolio as bp
import engine.regime as regime_mod

WINDOWS = [
    ("W1", "2015-01-01", "2019-01-01", 5.0),
    ("W2", "2017-07-01", "2021-07-01", 3.25),
    ("W3", "2020-01-01", "2024-01-01", 3.75),
    ("W4", "2022-07-01", "2026-07-03", 5.75),
]


def _window_slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


def main():
    rows = []

    for win_name, start, end, best_atr in WINDOWS:
        print(f"\n[{win_name}] {start} ~ {end}  fetching...")
        prepared = bp._prepare_backtest_data(bp.BACKTEST_STOCKS, start, end, usd_to_jpy=None)
        if prepared is None:
            print(f"[{win_name}] NO DATA, skipping")
            continue

        all_data = prepared["all_data"]
        benchmark_df = prepared["benchmark_df"]

        # ── 1 & 4: cross-sectional breadth ADX / ATR% across stock universe ──
        adx_means, atrpct_means = [], []
        for code, df in all_data.items():
            if code == bp.QQQ_CODE:
                continue
            sub = _window_slice(df, start, end)
            if len(sub) < 30:
                continue
            adx, _, _ = regime_mod._adx(sub["high"], sub["low"], sub["close"], 14)
            atr = regime_mod._atr(sub["high"], sub["low"], sub["close"], 14)
            atrpct = atr / sub["close"]
            adx_means.append(adx.mean())
            atrpct_means.append(atrpct.mean())

        avg_adx_breadth = float(np.nanmean(adx_means))
        avg_atr_pct     = float(np.nanmean(atrpct_means)) * 100  # as %

        # ── 2 & 3: QQQ-specific trend strength / realized volatility ──
        qsub = _window_slice(benchmark_df, start, end)
        q_adx, _, _ = regime_mod._adx(qsub["high"], qsub["low"], qsub["close"], 14)
        qqq_adx = float(q_adx.mean())

        qqq_ret = qsub["close"].astype(float).pct_change().dropna()
        qqq_realized_vol = float(qqq_ret.std() * np.sqrt(252) * 100)  # annualized %

        rows.append({
            "window": win_name, "start": start, "end": end, "best_atr": best_atr,
            "avg_adx_breadth": avg_adx_breadth,
            "qqq_adx": qqq_adx,
            "qqq_realized_vol_pct": qqq_realized_vol,
            "avg_atr_pct": avg_atr_pct,
        })
        print(f"[{win_name}] avg_adx_breadth={avg_adx_breadth:.2f}  qqq_adx={qqq_adx:.2f}"
              f"  qqq_realized_vol={qqq_realized_vol:.2f}%  avg_atr_pct={avg_atr_pct:.3f}%")

    print(f"\n{'='*90}")
    print("窗口市场特征 vs 最佳ATR")
    print(f"{'='*90}")
    print(f"{'窗口':<4} {'最佳ATR':>8} {'横截面ADX':>10} {'QQQ_ADX':>9} {'QQQ已实现波动%':>14} {'横截面ATR%':>11}")
    for r in rows:
        print(f"{r['window']:<4} {r['best_atr']:>8.2f} {r['avg_adx_breadth']:>10.2f}"
              f" {r['qqq_adx']:>9.2f} {r['qqq_realized_vol_pct']:>14.2f}"
              f" {r['avg_atr_pct']:>11.3f}")

    best_atrs = [r["best_atr"] for r in rows]
    print(f"\n{'='*90}")
    print("相关系数 (Pearson r, n=4 -- 极小样本，仅供参考，不构成统计显著性证据)")
    print(f"{'='*90}")
    metrics = ["avg_adx_breadth", "qqq_adx", "qqq_realized_vol_pct", "avg_atr_pct"]
    corr_results = []
    for m in metrics:
        vals = [r[m] for r in rows]
        r_val = float(np.corrcoef(best_atrs, vals)[0, 1])
        corr_results.append((m, r_val))
        print(f"  best_atr vs {m:<22}: r = {r_val:+.3f}")

    corr_results.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n相关性排序（绝对值从高到低）:")
    for m, r_val in corr_results:
        strength = ("强" if abs(r_val) >= 0.8 else
                    "中等" if abs(r_val) >= 0.5 else
                    "弱")
        print(f"  {m:<22}: r={r_val:+.3f}  ({strength})")


if __name__ == "__main__":
    main()
