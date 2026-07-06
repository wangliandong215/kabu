"""
engine/momentum.py — Cross-Sectional Momentum (动量轮动).

Strategy logic:
  1. Fetch historical closes for all codes.
  2. Compute each stock's return over `lookback_days` (skip last `skip_days`
     to avoid short-term reversal — standard Jegadeesh-Titman adjustment).
  3. Rank stocks by return descending.
  4. Hold the top `top_k` stocks; rotate out of anything that falls off the list.

Backtest:
  run_backtest_momentum(codes, lookback_days, skip_days, top_k, hold_days,
                        initial_cash, commission)
  → walks forward in time, rebalancing every `hold_days` bars.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd

import notify.alert as alert
from data.fetcher import fetch_kline


# ── Live scoring ──────────────────────────────────────────────────────────────

def compute_momentum(
    codes: List[str],
    lookback_days: int = 126,   # ~6 months of trading days
    skip_days: int = 5,         # skip most-recent week (reversal filter)
    ktype: str = "K_DAY",
) -> List[Dict]:
    """
    Return a list of dicts sorted by momentum score (highest first):
      [{"code": "US.NXPI", "return_pct": 34.2, "price": 281.8}, ...]

    Stocks with insufficient data are excluded.
    """
    scores = []
    total  = len(codes)

    for i, code in enumerate(codes, 1):
        try:
            bars_needed = lookback_days + skip_days + 5
            df = fetch_kline(code, ktype=ktype, bars=bars_needed)
            if df is None or len(df) < lookback_days + skip_days:
                continue

            closes = df["close"].astype(float).reset_index(drop=True)
            # end = most-recent bar minus skip_days
            end_idx   = len(closes) - 1 - skip_days
            start_idx = end_idx - lookback_days
            if start_idx < 0:
                continue

            p_start = closes.iloc[start_idx]
            p_end   = closes.iloc[end_idx]
            p_now   = closes.iloc[-1]

            if p_start <= 0:
                continue

            ret = (p_end - p_start) / p_start * 100
            scores.append({
                "code":       code,
                "return_pct": round(ret, 2),
                "price":      round(float(p_now), 4),
            })
            alert.log(f"momentum [{i}/{total}]: {code:<20s}  "
                      f"return={ret:+.1f}%  price={p_now:.2f}")
        except Exception as exc:
            alert.error(f"momentum [{i}/{total}]: {code} — {exc}")

    scores.sort(key=lambda x: x["return_pct"], reverse=True)
    return scores


def top_picks(
    codes: List[str],
    top_k: int = 5,
    lookback_days: int = 126,
    skip_days: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Return (top_k_buy_list, full_ranked_list).
    Prints a summary table.
    """
    ranked = compute_momentum(codes, lookback_days=lookback_days,
                              skip_days=skip_days)
    buys   = ranked[:top_k]

    print()
    print("=" * 62)
    print(f"Momentum Ranking  (lookback={lookback_days}d, skip={skip_days}d)")
    print("=" * 62)
    print(f"  {'Rank':<6} {'Code':<20} {'Return':>8}  {'Price':>12}")
    print("  " + "-" * 52)
    for rank, r in enumerate(ranked, 1):
        marker = " <-- BUY" if rank <= top_k else ""
        print(f"  {rank:<6} {r['code']:<20} {r['return_pct']:>+7.1f}%"
              f"  {r['price']:>12.2f}{marker}")
    print("=" * 62)

    return buys, ranked


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest_momentum(
    codes: List[str],
    lookback_days: int = 126,
    skip_days: int = 5,
    top_k: int = 5,
    hold_days: int = 21,        # rebalance every ~1 month
    initial_cash: float = 100_000.0,
    commission: float = 0.001,    # 0.1% per trade
    slippage: float = 0.0005,     # 0.05% per trade
    ktype: str = "K_DAY",
    bars: int = 500,
) -> dict:
    """
    Walk-forward cross-sectional momentum backtest.

    At each rebalancing date:
      - Score all codes on their lookback return up to that date.
      - Hold equal-weight top_k stocks.
      - Simulate trades with commission.

    Returns performance metrics dict.
    """
    import math

    # ── Fetch all close series ────────────────────────────────────────────────
    import time
    print(f"Fetching data for {len(codes)} codes…")
    close_map: Dict[str, pd.Series] = {}
    for i, code in enumerate(codes, 1):
        try:
            df = fetch_kline(code, ktype=ktype, bars=bars)
            if df is not None and len(df) >= lookback_days + skip_days + hold_days:
                close_map[code] = df["close"].astype(float).reset_index(drop=True)
                print(f"  [{i}/{len(codes)}] {code} ok ({len(df)} bars)")
            else:
                print(f"  [{i}/{len(codes)}] {code} skipped (insufficient data)")
        except Exception as e:
            print(f"  [{i}/{len(codes)}] {code} error: {e}")
        if i < len(codes):
            time.sleep(0.5)

    if not close_map:
        print("Error: no data available for any code")
        return {}

    # Align to shortest series length
    min_len = min(len(s) for s in close_map.values())
    for code in list(close_map):
        close_map[code] = close_map[code].iloc[-min_len:].reset_index(drop=True)

    n_bars  = min_len
    cash    = initial_cash
    holdings: Dict[str, float] = {}   # code → share qty (float for simulation)
    equity_curve = [initial_cash]

    rebalance_dates = list(range(lookback_days + skip_days, n_bars, hold_days))
    trades_total = 0

    for t in rebalance_dates:
        # ── Score at time t ───────────────────────────────────────────────────
        scored = []
        for code, closes in close_map.items():
            end_idx   = t - skip_days
            start_idx = end_idx - lookback_days
            if start_idx < 0 or end_idx >= len(closes):
                continue
            p_start = closes.iloc[start_idx]
            p_end   = closes.iloc[end_idx]
            if p_start <= 0:
                continue
            scored.append((code, (p_end - p_start) / p_start))

        scored.sort(key=lambda x: x[1], reverse=True)
        target_codes = {c for c, _ in scored[:top_k]}

        # ── Liquidate positions not in new top_k ──────────────────────────────
        for code in list(holdings):
            if code not in target_codes:
                price = float(close_map[code].iloc[t]) * (1 - slippage)
                proceeds = holdings[code] * price * (1 - commission)
                cash += proceeds
                trades_total += 1
                del holdings[code]

        # ── Buy new top_k in equal weight ────────────────────────────────────
        portfolio_value = cash + sum(
            holdings[c] * float(close_map[c].iloc[t]) for c in holdings
        )
        slot_value = portfolio_value / top_k

        for code in target_codes:
            if code not in holdings:
                price = float(close_map[code].iloc[t]) * (1 + slippage)
                if price <= 0:
                    continue
                qty = (slot_value * (1 - commission)) / price
                cost = qty * price * (1 + commission)
                if cost <= cash:
                    holdings[code] = qty
                    cash -= cost
                    trades_total += 1

        # ── Mark portfolio value ──────────────────────────────────────────────
        pv = cash + sum(
            holdings[c] * float(close_map[c].iloc[t]) for c in holdings
        )
        equity_curve.append(pv)

    # ── Final liquidation ─────────────────────────────────────────────────────
    final_t = n_bars - 1
    for code in list(holdings):
        price = float(close_map[code].iloc[final_t])
        cash += holdings[code] * price * (1 - commission)
    final_value = cash

    # ── Metrics ───────────────────────────────────────────────────────────────
    total_return = (final_value - initial_cash) / initial_cash
    n_years = (len(rebalance_dates) * hold_days) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    eq = pd.Series(equity_curve)
    roll_max = eq.cummax()
    drawdowns = (eq - roll_max) / roll_max
    max_dd = float(drawdowns.min()) * 100

    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)
              if daily_ret.std() > 0 else float("nan"))

    result = {
        "strategy":            "cross_sectional_momentum",
        "codes_scanned":       len(close_map),
        "top_k":               top_k,
        "lookback_days":       lookback_days,
        "skip_days":           skip_days,
        "hold_days":           hold_days,
        "bars":                n_bars,
        "initial_cash":        round(initial_cash, 2),
        "final_value":         round(final_value, 2),
        "total_return_pct":    round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return * 100, 2),
        "sharpe_ratio":        round(float(sharpe), 4) if not math.isnan(float(sharpe)) else None,
        "max_drawdown_pct":    round(max_dd, 2),
        "num_rebalances":      len(rebalance_dates),
        "num_trades":          trades_total,
    }

    _print_result(result)
    return result


def _print_result(r: dict) -> None:
    print()
    print("=" * 70)
    print("Backtest: Cross-Sectional Momentum")
    print("=" * 70)
    print(f"Universe:          {r['codes_scanned']} stocks")
    print(f"Top K held:        {r['top_k']}")
    print(f"Lookback:          {r['lookback_days']} days  "
          f"(skip last {r['skip_days']}d)")
    print(f"Rebalance every:   {r['hold_days']} days  "
          f"({r['num_rebalances']} rebalances, {r['num_trades']} trades)")
    print(f"Bars:              {r['bars']}")
    print()
    print(f"Initial Capital:   {r['initial_cash']:>14,.2f}")
    print(f"Final Value:       {r['final_value']:>14,.2f}")
    print()
    print(f"Total Return:      {r['total_return_pct']:>+.2f}%")
    print(f"Annualized Return: {r['annualized_return_pct']:>+.2f}%")
    print(f"Sharpe Ratio:      {r['sharpe_ratio'] if r['sharpe_ratio'] else 'N/A'}")
    print(f"Max Drawdown:      {r['max_drawdown_pct']:.2f}%")
    print("=" * 70)
    print("* Past performance does not guarantee future results.")
    print("=" * 70)
