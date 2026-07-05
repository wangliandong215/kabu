"""
kabu — moomoo quantitative trading
analyze.py — fetch historical K-line data and compute technical indicators.

Supported strategies:
  ma       Moving Average crossover (MA5 / MA20 by default)
  rsi      RSI mean-reversion  (period=14, oversold<30, overbought>70)
  macd     MACD histogram      (12 / 26 / 9)
  boll     Bollinger Bands %B  (period=20, 2 std)
  combined All four — majority-vote signal (default)

Usage:
  python analyze.py US.AAPL
  python analyze.py HK.00700 --strategy ma
  python analyze.py SH.600519 --strategy rsi
  python analyze.py US.NVDA --ktype K_60M --strategy combined --json
"""
import argparse
import json
import sys
from datetime import datetime

from data.fetcher import fetch_kline
from strategies import get_strategy


def analyze(
    code: str,
    strategy_name: str = "combined",
    ktype: str = "K_DAY",
    bars: int = 120,
    output_json: bool = False,
) -> dict:
    strategy = get_strategy(strategy_name)

    df = fetch_kline(code, ktype=ktype, bars=bars)
    if df is None or len(df) == 0:
        print(f"Error: no data returned for {code}")
        sys.exit(1)

    if len(df) < strategy.required_bars:
        print(
            f"Warning: only {len(df)} bars available, "
            f"strategy needs {strategy.required_bars}. Results may be inaccurate."
        )

    result = strategy.full_result(df)

    last   = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else last
    price  = float(last["close"])
    prev_c = float(prev["close"])
    change = price - prev_c
    pct    = change / prev_c * 100 if prev_c else 0.0

    recent = [
        {
            "time":   str(r["time_key"]),
            "open":   float(r["open"]),
            "high":   float(r["high"]),
            "low":    float(r["low"]),
            "close":  float(r["close"]),
            "volume": int(r["volume"]),
        }
        for _, r in df.tail(5).iterrows()
    ]

    output = {
        "code":             code,
        "strategy":         strategy_name,
        "ktype":            ktype,
        "analyzed_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price":    round(price, 4),
        "price_change":     round(change, 4),
        "price_change_pct": round(pct, 2),
        "data_bars":        len(df),
        "recent_bars":      recent,
        **result,
    }

    if output_json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        _print_result(output)

    return output


def _print_result(r: dict) -> None:
    price  = r["current_price"]
    change = r["price_change"]
    pct    = r["price_change_pct"]
    arrow  = "+" if change >= 0 else ""

    print("=" * 80)
    print(f"Quantitative Analysis: {r['code']}  [{r['strategy'].upper()} / {r['ktype']}]")
    print("=" * 80)
    print(f"Current Price:  {price:.4f}  ({arrow}{change:.4f}, {arrow}{pct:.2f}%)")
    print(f"Signal:         {r.get('signal', 'HOLD')}  "
          f"(strength {r.get('signal_strength', 0):.0%})")
    print(f"Analyzed At:    {r['analyzed_at']}  ({r['data_bars']} bars loaded)")

    # combined strategy has sub_indicators; individual strategies have flat indicators
    sub = r.get("sub_indicators") or {}
    if sub:
        print()
        print("Technical Indicators:")
        print(f"  {'Name':<12} {'Signal':<8} Detail")
        print("  " + "-" * 64)
        for name, ind in sub.items():
            lbl = f"[{ind.get('signal', 'HOLD')}]"
            print(f"  {name.upper():<12} {lbl:<8} {ind.get('detail', '')}")
    elif r.get("detail"):
        print(f"Detail:         {r['detail']}")

    buy_n  = r.get("buy_votes",  0)
    sell_n = r.get("sell_votes", 0)
    hold_n = r.get("hold_votes", 0)
    if buy_n + sell_n + hold_n > 0:
        print()
        print(f"Vote Summary:   BUY={buy_n}  SELL={sell_n}  HOLD={hold_n}  "
              f"→ Overall: {r.get('signal', 'HOLD')}")

    if r.get("recent_bars"):
        print()
        print("Recent Candles (latest 5):")
        print(f"  {'Time':<22} {'Open':>10} {'High':>10} {'Low':>10} "
              f"{'Close':>10} {'Volume':>14}")
        print("  " + "-" * 82)
        for b in r["recent_bars"]:
            print(
                f"  {b['time']:<22} {b['open']:>10.2f} {b['high']:>10.2f} "
                f"{b['low']:>10.2f} {b['close']:>10.2f} {b['volume']:>14,}"
            )

    print("=" * 80)
    print("* For informational purposes only. Not investment advice.")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Quantitative technical analysis (moomoo OpenD)"
    )
    p.add_argument("code", help="Stock code, e.g. US.AAPL, HK.00700, SH.600519")
    p.add_argument(
        "--strategy",
        choices=["ma", "rsi", "macd", "boll", "combined"],
        default="combined",
    )
    p.add_argument(
        "--ktype",
        choices=["K_1M", "K_5M", "K_15M", "K_30M", "K_60M", "K_DAY", "K_WEEK", "K_MON"],
        default="K_DAY",
    )
    p.add_argument("--bars", type=int, default=120, help="Number of bars to load")
    p.add_argument("--json", action="store_true", dest="output_json")
    args = p.parse_args()

    analyze(
        code=args.code,
        strategy_name=args.strategy,
        ktype=args.ktype,
        bars=args.bars,
        output_json=args.output_json,
    )
