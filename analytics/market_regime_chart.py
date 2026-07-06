"""
analytics/market_regime_chart.py -- v2.5 Market Regime Detection visual
verification tool. Plots close price with a colored background band per
classified regime, so a human can eyeball whether the rule-based classifier
(engine/market_regime.py) lines up with what the chart obviously looks like
(e.g. 2022's bear market should show up as high-volatility, 2023-2024's
melt-up should show up as low-volatility trend).

This is a read-only diagnostic — it does not feed into any trading decision.

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python analytics/market_regime_chart.py                      # QQQ, full cached history
  python analytics/market_regime_chart.py US.NVDA --start 2022-01-01 --end 2026-07-03
  python analytics/market_regime_chart.py --csv path/to/data.csv   # arbitrary OHLCV, no moomoo needed
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import backtest
import config
from engine.market_regime import Regime, classify_series

REGIME_COLORS = {
    Regime.UNKNOWN:           "#cccccc",
    Regime.LOW_VOL_SIDEWAYS:  "#a6cee3",
    Regime.HIGH_VOL_SIDEWAYS: "#fdbf6f",
    Regime.LOW_VOL_TREND:     "#b2df8a",
    Regime.HIGH_VOL_TREND:    "#fb9a99",
}

OUT_DIR = os.path.dirname(__file__)


def _load_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    raw = backtest.fetch_kline(code, start, end)
    df = raw[["time_key", "high", "low", "close", "volume"]].copy()
    df["time_key"] = pd.to_datetime(df["time_key"])
    df = df.set_index("time_key").astype(float)
    return df


def _load_csv(path: str) -> pd.DataFrame:
    """Load an arbitrary OHLCV CSV (Date,Open,High,Low,Close,Volume header,
    case-insensitive) — bypasses moomoo/backtest.fetch_kline entirely, so
    the classifier can be sanity-checked against hand-built or third-party
    data without needing OpenD or a real stock code."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    date_col = "date" if "date" in raw.columns else raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col])
    df = raw.set_index(date_col)[["high", "low", "close", "volume"]].astype(float)
    return df.sort_index()


def _plot(code: str, df: pd.DataFrame, regimes: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["close"], color="black", linewidth=1.0, label="close")

    # Shade contiguous same-regime runs as single spans (fewer, cleaner
    # rectangles than one axvspan per bar).
    regime_vals = regimes["regime"].to_numpy()
    seen_labels = set()
    start_i = 0
    for i in range(1, len(regime_vals) + 1):
        if i == len(regime_vals) or regime_vals[i] != regime_vals[start_i]:
            regime = regime_vals[start_i]
            label = regime.name if regime.name not in seen_labels else None
            seen_labels.add(regime.name)
            ax.axvspan(df.index[start_i], df.index[i - 1],
                       color=REGIME_COLORS.get(regime, "#ffffff"),
                       alpha=0.4, label=label)
            start_i = i

    ax.set_title(f"Market Regime Detection — {code} ({df.index[0].date()} ~ {df.index[-1].date()})")
    ax.set_xlabel("date")
    ax.set_ylabel("close")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f"_market_regime_{code.replace('.', '_')}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _print_summary(regimes: pd.DataFrame) -> None:
    counts = regimes["regime"].value_counts()
    total = len(regimes)
    print("\nRegime distribution:")
    for regime in Regime:
        n = int(counts.get(regime, 0))
        if n == 0:
            continue
        print(f"  {regime.name:<20} {n:>5} bars  ({n / total * 100:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Visualize Market Regime Detection (v2.5) on a K-line chart.")
    parser.add_argument("code", nargs="?", default=config.QQQ_CORE_CODE,
                         help=f"stock code (default: {config.QQQ_CORE_CODE}, the market proxy); "
                              f"used only as a chart/output-file label when --csv is given")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-07-03")
    parser.add_argument("--csv", default=None,
                         help="path to a local OHLCV CSV (Date,Open,High,Low,Close,Volume) — "
                              "bypasses moomoo entirely")
    args = parser.parse_args()

    if args.csv:
        label = args.code if args.code != config.QQQ_CORE_CODE else os.path.splitext(os.path.basename(args.csv))[0]
        df = _load_csv(args.csv)
    else:
        label = args.code
        df = _load_ohlcv(args.code, args.start, args.end)

    regimes = classify_series(df)

    out_path = _plot(label, df, regimes)
    _print_summary(regimes)
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    main()
