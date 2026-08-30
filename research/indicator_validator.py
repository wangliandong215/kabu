"""
research/indicator_validator.py — technical indicator Shadow Validation
(v2.11 Priority 4, long-term/manual — see config.py's "Research: Indicator
Shadow Validation" section).

OpenD's SDK does NOT expose a "give me RSI14's exact numeric value for this
stock on this day" API — only get_technical_unusual() (an "unusual
technical pattern" screener) and get_stock_filter() (a market-wide
condition screener). So this validator does NOT diff numbers; it checks
BUCKET agreement: does this system's own RSI14 computation
(engine.indicators.rsi_last — the exact function live sizing uses) classify
a code as oversold/overbought the same way moomoo's own get_stock_filter()
condition screen does, for the same market on the same day.

Deliberately a standalone, manually-invoked CLI (not on the automatic
research schedule yet) — this is a methodology being validated, not a
finished pipeline. Does NOT import portfolio/, risk/, strategies/, or
engine/runner.py, and never feeds back into a trading decision.

Usage:
  python -m research.indicator_validator --codes US.AAPL US.NVDA
  python -m research.indicator_validator          # defaults to config.WATCHLIST
"""
import argparse
from datetime import datetime
from typing import Dict, List, Set, Tuple

import common
import config
from data.fetcher import fetch_kline
from engine.indicators import rsi_last
from research.market_features_store import MarketFeaturesStore


def _bucket(rsi: float) -> str:
    if rsi is None:
        return "UNKNOWN"
    if rsi <= config.INDICATOR_SHADOW_RSI_OVERSOLD:
        return "OVERSOLD"
    if rsi >= config.INDICATOR_SHADOW_RSI_OVERBOUGHT:
        return "OVERBOUGHT"
    return "NEUTRAL"


def our_rsi_bucket(code: str) -> Tuple[str, float]:
    df = fetch_kline(code, "1d", 60)
    if df is None or len(df) == 0 or "close" not in df.columns:
        return "UNKNOWN", None
    rsi = rsi_last(df["close"], period=14)
    return _bucket(rsi), rsi


def moomoo_rsi_buckets(market: str) -> Tuple[Set[str], Set[str]]:
    """Return (oversold_codes, overbought_codes) per moomoo's own
    get_stock_filter() RSI14 condition screen for `market`, paginated up to
    a fixed cap (this is a manual/long-term tool, not called per-signal)."""
    oversold = _filtered_codes(market, config.INDICATOR_SHADOW_RSI_OVERSOLD, "LESS")
    overbought = _filtered_codes(market, config.INDICATOR_SHADOW_RSI_OVERBOUGHT, "MORE")
    return oversold, overbought


def run_validation(codes: List[str], store: MarketFeaturesStore) -> Dict[str, int]:
    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    now_iso = datetime.now().isoformat()

    by_market: Dict[str, List[str]] = {}
    for code in codes:
        by_market.setdefault(common.infer_market(code), []).append(code)

    stats = {"match": 0, "mismatch": 0, "unknown": 0}
    for market, market_codes in by_market.items():
        try:
            oversold, overbought = moomoo_rsi_buckets(market)
        except Exception as e:
            print(f"[error] moomoo_rsi_buckets({market}): {e!r}")
            continue

        for code in market_codes:
            try:
                our_bucket, rsi = our_rsi_bucket(code)
            except Exception as e:
                print(f"[error] our_rsi_bucket({code}): {e!r}")
                continue

            if code in oversold:
                moomoo_bucket = "OVERSOLD"
            elif code in overbought:
                moomoo_bucket = "OVERBOUGHT"
            else:
                moomoo_bucket = "NEUTRAL"

            if our_bucket == "UNKNOWN":
                stats["unknown"] += 1
                continue

            match = int(our_bucket == moomoo_bucket)
            stats["match" if match else "mismatch"] += 1
            store.upsert("indicator_shadow_log", {
                "code": code, "snapshot_date": snapshot_date, "indicator": "RSI14",
                "our_bucket": our_bucket, "moomoo_bucket": moomoo_bucket,
                "match": match, "created_at": now_iso,
            })
            if not match:
                print(f"[mismatch] {code}: our={our_bucket}(rsi={rsi}) moomoo={moomoo_bucket}")
    return stats


# ── Internal ──────────────────────────────────────────────────────────────────

@common.retry()
def _filtered_codes(market: str, threshold: float, direction: str) -> Set[str]:
    import moomoo as ft

    market_enum = getattr(ft.Market, market, ft.Market.US)
    relative_position = ft.RelativePosition.LESS if direction == "LESS" else ft.RelativePosition.MORE

    f = ft.CustomIndicatorFilter()
    f.stock_field1 = ft.StockField.RSI
    f.stock_field1_para = [14]
    # CustomIndicatorFilter compares stock_field1 against stock_field2, not
    # a bare number — StockField.VALUE + .value is the "compare against a
    # fixed constant" idiom (confirmed live 2026-08-30: omitting
    # stock_field2 fails immediately with "Missing nessary parameters:
    # stock_field2").
    f.stock_field2 = ft.StockField.VALUE
    f.relative_position = relative_position
    f.value = threshold
    f.ktype = ft.KLType.K_DAY

    ctx = common.make_quote_ctx()
    codes: Set[str] = set()
    try:
        begin = 0
        for _ in range(5):   # cap at 5 pages (1000 stocks) — long-term manual tool, not a hot path
            ret, data = ctx.get_stock_filter(market=market_enum, filter_list=[f], begin=begin, num=200)
            if ret != ft.RET_OK or data is None:
                break
            # get_stock_filter's exact reply shape (tuple vs raw list) wasn't
            # confirmed against a live call at implementation time — handle
            # both a bare list of row-likes and a (last_page, all_count,
            # ret_list) tuple defensively.
            last_page, ret_list = True, data
            if isinstance(data, tuple) and len(data) == 3:
                last_page, _all_count, ret_list = data
            for row in ret_list:
                stock_code = row.get("stock_code") if isinstance(row, dict) else getattr(row, "stock_code", None)
                if stock_code:
                    codes.add(stock_code)
            if last_page or not ret_list:
                break
            begin += 200
    finally:
        common.safe_close(ctx)
    return codes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Indicator Shadow Validation (RSI14 bucket agreement) — "
                     "observation only, never affects live trading. See module docstring.")
    parser.add_argument("--codes", nargs="+", default=None,
                         help="Override the code list (default: config.WATCHLIST).")
    args = parser.parse_args(argv)

    codes = args.codes or config.WATCHLIST
    store = MarketFeaturesStore()
    try:
        stats = run_validation(codes, store)
        print(f"\nRSI14 bucket validation: {stats['match']} match, "
              f"{stats['mismatch']} mismatch, {stats['unknown']} unknown (no data).")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    import sys
    sys.exit(main())
