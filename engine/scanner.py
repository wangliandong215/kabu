"""
engine/scanner.py — batch signal scan across a watchlist.

Usage:
    from engine.scanner import scan, smart_scan, rank_signals

    # Fixed strategy for all stocks:
    results = scan(["US.AAPL", "US.TSLA"], strategy_name="boll")

    # Auto-route strategy per stock based on market regime:
    results = smart_scan(["US.AAPL", "US.TSLA"])
"""
import time
from typing import Dict, List, Optional

import config
import notify.alert as alert
from data_provider.provider_factory import get_provider
from engine import router as regime_router
from engine.indicators import rsi_last
from strategies import get_strategy

_REQUEST_DELAY = 0.8  # seconds between API calls — moomoo limit: 60 req/30s
_provider = get_provider(config.MARKET)


def _strip_live_bar(df, ktype: str, code: str):
    """
    For K_DAY strategies, the last bar is live (unconfirmed) while the market
    is open.  Using it for indicator computation causes intraday flickering:
    %B / ADX values change bar by bar and may trigger false buy/sell signals
    that vanish once the day closes.

    Fix: when ktype=K_DAY and the stock's market is currently open, remove the
    last row before feeding data to the strategy.  The live price is fetched
    separately via get_price() and injected as current_price.
    """
    if ktype != "K_DAY" or len(df) < 2:
        return df
    from engine.market_hours import is_open
    if is_open(code):
        return df.iloc[:-1].copy()
    return df


def scan(
    codes: List[str],
    strategy_name: str = "combined",
    ktype: str = "K_DAY",
    bars: int = 120,
) -> Dict[str, dict]:
    """
    Run the given strategy on each code.

    Returns a dict mapping code → result dict from strategy.full_result().
    Codes that fail to fetch or compute are skipped (logged, not raised).
    A "current_price" key is injected into each result.
    """
    strategy = get_strategy(strategy_name)
    results: Dict[str, dict] = {}
    total = len(codes)

    for i, code in enumerate(codes, 1):
        try:
            df = _provider.get_history(code, interval=ktype, limit=bars)
            if df is None or len(df) < strategy.required_bars:
                alert.log(
                    f"scan [{i}/{total}]: {code} too few bars "
                    f"({0 if df is None else len(df)} < {strategy.required_bars}), skipped"
                )
                continue

            # Use only confirmed (closed) bars for indicator computation.
            # If the market is open, the last K_DAY bar is still live and causes
            # intraday flickering.  Current price is fetched separately.
            df_confirmed = _strip_live_bar(df, ktype, code)
            if len(df_confirmed) < strategy.required_bars:
                alert.log(f"scan [{i}/{total}]: {code} insufficient confirmed bars, skipped")
                continue

            result = strategy.full_result(df_confirmed)
            price = _provider.get_latest_price(code)
            result["current_price"] = price
            result["rsi14"] = rsi_last(df_confirmed["close"].astype(float))
            results[code] = result

            alert.log(
                f"scan [{i}/{total}]: {code:20s}  signal={result.get('signal','?'):4s}"
                f"  strength={result.get('signal_strength', 0):.0%}"
                f"  price={price:.4f}"
            )
        except Exception as exc:
            alert.error(f"扫描 {code}（第{i}/{total}只）失败 — {exc}")

        if i < total:
            time.sleep(_REQUEST_DELAY)

    return results


def smart_scan(
    codes: List[str],
    ktype: str = "K_DAY",
    bars: int = 120,
) -> Dict[str, dict]:
    """
    Auto-routing scan: detect market regime per stock, select the best
    strategy for that regime, then compute the signal.

    Each result dict includes extra keys:
      regime          (str)  detected regime label
      strategy_used   (str)  strategy that was applied
    """
    results: Dict[str, dict] = {}
    total = len(codes)

    for i, code in enumerate(codes, 1):
        try:
            df = _provider.get_history(code, interval=ktype, limit=bars)
            if df is None or len(df) < 30:
                alert.log(f"smart_scan [{i}/{total}]: {code} too few bars, skipped")
                continue

            # Strip live bar before regime + strategy computation (see _strip_live_bar).
            df_confirmed = _strip_live_bar(df, ktype, code)

            # Regime detection + strategy routing
            from engine.regime import detect, regime_label
            regime        = detect(df_confirmed)
            strategy_name = config.REGIME_STRATEGY_MAP.get(regime)

            if strategy_name is None:
                alert.log(
                    f"smart_scan [{i}/{total}]: {code:20s}"
                    f"  regime={regime_label(regime):12s}  -> SKIP"
                )
                results[code] = {
                    "signal": "HOLD", "signal_strength": 0.0,
                    "regime": regime, "strategy_used": None,
                    "detail": f"Regime={regime_label(regime)} — no long entry",
                    "current_price": _provider.get_latest_price(code),
                }
                continue

            strategy = get_strategy(strategy_name)
            if len(df_confirmed) < strategy.required_bars:
                alert.log(
                    f"smart_scan [{i}/{total}]: {code} insufficient bars "
                    f"for {strategy_name}, skipped"
                )
                continue

            result = strategy.full_result(df_confirmed)
            price  = _provider.get_latest_price(code)
            result["current_price"]  = price
            result["regime"]         = regime
            result["strategy_used"]  = strategy_name
            result["rsi14"]          = rsi_last(df_confirmed["close"].astype(float))
            results[code] = result

            alert.log(
                f"smart_scan [{i}/{total}]: {code:20s}"
                f"  regime={regime_label(regime):12s}"
                f"  strategy={strategy_name:12s}"
                f"  signal={result.get('signal','?'):4s}"
                f"  strength={result.get('signal_strength', 0):.0%}"
                f"  price={price:.4f}"
            )

        except Exception as exc:
            alert.error(f"智能扫描 {code}（第{i}/{total}只）失败 — {exc}")

        if i < total:
            time.sleep(_REQUEST_DELAY)

    return results


def rank_signals(
    results: Dict[str, dict],
    signal: str = "BUY",
    top_n: int = None,
) -> List[dict]:
    """
    Filter results by signal type and rank by signal_strength descending.

    Returns a list of dicts sorted strongest-first:
      [{"code": "US.NVDA", "signal": "BUY", "signal_strength": 0.75, ...}, ...]
    """
    filtered = [
        {"code": code, **data}
        for code, data in results.items()
        if data.get("signal") == signal
    ]
    filtered.sort(key=lambda x: x.get("signal_strength", 0), reverse=True)
    return filtered[:top_n] if top_n else filtered
