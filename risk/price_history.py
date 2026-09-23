"""
risk/price_history.py — the ONLY module beta_risk.py/var_risk.py/
correlation_risk.py's network access goes through.

Wraps data/fetcher.py::fetch_kline() (moomoo OpenD daily klines, disk-cached)
to build a code -> daily-return-Series map, once per run_once() pass, cached
onto risk/portfolio_state.py::PortfolioState.price_returns so a 20-position
pass fetches each symbol's klines at most once rather than once per order.

Every other risk/*_risk.py module takes an already-built returns dict as a
parameter and is otherwise a pure function — this isolates the one live
network dependency so unit tests never need to mock moomoo/OpenD, they just
construct a PortfolioState with price_returns already populated.

A symbol's fetch failing (network error, delisted, insufficient history) is
swallowed here and that symbol is simply absent from the returned dict —
callers treat "missing key" as DATA_UNAVAILABLE for that symbol, never as a
0 return series.
"""
from typing import Dict, Iterable, Optional

import pandas as pd

import config
from data.fetcher import fetch_kline


def fetch_price_returns(codes: Iterable[str], lookback_days: Optional[int] = None) -> Dict[str, pd.Series]:
    """Best-effort. Returns {code: daily_return_series}; codes whose kline
    fetch fails or comes back too short are simply omitted."""
    lookback_days = lookback_days or config.RISK_ENGINE_VAR_LOOKBACK_DAYS
    out: Dict[str, pd.Series] = {}
    for code in dict.fromkeys(codes):   # de-dupe, preserve order
        try:
            df = fetch_kline(code, "1d", bars=lookback_days)
            if df is None or len(df) < 2 or "close" not in df.columns:
                continue
            closes = pd.to_numeric(df["close"], errors="coerce").dropna()
            if len(closes) < 2:
                continue
            returns = closes.pct_change().dropna()
            if len(returns) == 0:
                continue
            out[code] = returns.reset_index(drop=True)
        except Exception:
            continue
    return out
