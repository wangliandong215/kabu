"""
data/fetcher.py — K-line fetching with disk cache.

Cache location: config.CACHE_DIR  (~/.kabu_cache by default)
Cache TTL:      config.CACHE_TTL_DAILY (daily+ bars)  /  config.CACHE_TTL_INTRADAY (sub-day)

The cache stores the full lookback DataFrame per (code, ktype).
A stale entry is re-fetched transparently.
"""
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

import common
import config

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(code: str, ktype: str) -> Path:
    safe = code.replace(".", "_")
    return Path(config.CACHE_DIR) / f"{safe}_{ktype}.pkl"


def _ttl(ktype: str) -> int:
    return config.CACHE_TTL_DAILY if ktype in ("1d", "1w", "1M") else config.CACHE_TTL_INTRADAY


def _fresh(path: Path, ktype: str) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < _ttl(ktype)


def _load(path: Path) -> Optional[pd.DataFrame]:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save(path: Path, df: pd.DataFrame) -> None:
    Path(config.CACHE_DIR).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(df, f)


# ── K-line type map ───────────────────────────────────────────────────────────

def _ktype_enum(ktype: str):
    from moomoo import KLType
    return {
        "1m": KLType.K_1M,  "5m": KLType.K_5M,
        "15m": KLType.K_15M, "30m": KLType.K_30M,
        "60m": KLType.K_60M, "1d": KLType.K_DAY,
        "1w": KLType.K_WEEK, "1M": KLType.K_MON,
    }.get(ktype, KLType.K_DAY)


_LOOKBACK_DAYS = {
    "1m": 5,  "5m": 10,  "15m": 20, "30m": 40,
    "60m": 60, "1d": 400, "1w": 800, "1M": 1200,
}

_DEFAULT_BARS = {
    "1m": 200, "5m": 200, "15m": 200, "30m": 200,
    "60m": 200, "1d": 250, "1w": 104, "1M": 36,
}


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_kline(code: str, ktype: str = "1d", bars: int = None) -> pd.DataFrame:
    """
    Return a forward-adjusted historical K-line DataFrame.
    Uses disk cache; re-fetches when cache is stale.
    """
    if bars is None:
        bars = _DEFAULT_BARS.get(ktype, 250)

    path = _cache_path(code, ktype)

    if _fresh(path, ktype):
        cached = _load(path)
        if cached is not None and len(cached) >= bars:
            return cached.tail(bars).reset_index(drop=True)

    df = _fetch_from_opend(code, ktype)
    _save(path, df)
    return df.tail(bars).reset_index(drop=True)


def get_price(code: str) -> float:
    """Return current price (live snapshot, falls back to last daily close)."""
    from moomoo import RET_OK
    ctx = common.make_quote_ctx()
    try:
        ret, data = ctx.get_market_snapshot([code])
        if ret == RET_OK and data is not None and len(data):
            for field in ("last_price", "cur_price"):
                val = data.iloc[0].get(field)
                if val and float(val) > 0:
                    return float(val)
    except Exception:
        pass
    finally:
        common.safe_close(ctx)

    df = fetch_kline(code, "1d", 2)
    return float(df.iloc[-1]["close"]) if df is not None and len(df) else 0.0


# ── Internal fetch ────────────────────────────────────────────────────────────

@common.retry()
def _fetch_from_opend(code: str, ktype: str) -> pd.DataFrame:
    from moomoo import AuType, RET_OK

    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=_LOOKBACK_DAYS.get(ktype, 400))).strftime("%Y-%m-%d")

    ctx = common.make_quote_ctx()
    frames = []
    try:
        ret, data, page_key = ctx.request_history_kline(
            code, start=start, end=end,
            ktype=_ktype_enum(ktype), autype=AuType.QFQ, max_count=1000,
        )
        if ret != RET_OK:
            raise RuntimeError(f"fetch_kline({code}): {data}")
        frames.append(data)

        page = 1
        while page_key is not None and page < 10:
            ret, data, page_key = ctx.request_history_kline(
                code, start=start, end=end,
                ktype=_ktype_enum(ktype), autype=AuType.QFQ,
                max_count=1000, page_req_key=page_key,
            )
            if ret == RET_OK and data is not None and len(data):
                frames.append(data)
            page += 1
    finally:
        common.safe_close(ctx)

    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
