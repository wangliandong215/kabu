"""
kabu — moomoo quantitative trading
Shared utilities: connection helpers, enum parsing, safe value extraction.
"""
import functools
import socket
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import config


def utc_now_iso() -> str:
    """Current time as UTC ISO 8601 (e.g. '2026-08-30T02:15:30.123456+00:00').

    The server's OS timezone is JST (confirmed 2026-08-30 — Windows
    TimeZoneId 'Tokyo Standard Time'), so a bare `datetime.now().isoformat()`
    silently returns JST wall-clock time with no timezone marker attached.
    Research tables use this for their `observed_at` column instead, so
    "when was this actually fetched" is unambiguous regardless of what
    timezone the server ever runs in."""
    return datetime.now(timezone.utc).isoformat()


def _check_opend(host: str, port: int) -> None:
    """Exit with a clear message if OpenD is not reachable."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        sock.connect((host, port))
    except (ConnectionRefusedError, OSError) as e:
        print(f"[ERROR] Cannot connect to OpenD ({host}:{port}): {e}")
        print("Please start moomoo OpenD and log in, then retry.")
        sys.exit(1)
    finally:
        sock.close()


def make_quote_ctx():
    """Create and return an OpenQuoteContext."""
    from moomoo import OpenQuoteContext
    _check_opend(config.OPEND_HOST, config.OPEND_PORT)
    return OpenQuoteContext(host=config.OPEND_HOST, port=config.OPEND_PORT)


def make_trade_ctx(market: Optional[str] = None):
    """Create and return an OpenSecTradeContext for the given market."""
    from moomoo import OpenSecTradeContext, TrdMarket, SecurityFirm
    _check_opend(config.OPEND_HOST, config.OPEND_PORT)

    market_map = {
        "US": TrdMarket.US, "HK": TrdMarket.HK,
        "SH": TrdMarket.CN, "SZ": TrdMarket.CN,
        "CN": TrdMarket.CN, "SG": TrdMarket.SG,
    }
    trd_market = market_map.get((market or "").upper(), TrdMarket.NONE)

    firm_map = {
        "FUTUSECURITIES": SecurityFirm.FUTUSECURITIES,
        "FUTUINC":        SecurityFirm.FUTUINC,
        "FUTUSG":         SecurityFirm.FUTUSG,
        "FUTUAU":         SecurityFirm.FUTUAU,
        "FUTUJP":         SecurityFirm.FUTUJP,
        "FUTUMY":         SecurityFirm.FUTUMY,
        "FUTUCA":         SecurityFirm.FUTUCA,
    }
    firm = firm_map.get(config.SECURITY_FIRM, SecurityFirm.FUTUSECURITIES)

    return OpenSecTradeContext(
        filter_trdmarket=trd_market,
        host=config.OPEND_HOST,
        port=config.OPEND_PORT,
        security_firm=firm,
    )


def safe_close(ctx) -> None:
    try:
        if ctx:
            ctx.close()
    except Exception:
        pass


def infer_market(code: str) -> str:
    """'US.AAPL' → 'US', 'HK.00700' → 'HK', etc."""
    if not code or "." not in code:
        return "US"
    return code.split(".")[0].upper()


def parse_trd_env():
    """Return TrdEnv based on config.TRD_ENV."""
    from moomoo import TrdEnv
    return TrdEnv.REAL if config.TRD_ENV == "REAL" else TrdEnv.SIMULATE


def retry(attempts: int = None, delay: float = None, backoff: float = None,
          exceptions=(Exception,)):
    """Decorator: retry the wrapped function on exception, sleeping `delay`
    seconds (multiplied by `backoff` after every failed attempt) before the
    next try. Re-raises the last exception once `attempts` is exhausted.
    Defaults come from config.API_RETRY_* so retry behavior is tunable
    without touching call sites — meant for moomoo OpenD API calls, where a
    transient network hiccup shouldn't fail a whole scan pass."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            n = attempts if attempts is not None else config.API_RETRY_ATTEMPTS
            wait = delay if delay is not None else config.API_RETRY_DELAY_SECONDS
            mult = backoff if backoff is not None else config.API_RETRY_BACKOFF
            last_exc = None
            for attempt in range(1, n + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == n:
                        break
                    import notify.alert as alert
                    alert.log(
                        f"{fn.__name__}: 第{attempt}/{n}次尝试失败（{exc}），"
                        f"{wait:.0f}秒后重试"
                    )
                    time.sleep(wait)
                    wait *= mult
            raise last_exc
        return wrapper
    return decorator
