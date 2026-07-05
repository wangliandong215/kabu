"""
kabu — moomoo quantitative trading
Shared utilities: connection helpers, enum parsing, safe value extraction.
"""
import socket
import sys
from typing import Optional

import config


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
