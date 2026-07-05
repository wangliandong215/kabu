"""
notify/alert.py — structured console logging + optional DingTalk push.

Set env var KABU_DINGTALK_WEBHOOK (or config.DINGTALK_WEBHOOK) to receive
DingTalk alerts.  All messages are always printed to stdout regardless.
"""
from datetime import datetime

import config


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _emit(level: str, msg: str) -> None:
    print(f"[{_ts()}] [{level:5s}] {msg}")
    _push_dingtalk(f"[{level}] {msg}")


def _push_dingtalk(msg: str) -> None:
    webhook = getattr(config, "DINGTALK_WEBHOOK", "")
    if not webhook:
        return
    try:
        import requests
        requests.post(
            webhook,
            json={"msgtype": "text", "text": {"content": f"[kabu] {msg}"}},
            timeout=5,
        )
    except Exception:
        pass


# ── Public helpers ────────────────────────────────────────────────────────────

def info(msg: str)  -> None: _emit("INFO",  msg)
def warn(msg: str)  -> None: _emit("WARN",  msg)
def error(msg: str) -> None: _emit("ERROR", msg)


def signal(code: str, side: str, price: float, qty: int,
           order_id: str = "", env: str = "SIMULATE") -> None:
    """Log (and push) a trade execution event."""
    icon = "BUY +" if side == "BUY" else "SELL-"
    _emit("TRADE", f"{icon} {qty}×{code} @ {price:.4f}  order={order_id}  [{env}]")
