"""
notify/alert.py — structured console logging + optional DingTalk / Telegram push.

Set env var KABU_DINGTALK_WEBHOOK (or config.DINGTALK_WEBHOOK) to receive
DingTalk alerts. Set KABU_TELEGRAM_BOT_TOKEN + KABU_TELEGRAM_CHAT_ID (or
config.TELEGRAM_BOT_TOKEN / config.TELEGRAM_CHAT_ID) to receive Telegram
alerts. All messages are always printed to stdout regardless.

Every message (including console-only log()/warn_skip() traffic) is also
written to a rotating file under config.LOG_DIR — the terminal/Telegram
history disappears when a console closes or a phone notification scrolls
away, but the file survives a reboot so past runs stay traceable.
"""
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path

import config

# Windows consoles default to the GBK/cp936 codepage, which can't encode the
# emoji used in trade_buy/trade_sell messages. Force UTF-8 stdout so those
# prints don't crash the process (best-effort — no-op on non-reconfigurable
# streams, e.g. some redirected/piped contexts).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

_logger = logging.getLogger("kabu")
_logger.setLevel(logging.INFO)
_logger.propagate = False
if not _logger.handlers:   # guard against duplicate handlers on module reload
    try:
        log_dir = Path(config.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        _handler = logging.handlers.RotatingFileHandler(
            log_dir / "kabu.log",
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        _handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        _logger.addHandler(_handler)
    except OSError:
        pass   # best-effort — console/push still work if the log file can't be created

_LEVEL_MAP = {"INFO": logging.INFO, "WARN": logging.WARNING, "ERROR": logging.ERROR}


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _emit(level: str, msg: str) -> None:
    # Console keeps the [LEVEL] tag for grepping log files; the phone push
    # drops [kabu]/[LEVEL] since msg is plain Chinese and self-explanatory.
    print(f"[{_ts()}] [{level:5s}] {msg}")
    _logger.log(_LEVEL_MAP.get(level, logging.INFO), msg)
    _push_all(msg, prefix="")


def _push_all(msg: str, prefix: str = "[kabu] ") -> None:
    _push_dingtalk(msg, prefix)
    _push_telegram(msg, prefix)


def push_raw(msg: str) -> None:
    """Push msg to DingTalk/Telegram with no [kabu]/[LEVEL] prefix at all —
    for callers that want a clean, self-contained message rather than the
    standard console-log-style formatting (e.g. watchdog.py, whose alerts
    are already self-explanatory without extra tagging)."""
    print(f"[{_ts()}] {msg}")
    _logger.info(msg)
    _push_all(msg, prefix="")


def _push_dingtalk(msg: str, prefix: str = "[kabu] ") -> None:
    webhook = getattr(config, "DINGTALK_WEBHOOK", "")
    if not webhook:
        return
    try:
        import requests
        requests.post(
            webhook,
            json={"msgtype": "text", "text": {"content": f"{prefix}{msg}"}},
            timeout=5,
        )
    except Exception:
        pass


def _push_telegram(msg: str, prefix: str = "[kabu] ") -> None:
    token = getattr(config, "TELEGRAM_BOT_TOKEN", "")
    chat_ids = getattr(config, "TELEGRAM_CHAT_IDS", [])
    if not token or not chat_ids:
        return
    import requests
    for chat_id in chat_ids:
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": f"{prefix}{msg}"},
                timeout=5,
            )
        except Exception:
            pass


# ── Public helpers ────────────────────────────────────────────────────────────

def info(msg: str)  -> None: _emit("INFO",  msg)
def warn(msg: str)  -> None: _emit("WARN",  msg)
def error(msg: str) -> None: _emit("ERROR", msg)


def log(msg: str) -> None:
    """Diagnostic/progress message — never pushed to DingTalk/Telegram, but
    still written to the log file (see module docstring) for traceability.
    Use for per-stock scan/routing chatter and routine status that isn't an
    actionable trading event (use info/warn/error for anything that should
    reach the phone)."""
    print(f"[{_ts()}] [LOG  ] {msg}")
    _logger.info(msg)


_skip_pushed_today: dict = {}  # code -> "YYYY-MM-DD" of last push


def warn_skip(code: str, msg: str) -> None:
    """Buy-signal skip notice: printed to console every scan pass, but pushed
    to DingTalk/Telegram at most once per calendar day per stock code — a
    persistent skip reason (capacity full, earnings blackout, sector limit,
    ...) would otherwise repush every --interval scan (e.g. every 5 minutes)
    for the same code."""
    print(f"[{_ts()}] [WARN ] {msg}")
    _logger.warning(msg)
    today = datetime.now().strftime("%Y-%m-%d")
    if _skip_pushed_today.get(code) == today:
        return
    _skip_pushed_today[code] = today
    _push_all(msg, prefix="")


# ── Exit reason display mapping ───────────────────────────────────────────────

_EXIT_REASON_LABELS = {
    "STOP_LOSS":          "固定止损",
    "ATR_TRAIL":          "ATR止损",
    "TAKE_PROFIT":        "止盈",
    "MA200_BREAK":        "风控退出(QQQ跌破MA200)",
    "ACTIVE_REPLACEMENT": "风控退出(仓位置换)",
}


def _format_exit_reason(reason: str) -> str:
    if reason.startswith("STRATEGY_EXIT("):
        strategy_name = reason[len("STRATEGY_EXIT("):-1]
        return f"策略退出({strategy_name})"
    return _EXIT_REASON_LABELS.get(reason, reason or "其它")


# ── Trade notifications ───────────────────────────────────────────────────────

def trade_buy(code: str, price: float, qty: int, sector: str,
              score, score_label, stop_price, position_pct: float,
              cash_available: float, position_count: int, trade_id: int,
              env: str = "SIMULATE") -> None:
    """Rich buy-fill notification (console + DingTalk + Telegram)."""
    name = config.NAME_MAP.get(code, code)
    notional = price * qty
    score_str = f"{score:.0f}" if score is not None else "N/A"
    stop_str = f"{stop_price:.2f}" if stop_price is not None else "N/A"
    msg = (
        f"🟢【买入成交】\n"
        f"📈 股票：{code}（{name}）\n"
        f"💲 买入价格：{price:.2f}\n"
        f"📦 买入数量：{qty} 股\n"
        f"💰 成交金额：{notional:,.2f}\n"
        f"📊 当前仓位：{position_pct:.2%}\n"
        f"⭐ 综合评分：{score_str}\n"
        f"🛑 初始止损：{stop_str}\n"
        f"🏷 所属板块：{sector}\n"
        f"💵 剩余可用资金：{cash_available:,.2f}\n"
        f"📈 当前持仓数量：{position_count}\n"
        f"🆔 交易编号：#{trade_id}\n"
        f"🕒 成交时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}  [{env}]"
    )
    print(msg)
    _logger.info(msg.replace("\n", " | "))
    _push_all(msg)


def trade_sell(code: str, entry_price: float, exit_price: float, qty: int,
               days_held: int, reason: str, cash_available: float,
               total_equity: float, position_count: int, trade_id: int,
               env: str = "SIMULATE") -> None:
    """Rich sell-fill notification (console + DingTalk + Telegram)."""
    name = config.NAME_MAP.get(code, code)
    pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0
    pnl_amt = (exit_price - entry_price) * qty
    msg = (
        f"🔴【卖出成交】\n"
        f"📈 股票：{code}（{name}）\n"
        f"💲 买入价格：{entry_price:.2f}\n"
        f"💲 卖出价格：{exit_price:.2f}\n"
        f"📈 收益率：{pnl_pct:+.2%}\n"
        f"💰 盈亏金额：{pnl_amt:+,.2f}\n"
        f"📅 持仓天数：{days_held} 天\n"
        f"📦 卖出数量：{qty} 股\n"
        f"📝 卖出原因：{_format_exit_reason(reason)}\n"
        f"💵 当前可用资金：{cash_available:,.2f}\n"
        f"💼 当前总资产：{total_equity:,.2f}\n"
        f"📊 当前持仓数量：{position_count}\n"
        f"🆔 交易编号：#{trade_id}\n"
        f"🕒 成交时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}  [{env}]"
    )
    print(msg)
    _logger.info(msg.replace("\n", " | "))
    _push_all(msg)
