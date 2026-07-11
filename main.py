"""
kabu — moomoo quantitative trading
main.py — command-line entry point.

Runs the full scan → risk → size → order pipeline on all watchlist codes.

Usage:
  # One-shot dry run (default: SIMULATE, no orders placed)
  python main.py

  # One-shot simulation pass (actually place simulated orders)
  python main.py --confirmed

  # Loop every 5 minutes in simulation
  python main.py --confirmed --interval 300

  # One-shot live preview (no orders submitted)
  python main.py --real

  # Live execution (requires --real AND --confirmed)
  python main.py --real --confirmed

  # Use a specific strategy and custom code list
  python main.py --strategy ma --codes US.AAPL US.TSLA

  # Auto-route strategy per stock based on market regime (recommended)
  python main.py --auto --confirmed --interval 300

Safety:
  - Trade unlock must be done MANUALLY in OpenD GUI — SDK unlock_trade is NEVER called.
  - Live orders require BOTH --real AND --confirmed.
  - Default is always SIMULATE.
"""
import argparse
import ctypes
import os
import sys
from pathlib import Path

import config
from engine.runner import run_once, run_loop
from notify import alert, telegram_bot

# Keep a reference alive so the ctypes callback isn't garbage-collected —
# SetConsoleCtrlHandler only stores a raw function pointer.
_shutdown_handler_ref = None

_LOCK_PATH = Path(__file__).resolve().parent / ".kabu_loop.lock"


def _is_pid_running(pid: int) -> bool:
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle:
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    return False


def _acquire_single_instance_lock() -> None:
    """Refuse to start a second concurrent trading loop against the same
    portfolio/positions.json — two live instances placing orders in
    parallel is a real risk (e.g. Windows' "restart apps after sign-in"
    reviving a stale process with an old command line). Whichever
    instance is already running keeps the lock; a new launch just exits."""
    if _LOCK_PATH.exists():
        try:
            old_pid = int(_LOCK_PATH.read_text().strip())
        except (ValueError, OSError):
            old_pid = None
        if old_pid and old_pid != os.getpid() and _is_pid_running(old_pid):
            alert.error(
                f"已有一个kabu监控在运行（pid={old_pid}），"
                f"本次启动被拒绝，避免重复下单"
            )
            sys.exit(1)
    _LOCK_PATH.write_text(str(os.getpid()))


def _register_shutdown_notifier() -> None:
    """Best-effort Telegram/DingTalk alert when Windows is shutting down,
    logging off, or closing this console — covers the process being killed
    by the OS (e.g. a scheduled `shutdown` task) rather than stopped
    gracefully with Ctrl+C, which engine/runner.py's loop already reports
    via alert.info("kabu 监控已手动停止")."""
    if sys.platform != "win32":
        return
    global _shutdown_handler_ref

    HANDLER_ROUTINE = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)
    EVENT_NAMES = {2: "控制台被关闭", 5: "用户注销", 6: "系统关机"}

    def _handler(ctrl_type: int) -> int:
        name = EVENT_NAMES.get(ctrl_type)
        if name:
            alert.warn(f"kabu 监控即将停止 — 原因：{name}")
        return 0  # not handled — let the OS proceed with default shutdown

    _shutdown_handler_ref = HANDLER_ROUTINE(_handler)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_shutdown_handler_ref, True)


def cli_main() -> None:
    _acquire_single_instance_lock()
    _register_shutdown_notifier()
    telegram_bot.start()
    p = argparse.ArgumentParser(
        description="kabu pipeline runner (moomoo OpenD)",
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --confirmed --interval 300\n"
            "  python main.py --real --confirmed --strategy combined\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--strategy",
        choices=["ma", "rsi", "macd", "boll", "combined", "atr_breakout",
                 "ema_rsi", "ma_rsi"],
        default="combined",
        help="Signal strategy (default: combined)",
    )
    p.add_argument(
        "--ktype",
        choices=["K_1M", "K_5M", "K_15M", "K_30M", "K_60M", "K_DAY", "K_WEEK", "K_MON"],
        default="K_DAY",
    )
    p.add_argument("--bars",     type=int, default=120,
                   help="K-line bars to load per code (default: 120)")
    p.add_argument("--codes",    nargs="*", default=None,
                   help="Stock codes to scan (default: config.WATCHLIST)")
    p.add_argument("--interval", type=int, default=0,
                   help="Loop interval in seconds (0 = run once, then exit)")
    p.add_argument("--real",      action="store_true",
                   help="Use live account (default: SIMULATE)")
    p.add_argument("--confirmed", action="store_true",
                   help="Place actual orders (without this flag, dry-run only)")
    p.add_argument("--auto",      action="store_true",
                   help="Auto-route strategy per stock via market regime detection")
    args = p.parse_args()

    # None (not config.WATCHLIST) when --codes is omitted: run_once()/select_watchlist()
    # then pick EU/US vs Asia-Pacific dynamically each pass based on JST time of day.
    codes = args.codes

    if args.interval > 0:
        run_loop(
            strategy_name=args.strategy,
            codes=codes,
            ktype=args.ktype,
            bars=args.bars,
            interval_seconds=args.interval,
            confirmed=args.confirmed,
            use_real=args.real,
            auto_route=args.auto,
        )
    else:
        run_once(
            strategy_name=args.strategy,
            codes=codes,
            ktype=args.ktype,
            bars=args.bars,
            confirmed=args.confirmed,
            use_real=args.real,
            auto_route=args.auto,
        )


if __name__ == "__main__":
    cli_main()
