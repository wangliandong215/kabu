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

import config
from engine.runner import run_once, run_loop


def cli_main() -> None:
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

    codes = args.codes or config.WATCHLIST

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
