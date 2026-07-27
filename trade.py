"""
kabu — moomoo quantitative trading
trade.py — run analysis on a single stock and optionally place an order.

Safety rules (hard-coded):
  1. Default env = SIMULATE (paper trading).  Live trading requires --real flag.
  2. Live trading also requires --confirmed to actually submit the order
     (first run shows a preview; re-run with --confirmed to execute).
  3. Trade unlock must be done MANUALLY in the OpenD GUI — this script
     never calls unlock_trade() via the SDK.

Typical workflow:
  # 1. Analyse and preview (simulation, no order placed)
  python trade.py US.AAPL

  # 2. Force a direction and preview in simulation
  python trade.py US.AAPL --side BUY

  # 3. Execute in simulation
  python trade.py US.AAPL --confirmed

  # 4. Live preview (shows order details, does NOT submit)
  python trade.py US.AAPL --real

  # 5. Live execution (submits the order — ensure trade is unlocked in OpenD)
  python trade.py US.AAPL --real --confirmed
"""
import argparse
import json
import sys
from datetime import datetime

import common
import config
from data_provider.provider_factory import get_provider
from strategies import get_strategy

_provider = get_provider(config.MARKET)


# ── Audit log ─────────────────────────────────────────────────────────────────

def _audit(entry: dict) -> None:
    import os
    path = os.path.join(os.path.dirname(__file__), "trade_audit.jsonl")
    entry["timestamp"] = datetime.now().isoformat()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ── Core logic ────────────────────────────────────────────────────────────────

def run_trade(
    code: str,
    strategy_name: str = "combined",
    ktype: str = "K_DAY",
    bars: int = 120,
    side: str = None,
    qty: int = None,
    order_type: str = "LIMIT",
    use_real: bool = False,
    confirmed: bool = False,
    output_json: bool = False,
) -> None:

    # ── Step 1: Run analysis ──────────────────────────────────────────────────
    strategy = get_strategy(strategy_name)
    df = _provider.get_history(code, interval=ktype, limit=bars)
    if df is None or len(df) == 0:
        print(f"Error: no data returned for {code}")
        sys.exit(1)

    result = strategy.full_result(df)
    overall   = result.get("signal", "HOLD")
    strength  = result.get("signal_strength", 0.0)
    buy_n     = result.get("buy_votes", 0)
    sell_n    = result.get("sell_votes", 0)

    # ── Step 2: Determine trade direction ─────────────────────────────────────
    direction = side.upper() if side else overall
    if direction not in ("BUY", "SELL", "HOLD"):
        print(f"Error: --side must be BUY or SELL (got {side})")
        sys.exit(1)

    if direction == "HOLD":
        msg = (f"Signal is HOLD (strength={strength:.0%}, "
               f"BUY votes={buy_n}, SELL votes={sell_n}). No order placed.")
        if output_json:
            print(json.dumps({"code": code, "signal": "HOLD",
                               "action": "no_order", "message": msg},
                              ensure_ascii=False))
        else:
            print(f"\n{msg}")
        return

    # ── Step 3: Fetch current price and compute qty ───────────────────────────
    price = _provider.get_latest_price(code)
    if price <= 0:
        price = float(df.iloc[-1]["close"])

    if qty is None:
        from risk.sizing import calculate
        from portfolio.tracker import Portfolio
        qty = calculate(Portfolio().available_cash(), price)
    if qty <= 0:
        print(f"Error: computed qty=0 at price={price:.4f}, set --qty manually")
        sys.exit(1)

    # ── Step 4: Build preview ─────────────────────────────────────────────────
    trd_env_label = "REAL" if use_real else "SIMULATE"
    sub_sigs = {k: v.get("signal") for k, v in (result.get("sub_indicators") or {}).items()}

    preview = {
        "code":             code,
        "side":             direction,
        "quantity":         qty,
        "price":            round(price, 4),
        "order_type":       order_type,
        "trd_env":          trd_env_label,
        "signal":           overall,
        "signal_strength":  round(strength, 2),
        "buy_votes":        buy_n,
        "sell_votes":       sell_n,
        "sub_indicators":   sub_sigs,
    }

    # ── Step 5: Show analysis summary ─────────────────────────────────────────
    if not output_json:
        print("=" * 70)
        print(f"Analysis Signal: {overall}  (strength {strength:.0%})")
        for name, ind in (result.get("sub_indicators") or {}).items():
            print(f"  {name.upper():<8} [{ind.get('signal','HOLD')}]  {ind.get('detail','')}")
        print()
        print("Order Preview:")
        print(f"  Code:        {code}")
        print(f"  Side:        {direction}")
        print(f"  Quantity:    {qty}")
        print(f"  Price:       {price:.4f}  ({order_type})")
        print(f"  Environment: {trd_env_label}")
        print("=" * 70)

    if use_real and not confirmed:
        msg = "Live trading preview — re-run with --confirmed to submit."
        if output_json:
            preview.update({"status": "preview_only", "message": msg})
            print(json.dumps(preview, ensure_ascii=False, indent=2))
        else:
            print(f"\n[WARNING] {msg}")
        _audit({"action": "preview", **preview})
        sys.exit(2)

    if not confirmed:
        msg = "Dry run — add --confirmed to place the order."
        if output_json:
            preview.update({"status": "dry_run", "message": msg})
            print(json.dumps(preview, ensure_ascii=False, indent=2))
        else:
            print(f"\n{msg}")
        return

    # ── Step 6: Place the order ───────────────────────────────────────────────
    from moomoo import RET_OK, TrdSide, OrderType, TrdEnv

    trd_env  = TrdEnv.REAL if use_real else TrdEnv.SIMULATE
    trd_side = TrdSide.BUY if direction == "BUY" else TrdSide.SELL
    otype    = OrderType.MARKET if order_type.upper() == "MARKET" else OrderType.NORMAL
    order_price = 0.0 if otype == OrderType.MARKET else float(price)

    market  = common.infer_market(code)
    trd_ctx = common.make_trade_ctx(market)
    try:
        ret, data = trd_ctx.place_order(
            price=order_price,
            qty=int(qty),
            code=code,
            trd_side=trd_side,
            order_type=otype,
            trd_env=trd_env,
        )
        if ret != RET_OK:
            err = str(data)
            _audit({"action": "place_order", "result": "error", **preview, "error": err})
            if "unlock" in err.lower():
                print("\n[ERROR] Trade is not unlocked.")
                print("Click 'Unlock Trade' in the OpenD GUI and enter your password.")
            else:
                print(f"\n[ERROR] Order failed: {err}")
            sys.exit(1)

        order_id = str(data.iloc[0].get("order_id", data)) if hasattr(data, "iloc") else str(data)
        out = {**preview, "status": "submitted", "order_id": order_id}
        _audit({"action": "place_order", "result": "success", **out})

        if output_json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"\nOrder submitted successfully")
            print(f"  Order ID:  {order_id}")
            print(f"  {direction} {qty} x {code} @ {order_price:.4f}  [{trd_env_label}]")

    except Exception as exc:
        _audit({"action": "place_order", "result": "exception", **preview, "error": str(exc)})
        if output_json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            print(f"\n[ERROR] {exc}")
        sys.exit(1)
    finally:
        common.safe_close(trd_ctx)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Quantitative signal → order execution (moomoo OpenD)",
        epilog=(
            "Examples:\n"
            "  python trade.py US.AAPL\n"
            "  python trade.py US.AAPL --side BUY --confirmed\n"
            "  python trade.py HK.00700 --qty 100 --real --confirmed\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("code", help="Stock code, e.g. US.AAPL, HK.00700")
    p.add_argument(
        "--strategy",
        choices=["ma", "rsi", "macd", "boll", "combined"],
        default="combined",
    )
    p.add_argument(
        "--ktype",
        choices=["K_1M", "K_5M", "K_15M", "K_30M", "K_60M", "K_DAY", "K_WEEK", "K_MON"],
        default="K_DAY",
    )
    p.add_argument("--bars",       type=int,   default=120)
    p.add_argument("--side",       choices=["BUY", "SELL"], default=None)
    p.add_argument("--qty",        type=int,   default=None,
                   help="Shares to trade (default: auto-sized via risk.sizing)")
    p.add_argument("--order-type", choices=["LIMIT", "MARKET"], default="LIMIT")
    p.add_argument("--real",       action="store_true",
                   help="Use live account (default: simulation)")
    p.add_argument("--confirmed",  action="store_true",
                   help="Actually submit the order (without this flag, only preview is shown)")
    p.add_argument("--json",       action="store_true", dest="output_json")
    args = p.parse_args()

    run_trade(
        code=args.code,
        strategy_name=args.strategy,
        ktype=args.ktype,
        bars=args.bars,
        side=args.side,
        qty=args.qty,
        order_type=args.order_type,
        use_real=args.real,
        confirmed=args.confirmed,
        output_json=args.output_json,
    )
