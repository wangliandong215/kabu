"""Ad-hoc: close every non-core_etf position (real SIMULATE sell orders), keep QQQ core untouched."""
import moomoo as ft

from common import make_trade_ctx, safe_close, infer_market
from data.fetcher import get_price
from portfolio.tracker import Portfolio

p = Portfolio()
codes = [c for c, pos in p.data["positions"].items() if pos.get("strategy") != "core_etf"]
print(f"closing {len(codes)} position(s): {codes}")

for code in sorted(codes):
    pos = p.get_position(code)
    qty = pos["qty"]
    price = get_price(code)
    if price <= 0:
        print(f"{code}: SKIP — no price")
        continue

    market = infer_market(code)
    trd_ctx = make_trade_ctx(market)
    order_id = ""
    try:
        ret, data = trd_ctx.place_order(
            price=price, qty=qty, code=code,
            trd_side=ft.TrdSide.SELL, order_type=ft.OrderType.NORMAL,
            trd_env=ft.TrdEnv.SIMULATE,
        )
        if ret == ft.RET_OK:
            order_id = str(data["order_id"].iloc[0])
        else:
            print(f"{code}: ORDER FAILED — {data}")
    except Exception as exc:
        print(f"{code}: EXCEPTION — {exc}")
    finally:
        safe_close(trd_ctx)

    if order_id:
        closed = p.close_position(code, price, reason="MANUAL_RESET")
        print(f"{code}: SOLD qty={qty} @ {price:.4f}  pnl={closed['pnl']:+.2f}  order_id={order_id}")

print()
print("remaining positions:", list(p.data["positions"].keys()))
print("realized_pnl:", round(p.data["realized_pnl"], 2))
print("total_capital:", round(p.total_capital(), 2))
print("deployed_capital:", round(p.deployed_capital(), 2))
print("exposure_pct:", round(p.exposure_pct() * 100, 2), "%")
