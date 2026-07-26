"""
engine/broker.py — Broker abstraction layer.

get_broker(code) is the single seam that decides how an order for `code`
actually gets executed:
  MoomooBroker — real order placement via moomoo OpenD (US/EU-listed codes).
  PaperBroker  — no broker call at all; simulates an instant fill (JP codes,
                 since moomoo does not support JP auto-trading today — see
                 common.py::make_trade_ctx()'s market_map, which has no "JP"
                 entry).

Strategy/risk/sizing/analytics code never talks to either class directly —
it only ever calls engine.runner._place_order(), which resolves the broker
via get_broker() and delegates. When moomoo eventually supports real JP
execution, swapping PaperBroker for a real JPBroker here is the only change
needed anywhere in the system.
"""
import uuid

import notify.alert as alert
from common import infer_market, make_trade_ctx, safe_close


class MoomooBroker:
    """Real order placement via moomoo OpenD. Behavior moved verbatim from
    the pre-v2.9 engine/runner.py::_place_order() body — no logic change."""

    @staticmethod
    def place_order(code: str, side: str, qty: int, price: float,
                     trd_env, env_label: str, confirmed: bool) -> str:
        if not confirmed:
            alert.log(f"[DRY RUN] would {side} {qty}×{code} @ {price:.4f}")
            return ""

        import moomoo as ft

        market = infer_market(code)
        # moomoo 要求美股价格精确到分（最小 tick = $0.01）
        if code.startswith("US."):
            price = round(price, 2)
        trd_ctx = make_trade_ctx(market)
        order_id = ""
        try:
            order_type = ft.OrderType.NORMAL
            trd_side = ft.TrdSide.BUY if side == "BUY" else ft.TrdSide.SELL

            ret, data = trd_ctx.place_order(
                price=price,
                qty=qty,
                code=code,
                trd_side=trd_side,
                order_type=order_type,
                trd_env=trd_env,
            )
            if ret == ft.RET_OK:
                order_id = str(data["order_id"].iloc[0])
            else:
                side_cn = "买入" if side == "BUY" else "卖出"
                alert.error(f"{code} {side_cn}下单失败 — {data}")
        except Exception as exc:
            alert.error(f"{code} 下单时发生异常 — {exc}")
        finally:
            safe_close(trd_ctx)

        return order_id


class PaperBroker:
    """Virtual fill — never calls moomoo. Always "succeeds" (no rejection
    modeling for v1). Used for markets moomoo doesn't support real trading
    on yet (JP today; see get_broker())."""

    @staticmethod
    def place_order(code: str, side: str, qty: int, price: float,
                     trd_env, env_label: str, confirmed: bool) -> str:
        if not confirmed:
            alert.log(f"[DRY RUN PAPER] would {side} {qty}×{code} @ {price:.4f}")
            return ""

        order_id = f"PAPER-{uuid.uuid4().hex[:10]}"
        alert.log(f"[PAPER FILL] {side} {qty}×{code} @ {price:.4f}  order_id={order_id}")
        return order_id


def get_broker(code: str):
    """Return the broker class to use for `code`. Single seam for future
    market/broker additions (real JP execution, HK, IBKR, ... per the
    "Future Upgrade" design)."""
    if infer_market(code) == "JP":
        return PaperBroker
    return MoomooBroker
