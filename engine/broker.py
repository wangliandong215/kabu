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

place_order() returns a fill dict (not just an order_id) — see
MoomooBroker.place_order docstring. 2026-08-12: this replaced a version that
returned order_id alone and let callers treat "order accepted" as "order
filled". Orders here are DAY limit orders (OrderType.NORMAL); if the limit
price isn't touched that session the broker auto-cancels it hours later —
callers used to update positions.json/trade_history the instant order_id
came back, so the local book kept a position the broker never actually
opened/closed. Real incident: US.CTAS/US.MNST sell orders and
US.ROST/US.MSFT/US.SNOW/US.AMZN buy orders all sat CANCELLED_ALL with
dealt_qty=0 while local state recorded them as executed. Fixed by polling
the order to a terminal state (or cancelling it) before returning.
"""
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import notify.alert as alert
from common import infer_market, make_trade_ctx, safe_close
from engine import order_audit

_DRY_RUN_FILL = {"order_id": "", "dealt_qty": 0.0, "dealt_avg_price": 0.0, "status": "DRY_RUN"}


class MoomooBroker:
    """Real order placement via moomoo OpenD."""

    _POLL_ATTEMPTS = 6
    _POLL_INTERVAL_SEC = 1.5
    # Terminal states where dealt_qty is final and won't change further.
    # CANCELLED_PART (V3.6 P7.3) = part filled, remainder cancelled; it used
    # to be missing, so such an order was polled to timeout, cancelled a
    # second time and recorded as unconfirmed.
    _TERMINAL = {"FILLED_ALL", "CANCELLED_ALL", "CANCELLED_PART", "FAILED", "DISABLED",
                 "DELETED", "SUBMIT_FAILED", "FILL_CANCELLED"}

    @staticmethod
    def place_order(code: str, side: str, qty: int, price: float,
                     trd_env, env_label: str, confirmed: bool) -> dict:
        """Place a DAY limit order and wait (briefly) for it to resolve.

        Returns {"order_id": str, "dealt_qty": float, "dealt_avg_price": float,
        "status": str}. dealt_qty is the ONLY field callers should trust to
        decide whether to touch local portfolio state — order_id being
        non-empty only means the broker accepted the order, not that it
        filled.
        """
        if not confirmed:
            alert.log(f"[DRY RUN] would {side} {qty}×{code} @ {price:.4f}")
            return dict(_DRY_RUN_FILL)

        import moomoo as ft

        market = infer_market(code)
        # moomoo 要求美股价格精确到分（最小 tick = $0.01）
        if code.startswith("US."):
            price = round(price, 2)
        trd_ctx = make_trade_ctx(market)
        result = {"order_id": "", "dealt_qty": 0.0, "dealt_avg_price": 0.0,
                  "status": "SUBMIT_FAILED"}
        side_cn = "买入" if side == "BUY" else "卖出"
        try:
            order_type = ft.OrderType.NORMAL
            trd_side = ft.TrdSide.BUY if side == "BUY" else ft.TrdSide.SELL

            # P5 Execution Audit — record-only calls, never raise (see
            # engine/order_audit.py); nothing below depends on them.
            order_audit.event("SUBMIT_SENT", price=price, qty=qty, side=side)
            ret, data = trd_ctx.place_order(
                price=price,
                qty=qty,
                code=code,
                trd_side=trd_side,
                order_type=order_type,
                trd_env=trd_env,
            )
            if ret != ft.RET_OK:
                order_audit.event("SUBMIT_REJECTED", error=str(data))
                alert.error(f"{code} {side_cn}下单失败 — {data}")
                return result

            order_id = str(data["order_id"].iloc[0])
            result["order_id"] = order_id
            order_audit.event("ACK", order_id=order_id)

            status = ""
            for _ in range(MoomooBroker._POLL_ATTEMPTS):
                time.sleep(MoomooBroker._POLL_INTERVAL_SEC)
                q_ret, q_data = trd_ctx.order_list_query(order_id=order_id, trd_env=trd_env)
                if q_ret != ft.RET_OK or q_data is None or len(q_data) == 0:
                    order_audit.event("POLL_FAILED", ret=str(q_ret))
                    continue
                row = q_data.iloc[0]
                status = str(row.get("order_status", ""))
                result["dealt_qty"] = float(row.get("dealt_qty", 0) or 0)
                result["dealt_avg_price"] = float(row.get("dealt_avg_price", 0) or 0)
                result["status"] = status
                order_audit.event("POLL", status=status, dealt_qty=result["dealt_qty"],
                                  dealt_avg_price=result["dealt_avg_price"],
                                  terminal=status in MoomooBroker._TERMINAL,
                                  create_time=row.get("create_time"),
                                  updated_time=row.get("updated_time"))
                if status in MoomooBroker._TERMINAL:
                    break

            # Still open after the poll window (unfilled or only partially
            # filled DAY order) — cancel the remainder now instead of
            # leaving a silent pending order neither local state nor this
            # function's caller has any record of.
            if status not in MoomooBroker._TERMINAL:
                try:
                    order_audit.event("CANCEL_SENT", order_id=order_id)
                    trd_ctx.modify_order(
                        modify_order_op=ft.ModifyOrderOp.CANCEL,
                        order_id=order_id, qty=qty, price=price, trd_env=trd_env,
                    )
                except Exception as exc:
                    order_audit.event("CANCEL_FAILED", error=f"{exc.__class__.__name__}: {exc}")
                    alert.log(f"{code} 撤销未成交订单失败 — {exc}")
                # Re-check once to capture whatever filled right up to the cancel.
                q_ret, q_data = trd_ctx.order_list_query(order_id=order_id, trd_env=trd_env)
                if q_ret == ft.RET_OK and q_data is not None and len(q_data) > 0:
                    row = q_data.iloc[0]
                    result["dealt_qty"] = float(row.get("dealt_qty", 0) or 0)
                    result["dealt_avg_price"] = float(row.get("dealt_avg_price", 0) or 0)
                    result["status"] = str(row.get("order_status", ""))
                    order_audit.event("POLL", status=result["status"],
                                      dealt_qty=result["dealt_qty"],
                                      dealt_avg_price=result["dealt_avg_price"],
                                      terminal=result["status"] in MoomooBroker._TERMINAL,
                                      create_time=row.get("create_time"),
                                      updated_time=row.get("updated_time"), after_cancel=True)
                else:
                    order_audit.event("POLL_FAILED", ret=str(q_ret), after_cancel=True)

            if result["dealt_qty"] <= 0:
                alert.warn(f"{code} {side_cn}未成交（{result['status']}），"
                           f"本地不记录持仓变化，order_id={order_id}")
            elif result["dealt_qty"] < qty:
                alert.warn(f"{code} {side_cn}部分成交 {result['dealt_qty']:.0f}/{qty}股 "
                           f"@{result['dealt_avg_price']:.2f}，剩余已撤单")
        except Exception as exc:
            order_audit.event("EXCEPTION", error=f"{exc.__class__.__name__}: {exc}",
                              order_id=result["order_id"] or None)
            alert.error(f"{code} 下单时发生异常 — {exc}")
        finally:
            safe_close(trd_ctx)

        return result

    # ── V3.6 P7 reconciliation helpers (engine/execution_guard.py) ─────────────

    @staticmethod
    def query_orders(code: str, trd_env, order_id: str = "",
                     since_date: Optional[str] = None) -> Tuple[bool, List[dict]]:
        """Read-only lookup of broker orders for `code`: today's orders
        (order_list_query), plus history_order_list_query from `since_date`
        (YYYY-MM-DD) when given. Filtered to `order_id` when given.
        Returns (ok, rows); ok=False when any query failed or raised, so the
        caller never mistakes "could not look" for "not there"."""
        import moomoo as ft

        trd_ctx = make_trade_ctx(infer_market(code))
        rows: Dict[str, dict] = {}
        ok = True
        try:
            queries = [("today", lambda: trd_ctx.order_list_query(
                order_id=order_id, code=code, trd_env=trd_env))]
            if since_date:
                queries.append(("history", lambda: trd_ctx.history_order_list_query(
                    code=code, start=since_date, end=datetime.now().strftime("%Y-%m-%d"),
                    trd_env=trd_env)))
            for _name, q in queries:
                try:
                    ret, data = q()
                except Exception:
                    ok = False
                    continue
                if ret != ft.RET_OK:
                    ok = False
                    continue
                if data is None:
                    continue
                for _, r in data.iterrows():
                    oid = str(r.get("order_id", "") or order_id)
                    if order_id and oid != str(order_id):
                        continue
                    rows[oid] = {
                        "order_id": oid,
                        "code": str(r.get("code", code) or code),
                        "trd_side": str(r.get("trd_side", "") or ""),
                        "qty": float(r.get("qty", 0) or 0),
                        "price": float(r.get("price", 0) or 0),
                        "order_status": str(r.get("order_status", "") or ""),
                        "dealt_qty": float(r.get("dealt_qty", 0) or 0),
                        "dealt_avg_price": float(r.get("dealt_avg_price", 0) or 0),
                        "create_time": str(r.get("create_time", "") or ""),
                        "updated_time": str(r.get("updated_time", "") or ""),
                    }
        finally:
            safe_close(trd_ctx)
        return ok, list(rows.values())

    @staticmethod
    def cancel_order(code: str, order_id: str, trd_env) -> Tuple[bool, str]:
        """Cancel a still-open order found during reconciliation. Returns (ok, error)."""
        import moomoo as ft

        trd_ctx = make_trade_ctx(infer_market(code))
        try:
            out = trd_ctx.modify_order(modify_order_op=ft.ModifyOrderOp.CANCEL,
                                       order_id=order_id, qty=0, price=0, trd_env=trd_env)
            if isinstance(out, tuple) and len(out) == 2 and out[0] != ft.RET_OK:
                return False, str(out[1])
            return True, ""
        except Exception as exc:
            return False, f"{exc.__class__.__name__}: {exc}"
        finally:
            safe_close(trd_ctx)


class PaperBroker:
    """Virtual fill — never calls moomoo. Always "succeeds" (no rejection
    modeling for v1). Used for markets moomoo doesn't support real trading
    on yet (JP today; see get_broker())."""

    @staticmethod
    def place_order(code: str, side: str, qty: int, price: float,
                     trd_env, env_label: str, confirmed: bool) -> dict:
        if not confirmed:
            alert.log(f"[DRY RUN PAPER] would {side} {qty}×{code} @ {price:.4f}")
            return dict(_DRY_RUN_FILL)

        order_id = f"PAPER-{uuid.uuid4().hex[:10]}"
        alert.log(f"[PAPER FILL] {side} {qty}×{code} @ {price:.4f}  order_id={order_id}")
        return {"order_id": order_id, "dealt_qty": float(qty),
                "dealt_avg_price": float(price), "status": "FILLED_ALL"}


def get_broker(code: str):
    """Return the broker class to use for `code`. Single seam for future
    market/broker additions (real JP execution, HK, IBKR, ... per the
    "Future Upgrade" design)."""
    if infer_market(code) == "JP":
        return PaperBroker
    return MoomooBroker
