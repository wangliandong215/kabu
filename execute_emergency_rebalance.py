"""
execute_emergency_rebalance.py — v2.10.1 manual approval entrypoint for a
real EMERGENCY-tier portfolio rebalance.

The automatic run_once()/run_loop() loop NEVER places real rebalance orders
on its own while config.PORTFOLIO_RISK_EMERGENCY_AUTO_EXECUTE is False (the
default) — it only previews and logs. Running THIS script, by hand, after
reading the printed preview, is the human-approval act that lets a real
rebalance happen; it calls engine.runner._run_emergency_rebalance(confirmed=
True, ...) directly, bypassing that config flag entirely (see config.py's
"v2.10.1 上线前安全加固" comment and risk/portfolio_risk_manager.py).

2026-08-29: a real test run submitted a QQQ DAY LIMIT SELL order while the
US market was closed — it sat as SUBMITTED, never filled, and the existing
fail-stop logic correctly cancelled it and stopped without touching any
state (dealt_qty=0, tracker/cash/positions all unchanged — the safety net
worked exactly as designed). Still wasteful to try in the first place, so
this script now refuses to even start when the market is closed —
engine.market_hours.is_open(code) already IS the generic, per-symbol,
holiday/DST-aware "is this market open" gate (used everywhere else in the
codebase, e.g. engine/runner.py's filter_open() at the top of every
run_once() pass) — reused as-is rather than adding a second one. A closed
market is treated as a normal, expected exit here (prints a message, exits
0), never as an error.

Usage:
    python execute_emergency_rebalance.py            # prints preview, asks to confirm
    python execute_emergency_rebalance.py --yes       # skips the interactive prompt
"""
import argparse
import sys

import config
import notify.alert as alert
from common import parse_trd_env
from engine.market_hours import is_open
from engine.runner import _run_emergency_rebalance
from engine.scanner import scan
from engine.trade_tracker import TradeTracker
from portfolio import broker_state as broker_state_mod
from portfolio.tracker import Portfolio
from risk import portfolio_risk_manager as prm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true",
                        help="skip the interactive confirmation prompt")
    args = parser.parse_args()

    # 市场关闭时，DAY LIMIT单只会挂单等成交然后被撤单——不是错误，是可预期的
    # 正常情况，直接安全退出，不消耗一次broker查询/scan，也不会走到下单那步。
    # 用config.QQQ_CORE_CODE代表整个账户的交易市场（当前组合的真实持仓全部是
    # US.*——JP代码走PaperBroker、不会经过这条真实下单路径，见engine/broker.py
    # get_broker()），不是只检查QQQ这一只股票本身。
    if not is_open(config.QQQ_CORE_CODE):
        print("当前美股处于闭市状态——不提交任何订单，安全退出。"
              "请等美股开盘（美东9:30-16:00）后再运行本脚本。")
        return 0

    trd_env = parse_trd_env()
    env_label = config.TRD_ENV
    print(f"env = {env_label}")

    try:
        bs = broker_state_mod.fetch_broker_state(trd_env)
    except Exception as exc:
        print(f"ERROR: broker状态查询失败，无法预览/执行 — {exc}")
        return 1

    assessment = prm.evaluate(bs)
    print(f"当前仓位 {assessment.exposure_pct:.2%}  tier={assessment.tier}  "
          f"现金 ${assessment.cash:,.2f}  总资产 ${assessment.total_assets:,.2f}")

    if assessment.tier != prm.TIER_EMERGENCY:
        print(f"当前不是EMERGENCY（{assessment.tier}），没有需要执行的再平衡，退出。")
        return 0

    portfolio = Portfolio()
    codes = list(portfolio.data["positions"].keys())
    print(f"扫描 {len(codes)} 个持仓标的以获取当前信号数据（用于「卖谁」排序）...")
    results = scan(codes, strategy_name="combined")

    plan = prm.plan_rebalance(bs, portfolio.data["positions"],
                              target_pct=config.PORTFOLIO_RISK_REBALANCE_TARGET,
                              results=results)
    if not plan:
        print("EMERGENCY但当前没有可执行的减仓计划（可能都低于最小交易额），退出。")
        return 0

    print()
    print("=" * 60)
    print("再平衡预览计划")
    print("=" * 60)
    projected_long_mv = bs.long_mv
    for order in plan:
        sell_value = order.sell_qty * order.price
        projected_long_mv -= sell_value
        print(f"  {order.code:10s} SELL {order.sell_qty:>6d}股 @{order.price:>10.4f}  "
              f"=${sell_value:>12,.2f}   [{order.reason}, 优先级{order.priority}]")
    projected_exposure = projected_long_mv / bs.total_assets if bs.total_assets else float("nan")
    print("-" * 60)
    print(f"  预计执行后仓位: {assessment.exposure_pct:.2%} -> {projected_exposure:.2%}")
    print("  (实际执行按逐笔重新查询broker的迭代流程进行，最终结果可能因价格"
          "变动/部分成交而与此预览略有差异 —— 结束后会打印真实的执行后状态)")
    print("=" * 60)
    print()

    if not args.yes:
        answer = input('输入 "EXECUTE" 以确认执行上述真实卖单（其他任意输入=取消）: ')
        if answer.strip() != "EXECUTE":
            print("已取消，未下任何真实订单。")
            return 0

    try:
        tracker = TradeTracker()
    except Exception as exc:
        alert.log(f"trade_tracker: init failed, trade DB disabled this run — {exc}")
        tracker = None

    print("开始执行真实再平衡...")
    executed = _run_emergency_rebalance(
        portfolio=portfolio, results=results, trd_env=trd_env, env_label=env_label,
        confirmed=True, ktype="K_DAY", bars=120, tracker=tracker,
    )

    print()
    print(f"执行完成，共 {len(executed)} 笔:")
    for o in executed:
        print(f"  {o['code']:10s} SOLD {o['sell_qty']:>6d}股 @{o['price']:>10.4f}  [{o['reason']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
