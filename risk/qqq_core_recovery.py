"""
risk/qqq_core_recovery.py — v2.13 QQQ Core Recovery / Reclaim (Phase 2.1,
Observation Mode).

背景：QQQ Beta 底仓（config.QQQ_CORE_TARGET_PCT，engine/runner.py 2d 段）
跌破 MA200 清仓、重新站上 MA200 补仓，这条逻辑本身不动。问题是"重新站上
MA200 之后"——2a/2b/2c 普通买入路径排在 2d 前面执行，共用同一个
remaining_broker_cash/pm_state.remaining_budget 池子，如果普通策略持续
把这两个池子花完，QQQ 的 shortfall 可能长期补不上，没有任何机制会主动
让路（详见 2026-09-23 与用户的设计评审）。

职责边界（跟 risk/portfolio_position_manager.py 同款约定）：本模块只做
判断，不执行任何交易——compute_shortfall()/advance_state()/
reserve_amount()/plan_level3_release() 都是纯函数，调用方（engine/
runner.py）负责把算出来的 reserve 数值用在 2c NEW_ENTRY 一处的可用现金/
敞口判断上。load_state()/save_state()/log_recovery_pass()/
log_level3_plan() 是本模块仅有的副作用（读写本地 JSON/JSONL 存档），同样
不涉及下单。

跟 macro_block/assessment.tier（v2.10，FROZEN）的关系：完全分离，本模块
不判断"现在能不能加仓"，只在 macro_block 已经放行的前提下判断"资源该先
给谁"。调用方必须在 macro_block=True 的 pass 完全跳过本模块（不 load/
advance/save state）——那种情况是系统性风险熔断，不是"被普通仓位挤占"，
不该计入下面的 no-progress 计数，否则会在熔断期间错误地把 Reserve/Level3
计数器推高。

三层升级（都基于同一个 consecutive_no_progress_passes 计数器，单调只会
触发不会回退）：
  Level 1（默认）  什么都不做 —— shortfall 靠2d既有逻辑自然补，前提是
                    "有实质进展"（本pass起点shortfall比上一次记录的pass
                    起点shortfall缩小超过PORTFOLIO_REBALANCE_MIN_TRADE_USD
                    噪声门槛，复用已验证参数，不新造一个）。
  Level 2（Reserve） 连续 QQQ_CORE_RECOVERY_TRIGGER_PASSES 个pass都没有
                    实质进展 -> reserve_active=True，reserve_amount()开始
                    返回非零值，调用方据此降低2c NEW_ENTRY可用的现金/敞口
                    （只影响判断用的"派生值"，不mutate remaining_broker_
                    cash/pm_state.remaining_budget本身——2d因此自动拿回
                    2c少花的部分，不需要任何"归还"步骤，异常安全性由此在
                    架构上保证：没有可回滚的临时状态）。
  Level 3（Plan Only） 连续 QQQ_CORE_RECOVERY_LEVEL3_TRIGGER_PASSES 个pass
                    仍无进展，且普通仓位占比超过
                    QQQ_CORE_RECOVERY_LEVEL3_ORDINARY_EXPOSURE_MIN_PCT ->
                    level3_eligible=True，plan_level3_release()计算"如果
                    要主动卖出部分普通持仓来补齐shortfall，会卖哪些/多少"
                    ——只产出计划+日志，本阶段代码里不存在把这个计划送去
                    下单的调用链，不是靠一个bool开关挡住。
"""
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import config
from risk import portfolio_risk_manager

_STATE_PATH        = Path(r"C:\KabuData\portfolio\qqq_recovery_state.json")
_RECOVERY_LOG_PATH = Path(r"C:\KabuData\portfolio\qqq_recovery_log.jsonl")
_LEVEL3_LOG_PATH   = Path(r"C:\KabuData\portfolio\qqq_recovery_level3_plan_log.jsonl")


@dataclass
class RecoveryState:
    active: bool = False
    last_shortfall: Optional[float] = None
    consecutive_no_progress_passes: int = 0
    reserve_active: bool = False
    level3_eligible: bool = False
    episode_started_at: Optional[str] = None


def load_state() -> RecoveryState:
    """Best-effort. Missing/corrupt file (or any I/O failure) degrades to a
    fresh inactive state — never raises, and never lets a bad state file
    block trading (same fail-safe contract as portfolio_risk_manager.py's
    _LAST_DIFF_PATH/_QQQ_TRIM_PREVIEW_PATH helpers)."""
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = asdict(RecoveryState())
        merged.update({k: v for k, v in data.items() if k in merged})
        return RecoveryState(**merged)
    except Exception:
        return RecoveryState()


def save_state(state: RecoveryState) -> None:
    """Best-effort — never raises."""
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def compute_shortfall(qqq_value: Optional[float], total_assets: Optional[float]) -> float:
    """Pure. max(0, QQQ_CORE_TARGET_PCT*total_assets - qqq_value) — broker
    mark-to-market口径（跟remaining_broker_cash/pm_state.remaining_budget
    同一个denominator），不是2d自己用的tracker成本价口径shortfall（两者
    服务不同目的，互不替代，见模块docstring）。"""
    if total_assets is None or total_assets <= 0:
        return 0.0
    return max(0.0, config.QQQ_CORE_TARGET_PCT * total_assets - (qqq_value or 0.0))


def advance_state(prev: RecoveryState, qqq_above_ma: bool, shortfall: float,
                   ordinary_exposure_pct: Optional[float]) -> RecoveryState:
    """Pure. 调用方必须只在macro_block=False时调用本函数（见模块docstring）。

    qqq_above_ma=False 或 shortfall<=0 -> 全新inactive状态（episode结束/
    不存在，任何遗留的reserve_active/level3_eligible/计数器一律清零，
    QQQ下次重新站上MA200时是一次全新的episode，不会带着上一次的旧计数）。
    """
    if not qqq_above_ma or shortfall <= 0:
        return RecoveryState()

    now = datetime.now().isoformat()
    if not prev.active:
        return RecoveryState(active=True, last_shortfall=shortfall,
                              consecutive_no_progress_passes=0,
                              reserve_active=False, level3_eligible=False,
                              episode_started_at=now)

    progress = (prev.last_shortfall - shortfall) if prev.last_shortfall is not None else 0.0
    if progress > config.PORTFOLIO_REBALANCE_MIN_TRADE_USD:
        counter = 0
    else:
        counter = prev.consecutive_no_progress_passes + 1

    # 单调 —— 一旦触发过，不会因为某一个pass恰好有进展就退回False，避免
    # 在"时好时坏"的边界附近反复开关Reserve。
    reserve_active = prev.reserve_active or (counter >= config.QQQ_CORE_RECOVERY_TRIGGER_PASSES)
    level3_eligible = prev.level3_eligible or (
        counter >= config.QQQ_CORE_RECOVERY_LEVEL3_TRIGGER_PASSES
        and ordinary_exposure_pct is not None
        and ordinary_exposure_pct >= config.QQQ_CORE_RECOVERY_LEVEL3_ORDINARY_EXPOSURE_MIN_PCT
    )
    return RecoveryState(active=True, last_shortfall=shortfall,
                          consecutive_no_progress_passes=counter,
                          reserve_active=reserve_active, level3_eligible=level3_eligible,
                          episode_started_at=prev.episode_started_at)


def reserve_amount(state: RecoveryState, shortfall: float,
                    total_assets: Optional[float], available_pool: Optional[float]) -> float:
    """Pure. reserve_active=False -> 0。否则三重封顶取最小：不超过实际
    缺口，不超过QQQ_CORE_RECOVERY_MAX_RESERVE_PCT×total_assets的单pass
    上限，不超过池子里实际有的钱/敞口——三者都是"不会让预留超过实际需要
    或实际存在的资源"的保护，不是可以叠加放大的额度。"""
    if not state.reserve_active or shortfall <= 0:
        return 0.0
    if total_assets is None or total_assets <= 0 or available_pool is None:
        return 0.0
    cap = config.QQQ_CORE_RECOVERY_MAX_RESERVE_PCT * total_assets
    return max(0.0, min(shortfall, cap, available_pool))


def plan_level3_release(tracker_positions: Dict[str, dict], results: Dict[str, dict],
                         shortfall_remaining: float) -> List[portfolio_risk_manager.RebalanceOrder]:
    """Pure. 调用方只应在state.level3_eligible=True时调用本函数（本函数
    自己不检查这个开关——跟portfolio_risk_manager.plan_qqq_core_trim()一样
    "纯函数不管ENABLED"的既有约定）。排序完全复用
    portfolio_risk_manager.rank_satellites_by_current_weakness()（不新造
    排序规则），从最弱信号开始按市值累加，直到覆盖shortfall_remaining或
    候选耗尽，单笔低于PORTFOLIO_REBALANCE_MIN_TRADE_USD的零头跳过（跟
    plan_qqq_core_trim()同款约定）。

    本函数只返回一份计划，不下单、不mutate portfolio、不调用_place_order
    ——本阶段代码里物理不存在从这里到真实下单的调用链，调用方（engine/
    runner.py）只应该把返回值传给log_level3_plan()记录，不应该传给任何
    执行路径。"""
    if shortfall_remaining <= 0:
        return []

    ranked = portfolio_risk_manager.rank_satellites_by_current_weakness(tracker_positions, results)
    min_trade = config.PORTFOLIO_REBALANCE_MIN_TRADE_USD
    orders: List[portfolio_risk_manager.RebalanceOrder] = []
    remaining = shortfall_remaining

    for code in ranked:
        if remaining <= 0:
            break
        pos = tracker_positions.get(code, {})
        r = results.get(code) or {}
        price = r.get("current_price") or 0.0
        qty = pos.get("qty", 0.0)
        if price <= 0 or qty <= 0:
            continue
        value = price * qty
        if value <= remaining:
            sell_qty = int(qty)
            sell_value = value
        else:
            sell_qty = int(remaining // price)
            sell_value = sell_qty * price
        if sell_qty <= 0 or sell_value < min_trade:
            continue
        orders.append(portfolio_risk_manager.RebalanceOrder(
            code, sell_qty, price, "QQQ_RECOVERY_LEVEL3_PLAN", 1))
        remaining -= sell_value

    return orders


def log_recovery_pass(state: RecoveryState, shortfall: float, reserved_cash: float,
                       reserved_pm_budget: float, enabled: bool) -> None:
    """Append one JSON line per pass an episode is active (caller skips this
    entirely on macro_block=True passes and on passes with no active
    episode) — never raises."""
    try:
        _RECOVERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            **asdict(state),
            "shortfall": shortfall,
            "reserved_cash": reserved_cash,
            "reserved_pm_budget": reserved_pm_budget,
            "enabled": enabled,
        }
        with open(_RECOVERY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def log_level3_plan(state: RecoveryState, shortfall: float,
                     plan: List[portfolio_risk_manager.RebalanceOrder],
                     ordinary_exposure_pct: Optional[float]) -> None:
    """Append one JSON line per pass level3_eligible=True — never raises.
    would_execute is hardcoded False: this phase has no code path that could
    ever turn this plan into a real order (see module docstring)."""
    try:
        _LEVEL3_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            **asdict(state),
            "shortfall": shortfall,
            "ordinary_exposure_pct": ordinary_exposure_pct,
            "plan": [asdict(o) for o in plan],
            "would_execute": False,
        }
        with open(_LEVEL3_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
