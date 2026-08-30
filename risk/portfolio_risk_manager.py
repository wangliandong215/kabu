"""
risk/portfolio_risk_manager.py — v2.10 Portfolio Risk Manager.

职责边界（跟 portfolio/capacity_manager.py 同款约定）：本模块只做判断，
不执行任何交易——evaluate()/reconcile()/plan_rebalance() 都是纯函数，
调用方（engine/runner.py 的 _run_emergency_rebalance()）负责实际下单、
更新 portfolio/tracker.py、发通知。log_snapshot()/log_diff()/
try_acquire_rebalance_lock() 等是本模块仅有的副作用（写本地JSONL/锁文件
存档），不涉及下单。

判断依据只有 portfolio/broker_state.py 查到的 broker 真实持仓/现金/总
资产（市值口径）——不使用 portfolio/tracker.py 的 exposure_pct()/
available_cash()（成本价口径，浮盈测不出来，且已实测跟broker真实持仓
对不上，见 config.py 里 Portfolio Risk Manager 那段注释）。
"""
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import config
from portfolio.broker_state import BrokerState

TIER_NORMAL    = "NORMAL"
TIER_PAUSE     = "PAUSE_NEW"
TIER_WARNING   = "WARNING"
TIER_EMERGENCY = "EMERGENCY"
TIER_UNKNOWN   = "UNKNOWN"

LOCK_ACQUIRED      = "ACQUIRED"
LOCK_ACTIVE         = "LOCKED_ACTIVE"
LOCK_STALE          = "LOCKED_STALE"

_RISK_LOG_PATH  = Path(r"C:\KabuData\portfolio\risk_state_log.jsonl")
_DIFF_LOG_PATH  = Path(r"C:\KabuData\portfolio\reconciliation_diffs.jsonl")
_LAST_DIFF_PATH = Path(r"C:\KabuData\portfolio\reconciliation_last_diff.json")
_LOCK_PATH      = Path(r"C:\KabuData\portfolio\rebalance_lock.json")


@dataclass
class RiskAssessment:
    tier: str
    exposure_pct: Optional[float]
    cash: Optional[float]
    total_assets: Optional[float]
    long_mv: Optional[float]
    reason: str = ""


@dataclass
class ReconciliationDiff:
    code: str
    broker_qty: float
    tracker_qty: float
    diff_qty: float


@dataclass
class RebalanceOrder:
    code: str
    sell_qty: int
    price: float
    reason: str
    priority: int


def evaluate(broker_state: Optional[BrokerState]) -> RiskAssessment:
    """Pure. Classifies exposure tier from broker mark-to-market state.
    exposure_pct = long_mv / total_assets — correct even when cash is
    negative (total_assets = cash + long_mv already nets it out)."""
    if broker_state is None or broker_state.total_assets is None or broker_state.total_assets <= 0:
        return RiskAssessment(tier=TIER_UNKNOWN, exposure_pct=None,
                               cash=getattr(broker_state, "cash", None),
                               total_assets=getattr(broker_state, "total_assets", None),
                               long_mv=getattr(broker_state, "long_mv", None),
                               reason="broker state unavailable or total_assets<=0")

    exposure_pct = broker_state.long_mv / broker_state.total_assets
    if exposure_pct <= config.MAX_TOTAL_EXPOSURE_PCT:
        tier = TIER_NORMAL
    elif exposure_pct <= config.PORTFOLIO_RISK_TIER3_PCT:
        tier = TIER_PAUSE
    elif exposure_pct <= config.PORTFOLIO_RISK_EMERGENCY_PCT:
        tier = TIER_WARNING
    else:
        tier = TIER_EMERGENCY

    return RiskAssessment(tier=tier, exposure_pct=exposure_pct,
                           cash=broker_state.cash, total_assets=broker_state.total_assets,
                           long_mv=broker_state.long_mv)


def reconcile(broker_state: Optional[BrokerState],
              tracker_positions: Dict[str, dict]) -> List[ReconciliationDiff]:
    """Pure. Every qty mismatch between broker and tracker, both directions.
    No filtering here — config.RECONCILIATION_IGNORE_CODES is applied by the
    caller when deciding alert severity; every diff is still returned/logged."""
    if broker_state is None:
        return []

    diffs: List[ReconciliationDiff] = []
    codes = set(broker_state.positions) | set(tracker_positions)
    for code in codes:
        broker_qty = broker_state.positions[code].qty if code in broker_state.positions else 0.0
        tracker_qty = tracker_positions.get(code, {}).get("qty", 0.0)
        if abs(broker_qty - tracker_qty) > 1e-6:
            diffs.append(ReconciliationDiff(code, broker_qty, tracker_qty, broker_qty - tracker_qty))
    return diffs


def _rank_satellites_by_current_weakness(tracker_positions: Dict[str, dict],
                                          results: Dict[str, dict]) -> List[str]:
    """Pure. Orders currently-held satellite codes (excludes core_etf) from
    weakest to strongest using ONLY this pass's already-computed technical
    data (the `results` dict engine/scanner.py's scan()/smart_scan() already
    produces for every watchlist code, held or not) plus the ATR trailing
    stop tracker.py already maintains — no new fundamental/news calls, so an
    emergency de-risk decision never grows a dependency on an external
    service beyond the broker itself (2026-08-29 decision).

    v2.10 originally reused capacity_manager._rank_candidates() here, which
    sorts by entry-time total_score — but every satellite in this account
    was scored 100/FULL at entry and that score is never refreshed, so the
    ranking silently degenerated into "oldest position first" (see the
    2026-08-29 rebalance preview report / project memory). This replaces
    that with a ranking built from TODAY's data instead of entry-time data.

    Sort key (ascending — first element = weakest = sold first):
      1. missing this pass's scan data (fetch failure, or a restricted
         --codes run) sorts LAST — never guess a position's current
         strength from stale/absent data.
      2. current signal == 'SELL' first — the strategy's own live vote has
         already flipped bearish; the strongest available "weak" signal.
      3. conviction, direction-aware: strategies/combined.py's
         signal_strength is "winning_votes / total_votes" — i.e. conviction
         IN WHICHEVER DIRECTION WON, always a positive fraction. That means
         "weakest" is NOT simply ascending signal_strength for both groups:
           - within the SELL group, a HIGHER strength (more indicators
             agree it should be sold) is the weaker/more-sellable holding,
             so it must rank BEFORE a barely-SELL position, not after.
           - within the non-SELL group (BUY/HOLD — HOLD is strength=0.0 by
             construction, so it naturally sorts first here), a LOWER
             strength is the weaker holding, exactly as before.
         (2026-08-29: an earlier version of this function used plain
         ascending signal_strength for both groups, which put a weakly-
         convicted SELL ahead of a strongly-convicted one — caught by a live
         dry-run against the real account, where a 50%-strength SELL
         out-ranked a 75%-strength SELL for no good reason.)
      4. ascending distance to the ATR trailing stop, (price-trail_stop)/price
         — only set for trend-strategy positions; closer to its own stop is
         weaker. Positions without a trail_stop never win this tie-break.
      5. entry_time ascending (longest-held first) — last-resort tie-break
         only, not the de facto primary sort it used to be.
    """
    satellite_codes = [code for code, pos in tracker_positions.items()
                        if pos.get("strategy") != "core_etf"]

    def _key(code):
        pos = tracker_positions[code]
        r = results.get(code)
        has_data = r is not None
        signal = r.get("signal") if has_data else None
        sell_first = 0 if signal == "SELL" else 1
        strength = r.get("signal_strength", 0.0) if has_data else 0.0
        # Direction-aware conviction: within SELL, stronger conviction is
        # weaker (more negative -> sorts earlier); within BUY/HOLD, weaker
        # conviction is weaker (sorts earlier) exactly as raw strength already is.
        conviction_rank = -strength if signal == "SELL" else strength
        price = (r.get("current_price") or 0.0) if has_data else 0.0
        trail_stop = pos.get("trail_stop")
        stop_distance = (max(0.0, (price - trail_stop) / price)
                          if price > 0 and trail_stop is not None else float("inf"))
        entry_time = pos.get("entry_time", "")
        return (0 if has_data else 1, sell_first, conviction_rank, stop_distance, entry_time)

    return sorted(satellite_codes, key=_key)


def plan_rebalance(broker_state: BrokerState, tracker_positions: Dict[str, dict],
                    target_pct: float, results: Dict[str, dict]) -> List[RebalanceOrder]:
    """Pure. Cascading 3-priority sell plan to bring exposure back to
    ~target_pct, cheapest-in-trades-first (fewest, largest orders):
      1. QQQ excess above config.QQQ_CORE_TARGET_PCT of total_assets —
         never cuts below the strategic floor.
      2. Weakest-signal satellites first, per _rank_satellites_by_current_
         weakness() above — whole position if it fits inside the remaining
         excess, else a single partial trim and stop.
      3. Any remaining satellite whose broker market value exceeds its
         tracker cost basis — trims only the excess-over-cost, largest
         drift first.
    Every leg skips fragments below config.PORTFOLIO_REBALANCE_MIN_TRADE_USD.

    `results` is the current pass's scan() output (code -> signal/strength/
    current_price for every watchlist code) — pass whatever the caller
    already has in scope; missing entries degrade gracefully (see rank
    function above), they don't error."""
    total_assets = broker_state.total_assets
    if total_assets is None or total_assets <= 0:
        return []

    excess = broker_state.long_mv - target_pct * total_assets
    if excess <= 0:
        return []

    orders: List[RebalanceOrder] = []
    min_trade = config.PORTFOLIO_REBALANCE_MIN_TRADE_USD

    # ── Priority 1: QQQ excess above its 25% strategic floor ──────────────────
    qqq_pos = broker_state.positions.get(config.QQQ_CORE_CODE)
    if qqq_pos and qqq_pos.market_val > 0 and qqq_pos.current_price > 0:
        qqq_target_value = config.QQQ_CORE_TARGET_PCT * total_assets
        qqq_excess_value = qqq_pos.market_val - qqq_target_value
        if qqq_excess_value > min_trade:
            sell_value = min(qqq_excess_value, excess)
            sell_qty = int(sell_value // qqq_pos.current_price)
            sell_qty = min(sell_qty, int(qqq_pos.qty))
            sell_value = sell_qty * qqq_pos.current_price
            if sell_qty > 0 and sell_value >= min_trade:
                orders.append(RebalanceOrder(config.QQQ_CORE_CODE, sell_qty,
                                              qqq_pos.current_price, "QQQ_EXCESS_TRIM", 1))
                excess -= sell_value

    already_ordered = {o.code for o in orders}

    # ── Priority 2: weakest-signal satellites first ────────────────────────────
    ranked = _rank_satellites_by_current_weakness(tracker_positions, results)
    for code in ranked:
        if excess <= 0:
            break
        if code in already_ordered:
            continue
        bpos = broker_state.positions.get(code)
        if bpos is None or bpos.market_val <= 0 or bpos.current_price <= 0:
            continue
        if bpos.market_val <= excess:
            sell_qty = int(bpos.qty)
            sell_value = bpos.market_val
        else:
            sell_qty = int(excess // bpos.current_price)
            sell_value = sell_qty * bpos.current_price
        if sell_qty <= 0 or sell_value < min_trade:
            continue
        orders.append(RebalanceOrder(code, sell_qty, bpos.current_price, "WEAK_SIGNAL_TRIM", 2))
        already_ordered.add(code)
        excess -= sell_value

    # ── Priority 3: remaining satellites whose market value has drifted
    #    above their tracker cost basis — trim only the drift ─────────────────
    if excess > 0:
        drift_candidates = []
        for code, pos in tracker_positions.items():
            if pos.get("strategy") == "core_etf" or code in already_ordered:
                continue
            bpos = broker_state.positions.get(code)
            if bpos is None or bpos.current_price <= 0:
                continue
            cost_basis = pos.get("entry_price", 0.0) * pos.get("qty", 0.0)
            drift = bpos.market_val - cost_basis
            if drift > 0:
                drift_candidates.append((code, drift, bpos))
        drift_candidates.sort(key=lambda c: -c[1])

        for code, drift, bpos in drift_candidates:
            if excess <= 0:
                break
            sell_value = min(drift, excess)
            sell_qty = int(sell_value // bpos.current_price)
            sell_qty = min(sell_qty, int(bpos.qty))
            sell_value = sell_qty * bpos.current_price
            if sell_qty <= 0 or sell_value < min_trade:
                continue
            orders.append(RebalanceOrder(code, sell_qty, bpos.current_price, "POSITION_DRIFT_TRIM", 3))
            already_ordered.add(code)
            excess -= sell_value

    return orders


def log_snapshot(assessment: RiskAssessment, diffs: List[ReconciliationDiff],
                  executed_orders: List[dict]) -> None:
    """Append one JSON line per run_once() pass — never raises (best-effort,
    like every other trade_tracker.* logging call site in engine/runner.py)."""
    try:
        _RISK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            **asdict(assessment),
            "diff_count": len(diffs),
            "executed_orders": executed_orders,
        }
        with open(_RISK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def log_diff(diff: ReconciliationDiff) -> None:
    """Append one JSON line per reconciliation mismatch — always written,
    regardless of config.RECONCILIATION_IGNORE_CODES (that list only
    controls alert severity, never whether the diff gets recorded)."""
    try:
        _DIFF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": datetime.now().isoformat(), **asdict(diff)}
        with open(_DIFF_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def check_and_update_diff_growth(diffs: List[ReconciliationDiff]) -> Dict[str, bool]:
    """Best-effort. Compares each diff's |diff_qty| against the last-seen
    value for that code (persisted in a tiny JSON file — same idea as
    tracker.json's own persistence, just for this one purpose) and reports
    which codes have a GROWING gap since last time. Applies to every code,
    including ones in config.RECONCILIATION_IGNORE_CODES — a growing gap on
    an otherwise-muted code (e.g. the delisted-EA phantom row unexpectedly
    changing qty) is exactly the exception that should still surface.

    Returns {code: True/False}. Never raises — a failure here must not
    affect the pass's actual risk decisions, only this one growth callout;
    on failure returns {} (caller treats as "nothing flagged as growing")."""
    try:
        _LAST_DIFF_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(_LAST_DIFF_PATH, "r", encoding="utf-8") as f:
                last_seen = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            last_seen = {}

        grew: Dict[str, bool] = {}
        for d in diffs:
            # No prior record for this code -> establishing the baseline,
            # not "growth" (nothing to have grown from yet).
            grew[d.code] = d.code in last_seen and abs(d.diff_qty) > abs(last_seen[d.code])
            last_seen[d.code] = d.diff_qty

        with open(_LAST_DIFF_PATH, "w", encoding="utf-8") as f:
            json.dump(last_seen, f, ensure_ascii=False, indent=2)
        return grew
    except Exception:
        return {}


def try_acquire_rebalance_lock() -> str:
    """Best-effort file lock so two emergency-rebalance executions (the
    automatic loop and a manually-run execute_emergency_rebalance.py, or two
    manual runs) never overlap. Returns LOCK_ACQUIRED (safe to proceed —
    caller MUST call release_rebalance_lock() when done, success or not),
    LOCK_ACTIVE (a fresh lock already exists — another execution is
    presumably in flight, caller must not proceed), or LOCK_STALE (a lock
    exists but is older than config.PORTFOLIO_REBALANCE_LOCK_TIMEOUT_SEC —
    implies a crash mid-execution; deliberately NOT auto-cleared, since
    auto-clearing could mask exactly the kind of interrupted-execution state
    this lock exists to catch — a human must delete the file by hand).
    On any I/O error, fails conservatively by returning LOCK_ACTIVE (never
    silently proceed as if no lock existed)."""
    try:
        _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _LOCK_PATH.exists():
            try:
                with open(_LOCK_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                age = time.time() - existing.get("started_at_epoch", 0)
            except Exception:
                age = 0  # unreadable lock file -> treat as fresh/active, not stale
            if age > config.PORTFOLIO_REBALANCE_LOCK_TIMEOUT_SEC:
                return LOCK_STALE
            return LOCK_ACTIVE

        record = {"pid": os.getpid(), "started_at_epoch": time.time(),
                  "started_at": datetime.now().isoformat()}
        with open(_LOCK_PATH, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return LOCK_ACQUIRED
    except Exception:
        return LOCK_ACTIVE


def release_rebalance_lock() -> None:
    """Best-effort — only call after a successful try_acquire_rebalance_lock()
    == LOCK_ACQUIRED, always in a finally block so a mid-execution exception
    still releases it."""
    try:
        _LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        pass
