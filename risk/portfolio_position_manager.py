"""
risk/portfolio_position_manager.py — v2.12 Portfolio Position Manager (Phase 2,
Observation Mode).

职责：账户级别的"总仓位天花板"——在现有 Entry Signal -> Risk Check -> Order
之间增加一层更早、更细粒度的敞口预算，专门解决"大跌卖出后现金增加、Entry
Engine又把仓位买回去"的仓位来回震荡问题（用户2026-09-15需求）。跟
risk/portfolio_risk_manager.py是姊妹模块，职责边界同款约定：本模块只做
判断，不执行任何交易——classify_market_regime()/max_exposure_for_regime()/
compute_exposure_budget()都是纯函数，调用方（engine/runner.py）负责用
这个预算去clamp各买入路径的qty、发通知、写日志。log_attempt()是本模块
仅有的副作用（写本地JSONL存档），不涉及下单。

不新增第二套broker敞口数据源：current_exposure直接复用
risk/portfolio_risk_manager.py::evaluate()已经算出的
assessment.long_mv/assessment.total_assets（broker市值口径），本模块不
重新fetch broker状态。

跟已有风控层级的关系（互不替代，取更严格的那个）：
  macro_block / assessment.tier      v2.10 Layer2，总仓位95/100/105% —— 不动
  QQQ CORE concentration             v2.11 Layer1，QQQ自身30/35%     —— 不动
  Portfolio Position Manager（本模块） v2.12，regime驱动的总仓位天花板 —— 新增

config.PORTFOLIO_REGIME_MAX_EXPOSURE里的数值全部标注INITIAL/NOT VALIDATED
（见config.py同名注释）——本模块BULL档的上限(100%)可能高于v2.10 Layer2已有
的95% PAUSE_NEW门槛，这不是bug：Layer2那95%门槛完全不受本模块影响，仍然是
既有硬顶；本模块的regime天花板只在它比95%更严格时才会真正收紧可买空间
（比如CAUTION 75%/RISK_OFF 50%）——本模块只能让可买空间变小或不变，不会
让它变大。

config.PORTFOLIO_POSITION_MANAGER_ENABLED=False（Phase 2默认状态）时，本
模块仍然正常计算regime/current_exposure/remaining_budget并写日志，只是
engine/runner.py不会把算出来的allowed_qty真正应用到下单数量上——这是
"Observation Mode"的实现方式，不是本模块自己判断要不要生效。
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import config

REGIME_BULL     = "BULL"
REGIME_NORMAL   = "NORMAL"
REGIME_CAUTION  = "CAUTION"
REGIME_RISK_OFF = "RISK_OFF"

_PM_LOG_PATH = Path(r"C:\KabuData\portfolio\position_manager_log.jsonl")


def classify_market_regime(weather_code: Optional[int], qqq_above_ma: Optional[bool],
                            drawdown_halt: bool) -> str:
    """Pure. Deterministic, zero new data dependencies — reuses three signals
    engine/runner.py already computes every pass for other purposes:
      weather_code  — engine/market_weather.py's 0/1/2 crisis/chop/safe scale
                       (already backtested/validated, see that module's
                       docstring — not re-derived here, just consumed).
      qqq_above_ma  — the same MA200 trend gate the QQQ Beta core position
                       itself uses (runner.py::_qqq_above_ma()).
      drawdown_halt — risk/guard.py::check_max_drawdown()'s halt flag.

    No local LLM / risk-score input in this phase — see module docstring's
    "future" note in config.py. Missing weather_code/qqq_above_ma (this
    pass's broker/kline fetch failed) degrades to CAUTION, never silently
    treated as BULL/NORMAL on absent data.

    NORMAL is a defensive fallback only — with today's three boolean-ish
    inputs it is not actually reachable (weather_code==2 and qqq_above_ma
    together already return BULL). Kept so a future extra regime signal
    can land here without every branch needing to be re-derived.
    """
    if drawdown_halt or weather_code == 0:
        return REGIME_RISK_OFF
    if weather_code is None or qqq_above_ma is None:
        return REGIME_CAUTION
    if weather_code == 1 or not qqq_above_ma:
        return REGIME_CAUTION
    if weather_code == 2 and qqq_above_ma:
        return REGIME_BULL
    return REGIME_NORMAL


def max_exposure_for_regime(market_regime: str) -> float:
    """Pure. Looks up config.PORTFOLIO_REGIME_MAX_EXPOSURE (INITIAL/NOT
    VALIDATED placeholder table, see config.py). Falls back to the most
    conservative configured tier for an unrecognized regime label — never
    fails open to an unbounded 100%."""
    table = config.PORTFOLIO_REGIME_MAX_EXPOSURE
    if market_regime in table:
        return table[market_regime]
    return min(table.values())


@dataclass
class ExposureBudget:
    regime: str
    max_exposure_pct: float
    current_exposure_pct: Optional[float]
    current_exposure_mv: Optional[float]
    total_assets: Optional[float]
    max_exposure_mv: Optional[float]
    remaining_budget: Optional[float]   # $, broker mark-to-market — can be negative (already over cap)


def compute_exposure_budget(market_regime: str, long_mv: Optional[float],
                             total_assets: Optional[float]) -> ExposureBudget:
    """Pure. remaining_budget = MAX_ALLOWED_EXPOSURE_MV - CURRENT_EXPOSURE_MV,
    both in broker mark-to-market dollars (never cost-basis, never a plain
    cash balance — see module docstring). None in (broker state unavailable
    this pass) -> None out, rather than guessing a number that could look
    like real capacity."""
    max_pct = max_exposure_for_regime(market_regime)
    if long_mv is None or total_assets is None or total_assets <= 0:
        return ExposureBudget(regime=market_regime, max_exposure_pct=max_pct,
                               current_exposure_pct=None, current_exposure_mv=long_mv,
                               total_assets=total_assets, max_exposure_mv=None,
                               remaining_budget=None)
    max_mv = max_pct * total_assets
    return ExposureBudget(regime=market_regime, max_exposure_pct=max_pct,
                           current_exposure_pct=long_mv / total_assets,
                           current_exposure_mv=long_mv, total_assets=total_assets,
                           max_exposure_mv=max_mv, remaining_budget=max_mv - long_mv)


def log_attempt(section: str, code: str, regime: str, budget: ExposureBudget,
                 remaining_budget: Optional[float], requested_qty: int, allowed_qty: int,
                 applied_qty: int, price: float, action: str, enabled: bool) -> None:
    """Append one JSON line per buy attempt PM evaluated (2a/2b/2c/2d, every
    pass, regardless of config.PORTFOLIO_POSITION_MANAGER_ENABLED) — this is
    what lets Observation Mode answer 'how much would PM have blocked'
    without ever touching real order sizes. Never raises, like every other
    trade_tracker.*/portfolio_risk_manager.log_*() call site in this
    codebase."""
    try:
        _PM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "section": section,
            "code": code,
            "regime": regime,
            "max_exposure_pct": budget.max_exposure_pct,
            "current_exposure_pct": budget.current_exposure_pct,
            "total_assets": budget.total_assets,
            "remaining_capacity_usd": remaining_budget,
            "price": price,
            "requested_qty": requested_qty,
            "requested_usd": requested_qty * price,
            "allowed_qty": allowed_qty,
            "allowed_usd": allowed_qty * price,
            "blocked_usd": max(0.0, (requested_qty - allowed_qty) * price),
            "applied_qty": applied_qty,
            "action": action,
            "enabled": enabled,
        }
        with open(_PM_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
