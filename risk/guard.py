"""
risk/guard.py — position-level and portfolio-level risk checks.

Position checks:
  check_stop_loss      — price fell more than STOP_LOSS_PCT from entry
  check_take_profit    — price rose more than TAKE_PROFIT_PCT from entry
  check_exit_ordered   — ordered exit check with strategy-aware priority

Exit priority (enforced by check_exit_ordered):
  1. Hard stop-loss (-5%)               — universal, highest priority
  2. Trend strategy SELL signal         — ATR trailing stop, etc.  [trend only]
  3. Hard take-profit (+15%)            — universal
  4. Mean-reversion strategy SELL signal — %B flip, RSI exit, etc. [ranging only]

Rationale: for trend strategies the position should ride momentum past the
+15% mark (hence strategy exit beats hard take-profit); for mean-reversion
we want to lock in the 15% gain rather than wait for a slow pct_b reversal.

Portfolio checks:
  can_open_position  — number of open positions < MAX_POSITIONS
  check_max_drawdown — portfolio value has not fallen > MAX_DRAWDOWN_PCT from peak
"""
import config


# Strategies that follow momentum — their built-in SELL signal (e.g. ATR
# trailing stop) should take priority over the hard take-profit.
_TREND_STRATEGIES = {"atr_breakout", "atr_breakout_early", "ema_rsi", "ma_rsi", "ma", "macd"}


# ── Position-level ────────────────────────────────────────────────────────────

def check_stop_loss(entry_price: float, current_price: float) -> bool:
    """True when current price has dropped >= STOP_LOSS_PCT below entry."""
    if entry_price <= 0:
        return False
    return (entry_price - current_price) / entry_price >= config.STOP_LOSS_PCT


def check_take_profit(entry_price: float, current_price: float) -> bool:
    """True when current price has risen >= TAKE_PROFIT_PCT above entry."""
    if entry_price <= 0:
        return False
    return (current_price - entry_price) / entry_price >= config.TAKE_PROFIT_PCT


def check_exit(entry_price: float, current_price: float) -> str:
    """Legacy convenience wrapper — use check_exit_ordered for new code."""
    if check_stop_loss(entry_price, current_price):
        return "STOP_LOSS"
    if check_take_profit(entry_price, current_price):
        return "TAKE_PROFIT"
    return ""


def check_exit_ordered(
    entry_price: float,
    current_price: float,
    strategy_name: str = "",
    strategy_signal: str = "",
    trail_stop: float = None,
) -> str:
    """
    Evaluate all exit triggers in the correct priority order.

    Parameters
    ----------
    strategy_name   : strategy that opened this position
    strategy_signal : current SELL/HOLD signal from that strategy
    trail_stop      : current ATR trailing stop price (from update_trailing_stop)

    Returns: reason string or "" (no exit triggered)
    """
    # Priority 1: Hard stop-loss — always overrides everything
    if check_stop_loss(entry_price, current_price):
        return "STOP_LOSS"

    is_trend = strategy_name in _TREND_STRATEGIES

    if is_trend:
        # Trend strategy exit order:
        #   2. ATR trailing stop (replaces hard take-profit — lets winners run)
        #   3. Any residual strategy SELL signal
        # config.TAKE_PROFIT_TREND = None → no hard cap for trending positions.
        if trail_stop is not None and current_price < trail_stop:
            return "ATR_TRAIL"
        if strategy_signal == "SELL":
            return f"STRATEGY_EXIT({strategy_name})"
    else:
        # Mean-reversion exit order:
        #   2. Hard take-profit (+15%)  — lock in the bounce target
        #   3. Strategy indicator SELL  — %B flip, death cross, etc.
        if check_take_profit(entry_price, current_price):
            return "TAKE_PROFIT"
        if strategy_signal == "SELL":
            return f"STRATEGY_EXIT({strategy_name})"

    return ""


def breakeven_lock_floor(entry: float, entry_atr: float, current_price: float,
                          was_locked: bool) -> tuple:
    """
    单一职责：判断保本止损锁是否应该触发/维持，返回 (是否锁定, 止损地板价)。
    一旦价格清出 1×entry_ATR 的浮盈就永久锁定（只会锁上，不会解锁），地板
    价固定在成本价——`update_trailing_stop()`（实盘/模拟盘）和
    `backtest_portfolio.py`（回测）都调用这一个函数做判断，2026-07-06
    之前backtest那份是完全没有实现这段逻辑（不是数值凑巧一致，是真的
    缺失），修复时收敛成这一处共享实现，避免以后又出现"两边各写一次，
    改一边忘了改另一边"的漂移。

    返回的 floor 在 was_locked=False 且未触发时是 None（调用方不应该用它
    去限制trail），触发/已锁定时是 entry（成本价）。
    """
    locked = was_locked or current_price > entry + config.ATR_BREAKEVEN_TRIGGER * entry_atr
    return locked, (entry if locked else None)


def update_trailing_stop(pos: dict, current_price: float, current_atr: float) -> None:
    """
    Advance the ATR trailing stop in-place for a trend-following position.

    Tier progression (ratchets stop up, never down):
      Level 0 — initial:          stop = entry − 2.0 × ATR
      Breakeven lock:             when price > entry + 1×entry_ATR → floor = avg_cost
      Level 1 — float > 20%:     mult → 1.5 × (tighter trailing)
      Level 2 — float > 40%:     mult → 1.2 × (aggressive lock-in)
    """
    if current_atr <= 0:
        return

    entry     = pos.get("avg_cost", pos.get("entry_price", current_price))
    entry_atr = pos.get("entry_atr", current_atr)
    atr_mult  = pos.get("atr_mult",  config.ATR_MULT_BASE)
    be_locked = pos.get("breakeven_locked", False)
    float_pct = (current_price - entry) / entry if entry > 0 else 0.0

    # Ratchet multiplier down only (never relaxes once tightened)
    if float_pct >= 0.40 and atr_mult > config.ATR_MULT_TIGHT:
        atr_mult = config.ATR_MULT_TIGHT
    elif float_pct >= 0.20 and atr_mult > config.ATR_MULT_MID:
        atr_mult = config.ATR_MULT_MID

    be_locked, floor = breakeven_lock_floor(entry, entry_atr, current_price, be_locked)

    # Advance trailing stop (only moves up)
    new_trail = current_price - atr_mult * current_atr
    # pos.get(key, default) only falls back when the key is ABSENT — a
    # position freshly opened with entry_atr=0 stores trail_stop=None
    # explicitly (see Portfolio.open_position), so the key exists and .get()
    # returns None itself, not the default, and max(None, new_trail) crashes.
    old_trail = pos.get("trail_stop")
    if old_trail is None:
        old_trail = entry - config.ATR_MULT_BASE * entry_atr
    trail     = max(old_trail, new_trail)
    if floor is not None:
        trail = max(trail, floor)

    pos["atr_mult"]         = atr_mult
    pos["breakeven_locked"] = be_locked
    pos["trail_stop"]       = trail


# ── Portfolio-level ───────────────────────────────────────────────────────────

def can_open_position(portfolio) -> bool:
    """True if another active signal position can be opened (QQQ core excluded from count)."""
    active = sum(1 for p in portfolio.data["positions"].values()
                 if p.get("strategy") != "core_etf")
    return active < config.MAX_POSITIONS


def check_total_exposure(portfolio) -> bool:
    """True if total deployed capital is below MAX_TOTAL_EXPOSURE_PCT."""
    return portfolio.exposure_pct() < config.MAX_TOTAL_EXPOSURE_PCT


def check_sector_exposure(portfolio, code: str) -> bool:
    """True if adding this code won't push the sector above MAX_SECTOR_EXPOSURE_PCT."""
    sector = config.SECTOR_MAP.get(code, "other")
    return portfolio.sector_exposure_pct(sector) < config.MAX_SECTOR_EXPOSURE_PCT


def check_max_drawdown(portfolio) -> bool:
    """
    True if trading should continue.
    False (halt) when realized equity has fallen more than MAX_DRAWDOWN_PCT
    from its peak.

    Uses portfolio.equity_drawdown_pct() (realized P&L based — same
    mechanism the Half-Kelly throttle in engine/runner.py already relies on
    via is_headwind()). Deliberately NOT based on deployed_capital(): being
    under-invested (idle cash) is not a drawdown, and the old peak_value/
    current_value() pair conflated the two — a 40%-invested account with
    zero losses could show as a 60% "drawdown" and halt all trading.
    """
    return portfolio.equity_drawdown_pct() < config.MAX_DRAWDOWN_PCT
