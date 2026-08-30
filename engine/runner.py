"""
engine/runner.py — full pipeline: scan → risk checks → size → order → notify.

Two entry points:
  run_once(...)  — one pass over the watchlist
  run_loop(...)  — blocks, calling run_once every `interval_seconds`

Security constraints enforced here:
  • Trade unlock must be done manually in OpenD GUI — SDK unlock_trade is NEVER called.
  • Live trading requires use_real=True AND confirmed=True simultaneously.
  • Default environment is always SIMULATE.
"""
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import config
import notify.alert as alert
from common import infer_market, parse_trd_env
from data.fetcher import get_price
from engine.broker import get_broker
from engine.scanner import scan, smart_scan
from engine import news as news_sentiment
from engine import pipeline
from engine import scoring
from engine import entry_quality
from engine import event_risk
from engine import market_preflight
from engine import regime
from engine import research_snapshot
from engine import market_weather
from engine import market_context
from engine.market_hours import filter_open, should_notify_close, should_notify_open, is_daytime_jst
from portfolio import capacity_manager
from portfolio import broker_state as broker_state_mod
from portfolio.tracker import Portfolio
from risk import guard, sizing, portfolio_risk_manager
from engine.trade_tracker import (TradeTracker, build_regime_ctx_preferring_hmm,
                                   default_parameter_snapshot,
                                   compute_parameter_hash)


def select_watchlist() -> List[str]:
    """Default watchlist when no --codes override is given: JP trading
    hours (09:00-15:30 JST) scan/trade WATCHLIST_ASIA_PACIFIC, everything
    else (evening through next morning, covering the US session) scans
    WATCHLIST_EUROPE_US. Re-evaluated on every run_once() call so a
    long-running loop naturally switches pools as the day/night boundary
    is crossed, instead of freezing whichever pool was current at
    process launch.

    config.JP_TRADING_ENABLED=False (set 2026-07-21, moomoo JP quote
    permission outage) skips the JP pool entirely during JP hours instead
    of returning it — see config.py comment for how to re-enable."""
    if not is_daytime_jst():
        return config.WATCHLIST_EUROPE_US
    if not config.JP_TRADING_ENABLED:
        alert.log("runner: JP trading disabled (config.JP_TRADING_ENABLED=False) — skipping JP session")
        return []
    return config.WATCHLIST_ASIA_PACIFIC


def run_once(
    strategy_name: str = "combined",
    codes: Optional[List[str]] = None,
    ktype: str = "K_DAY",
    bars: int = 120,
    confirmed: bool = False,
    use_real: bool = False,
    auto_route: bool = False,
    run_id: Optional[str] = None,
) -> None:
    """
    One full pipeline pass:
      1. Check portfolio-level drawdown guard.
      2. Scan all codes with the chosen strategy.
      3. For open positions: check stop-loss / take-profit → place sell order.
      4. For new signals: check position limits → size → place buy order.

    Parameters
    ----------
    confirmed : bool
        Must be True to actually place orders (even in SIMULATE mode).
    use_real : bool
        Must be True to use live/real-money environment. Ignored unless
        confirmed=True. When False, forces SIMULATE regardless of config.
    run_id : Optional[str]
        v2.8 Trade Intelligence Database run grouping. run_loop() generates
        one run_id per loop invocation and threads it through every pass so
        a whole trading session's trades share one metadata row; a
        standalone one-off run_once() call auto-generates its own if not
        given, so its trades still get tagged with *some* run_id.
    """
    if run_id is None:
        run_id = uuid.uuid4().hex

    watchlist = codes or select_watchlist()

    # Filter to markets currently open (prevents dead scans on closed exchanges)
    open_codes = filter_open(watchlist)
    skipped = len(watchlist) - len(open_codes)
    if skipped:
        alert.log(f"runner: {skipped} code(s) skipped — market closed")
    if not open_codes:
        alert.log("runner: no markets open — pass skipped")
        return

    # ── JP Paper Trading (v2.9) ───────────────────────────────────────────────
    # moomoo doesn't support real JP order execution today (see
    # engine/broker.py docstring) — a pass scanning only JP codes runs the
    # exact same pipeline against a separate virtual capital pool / position
    # file instead, via engine.broker.PaperBroker (resolved per-code inside
    # _place_order()). Mixed JP + non-JP watchlists (only possible via a
    # manual --codes override) are treated as a real/non-JP pass — not
    # supported in v1, see config.py JP_PAPER_* comment.
    is_jp_pass = bool(open_codes) and all(infer_market(c) == "JP" for c in open_codes)

    # ── Determine trade environment ───────────────────────────────────────────
    if is_jp_pass:
        env_label = "JP-PAPER"
        trd_env = None   # PaperBroker never reads trd_env
    elif use_real and confirmed:
        trd_env = parse_trd_env(config.TRD_ENV)
        env_label = config.TRD_ENV
    else:
        import moomoo as ft
        trd_env = ft.TrdEnv.SIMULATE
        env_label = "SIMULATE"

    # v2.1 多因子总分模型的 env 参数（engine.fundamental / engine.scoring）：
    # SIMULATE/JP-PAPER 都不是真实资金，同样按 paper 处理。
    # backtest_portfolio.py 是完全独立的脚本入口，走的是 env="backtest"，
    # 不会经过这里。
    score_env = "paper" if env_label in ("SIMULATE", "JP-PAPER") else "live"

    mode_label = "AUTO-ROUTE" if auto_route else strategy_name
    alert.log(f"runner: starting pass  env={env_label}  mode={mode_label}"
              f"  dry_run={not confirmed}")

    if is_jp_pass:
        portfolio = Portfolio(path=config.JP_PAPER_POSITIONS_PATH,
                              initial_cash=config.JP_PAPER_INITIAL_CAPITAL)
    else:
        portfolio = Portfolio()
    try:
        tracker = TradeTracker()   # v2.6 Trade Intelligence Database — side-effect-only
    except Exception as exc:
        alert.log(f"trade_tracker: init failed, trade DB disabled this pass — {exc}")
        tracker = None

    # ── Portfolio-level guard ─────────────────────────────────────────────────
    # 2026-07-11 fix: this used to `return` here, which skipped scan() and the
    # exit-check loop below entirely — meaning a drawdown breach (exactly the
    # moment stop-losses matter most) silently disabled hard stop-loss/ATR
    # trailing/take-profit checks for every open position for that whole
    # pass. Now it only feeds into `macro_block` (set further below, after
    # the exit-check loop runs) so it blocks new BUY/PROMOTE/Replacement the
    # same way the news/QQQ-technical/market-weather breakers already do —
    # SELL/EXIT is never gated by any of the four breakers.
    drawdown_halt = not guard.check_max_drawdown(portfolio)
    if drawdown_halt:
        alert.warn("触发最大回撤熔断，本轮禁止新开仓（已持仓位仍正常检查退出）")

    if auto_route:
        results = smart_scan(open_codes, ktype=ktype, bars=bars)
    else:
        results = scan(open_codes, strategy_name=strategy_name, ktype=ktype, bars=bars)

    # ── 1. Check exits for open positions ─────────────────────────────────────
    # Rule: "who buys, who exits" — use the strategy that opened the position,
    # regardless of what the current regime router would select today.
    #
    # Exit priority (enforced by guard.check_exit_ordered):
    #   1. Hard stop-loss (-5%)               — highest priority, always
    #   2. Trend strategy SELL (ATR trailing)  — before hard take-profit
    #   3. Hard take-profit (+15%)
    #   4. Mean-reversion SELL (%B flip, etc.) — lowest priority
    for code, pos in list(portfolio.data["positions"].items()):
        result       = results.get(code, {})
        price        = result.get("current_price") or get_price(code)
        entry_strat  = pos.get("strategy", strategy_name)

        if price <= 0:
            continue

        # v2.8 Trade Intelligence Database — daily MFE/MAE snapshot. Runs for
        # every open position every pass (including core_etf, before its
        # early `continue` below), purely additive/side-effect-free like
        # every other tracker.* call site. Uses entry_atr (not a freshly
        # re-fetched ATR) to avoid duplicating the fetch the normal branch
        # already does further below — an acceptable approximation for a
        # daily high-water-mark snapshot.
        if tracker is not None:
            try:
                tracker.update_position_metrics(
                    trade_id=_trade_id(code, pos.get("entry_time")),
                    date=datetime.now().strftime("%Y-%m-%d"),
                    close=price, entry_price=pos["entry_price"],
                    shares=pos["qty"], direction="LONG",
                    atr=pos.get("entry_atr"),
                    regime_ctx=_trade_regime_ctx(code, ktype, bars),
                )
            except Exception as exc:
                alert.log(f"trade_tracker: update_position_metrics failed {code} — {exc}")

        # ── QQQ Beta 底仓："死仓"，唯一退出条件是跌破 MA200 ──────────────────
        # 死命令（用户明确要求）：这里 continue 提前跳出，不会走到下面的
        # guard.update_trailing_stop / check_exit_ordered（ATR跟踪止损）——
        # 哪怕价格短期内剧烈震荡（4×ATR级别）也绝对不会触发卖出，只有跌破
        # MA200 才清仓。不要在这个分支之外给 core_etf 加任何止损/止盈判断。
        if entry_strat == "core_etf":
            if not _qqq_above_ma():
                alert.warn(f"QQQ底仓跌破MA{config.QQQ_MA_PERIOD}"
                           f"（现价{price:.2f}），清仓退出")
                fill = _place_order(
                    code=code, side="SELL", qty=pos["qty"], price=price,
                    trd_env=trd_env, env_label=env_label, confirmed=confirmed,
                )
                if confirmed and fill["dealt_qty"] >= pos["qty"]:
                    exit_price = fill["dealt_avg_price"] or price
                    closed = portfolio.close_position(code, exit_price, reason="MA200_BREAK")
                    alert.trade_sell(
                        code, closed["avg_cost"], exit_price, closed["qty"],
                        days_held=_days_held(closed.get("entry_time")),
                        reason="MA200_BREAK",
                        cash_available=portfolio.available_cash(),
                        total_equity=portfolio.current_equity(),
                        position_count=portfolio.position_count(),
                        trade_id=portfolio.next_trade_id(), env=env_label,
                    )
                    if tracker is not None:
                        try:
                            tracker.log_exit(
                                trade_id=_trade_id(code, closed.get("entry_time")),
                                price=exit_price, cash=portfolio.available_cash(),
                                equity=portfolio.current_equity(),
                                timestamp=datetime.now().isoformat(),
                                exit_reason="MA200_BREAK",
                                regime_ctx=_trade_regime_ctx(code, ktype, bars),
                            )
                        except Exception as exc:
                            alert.log(f"trade_tracker: log_exit failed {code} — {exc}")
            continue   # 跳过下面的普通持仓退出逻辑

        # Gather the SELL/HOLD signal from the strategy that opened this position.
        entry_result = result if result.get("strategy_used") == entry_strat else {}
        if not entry_result and entry_strat:
            try:
                from data.fetcher import fetch_kline
                from strategies import get_strategy as _get
                df_exit = fetch_kline(code, ktype=ktype, bars=bars)
                if df_exit is not None:
                    entry_result = _get(entry_strat).full_result(df_exit)
            except Exception:
                pass
        strat_signal = entry_result.get("signal", "HOLD") if entry_result else "HOLD"

        # Update ATR trailing stop state before checking exits
        current_atr = (result.get("atr") or entry_result.get("atr") or
                       pos.get("entry_atr", 0.0))
        if entry_strat in guard._TREND_STRATEGIES and current_atr and price > 0:
            guard.update_trailing_stop(pos, price, float(current_atr))

        # Ordered exit check with tiered trailing stop
        reason = guard.check_exit_ordered(
            pos["entry_price"], price, entry_strat, strat_signal,
            trail_stop=pos.get("trail_stop"),
        )

        if reason:
            alert.warn(f"{code} 触发退出（{reason}），策略={entry_strat}，"
                       f"成本={pos['entry_price']:.4f}，现价={price:.4f}")
            fill = _place_order(
                code=code,
                side="SELL",
                qty=pos["qty"],
                price=price,
                trd_env=trd_env,
                env_label=env_label,
                confirmed=confirmed,
            )
            if confirmed and fill["dealt_qty"] >= pos["qty"]:
                exit_price = fill["dealt_avg_price"] or price
                closed = portfolio.close_position(code, exit_price, reason=reason)
                alert.trade_sell(
                    code, closed["avg_cost"], exit_price, closed["qty"],
                    days_held=_days_held(closed.get("entry_time")),
                    reason=reason,
                    cash_available=portfolio.available_cash(),
                    total_equity=portfolio.current_equity(),
                    position_count=portfolio.position_count(),
                    trade_id=portfolio.next_trade_id(), env=env_label,
                )
                if tracker is not None:
                    try:
                        tracker.log_exit(
                            trade_id=_trade_id(code, closed.get("entry_time")),
                            price=exit_price, cash=portfolio.available_cash(),
                            equity=portfolio.current_equity(),
                            timestamp=datetime.now().isoformat(),
                            exit_reason=reason,
                            regime_ctx=_trade_regime_ctx(code, ktype, bars),
                        )
                    except Exception as exc:
                        alert.log(f"trade_tracker: log_exit failed {code} — {exc}")

    # ── v2.10 Portfolio Risk Manager — broker ground-truth exposure control ──
    # 用真实broker持仓市值/现金分级（见risk/portfolio_risk_manager.py docstring
    # 和config.py同名注释）——不用portfolio.exposure_pct()/available_cash()
    # （成本价口径，浮盈测不出来，且已实测跟broker真实持仓对不上）。JP Paper
    # pass跳过：那是独立的虚拟资金池，跟真实US broker账户无关。
    risk_block = None
    remaining_broker_cash = None
    if not is_jp_pass:
        broker_state = None
        try:
            broker_state = broker_state_mod.fetch_broker_state(trd_env)
        except Exception as exc:
            alert.error(f"portfolio_risk_manager: broker状态查询失败（{exc}），"
                        f"本轮暂停新增买入（保守降级，不做强制平仓/对账）")
            risk_block = "PORTFOLIO_RISK_DATA_UNAVAILABLE"

        if broker_state is not None:
            remaining_broker_cash = broker_state.cash

            diffs = portfolio_risk_manager.reconcile(broker_state, portfolio.data["positions"])
            diff_grew = portfolio_risk_manager.check_and_update_diff_growth(diffs)
            for d in diffs:
                portfolio_risk_manager.log_diff(d)
                grew = diff_grew.get(d.code, False)
                if d.code not in config.RECONCILIATION_IGNORE_CODES or grew:
                    growth_note = "（差异较上次扩大）" if grew else ""
                    alert.error(f"持仓对账不一致{growth_note}: {d.code} broker={d.broker_qty:.0f}股 "
                                f"tracker={d.tracker_qty:.0f}股 差={d.diff_qty:+.0f}股")

            assessment = portfolio_risk_manager.evaluate(broker_state)
            alert.log(f"portfolio_risk_manager: 仓位{assessment.exposure_pct:.1%}"
                      f"  tier={assessment.tier}  现金${assessment.cash:,.0f}"
                      f"  总资产${assessment.total_assets:,.0f}")

            executed_orders = []
            if assessment.tier in (portfolio_risk_manager.TIER_PAUSE, portfolio_risk_manager.TIER_WARNING):
                risk_block = f"PORTFOLIO_RISK_{assessment.tier}: exposure {assessment.exposure_pct:.1%}"
                alert.warn(risk_block + " — 暂停新增买入，不强制卖出")
            elif assessment.tier == portfolio_risk_manager.TIER_EMERGENCY:
                risk_block = f"PORTFOLIO_RISK_EMERGENCY: exposure {assessment.exposure_pct:.1%}"
                alert.error(risk_block + f" — 触发强制再平衡，目标回落至"
                            f"{config.PORTFOLIO_RISK_REBALANCE_TARGET:.0%}")
                # v2.10.1: 自动循环里的再平衡，哪怕这一轮confirmed=True，
                # 也强制走DRY RUN，直到人工把
                # config.PORTFOLIO_RISK_EMERGENCY_AUTO_EXECUTE打开，或者手动
                # 跑一次 execute_emergency_rebalance.py（那个脚本直接传
                # confirmed=True，绕开这个开关——跑那个脚本本身就是人工确认
                # 这个动作）。
                rebalance_confirmed = confirmed and config.PORTFOLIO_RISK_EMERGENCY_AUTO_EXECUTE
                executed_orders = _run_emergency_rebalance(
                    portfolio=portfolio, results=results, trd_env=trd_env, env_label=env_label,
                    confirmed=rebalance_confirmed, ktype=ktype, bars=bars, tracker=tracker,
                )
                remaining_broker_cash += sum(o["price"] * o["sell_qty"] for o in executed_orders)

            portfolio_risk_manager.log_snapshot(assessment, diffs, executed_orders)

    # ── Macro news circuit breaker (checked once per pass) ───────────────────
    macro_block = news_sentiment.macro_circuit_breaker()
    if macro_block:
        alert.warn(f"新闻面熔断触发（{macro_block}），本轮不再开新仓")
    if not macro_block and drawdown_halt:
        macro_block = "MAX_DRAWDOWN_HALT: realized drawdown exceeded threshold"
    if not macro_block and risk_block:
        macro_block = risk_block

    # ── QQQ macro technical halt + Market Weather regime (checked once per
    #    pass) ───────────────────────────────────────────────────────────────
    # Below MA200 AND own MA20 momentum falling steeply -> block new stock
    # entries entirely (existing positions still exit via their own rules —
    # this only gates NEW risk, doesn't force-liquidate, per user's choice).
    # v1.0-RELEASE-FINAL: locked as a hard block — position-scale and
    # dual-watchlist variants were tried and reverted (see engine/regime.py
    # docstring / config.py / project memory).
    #
    # Always fetch QQQ (not gated by "if not macro_block") because the
    # Market Weather regime code (模块一) feeds sizing.calculate() for the
    # pyramid scale-in step below too, which is NOT itself gated by
    # macro_block — so the code is needed even on days the news breaker
    # already fired.
    weather_code = 1   # safe/cautious default if the fetch or compute fails
    try:
        from data.fetcher import fetch_kline
        bars_needed = max(config.QQQ_MA_PERIOD,
                          regime._MACRO_HALT_MA_PERIOD + regime._MACRO_HALT_SLOPE_LOOKBACK) + 5
        qqq_df = fetch_kline(config.QQQ_CORE_CODE, ktype="K_DAY", bars=bars_needed)

        if not macro_block and regime.qqq_macro_halt(qqq_df):
            macro_block = f"QQQ_TECHNICAL_HALT: below MA{config.QQQ_MA_PERIOD} + steep MA{regime._MACRO_HALT_MA_PERIOD} downslope"
            alert.warn(f"大盘技术面熔断（{macro_block}），本轮不再开新仓")

        weather_code = market_weather.market_weather(qqq_df)
        if weather_code == 0:
            macro_block = macro_block or "MARKET_WEATHER_CODE_0: crisis regime — no new entries"
            alert.warn("大盘天气进入危机状态（状态0），本轮不再开新仓")
        else:
            alert.log(f"runner: market weather code={weather_code}")
    except Exception as exc:
        alert.warn(f"大盘熔断/天气检测失败 — {exc}")

    # ── 2a. Half-Kelly state ──────────────────────────────────────────────────
    kelly_factor = 0.5 if portfolio.is_headwind() else 1.0
    if kelly_factor < 1.0:
        dd = portfolio.equity_drawdown_pct()
        alert.warn(f"进入逆风模式（回撤{dd:.1%}），仓位系数减半")
    else:
        alert.log(f"runner: TAILWIND mode  kelly=1.0x")

    # ── 2a-2. Promote TRENDING_EARLY trials whose regime has confirmed to
    #        TRENDING_UP: top the position up from its 30% trial size to 100%.
    #        Only meaningful under auto_route (smart_scan populates "regime").
    #        Skipped during a macro halt — don't add risk into a black-swan event.
    if auto_route and not macro_block:
        for code, pos in list(portfolio.data["positions"].items()):
            if pos.get("strategy") != "atr_breakout_early":
                continue
            result = results.get(code, {})
            if result.get("regime") != "TRENDING_UP":
                continue

            price = result.get("current_price") or get_price(code)
            if price <= 0:
                continue
            strength = result.get("signal_strength", pos.get("signal_strength", 0.5))
            stop_pct = float(result.get("stop_loss_pct", config.STOP_LOSS_PCT))

            # Hypothetical cash = what's actually free + what this trial already
            # deployed, so the target reflects "what a normal full entry would
            # have sized to", not "what's left after the trial ate into cash".
            pos_value = pos["avg_cost"] * pos["qty"]
            target_qty = sizing.calculate(
                portfolio.available_cash() + pos_value, price, strength,
                total_capital=portfolio.total_capital(),
                stop_loss_pct=stop_pct,
                kelly_factor=kelly_factor,
                strategy="atr_breakout",   # confirmed now — full-size cap
                open_positions=sum(1 for p in portfolio.data["positions"].values()
                                   if p.get("strategy") != "core_etf"),
                rsi_val=result.get("rsi14"),
                market_weather_code=weather_code,
            )
            add_qty = target_qty - pos["qty"]
            if add_qty <= 0:
                continue
            if not guard.check_total_exposure(portfolio):
                continue   # portfolio-wide exposure cap — don't grow into it
            cost = price * add_qty
            if cost > portfolio.available_cash():
                add_qty = int((portfolio.available_cash() * 0.98) / price)
            if add_qty <= 0:
                continue

            alert.info(f"{code} 试错仓位转正（early→confirmed），"
                       f"加仓{add_qty}股 @{price:.4f}")
            fill = _place_order(
                code=code, side="BUY", qty=add_qty, price=price,
                trd_env=trd_env, env_label=env_label, confirmed=confirmed,
            )
            if confirmed and fill["dealt_qty"] > 0:
                filled_qty = int(fill["dealt_qty"])
                fill_price = fill["dealt_avg_price"] or price
                portfolio.add_to_position(code, fill_price, filled_qty, strength,
                                          strategy="atr_breakout")
                alert.trade_buy(
                    code, fill_price, filled_qty,
                    sector=config.SECTOR_MAP.get(code, "other"),
                    score=pos.get("total_score"), score_label=pos.get("score_label"),
                    stop_price=pos.get("trail_stop"),
                    position_pct=(pos["avg_cost"] * pos["qty"]) / portfolio.total_capital(),
                    cash_available=portfolio.available_cash(),
                    position_count=portfolio.position_count(),
                    trade_id=portfolio.next_trade_id(), env=env_label,
                )

    # ── 2b. Pyramid scale-in for existing positions ───────────────────────────
    # 2026-07-11 fix: pyramid adds increase portfolio risk exposure exactly
    # like a new BUY/PROMOTE/Replacement — must be gated by the same
    # macro_block used everywhere else (news/QQQ-technical/market-weather/
    # MAX_DRAWDOWN), not a second independent risk check.
    if config.PYRAMID_ENABLED and not macro_block:
        for code, pos in list(portfolio.data["positions"].items()):
            if pos.get("strategy") == "core_etf":
                continue   # QQQ Beta floor is a fixed "dead" position — never resized
            result    = results.get(code, {})
            new_sig   = result.get("signal") or "HOLD"
            new_str   = result.get("signal_strength", 0.0)
            old_str   = pos.get("signal_strength", 0.0)
            price     = result.get("current_price", 0.0)
            avg_cost  = pos.get("avg_cost", pos["entry_price"])

            # Only add when signal is still BUY and meaningfully stronger
            if new_sig != "BUY":
                continue
            if event_risk.has_earnings_risk(code):
                alert.log(f"runner: {code} 处于财报窗口期，跳过加仓")
                continue
            if new_str - old_str < config.PYRAMID_STRENGTH_UPGRADE_MIN:
                continue
            # Right-side pyramid: price must not be too far above avg cost
            if price <= 0 or price > avg_cost * (1 + config.PYRAMID_MAX_ADD_ABOVE_COST):
                alert.log(f"runner: {code} pyramid skip — price {price:.2f} "
                          f"too far above avg cost {avg_cost:.2f}")
                continue

            stop_pct = float(result.get("stop_loss_pct", config.STOP_LOSS_PCT))
            add_qty = sizing.calculate(
                portfolio.available_cash(), price, new_str,
                total_capital=portfolio.total_capital(),
                stop_loss_pct=stop_pct,
                kelly_factor=kelly_factor,
                market_weather_code=weather_code,
            ) - pos["qty"]
            # NOTE: strategy intentionally omitted here (pre-existing behavior) —
            # pyramid adds are not strategy-capped, so RSI dynamic sizing (which
            # is strategy-gated) does not apply to scale-ins, only new entries.

            if add_qty <= 0:
                continue

            alert.info(f"{code} 加仓（金字塔），+{add_qty}股，"
                       f"信号强度{old_str:.0%}→{new_str:.0%}，"
                       f"均本{avg_cost:.2f}，现价{price:.2f}")
            fill = _place_order(
                code=code, side="BUY", qty=add_qty, price=price,
                trd_env=trd_env, env_label=env_label, confirmed=confirmed,
            )
            if confirmed and fill["dealt_qty"] > 0:
                filled_qty = int(fill["dealt_qty"])
                fill_price = fill["dealt_avg_price"] or price
                portfolio.add_to_position(code, fill_price, filled_qty, new_str)
                alert.trade_buy(
                    code, fill_price, filled_qty,
                    sector=config.SECTOR_MAP.get(code, "other"),
                    score=pos.get("total_score"), score_label=pos.get("score_label"),
                    stop_price=pos.get("trail_stop"),
                    position_pct=(pos["avg_cost"] * pos["qty"]) / portfolio.total_capital(),
                    cash_available=portfolio.available_cash(),
                    position_count=portfolio.position_count(),
                    trade_id=portfolio.next_trade_id(), env=env_label,
                )

    # ── 2c. Open new positions — Two-Phase Execution ──────────────────────────
    # Candidate Pool + Global Filters + Ranking Engine live in engine/pipeline.py
    # (already-held/cooldown/min-strength/earnings-blackout/macro_block filtering,
    # plus full total_score computed for every survivor up front — see that
    # module's docstring for why: it's what fixes the old bug where a
    # later-ranked-but-equal-or-better candidate never got scored before capacity
    # filled up, see the 2026-07-29 US.BKNG/US.ABNB case). Portfolio Construction
    # and Order Executor stay here — they mutate `portfolio` sequentially
    # (capacity/exposure/sector recheck + Active Replacement + sizing + order
    # placement per candidate), which is unchanged existing behavior, not part
    # of the ranking-order bug this refactor fixes.
    candidates = pipeline.build_candidate_pool(
        results, portfolio, macro_block, score_env, weather_code)
    ranked = pipeline.rank_candidates(candidates)
    # Active count excludes QQQ core position
    active_count = sum(1 for p in portfolio.data["positions"].values()
                       if p.get("strategy") != "core_etf")
    slots_free   = config.MAX_POSITIONS - active_count
    alert.log(f"runner: {len(ranked)} BUY candidate(s) ranked, "
              f"{slots_free} slot(s) available  (active={active_count})")

    for entry_rank, cand in enumerate(ranked, start=1):
        code   = cand.code
        result = cand.raw

        if not guard.can_open_position(portfolio):
            # ── v2.3/v2.4 Portfolio Capacity Manager（主动置换）────────────
            # 名额已满时，只有新信号是FULL才尝试换出一个OBSERVATION持仓
            # 腾出名额，而不是直接放弃——跟backtest_portfolio.py共用同一个
            # portfolio.capacity_manager.evaluate_replacement()判断入口
            # （Single Source of Truth，2026-07-06统一，见该函数docstring）。
            # 找不到victim就维持原有行为（break）：候选已经按总分从高到低排好
            # 序了，如果排名最靠前的这个候选都换不出名额，排名更靠后（总分
            # 更低）的候选面对的是同一批持仓（victim池不变），置换优势只会
            # 更难满足（REPLACEMENT_MARGIN 是跟 incoming_score 比较），不会
            # 有更好的结果，所以不用继续试，直接break跟原有语义一致。
            victim_code = None
            if config.ENABLE_ACTIVE_REPLACEMENT and cand.score_label == scoring.LABEL_FULL:
                victim_code = _attempt_active_replacement(
                    portfolio, incoming_code=code, incoming_score=cand.total_score,
                    trd_env=trd_env, env_label=env_label, confirmed=confirmed,
                    results=results, tracker=tracker, ktype=ktype, bars=bars,
                )
            if victim_code is None:
                skip_price = cand.current_price
                alert.warn_skip(code, f"持仓数已达上限，跳过 {code}"
                                f"（现价{skip_price:.4f}，评分{cand.total_score:.0f}/{cand.score_label}）")
                break
            # 置换成立，名额已腾出——直接往下走已有的敞口/板块/
            # sizing/BUY逻辑，就像这个名额本来就空着一样。
        # 总仓位敞口的判断已经上移到v2.10 Portfolio Risk Manager（见本函数前面
        # "Portfolio Risk Manager"那一段）——它在macro_block里，通过
        # pipeline.build_candidate_pool()的macro_block短路（macro_block为真时
        # 直接返回空候选池）挡住这里的整个循环，不需要在这里重复判断一次
        # portfolio.exposure_pct()（成本价口径，已废弃用于风控判断）。
        if not guard.check_sector_exposure(portfolio, code):
            sector = config.SECTOR_MAP.get(code, "other")
            alert.warn_skip(code, f"{code} 所属板块({sector})仓位占比"
                            f"{portfolio.sector_exposure_pct(sector):.0%}"
                            f"已达上限{config.MAX_SECTOR_EXPOSURE_PCT:.0%}，跳过")
            continue

        price = cand.current_price
        if price <= 0:
            alert.warn_skip(code, f"{code} 无法获取价格，跳过")
            continue

        # Resolve entry strategy before sizing (used by strategy cap lookup)
        entry_strategy = result.get("strategy_used") or strategy_name

        # Derive technical stop distance: use strategy-reported value,
        # ATR-based estimate, or fall back to config default.
        if "stop_loss_pct" in result:
            stop_pct = float(result["stop_loss_pct"])
        elif result.get("atr") and price > 0:
            stop_pct = 2.0 * result["atr"] / price   # 2×ATR as stop distance
        else:
            stop_pct = config.STOP_LOSS_PCT

        # v2.1 总分模型的 position_scale 和既有 TRENDING_EARLY 试错仓位系数
        # 相乘——两者都只裁剪、不放大（跟 risk/sizing.py::calculate() 里
        # position_scale 参数"trims, never expands"的既有语义一致）。
        trial_scale = (config.TRENDING_EARLY_POSITION_SCALE
                       if entry_strategy == "atr_breakout_early" else 1.0)
        position_scale = trial_scale * cand.score_components["position_scale"]
        qty = sizing.calculate(
            portfolio.available_cash(), price, cand.signal_strength,
            total_capital=portfolio.total_capital(),
            stop_loss_pct=stop_pct,
            kelly_factor=kelly_factor,
            strategy=entry_strategy,
            open_positions=sum(1 for p in portfolio.data["positions"].values()
                               if p.get("strategy") != "core_etf"),
            rsi_val=result.get("rsi14"),
            position_scale=position_scale,
            market_weather_code=weather_code,
            score_label=cand.score_label,
        )
        # v2.10 现金红线：不能因为新买入信号把broker真实现金买成负数（哪怕总
        # 仓位还在95%以内）——用remaining_broker_cash（本轮从broker真实现金
        # 起算、每次成交后扣减，见前面Portfolio Risk Manager那段）做硬约束，
        # 不用portfolio.available_cash()（成本价口径，floor在0，测不出真实
        # 欠款）。broker状态本轮取数失败时remaining_broker_cash为None，不做
        # 这层约束——那种情况risk_block已经把macro_block整体挡死了。
        if remaining_broker_cash is not None:
            max_affordable = int(remaining_broker_cash // price)
            if max_affordable < qty:
                qty = max_affordable
        if qty <= 0:
            alert.warn_skip(code, f"{code} 现价{price:.4f}下可买股数为0（含真实现金约束），跳过")
            continue

        alert.info(f"准备买入 {code}，{qty}股 @{price:.4f}，"
                   f"信号强度{cand.signal_strength:.0%}，"
                   f"排名#{entry_rank}")
        fill = _place_order(
            code=code,
            side="BUY",
            qty=qty,
            price=price,
            trd_env=trd_env,
            env_label=env_label,
            confirmed=confirmed,
        )
        if confirmed and fill["dealt_qty"] > 0:
            filled_qty = int(fill["dealt_qty"])
            fill_price = fill["dealt_avg_price"] or price
            if remaining_broker_cash is not None:
                remaining_broker_cash -= fill_price * filled_qty
            # Persist entry_atr so ATR trailing stop can be reconstructed after restart.
            # score_label/total_score persisted too so this position can itself be
            # considered as a future Active Replacement victim (see
            # _attempt_active_replacement above).
            portfolio.open_position(code, "BUY", fill_price, filled_qty, cand.signal_strength, entry_strategy,
                                    entry_atr=result.get("atr", 0.0),
                                    score_label=cand.score_label, total_score=cand.total_score)
            alert.trade_buy(
                code, fill_price, filled_qty,
                sector=config.SECTOR_MAP.get(code, "other"),
                score=cand.total_score, score_label=cand.score_label,
                stop_price=portfolio.get_position(code).get("trail_stop"),
                position_pct=(fill_price * filled_qty) / portfolio.total_capital(),
                cash_available=portfolio.available_cash(),
                position_count=portfolio.position_count(),
                trade_id=portfolio.next_trade_id(), env=env_label,
            )
            if tracker is not None:
                try:
                    pos_after = portfolio.get_position(code)
                    tracker.log_entry(
                        trade_id=_trade_id(code, pos_after.get("entry_time")),
                        ticker=code, strategy_name=entry_strategy,
                        strategy_version=config.SYSTEM_VERSION,
                        direction="LONG", price=fill_price, shares=filled_qty,
                        position_value=fill_price * filled_qty,
                        position_pct=(fill_price * filled_qty) / portfolio.total_capital(),
                        cash=portfolio.available_cash(), equity=portfolio.current_equity(),
                        timestamp=pos_after.get("entry_time"),
                        regime_ctx=_trade_regime_ctx(code, ktype, bars),
                        run_id=run_id, atr_entry=result.get("atr", 0.0),
                        sector=config.SECTOR_MAP.get(code, "other"),
                        market_environment=weather_code, entry_rank=entry_rank,
                        risk_per_trade=config.RISK_PER_TRADE_PCT,
                        market=infer_market(code),
                        execution="PAPER" if is_jp_pass else "REAL",
                    )
                except Exception as exc:
                    alert.log(f"trade_tracker: log_entry failed {code} — {exc}")
                try:
                    tracker.log_market_context(
                        trade_id=_trade_id(code, pos_after.get("entry_time")),
                        ctx=market_context.get_latest_context(),
                    )
                except Exception as exc:
                    alert.log(f"trade_tracker: log_market_context failed {code} — {exc}")
                try:
                    # v2.9.x Entry Quality Tracking — observation only, see
                    # engine/entry_quality.py's module docstring. Re-fetches
                    # kline (same bounded, try/except-wrapped pattern as
                    # _trade_regime_ctx's fallback fetch above) rather than
                    # threading `df` through scan()/pipeline.py, so this can
                    # never perturb the signal/candidate data those use.
                    from data.fetcher import fetch_kline
                    df_eq = fetch_kline(code, ktype=ktype, bars=bars)
                    snapshot = entry_quality.build_entry_quality_snapshot(
                        df_eq, entry_price=fill_price,
                        donchian_breakout_price=result.get("donchian_high"),
                        atr_at_entry=result.get("atr"),
                        rule_score=cand.total_score,
                        confidence_score=cand.confidence,
                    )
                    if snapshot is not None:
                        tracker.log_entry_quality(
                            trade_id=_trade_id(code, pos_after.get("entry_time")),
                            **snapshot,
                        )
                except Exception as exc:
                    alert.log(f"trade_tracker: log_entry_quality failed {code} — {exc}")
                try:
                    # v2.11.1 Trade Research Snapshot — permanent, write-once
                    # record of what the system saw at trade time (see
                    # engine/research_snapshot.py's module docstring). Purely
                    # additive/observational, same never-block-the-trade
                    # contract as the three hooks above — the BUY has already
                    # filled by the time this runs.
                    snapshot = research_snapshot.build_trade_research_snapshot(
                        code, result, cand, signal_time=pos_after.get("entry_time"),
                    )
                    tracker.log_research_snapshot(
                        trade_id=_trade_id(code, pos_after.get("entry_time")),
                        snapshot=snapshot,
                    )
                except Exception as exc:
                    alert.log(f"trade_tracker: log_research_snapshot failed {code} — {exc}")

    # ── 2d. QQQ Beta 底仓：固定目标仓位，站上MA200时买入/补仓到目标比例 ──────
    # 不再看活跃仓位数量——这是永远划出的固定死仓，不是"信号不够时的填充"。
    # JP Paper pass 跳过这一段——用JP虚拟资金买US.QQQ底仓没有意义。
    # 2026-08-12 fix: 原来是"只要没持有就一次性买入"语义——如果建仓当天
    # 现金紧张（被2c的活跃信号仓位先占用），QQQ底仓就会永久卡在一个远低于
    # 目标的仓位上（实测卡在1.2%，目标25%），之后哪怕现金充裕了也不会补。
    # 改成按 shortfall = 目标市值 − 当前市值 逐步补仓，每次run现金充裕时
    # 都会继续往目标比例补，直到到位；跌破MA200时shortfall天然按0处理
    # （见下方else分支——不追高也不在熊市补仓）。
    qqq_position   = None if is_jp_pass else portfolio.get_position(config.QQQ_CORE_CODE)
    qqq_held_value = (qqq_position["avg_cost"] * qqq_position["qty"]) if qqq_position else 0.0

    if not is_jp_pass and not macro_block:
        if _qqq_above_ma():
            target_value = config.QQQ_CORE_TARGET_PCT * portfolio.total_capital()
            shortfall    = target_value - qqq_held_value
            qqq_price    = get_price(config.QQQ_CORE_CODE) if shortfall > 0 else 0.0
            spend        = min(portfolio.available_cash() * 0.98, shortfall) if shortfall > 0 else 0.0
            # v2.10 现金红线：跟2c的活跃仓位买入同一约束，不能把broker真实现金
            # 买成负数——remaining_broker_cash为None（本轮broker取数失败）时不
            # 加这层，risk_block已经把macro_block整体挡死。
            if remaining_broker_cash is not None:
                spend = min(spend, remaining_broker_cash)
            if qqq_price > 0 and spend > qqq_price:
                qty = int(spend / qqq_price)
                if qty > 0:
                    action = "补仓" if qqq_position is not None else "买入"
                    alert.info(f"QQQ底仓{action} {qty}股 @{qqq_price:.2f}"
                               f"（目标仓位{config.QQQ_CORE_TARGET_PCT:.0%}，"
                               f"当前{qqq_held_value / portfolio.total_capital():.1%}，"
                               f"站上MA{config.QQQ_MA_PERIOD}）")
                    fill = _place_order(
                        code=config.QQQ_CORE_CODE, side="BUY",
                        qty=qty, price=qqq_price,
                        trd_env=trd_env, env_label=env_label, confirmed=confirmed,
                    )
                    if confirmed and fill["dealt_qty"] > 0:
                        filled_qty = int(fill["dealt_qty"])
                        fill_price = fill["dealt_avg_price"] or qqq_price
                        if remaining_broker_cash is not None:
                            remaining_broker_cash -= fill_price * filled_qty
                        if qqq_position is not None:
                            # 补仓走pyramid同款的add_to_position——跟2b的scale-in
                            # 一样只更新qty/avg_cost，不新开Trade Intelligence DB
                            # 记录（那条记录的trade_id绑定在原始entry_time上）。
                            portfolio.add_to_position(
                                config.QQQ_CORE_CODE, fill_price, filled_qty,
                                new_strength=1.0, strategy="core_etf",
                            )
                            alert.trade_buy(
                                config.QQQ_CORE_CODE, fill_price, filled_qty,
                                sector="etf", score=None, score_label="CORE_ETF",
                                stop_price=None,
                                position_pct=(qqq_held_value + fill_price * filled_qty) / portfolio.total_capital(),
                                cash_available=portfolio.available_cash(),
                                position_count=portfolio.position_count(),
                                trade_id=portfolio.next_trade_id(), env=env_label,
                            )
                        else:
                            portfolio.open_position(
                                config.QQQ_CORE_CODE, "BUY", fill_price, filled_qty,
                                signal_strength=1.0, strategy="core_etf",
                            )
                            alert.trade_buy(
                                config.QQQ_CORE_CODE, fill_price, filled_qty,
                                sector="etf", score=None, score_label="CORE_ETF",
                                stop_price=None,
                                position_pct=(fill_price * filled_qty) / portfolio.total_capital(),
                                cash_available=portfolio.available_cash(),
                                position_count=portfolio.position_count(),
                                trade_id=portfolio.next_trade_id(), env=env_label,
                            )
                            if tracker is not None:
                                try:
                                    pos_after = portfolio.get_position(config.QQQ_CORE_CODE)
                                    tracker.log_entry(
                                        trade_id=_trade_id(config.QQQ_CORE_CODE,
                                                           pos_after.get("entry_time")),
                                        ticker=config.QQQ_CORE_CODE, strategy_name="core_etf",
                                        strategy_version=config.SYSTEM_VERSION,
                                        direction="LONG", price=fill_price, shares=filled_qty,
                                        position_value=fill_price * filled_qty,
                                        position_pct=(fill_price * filled_qty) / portfolio.total_capital(),
                                        cash=portfolio.available_cash(),
                                        equity=portfolio.current_equity(),
                                        timestamp=pos_after.get("entry_time"),
                                        regime_ctx=_trade_regime_ctx(
                                            config.QQQ_CORE_CODE, ktype, bars),
                                        run_id=run_id, sector="etf",
                                        market_environment=weather_code,
                                        market="US", execution="REAL",
                                    )
                                except Exception as exc:
                                    alert.log(f"trade_tracker: log_entry failed "
                                              f"{config.QQQ_CORE_CODE} — {exc}")
                                try:
                                    tracker.log_market_context(
                                        trade_id=_trade_id(
                                            config.QQQ_CORE_CODE,
                                            pos_after.get("entry_time")),
                                        ctx=market_context.get_latest_context(),
                                    )
                                except Exception as exc:
                                    alert.log(f"trade_tracker: log_market_context "
                                              f"failed {config.QQQ_CORE_CODE} — {exc}")
        else:
            alert.log(f"runner: QQQ Beta floor skip — QQQ below MA{config.QQQ_MA_PERIOD}"
                      f"  (bear market, stay in cash)")

    alert.log("runner: pass complete")
    portfolio.print_summary()


def run_loop(
    strategy_name: str = "combined",
    codes: Optional[List[str]] = None,
    ktype: str = "K_DAY",
    bars: int = 120,
    interval_seconds: int = 300,
    confirmed: bool = False,
    use_real: bool = False,
    auto_route: bool = False,
) -> None:
    """Block and call run_once every interval_seconds.

    v2.8: generates one run_id for this whole loop invocation and threads it
    through every run_once() pass, so every trade from this trading session
    shares one metadata row in the Trade Intelligence Database — best
    effort, never blocks the loop from starting if it fails."""
    run_id = uuid.uuid4().hex
    try:
        params = default_parameter_snapshot()
        TradeTracker().log_run_metadata(
            run_id=run_id, strategy_version=config.SYSTEM_VERSION,
            market="SIMULATE" if not use_real else config.TRD_ENV,
            parameter_hash=compute_parameter_hash(params),
        )
    except Exception as exc:
        alert.log(f"trade_tracker: log_run_metadata failed — {exc}")

    alert.info(f"kabu 监控已启动，每{interval_seconds // 60}分钟扫描一次")
    while True:
        _write_heartbeat()

        # Checked before run_once() (not after) so the notification fires the
        # moment the open window is entered, rather than waiting for the
        # ~5min scan of the full watchlist to finish first.
        if should_notify_open():
            alert.info("美股开盘，开始扫描")

        # v2.9 Market Context Logging — once per ET calendar day (08:30 ET,
        # before market open), never on the 5-minute scan cadence. Failure
        # here must never stop the scan loop (see engine/market_context.py
        # module docstring).
        try:
            if market_context.should_run_daily_update():
                ctx = market_context.update_market_context()
                alert.log(f"market_context: daily update saved — "
                          f"{ctx.get('data_status')}")
        except Exception as exc:
            alert.log(f"market_context: daily update failed — {exc}")

        try:
            run_once(
                strategy_name=strategy_name,
                codes=codes,
                ktype=ktype,
                bars=bars,
                confirmed=confirmed,
                use_real=use_real,
                auto_route=auto_route,
                run_id=run_id,
            )
        except KeyboardInterrupt:
            alert.info("kabu 监控已手动停止")
            break
        except Exception as exc:
            alert.error(f"本轮扫描出现未处理异常 — {exc}")

        if should_notify_close():
            alert.warn("美股收盘，停止扫描")

        alert.log(f"runner: sleeping {interval_seconds}s …")
        time.sleep(interval_seconds)


# ── Internal helpers ──────────────────────────────────────────────────────────

_HEARTBEAT_PATH = Path(__file__).resolve().parent.parent / ".kabu_heartbeat"


def _write_heartbeat() -> None:
    """Timestamp written at the top of every run_loop() iteration — read by
    the external watchdog.py to detect a hung process (one that's still
    running but stopped making progress, e.g. blocked forever inside a
    moomoo API call with no timeout). Written before run_once() rather than
    after, so a pass that never returns leaves a stale-but-present timestamp
    for the watchdog to compare against, instead of no file at all."""
    try:
        _HEARTBEAT_PATH.write_text(datetime.now().isoformat())
    except OSError:
        pass


def _trade_id(code: str, entry_time_iso: str) -> str:
    """v2.6 Trade Intelligence Database round-trip key — must be derived the
    same way at both open and close time without adding any new field to
    Portfolio's position dict. entry_time is already unique per fill
    (microsecond-precision ISO string set by Portfolio.open_position), so
    f"{code}_{entry_time}" is a stable trade_id recoverable from the dict
    portfolio.close_position() returns (it always includes entry_time)."""
    return f"{code}_{entry_time_iso}"


def _trade_regime_ctx(code: str, ktype: str = "K_DAY", bars: int = 120):
    """Best-effort regime snapshot for trade_tracker attribution. v2.8:
    prefers engine.regime_store's HMM output (engine/hmm_shadow.py's daily
    batch) — tried first with no kline fetch at all. Only falls back to a
    fresh fetch_kline() + the rule-based classifier when the HMM store has
    no entry yet for `code`. Never raises — a failed regime fetch must not
    block or crash a trading pass, so this degrades to regime_ctx=None (a
    NULL attribution row) on any error."""
    try:
        ctx = build_regime_ctx_preferring_hmm(code)
        if ctx is not None:
            return ctx
        from data.fetcher import fetch_kline
        df = fetch_kline(code, ktype=ktype, bars=bars)
        return build_regime_ctx_preferring_hmm(code, df)
    except Exception:
        return None


def _qqq_above_ma() -> bool:
    """
    True if QQQ's last close is above its MA(config.QQQ_MA_PERIOD) — the sole
    gate for the fixed Beta-floor position (config.QQQ_CORE_TARGET_PCT).
    Used both to decide whether to buy the floor back and whether to exit it.
    """
    try:
        from data.fetcher import fetch_kline
        qqq_df = fetch_kline(config.QQQ_CORE_CODE, ktype="K_DAY",
                             bars=config.QQQ_MA_PERIOD + 5)
        qqq_close = qqq_df["close"].astype(float)
        return float(qqq_close.iloc[-1]) > float(qqq_close.tail(config.QQQ_MA_PERIOD).mean())
    except Exception:
        return False


def _day_ordinal(entry_time_iso) -> int:
    """把 entry_time（ISO字符串）换算成可比较的整数序数，供
    portfolio.capacity_manager.evaluate_replacement() 的
    current_day_idx 参数使用——只用于多个OBSERVATION候选打平时按持仓时长
    排序（tie-break），不影响主判断，用日历日代替回测里的交易日不影响
    正确性。解析失败/缺失时退化为"今天"（等价于holding_days=0，不会被
    优先换出，是保守的默认值，不是报错）。"""
    if entry_time_iso:
        try:
            return datetime.fromisoformat(entry_time_iso).toordinal()
        except (ValueError, TypeError):
            pass
    return datetime.now().toordinal()


def _days_held(entry_time_iso) -> int:
    """Calendar days between entry_time (ISO string) and now, for the SELL
    notification's "持仓天数" field. Falls back to 0 if unparseable."""
    if entry_time_iso:
        try:
            return (datetime.now() - datetime.fromisoformat(entry_time_iso)).days
        except (ValueError, TypeError):
            pass
    return 0


def _run_emergency_rebalance(portfolio: Portfolio, results: dict, trd_env, env_label: str,
                             confirmed: bool, ktype: str, bars: int,
                             tracker: Optional[TradeTracker] = None) -> List[dict]:
    """v2.10.1 EMERGENCY-tier rebalance executor.

    confirmed=False (the automatic loop's default until config.
    PORTFOLIO_RISK_EMERGENCY_AUTO_EXECUTE is turned on, or a caller like
    execute_emergency_rebalance.py passes True directly) — preview mode:
    computes ONE static plan against the current broker snapshot and walks
    every leg through _place_order(confirmed=False), which only logs
    "[DRY RUN] would SELL ..." and never touches state. No lock needed
    (nothing real happens), no re-fetching between legs (there's nothing to
    re-verify since nothing changed).

    confirmed=True — real execution: iterative loop, one leg at a time —
    fetch fresh broker state, confirm still EMERGENCY, (re)compute the plan
    from that fresh snapshot, execute only its first leg, then loop back to
    fetch again. This is deliberately NOT "compute the whole plan once and
    execute every leg" (that was v2.10's original design) — a frozen batch
    plan can't account for a partial fill on an earlier leg or price drift
    between orders (see the 2026-08-29 rebalance preview report). Every
    iteration re-verifies against real broker qty before sizing the sell
    (belt-and-suspenders on top of plan_rebalance() already using a fresh
    snapshot), books the REAL dealt_qty into tracker.py (never the planned
    qty), and stops immediately (fail-stop, no retry-forever) on a broker
    query failure, a zero fill, or hitting config.
    PORTFOLIO_REBALANCE_MAX_ORDERS_PER_RUN. Wrapped in a file lock
    (risk/portfolio_risk_manager.try_acquire_rebalance_lock()) so this can
    never overlap with another real execution (the automatic loop and a
    manually-run execute_emergency_rebalance.py, or two manual runs).
    Finishes with one fresh post-trade broker re-fetch, logged as an
    explicit before/after summary.

    Returns the list of {"code","sell_qty","price","reason"} dicts actually
    executed (or, in preview mode, that WOULD be executed) — for
    portfolio_risk_manager.log_snapshot()'s audit trail.
    """
    target_pct = config.PORTFOLIO_RISK_REBALANCE_TARGET

    if not confirmed:
        try:
            bs = broker_state_mod.fetch_broker_state(trd_env)
        except Exception as exc:
            alert.error(f"再平衡预览：broker状态查询失败，跳过本轮预览 — {exc}")
            return []
        plan = portfolio_risk_manager.plan_rebalance(
            bs, portfolio.data["positions"], target_pct=target_pct, results=results)
        previewed = []
        for order in plan:
            alert.info(f"[预览] 再平衡将卖出 {order.code} {order.sell_qty}股 "
                      f"@{order.price:.4f}（{order.reason}，优先级{order.priority}）")
            _place_order(code=order.code, side="SELL", qty=order.sell_qty, price=order.price,
                        trd_env=trd_env, env_label=env_label, confirmed=False)
            previewed.append({"code": order.code, "sell_qty": order.sell_qty,
                              "price": order.price, "reason": order.reason})
        return previewed

    lock_state = portfolio_risk_manager.try_acquire_rebalance_lock()
    if lock_state == portfolio_risk_manager.LOCK_STALE:
        alert.error("再平衡：发现一个超时未释放的执行锁（可能是上次执行崩溃/卡死），"
                    r"为避免掩盖异常状态不会自动清除——需要人工确认账户真实状态后"
                    r"手动删除 C:\KabuData\portfolio\rebalance_lock.json 才会继续")
        return []
    if lock_state == portfolio_risk_manager.LOCK_ACTIVE:
        alert.warn("再平衡：已有一次执行正在进行中（或锁文件无法读取），本轮跳过，避免重复下单")
        return []

    executed_orders: List[dict] = []
    before_assessment = None
    try:
        for i in range(config.PORTFOLIO_REBALANCE_MAX_ORDERS_PER_RUN):
            try:
                bs = broker_state_mod.fetch_broker_state(trd_env)
            except Exception as exc:
                alert.error(f"再平衡：broker状态查询失败，停止本轮再平衡 — {exc}")
                break
            assessment = portfolio_risk_manager.evaluate(bs)
            if before_assessment is None:
                before_assessment = assessment
            if assessment.tier != portfolio_risk_manager.TIER_EMERGENCY:
                alert.log(f"再平衡：仓位已回落到{assessment.exposure_pct:.1%}"
                          f"（{assessment.tier}），停止（本轮已执行{len(executed_orders)}笔）")
                break
            plan = portfolio_risk_manager.plan_rebalance(
                bs, portfolio.data["positions"], target_pct=target_pct, results=results)
            if not plan:
                alert.log("再平衡：仍处于EMERGENCY但已无可执行的减仓计划（可能都低于最小交易额），停止")
                break

            order = plan[0]
            bpos = bs.positions.get(order.code)
            broker_qty = int(bpos.qty) if bpos is not None else 0
            sell_qty = min(order.sell_qty, broker_qty)   # never oversell vs. just-refetched real qty
            if sell_qty <= 0:
                alert.error(f"再平衡：{order.code}计划卖出但broker真实股数为{broker_qty}，"
                            f"停止本轮再平衡")
                break

            alert.info(f"再平衡卖出 {order.code} {sell_qty}股 @{order.price:.4f}"
                      f"（{order.reason}，优先级{order.priority}，第{i + 1}笔）")
            fill = _place_order(code=order.code, side="SELL", qty=sell_qty, price=order.price,
                                trd_env=trd_env, env_label=env_label, confirmed=True)
            if fill["dealt_qty"] <= 0:
                alert.error(f"再平衡：{order.code}未成交（{fill['status']}），停止本轮再平衡，"
                            f"等待下次评估重新计算")
                break

            dealt_qty  = int(fill["dealt_qty"])
            exit_price = fill["dealt_avg_price"] or order.price
            closed = portfolio.reduce_position(order.code, exit_price, dealt_qty, reason=order.reason)
            alert.trade_sell(
                order.code, closed.get("avg_cost", exit_price), exit_price, dealt_qty,
                days_held=_days_held(closed.get("entry_time")), reason=order.reason,
                cash_available=portfolio.available_cash(), total_equity=portfolio.current_equity(),
                position_count=portfolio.position_count(), trade_id=portfolio.next_trade_id(), env=env_label,
            )
            if tracker is not None:
                try:
                    tracker.log_exit(
                        trade_id=_trade_id(order.code, closed.get("entry_time")),
                        price=exit_price, cash=portfolio.available_cash(),
                        equity=portfolio.current_equity(), timestamp=datetime.now().isoformat(),
                        exit_reason=order.reason, regime_ctx=_trade_regime_ctx(order.code, ktype, bars),
                    )
                except Exception as exc:
                    alert.log(f"trade_tracker: log_exit failed {order.code} — {exc}")
            executed_orders.append({"code": order.code, "sell_qty": dealt_qty,
                                    "price": exit_price, "reason": order.reason})
            if dealt_qty < sell_qty:
                alert.warn(f"再平衡：{order.code}部分成交 {dealt_qty}/{sell_qty}股，"
                          f"下一轮循环会用最新broker状态重新计算剩余缺口")
        else:
            alert.error(f"再平衡：达到单轮最大下单数上限"
                        f"({config.PORTFOLIO_REBALANCE_MAX_ORDERS_PER_RUN})仍未回落到EMERGENCY以下，"
                        f"停止并等待人工检查")

        try:
            bs_after = broker_state_mod.fetch_broker_state(trd_env)
            after = portfolio_risk_manager.evaluate(bs_after)
            before_desc = f"{before_assessment.exposure_pct:.1%}" if before_assessment else "?"
            alert.log(f"再平衡结束：本轮共{len(executed_orders)}笔  "
                      f"仓位 {before_desc} -> {after.exposure_pct:.1%}（{after.tier}）  "
                      f"现金 -> ${after.cash:,.0f}")
            if after.tier == portfolio_risk_manager.TIER_EMERGENCY:
                alert.error("再平衡结束后仓位仍处于EMERGENCY，需要人工介入检查")
        except Exception as exc:
            alert.error(f"再平衡后验证查询失败 — {exc}")
    finally:
        if lock_state == portfolio_risk_manager.LOCK_ACQUIRED:
            portfolio_risk_manager.release_rebalance_lock()

    return executed_orders


def _attempt_active_replacement(portfolio: Portfolio, incoming_code: str, incoming_score: float,
                                trd_env, env_label: str, confirmed: bool,
                                results: dict, tracker: Optional[TradeTracker] = None,
                                ktype: str = "K_DAY", bars: int = 120) -> Optional[str]:
    """v2.3/v2.4 Portfolio Capacity Manager 移植进实盘：MAX_POSITIONS已满
    时，尝试找一个可换出的持仓、真正执行SELL，为新的FULL信号腾出名额。
    判断逻辑调用 portfolio.capacity_manager.evaluate_replacement()——跟
    backtest_portfolio.py非RSL分支用的是**同一个函数**，不是重新实现或
    复制的一份规则，REPLACEMENT_MARGIN/REPLACEMENT_MIN_NEW_SCORE/
    REPLACEMENT_BLOCK_SAME_SECTOR 三个配置在回测/模拟盘/实盘（后两者都
    走这条 runner.py 路径，只是 trd_env 不同）三条路径下行为完全一致
    （由 engine/test_runner_replacement.py 的跨路径一致性测试保证）。

    已知的一处不完整对齐：本函数不传 full_score_history（runner.py 未
    维护滚动FULL分数历史），所以WEAK_FULL独立通道（config.
    REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED）即使打开也不会在这里触发
    ——该开关当前默认False，不影响现行为；如果以后要把这个开关也打开到
    实盘，需要先给runner.py补上历史追踪再移除这个限制。

    返回被换出的code表示置换成立（调用方应紧接着往下走已有的敞口/板块/
    sizing/BUY逻辑，就像这个名额本来就空着一样）；返回None表示没有可换的
    victim（不存在合格候选、分数优势不够，或被三个消融过滤门拦截），
    调用方应回退到原有的"容量已满，放弃开仓"行为（break）。

    跟其余退出路径一样：现在统一按**dealt_qty是否覆盖持仓量**门控是否
    调用close_position——SELL未成交/未完全成交时不会把本地仓位清掉但
    broker其实没成交（或只成交一部分），这次置换机会直接作废（返回
    None），不会强行认为名额已经腾出。dry-run（confirmed=False）下只打日志、不touch
    任何状态，跟runner.py其余下单路径的既有约定一致。
    """
    held = {c: p for c, p in portfolio.data["positions"].items()
            if p.get("strategy") != "core_etf"}
    today_ord = datetime.now().toordinal()
    held_for_review = {
        c: {**p, "entry_day_idx": _day_ordinal(p.get("entry_time"))}
        for c, p in held.items()
    }
    evaluation = capacity_manager.evaluate_replacement(
        incoming_code=incoming_code, incoming_score=incoming_score,
        held_positions=held_for_review, current_day_idx=today_ord)
    if evaluation.decision != "REPLACE":
        return None
    victim_code = evaluation.victim_code

    victim_pos = held[victim_code]
    victim_price = results.get(victim_code, {}).get("current_price") or get_price(victim_code)
    if not victim_price or victim_price <= 0:
        alert.warn(f"置换候选{victim_code}无法获取价格，本次置换取消")
        return None

    alert.warn(f"主动置换：卖出{victim_code}（评分{victim_pos.get('total_score')}）"
               f"为新的满分信号（评分{incoming_score}）腾出仓位")
    fill = _place_order(
        code=victim_code, side="SELL", qty=victim_pos["qty"], price=victim_price,
        trd_env=trd_env, env_label=env_label, confirmed=confirmed,
    )
    if confirmed and fill["dealt_qty"] < victim_pos["qty"]:
        alert.error(f"主动置换卖出{victim_code}未完全成交"
                    f"（{fill['dealt_qty']:.0f}/{victim_pos['qty']}股，{fill['status']}），"
                    f"置换取消，持仓保持不变")
        return None
    if confirmed and fill["dealt_qty"] >= victim_pos["qty"]:
        exit_price = fill["dealt_avg_price"] or victim_price
        closed = portfolio.close_position(victim_code, exit_price, reason="ACTIVE_REPLACEMENT")
        alert.trade_sell(
            victim_code, closed["avg_cost"], exit_price, closed["qty"],
            days_held=_days_held(closed.get("entry_time")),
            reason="ACTIVE_REPLACEMENT",
            cash_available=portfolio.available_cash(),
            total_equity=portfolio.current_equity(),
            position_count=portfolio.position_count(),
            trade_id=portfolio.next_trade_id(), env=env_label,
        )
        if tracker is not None:
            try:
                tracker.log_exit(
                    trade_id=_trade_id(victim_code, closed.get("entry_time")),
                    price=exit_price, cash=portfolio.available_cash(),
                    equity=portfolio.current_equity(),
                    timestamp=datetime.now().isoformat(),
                    exit_reason="ACTIVE_REPLACEMENT",
                    regime_ctx=_trade_regime_ctx(victim_code, ktype, bars),
                )
            except Exception as exc:
                alert.log(f"trade_tracker: log_exit failed {victim_code} — {exc}")
    return victim_code


def _place_order(
    code: str,
    side: str,
    qty: int,
    price: float,
    trd_env,
    env_label: str,
    confirmed: bool,
) -> dict:
    """
    Place an order via whichever broker engine.broker.get_broker(code)
    resolves for this code (real moomoo order for US/EU codes, virtual
    paper fill for JP codes — see engine/broker.py). Returns a fill dict
    {"order_id", "dealt_qty", "dealt_avg_price", "status"} — callers MUST
    gate on dealt_qty > 0, not on order_id, before touching portfolio state
    (order_id only means the broker accepted the order, not that it filled;
    see engine/broker.py module docstring for the incident this fixed).
    """
    if confirmed and config.PREFLIGHT_ENABLED:
        result = market_preflight.check(code)
        if not result.ok:
            alert.error(f"{code} preflight检查未通过（{result.reason}），取消下单")
            return {"order_id": "", "dealt_qty": 0.0, "dealt_avg_price": 0.0,
                    "status": "PREFLIGHT_FAILED"}

    broker = get_broker(code)
    return broker.place_order(
        code=code, side=side, qty=qty, price=price,
        trd_env=trd_env, env_label=env_label, confirmed=confirmed,
    )
