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
from common import make_trade_ctx, safe_close, infer_market, parse_trd_env
from data.fetcher import get_price
from engine.scanner import scan, smart_scan, rank_signals
from engine import news as news_sentiment
from engine import news_filter
from engine import fundamental
from engine import scoring
from engine import regime
from engine import market_weather
from engine.market_hours import filter_open, should_notify_close, is_daytime_jst
from risk.earnings import is_earnings_blackout
from portfolio import capacity_manager
from portfolio.tracker import Portfolio
from risk import guard, sizing
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
    process launch."""
    return config.WATCHLIST_ASIA_PACIFIC if is_daytime_jst() else config.WATCHLIST_EUROPE_US


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

    # ── Determine trade environment ───────────────────────────────────────────
    if use_real and confirmed:
        trd_env = parse_trd_env(config.TRD_ENV)
        env_label = config.TRD_ENV
    else:
        import moomoo as ft
        trd_env = ft.TrdEnv.SIMULATE
        env_label = "SIMULATE"

    # v2.1 多因子总分模型的 env 参数（engine.fundamental / engine.scoring）：
    # 只有 SIMULATE(模拟盘)/真实两档，backtest_portfolio.py 是完全独立的脚本
    # 入口，走的是 env="backtest"，不会经过这里。
    score_env = "paper" if env_label == "SIMULATE" else "live"

    mode_label = "AUTO-ROUTE" if auto_route else strategy_name
    alert.log(f"runner: starting pass  env={env_label}  mode={mode_label}"
              f"  dry_run={not confirmed}")

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

    watchlist = codes or select_watchlist()

    # Filter to markets currently open (prevents dead scans on closed exchanges)
    open_codes = filter_open(watchlist)
    skipped = len(watchlist) - len(open_codes)
    if skipped:
        alert.log(f"runner: {skipped} code(s) skipped — market closed")
    if not open_codes:
        alert.log("runner: no markets open — pass skipped")
        return

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
                order_id = _place_order(
                    code=code, side="SELL", qty=pos["qty"], price=price,
                    trd_env=trd_env, env_label=env_label, confirmed=confirmed,
                )
                if confirmed and order_id:
                    closed = portfolio.close_position(code, price, reason="MA200_BREAK")
                    alert.trade_sell(
                        code, closed["avg_cost"], price, closed["qty"],
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
                                price=price, cash=portfolio.available_cash(),
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
            order_id = _place_order(
                code=code,
                side="SELL",
                qty=pos["qty"],
                price=price,
                trd_env=trd_env,
                env_label=env_label,
                confirmed=confirmed,
            )
            if confirmed and order_id:
                closed = portfolio.close_position(code, price, reason=reason)
                alert.trade_sell(
                    code, closed["avg_cost"], price, closed["qty"],
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
                            price=price, cash=portfolio.available_cash(),
                            equity=portfolio.current_equity(),
                            timestamp=datetime.now().isoformat(),
                            exit_reason=reason,
                            regime_ctx=_trade_regime_ctx(code, ktype, bars),
                        )
                    except Exception as exc:
                        alert.log(f"trade_tracker: log_exit failed {code} — {exc}")

    # ── Macro news circuit breaker (checked once per pass) ───────────────────
    macro_block = news_sentiment.macro_circuit_breaker()
    if macro_block:
        alert.warn(f"新闻面熔断触发（{macro_block}），本轮不再开新仓")
    if not macro_block and drawdown_halt:
        macro_block = "MAX_DRAWDOWN_HALT: realized drawdown exceeded threshold"

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
            order_id = _place_order(
                code=code, side="BUY", qty=add_qty, price=price,
                trd_env=trd_env, env_label=env_label, confirmed=confirmed,
            )
            if confirmed and order_id:
                portfolio.add_to_position(code, price, add_qty, strength,
                                          strategy="atr_breakout")
                alert.trade_buy(
                    code, price, add_qty,
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
            order_id = _place_order(
                code=code, side="BUY", qty=add_qty, price=price,
                trd_env=trd_env, env_label=env_label, confirmed=confirmed,
            )
            if confirmed and order_id:
                portfolio.add_to_position(code, price, add_qty, new_str)
                alert.trade_buy(
                    code, price, add_qty,
                    sector=config.SECTOR_MAP.get(code, "other"),
                    score=pos.get("total_score"), score_label=pos.get("score_label"),
                    stop_price=pos.get("trail_stop"),
                    position_pct=(pos["avg_cost"] * pos["qty"]) / portfolio.total_capital(),
                    cash_available=portfolio.available_cash(),
                    position_count=portfolio.position_count(),
                    trade_id=portfolio.next_trade_id(), env=env_label,
                )

    # ── 2c. Open new positions — ranked by signal strength ────────────────────
    buy_signals = rank_signals(results, signal="BUY")
    # Active count excludes QQQ core position
    active_count = sum(1 for p in portfolio.data["positions"].values()
                       if p.get("strategy") != "core_etf")
    slots_free   = config.MAX_POSITIONS - active_count
    alert.log(f"runner: {len(buy_signals)} BUY signal(s) found, "
              f"{slots_free} slot(s) available  (active={active_count})")

    for ranked in buy_signals:
        code   = ranked["code"]
        result = ranked

        if portfolio.get_position(code):
            alert.log(f"runner: {code} already held — skip")
            continue

        if portfolio.is_cooldown(code):
            alert.log(f"runner: {code} in TRENDING_EARLY stop-out cooldown — skip")
            continue

        # Minimum signal quality gate — filter weak signals to reduce commission drag
        if result.get("signal_strength", 0) < config.MIN_ENTRY_STRENGTH:
            alert.log(f"runner: {code} strength {result.get('signal_strength',0):.0%} "
                      f"< {config.MIN_ENTRY_STRENGTH:.0%} threshold — skip")
            continue

        if macro_block:
            alert.log(f"runner: {code} skipped — macro circuit breaker active")
            break

        # Earnings blackout — no new positions within ±1 day of earnings
        if is_earnings_blackout(code):
            alert.warn(f"{code} 处于财报窗口期，跳过开仓")
            continue

        # ── v2.1 横向多因子总分：趋势(40%)+基本面(20%)+新闻(20%)+天气(20%) ──
        # 见 engine/scoring.py。新闻/基本面都走真实API（各自4h缓存，见
        # engine/news_filter.py::classify_code / engine/fundamental.py），
        # 不是 backtest_portfolio.py 里那个固定中性分的版本。
        strength = result.get("signal_strength", 0.5)
        news_result = news_filter.classify_code(code)
        fund_result = fundamental.score(code, env=score_env)
        total = scoring.compute_total_score(
            trend_strength=strength,
            weather_code=weather_code,
            fundamental_score=fund_result["score"],
            news_score=news_result["score"],
        )

        def _fmt(v):   # score components can be None (missing data, excluded/renormalized)
            return f"{v:5.1f}" if v is not None else "  N/A"

        alert.log(
            f"SCORE {code:8s} trend={_fmt(total.trend_score)} "
            f"fund={_fmt(total.fundamental_score)} news={_fmt(total.news_score)} "
            f"weather={_fmt(total.weather_score)} total={_fmt(total.total)} "
            f"-> {total.label}"
            + (f"  news_tier={news_result['tier']}({news_result.get('matched_keyword')})"
               if news_result["tier"] != 3 else "")
            + (f"  fund_tier={fund_result['tier']}({fund_result['reason']})"
               if fund_result["tier"] != 3 else "")
        )
        if total.label == scoring.LABEL_SKIP:
            continue   # 总分<40，或天气state0（已在上面macro_block短路，这里是双保险）
        if not guard.can_open_position(portfolio):
            # ── v2.3/v2.4 Portfolio Capacity Manager（主动置换）────────────
            # 名额已满时，只有新信号是FULL才尝试换出一个OBSERVATION持仓
            # 腾出名额，而不是直接放弃——跟backtest_portfolio.py共用同一个
            # portfolio.capacity_manager.evaluate_replacement()判断入口
            # （Single Source of Truth，2026-07-06统一，见该函数docstring）。
            # 找不到victim就维持原有行为（break）。
            victim_code = None
            if config.ENABLE_ACTIVE_REPLACEMENT and total.label == scoring.LABEL_FULL:
                victim_code = _attempt_active_replacement(
                    portfolio, incoming_code=code, incoming_score=total.total,
                    trd_env=trd_env, env_label=env_label, confirmed=confirmed,
                    results=results, tracker=tracker, ktype=ktype, bars=bars,
                )
            if victim_code is None:
                skip_price = result.get("current_price", 0.0)
                alert.warn(f"持仓数已达上限，跳过 {code}"
                           f"（现价{skip_price:.4f}，评分{total.total:.0f}/{total.label}）")
                break
            # 置换成立，名额已腾出——不 break，直接往下走已有的敞口/板块/
            # sizing/BUY逻辑，就像这个名额本来就空着一样。
        if not guard.check_total_exposure(portfolio):
            alert.log(f"runner: total exposure {portfolio.exposure_pct():.0%} "
                      f">= {config.MAX_TOTAL_EXPOSURE_PCT:.0%} — skip new entries")
            break
        if not guard.check_sector_exposure(portfolio, code):
            sector = config.SECTOR_MAP.get(code, "other")
            alert.warn(f"{code} 所属板块({sector})仓位占比"
                       f"{portfolio.sector_exposure_pct(sector):.0%}"
                       f"已达上限{config.MAX_SECTOR_EXPOSURE_PCT:.0%}，跳过")
            continue

        price = result.get("current_price", 0.0)
        if price <= 0:
            alert.warn(f"{code} 无法获取价格，跳过")
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
        position_scale = trial_scale * total.position_scale
        qty = sizing.calculate(
            portfolio.available_cash(), price, strength,
            total_capital=portfolio.total_capital(),
            stop_loss_pct=stop_pct,
            kelly_factor=kelly_factor,
            strategy=entry_strategy,
            open_positions=sum(1 for p in portfolio.data["positions"].values()
                               if p.get("strategy") != "core_etf"),
            rsi_val=result.get("rsi14"),
            position_scale=position_scale,
            market_weather_code=weather_code,
            score_label=total.label,
        )
        if qty <= 0:
            alert.warn(f"{code} 现价{price:.4f}下可买股数为0，跳过")
            continue

        entry_rank = buy_signals.index(ranked) + 1
        alert.info(f"准备买入 {code}，{qty}股 @{price:.4f}，"
                   f"信号强度{result.get('signal_strength', 0):.0%}，"
                   f"排名#{entry_rank}")
        order_id = _place_order(
            code=code,
            side="BUY",
            qty=qty,
            price=price,
            trd_env=trd_env,
            env_label=env_label,
            confirmed=confirmed,
        )
        if confirmed and order_id:
            # Persist entry_atr so ATR trailing stop can be reconstructed after restart.
            # score_label/total_score persisted too so this position can itself be
            # considered as a future Active Replacement victim (see
            # _attempt_active_replacement above).
            portfolio.open_position(code, "BUY", price, qty, strength, entry_strategy,
                                    entry_atr=result.get("atr", 0.0),
                                    score_label=total.label, total_score=total.total)
            alert.trade_buy(
                code, price, qty,
                sector=config.SECTOR_MAP.get(code, "other"),
                score=total.total, score_label=total.label,
                stop_price=portfolio.get_position(code).get("trail_stop"),
                position_pct=(price * qty) / portfolio.total_capital(),
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
                        direction="LONG", price=price, shares=qty,
                        position_value=price * qty,
                        position_pct=(price * qty) / portfolio.total_capital(),
                        cash=portfolio.available_cash(), equity=portfolio.current_equity(),
                        timestamp=pos_after.get("entry_time"),
                        regime_ctx=_trade_regime_ctx(code, ktype, bars),
                        run_id=run_id, atr_entry=result.get("atr", 0.0),
                        sector=config.SECTOR_MAP.get(code, "other"),
                        market_environment=weather_code, entry_rank=entry_rank,
                        risk_per_trade=config.RISK_PER_TRADE_PCT,
                    )
                except Exception as exc:
                    alert.log(f"trade_tracker: log_entry failed {code} — {exc}")

    # ── 2d. QQQ Beta 底仓：固定目标仓位，只要不在持有中且 QQQ>MA200 就买回 ──
    # 不再看活跃仓位数量——这是永远划出的固定死仓，不是"信号不够时的填充"。
    qqq_held = portfolio.get_position(config.QQQ_CORE_CODE) is not None

    if not qqq_held and not macro_block:
        if _qqq_above_ma():
            qqq_price = get_price(config.QQQ_CORE_CODE)
            target_value = config.QQQ_CORE_TARGET_PCT * portfolio.total_capital()
            spend        = min(portfolio.available_cash() * 0.98, target_value)
            if qqq_price > 0 and spend > qqq_price:
                qty = int(spend / qqq_price)
                if qty > 0:
                    alert.info(f"QQQ底仓买入 {qty}股 @{qqq_price:.2f}"
                               f"（目标仓位{config.QQQ_CORE_TARGET_PCT:.0%}，"
                               f"站上MA{config.QQQ_MA_PERIOD}）")
                    order_id = _place_order(
                        code=config.QQQ_CORE_CODE, side="BUY",
                        qty=qty, price=qqq_price,
                        trd_env=trd_env, env_label=env_label, confirmed=confirmed,
                    )
                    if confirmed and order_id:
                        portfolio.open_position(
                            config.QQQ_CORE_CODE, "BUY", qqq_price, qty,
                            signal_strength=1.0, strategy="core_etf",
                        )
                        alert.trade_buy(
                            config.QQQ_CORE_CODE, qqq_price, qty,
                            sector="etf", score=None, score_label="CORE_ETF",
                            stop_price=None,
                            position_pct=(qqq_price * qty) / portfolio.total_capital(),
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
                                    direction="LONG", price=qqq_price, shares=qty,
                                    position_value=qqq_price * qty,
                                    position_pct=(qqq_price * qty) / portfolio.total_capital(),
                                    cash=portfolio.available_cash(),
                                    equity=portfolio.current_equity(),
                                    timestamp=pos_after.get("entry_time"),
                                    regime_ctx=_trade_regime_ctx(
                                        config.QQQ_CORE_CODE, ktype, bars),
                                    run_id=run_id, sector="etf",
                                    market_environment=weather_code,
                                )
                            except Exception as exc:
                                alert.log(f"trade_tracker: log_entry failed "
                                          f"{config.QQQ_CORE_CODE} — {exc}")
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

    跟现有退出循环的SELL不同：这里的SELL用BUY侧那种更严格的**order_id
    成功与否**门控是否调用close_position——SELL静默失败时不会把本地
    仓位清掉但broker其实没成交，这次置换机会直接作废（返回None），不会
    强行认为名额已经腾出。dry-run（confirmed=False）下只打日志、不touch
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
    order_id = _place_order(
        code=victim_code, side="SELL", qty=victim_pos["qty"], price=victim_price,
        trd_env=trd_env, env_label=env_label, confirmed=confirmed,
    )
    if confirmed and not order_id:
        alert.error(f"主动置换卖出{victim_code}失败，置换取消，持仓保持不变")
        return None
    if confirmed and order_id:
        closed = portfolio.close_position(victim_code, victim_price, reason="ACTIVE_REPLACEMENT")
        alert.trade_sell(
            victim_code, closed["avg_cost"], victim_price, closed["qty"],
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
                    price=victim_price, cash=portfolio.available_cash(),
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
) -> str:
    """
    Place a market-price order.  Returns order_id string on success, "" on dry
    run or failure.

    Dry-run mode (confirmed=False): logs intent only, never touches the broker.
    """
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
