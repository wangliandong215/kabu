"""
risk/sizing.py — position size calculation.

Four methods (set via config.SIZING_METHOD):
  fixed_amount   Buy floor(SIZING_AMOUNT / price) shares each trade.
  fixed_percent  Buy floor(available_cash * SIZING_PCT / price) shares.
  fixed_qty      Always buy SIZING_QTY shares regardless of price.
  signal_based   Tiered sizing by signal strength:
                   strong (>=0.7) → SIZING_PCT_STRONG of cash
                   medium (>=0.4) → SIZING_PCT_MEDIUM of cash
                   weak   (<0.4)  → SIZING_PCT_WEAK   of cash (observation)
                 v2.2: when score_label==OBSERVATION (engine.scoring 总分
                 40-59分), uses config.OBSERVATION_POSITION_PCT instead —
                 an independent knob from SIZING_PCT_WEAK above (that one
                 only fires on the legacy strength-threshold fallback path,
                 e.g. PROMOTE/pyramid calls that don't pass score_label).
                 Further weighted by RSI-14 momentum (rsi_multiplier()):
                 strong atr_breakout entries get a bigger slice of cash,
                 weak boll/boll_mr entries get a smaller one — but the risk
                 cap (RISK_PER_TRADE_PCT) and strategy cap are untouched, so
                 the multiplier can only redistribute within the existing
                 risk budget, never expand it.

                 v2.1 新开仓（engine/runner.py 的 buy_signals 循环、
                 backtest_portfolio.py 的 buy_candidates 循环）改为传
                 score_label（engine.scoring.compute_total_score() 的
                 FULL/PARTIAL），直接按 label 选 strong/medium 档位，不再用
                 signal_strength 阈值重新分档——total 本身已经含40%权重的
                 signal_strength，两边都用 signal_strength 分档会把同一个
                 趋势强度信息打两次折。PROMOTE/pyramid 调用方仍不传
                 score_label，走原来的 signal_strength 阈值分档，不受影响。
"""
import config
from engine.scoring import LABEL_FULL, LABEL_OBSERVATION


def rsi_multiplier(rsi_val, strategy: str) -> float:
    """
    Momentum-weighted sizing multiplier for the cash tier (not the risk cap).

    rsi_val : latest RSI-14 value, or None if unavailable (→ 1.0, no bias)
    """
    if rsi_val is None:
        return 1.0
    if (rsi_val > config.DYNAMIC_SIZING_RSI_STRONG
            and strategy in config.DYNAMIC_SIZING_STRONG_STRATEGIES):
        return config.DYNAMIC_SIZING_MULT_STRONG
    if (rsi_val < config.DYNAMIC_SIZING_RSI_WEAK
            and strategy in config.DYNAMIC_SIZING_WEAK_STRATEGIES):
        return config.DYNAMIC_SIZING_MULT_WEAK
    return 1.0


def calculate(available_cash: float, price: float,
              signal_strength: float = 0.5,
              total_capital: float = 0.0,
              stop_loss_pct: float = None,
              kelly_factor: float = 1.0,
              strategy: str = "",
              open_positions: int = 0,
              rsi_val: float = None,
              position_scale: float = 1.0,
              market_weather_code: int = 2,
              score_label: str = None) -> int:
    """
    Return integer share quantity to buy, or 0 if price is zero.

    open_positions : current number of open positions — used for dynamic sizing.
                     When the portfolio has few open positions and the signal is
                     strong, the allocation scales up to SIZING_PCT_STRONG_DYNAMIC
                     so idle cash is deployed more aggressively on high-quality entries.
    rsi_val        : latest RSI-14, used to weight the cash tier by momentum
                     quality (see rsi_multiplier()). None → no adjustment.
    position_scale : final multiplier applied after all tier/risk/strategy caps
                     (e.g. config.TRENDING_EARLY_POSITION_SCALE=0.3 for a
                     trial entry). Same mechanism as kelly_factor — trims the
                     result, never expands it beyond the caps above.
    market_weather_code : engine.market_weather regime code (2/1/0, see that
                     module). Scales RISK_PER_TRADE_PCT itself via
                     config.MARKET_WEATHER_RISK_MULTIPLIER (2/1 both 1.0x,
                     0 -> 0.0x, hard-blocking new entries) — this reuses the
                     EXISTING risk-per-trade cap below rather than adding a
                     second, parallel ATR-based one (an earlier version did
                     that and it double-counted against this same cap for
                     ATR-stop strategies, see config.py comment + project
                     memory for the backtest evidence). Also applies the
                     20% single-position cap (config.MARKET_WEATHER_MAX_POSITION_PCT).
    score_label    : engine.scoring.compute_total_score() 的 FULL/OBSERVATION/
                     PARTIAL 标签（v2.1/v2.2 新开仓专用；PROMOTE/pyramid 调用
                     方不传，保持旧的 signal_strength 阈值分档不变）。传入时
                     直接按 label 选仓位比例档位，而不是让基本面/新闻/天气
                     三项真正参与加权时也去改用旧的 signal_strength 阈值——
                     那样会让FULL/PARTIAL 的分档结果和总分（可能因为强新闻/
                     强基本面把中等趋势拉到FULL，或因为负面新闻/天气把强趋势
                     拉到PARTIAL）脱节。注意：当 fund/news 都缺失时（回测无
                     --news-file的常态），label 的 FULL/PARTIAL 边界和旧
                     signal_strength 的 0.7/0.4 阈值恰好重合，这时候改用
                     label 选档不改变任何结果（实测验证过，byte-for-byte
                     一致）——这个参数真正发挥作用是在 fund/news 有真实数据、
                     total 因此偏离纯趋势线性变换的时候。SKIP 已在调用方过滤
                     掉，这里只会收到 FULL、OBSERVATION 或 PARTIAL。
                     OBSERVATION（v2.2新增，总分40-59分）直接用
                     config.OBSERVATION_POSITION_PCT 作为仓位比例，不走
                     strong/medium 阈值分档——这是"技术面弱信号观察仓"，
                     刻意跟FULL/PARTIAL的强度分档逻辑分开、独立配置。
    """
    if price <= 0:
        return 0
    if stop_loss_pct is None:
        stop_loss_pct = config.STOP_LOSS_PCT

    weather_mult = config.MARKET_WEATHER_RISK_MULTIPLIER.get(market_weather_code, 1.0)
    method = config.SIZING_METHOD

    if method == "fixed_amount":
        qty = max(0, int(config.SIZING_AMOUNT / price))

    elif method == "fixed_percent":
        qty = max(0, int(available_cash * config.SIZING_PCT / price))

    elif method == "fixed_qty":
        qty = config.SIZING_QTY

    elif method == "signal_based":
        if score_label is not None:
            if score_label == LABEL_FULL:
                pct = (config.SIZING_PCT_STRONG_DYNAMIC
                       if open_positions <= config.DYNAMIC_SIZING_MAX_OPEN
                       else config.SIZING_PCT_STRONG) if stop_loss_pct <= config.STOP_MAX_STRONG \
                      else config.SIZING_PCT_MEDIUM
            elif score_label == LABEL_OBSERVATION:
                pct = config.OBSERVATION_POSITION_PCT
            else:   # PARTIAL
                pct = (config.SIZING_PCT_MEDIUM if stop_loss_pct <= config.STOP_MAX_MEDIUM
                       else config.SIZING_PCT_WEAK)
        # Strong signal base tier
        elif signal_strength >= 0.7:
            if stop_loss_pct <= config.STOP_MAX_STRONG:
                # Dynamic sizing: if portfolio is lightly loaded, use expanded cap
                if open_positions <= config.DYNAMIC_SIZING_MAX_OPEN:
                    pct = config.SIZING_PCT_STRONG_DYNAMIC   # 30%
                else:
                    pct = config.SIZING_PCT_STRONG           # 20%
            else:
                pct = config.SIZING_PCT_MEDIUM               # downgrade: stop too wide
        elif signal_strength >= 0.4:
            if stop_loss_pct <= config.STOP_MAX_MEDIUM:
                pct = config.SIZING_PCT_MEDIUM               # 10%
            else:
                pct = config.SIZING_PCT_WEAK                 # downgrade: stop too wide
        else:
            pct = config.SIZING_PCT_WEAK                     # 3% — observation

        pct *= rsi_multiplier(rsi_val, strategy)
        qty_by_tier = int(available_cash * pct / price)

        # Risk cap: single trade loss ≤ RISK_PER_TRADE_PCT × weather_mult × total
        # capital. stop_loss_pct is already strategy-type-aware (2×ATR/price for
        # ATR-trailing strategies, config.STOP_LOSS_PCT flat % for mean-reversion),
        # so scaling the risk budget itself by weather_mult correctly derives
        # "shares = (capital × risk% × regime mult) / (ATR × 2)" for the former
        # and "shares = (capital × risk% × regime mult) / (price × 5%)" for the
        # latter — no separate ATR parameter needed.
        risk_budget = total_capital * config.RISK_PER_TRADE_PCT * weather_mult
        qty_by_risk = (int(risk_budget / (price * stop_loss_pct))
                       if stop_loss_pct > 0 else qty_by_tier)

        # Per-strategy cap (atr_breakout now allows 30%, boll stays at 10%)
        strat_max    = config.STRATEGY_MAX_SIZE_PCT.get(strategy, 1.0)
        qty_by_strat = int(available_cash * strat_max / price)

        qty = min(qty_by_tier, qty_by_risk, qty_by_strat)
        qty = max(0, int(qty * kelly_factor * position_scale))

    else:
        raise ValueError(
            f"Unknown SIZING_METHOD '{method}'. "
            "Choose: fixed_amount | fixed_percent | fixed_qty | signal_based"
        )

    # 状态0：不管走的是哪个 SIZING_METHOD，直接归零（新开仓硬拦截）。对
    # signal_based 来说 qty_by_risk 已经是0，这里主要是给其他方法兜底。
    if weather_mult <= 0:
        return 0

    # 单股仓位硬顶——跟风险预算缩放无关，是独立的资金集中度上限。
    # 守卫：fixed_qty/fixed_amount 等调用方默认不传 total_capital(0.0)，不加
    # 这个guard会把它们误伤成0，是回归不是增强。
    if total_capital > 0:
        qty = min(qty, int(config.MARKET_WEATHER_MAX_POSITION_PCT * total_capital / price))

    return max(0, qty)
