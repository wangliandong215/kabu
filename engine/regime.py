"""
engine/regime.py — Market Regime Detection per stock.

Detects which of five regimes a stock is currently in.

Regimes:
  TRENDING_UP    confirmed uptrend (ADX) OR early uptrend (ROC + MACD histogram)
  TRENDING_EARLY pre-ADX-confirmation breakout: 18<=ADX<25, price>MA50, price
                 clears the upper Bollinger Band on a volume spike (>=1.5x
                 20-day average volume). A lighter-conviction cousin of
                 TRENDING_UP routed to a smaller position cap (see
                 config.STRATEGY_MAX_SIZE_PCT["atr_breakout_early"]) — catches
                 the early "fish head" of a breakout without waiting for ADX
                 to confirm, while capping the blast radius if it's a fakeout.
  TRENDING_DOWN  confirmed downtrend (ADX) OR early downtrend (ROC + MACD histogram)
  RANGING        no trend, low volatility
  VOLATILE       no trend, high intraday volatility (ATR/price)

ADX-only detection lags: Wilder's smoothing needs ~1-2x the period to react,
so by the time ADX crosses 25 a breakout has often already run 5-10%. This
module CAN promote a stock to TRENDING_* earlier — as soon as a lower ADX
threshold is corroborated by two faster, independent momentum signals
(ROC(10) and MACD histogram slope) — via _ADX_FAST_MIN below.

THIS IS DISABLED BY DEFAULT (_ADX_FAST_MIN == _ADX_TREND_MIN, a no-op).
Backtested on 2024-01-01..2026-07-03 (101-stock watchlist, see
backtest_portfolio.py), enabling it made results WORSE, not better, under
two different threshold configs:

  variant                          return   Sharpe   win-rate   trades
  boll-filter only (this default)  +22.55%   0.711     40.7%      236
  + early-trend, ADX>=20, ROC>=3%  +19.28%   0.612     33.2%      259
  + early-trend, ADX>=22, ROC>=5%  +14.49%   0.528     36.8%      247

Earlier entries did catch a few winners sooner, but mostly added whipsaw
entries on ADX 20-24 stocks that never confirmed into a real trend — net
negative on both Sharpe and win-rate in both configurations tried. Left in
place (rather than deleted) because the mechanism is sound in principle and
a future indicator combination or threshold sweep might still recover edge;
flip _ADX_FAST_MIN below to < _ADX_TREND_MIN to re-enable and re-backtest
before trusting it live.

TRENDING_EARLY (added 2026-07-03, per user-specified PRD) is a SEPARATE
mechanism from the ROC/MACD early-trend path above — it confirms with a
Bollinger Band breakout + volume spike instead of ROC/MACD slope, and instead
of being folded into TRENDING_UP it is its own regime routed to a 30%-sized
trial position (config.TRENDING_EARLY_POSITION_SCALE) that gets topped up to
full size if the regime later confirms to TRENDING_UP while still held (see
"promote" step in engine/runner.py / backtest_portfolio.py), or cools the
stock down for 10 trading days if stopped out within 5 days of entry (see
config.TRENDING_EARLY_COOLDOWN_DAYS, portfolio.tracker.Portfolio.is_cooldown).
A hysteresis lock (_HYSTERESIS_EXIT_ADX=20) keeps a stock in the trending
bucket once entered, to stop it ping-ponging RANGING<->TRENDING_EARLY at the
ADX 20-25 boundary.

Backtested 2024-01-01..2026-07-03 (82 of 142 watchlist stocks actually
fetched — the rest were skipped by OpenD rate-limiting in both runs, same 60
stocks skipped in both so the comparison is apples-to-apples):

  variant                  return   Sharpe   max-dd   trades  win-rate
  TRENDING_EARLY disabled  +24.29%   0.756   -13.86%     201     40.8%
  TRENDING_EARLY enabled   +35.07%   0.955   -12.51%     211     38.9%

Portfolio-level numbers improved. But do NOT read that as "the light-trial-
then-promote thesis worked" — the atr_breakout_early trades themselves lost
money standalone (20 trades, 20.0% win rate, -57,073 net) and NOT ONE of them
was ever promoted to TRENDING_UP in this entire 2.5-year window across 82
stocks: either the trial gets stopped out before ADX can confirm, or it exits
some other way first. The portfolio-level improvement instead traces to the
confirmed atr_breakout strategy's own P&L being ~JPY 788k higher with
TRENDING_EARLY on (171 trades, +2,245,712) vs off (176 trades, +1,457,269) —
nearly the same trade COUNT, very different P&L. That's consistent with a
path-dependent effect: trial entries and cooldowns compete for the same
MAX_POSITIONS slots and cash pool as confirmed atr_breakout signals, which
changes the day-by-day selection order across the whole 2.5-year simulation
and, on this particular historical path, happened to land on a more
profitable sequence — not necessarily a repeatable structural edge. Treat
this result as "not disproven, not confirmed either" — the promotion
mechanism specifically has zero live evidence behind it and deserves a longer
/ different backtest window (or loosened entry conditions so a promotion can
actually occur) before being trusted with real capital.
"""
import numpy as np
import pandas as pd


# ── Tuneable constants ────────────────────────────────────────────────────────
_ADX_PERIOD        = 14
_ADX_TREND_MIN     = 25    # ADX >= this → confirmed trending (original threshold)
_ADX_FAST_MIN      = 25    # set < _ADX_TREND_MIN to re-enable early-trend detection
                            # (backtested worse as of 2026-07 — see module docstring)
_MA_PERIOD         = 50    # trend direction reference
_VOLATILE_ATR_PCT  = 0.03  # ATR/price >= 3% → VOLATILE

_ROC_PERIOD        = 10    # bars for rate-of-change momentum
_ROC_THRESHOLD     = 0.03  # +/-3% over ROC_PERIOD → meaningful momentum
_MACD_FAST         = 12
_MACD_SLOW         = 26
_MACD_SIGNAL       = 9
_MACD_SLOPE_BARS   = 3     # bars over which histogram slope is measured

# TRENDING_EARLY: Bollinger breakout + volume spike, pre-ADX-confirmation.
# Independent of the ROC/MACD _ADX_FAST_MIN mechanism above (see docstring).
_ADX_EARLY_MIN     = 18    # lower bound of the "early" zone (ADX < _ADX_TREND_MIN)
_BB_PERIOD         = 20
_BB_STD            = 2.0
_VOL_MA_PERIOD     = 20
_VOL_SPIKE_MULT    = 1.5   # volume >= 1.5x its 20-bar average → "spike"

# Hysteresis: once a stock enters the trending bucket (TRENDING_EARLY or
# TRENDING_UP), only release it back to RANGING/VOLATILE when ADX drops below
# this (lower than _ADX_TREND_MIN=25) — prevents ping-ponging RANGING<->
# TRENDING_EARLY every bar for a stock hovering right at the ADX 20-25
# boundary. A genuine reversal into TRENDING_DOWN (ADX>=25 with price<=MA50)
# still releases the lock immediately — this only dampens noise on the way
# back down to "no trend", not real direction changes.
_HYSTERESIS_EXIT_ADX = 20

# V_REVERSAL_MODE (added 2026-07-03, promoted to default 2026-07-03):
# bypasses ADX entirely — TRENDING_UP can be declared purely from short-term
# momentum (RSI(6)>50, ROC(5)>0, price>MA20), no ADX floor at all. Much more
# aggressive than TRENDING_EARLY (which still requires ADX>=_ADX_EARLY_MIN).
# Built to catch V-shaped bear->bull reversals (regime.py's ADX-lag problem
# is worst exactly here — see 2023's -40% alpha gap in project memory).
#
# Validated on the FULL 2022-2026 sample (92 stocks), not just the reversal
# windows it was designed for — same discipline that got the ROC/MACD
# _ADX_FAST_MIN variant rejected (that one only ever helped in isolation,
# hurt full-sample results, stayed OFF). This one is different: full-sample
# return +46.19%->+60.08%, Sharpe 0.802->0.880, Alpha-vs-QQQ gap -36.43%->
# -22.54%. Costs are real but bounded: win rate 34.8%->32.8%, max drawdown
# -13.67%->-14.66%, and 2022 (bear market) specifically got ~0.7pp worse
# (-6.92%->-7.65%) from false signals on bear-market bounces — that's the
# risk to watch if results degrade going forward, not a hypothetical.
V_REVERSAL_MODE        = True
_V_REVERSAL_MA_PERIOD  = 20
_V_REVERSAL_RSI_PERIOD = 6
_V_REVERSAL_RSI_MIN    = 50
_V_REVERSAL_ROC_PERIOD = 5

# ── QQQ macro technical halt (added 2026-07-03) ──────────────────────────────
# When QQQ itself is below MA(config.QQQ_MA_PERIOD) AND its own short-term
# momentum (MA20 slope) is falling steeply, block ALL new stock-engine
# entries regardless of individual signal quality — a confirmed, accelerating
# macro downtrend, not just "QQQ dipped below its long MA." A QQQ that's
# below MA200 but stabilizing (flat/rising MA20) does NOT trigger this.
#
# By user's explicit choice: this blocks NEW entries only (same behavior as
# the news-based engine.news.macro_circuit_breaker) — it does NOT force-close
# already-open stock positions, which keep exiting via their own stop-loss /
# trailing-stop / strategy-SELL rules. It is independent of (but usually
# coincides with) the QQQ Beta floor's own MA200 exit — that's a separate
# "death command" mechanism (config.QQQ_CORE_TARGET_PCT), not driven by this.
_MACRO_HALT_MA_PERIOD       = 20      # QQQ's own short MA for momentum slope
_MACRO_HALT_SLOPE_LOOKBACK  = 5       # bars over which the MA20 slope is measured
_MACRO_HALT_SLOPE_THRESHOLD = -0.006  # MA20 slope steeper than -0.6% of price
                                       # over the lookback counts as "severely down"
# Calibrated 2026-07-03 from the worst per-year QQQ MA20 slope across the full
# 2015-2026 history: -1.0% (the original value) never fired outside 2020's
# COVID crash — 2022's grinding bear market only reached -0.82% and got zero
# coverage. -0.6% is the natural separation point: every non-crisis year
# (2015-19, 2021, 2023, 2026) tops out at -0.58%, never crossing it, while
# 2022 crosses it on 30 days and 2020 (COVID) on 17. Costs a handful of false
# triggers in 2024/2025 (3 days each, mild pullbacks, not real crises) — that
# trade-off was accepted deliberately, not discovered after the fact by
# tuning until 2022 "looked good"; see project memory for the full table and
# the resulting full-sample backtest before trusting this value further.


def qqq_macro_halt_series(close: pd.Series) -> pd.Series:
    """
    Per-bar boolean (vectorised — shared by live scanning and
    backtest_portfolio.py, so the two never drift apart): True when the
    satellite stock engine should block all new entries — QQQ is in a
    confirmed, accelerating downtrend (below MA200 + steep MA20 downslope).

    2026-07-03: briefly replaced with a position-size scale-down instead of
    a hard block (qqq_position_scale_series, now removed) — reverted after
    full 2015-2026 testing showed the scale-down was WORSE than this hard
    block on every metric that mattered (2022 trades rose 130->150, 2022
    return -15.11%->-15.23%, global MDD -23.54%->-25.09%), because letting
    entries through at a discount still let V_REVERSAL_MODE fire repeatedly
    through 2022's grind instead of stopping it. This hard-block version is
    the locked baseline (v1.0-RELEASE-FINAL) — do not reintroduce a
    scale-based variant without re-testing the full sample first; see
    project memory for the full writeup.
    """
    import config

    ma_long = close.rolling(config.QQQ_MA_PERIOD).mean()
    below_ma_long = close < ma_long

    ma_short = close.rolling(_MACRO_HALT_MA_PERIOD).mean()
    slope = (ma_short - ma_short.shift(_MACRO_HALT_SLOPE_LOOKBACK)) / _MACRO_HALT_SLOPE_LOOKBACK
    slope_pct = slope / close
    momentum_down = slope_pct < _MACRO_HALT_SLOPE_THRESHOLD

    halt = below_ma_long & momentum_down
    return halt.fillna(False)


def qqq_macro_halt(df: pd.DataFrame) -> bool:
    """Last-bar convenience wrapper for live scanning. df needs a 'close' column."""
    close = df["close"].astype(float)
    series = qqq_macro_halt_series(close)
    if len(series) == 0:
        return False
    return bool(series.iloc[-1])


def detect(df: pd.DataFrame) -> str:
    """
    Return one of: "TRENDING_UP", "TRENDING_EARLY", "TRENDING_DOWN",
    "RANGING", "VOLATILE" for the LAST bar of df.
    df must have columns: high, low, close (float); volume is optional
    (without it, TRENDING_EARLY can never trigger — see detect_series).
    Returns "RANGING" if data is insufficient.
    """
    series = detect_series(df)
    if series is None or len(series) == 0:
        return "RANGING"
    val = series.iloc[-1]
    return "RANGING" if pd.isna(val) else str(val)


def detect_series(df: pd.DataFrame) -> pd.Series:
    """
    Per-bar regime label for the whole df (vectorised — used by both live
    scanning and backtest_portfolio.py, so the two never drift apart).

    df must have columns: high, low, close (float). volume (float) is
    optional — without it TRENDING_EARLY never fires (falls through to
    RANGING/VOLATILE as before), everything else is unaffected.
    """
    min_bars = _ADX_PERIOD * 3 + _MA_PERIOD
    if df is None or len(df) < min_bars:
        return pd.Series("RANGING", index=df.index if df is not None else [])

    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    adx, _plus_di, _minus_di = _adx(high, low, close, _ADX_PERIOD)
    atr = _atr(high, low, close, _ADX_PERIOD)
    ma50 = close.rolling(_MA_PERIOD).mean()
    atr_pct = (atr / close).rolling(5).mean()   # smoothed ATR/price

    roc = close.pct_change(_ROC_PERIOD)
    macd_line   = close.ewm(span=_MACD_FAST, adjust=False).mean() - \
                  close.ewm(span=_MACD_SLOW, adjust=False).mean()
    signal_line = macd_line.ewm(span=_MACD_SIGNAL, adjust=False).mean()
    histogram   = macd_line - signal_line
    hist_slope  = histogram.diff(_MACD_SLOPE_BARS)

    momentum_up   = (roc >= _ROC_THRESHOLD)  & (hist_slope > 0) & (histogram > 0)
    momentum_down = (roc <= -_ROC_THRESHOLD) & (hist_slope < 0) & (histogram < 0)

    confirmed_up   = (adx >= _ADX_TREND_MIN) & (close > ma50)
    confirmed_down = (adx >= _ADX_TREND_MIN) & (close <= ma50)
    early_up       = (adx >= _ADX_FAST_MIN)  & (close > ma50) & momentum_up
    early_down     = (adx >= _ADX_FAST_MIN)  & (close <= ma50) & momentum_down

    trending_up   = confirmed_up | early_up
    trending_down = confirmed_down | early_down

    # V_REVERSAL_MODE: test-only ADX bypass (see constant docstring above).
    if V_REVERSAL_MODE:
        from engine.indicators import rsi_series
        ma20 = close.rolling(_V_REVERSAL_MA_PERIOD).mean()
        rsi6 = rsi_series(close, _V_REVERSAL_RSI_PERIOD)
        roc5 = close.pct_change(_V_REVERSAL_ROC_PERIOD)
        v_reversal_up = (close > ma20) & (rsi6 > _V_REVERSAL_RSI_MIN) & (roc5 > 0)
        trending_up = trending_up | v_reversal_up

    # TRENDING_EARLY: pre-ADX breakout confirmed by price + volume instead of
    # ADX. Only considered in the 18<=ADX<25 zone that isn't already a
    # confirmed/ROC-MACD trend, so it can never override TRENDING_UP/DOWN.
    bb_mid   = close.rolling(_BB_PERIOD).mean()
    bb_upper = bb_mid + _BB_STD * close.rolling(_BB_PERIOD).std()
    is_bb_breakout = close > bb_upper

    if "volume" in df.columns:
        volume   = df["volume"].astype(float)
        vol_ma20 = volume.rolling(_VOL_MA_PERIOD).mean()
        is_volume_spike = volume > (_VOL_SPIKE_MULT * vol_ma20)
    else:
        is_volume_spike = pd.Series(False, index=df.index)

    trend_early = (
        (adx >= _ADX_EARLY_MIN) & (adx < _ADX_TREND_MIN)
        & (close > ma50) & is_bb_breakout & is_volume_spike
        & ~trending_up & ~trending_down
    )

    regime = pd.Series("RANGING", index=df.index)
    regime[~trending_up & ~trending_down & ~trend_early
           & (atr_pct >= _VOLATILE_ATR_PCT)] = "VOLATILE"
    regime[trending_down & ~trending_up]     = "TRENDING_DOWN"
    regime[trend_early]                      = "TRENDING_EARLY"
    regime[trending_up]                      = "TRENDING_UP"
    regime[adx.isna() | ma50.isna()] = "RANGING"   # warm-up

    return _apply_hysteresis(regime, adx)


def _apply_hysteresis(regime_raw: pd.Series, adx: pd.Series) -> pd.Series:
    """
    Sequential state-machine pass: once locked into the trending bucket
    (TRENDING_EARLY/TRENDING_UP), stay there — even if the raw per-bar
    conditions above would say RANGING/VOLATILE this bar — until ADX drops
    below _HYSTERESIS_EXIT_ADX. A raw TRENDING_DOWN reading (a genuine
    reversal) always releases the lock immediately.

    Inherently stateful (depends on the previous bar's lock state), so this
    can't be a vectorised boolean mask like the rest of detect_series — it's
    a single forward pass over the bars instead (cheap: O(n) per stock).
    """
    vals = regime_raw.to_numpy(copy=True)
    adx_vals = adx.to_numpy()
    locked = False

    for i in range(len(vals)):
        cur = vals[i]
        a = adx_vals[i]

        if locked:
            if cur == "TRENDING_DOWN":
                locked = False   # genuine reversal — let it through as-is
            elif pd.isna(a) or a < _HYSTERESIS_EXIT_ADX:
                locked = False   # trend has genuinely faded — release the lock
            else:
                # still elevated ADX — stay in the trending bucket regardless
                # of what the raw per-bar mask said this bar
                if cur not in ("TRENDING_UP", "TRENDING_EARLY"):
                    cur = "TRENDING_UP" if a >= _ADX_TREND_MIN else "TRENDING_EARLY"
                    vals[i] = cur

        if cur in ("TRENDING_UP", "TRENDING_EARLY"):
            locked = True

    return pd.Series(vals, index=regime_raw.index)


def regime_label(regime: str) -> str:
    """Human-readable label for logging."""
    return {
        "TRENDING_UP":    "Trend-Up",
        "TRENDING_EARLY": "Trend-Early",
        "TRENDING_DOWN":  "Trend-Down",
        "RANGING":        "Ranging",
        "VOLATILE":       "Volatile",
    }.get(regime, regime)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _atr(high, low, close, period):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(high, low, close, period):
    """Return (ADX, +DI, -DI) series using Wilder's smoothing."""
    up   = high.diff()
    down = -low.diff()

    plus_dm  = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)
    plus_dm[ (up > down) & (up > 0)]   = up
    minus_dm[(down > up) & (down > 0)] = down

    atr_s    = _atr(high, low, close, period)
    plus_di  = 100 * plus_dm.ewm( alpha=1/period, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_s

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx, plus_di, minus_di


# ── Dynamic ATR multiplier (inverse-volatility scaling) ──────────────────────
# REJECTED 2026-07-04 — kept only as a documented dead end, NOT wired into
# any default path (config.ATR_MULT_BASE / backtest_portfolio.ATR_TRAIL_MULT
# are untouched by this section; nothing calls these functions unless a
# script explicitly imports and does so).
#
# Design doc: ATR_Multiplier_t = Base * (Historical_Vol_Mean / Current_Vol_t)^alpha,
# clipped to [min_mult, max_mult]. Motivated by the 4-window walk-forward ATR
# sweep in backtest_portfolio.py (best fixed ATR ranged 3.25~5.75 across
# windows) and a follow-up regime-correlation check, which found ADX
# essentially flat across windows (|r|<0.22, no relationship) while
# volatility-type measures (QQQ realized vol, cross-sectional ATR%) had a
# moderate (|r|~0.65-0.69, not proof) relationship with the best per-window
# ATR.
#
# That correlation motivated a full nested walk-forward test (42 rolling
# folds, 12mo in-sample grid search over base x alpha -> 3mo out-of-sample,
# 2016-01~2026-07, see _atr_dynamic_walkforward_ab.py) against two fixed-ATR
# benchmarks. Result: the dynamic model LOST to simply fixing ATR=5.5 the
# whole time, on every headline metric (stitched OOS Sharpe 0.891 vs 0.938,
# Calmar 0.479 vs 0.556, total return 280% vs 328%). It only narrowly beat
# fixed ATR=3.5, a low bar. The core "loosen in low vol to cut whipsaw"
# mechanism didn't even hold empirically -- the dynamic model generated MORE
# trades than fixed-5.5 in the below-median-volatility folds (453 vs 408),
# not fewer. In 11 of 42 folds (26%) the in-sample grid search's own best
# pick degenerated to alpha=0 (no dynamic adjustment at all), meaning the
# "dynamic beats fixed" premise often failed even in-sample, before getting
# to the out-of-sample question.
#
# Conclusion actually worth carrying forward: the widest fixed value tested
# (5.5) beat the current locked default of 4.0 (config.ATR_MULT_BASE /
# ATR_TRAIL_MULT) by a wide margin in this same test range — see the
# separate fixed-ATR 4.0/5.0/5.5 full-history comparison run right after
# this experiment for the number that matters. Do not resume the dynamic-ATR
# direction without new evidence; if asked "did dynamic ATR work", the
# answer is no, and re-running the same sweep isn't likely to change that.
def qqq_realized_vol_series(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized volatility of daily log returns, rolling `window`
    days. Causal (each point only uses days up to and including it) — safe
    to use inside a backtest without look-ahead bias."""
    log_ret = np.log(close.astype(float) / close.astype(float).shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def dynamic_atr_multiplier_series(close: pd.Series,
                                   base_multiplier: float = 4.5,
                                   alpha: float = 0.5,
                                   min_mult: float = 3.0,
                                   max_mult: float = 6.0,
                                   vol_window: int = 20,
                                   hist_window: int = 252) -> pd.Series:
    """Causal daily series of ATR_Multiplier_t for backtest/live use.

    current_vol_t      = `vol_window`-day annualized realized volatility of
                          `close` (e.g. QQQ), see qqq_realized_vol_series().
    historical_vol_mean_t = trailing `hist_window`-day rolling mean of
                          current_vol_t (NOT a whole-sample/expanding mean —
                          using future data to normalize past days would be
                          look-ahead bias). Every point in the returned
                          series only depends on `close` values up to that
                          date.
    Days without enough warmup for either rolling stat (NaN) fall back to
    `base_multiplier` (equivalent to alpha's effect being neutral until
    enough history exists).
    """
    current_vol = qqq_realized_vol_series(close, vol_window)
    historical_vol_mean = current_vol.rolling(hist_window).mean()

    ratio = historical_vol_mean / current_vol.replace(0, np.nan)
    dynamic_mult = base_multiplier * (ratio ** alpha)
    dynamic_mult = dynamic_mult.clip(lower=min_mult, upper=max_mult)
    return dynamic_mult.fillna(base_multiplier)


def calculate_dynamic_atr_multiplier(current_prices,
                                      historical_vol_mean: float,
                                      base_multiplier: float = 4.5,
                                      alpha: float = 0.5,
                                      min_mult: float = 3.0,
                                      max_mult: float = 6.0) -> float:
    """Single-point convenience version (e.g. for a live 5-minute poll loop
    where `historical_vol_mean` is maintained/updated separately). For
    backtesting, prefer dynamic_atr_multiplier_series() — it computes
    historical_vol_mean causally per day instead of requiring the caller to
    supply one precomputed value."""
    current_prices = np.asarray(current_prices, dtype=float)
    log_returns = np.log(current_prices[1:] / current_prices[:-1])
    current_vol = float(np.std(log_returns) * np.sqrt(252))

    if current_vol == 0:
        return base_multiplier

    dynamic_mult = base_multiplier * ((historical_vol_mean / current_vol) ** alpha)
    return float(np.clip(dynamic_mult, min_mult, max_mult))


# ── Macro Risk-Off half-size circuit breaker — TRIED AND ROLLED BACK 2026-07-04
# NOT wired into any default path. backtest_portfolio.py briefly called this
# to halve position_scale on Risk-Off days (close<MA252 OR 20d vol >= its own
# trailing 90th percentile), on top of the existing qqq_macro_halt() hard
# block above (which fully vetoes entries under its own separate condition —
# below MA200 AND own MA20 momentum falling steeply). That wiring has been
# removed; qqq_macro_halt is the only halt mechanism in the locked baseline.
#
# Full-history (2015-01-01~2026-07-03) result at ATR_TRAIL_MULT=5.5:
#   Risk-Off OFF: return=277.18%  Sharpe=0.920  maxdd=-19.85%  trades=873
#   Risk-Off ON:  return=198.29%  Sharpe=0.832  maxdd=-18.48%  trades=926
# i.e. halving position size on ~15-25% of days (93% of 2022 alone) cost
# 28.5% of baseline return and 9.6% of baseline Sharpe. In exchange:
#   - 2x-commission stress test: relative return drop improved from -35.6%
#     (Risk-Off off) to -19.2% (Risk-Off on) -- a real, apples-to-apples win,
#     better even than the old ATR=4.0 baseline's -20.9%.
#   - Trade-level Monte Carlo (1000x reshuffled realized trade P&Ls, SAME
#     methodology both sides so this comparison is clean): 5th-pct-worst
#     drawdown improved only -19.55%->-17.93%, worst single -29.50%->-25.65%
#     -- real but modest, NOT the dramatic-looking improvement an earlier,
#     methodologically-inconsistent comparison (different Monte Carlo method
#     on each side) initially suggested.
# Decision: the cost-sensitivity improvement was real but not judged worth
# giving up ~28% of baseline return for. Kept here only as a documented dead
# end -- do not re-wire without new evidence, and if re-tried, compare
# Monte Carlo methodology consistently on both sides of the on/off toggle.
def qqq_risk_off_series(close: pd.Series,
                         ma_period: int = 252,
                         vol_window: int = 20,
                         vol_percentile: float = 90.0,
                         min_history: int = 252) -> pd.Series:
    """Daily boolean Series: True on days classified Macro Risk-Off.

    Risk-Off triggers on EITHER:
      1. close < its `ma_period`-day moving average, OR
      2. the `vol_window`-day realized vol (qqq_realized_vol_series) is at or
         above the `vol_percentile`-th percentile of ITS OWN trailing history
         up to and including that day (expanding percentile rank -- causal,
         no look-ahead: day t's threshold only reflects vol values known by
         day t, not the whole-sample distribution).

    Before `min_history` days of vol history exist, condition 2 defaults to
    False (not enough sample to rank a percentile) -- condition 1 (MA) can
    still fire once `ma_period` days of price history exist.
    """
    close = close.astype(float)
    below_ma = close < close.rolling(ma_period).mean()

    vol = qqq_realized_vol_series(close, window=vol_window)

    def _expanding_percentile_rank(arr: np.ndarray) -> float:
        return (arr <= arr[-1]).mean() * 100.0

    vol_pct_rank = vol.expanding(min_periods=min_history).apply(
        _expanding_percentile_rank, raw=True)
    high_vol = vol_pct_rank >= vol_percentile

    risk_off = (below_ma.fillna(False)) | (high_vol.fillna(False))
    return risk_off


def market_volatility_percentile(close: pd.Series,
                                  window: int = 5,
                                  min_history: int = 60) -> pd.Series:
    """v2.4 RSL: `window`-day realized volatility (see qqq_realized_vol_series)
    expressed as a causal percentile rank (0-100) against its own trailing
    history -- same _expanding_percentile_rank technique as
    qqq_risk_off_series, extracted here so a short (default 5-day) lookback
    can be reused independently of that function's 20-day default.

    Before `min_history` days of vol history exist, returns 0.0 (no
    volatility premium applied yet -- consistent with this project's existing
    warm-up convention of defaulting new indicators to their most permissive
    value rather than erroring or guessing).
    """
    vol = qqq_realized_vol_series(close.astype(float), window=window)

    def _expanding_percentile_rank(arr: np.ndarray) -> float:
        return (arr <= arr[-1]).mean() * 100.0

    pct_rank = vol.expanding(min_periods=min_history).apply(
        _expanding_percentile_rank, raw=True)
    return pct_rank.fillna(0.0)
