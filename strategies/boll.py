"""
strategies/boll.py — Bollinger Bands mean-reversion.

Entry:  price closes below lower band (oversold) AND has stopped falling
Exit:   price closes back above middle band OR touches upper band
Stop:   hard stop-loss from entry price (tracked via entry_price param)

Signal values:
  BUY  — price below lower band, stabilization confirmed
  SELL — price above middle band (take profit) or above upper band
  HOLD — between lower and middle band (in position, waiting), or still
         falling / not yet based

Basing filter (transition-from-trend protection):
  A single higher close is a weak confirmation — if the stock just fell hard
  (a recent downtrend), one green bar can still be a dead-cat bounce inside a
  falling knife. When a sharp recent decline is detected, BUY additionally
  requires:
    1. Two consecutive non-declining closes (not just one), AND
    2. The lower band itself has flattened (no longer dropping fast) —
       otherwise the band is still chasing price down and "below the band"
       doesn't mean much.
  Established ranges (no sharp recent decline) keep the original, looser
  1-bar stabilization check — this filter only kicks in for the risky
  trend-to-range transition case it was designed for.
"""
import pandas as pd
from .base import Strategy

# ── Basing-filter tuneables ─────────────────────────────────────────────────
_DOWNTREND_LOOKBACK   = 10     # bars to look back for "was this a hard drop?"
_DOWNTREND_DROP_PCT   = 0.08   # >=8% drop over lookback → treat as recent downtrend
_BAND_SLOPE_LOOKBACK  = 5      # bars over which lower-band slope is measured
_BAND_FLATTEN_PCT     = 0.0015 # |lower-band slope| per bar, as % of price, to count as "flat"


class BollStrategy(Strategy):
    name = "boll"

    def __init__(self, period: int = 20, std_dev: float = 2.0,
                 stop_loss_pct: float = 0.03):
        self.period        = period
        self.std_dev       = std_dev
        self.stop_loss_pct = stop_loss_pct
        self.required_bars = period + _DOWNTREND_LOOKBACK + 5

    def compute(self, close: pd.Series) -> dict:
        mid   = close.rolling(self.period).mean()
        band  = close.rolling(self.period).std()
        upper = mid + self.std_dev * band
        lower = mid - self.std_dev * band

        price = self._v(close)
        vu, vm, vl = self._v(upper), self._v(mid), self._v(lower)

        pct_b = None
        if None not in (price, vu, vl):
            bw    = vu - vl
            pct_b = (price - vl) / bw if bw > 0 else 0.5

        basing_ok, stab_label = self._basing_check(close, lower, price)

        if pct_b is None:
            sig = "HOLD"
        elif pct_b < 0.0 and basing_ok:
            sig = "BUY"          # below lower band AND basing confirmed
        elif pct_b < 0.0 and not basing_ok:
            sig = "HOLD"         # below lower band but not based yet — wait
        elif pct_b >= 0.5:
            sig = "SELL"
        else:
            sig = "HOLD"

        # Signal strength: how far below the lower band (deeper = stronger).
        # Capped at 1.0 even for extreme oversold (boll is contrarian — never
        # go heavier than STRATEGY_MAX_SIZE_PCT["boll"] = 10%).
        if sig == "BUY" and pct_b is not None:
            strength = min(1.0, abs(pct_b) / 0.20)
        else:
            strength = 0.0

        detail = (
            f"Upper={vu:.2f}, Mid={vm:.2f}, Lower={vl:.2f}, %B={pct_b:.2%}"
            f", strength={strength:.0%}, {stab_label}"
            if vu else "Bollinger=N/A"
        )

        return {
            "period":         self.period,
            "upper":          round(vu, 4)       if vu    else None,
            "middle":         round(vm, 4)       if vm    else None,
            "lower":          round(vl, 4)       if vl    else None,
            "pct_b":          round(pct_b, 4)    if pct_b is not None else None,
            "signal_strength": round(strength, 4),
            "stop_loss_pct":  self.stop_loss_pct,
            "signal":         sig,
            "detail":         detail,
        }

    def compute_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorised per-bar signal/strength/atr/stop_pct — shared by live
        scanning (via compute(), single last bar) and backtest_portfolio.py
        (whole series), so the two paths never drift apart.

        df must have columns: high, low, close (float).
        """
        close = df["close"].astype(float)
        mid   = close.rolling(self.period).mean()
        band  = close.rolling(self.period).std()
        upper = mid + self.std_dev * band
        lower = mid - self.std_dev * band
        bw    = (upper - lower).replace(0, float("nan"))
        pct_b = (close - lower) / bw

        basing_ok = self._basing_series(close, lower)
        atr = self._atr_series(df)

        signal   = pd.Series("HOLD", index=df.index)
        strength = pd.Series(0.0, index=df.index)

        buy = (pct_b < 0) & basing_ok
        signal[buy] = "BUY"
        signal[pct_b >= 0.5] = "SELL"

        strength[buy] = (pct_b[buy].abs() / 0.20).clip(upper=1.0)

        stop = pd.Series(self.stop_loss_pct, index=df.index)

        return pd.DataFrame({"signal": signal, "strength": strength,
                             "atr": atr, "stop_pct": stop})

    # ── Basing-filter helpers ────────────────────────────────────────────────

    def _basing_check(self, close: pd.Series, lower: pd.Series, price):
        """Single-bar (live) version. Returns (basing_ok: bool, label: str)."""
        p0, p1, p2 = self._v(close, -1), self._v(close, -2), self._v(close, -3)
        recent_ref = self._v(close, -1 - _DOWNTREND_LOOKBACK)

        recent_downtrend = (
            recent_ref is not None and price is not None and recent_ref > 0
            and (recent_ref - price) / recent_ref >= _DOWNTREND_DROP_PCT
        )

        stabilized_1bar = p0 is not None and p1 is not None and p0 >= p1
        stabilized_2bar = (stabilized_1bar and p2 is not None and p1 >= p2)

        l0 = self._v(lower, -1)
        l5 = self._v(lower, -1 - _BAND_SLOPE_LOOKBACK)
        flattening = (
            l0 is not None and l5 is not None and price and price > 0
            and abs((l0 - l5) / _BAND_SLOPE_LOOKBACK / price) < _BAND_FLATTEN_PCT
        )

        if recent_downtrend:
            ok = stabilized_2bar and flattening
            label = f"basing={'yes' if ok else 'no'} (post-downtrend, 2bar+flat)"
        else:
            ok = stabilized_1bar
            label = "stabilized" if ok else "falling"

        return ok, label

    @staticmethod
    def _basing_series(close: pd.Series, lower: pd.Series) -> pd.Series:
        """Vectorised version of _basing_check for the whole series."""
        ref = close.shift(_DOWNTREND_LOOKBACK)
        recent_downtrend = (ref > 0) & (((ref - close) / ref) >= _DOWNTREND_DROP_PCT)

        stabilized_1bar = close >= close.shift(1)
        stabilized_2bar = stabilized_1bar & (close.shift(1) >= close.shift(2))

        band_slope = (lower - lower.shift(_BAND_SLOPE_LOOKBACK)) / _BAND_SLOPE_LOOKBACK
        flattening = (close > 0) & ((band_slope / close).abs() < _BAND_FLATTEN_PCT)

        strict_ok = stabilized_2bar & flattening
        loose_ok  = stabilized_1bar

        ok = pd.Series(False, index=close.index)
        ok[recent_downtrend]  = strict_ok[recent_downtrend]
        ok[~recent_downtrend] = loose_ok[~recent_downtrend]
        return ok

    @staticmethod
    def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()
