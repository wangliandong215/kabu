"""
kabu — moomoo quantitative trading
backtest.py — run a strategy on historical data using backtrader.

Strategies mirror analyze.py:
  ma       Moving Average crossover
  rsi      RSI mean-reversion
  macd     MACD crossover
  boll     Bollinger Bands breakout
  combined Requires ≥3 of the 4 indicators to agree

Usage:
  python backtest.py US.AAPL
  python backtest.py US.AAPL --strategy ma --start 2023-01-01 --end 2024-12-31
  python backtest.py HK.00700 --strategy combined --cash 500000 --json
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import common
from data import fetcher as _fetcher


# ── Data fetching ─────────────────────────────────────────────────────────────
# Disk cache for backtest history (data_cache/{code}.pkl): backtests over the
# same historical window used to re-hit OpenD every single run, which is not
# just slow but a real source of run-to-run drift — rate-limit skips, OpenD
# truncating a page, and QFQ (forward-adjusted) prices for OLD bars shifting
# slightly as new dividends land all make "the same historical date" return
# different numbers on different days. Caching once and reusing freezes the
# numbers for any date range already covered; only genuinely new dates (the
# tail beyond what's cached, or an earlier start not yet covered) hit OpenD.

_CACHE_DIR = Path(__file__).parent / "data_cache"


def _cache_path(code: str) -> Path:
    return _CACHE_DIR / f"{code.replace('.', '_')}.pkl"


def _load_cache(code: str) -> dict | None:
    """Returns {"df": DataFrame, "queried_start": str, "queried_end": str} or None."""
    path = _cache_path(code)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "df" in payload:
            return payload
        return None   # stale format from an older cache version — refetch clean
    except Exception:
        return None


def _save_cache(code: str, df: pd.DataFrame, queried_start: str, queried_end: str) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(code)
    tmp = path.with_suffix(".tmp")
    payload = {"df": df, "queried_start": queried_start, "queried_end": queried_end}
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def fetch_kline(code: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch daily bars for [start, end], disk-cached in data_cache/.

    On a cache hit that fully covers [start, end], this never touches OpenD —
    pure local disk read. On a partial hit, only the missing gap (older
    history before the cached range, and/or newer bars after it — e.g.
    catching up to "yesterday" on a re-run) is fetched from OpenD and merged
    into the cache. Pass use_cache=False to force a full fresh refetch
    without touching or updating the cache.

    Coverage is tracked via queried_start/queried_end metadata, NOT inferred
    from the min/max date actually present in the cached bars — a date range
    that happens to be all weekend/holiday still counts as "covered" once
    queried, so re-running the same window never re-triggers a network call
    just because the requested boundary doesn't land on a trading day.
    """
    if not use_cache:
        merged = _fetch_range(code, start, end)
        mask = (merged["time_key"].str[:10] >= start) & (merged["time_key"].str[:10] <= end)
        return merged[mask].reset_index(drop=True)

    cached = _load_cache(code)

    if cached is None:
        df = _fetch_range(code, start, end)
        q_start, q_end = start, end
    else:
        df = cached["df"]
        q_start, q_end = cached["queried_start"], cached["queried_end"]
        frames = [df]

        if start < q_start:
            gap_end = (pd.Timestamp(q_start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            frames.insert(0, _fetch_range(code, start, gap_end))
            q_start = start

        if end > q_end:
            gap_start = (pd.Timestamp(q_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            frames.append(_fetch_range(code, gap_start, end))
            q_end = end

        if len(frames) > 1:
            df = (pd.concat(frames, ignore_index=True)
                    .drop_duplicates(subset="time_key")
                    .sort_values("time_key")
                    .reset_index(drop=True))

    _save_cache(code, df, q_start, q_end)
    mask = (df["time_key"].str[:10] >= start) & (df["time_key"].str[:10] <= end)
    return df[mask].reset_index(drop=True)


def _fetch_range(code: str, start: str, end: str) -> pd.DataFrame:
    """Raw OpenD fetch for [start, end] — no caching, always hits the network."""
    from moomoo import KLType, AuType, RET_OK

    ctx = common.make_quote_ctx()
    frames = []
    try:
        ret, data, page_key = ctx.request_history_kline(
            code, start=start, end=end,
            ktype=KLType.K_DAY, autype=AuType.QFQ, max_count=1000,
        )
        time.sleep(0.5)   # OpenD rate limit: 60 req/30s — throttle every request,
                           # including pagination pages below (see page loop)
        if ret != RET_OK:
            raise RuntimeError(f"fetch_kline({code}): {data}")
        frames.append(data)

        page = 1
        while page_key is not None and page < 20:
            ret, data, page_key = ctx.request_history_kline(
                code, start=start, end=end,
                ktype=KLType.K_DAY, autype=AuType.QFQ,
                max_count=1000, page_req_key=page_key,
            )
            time.sleep(0.5)   # same throttle — long windows (>1000 daily bars,
                               # e.g. the 2022-2026 default) paginate and used to
                               # fire these back-to-back with zero delay, which
                               # was the actual rate-limit trigger for long runs
            if ret == RET_OK and data is not None and len(data):
                frames.append(data)
            page += 1
    finally:
        common.safe_close(ctx)

    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _to_bt_feed(df: pd.DataFrame):
    """Convert moomoo DataFrame to a backtrader PandasData feed."""
    import backtrader as bt

    result = df[["time_key", "open", "high", "low", "close", "volume"]].copy()
    result.columns = ["datetime", "open", "high", "low", "close", "volume"]
    result["datetime"] = pd.to_datetime(result["datetime"])
    result = result.set_index("datetime").astype(float)
    result["openinterest"] = 0.0
    return bt.feeds.PandasData(dataname=result)


# ── Backtrader strategies ─────────────────────────────────────────────────────

def _make_ma_strategy(fast: int = 5, slow: int = 20):
    import backtrader as bt

    class MACross(bt.Strategy):
        params = (("fast", fast), ("slow", slow))

        def __init__(self):
            self.fast = bt.indicators.SMA(self.data.close, period=self.p.fast)
            self.slow = bt.indicators.SMA(self.data.close, period=self.p.slow)
            self.cross = bt.indicators.CrossOver(self.fast, self.slow)
            self._order = None

        def next(self):
            if self._order:
                return
            if not self.position:
                if self.cross > 0:
                    self._order = self.buy()
            elif self.cross < 0:
                self._order = self.sell()

        def notify_order(self, order):
            if order.status in (order.Completed, order.Canceled, order.Rejected):
                self._order = None

    return MACross


def _make_rsi_strategy(period: int = 14, oversold: int = 30, overbought: int = 70):
    import backtrader as bt

    class RSIStrategy(bt.Strategy):
        params = (("period", period), ("oversold", oversold), ("overbought", overbought))

        def __init__(self):
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)
            self._order = None

        def next(self):
            if self._order:
                return
            if not self.position:
                if self.rsi < self.p.oversold:
                    self._order = self.buy()
            elif self.rsi > self.p.overbought:
                self._order = self.sell()

        def notify_order(self, order):
            if order.status in (order.Completed, order.Canceled, order.Rejected):
                self._order = None

    return RSIStrategy


def _make_macd_strategy():
    import backtrader as bt

    class MACDStrategy(bt.Strategy):
        def __init__(self):
            self.macd = bt.indicators.MACD(self.data.close)
            self._order = None

        def next(self):
            if self._order:
                return
            if not self.position:
                if self.macd.macd[0] > self.macd.signal[0]:
                    self._order = self.buy()
            elif self.macd.macd[0] < self.macd.signal[0]:
                self._order = self.sell()

        def notify_order(self, order):
            if order.status in (order.Completed, order.Canceled, order.Rejected):
                self._order = None

    return MACDStrategy


def _make_boll_strategy(period: int = 20, devfactor: float = 2.0):
    import backtrader as bt

    class BollStrategy(bt.Strategy):
        params = (("period", period), ("devfactor", devfactor))

        def __init__(self):
            self.boll = bt.indicators.BollingerBands(
                self.data.close, period=self.p.period, devfactor=self.p.devfactor
            )
            self._order = None

        def next(self):
            if self._order:
                return
            if not self.position:
                if self.data.close[0] < self.boll.lines.bot[0]:
                    self._order = self.buy()
            elif self.data.close[0] > self.boll.lines.top[0]:
                self._order = self.sell()

        def notify_order(self, order):
            if order.status in (order.Completed, order.Canceled, order.Rejected):
                self._order = None

    return BollStrategy


def _make_combined_strategy(fast: int = 5, slow: int = 20):
    import backtrader as bt

    class Combined(bt.Strategy):
        """Enter when ≥3 out of 4 indicators say BUY; exit when ≥3 say SELL."""
        params = (("fast", fast), ("slow", slow))

        def __init__(self):
            self.maf   = bt.indicators.SMA(self.data.close, period=self.p.fast)
            self.mas   = bt.indicators.SMA(self.data.close, period=self.p.slow)
            self.rsi   = bt.indicators.RSI(self.data.close, period=14)
            self.macd  = bt.indicators.MACD(self.data.close)
            self.boll  = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=2.0)
            self._order = None

        def _buy_count(self):
            score = 0
            if self.maf[0] > self.mas[0]:               score += 1
            if self.rsi[0] < 40:                         score += 1
            if self.macd.macd[0] > self.macd.signal[0]: score += 1
            if self.data.close[0] < self.boll.lines.mid[0]: score += 1
            return score

        def _sell_count(self):
            score = 0
            if self.maf[0] < self.mas[0]:               score += 1
            if self.rsi[0] > 60:                         score += 1
            if self.macd.macd[0] < self.macd.signal[0]: score += 1
            if self.data.close[0] > self.boll.lines.mid[0]: score += 1
            return score

        def next(self):
            if self._order:
                return
            if not self.position:
                if self._buy_count() >= 3:
                    self._order = self.buy()
            elif self._sell_count() >= 3:
                self._order = self.sell()

        def notify_order(self, order):
            if order.status in (order.Completed, order.Canceled, order.Rejected):
                self._order = None

    return Combined


def _make_ma_rsi_strategy(fast: int = 5, slow: int = 20,
                           rsi_period: int = 14,
                           rsi_buy: float = 70.0, rsi_sell: float = 70.0):
    import backtrader as bt

    class MARSIStrategy(bt.Strategy):
        params = (("fast", fast), ("slow", slow),
                  ("rsi_period", rsi_period),
                  ("rsi_buy", rsi_buy), ("rsi_sell", rsi_sell))

        def __init__(self):
            self.maf   = bt.indicators.SMA(self.data.close, period=self.p.fast)
            self.mas   = bt.indicators.SMA(self.data.close, period=self.p.slow)
            self.rsi   = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
            self.cross = bt.indicators.CrossOver(self.maf, self.mas)
            self._order = None

        def next(self):
            if self._order:
                return
            if not self.position:
                # Golden cross AND RSI not overbought
                if self.cross > 0 and self.rsi[0] < self.p.rsi_buy:
                    self._order = self.buy()
            else:
                # Death cross OR RSI overbought
                if self.cross < 0 or self.rsi[0] > self.p.rsi_sell:
                    self._order = self.sell()

        def notify_order(self, order):
            if order.status in (order.Completed, order.Canceled, order.Rejected):
                self._order = None

    return MARSIStrategy


def _make_boll_mean_reversion():
    """Boll mean-reversion: buy below lower band, exit at middle band + 3% stop."""
    import backtrader as bt

    class BollMeanReversion(bt.Strategy):
        params = (("period", 20), ("devfactor", 2.0), ("stop_pct", 0.03))

        def __init__(self):
            self.boll         = bt.indicators.BollingerBands(
                self.data.close, period=self.p.period, devfactor=self.p.devfactor
            )
            self._order       = None
            self._entry_price = None

        def next(self):
            if self._order:
                return
            price = self.data.close[0]
            if not self.position:
                if price < self.boll.lines.bot[0]:
                    self._order       = self.buy()
                    self._entry_price = price
            else:
                if price >= self.boll.lines.mid[0]:
                    self._order = self.sell()
                elif self._entry_price and price < self._entry_price * (1 - self.p.stop_pct):
                    self._order = self.sell()

        def notify_order(self, order):
            if order.status in (order.Completed, order.Canceled, order.Rejected):
                self._order = None

    return BollMeanReversion


def _make_ema_rsi_strategy(fast: int = 5, slow: int = 20):
    """EMA crossover + RSI: buy golden cross & RSI>50, sell death cross or RSI<40."""
    import backtrader as bt

    class EMARSIStrategy(bt.Strategy):
        params = (("fast", fast), ("slow", slow), ("rsi_period", 14))

        def __init__(self):
            self.emaf   = bt.indicators.EMA(self.data.close, period=self.p.fast)
            self.emas   = bt.indicators.EMA(self.data.close, period=self.p.slow)
            self.rsi    = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
            self.cross  = bt.indicators.CrossOver(self.emaf, self.emas)
            self._order = None

        def next(self):
            if self._order:
                return
            if not self.position:
                if self.cross > 0 and self.rsi[0] > 50:
                    self._order = self.buy()
            else:
                if self.cross < 0 or self.rsi[0] < 40:
                    self._order = self.sell()

        def notify_order(self, order):
            if order.status in (order.Completed, order.Canceled, order.Rejected):
                self._order = None

    return EMARSIStrategy


def _make_atr_breakout_strategy():
    """ATR Donchian breakout with trailing stop = highest_close - 2*ATR."""
    import backtrader as bt

    class ATRBreakout(bt.Strategy):
        params = (("channel", 20), ("atr_period", 14), ("atr_mult", 2.0))

        def __init__(self):
            self.atr     = bt.indicators.ATR(self.data, period=self.p.atr_period)
            self.high_ch = bt.indicators.Highest(self.data.high, period=self.p.channel)
            self._order  = None
            self._trail  = None

        def next(self):
            if self._order:
                return
            price = self.data.close[0]
            if not self.position:
                if price > self.high_ch[-1]:
                    self._order = self.buy()
                    self._trail = price - self.p.atr_mult * self.atr[0]
            else:
                new_stop = price - self.p.atr_mult * self.atr[0]
                if self._trail is None or new_stop > self._trail:
                    self._trail = new_stop
                if price < self._trail:
                    self._order = self.sell()
                    self._trail = None

        def notify_order(self, order):
            if order.status in (order.Completed, order.Canceled, order.Rejected):
                self._order = None

    return ATRBreakout


def _with_tax(strategy_cls, tax_rate: float = 0.20):
    """
    Wrap any backtrader Strategy class with capital gains tax.
    On each closed profitable trade: deduct tax_rate × profit from broker cash.
    Losing trades are not taxed.
    """
    class TaxedStrategy(strategy_cls):
        def __init__(self):
            super().__init__()
            self._total_tax = 0.0

        def notify_trade(self, trade):
            if trade.isclosed and trade.pnlcomm > 0:
                tax = trade.pnlcomm * tax_rate
                self._total_tax += tax
                self.broker.add_cash(-tax)

    TaxedStrategy.__name__ = f"Taxed_{strategy_cls.__name__}"
    return TaxedStrategy


_STRATEGY_FACTORIES = {
    "ma":           _make_ma_strategy,
    "rsi":          _make_rsi_strategy,
    "macd":         _make_macd_strategy,
    "boll":         _make_boll_strategy,
    "boll_mr":      _make_boll_mean_reversion,
    "ema_rsi":      _make_ema_rsi_strategy,
    "atr_breakout": _make_atr_breakout_strategy,
    "combined":     _make_combined_strategy,
    "ma_rsi":       _make_ma_rsi_strategy,
}


# ── Main backtest runner ──────────────────────────────────────────────────────

def run_backtest(code: str, strategy: str = "combined",
                 start: str = None, end: str = None,
                 cash: float = 100_000.0, commission: float = 0.001,
                 fast: int = 5, slow: int = 20,
                 tax_rate: float = 0.0,
                 output_json: bool = False) -> dict:
    try:
        import backtrader as bt
    except ImportError:
        print("Error: backtrader not installed.  Run: pip install backtrader")
        sys.exit(1)

    if start is None:
        start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    df_raw = fetch_kline(code, start, end)
    if df_raw is None or len(df_raw) == 0:
        print(f"Error: no data for {code} between {start} and {end}")
        sys.exit(1)
    num_bars = len(df_raw)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(perc=0.0005)   # 0.05% slippage per trade
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
    cerebro.adddata(_to_bt_feed(df_raw))

    factory = _STRATEGY_FACTORIES.get(strategy)
    if factory is None:
        print(f"Unknown strategy: {strategy}")
        sys.exit(1)

    if strategy in ("ma", "combined", "ma_rsi", "ema_rsi"):
        strategy_cls = factory(fast, slow)
    else:
        strategy_cls = factory()

    if tax_rate > 0:
        strategy_cls = _with_tax(strategy_cls, tax_rate)

    cerebro.addstrategy(strategy_cls)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,   _name="sharpe",   riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown,       _name="drawdown")

    results    = cerebro.run()
    strat      = results[0]
    final      = cerebro.broker.getvalue()
    total_tax  = getattr(strat, "_total_tax", 0.0)
    ret_total  = (final - cash) / cash

    ta = strat.analyzers.trades.get_analysis()
    sa = strat.analyzers.sharpe.get_analysis()
    da = strat.analyzers.drawdown.get_analysis()

    n_trades = ta.get("total", {}).get("total", 0)   if ta else 0
    n_wins   = ta.get("won",   {}).get("total", 0)   if ta else 0
    n_loss   = ta.get("lost",  {}).get("total", 0)   if ta else 0

    sharpe   = sa.get("sharperatio") if sa else None
    max_dd   = da.get("max", {}).get("drawdown", 0)  if da else 0

    days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
    years = days / 365.0
    ann   = (1 + ret_total) ** (1 / years) - 1 if years > 0 else 0

    result = {
        "code":                  code,
        "strategy":              strategy,
        "start":                 start,
        "end":                   end,
        "data_bars":             num_bars,
        "initial_cash":          cash,
        "final_value":           round(final, 2),
        "total_return_pct":      round(ret_total * 100, 2),
        "annualized_return_pct": round(ann * 100, 2),
        "sharpe_ratio":          round(float(sharpe), 4) if sharpe and not math.isnan(float(sharpe)) else None,
        "max_drawdown_pct":      round(float(max_dd), 2),
        "num_trades":            n_trades,
        "num_wins":              n_wins,
        "num_losses":            n_loss,
        "win_rate_pct":          round(n_wins / n_trades * 100, 1) if n_trades else 0,
        "tax_rate_pct":          round(tax_rate * 100, 1),
        "total_tax_paid":        round(total_tax, 2),
    }

    if output_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_result(result)

    return result


def _print_result(r: dict) -> None:
    tr   = r["total_return_pct"]
    ar   = r["annualized_return_pct"]
    sh   = r["sharpe_ratio"]
    dd   = r["max_drawdown_pct"]
    tax  = r.get("tax_rate_pct", 0)
    paid = r.get("total_tax_paid", 0.0)

    print("=" * 80)
    print(f"Backtest Results: {r['code']}  [{r['strategy'].upper()} strategy]")
    print("=" * 80)
    print(f"Period:              {r['start']} ~ {r['end']}  ({r['data_bars']} bars)")
    print(f"Initial Capital:     {r['initial_cash']:>14,.2f}")
    print(f"Final Value:         {r['final_value']:>14,.2f}")
    if tax > 0:
        pretax = r["final_value"] + paid
        print(f"  Pre-tax Value:     {pretax:>14,.2f}")
        print(f"  Tax Paid ({tax:.0f}%):   {paid:>14,.2f}")
    print()
    print(f"Total Return:        {tr:>+.2f}%")
    print(f"Annualized Return:   {ar:>+.2f}%")
    print(f"Sharpe Ratio:        {sh if sh is not None else 'N/A'}")
    print(f"Max Drawdown:        -{dd:.2f}%")
    print()
    print(f"Trades:              {r['num_trades']}")
    print(f"Win / Loss:          {r['num_wins']} / {r['num_losses']}")
    print(f"Win Rate:            {r['win_rate_pct']:.1f}%")
    if tax > 0:
        print(f"Capital Gains Tax:   {tax:.0f}%  (applied to each profitable trade)")
    print("=" * 80)
    print("* Past performance does not guarantee future results.")
    print("=" * 80)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Strategy backtesting (moomoo OpenD + backtrader)")
    p.add_argument("code",  help="Stock code, e.g. US.AAPL, HK.00700")
    p.add_argument("--strategy",
                   choices=["ma", "rsi", "macd", "boll", "boll_mr",
                            "ema_rsi", "atr_breakout", "combined", "ma_rsi"],
                   default="combined")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: 2 years ago)")
    p.add_argument("--end",   default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--cash",       type=float, default=100_000.0)
    p.add_argument("--commission", type=float, default=0.001,
                   help="Commission rate, e.g. 0.001 = 0.1%% (default: 0.001)")
    p.add_argument("--fast-period", type=int, default=5)
    p.add_argument("--slow-period", type=int, default=20)
    p.add_argument("--tax-rate", type=float, default=0.0,
                   help="Capital gains tax on profitable trades, e.g. 0.20 = 20%% (default: 0)")
    p.add_argument("--json", action="store_true", dest="output_json")
    args = p.parse_args()

    run_backtest(
        code=args.code, strategy=args.strategy,
        start=args.start, end=args.end,
        cash=args.cash, commission=args.commission,
        fast=args.fast_period, slow=args.slow_period,
        tax_rate=args.tax_rate,
        output_json=args.output_json,
    )
