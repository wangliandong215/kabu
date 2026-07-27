"""
backtest_crypto.py — single-symbol crypto backtest that drives the SAME
shared ATR Breakout strategy / Risk Engine / Position Manager / Trade Logger
engine/runner.py already uses for live stock trading — fed from the locally
synced crypto_kline table instead of live moomoo/Binance data.

Goal: prove the stock ATR Breakout strategy runs UNMODIFIED against crypto
OHLC data. Nothing here is a new strategy or a copy of one.

Reused as-is (imported, never reimplemented):
  strategies.get_strategy("atr_breakout")       — signal generation
  risk.guard.check_exit_ordered / update_trailing_stop — exit rules
  risk.sizing.calculate                          — position sizing
  portfolio.tracker.Portfolio                    — position state
  engine.trade_tracker.TradeTracker              — trade log persistence

Deliberately NOT ported — stock-portfolio concepts that don't apply to a
single always-open crypto symbol: QQQ macro halt, Half-Kelly drawdown
throttle, Market Weather regime multiplier, sector caps, capacity
manager/active replacement, earnings blackout, market-hours gating.
backtest_portfolio.py (stocks) and its trade_history.db/positions.json are
untouched by this file.

Data: read-only via data_provider.provider_factory.get_provider("CRYPTO",
source="db") -> BinanceDBProvider, which reads the local crypto_kline table
only — this script never calls the Binance API directly. Run
data_sync.binance_sync.sync_symbol(...) first to populate crypto_kline.

Sizing note: risk.sizing.calculate() returns an INTEGER "share" count, built
for stocks priced in the tens-to-hundreds range. BTCUSDT trades at tens of
thousands of USDT per whole coin, so with $50,000 capital an unscaled call
would always round down to qty=0 (can't even buy 1 whole BTC) — zero trades,
every time, regardless of signal. To reuse calculate() unmodified while
still getting meaningful fractional-BTC sizing, this script prices in
"lots" of _QTY_LOT_SIZE BTC (default 1e-6) when calling it, then converts
the returned integer lot count back to real BTC quantity. All P&L/equity/
guard calculations after that use the real price and real (fractional) qty
— only the sizing.calculate() call itself sees the rescaled price.

Usage:
  python backtest_crypto.py
  python backtest_crypto.py --symbol BTCUSDT --days 365 --cash 50000
  python backtest_crypto.py --json
"""
import argparse
import json as json_module
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from data_provider.provider_factory import get_provider
from engine.trade_tracker import TradeTracker
from portfolio.tracker import Portfolio
from risk import guard, sizing
from strategies import get_strategy

_BARS_WINDOW = 120           # matches engine/scanner.py's / runner.py's default live scan window
_ANNUALIZATION_DAYS = 365    # crypto trades 24/7 — no 252-trading-day-year assumption
_QTY_LOT_SIZE = 1e-6         # see "Sizing note" above

_CRYPTO_PORTFOLIO_PATH = Path(r"I:\bianceData\portfolio\crypto_positions.json")
_CRYPTO_TRADE_DB_PATH = Path(r"I:\bianceData\trade_history\crypto_trade_history.db")


def _seed_portfolio(path: Path, initial_cash: float) -> Portfolio:
    """Always start a fresh account state for a backtest run — unlike live
    trading, re-running the same backtest must not inherit a stale open
    position from a previous run. Writes the JSON directly (Portfolio has no
    constructor override for a non-default starting balance, and this file
    is not the place to add one) rather than going through Portfolio._save()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json_module.dump({
            "positions": {}, "peak_value": initial_cash,
            "initial_cash": initial_cash, "realized_pnl": 0.0,
            "peak_equity": initial_cash, "cooldowns": {},
        }, f)
    return Portfolio(path=path)


def _set_entry_time(portfolio: Portfolio, symbol: str, as_of: pd.Timestamp) -> None:
    """Portfolio.open_position() stamps entry_time with datetime.now() (the
    live-trading fill time). A backtest needs the SIMULATED bar date there
    instead, so trade_id / trade-log dates are historically meaningful.
    Patched via the public `data` dict — portfolio/tracker.py is untouched."""
    portfolio.data["positions"][symbol]["entry_time"] = as_of.isoformat()
    portfolio._save()


def run_backtest(symbol: str, days: int, initial_cash: float) -> dict:
    provider = get_provider("CRYPTO", source="db")
    strategy = get_strategy("atr_breakout")

    end = pd.Timestamp.now().normalize()
    sim_start = end - pd.Timedelta(days=days)
    # Extra lookback so the strategy has warmup bars before the reporting
    # window starts — get_history() just returns whatever's actually synced
    # if less is available; the required_bars check below handles the rest.
    fetch_start = sim_start - pd.Timedelta(days=_BARS_WINDOW + 10)

    df = provider.get_history(symbol, interval="1d", start=fetch_start, end=end)
    df = df.sort_values("datetime").reset_index(drop=True)

    sim_idx = df.index[df["datetime"] >= sim_start]
    if len(sim_idx) == 0:
        raise ValueError(f"No bars on/after {sim_start} for {symbol} in crypto_kline — "
                          f"run data_sync.binance_sync.sync_symbol() first.")

    portfolio = _seed_portfolio(_CRYPTO_PORTFOLIO_PATH, initial_cash)
    tracker = TradeTracker(db_path=_CRYPTO_TRADE_DB_PATH)
    run_id = f"crypto_backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    equity_curve = []
    trade_log = []

    for i in sim_idx:
        window = df.iloc[max(0, i - _BARS_WINDOW + 1): i + 1].reset_index(drop=True)
        today = df.iloc[i]["datetime"]
        if len(window) < strategy.required_bars:
            equity_curve.append((today, portfolio.current_equity()))
            continue

        result = strategy.full_result(window)
        price = float(window.iloc[-1]["close"])

        pos = portfolio.get_position(symbol)
        if pos is not None:
            current_atr = result.get("atr") or pos.get("entry_atr", 0.0)
            if current_atr:
                guard.update_trailing_stop(pos, price, float(current_atr))
                portfolio._save()

            reason = guard.check_exit_ordered(
                pos["entry_price"], price, "atr_breakout", result.get("signal", "HOLD"),
                trail_stop=pos.get("trail_stop"),
            )
            if reason:
                closed = portfolio.close_position(symbol, price, reason=reason)
                trade_id = f"{symbol}_{closed['entry_time']}"
                tracker.log_exit(
                    trade_id=trade_id, price=price,
                    cash=portfolio.available_cash(), equity=portfolio.current_equity(),
                    timestamp=today.isoformat(), exit_reason=reason,
                )
                trade_log.append({
                    "symbol": symbol, "side": "SELL", "date": today.isoformat(),
                    "price": price, "qty": closed["qty"], "reason": reason,
                    "pnl": closed["pnl"],
                })
        else:
            if result.get("signal") == "BUY":
                stop_pct = float(result.get("stop_loss_pct", config.STOP_LOSS_PCT))
                lots = sizing.calculate(
                    portfolio.available_cash(), price * _QTY_LOT_SIZE,
                    signal_strength=result.get("signal_strength", 0.5),
                    total_capital=portfolio.total_capital(),
                    stop_loss_pct=stop_pct, strategy="atr_breakout",
                    open_positions=portfolio.position_count(),
                )
                qty = lots * _QTY_LOT_SIZE
                if qty > 0:
                    portfolio.open_position(
                        symbol, "BUY", price, qty,
                        signal_strength=result.get("signal_strength", 0.5),
                        strategy="atr_breakout", entry_atr=result.get("atr") or 0.0,
                    )
                    _set_entry_time(portfolio, symbol, today)
                    pos_after = portfolio.get_position(symbol)
                    trade_id = f"{symbol}_{pos_after['entry_time']}"
                    tracker.log_entry(
                        trade_id=trade_id, ticker=symbol, strategy_name="atr_breakout",
                        strategy_version=config.SYSTEM_VERSION, direction="LONG",
                        price=price, shares=qty, position_value=price * qty,
                        position_pct=(price * qty) / portfolio.total_capital(),
                        cash=portfolio.available_cash(), equity=portfolio.current_equity(),
                        timestamp=pos_after["entry_time"], run_id=run_id,
                        atr_entry=result.get("atr"),
                    )
                    trade_log.append({
                        "symbol": symbol, "side": "BUY", "date": today.isoformat(),
                        "price": price, "qty": qty, "reason": None, "pnl": None,
                    })

        equity_curve.append((today, portfolio.current_equity()))

    # ── Final liquidation at the last available close ─────────────────────────
    pos = portfolio.get_position(symbol)
    if pos is not None:
        last_price = float(df.iloc[sim_idx[-1]]["close"])
        last_date = df.iloc[sim_idx[-1]]["datetime"]
        closed = portfolio.close_position(symbol, last_price, reason="BACKTEST_END")
        trade_id = f"{symbol}_{closed['entry_time']}"
        tracker.log_exit(
            trade_id=trade_id, price=last_price,
            cash=portfolio.available_cash(), equity=portfolio.current_equity(),
            timestamp=last_date.isoformat(), exit_reason="BACKTEST_END",
        )
        trade_log.append({
            "symbol": symbol, "side": "SELL", "date": last_date.isoformat(),
            "price": last_price, "qty": closed["qty"], "reason": "BACKTEST_END",
            "pnl": closed["pnl"],
        })
        equity_curve[-1] = (last_date, portfolio.current_equity())

    metrics = _compute_metrics(equity_curve, trade_log, initial_cash)
    return {"metrics": metrics, "trades": trade_log, "run_id": run_id}


def _compute_metrics(equity_curve, trade_log, initial_cash: float) -> dict:
    eq = pd.Series([v for _, v in equity_curve],
                   index=pd.to_datetime([d for d, _ in equity_curve]))
    total_return_pct = (eq.iloc[-1] - initial_cash) / initial_cash * 100

    roll_max = eq.cummax()
    drawdown = (eq - roll_max) / roll_max.replace(0, pd.NA)
    max_dd_pct = float(drawdown.min()) * 100 if not drawdown.dropna().empty else 0.0

    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * math.sqrt(_ANNUALIZATION_DAYS)
              if daily_ret.std() > 0 else None)

    downside = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / downside.std() * math.sqrt(_ANNUALIZATION_DAYS)
               if len(downside) > 1 and downside.std() > 0 else None)

    sells = [t for t in trade_log if t["side"] == "SELL"]
    n_trades = len(sells)
    wins = [t for t in sells if (t["pnl"] or 0) > 0]
    losses = [t for t in sells if (t["pnl"] or 0) < 0]
    win_rate = len(wins) / n_trades if n_trades else None

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else None

    return {
        "total_return_pct": round(float(total_return_pct), 2),
        "sharpe_ratio": round(float(sharpe), 4) if sharpe is not None else None,
        "sortino_ratio": round(float(sortino), 4) if sortino is not None else None,
        "max_drawdown_pct": round(max_dd_pct, 2),
        "profit_factor": (round(profit_factor, 4)
                          if isinstance(profit_factor, float) and math.isfinite(profit_factor)
                          else profit_factor),
        "num_trades": n_trades,
        "win_rate_pct": round(win_rate * 100, 2) if win_rate is not None else None,
        "final_equity": round(float(eq.iloc[-1]), 2),
    }


def _print_result(symbol: str, result: dict) -> None:
    m = result["metrics"]
    print("=" * 70)
    print(f"Crypto Backtest: ATR Breakout (stock strategy, unmodified) — {symbol}")
    print("=" * 70)
    print(f"Total Return:   {m['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:   {m['sharpe_ratio']}")
    print(f"Sortino Ratio:  {m['sortino_ratio']}")
    print(f"Max Drawdown:   {m['max_drawdown_pct']:.2f}%")
    print(f"Profit Factor:  {m['profit_factor']}")
    print(f"Num Trades:     {m['num_trades']}")
    print(f"Win Rate:       {m['win_rate_pct']}%")
    print(f"Final Equity:   {m['final_equity']:,.2f} USDT")
    print("=" * 70)
    print("Trade Log:")
    for t in result["trades"]:
        pnl_str = f"  pnl={t['pnl']:+.2f}" if t["pnl"] is not None else ""
        reason_str = f"  reason={t['reason']}" if t["reason"] else ""
        print(f"  {t['date'][:19]}  {t['side']:4s}  qty={t['qty']:<12.6f}"
              f"  price={t['price']:<12.4f}{pnl_str}{reason_str}")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Crypto backtest driving the shared (unmodified) ATR Breakout strategy")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--cash", type=float, default=50_000.0)
    p.add_argument("--json", action="store_true", dest="output_json")
    args = p.parse_args()

    result = run_backtest(args.symbol, args.days, args.cash)

    if args.output_json:
        print(json_module.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        _print_result(args.symbol, result)
