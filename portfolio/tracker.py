"""
portfolio/tracker.py — persistent position state.

Positions are stored in portfolio/positions.json so they survive restarts.
The tracker also maintains peak_value for max-drawdown calculation.

Usage:
    from portfolio.tracker import Portfolio
    p = Portfolio()
    p.open_position("US.AAPL", "BUY", 150.23, 10)
    p.close_position("US.AAPL", 165.00)
    p.print_summary()
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

import config


_DEFAULT_PATH = Path(__file__).parent / "positions.json"

# Module-level write lock: prevents concurrent saves from corrupting the JSON
# when two passes overlap (e.g. slow network delays a previous run).
# For cross-process safety, install `filelock` and replace with FileLock.
_write_lock = threading.Lock()


class Portfolio:

    def __init__(self, path: Path = None):
        self.path = Path(path or _DEFAULT_PATH)
        self.data = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Back-fill fields added in later versions
                data.setdefault("realized_pnl", 0.0)
                data.setdefault("peak_equity",  data.get("initial_cash", config.INITIAL_CAPITAL))
                data.setdefault("cooldowns",    {})   # code -> ISO date string, blocked through this date
                for pos in data.get("positions", {}).values():
                    pos.setdefault("avg_cost",          pos["entry_price"])
                    pos.setdefault("signal_strength",    0.5)
                    pos.setdefault("entry_atr",          0.0)
                    pos.setdefault("atr_mult",           config.ATR_MULT_BASE)
                    pos.setdefault("breakeven_locked",   False)
                    pos.setdefault("trail_stop",         None)
                    pos.setdefault("score_label",        None)
                    pos.setdefault("total_score",        None)
                return data
            except Exception:
                pass
        return {
            "positions":    {},
            "peak_value":   0.0,
            "initial_cash": config.INITIAL_CAPITAL,
            "realized_pnl": 0.0,
            "peak_equity":  config.INITIAL_CAPITAL,
            "cooldowns":    {},   # code -> ISO date string, blocked through this date
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            # Atomic write: serialize to a temp file, then rename over the
            # target.  os.replace() is atomic on NTFS (Windows) and POSIX,
            # so a crash mid-write never leaves a half-written JSON.
            tmp = self.path.with_suffix(".tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self.path)
            except Exception:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                raise

    # ── Position operations ───────────────────────────────────────────────────

    def get_position(self, code: str) -> dict | None:
        """Return position dict or None if not held."""
        return self.data["positions"].get(code)

    def position_count(self) -> int:
        return len(self.data["positions"])

    def open_position(self, code: str, side: str,
                      entry_price: float, qty: int,
                      signal_strength: float = 0.5,
                      strategy: str = "",
                      entry_atr: float = 0.0,
                      score_label: str = None,
                      total_score: float = None) -> None:
        entry_trail = (entry_price - config.ATR_MULT_BASE * entry_atr
                       if entry_atr > 0 else None)
        self.data["positions"][code] = {
            "side":             side,
            "entry_price":      entry_price,
            "avg_cost":         entry_price,
            "qty":              qty,
            "signal_strength":  signal_strength,
            "strategy":         strategy,        # locked — exit must use same strategy
            "entry_atr":        entry_atr,       # ATR at entry — for trailing stop rebuild
            "atr_mult":         config.ATR_MULT_BASE,
            "breakeven_locked": False,
            "trail_stop":       entry_trail,     # initial stop = entry − 2×ATR
            "entry_time":       datetime.now().isoformat(),
            # v2.3主动置换（Portfolio Capacity Manager）移植进实盘用——跟
            # backtest_portfolio.py的positions[code]同名字段同语义，供
            # portfolio.capacity_manager.find_replaceable_position()判断
            # 这个持仓将来能不能被换出。None = 未传入分数（旧存档/其他
            # 调用点如pyramid加仓，不影响其余逻辑）。
            "score_label":      score_label,
            "total_score":      total_score,
        }
        self._update_peak()
        self._save()

    def add_to_position(self, code: str, add_price: float,
                        add_qty: int, new_strength: float,
                        strategy: str = None) -> None:
        """
        Scale into an existing position (right-side pyramid, or a
        TRENDING_EARLY -> TRENDING_UP promotion top-up).

        strategy : if given, overwrites pos["strategy"] — used by the
                   promotion step in engine/runner.py to relabel a filled
                   trial (atr_breakout_early) as a confirmed atr_breakout
                   position once it's been topped up to full size.
        """
        pos = self.data["positions"].get(code)
        if pos is None:
            return
        old_qty  = pos["qty"]
        old_cost = pos["avg_cost"]
        total_qty = old_qty + add_qty
        new_avg   = (old_qty * old_cost + add_qty * add_price) / total_qty
        pos["qty"]             = total_qty
        pos["avg_cost"]        = round(new_avg, 4)
        pos["signal_strength"] = new_strength
        if strategy is not None:
            pos["strategy"] = strategy
        self._save()

    def close_position(self, code: str, exit_price: float, reason: str = "") -> dict:
        """Remove position, accumulate realized P&L, return trade summary."""
        pos = self.data["positions"].pop(code, None)
        if pos is None:
            return {}
        pnl = (exit_price - pos["avg_cost"]) * pos["qty"]
        if pos["side"] == "SELL":
            pnl = -pnl
        self.data["realized_pnl"] = self.data.get("realized_pnl", 0.0) + pnl
        self._maybe_set_trial_cooldown(code, pos, reason)
        self._update_peak_equity()
        self._save()
        return {**pos, "exit_price": exit_price, "pnl": round(pnl, 4)}

    # ── TRENDING_EARLY cooldown ──────────────────────────────────────────────

    def _maybe_set_trial_cooldown(self, code: str, pos: dict, reason: str) -> None:
        """
        If an unpromoted TRENDING_EARLY trial position gets stopped out within
        config.TRENDING_EARLY_STOPOUT_LOOKBACK_DAYS trading days of entry,
        block new entries in this stock for config.TRENDING_EARLY_COOLDOWN_DAYS
        trading days — prevents repeated whipsaw re-entries in a choppy range.
        A promoted position (strategy already relabeled to "atr_breakout" by
        add_to_position) never triggers this — only a trial that failed while
        still unconfirmed does.
        """
        if reason != "STOP_LOSS" or pos.get("strategy") != "atr_breakout_early":
            return
        entry_date = np.datetime64(pos.get("entry_time", datetime.now().isoformat())[:10])
        today = np.datetime64(datetime.now().date().isoformat())
        held_days = int(np.busday_count(entry_date, today))
        if held_days > config.TRENDING_EARLY_STOPOUT_LOOKBACK_DAYS:
            return
        until = np.busday_offset(today, config.TRENDING_EARLY_COOLDOWN_DAYS, roll="forward")
        self.data.setdefault("cooldowns", {})[code] = str(until)

    def is_cooldown(self, code: str) -> bool:
        """True if `code` is still inside its TRENDING_EARLY stop-out cooldown."""
        until = self.data.get("cooldowns", {}).get(code)
        if not until:
            return False
        today = datetime.now().date().isoformat()
        if today >= until:
            del self.data["cooldowns"][code]
            self._save()
            return False
        return True

    # ── Portfolio-level metrics ───────────────────────────────────────────────

    def total_capital(self) -> float:
        return self.data.get("initial_cash", config.INITIAL_CAPITAL)

    def deployed_capital(self) -> float:
        return sum(p["entry_price"] * p["qty"]
                   for p in self.data["positions"].values())

    def available_cash(self) -> float:
        return max(0.0, self.total_capital() - self.deployed_capital())

    def exposure_pct(self) -> float:
        tc = self.total_capital()
        return self.deployed_capital() / tc if tc > 0 else 0.0

    def sector_exposure_pct(self, sector: str) -> float:
        tc = self.total_capital()
        if tc <= 0:
            return 0.0
        deployed = sum(
            p["entry_price"] * p["qty"]
            for code, p in self.data["positions"].items()
            if config.SECTOR_MAP.get(code, "other") == sector
        )
        return deployed / tc

    def current_equity(self) -> float:
        """Realized equity = initial capital + all closed-trade P&L."""
        return self.total_capital() + self.data.get("realized_pnl", 0.0)

    def equity_drawdown_pct(self) -> float:
        """Drawdown from peak realized equity (0.0 → no drawdown)."""
        peak = self.data.get("peak_equity", self.total_capital())
        curr = self.current_equity()
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - curr) / peak)

    def is_headwind(self) -> bool:
        """True when realized drawdown exceeds the half-Kelly threshold."""
        return self.equity_drawdown_pct() >= config.HALFKELLY_DRAWDOWN_THRESHOLD

    def current_value(self) -> float:
        return self.deployed_capital()

    def _update_peak(self) -> None:
        v = self.total_capital()
        if v > self.data.get("peak_value", 0.0):
            self.data["peak_value"] = v

    def _update_peak_equity(self) -> None:
        eq = self.current_equity()
        if eq > self.data.get("peak_equity", 0.0):
            self.data["peak_equity"] = eq

    # ── Display ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "positions":      self.data["positions"],
            "position_count": self.position_count(),
            "peak_value":     self.data.get("peak_value", 0.0),
        }

    def print_summary(self) -> None:
        positions = self.data["positions"]
        print("=" * 55)
        print(f"Portfolio: {len(positions)} open position(s)")
        print("=" * 55)
        if not positions:
            print("  (no open positions)")
        else:
            for code, pos in positions.items():
                print(f"  {code:20s}  {pos['side']:4s}  qty={pos['qty']:>6d}"
                      f"  entry={pos['entry_price']:.4f}")
                print(f"  {'':20s}  opened {pos['entry_time'][:19]}")
        print("=" * 55)
