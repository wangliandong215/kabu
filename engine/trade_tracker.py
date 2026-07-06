"""
engine/trade_tracker.py — Trade Intelligence Database (v2.6).

Side-effect-only persistence layer: it records what already happened (fills
decided by existing strategy / risk / sizing logic elsewhere) into SQLite so
the system has a queryable trade history. It never influences a trading
decision — every call site that wires this in (engine/runner.py's hook
calls, ingest_backtest_trade_log() below) sits strictly AFTER an order/fill
is already final, and every hook call at those sites is wrapped in
try/except so a logging failure can never break a trading pass.

Schema (see _SCHEMA below): trades (one row per round-trip trade) +
trade_attribution (entry/exit market-regime snapshot, 1:1 with trades) +
trade_events (append-only OPEN/EXIT log, ADD_POSITION/PARTIAL_EXIT/MOVE_STOP/
TRAIL/TAKE_PROFIT reserved for a future version — not written by this one).

trade_id is caller-supplied and must be the SAME string at log_entry() and
the matching log_exit() for one round-trip trade. engine/runner.py's hook
call sites use f"{code}_{entry_time}" (entry_time = the position's own
ISO-timestamp field, already unique per fill); ingest_backtest_trade_log()
below uses the equivalent f"{code}_{entry_date_iso}" convention for backtest
trade_log rows. Calling log_entry() twice with the same trade_id overwrites
(INSERT OR REPLACE) rather than raising — this makes re-running the same
backtest window idempotent instead of crashing on a PRIMARY KEY conflict.
"""
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.market_regime import MRD_VERSION, classify, to_regime_ctx

_DEFAULT_DB_PATH = Path(__file__).parent / "trade_history.db"

# Module-level write lock, same rationale as portfolio/tracker.py's
# _write_lock: prevents interleaved writes from two overlapping passes
# (e.g. a slow network delaying a previous run_once() pass) from corrupting
# multi-statement transactions.
_write_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id         TEXT PRIMARY KEY,
    ticker           TEXT NOT NULL,
    strategy_name    TEXT,
    strategy_version TEXT,
    direction        TEXT,
    entry_time       TEXT,
    entry_price      REAL,
    exit_time        TEXT,
    exit_price       REAL,
    shares           REAL,
    position_value   REAL,
    position_pct     REAL,
    holding_days     REAL,
    holding_hours    REAL,
    exit_reason      TEXT,
    pnl              REAL,
    pnl_pct          REAL,
    cash_before      REAL,
    equity_before    REAL,
    cash_after       REAL,
    equity_after     REAL
);

CREATE TABLE IF NOT EXISTS trade_attribution (
    trade_id           TEXT PRIMARY KEY REFERENCES trades(trade_id),
    entry_regime       INTEGER,
    entry_regime_label TEXT,
    entry_confidence   REAL,
    exit_regime        INTEGER,
    exit_regime_label  TEXT,
    exit_confidence    REAL,
    regime_drifted     INTEGER,
    mrd_version        TEXT,
    entry_context_json TEXT,
    exit_context_json  TEXT
);

CREATE TABLE IF NOT EXISTS trade_events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id   TEXT REFERENCES trades(trade_id),
    timestamp  TEXT,
    event_type TEXT,
    price      REAL,
    shares     REAL,
    note       TEXT
);
"""


class TradeTracker:
    """SQLite-backed recorder. Two public write methods (log_entry/log_exit)
    model one round-trip trade each; everything else (pnl/pnl_pct/holding_days
    /holding_hours/regime_drifted) is derived internally so callers never
    have to compute it themselves."""

    def __init__(self, db_path: Optional[Path] = None):
        self.path = Path(db_path or _DEFAULT_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with _write_lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def log_entry(self, trade_id: str, ticker: str, strategy_name: str,
                  strategy_version: str, direction: str, price: float,
                  shares: float, position_value: float, position_pct: float,
                  cash: Optional[float], equity: Optional[float],
                  timestamp: Optional[str] = None,
                  regime_ctx: Optional[dict] = None,
                  mrd_version: str = MRD_VERSION) -> None:
        """Record a new round-trip trade's open leg."""
        ts = timestamp or datetime.now().isoformat()
        ctx = regime_ctx or {}
        with _write_lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO trades (
                    trade_id, ticker, strategy_name, strategy_version, direction,
                    entry_time, entry_price, shares, position_value, position_pct,
                    cash_before, equity_before
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (trade_id, ticker, strategy_name, strategy_version, direction,
                 ts, price, shares, position_value, position_pct, cash, equity),
            )
            self._conn.execute(
                """INSERT INTO trade_events
                   (trade_id, timestamp, event_type, price, shares, note)
                   VALUES (?,?,?,?,?,?)""",
                (trade_id, ts, "OPEN", price, shares, None),
            )
            self._conn.execute(
                """INSERT OR REPLACE INTO trade_attribution (
                    trade_id, entry_regime, entry_regime_label, entry_confidence,
                    mrd_version, entry_context_json
                ) VALUES (?,?,?,?,?,?)""",
                (trade_id, ctx.get("regime"), ctx.get("regime_label"),
                 ctx.get("confidence"), mrd_version,
                 json.dumps(regime_ctx, ensure_ascii=False) if regime_ctx else None),
            )
            self._conn.commit()

    def log_exit(self, trade_id: str, price: float, cash: Optional[float],
                equity: Optional[float], timestamp: Optional[str] = None,
                exit_reason: str = "", regime_ctx: Optional[dict] = None) -> None:
        """Record a round-trip trade's close leg. pnl/pnl_pct/holding_days/
        holding_hours are computed here from the row log_entry() wrote —
        raises ValueError if trade_id was never opened (a hook bug, not a
        normal runtime condition, so this is allowed to raise; callers wrap
        this in try/except so it never propagates into the trading pass)."""
        ts = timestamp or datetime.now().isoformat()
        with _write_lock:
            row = self._conn.execute(
                "SELECT entry_price, entry_time, shares, direction "
                "FROM trades WHERE trade_id=?", (trade_id,),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"trade_tracker.log_exit: unknown trade_id {trade_id!r} "
                    f"— log_entry() must be called first")

            entry_price, entry_time, shares, direction = (
                row["entry_price"], row["entry_time"], row["shares"], row["direction"])
            sign = -1.0 if direction == "SHORT" else 1.0
            pnl = (price - entry_price) * shares * sign
            cost_basis = entry_price * shares
            pnl_pct = pnl / cost_basis if cost_basis else 0.0

            entry_dt = datetime.fromisoformat(entry_time)
            exit_dt = datetime.fromisoformat(ts)
            holding_hours = (exit_dt - entry_dt).total_seconds() / 3600.0
            holding_days = holding_hours / 24.0

            self._conn.execute(
                """UPDATE trades SET exit_time=?, exit_price=?, holding_days=?,
                   holding_hours=?, exit_reason=?, pnl=?, pnl_pct=?,
                   cash_after=?, equity_after=? WHERE trade_id=?""",
                (ts, price, holding_days, holding_hours, exit_reason,
                 pnl, pnl_pct, cash, equity, trade_id),
            )
            self._conn.execute(
                """INSERT INTO trade_events
                   (trade_id, timestamp, event_type, price, shares, note)
                   VALUES (?,?,?,?,?,?)""",
                (trade_id, ts, "EXIT", price, shares, exit_reason),
            )

            attr_row = self._conn.execute(
                "SELECT entry_regime FROM trade_attribution WHERE trade_id=?",
                (trade_id,),
            ).fetchone()
            entry_regime = attr_row["entry_regime"] if attr_row else None
            ctx = regime_ctx or {}
            exit_regime = ctx.get("regime")
            both_known = entry_regime is not None and exit_regime is not None
            drifted = int(entry_regime != exit_regime) if both_known else None

            self._conn.execute(
                """UPDATE trade_attribution SET exit_regime=?, exit_regime_label=?,
                   exit_confidence=?, regime_drifted=?, exit_context_json=?
                   WHERE trade_id=?""",
                (exit_regime, ctx.get("regime_label"), ctx.get("confidence"),
                 drifted,
                 json.dumps(regime_ctx, ensure_ascii=False) if regime_ctx else None,
                 trade_id),
            )
            self._conn.commit()


# ── Shared helper: OHLCV DataFrame -> regime_ctx dict ─────────────────────────

def build_regime_ctx(df, mrd_version: str = MRD_VERSION) -> Optional[dict]:
    """Compute a JSON-ready regime snapshot for use as the `regime_ctx`
    argument to log_entry/log_exit. Never raises — returns None if
    classification isn't possible (missing/empty df, etc.), so a caller can
    pass regime_ctx=None and get a NULL attribution rather than a crashed
    trading pass."""
    try:
        if df is None or len(df) == 0:
            return None
        return to_regime_ctx(classify(df))
    except Exception:
        return None


# ── Backtest ingestion (post-hoc, does not touch backtest_portfolio.py) ──────

# Reasons that mark a genuinely NEW round-trip trade opening in
# backtest_portfolio.py's trade_log. "PROMOTE" (TRENDING_EARLY -> confirmed
# top-up) is deliberately excluded: it enlarges an already-open trade rather
# than starting one, and log_entry/log_exit only model whole round-trips.
_BACKTEST_OPEN_REASONS = {"SIGNAL", "QQQ_BETA_FLOOR"}


def ingest_backtest_trade_log(tracker: TradeTracker, trade_log: list,
                              all_data: dict,
                              equity_curve: Optional[pd.Series] = None,
                              strategy_version: str = "backtest",
                              mrd_version: str = MRD_VERSION) -> int:
    """Post-hoc ingestion of one completed backtest_portfolio.py run into the
    Trade Intelligence Database.

    Reads ONLY the already-finalized `result["trade_log"]` list returned by
    simulate_from_prepared() — any Active Replacement attempt that got rolled
    back by that function's _undo_replacement() is already popped out of
    trade_log before it's ever returned, so this function never needs to
    reason about rollback itself, and never touches backtest_portfolio.py's
    simulation loop at all. That makes it impossible for this function to
    affect any backtest result, by construction (the highest-priority
    constraint for this module).

    Pairs each BUY row (reason in _BACKTEST_OPEN_REASONS — a true new-trade
    open) with the next SELL row for the same ticker: this system holds at
    most one open position per code at a time, so that pairing is
    unambiguous FIFO per code. A position still open when trade_log ends
    (never closed within the simulated window) still gets a log_entry() call
    with no matching log_exit() — it shows up as an open trade in the DB,
    which is correct.

    all_data : the same {code: OHLCV DataFrame} dict produced by
        _prepare_backtest_data() and passed into simulate_from_prepared() —
        reused here (not re-fetched) purely to compute a regime_ctx snapshot
        as of each trade's date.
    equity_curve : optional date-indexed Series (result["equity_curve"]) used
        to fill equity_before/equity_after with that day's mark-to-market
        equity. cash_before/cash_after are left NULL — backtest_portfolio.py
        has no per-trade (only end-of-day) cash ledger, and fabricating a
        number would overstate this data's actual precision.

    Returns the number of round-trip trades ingested (open+closed pairs;
    a dangling still-open position at the end counts too).
    """
    open_rows: dict = {}
    count = 0

    def _equity_at(date) -> Optional[float]:
        if equity_curve is None:
            return None
        try:
            return float(equity_curve.loc[pd.Timestamp(date)])
        except KeyError:
            return None

    def _regime_ctx_at(code, date) -> Optional[dict]:
        df = all_data.get(code)
        if df is None:
            return None
        sliced = df.loc[:pd.Timestamp(date)]
        return build_regime_ctx(sliced, mrd_version)

    for row in sorted(trade_log, key=lambda r: r["date"]):
        code = row["code"]
        if row["side"] == "BUY":
            if row.get("reason") not in _BACKTEST_OPEN_REASONS:
                continue   # PROMOTE etc. — enlarges an existing trade, not a new one
            open_rows[code] = row
            continue

        entry_row = open_rows.pop(code, None)
        if entry_row is None:
            continue   # defensive: shouldn't happen given one-position-per-code

        entry_time = pd.Timestamp(entry_row["date"]).isoformat()
        exit_time = pd.Timestamp(row["date"]).isoformat()
        trade_id = f"{code}_{entry_time}"
        shares = entry_row["qty"]
        entry_price = entry_row["price"]
        position_value = entry_price * shares
        equity_before = _equity_at(entry_row["date"])

        tracker.log_entry(
            trade_id=trade_id, ticker=code,
            strategy_name=entry_row.get("strategy"),
            strategy_version=strategy_version,
            direction="LONG", price=entry_price, shares=shares,
            position_value=position_value,
            position_pct=(position_value / equity_before) if equity_before else None,
            cash=None, equity=equity_before, timestamp=entry_time,
            regime_ctx=_regime_ctx_at(code, entry_row["date"]),
            mrd_version=mrd_version,
        )
        tracker.log_exit(
            trade_id=trade_id, price=row["price"], cash=None,
            equity=_equity_at(row["date"]), timestamp=exit_time,
            exit_reason=row.get("reason", ""),
            regime_ctx=_regime_ctx_at(code, row["date"]),
        )
        count += 1

    # Positions opened but never closed within the simulated window.
    for code, entry_row in open_rows.items():
        entry_time = pd.Timestamp(entry_row["date"]).isoformat()
        shares = entry_row["qty"]
        entry_price = entry_row["price"]
        position_value = entry_price * shares
        equity_before = _equity_at(entry_row["date"])
        tracker.log_entry(
            trade_id=f"{code}_{entry_time}", ticker=code,
            strategy_name=entry_row.get("strategy"),
            strategy_version=strategy_version,
            direction="LONG", price=entry_price, shares=shares,
            position_value=position_value,
            position_pct=(position_value / equity_before) if equity_before else None,
            cash=None, equity=equity_before, timestamp=entry_time,
            regime_ctx=_regime_ctx_at(code, entry_row["date"]),
            mrd_version=mrd_version,
        )
        count += 1

    return count
