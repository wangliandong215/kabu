"""
engine/trade_tracker.py — Trade Intelligence Database (v2.6, extended v2.8).

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
TRAIL/TAKE_PROFIT reserved for a future version — not written by this one) +
trade_daily (v2.8, one row per trade per calendar date — running MFE/MAE
snapshot while a position is open) + metadata (v2.8, one row per run —
backtest window or live/paper trading session — so results from different
strategy versions/parameter sets are never mixed up downstream).

trade_id is caller-supplied and must be the SAME string at log_entry() and
the matching log_exit() for one round-trip trade. engine/runner.py's hook
call sites use f"{code}_{entry_time}" (entry_time = the position's own
ISO-timestamp field, already unique per fill); ingest_backtest_trade_log()
below uses the equivalent f"{code}_{entry_date_iso}" convention for backtest
trade_log rows. Calling log_entry() twice with the same trade_id overwrites
(INSERT OR REPLACE) rather than raising — this makes re-running the same
backtest window idempotent instead of crashing on a PRIMARY KEY conflict.
This is also why trade_id is a deterministic string and not a random UUID
(v2.8 considered switching to UUID per an earlier draft spec, but that would
break this idempotency and require persisting a new field on Portfolio's
position dict — kept as-is, deliberately).
"""
import hashlib
import json
import sqlite3
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.market_regime import MRD_VERSION, Regime, classify, to_regime_ctx
from engine import regime_store

_DEFAULT_DB_PATH = Path(__file__).parent / "trade_history.db"

# ── V2.8 exit-reason enum ─────────────────────────────────────────────────────
# Maps the free-text reason strings risk/guard.py and backtest_portfolio.py
# actually produce today onto a small, stable set of codes for analytics
# grouping (e.g. "win rate by exit type"). EXIT_MANUAL/EXIT_TIMEOUT/
# EXIT_MARGIN are reserved for exit types that don't exist anywhere in this
# codebase yet — defined now so a future exit type doesn't need another
# schema migration, but normalize_exit_reason() never produces them today.
EXIT_STOP_LOSS = "EXIT_STOP_LOSS"
EXIT_TRAILING = "EXIT_TRAILING"
EXIT_TAKE_PROFIT = "EXIT_TAKE_PROFIT"
EXIT_STRATEGY_SIGNAL = "EXIT_STRATEGY_SIGNAL"
EXIT_REGIME_BREAK = "EXIT_REGIME_BREAK"
EXIT_REPLACED = "EXIT_REPLACED"
EXIT_MANUAL = "EXIT_MANUAL"      # reserved — not produced by any code path yet
EXIT_TIMEOUT = "EXIT_TIMEOUT"    # reserved — not produced by any code path yet
EXIT_MARGIN = "EXIT_MARGIN"      # reserved — not produced by any code path yet
EXIT_UNKNOWN = "EXIT_UNKNOWN"


def normalize_exit_reason(raw: Optional[str]) -> str:
    """Classify a raw exit_reason string (risk/guard.py::check_exit_ordered,
    engine/runner.py's MA200_BREAK/ACTIVE_REPLACEMENT, or
    backtest_portfolio.py's STRAT_EXIT) into one of the EXIT_* codes above.
    Never raises — unrecognized/empty input maps to EXIT_UNKNOWN so this is
    always safe to call from inside log_exit()."""
    if not raw:
        return EXIT_UNKNOWN
    if raw == "STOP_LOSS":
        return EXIT_STOP_LOSS
    if raw == "ATR_TRAIL":
        return EXIT_TRAILING
    if raw == "TAKE_PROFIT":
        return EXIT_TAKE_PROFIT
    if raw == "STRAT_EXIT" or raw.startswith("STRATEGY_EXIT("):
        return EXIT_STRATEGY_SIGNAL
    if raw == "MA200_BREAK":
        return EXIT_REGIME_BREAK
    if raw == "ACTIVE_REPLACEMENT":
        return EXIT_REPLACED
    return EXIT_UNKNOWN

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
    exit_reason_code TEXT,
    pnl              REAL,
    pnl_pct          REAL,
    cash_before      REAL,
    equity_before    REAL,
    cash_after       REAL,
    equity_after     REAL,
    run_id           TEXT REFERENCES metadata(run_id),
    atr_entry        REAL,
    sector           TEXT,
    market_environment INTEGER,
    entry_rank       INTEGER,
    confidence_score REAL,
    risk_per_trade   REAL,
    commission       REAL,
    slippage         REAL,
    mfe              REAL,
    mae              REAL
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

CREATE TABLE IF NOT EXISTS trade_daily (
    trade_id          TEXT REFERENCES trades(trade_id),
    date              TEXT,
    close             REAL,
    floating_pnl      REAL,
    floating_pnl_pct  REAL,
    mfe               REAL,
    mae               REAL,
    atr               REAL,
    regime            INTEGER,
    regime_label      TEXT,
    regime_confidence REAL,
    PRIMARY KEY (trade_id, date)
);

CREATE TABLE IF NOT EXISTS metadata (
    run_id           TEXT PRIMARY KEY,
    strategy_version TEXT,
    market           TEXT,
    start_date       TEXT,
    end_date         TEXT,
    created_at       TEXT,
    hmm_version      TEXT,
    parameter_hash   TEXT,
    git_commit       TEXT
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
                  mrd_version: str = MRD_VERSION,
                  run_id: Optional[str] = None,
                  atr_entry: Optional[float] = None,
                  sector: Optional[str] = None,
                  market_environment: Optional[int] = None,
                  entry_rank: Optional[int] = None,
                  confidence_score: Optional[float] = None,
                  risk_per_trade: Optional[float] = None,
                  commission: Optional[float] = None,
                  slippage: Optional[float] = None) -> None:
        """Record a new round-trip trade's open leg. All v2.8 kwargs
        (run_id..slippage) are optional and default to None — existing
        call sites that don't pass them are unaffected."""
        ts = timestamp or datetime.now().isoformat()
        ctx = regime_ctx or {}
        with _write_lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO trades (
                    trade_id, ticker, strategy_name, strategy_version, direction,
                    entry_time, entry_price, shares, position_value, position_pct,
                    cash_before, equity_before, run_id, atr_entry, sector,
                    market_environment, entry_rank, confidence_score,
                    risk_per_trade, commission, slippage
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (trade_id, ticker, strategy_name, strategy_version, direction,
                 ts, price, shares, position_value, position_pct, cash, equity,
                 run_id, atr_entry, sector, market_environment, entry_rank,
                 confidence_score, risk_per_trade, commission, slippage),
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
                exit_reason: str = "", regime_ctx: Optional[dict] = None,
                commission: Optional[float] = None) -> None:
        """Record a round-trip trade's close leg. pnl/pnl_pct/holding_days/
        holding_hours/exit_reason_code/mfe/mae are computed here — raises
        ValueError if trade_id was never opened (a hook bug, not a normal
        runtime condition, so this is allowed to raise; callers wrap this in
        try/except so it never propagates into the trading pass).

        commission (v2.8, optional): round-trip total, if the caller has one
        (e.g. ingest_backtest_trade_log() passes backtest_portfolio.py's
        already-computed fee). Left None for live/paper trades — no
        commission model exists in that path today.

        mfe/mae are aggregated from trade_daily (MAX(mfe)/MIN(mae) across
        every day this trade was open, written by update_position_metrics())
        rather than passed in — a trade with no trade_daily rows (e.g.
        opened and closed within the same pass) simply keeps mfe/mae NULL."""
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
            exit_reason_code = normalize_exit_reason(exit_reason)

            agg = self._conn.execute(
                "SELECT MAX(mfe) AS mfe, MIN(mae) AS mae FROM trade_daily "
                "WHERE trade_id=?", (trade_id,),
            ).fetchone()
            mfe = agg["mfe"] if agg else None
            mae = agg["mae"] if agg else None

            self._conn.execute(
                """UPDATE trades SET exit_time=?, exit_price=?, holding_days=?,
                   holding_hours=?, exit_reason=?, exit_reason_code=?, pnl=?,
                   pnl_pct=?, cash_after=?, equity_after=?, commission=?,
                   mfe=?, mae=? WHERE trade_id=?""",
                (ts, price, holding_days, holding_hours, exit_reason,
                 exit_reason_code, pnl, pnl_pct, cash, equity, commission,
                 mfe, mae, trade_id),
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

    def update_position_metrics(self, trade_id: str, date: str, close: float,
                                 entry_price: float, shares: float,
                                 direction: str = "LONG",
                                 high: Optional[float] = None,
                                 low: Optional[float] = None,
                                 atr: Optional[float] = None,
                                 regime_ctx: Optional[dict] = None,
                                 timestamp: Optional[str] = None) -> None:
        """v2.8 — upsert one trade_daily row for (trade_id, date). Safe to
        call multiple times for the same (trade_id, date) (e.g. once per
        live scan pass, default every 300s) — mfe/mae accumulate as that
        day's running high-water-mark via SQLite's ON CONFLICT DO UPDATE
        (MAX/MIN against the existing stored value), everything else
        (close/floating_pnl/atr/regime) is simply overwritten with the
        latest snapshot.

        Unlike log_exit(), this never raises on an unknown trade_id — it's
        called far more often (every open position, every pass) and every
        call site already wraps it in try/except per the shadow-logging
        discipline, so a silent no-op-into-orphan-row is preferable to a
        ValueError callers would just swallow anyway.

        high/low default to close when omitted (live path — a single price
        sample per pass) — backtest ingestion passes the day's true bar
        high/low for a more accurate MFE/MAE."""
        ts = timestamp or datetime.now().isoformat()
        sign = -1.0 if direction == "SHORT" else 1.0
        cost_basis = entry_price * shares

        floating_pnl = (close - entry_price) * shares * sign
        floating_pnl_pct = floating_pnl / cost_basis if cost_basis else None

        fav_price = (high if high is not None else close) if direction != "SHORT" \
            else (low if low is not None else close)
        adv_price = (low if low is not None else close) if direction != "SHORT" \
            else (high if high is not None else close)
        mfe = (fav_price - entry_price) * shares * sign
        mae = (adv_price - entry_price) * shares * sign

        ctx = regime_ctx or {}
        with _write_lock:
            self._conn.execute(
                """INSERT INTO trade_daily (
                    trade_id, date, close, floating_pnl, floating_pnl_pct,
                    mfe, mae, atr, regime, regime_label, regime_confidence
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(trade_id, date) DO UPDATE SET
                    close=excluded.close,
                    floating_pnl=excluded.floating_pnl,
                    floating_pnl_pct=excluded.floating_pnl_pct,
                    mfe=MAX(trade_daily.mfe, excluded.mfe),
                    mae=MIN(trade_daily.mae, excluded.mae),
                    atr=excluded.atr,
                    regime=excluded.regime,
                    regime_label=excluded.regime_label,
                    regime_confidence=excluded.regime_confidence""",
                (trade_id, date, close, floating_pnl, floating_pnl_pct,
                 mfe, mae, atr, ctx.get("regime"), ctx.get("regime_label"),
                 ctx.get("confidence")),
            )
            self._conn.commit()

    def log_run_metadata(self, run_id: Optional[str] = None,
                          strategy_version: Optional[str] = None,
                          market: Optional[str] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          hmm_version: Optional[str] = None,
                          parameter_hash: Optional[str] = None,
                          git_commit: Optional[str] = None) -> str:
        """v2.8 — record one row describing a run (a backtest window or one
        live/paper trading session), so trades from different strategy
        versions/parameter sets are never mixed up downstream. Returns the
        run_id used (auto-generated uuid4 hex if not supplied) so the caller
        can thread it into log_entry(run_id=...) calls for that same run."""
        run_id = run_id or uuid.uuid4().hex
        if git_commit is None:
            git_commit = _detect_git_commit()
        with _write_lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO metadata (
                    run_id, strategy_version, market, start_date, end_date,
                    created_at, hmm_version, parameter_hash, git_commit
                ) VALUES (?,?,?,?,?,?,?,?,?)""",
                (run_id, strategy_version, market, start_date, end_date,
                 datetime.now().isoformat(), hmm_version, parameter_hash,
                 git_commit),
            )
            self._conn.commit()
        return run_id

    def flush(self) -> None:
        """v2.8 — every write already commits synchronously (see each method
        above), so this is a checkpoint convenience, not a durability
        requirement: folds the WAL file back into the main DB file."""
        with _write_lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    _EXPORTABLE_TABLES = {"trades", "trade_attribution", "trade_events",
                           "trade_daily", "metadata"}

    def export_csv(self, table: str, path) -> None:
        """v2.8 — dump one table to CSV. `table` is checked against a fixed
        allowlist before being interpolated into SQL (there's no
        parameterized-identifier syntax in SQLite) since it may originate
        from caller input."""
        if table not in self._EXPORTABLE_TABLES:
            raise ValueError(
                f"trade_tracker.export_csv: unknown table {table!r} — "
                f"must be one of {sorted(self._EXPORTABLE_TABLES)}")
        df = pd.read_sql_query(f"SELECT * FROM {table}", self._conn)
        df.to_csv(path, index=False)


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


def build_regime_ctx_preferring_hmm(code: str, df=None,
                                     mrd_version: str = MRD_VERSION) -> Optional[dict]:
    """v2.8 — live-path regime attribution: reads engine.regime_store's
    durable Regime Interface (engine/hmm_shadow.py's daily batch output)
    first, falling back to the rule-based build_regime_ctx(df) only when the
    store has no entry for `code` yet (its HMM model hasn't been trained, or
    the daily Shadow Mode batch hasn't been run for it — a manual/cron job,
    not automatic). Only imports engine.regime_store (json/os/threading) —
    NOT engine.hmm_regime, which pulls in hmmlearn/sklearn and stays
    confined to the offline training + shadow-batch path so those remain
    optional dependencies for the live trading path.

    Tags which source actually produced the snapshot via a "source" key
    ("HMM" or "RULE") inside the returned dict, so it's visible in the
    entry_context_json/exit_context_json JSON blob without a schema change.
    Never raises — same contract as build_regime_ctx()."""
    try:
        store_entry = regime_store.get_regime_interface(code)
        label = store_entry.get("current_regime")
        if label is not None:
            regime_member = Regime[f"HMM_{label.upper()}"]
            return {
                "regime": regime_member.value,
                "regime_label": regime_member.name,
                "confidence": store_entry.get("regime_confidence"),
                "features": {
                    "regime_duration": store_entry.get("regime_duration"),
                    "regime_changed_today": store_entry.get("regime_changed_today"),
                },
                "timestamp": None,
                "hmm_version": store_entry.get("hmm_version"),
                "source": "HMM",
            }
    except Exception:
        pass

    ctx = build_regime_ctx(df, mrd_version=mrd_version)
    if ctx is not None:
        ctx["source"] = "RULE"
    return ctx


def _detect_git_commit() -> Optional[str]:
    """v2.8 — best-effort `git rev-parse --short HEAD`, run from this file's
    repo. Never raises (missing git binary, not a repo, detached weirdness,
    etc. all degrade to None) since it's only ever used to fill an
    analytics-only metadata column."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        pass
    return None


def default_parameter_snapshot() -> dict:
    """v2.8 — a small dict of the config constants that actually govern
    trading behavior, for compute_parameter_hash(). Deliberately not
    exhaustive — extend this dict (no schema change needed, it's just
    hashed) as more parameters prove worth distinguishing runs by."""
    import config
    return {
        "system_version": getattr(config, "SYSTEM_VERSION", None),
        "risk_per_trade_pct": getattr(config, "RISK_PER_TRADE_PCT", None),
        "stop_loss_pct": getattr(config, "STOP_LOSS_PCT", None),
        "take_profit_pct": getattr(config, "TAKE_PROFIT_PCT", None),
        "max_positions": getattr(config, "MAX_POSITIONS", None),
        "atr_mult_base": getattr(config, "ATR_MULT_BASE", None),
        "qqq_core_target_pct": getattr(config, "QQQ_CORE_TARGET_PCT", None),
        "watchlist_size": len(getattr(config, "WATCHLIST", [])),
    }


def compute_parameter_hash(params: dict) -> str:
    """v2.8 — stable short hash of an arbitrary parameter dict (typically
    default_parameter_snapshot()'s output) for metadata.parameter_hash, so
    two runs with different config are never silently conflated."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ── Backtest ingestion (post-hoc, does not touch backtest_portfolio.py) ──────

# Reasons that mark a genuinely NEW round-trip trade opening in
# backtest_portfolio.py's trade_log. "PROMOTE" (TRENDING_EARLY -> confirmed
# top-up) is deliberately excluded: it enlarges an already-open trade rather
# than starting one, and log_entry/log_exit only model whole round-trips.
_BACKTEST_OPEN_REASONS = {"SIGNAL", "QQQ_BETA_FLOOR"}


def _hmm_label_valid(label) -> bool:
    """True if `label` is a usable HMM regime label string (not None/NaN —
    decode_regime_series_causal() returns NaN before HMM_MIN_DECODE_BARS of
    warmup)."""
    if label is None:
        return False
    if isinstance(label, float) and pd.isna(label):
        return False
    return True


def _hmm_regime_ctx_from_label(label, confidence, duration=None) -> Optional[dict]:
    """Build a regime_ctx dict (same shape as build_regime_ctx()'s output)
    from an already-decoded HMM label/confidence pair — shared by the
    trade_log regime_at_entry/_exit fields and daily_position_log rows
    below. Returns None if `label` isn't usable (see _hmm_label_valid)."""
    if not _hmm_label_valid(label):
        return None
    try:
        regime_member = Regime[f"HMM_{label.upper()}"]
    except KeyError:
        return None
    return {
        "regime": regime_member.value,
        "regime_label": regime_member.name,
        "confidence": confidence,
        "features": {"duration": duration},
        "timestamp": None,
        "source": "HMM",
    }


def ingest_backtest_trade_log(tracker: TradeTracker, trade_log: list,
                              all_data: dict,
                              equity_curve: Optional[pd.Series] = None,
                              strategy_version: str = "backtest",
                              mrd_version: str = MRD_VERSION,
                              run_id: Optional[str] = None,
                              daily_position_log: Optional[list] = None) -> int:
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
        used as the rule-based regime FALLBACK (see regime source note
        below), not re-fetched.
    equity_curve : optional date-indexed Series (result["equity_curve"]) used
        to fill equity_before/equity_after with that day's mark-to-market
        equity. cash_before/cash_after are left NULL — backtest_portfolio.py
        has no per-trade (only end-of-day) cash ledger, and fabricating a
        number would overstate this data's actual precision.
    run_id : v2.8, optional — threaded into every log_entry() call so all
        trades from this ingestion share one metadata row (see
        TradeTracker.log_run_metadata(), called separately by the caller —
        this function only consumes an already-created run_id, it never
        creates the metadata row itself, keeping ingestion a pure read of
        already-finalized results).
    daily_position_log : v2.8, optional — {code, date, close, high, low,
        atr, regime, confidence, entry_date, entry_price, shares} rows
        collected during simulate_from_prepared()'s day loop (one per open
        position per day — purely additive analytics bookkeeping, see that
        function's docstring). When supplied, replayed into
        update_position_metrics() calls BEFORE the open/close pairs below
        are ingested — trade_daily needs to already be populated by the time
        log_exit() runs its internal MAX(mfe)/MIN(mae) aggregation, and
        update_position_metrics() never requires the matching `trades` row
        to exist first, so this ordering is safe.

    Regime source: prefers each trade_log row's own regime_at_entry/
    regime_at_exit (walk-forward, no-look-ahead HMM decode — set only when
    the caller passed regime_lookup= into simulate_from_prepared(), see its
    docstring) over recomputing via the rule-based build_regime_ctx() from
    raw OHLCV. Falls back to the rule-based path when those fields are
    absent/NaN — the common case, since regime_lookup is opt-in.

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

    def _regime_ctx_rule_fallback(code, date) -> Optional[dict]:
        df = all_data.get(code)
        if df is None:
            return None
        sliced = df.loc[:pd.Timestamp(date)]
        ctx = build_regime_ctx(sliced, mrd_version)
        if ctx is not None:
            ctx["source"] = "RULE"
        return ctx

    def _regime_ctx_for(row_dict, suffix, code, date) -> Optional[dict]:
        """suffix is 'entry' or 'exit' — reads regime_at_{suffix}/
        confidence_at_{suffix}/duration_at_{suffix} off row_dict (a
        trade_log row) if present, else falls back to the rule-based
        classifier at `date`."""
        ctx = _hmm_regime_ctx_from_label(
            row_dict.get(f"regime_at_{suffix}"),
            row_dict.get(f"confidence_at_{suffix}"),
            row_dict.get(f"duration_at_{suffix}"),
        )
        return ctx if ctx is not None else _regime_ctx_rule_fallback(code, date)

    # Replay daily_position_log into trade_daily BEFORE the log_exit() calls
    # below, so their internal MAX(mfe)/MIN(mae) aggregation (which reads
    # trade_daily at the moment log_exit runs) actually has rows to
    # aggregate. update_position_metrics() never requires the `trades` row
    # to exist first (see its docstring), so this ordering is safe even
    # though the matching log_entry() call hasn't happened yet.
    if daily_position_log:
        for r in daily_position_log:
            entry_time = pd.Timestamp(r["entry_date"]).isoformat()
            trade_id = f"{r['code']}_{entry_time}"
            regime_ctx = _hmm_regime_ctx_from_label(
                r.get("regime"), r.get("confidence"))
            try:
                tracker.update_position_metrics(
                    trade_id=trade_id, date=str(pd.Timestamp(r["date"]).date()),
                    close=r["close"], entry_price=r["entry_price"],
                    shares=r["shares"], direction="LONG",
                    high=r.get("high"), low=r.get("low"), atr=r.get("atr"),
                    regime_ctx=regime_ctx,
                )
            except Exception:
                continue   # analytics-only replay — never abort the ingestion

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
            regime_ctx=_regime_ctx_for(entry_row, "entry", code, entry_row["date"]),
            mrd_version=mrd_version, run_id=run_id,
        )
        tracker.log_exit(
            trade_id=trade_id, price=row["price"], cash=None,
            equity=_equity_at(row["date"]), timestamp=exit_time,
            exit_reason=row.get("reason", ""),
            regime_ctx=_regime_ctx_for(row, "exit", code, row["date"]),
            commission=row.get("fee"),
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
            regime_ctx=_regime_ctx_for(entry_row, "entry", code, entry_row["date"]),
            mrd_version=mrd_version, run_id=run_id,
        )
        count += 1

    return count
