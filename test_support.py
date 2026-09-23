"""
test_support.py -- shared test-only helper.

notify/alert.py's _emit() unconditionally writes every log()/warn()/error()/
info() call to THREE places: stdout, a RotatingFileHandler pointed at the
real production C:\\KabuData\\logs\\kabu.log, and DingTalk/Telegram (already
gated by config.NOTIFY_MUTE_FILE). None of the engine/test_runner_*.py-style
suites mock alert.log/warn/error/info (only alert.trade_sell/trade_buy),
because those are legitimate log lines from real code paths under test --
but that means every test run was writing realistic-looking "再平衡卖出..."
etc. lines into the real production log file, discovered 2026-08-31 while
auditing a real EMERGENCY rebalance execution (the review had to distinguish
genuine log lines from test-run noise by cross-checking order IDs/prices
against the actual broker).

mute_alert_file_logging()/unmute_alert_file_logging() detach just the file
handler for the duration of a test -- stdout printing and the (already
config-gated) DingTalk/Telegram push are untouched, since neither is the
production audit trail this exists to protect and tests already rely on/
tolerate console noise.
"""
import notify.alert as alert


def mute_alert_file_logging() -> list:
    """Detach notify.alert._logger's handlers (the RotatingFileHandler onto
    kabu.log) and return them so unmute_alert_file_logging() can restore
    them. Call from setUp()."""
    handlers = list(alert._logger.handlers)
    alert._logger.handlers = []
    return handlers


def unmute_alert_file_logging(saved_handlers: list) -> None:
    """Restore handlers saved by mute_alert_file_logging(). Call from
    tearDown() with the list that setUp() got back."""
    alert._logger.handlers = saved_handlers


# ── Live state/log isolation ──────────────────────────────────────────────────
# Every module-level C:\KabuData path that run_once() and friends read or
# write during a pass. Discovered 2026-09-23: the engine/test_runner_*.py-style
# suites only stubbed a few of these (log_snapshot/log_qqq_snapshot), so each
# test run appended fake rows (total_assets=1,000,000, long_mv=0) to the real
# risk_state_log.jsonl, left fake codes (US.TEST/US.WEAK) in the real
# reconciliation_last_diff.json that live diff-growth alerts compare against,
# and overwrote the real qqq_recovery_state.json — and read the real state
# files back, so results also depended on live data.
_LIVE_PATH_ATTRS = [
    ("risk.portfolio_risk_manager", "_RISK_LOG_PATH"),
    ("risk.portfolio_risk_manager", "_DIFF_LOG_PATH"),
    ("risk.portfolio_risk_manager", "_LAST_DIFF_PATH"),
    ("risk.portfolio_risk_manager", "_LOCK_PATH"),
    ("risk.portfolio_risk_manager", "_QQQ_RISK_LOG_PATH"),
    ("risk.portfolio_risk_manager", "_QQQ_TRIM_PREVIEW_PATH"),
    ("risk.qqq_core_recovery", "_STATE_PATH"),
    ("risk.qqq_core_recovery", "_RECOVERY_LOG_PATH"),
    ("risk.qqq_core_recovery", "_LEVEL3_LOG_PATH"),
    ("risk.portfolio_position_manager", "_PM_LOG_PATH"),
    ("risk.portfolio_risk_engine", "_LOG_PATH"),
    ("ai_decision.audit_log", "_LOG_PATH"),
    ("exit_engine", "EXIT_ENGINE_LOG_PATH"),
    ("exit_engine.state_store", "_STORE_PATH"),
    ("position_manager", "POSITION_MANAGER_LOG_PATH"),
    ("position_manager.state_store", "_STORE_PATH"),
    ("regime", "REGIME_LOG_PATH"),
]

_isolation_stack: list = []


def isolate_live_state() -> None:
    """Point every _LIVE_PATH_ATTRS path at a fresh temp dir. Call from a
    test module's setUpModule() and pair with restore_live_state() in
    tearDownModule(). Tests that further patch one of these attrs (saving
    and restoring around it) keep working — they just restore to the temp
    path instead of the live one."""
    import importlib
    import tempfile
    from pathlib import Path

    tmp = tempfile.TemporaryDirectory(prefix="kabu_test_")
    saved = []
    for mod_name, attr in _LIVE_PATH_ATTRS:
        mod = importlib.import_module(mod_name)
        orig = getattr(mod, attr)
        saved.append((mod, attr, orig))
        setattr(mod, attr, Path(tmp.name) / mod_name.replace(".", "_") / Path(orig).name)
    _isolation_stack.append((tmp, saved))


def restore_live_state() -> None:
    """Undo the most recent isolate_live_state()."""
    tmp, saved = _isolation_stack.pop()
    for mod, attr, orig in saved:
        setattr(mod, attr, orig)
    tmp.cleanup()
