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
