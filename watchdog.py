"""
watchdog.py — external liveness check for the main.py monitoring loop.

Run periodically (e.g. every 5-10 minutes via a Windows scheduled task)
alongside main.py. Detects two failure modes main.py cannot self-report,
because in both cases the process that would normally call alert.error()
is either frozen or gone:

  1. Hung: the process from .kabu_loop.lock is still running, but
     .kabu_heartbeat (written at the top of every engine/runner.py
     run_loop() iteration) hasn't advanced in longer than a normal
     scan+sleep cycle ever takes — e.g. stuck forever inside a moomoo API
     call with no timeout (this happened 2026-07-10 23:57 JST).
  2. Dead: the .kabu_loop.lock PID is no longer running at all (crashed,
     force-killed, or the machine rebooted without main.py being
     relaunched).

Sends one DingTalk/Telegram alert per incident (not once per poll) via a
dedup marker file, mirroring engine/market_hours.py's should_notify_close()
one-shot pattern — clears automatically once the loop is healthy again so a
future incident re-alerts instead of staying silently suppressed forever.
"""
import ctypes
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import notify.alert as alert

_ROOT = Path(__file__).resolve().parent
_HEARTBEAT_PATH = _ROOT / ".kabu_heartbeat"
_LOCK_PATH = _ROOT / ".kabu_loop.lock"
_ALERTED_MARKER = _ROOT / ".kabu_watchdog_alerted"

# Normal cycle observed in production is ~420-425s (scan time + 300s sleep);
# 15 minutes gives a wide margin before crying wolf on a slow-but-fine pass.
_STALE_THRESHOLD = timedelta(minutes=15)


def _is_pid_running(pid: int) -> bool:
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle:
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    return False


def main() -> None:
    if not _LOCK_PATH.exists():
        return   # main.py has never been started — nothing to watch

    try:
        pid = int(_LOCK_PATH.read_text().strip())
    except (ValueError, OSError):
        return

    if not _HEARTBEAT_PATH.exists():
        return   # started but hasn't completed its first iteration yet

    try:
        last_beat = datetime.fromisoformat(_HEARTBEAT_PATH.read_text().strip())
    except (ValueError, OSError):
        return

    age = datetime.now() - last_beat
    if age <= _STALE_THRESHOLD:
        if _ALERTED_MARKER.exists():
            _ALERTED_MARKER.unlink()
        return

    if _ALERTED_MARKER.exists():
        return   # already alerted for this ongoing incident

    if _is_pid_running(pid):
        alert.push_raw(
            f"监控进程(pid={pid})还活着，但已经"
            f"{int(age.total_seconds() // 60)}分钟没有新的scan pass了"
            f" —— 大概率卡死了，需要手动重启"
        )
    else:
        alert.push_raw(
            f"监控进程(pid={pid})已经不在运行了"
            f"（崩溃/被杀/电脑重启），最后一次活动是"
            f"{last_beat.isoformat(timespec='seconds')} —— 需要重新启动"
        )
    _ALERTED_MARKER.write_text(datetime.now().isoformat())


if __name__ == "__main__":
    main()
