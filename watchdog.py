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
     call with no timeout (this happened 2026-07-10 23:57 JST, and again
     2026-07-22 when OpenD silently stopped listening on OPEND_PORT for
     ~7h while main.py kept retrying). Since a dead/unresponsive OpenD
     gateway is the observed cause each time, this case auto-restarts
     OpenD (see _restart_opend()) on every poll for as long as the hang
     persists — OpenD is configured for auto-login so no credentials
     need to be re-entered.
  2. Dead: the .kabu_loop.lock PID is no longer running at all (crashed,
     force-killed, or the machine rebooted without main.py being
     relaunched). Observed 2026-07-22 right at US market open: OpenD
     dropped for a few seconds and common._check_opend() (called when a
     fresh quote/trade context is created mid-loop) did sys.exit(1) on the
     refused connection — a SystemExit, which unlike a normal Exception
     is NOT caught by run_loop()'s `except Exception`, so main.py exited
     outright instead of logging and continuing. This case restarts
     OpenD, waits for its port to actually accept connections, then
     relaunches main.py itself (see _restart_main()) — relaunching main.py
     before OpenD's port is up just reproduces the same sys.exit(1).

Sends one DingTalk/Telegram alert per incident (not once per poll) via a
dedup marker file, mirroring engine/market_hours.py's should_notify_close()
one-shot pattern — clears automatically once the loop is healthy again so a
future incident re-alerts instead of staying silently suppressed forever.
The restart actions themselves are deliberately NOT deduped the same way:
they're retried every ~5min poll for as long as the failure persists,
since one restart isn't always enough (observed 2026-07-22: OpenD came
back, then died again ~18min later) and re-running them against an
already-healthy OpenD/main.py is a harmless ~30s blip.
"""
import ctypes
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import config
import notify.alert as alert

_ROOT = Path(__file__).resolve().parent
_HEARTBEAT_PATH = _ROOT / ".kabu_heartbeat"
_LOCK_PATH = _ROOT / ".kabu_loop.lock"
_ALERTED_MARKER = _ROOT / ".kabu_watchdog_alerted"
_OPEND_EXE = Path(config.OPEND_EXE_PATH)
_MAIN_PY = _ROOT / "main.py"
_MAIN_ARGS = ["--auto", "--confirmed", "--interval", "300"]
_CONSOLE_LOG = _ROOT / "kabu_console.log"
_CONSOLE_ERR = _ROOT / "kabu_console.err.log"

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


def _restart_opend() -> None:
    """Kill and relaunch the moomoo OpenD gateway. Best-effort: taskkill
    exits non-zero if OpenD wasn't running (e.g. it crashed and exited
    rather than hanging), which is fine — Popen still (re)launches it.

    Sends one push per call, not deduped like the diagnostic alert in
    main() — this is called on every ~5min poll for as long as a hang
    persists, so a long incident means one message per restart attempt
    (by design, per user request: they want to see each restart happen)."""
    subprocess.run(
        ["taskkill", "/IM", "moomoo_OpenD.exe", "/F"],
        capture_output=True,
    )
    time.sleep(2)
    subprocess.Popen(
        [str(_OPEND_EXE)],
        cwd=str(_OPEND_EXE.parent),
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    alert.push_raw(f"OpenD 网关已重启（{datetime.now().strftime('%H:%M:%S')}）")


def _opend_port_open() -> bool:
    try:
        with socket.create_connection((config.OPEND_HOST, config.OPEND_PORT), timeout=1.5):
            return True
    except OSError:
        return False


def _wait_for_opend(max_wait_s: int = 60) -> bool:
    """Poll OPEND_PORT until it accepts a connection, or give up after
    max_wait_s. See module docstring — relaunching main.py before this
    is true just re-triggers common._check_opend()'s sys.exit(1)."""
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        if _opend_port_open():
            return True
        time.sleep(3)
    return False


def _restart_main() -> None:
    """Relaunch main.py detached with the same flags used in production.
    Scheduled-task launches (unlike a long-lived interactive shell) pick
    up the current User-level env vars fresh, so KABU_TELEGRAM_BOT_TOKEN
    etc. come through without needing to be re-injected here."""
    with open(_CONSOLE_LOG, "w") as out, open(_CONSOLE_ERR, "w") as err:
        subprocess.Popen(
            [sys.executable, str(_MAIN_PY), *_MAIN_ARGS],
            cwd=str(_ROOT),
            stdout=out,
            stderr=err,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )


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

    if _is_pid_running(pid):
        _restart_opend()
        if not _ALERTED_MARKER.exists():
            alert.push_raw(
                f"监控进程(pid={pid})还活着，但已经"
                f"{int(age.total_seconds() // 60)}分钟没有新的scan pass了"
                f" —— 大概率是OpenD网关卡死/掉线，已自动重启OpenD"
                f"（如果重启后几分钟内仍未恢复，需要手动检查）"
            )
            _ALERTED_MARKER.write_text(datetime.now().isoformat())
        return

    # Dead: main.py itself is gone. Restart OpenD, wait for its port to
    # actually come up, then relaunch main.py — every poll, until healthy.
    _restart_opend()
    if _wait_for_opend():
        _restart_main()
        outcome = "已自动重启OpenD和main.py"
    else:
        outcome = "已重启OpenD，但60秒内端口仍未监听，main.py暂未重启——需要人工检查"

    if not _ALERTED_MARKER.exists():
        alert.push_raw(
            f"监控进程(pid={pid})已经不在运行了"
            f"（崩溃/被杀/电脑重启），最后一次活动是"
            f"{last_beat.isoformat(timespec='seconds')} —— {outcome}"
        )
        _ALERTED_MARKER.write_text(datetime.now().isoformat())


if __name__ == "__main__":
    main()
