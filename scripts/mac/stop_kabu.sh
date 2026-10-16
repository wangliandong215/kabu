#!/bin/bash
# stop_kabu.sh - stop the main.py monitoring loop (leaves OpenD gateway running).
# macOS counterpart of .claude/scripts/stop_kabu.ps1. Safe to re-run.
#
# Note: the watchdog launchd agent relaunches main.py ~15 min after it stops
# heartbeating. To keep it stopped, also run: scripts/mac/watchdog_agent.sh uninstall

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCK_PATH="$ROOT/.kabu_loop.lock"

kabu_pid_alive() {
    local comm
    comm="$(ps -p "$1" -o comm= 2>/dev/null)" || return 1
    [[ "$(basename "$comm" | tr '[:upper:]' '[:lower:]')" == python* ]]
}

if [[ ! -f "$LOCK_PATH" ]]; then
    echo "Not running (no lock file)."
    exit 0
fi

TARGET_PID="$(cat "$LOCK_PATH")"
if ! kabu_pid_alive "$TARGET_PID"; then
    echo "Not running (stale lock file, no live process)."
    rm -f "$LOCK_PATH"
    exit 0
fi

echo "Stopping main.py (pid=$TARGET_PID)..."
kill "$TARGET_PID"   # SIGTERM -> main.py sends the "即将停止" alert, then exits
for _ in $(seq 10); do
    kabu_pid_alive "$TARGET_PID" || break
    sleep 1
done
if kabu_pid_alive "$TARGET_PID"; then
    kill -9 "$TARGET_PID"
    sleep 1
fi

if kabu_pid_alive "$TARGET_PID"; then
    echo "WARNING: process $TARGET_PID still alive after kill -9"
    exit 1
fi
rm -f "$LOCK_PATH"
echo "Stopped (pid=$TARGET_PID)."
