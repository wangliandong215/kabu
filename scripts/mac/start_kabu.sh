#!/bin/bash
# start_kabu.sh - idempotent kabu startup on macOS: OpenD gateway + main.py
# monitoring loop. macOS counterpart of .claude/scripts/start_kabu.ps1; mirrors
# watchdog.py's _restart_opend()/_restart_main() production sequence.
# Safe to re-run: no-ops on whichever half (OpenD / main.py) is already alive.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
OPEND_APP="${OPEND_EXE_PATH:-/Applications/moomoo_OpenD.app}"
OPEND_PORT="${FUTU_PORT:-11111}"
LOCK_PATH="$ROOT/.kabu_loop.lock"
HB_PATH="$ROOT/.kabu_heartbeat"
CONSOLE_LOG="$ROOT/kabu_console.log"
CONSOLE_ERR="$ROOT/kabu_console.err.log"

opend_port_open() {
    nc -z -G 2 127.0.0.1 "$OPEND_PORT" >/dev/null 2>&1
}

kabu_pid_alive() {
    local comm
    comm="$(ps -p "$1" -o comm= 2>/dev/null)" || return 1
    [[ "$(basename "$comm" | tr '[:upper:]' '[:lower:]')" == python* ]]
}

# 1. OpenD gateway
if ! pgrep -x moomoo_OpenD >/dev/null || ! opend_port_open; then
    echo "Starting OpenD..."
    open -a "$OPEND_APP"
    for _ in $(seq 20); do
        opend_port_open && break
        sleep 3
    done
    if ! opend_port_open; then
        echo "ERROR: OpenD port $OPEND_PORT did not come up within 60s"
        echo "       (first run: log in once in the OpenD window and enable auto-login)"
        exit 1
    fi
    echo "OpenD is up."
else
    echo "OpenD already running."
fi

# 2. main.py monitoring loop
if [[ -f "$LOCK_PATH" ]] && kabu_pid_alive "$(cat "$LOCK_PATH")"; then
    echo "main.py already running (pid=$(cat "$LOCK_PATH"))."
    exit 0
fi

rm -f "$LOCK_PATH" "$HB_PATH"
echo "Starting main.py --auto --confirmed --interval 300 ..."
cd "$ROOT" || exit 1
nohup "$PYTHON" main.py --auto --confirmed --interval 300 \
    >"$CONSOLE_LOG" 2>"$CONSOLE_ERR" </dev/null &
disown
sleep 5
if [[ -f "$LOCK_PATH" ]]; then
    echo "main.py started, pid=$(cat "$LOCK_PATH")"
else
    echo "WARNING: lock file not found after launch attempt - check $CONSOLE_ERR"
fi
