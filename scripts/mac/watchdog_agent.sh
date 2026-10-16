#!/bin/bash
# watchdog_agent.sh - install/uninstall/status of the launchd agent that runs
# watchdog.py every 5 minutes (macOS counterpart of the Windows scheduled task).
#
#   scripts/mac/watchdog_agent.sh install
#   scripts/mac/watchdog_agent.sh uninstall
#   scripts/mac/watchdog_agent.sh status

LABEL="com.kabu.watchdog"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$ROOT/kabu_watchdog.log"
DOMAIN="gui/$(id -u)"

case "$1" in
install)
    mkdir -p "$HOME/Library/LaunchAgents"
    cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$ROOT/.venv/bin/python</string>
        <string>$ROOT/watchdog.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$ROOT</string>
    <key>StartInterval</key>
    <integer>300</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG</string>
    <key>StandardErrorPath</key>
    <string>$LOG</string>
</dict>
</plist>
EOF
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
    launchctl bootstrap "$DOMAIN" "$PLIST" && echo "Installed $LABEL (every 300s), log: $LOG"
    ;;
uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
    rm -f "$PLIST"
    echo "Uninstalled $LABEL"
    ;;
status)
    launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E 'state|last exit|run interval' || echo "$LABEL not loaded"
    ;;
*)
    echo "usage: $0 install|uninstall|status"
    exit 1
    ;;
esac
