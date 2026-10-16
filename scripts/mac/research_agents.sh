#!/bin/bash
# research_agents.sh - install/uninstall/status of the launchd agents that run
# the two daily research scans (macOS counterpart of the Windows scheduled
# tasks "Kabu Research Scan" / "Kabu Market Features Scan"). Both scripts are
# research-only: they never import runner/portfolio/risk and never place orders.
#
# launchd fires on the Mac's local clock (JST). 08:00 / 08:15 JST is after the
# US close all year: 19:00/19:15 ET during US daylight time, 18:00/18:15 ET
# after it ends — so no seasonal adjustment is needed.
#
#   scripts/mac/research_agents.sh install
#   scripts/mac/research_agents.sh uninstall
#   scripts/mac/research_agents.sh status
#   scripts/mac/research_agents.sh run-now    # trigger both once, right now

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOMAIN="gui/$(id -u)"

# label|python module|hour|minute|log file
AGENTS=(
    "com.kabu.research-scan|research.scan_mean_reversion_candidates|8|0|$ROOT/kabu_research_scan.log"
    "com.kabu.market-features-scan|research.collect_market_features|8|15|$ROOT/kabu_market_features_scan.log"
)

install_agent() {
    local label="$1" module="$2" hour="$3" minute="$4" log="$5"
    local plist="$HOME/Library/LaunchAgents/$label.plist"
    cat >"$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$label</string>
    <key>ProgramArguments</key>
    <array>
        <string>$ROOT/.venv/bin/python</string>
        <string>-m</string>
        <string>$module</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$ROOT</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>$hour</integer>
        <key>Minute</key>
        <integer>$minute</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$log</string>
    <key>StandardErrorPath</key>
    <string>$log</string>
</dict>
</plist>
EOF
    launchctl bootout "$DOMAIN/$label" 2>/dev/null
    launchctl bootstrap "$DOMAIN" "$plist" &&
        printf "Installed %s (daily %02d:%02d local), log: %s\n" "$label" "$hour" "$minute" "$log"
}

case "$1" in
install)
    mkdir -p "$HOME/Library/LaunchAgents"
    for a in "${AGENTS[@]}"; do
        IFS='|' read -r label module hour minute log <<<"$a"
        install_agent "$label" "$module" "$hour" "$minute" "$log"
    done
    ;;
uninstall)
    for a in "${AGENTS[@]}"; do
        IFS='|' read -r label _ <<<"$a"
        launchctl bootout "$DOMAIN/$label" 2>/dev/null
        rm -f "$HOME/Library/LaunchAgents/$label.plist"
        echo "Uninstalled $label"
    done
    ;;
status)
    for a in "${AGENTS[@]}"; do
        IFS='|' read -r label _ <<<"$a"
        echo "== $label"
        launchctl print "$DOMAIN/$label" 2>/dev/null | grep -E 'state =|last exit|runs =' || echo "not loaded"
    done
    ;;
run-now)
    for a in "${AGENTS[@]}"; do
        IFS='|' read -r label _ <<<"$a"
        launchctl kickstart "$DOMAIN/$label" && echo "Started $label"
    done
    ;;
*)
    echo "usage: $0 install|uninstall|status|run-now"
    exit 1
    ;;
esac
