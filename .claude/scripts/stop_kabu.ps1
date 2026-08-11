# stop_kabu.ps1 - stop the main.py monitoring loop (leaves OpenD gateway running).
# Companion to start_kabu.ps1. Safe to re-run: no-ops if nothing is running.

$root     = 'c:\Kabu\kabu'
$lockPath = Join-Path $root '.kabu_loop.lock'

function Test-KabuPidAlive([int]$procId) {
    $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
    return ($null -ne $p -and $p.ProcessName -match '^python')
}

if (-not (Test-Path $lockPath)) {
    Write-Output "Not running (no lock file)."
    exit 0
}

$lockPidRaw = Get-Content $lockPath -ErrorAction SilentlyContinue
if (-not $lockPidRaw -or -not (Test-KabuPidAlive([int]$lockPidRaw))) {
    Write-Output "Not running (stale lock file, no live process)."
    Remove-Item -Path $lockPath -ErrorAction SilentlyContinue
    exit 0
}

$targetPid = [int]$lockPidRaw
Write-Output "Stopping main.py (pid=$targetPid)..."
Stop-Process -Id $targetPid -Force
Start-Sleep -Seconds 2

if (Test-KabuPidAlive($targetPid)) {
    Write-Output "WARNING: process $targetPid still alive after Stop-Process"
    exit 1
} else {
    Remove-Item -Path $lockPath -ErrorAction SilentlyContinue
    Write-Output "Stopped (pid=$targetPid)."
}
