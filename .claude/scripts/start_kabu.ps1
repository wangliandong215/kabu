# start_kabu.ps1 - idempotent kabu startup: OpenD gateway + main.py monitoring loop.
# Mirrors watchdog.py's _restart_opend()/_restart_main() production sequence.
# Safe to re-run: no-ops on whichever half (OpenD / main.py) is already alive.

$root       = 'c:\Kabu\kabu'
$pythonExe  = 'C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe'
$opendExe   = 'C:\Users\Administrator\AppData\Roaming\moomoo_OpenD\moomoo_OpenD.exe'
$lockPath   = Join-Path $root '.kabu_loop.lock'
$hbPath     = Join-Path $root '.kabu_heartbeat'
$consoleLog = Join-Path $root 'kabu_console.log'
$consoleErr = Join-Path $root 'kabu_console.err.log'

function Test-OpenDPort {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $c.Connect('127.0.0.1', 11111)
        $c.Close()
        return $true
    } catch {
        return $false
    }
}

function Test-KabuPidAlive([int]$procId) {
    $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
    return ($null -ne $p -and $p.ProcessName -match '^python')
}

# 1. OpenD gateway
$opendProc = Get-Process moomoo_OpenD -ErrorAction SilentlyContinue
if (-not $opendProc -or -not (Test-OpenDPort)) {
    Write-Output "Starting OpenD..."
    Start-Process -FilePath $opendExe -WorkingDirectory (Split-Path $opendExe)
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline -and -not (Test-OpenDPort)) {
        Start-Sleep -Seconds 3
    }
    if (-not (Test-OpenDPort)) {
        Write-Output "ERROR: OpenD port 11111 did not come up within 60s"
        exit 1
    }
    Write-Output "OpenD is up."
} else {
    Write-Output "OpenD already running."
}

# 2. main.py monitoring loop
$mainAlreadyRunning = $false
if (Test-Path $lockPath) {
    $lockPidRaw = Get-Content $lockPath -ErrorAction SilentlyContinue
    if ($lockPidRaw -and (Test-KabuPidAlive([int]$lockPidRaw))) {
        $mainAlreadyRunning = $true
        Write-Output "main.py already running (pid=$lockPidRaw)."
    }
}

if (-not $mainAlreadyRunning) {
    Remove-Item -Path $lockPath -ErrorAction SilentlyContinue
    Remove-Item -Path $hbPath -ErrorAction SilentlyContinue
    Write-Output "Starting main.py --auto --confirmed --interval 300 ..."
    Start-Process -FilePath $pythonExe `
        -ArgumentList 'main.py', '--auto', '--confirmed', '--interval', '300' `
        -WorkingDirectory $root `
        -RedirectStandardOutput $consoleLog `
        -RedirectStandardError $consoleErr `
        -WindowStyle Hidden
    Start-Sleep -Seconds 5
    if (Test-Path $lockPath) {
        $newPid = Get-Content $lockPath
        Write-Output "main.py started, pid=$newPid"
    } else {
        Write-Output "WARNING: lock file not found after launch attempt - check $consoleErr"
    }
}
