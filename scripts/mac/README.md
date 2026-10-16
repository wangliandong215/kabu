# kabu on macOS

macOS counterparts of the Windows production setup (`.claude/scripts/*.ps1` +
the watchdog scheduled task).

## One-time setup

1. **Runtime data**: copy the Windows `C:\KabuData` folder to `~/KabuData`
   (positions, `trade_history.db`, `secrets.json`, HMM models, logs...).
   `config.DATA_ROOT` points there automatically on macOS; set
   `KABU_DATA_DIR` to use a different folder.
2. **Python 3.12 venv** (system Python 3.9 is too old):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ~/.local/bin/uv venv --python 3.12 .venv
   ~/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt
   ```
3. **OpenD**: install `/Applications/moomoo_OpenD.app`, log in once in its
   window and enable auto-login (watchdog.py restarts it unattended).
   Override the location with `OPEND_EXE_PATH`.

## Daily operation

```bash
scripts/mac/start_kabu.sh              # start OpenD (if needed) + main.py loop
scripts/mac/stop_kabu.sh               # stop main.py (OpenD keeps running)
scripts/mac/watchdog_agent.sh install  # run watchdog.py every 5 min via launchd
scripts/mac/watchdog_agent.sh status
scripts/mac/watchdog_agent.sh uninstall
```

Logs: `kabu_console.log` / `kabu_console.err.log` (main.py stdout/stderr),
`kabu_watchdog.log` (watchdog runs), `~/KabuData/logs/kabu.log` (alert log).

Pause phone notifications: `touch ~/KabuData/notify_paused.flag`
(resume: `rm` it).

Keep the Mac awake while trading (System Settings → Battery/Energy →
prevent automatic sleeping), otherwise the loop and watchdog pause during
sleep and the watchdog will report a stale heartbeat on wake.

## Tests

```bash
.venv/bin/python -m pytest -q -o python_files='test_*.py' --ignore=.claude
```
A few `engine/test_runner_*` tests reach a live OpenD and fail when it is not
logged in.
