# Evaluation Monitor

## Commands

Set the lifecycle script path:

```bash
MONITOR_TOOL="$HOME/.codex/skills/streamlake-experiment-analyst/scripts/eval_monitor.py"
```

Deploy or upgrade the bundled app, create the local cache directory, and start a detached loopback-only server:

```bash
python "$MONITOR_TOOL" deploy
```

Use explicit paths or a different port when needed:

```bash
python "$MONITOR_TOOL" deploy \
  --target-dir "$HOME/.local/share/streamlake-eval-monitor" \
  --output-dir "$HOME/.local/share/streamlake-eval-monitor/data" \
  --port 18280
```

Manage an existing deployment:

```bash
python "$MONITOR_TOOL" status
python "$MONITOR_TOOL" stop
python "$MONITOR_TOOL" start
```

Re-run `deploy` after updating the skill. It replaces only managed application files and `monitor_config.json`; it does not delete the experiment cache or credentials.

## First configuration

Open the page and click the settings button. The user should:

1. Open the StreamLake evaluation task list in a logged-in browser.
2. Open developer tools, select Network, and refresh.
3. Find `GetIamProjectList` or another StreamLake request whose Referer contains `/wanqing/proj-.../`.
4. Copy the complete request headers, including the request line, `Cookie`, and `Referer`.
5. Paste the headers into the monitor and save.

The server extracts the Cookie and project ID, writes them under `<target-dir>/config`, and never echoes the Cookie. Saved configuration remains active until replaced or deleted. Synchronized data, logs, exports, and analysis caches default to `<target-dir>/data`; the PID, service log, and deployed app also stay inside `<target-dir>`.

## SSH access

The server intentionally binds only to `127.0.0.1`. First start it on the remote server:

```bash
python "$MONITOR_TOOL" deploy
```

Then open a new terminal on the computer where the browser runs. Do not run the following tunnel inside the remote SSH shell:

```bash
ssh -N -L 18280:127.0.0.1:18280 USER@SERVER
```

Replace `USER` with the remote login name and `SERVER` with the server hostname or IP. For a non-default SSH port:

```bash
ssh -p SSH_PORT -N -L 18280:127.0.0.1:18280 USER@SERVER_IP
```

Keep that terminal running, then open `http://127.0.0.1:18280/` in the local browser. Closing the SSH command closes the tunnel but does not stop the remote Monitor.

A personal SSH alias works only on a computer whose SSH config already defines it. For example, Windows uses `%USERPROFILE%\.ssh\config`, while Linux and macOS use `~/.ssh/config`:

```sshconfig
Host SSH_ALIAS
    HostName SERVER_IP_OR_HOSTNAME
    User SSH_USER
    Port SSH_PORT
    IdentityFile ~/.ssh/id_ed25519
```

After configuring the alias, the local command may be shortened to:

```bash
ssh -N -L 18280:127.0.0.1:18280 SSH_ALIAS
```

The `IdentityFile` line is optional. If SSH normally uses port 22, the `Port` line and command-line `-p` option may be omitted. Replace both remote Monitor port occurrences when deploying the Monitor on another port. If local port `18280` is occupied, keep the remote port and choose another local port, for example `ssh -N -L 18281:127.0.0.1:18280 USER@SERVER`, then open `http://127.0.0.1:18281/`.

Windows users who have the generated helper script available on their local computer may run:

```powershell
powershell -ExecutionPolicy Bypass -File .\open_monitor_windows.ps1 -SshTarget USER@SERVER
```

Pass `-SshPort PORT` only when the SSH port is not already defined in `~/.ssh/config`.

## Behavior

- The left list and sample detail panes scroll independently.
- Completed evaluations expose platform scores, 11 tasks, parsed samples, Think/No-think outputs, filtered logs, and comparisons.
- Multi-output samples show one output body per mode. Selectors use 16 columns in a wide pane and 8 columns below the container threshold.
- Running evaluations remain visible without requesting output until `hasOutput=true`.
- The Sync button invokes the bundled `streamlake_experiments.py sync`, including the 2026-08-01 cutoff and complete-log download rules.
- Each successful sync starts background analysis only for new or changed logs. Full parsed results are stored as compressed files under `<output-dir>/analysis_cache`; the cache is reused after service restarts and invalidated only when the log size/mtime or parser version changes.
- Opening or polling the page does not start batch analysis, and the interface does not display analysis-cache progress. Opening an uncached evaluation still analyzes and persists that individual log on demand.

## Troubleshooting

- `configured: false`: paste fresh complete request headers in the settings dialog.
- `401`, `403`, or login redirect: replace the expired Cookie through the settings dialog.
- `running: false`: inspect `<target-dir>/monitor_server.log`, then run `start`.
- Port conflict: re-run `deploy --port ANOTHER_PORT` and use the same remote port in the SSH tunnel.
- A running evaluation has no log: this is expected until a later sync observes `hasOutput=true`.
