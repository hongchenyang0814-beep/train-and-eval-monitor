# Unified Evaluation / Training Monitor

## Commands

Set the lifecycle script path:

```bash
MONITOR_TOOL="$HOME/.codex/skills/streamlake-experiment-analyst/scripts/eval_monitor.py"
```

Deploy or upgrade the bundled app, create the local cache directory, and start a detached loopback-only server. The same server exposes `/` (redirecting to `/train/`) for the local training monitor and `/eval/` for evaluation-log analysis:

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

Re-run `deploy` after updating the skill. It replaces only managed application files and `monitor_config.json`; it does not delete the experiment cache or credentials. The training monitor runtime config is `<target-dir>/training_monitor_config.json`; it is created on first deploy and then preserved across upgrades so user-specific output roots, explicit targets, and optional Hugging Face upload settings are not overwritten.

Every installation must verify paths for its own machine. Pass one or more existing absolute training roots, for example `--training-output-root /absolute/path/to/training/output`; repeat the option for multiple roots. A new deployment also checks supported training-root environment variables and a small set of current-user-relative candidates. Never copy another machine's absolute paths, credentials, PID, cache, or service log. Review `outputs_roots` and every `targets.*` path after deployment and after an upgrade.

When a runtime config came from another computer, repair it before starting the service:

```bash
python "$MONITOR_TOOL" repair-paths --target-dir "$TARGET_DIR"
```

The command keeps only existing paths on the current machine and removes stale explicit targets. To combine repair with deployment, add `--repair-paths` to `deploy`.

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

Keep that terminal running, then open `http://127.0.0.1:18280/` in the local browser for training or `http://127.0.0.1:18280/eval/` for evaluation logs. Closing the SSH command closes the tunnel but does not stop the remote Monitor.

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

The `IdentityFile` line is optional. If SSH normally uses port 22, the `Port` line and command-line `-p` option may be omitted. Replace both remote Monitor port occurrences when deploying the Monitor on another port. If local port `18280` is occupied, keep the remote port and choose another local port, for example `ssh -N -L 18281:127.0.0.1:18280 USER@SERVER`, then open `http://127.0.0.1:18281/` for training or `http://127.0.0.1:18281/eval/` for evaluation.

Windows users who have the generated helper script available on their local computer may run:

```powershell
powershell -ExecutionPolicy Bypass -File .\open_monitor_windows.ps1 -SshTarget USER@SERVER
```

Pass `-SshPort PORT` only when the SSH port is not already defined in `~/.ssh/config`.

## Behavior

- The left list and sample detail panes scroll independently.
- The top workspace switcher opens the training page (`/`, redirecting to `/train/`) and evaluation-log page (`/eval/`) on the same loopback server and SSH tunnel.
- Completed evaluations expose platform scores, 11 tasks, parsed samples, Think/No-think outputs, filtered logs, and comparisons. Task rows show displayed samples, generation requests, and generation time; the sample browser groups its 11 tasks under 懂物料, 懂用户, 懂推荐, and 懂世界.
- The ranking tab sorts all synchronized evaluations by total score, family scores, sample count, and cached local automatic metrics such as recommendation copy-answer rate, Think/No-think SID overlap, action hallucination rate, repeated SID rate, and loop output count. Risk-style metrics default to ascending order.
- The training page reuses the local training Monitor: it reads `trainer_log.jsonl`, GPU state from `nvidia-smi`, explicit PIDs, checkpoint directories, run manifests, recent logs, and optional configured Hugging Face upload targets. The settings dialog only stores or replaces the Hugging Face Write token; existing upload targets and repository behavior remain controlled by the runtime training configuration. It does not stop training or delete files.
- The training settings dialog only binds or replaces the Hugging Face Write token. It stores the token separately under `<target-dir>/config/huggingface_token` with mode `600`; endpoint, namespace, repository prefix, privacy, index repository, base model, and explicit target settings remain in the runtime training configuration. Automatic uploads use exact category profiles: `<output-root>/<category>/<run-id>/` and `<output-root>/<category>/<subcategory>/<run-id>/` are supported when the matching profile exists and the run includes run-local `trainer_log.jsonl` and `training_config.yaml`. Other runs remain read-only. After binding, a selected eligible checkpoint follows the original flow to create its destination model repository and upload the staged evaluation package.
- The Log tab downloads the selected evaluation's complete original log to the browser device. The server streams the file without a size limit, and browser settings determine its final local path.
- Multi-output samples show one output body per mode. Selectors use 16 columns in a wide pane and 8 columns below the container threshold.
- Running evaluations remain visible without requesting output until `hasOutput=true`.
- The Sync button invokes the bundled `streamlake_experiments.py sync`, including the 2026-08-01 cutoff and complete-log download rules.
- Each successful sync starts background analysis only for new or changed logs. Full parsed results and lightweight ranking summaries are stored under `<output-dir>/analysis_cache`; the cache is reused after service restarts and invalidated only when the log size/mtime or parser version changes.
- Opening or polling the page does not start batch analysis, and the interface does not display analysis-cache progress. Opening an uncached evaluation still analyzes and persists that individual log on demand.
- AI-operated training runs must create `run_manifest.json`, run `scripts/experiment_manifest.py validate`, and render `training_task.md` before starting. The required fields are documented in the root README; a failed validation is a hard stop. Evaluation detail can bind only to a discovered training run with a valid manifest and displays its sanitized manifest/document.
- The evaluation toolbar places manual log upload before Sync and uses a distinct color for it. The parser rejects files without recognizable task, sample, or evaluation metadata. Accepted originals are permanent under `<output-dir>/logs/<eval-task-id>/evaluation.log`, the same path used by synchronization, with `evaluation_note.md` and analysis cache; list rows carry the `自主上传` badge and optional training binding.

## Troubleshooting

- `configured: false`: paste fresh complete request headers in the settings dialog.
- `401`, `403`, or login redirect: replace the expired Cookie through the settings dialog.
- `running: false`: inspect `<target-dir>/monitor_server.log`, then run `start`.
- Port conflict: re-run `deploy --port ANOTHER_PORT` and use the same remote port in the SSH tunnel.
- A running evaluation has no log: this is expected until a later sync observes `hasOutput=true`.
- No training runs shown: edit `<target-dir>/training_monitor_config.json` and set `outputs_roots` or explicit `targets`, then restart the Monitor.
