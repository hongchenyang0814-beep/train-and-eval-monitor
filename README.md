# StreamLake Experiment Analyst

[![Tests](https://github.com/hongchenyang0814-beep/streamlake-experiment-analyst/actions/workflows/test.yml/badge.svg)](https://github.com/hongchenyang0814-beep/streamlake-experiment-analyst/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文说明](README.zh-CN.md)

A Codex skill and dependency-free Python CLI for synchronizing StreamLake Wanqing fine-tuning and competition-evaluation metadata, comparing experiments, and deploying an SSH-accessible evaluation-log Monitor. It explicitly downloads complete evaluation logs without downloading checkpoints, datasets, or prediction files.

> Stop opening experiments one by one and copying metrics by hand. StreamLake Experiment Analyst synchronizes Wanqing training and evaluation records, organizes metrics, parameters, and experiment relationships locally, and prepares context that Codex and other vibe-coding tools can analyze directly—all without downloading model weights, checkpoints, datasets, or prediction files.

## Features

- Read-only synchronization of fine-tuning and formal evaluation experiments.
- Full pagination with ID deduplication.
- Synchronization is hard-restricted to experiments created on or after `2026-08-01T00:00:00Z`; `sync --since ISO-8601-TIMESTAMP` may only narrow that range.
- Evaluation tasks marked `FAILED` with `hasOutput=false` are excluded before result retrieval.
- Complete evaluation logs can be downloaded by exact evaluation ID without a file-size limit or forwarding the StreamLake Cookie to the log CDN.
- Every normal `sync` automatically fills in complete logs for all successful evaluations under `<output-dir>/logs/<evalTaskId>/evaluation.log`; existing complete files are reused and re-hashed.
- Running or pending evaluations retain their metadata but do not request a log until `hasOutput=true`; a later sync downloads the log after successful completion.
- SQLite, sanitized JSON, CSV, catalog, and bounded LLM context outputs.
- Metric deltas, missing-value handling, parameter differences, and experiment-ID resolution.
- Credential redaction, signed-URL cleanup, large-body omission, endpoint allowlists, and same-host redirect enforcement.
- No third-party Python dependencies.
- One-command deployment of a loopback-only unified Monitor: `/` for evaluation-log analysis and `/train/` for the local LLaMA-Factory training monitor.
- The training tab reads local `trainer_log.jsonl`, GPU state, checkpoints, run manifests, and recent logs; optional Hugging Face checkpoint upload is configured only in the runtime `training_monitor_config.json`, not in the skill source.
- The evaluation tab supports platform configuration, synchronization, 11 structured tasks, Think/No-think samples, complete-log download, pairwise comparison, and global ranking by platform scores or cached local automatic metrics.
- Task binding falls back to task-family prefixes, so unfamiliar `challenge_itemic_*`, `challenge_evolution_*`, `challenge_recommendation_*`, and `challenge_common_sense` aliases still stay in the right family.
- Incremental background analysis after each successful sync, limited to new or changed logs, with compressed persistent results under `<output-dir>/analysis_cache`; page loads do not trigger batch analysis or display cache progress, and valid caches are reused across page loads and service restarts.

## Install as a Codex skill

Requirements: Git, Python 3.10 or later, and Codex.

### First-time installation

```bash
mkdir -p ~/.codex/skills
git clone https://github.com/hongchenyang0814-beep/streamlake-experiment-analyst.git \
  ~/.codex/skills/streamlake-experiment-analyst
```

### Verify the installation

These checks are local and do not contact StreamLake:

```bash
test -f ~/.codex/skills/streamlake-experiment-analyst/SKILL.md
python ~/.codex/skills/streamlake-experiment-analyst/scripts/streamlake_experiments.py --help
```

Restart Codex or begin a new task so the skill is rediscovered. You can then ask Codex to use `streamlake-experiment-analyst` to synchronize or compare experiments.

### Update an existing installation

```bash
git -C ~/.codex/skills/streamlake-experiment-analyst pull --ff-only
```

Restart Codex or begin a new task after updating. Authentication and project configuration are separate from installation; never place a Cookie in the clone or update command.

## Configure authentication

Prefer deploying the Monitor and pasting the complete request headers through its settings dialog. Cookie and project ID values are then saved automatically under the unified runtime directory. Never commit a Cookie or pass it as a command-line argument.

```bash
mkdir -p ~/.local/share/streamlake-eval-monitor/config
chmod 700 ~/.local/share/streamlake-eval-monitor/config
# Write the Cookie value to this file using your preferred secret-safe editor.
chmod 600 ~/.local/share/streamlake-eval-monitor/config/cookie
```

The CLI reads `~/.local/share/streamlake-eval-monitor/config/cookie` by default. Override the path with `STREAMLAKE_COOKIE_FILE`, or provide the value through `STREAMLAKE_COOKIE` in a trusted local environment.

Set your project ID:

```bash
export STREAMLAKE_PROJECT_ID="proj-your-project-id"
```

You may instead save the project ID in `~/.local/share/streamlake-eval-monitor/config/project_id`, or pass `--project-id` to `probe` or `sync`. The precedence order is command-line argument, environment variable, then the configuration file.

## Usage

```bash
TOOL="$HOME/.codex/skills/streamlake-experiment-analyst/scripts/streamlake_experiments.py"

python "$TOOL" probe
python "$TOOL" sync
python "$TOOL" download-log EVALUATION_ID
# Optional later cutoff:
python "$TOOL" sync --since 2026-08-15T00:00:00Z
python "$TOOL" status
python "$TOOL" list
python "$TOOL" compare EVAL_ID_A EVAL_ID_B \
  --baseline EVAL_ID_A --primary-metric score
python "$TOOL" context
```

Use `sync --log-dir PATH` to override the automatic log root. Downloaded, reused, and failed log counts are stored in `sync_state.json` under `log_downloads`; both `error_count` and `log_downloads.error_count` must be zero for complete metadata and log coverage.

### Deploy the unified Monitor

```bash
MONITOR_TOOL="$HOME/.codex/skills/streamlake-experiment-analyst/scripts/eval_monitor.py"

# Deploy or upgrade, then start on 127.0.0.1:18280 in the background.
python "$MONITOR_TOOL" deploy

python "$MONITOR_TOOL" status
python "$MONITOR_TOOL" stop
python "$MONITOR_TOOL" start
```

On first use, open the settings dialog on the evaluation tab and paste complete request headers copied from the authenticated StreamLake browser session. The server extracts the Cookie and project ID without echoing the Cookie. Use the top switcher to open the training tab at `/train/`.

The training tab default runtime config is `~/.local/share/streamlake-eval-monitor/training_monitor_config.json`; it initially scans `/root/output`. Edit that runtime file to add output roots, explicit targets, PID/log paths, or optional Hugging Face upload settings.

Run `deploy` on the server. Then run the tunnel in a new terminal on the computer where the browser runs, not inside the remote SSH shell:

```bash
ssh -N -L 18280:127.0.0.1:18280 USER@SERVER
```

Replace `USER@SERVER` with the real SSH login and add `-p SSH_PORT` when required. Use a personal SSH alias only when it is already configured on the current computer; the skill does not embed or assume one. Keep the command running, then open `http://127.0.0.1:18280/` locally. Run `python "$MONITOR_TOOL" deploy --help` for custom deployment paths and ports.

Generated outputs:

- `experiments.sqlite`: normalized experiments, parameters, metrics, relations, and synchronization errors.
- `raw/`: lightweight sanitized API evidence.
- `exports/`: CSV tables for external analysis.
- `catalog.md`: experiment discovery index.
- `context.md`: bounded context for Codex and other coding assistants.
- `sync_state.json`: freshness, coverage, and `error_count`.
- `analysis_cache/`: compressed Monitor-derived log analysis reused across later page loads and service restarts.

Treat a nonzero `error_count` as partial coverage. Missing metrics remain missing and are never converted to zero.

## Safety boundary

The client permits only the documented read-only list, detail, dashboard, metric-query, and evaluation-output endpoints on `https://console.streamlake.com`. `download-log` may then access the HTTPS `safetyimg.com` log URL returned by the evaluation-output endpoint without forwarding the StreamLake Cookie. Other hosts, mutating methods/endpoints, model-package downloads, and inference-result downloads remain blocked.

The tool intentionally does not download model weights, checkpoints, datasets, or predictions. Evaluation logs are downloaded only through `download-log`. Review [the API contract](references/api-contract.md) before adapting it to a changed StreamLake deployment.

## Testing

```bash
python tests/test_streamlake_experiments.py
python tests/test_eval_monitor.py
python -m py_compile scripts/streamlake_experiments.py
```

## Limitations

- The API contract was inferred from the Wanqing web console and may change.
- Evaluation protocols should be compared only when task, dataset version, split, candidate set, cutoff, and inference settings are compatible.
- Configuration/result relationships are correlations unless established through controlled repeated experiments.

## License

MIT. See [LICENSE](LICENSE).
