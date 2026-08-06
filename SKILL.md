---
name: streamlake-experiment-analyst
description: Use when working with StreamLake or Wanqing fine-tuning and competition-evaluation experiments, synchronizing records and complete evaluation logs locally, deploying or operating the bundled evaluation-log Monitor website, comparing runs, ranking metrics, investigating regressions, or preparing experiment context for coding assistants.
---

# StreamLake Experiment Analyst

## Overview

Maintain a read-only local experiment repository, then analyze from SQLite, CSV, sanitized JSON, Markdown, and explicitly downloaded evaluation logs. Never infer results from names alone. Evaluation log downloads are allowed only through the guarded `download-log` workflow; never download checkpoints, weights, datasets, prediction files, or other artifact bodies.

Path hard constraint: never assume the author's home directory, repository checkout, training output root, run directory, Windows drive, or SSH alias. On every new machine, deploy with that machine's `--target-dir`, `--output-dir`, and one or more existing absolute `--training-output-root` values, then verify every `outputs_roots` and `targets.*` path in `<target-dir>/training_monitor_config.json`. Replace or remove paths that do not exist locally. Never copy personal paths, credentials, caches, PIDs, or service logs into the skill source.

The runtime location is intentionally stable and machine-relative: `$HOME/.local/share/streamlake-eval-monitor` with `config/` and `data/` below it. The training root is a `<TRAINING_OUTPUT_ROOT>` placeholder, not an author path. After loading this Skill on another computer, run `python "$MONITOR_TOOL" repair-paths` (or deploy with `--repair-paths`) to remove stale absolute roots and targets and keep only directories that exist on that computer.

## Quick reference

Set the script path once:

```bash
STREAMLAKE_TOOL="$HOME/.codex/skills/streamlake-experiment-analyst/scripts/streamlake_experiments.py"
MONITOR_TOOL="$HOME/.codex/skills/streamlake-experiment-analyst/scripts/eval_monitor.py"
MANIFEST_TOOL="$HOME/.codex/skills/streamlake-experiment-analyst/scripts/experiment_manifest.py"
```

| Task | Command |
|---|---|
| Check authentication and API shape | `python "$STREAMLAKE_TOOL" probe` |
| Synchronize experiments and complete logs created since 2026-08-01 | `python "$STREAMLAKE_TOOL" sync` |
| Download one complete evaluation log | `python "$STREAMLAKE_TOOL" download-log EVAL_ID` |
| Show cache freshness and counts | `python "$STREAMLAKE_TOOL" status` |
| List cached experiments | `python "$STREAMLAKE_TOOL" list` |
| Compare exact IDs | `python "$STREAMLAKE_TOOL" compare ID1 ID2 --baseline ID1 --primary-metric METRIC` |
| Print the bounded AI context pack | `python "$STREAMLAKE_TOOL" context` |
| Deploy or upgrade and start the evaluation Monitor | `python "$MONITOR_TOOL" deploy` |
| Repair paths on a new machine | `python "$MONITOR_TOOL" repair-paths` |
| Check the Monitor | `python "$MONITOR_TOOL" status` |
| Stop or start the Monitor | `python "$MONITOR_TOOL" stop` / `python "$MONITOR_TOOL" start` |
| Create a required training explanation | `python "$MANIFEST_TOOL" init --output-dir /absolute/run-dir` |
| Validate before training | `python "$MANIFEST_TOOL" validate --output-dir /absolute/run-dir` |
| Render the readable training document | `python "$MANIFEST_TOOL" render --output-dir /absolute/run-dir` |
| Verify portable sanitized checkout | `python "$HOME/.codex/skills/streamlake-experiment-analyst/scripts/release_check.py"` |

Before sharing this Skill or publishing a checkout, run `python "$HOME/.codex/skills/streamlake-experiment-analyst/scripts/release_check.py"`. It verifies that the bundled Monitor and helper scripts exist, the training template contains no machine-specific roots or targets, and no runtime artifacts, personal paths, account URLs, or credentials are present.

Run each subcommand with `--help` for filters and overrides. By default, both the CLI and Monitor use `~/.local/share/streamlake-eval-monitor`, with credentials under `config/` and synchronized artifacts under `data/`. Set `STREAMLAKE_PROJECT_ID`, save the ID in `config/project_id`, or pass `--project-id`; the skill has no built-in project identifier.

Synchronization is hard-restricted to records created on or after `2026-08-01T00:00:00Z`; records without a parseable creation timestamp are excluded. `sync --since ISO-8601-TIMESTAMP` may narrow the range further but cannot request an earlier time. Pagination still scans every page before filtering, and older records are excluded before detail, metric, and output requests.

Evaluation records marked `FAILED` with `hasOutput=false` are excluded before detail and output requests, because the platform has explicitly declared that no result is available.

Running or pending evaluations remain in the synchronized metadata but do not trigger an output request until `hasOutput=true`; a later sync downloads their logs after successful completion.

Every normal `sync` also ensures that the complete logs for all synchronized `SUCCEEDED` evaluations with `hasOutput=true` exist under `<output-dir>/logs/<evalTaskId>/evaluation.log`. Missing logs are downloaded without a file-size limit; existing atomically completed files are reused and re-hashed. Per-log failures are recorded under `log_downloads` in `sync_state.json` without rolling back successfully synchronized metadata. Use `sync --log-dir PATH` to override the log root.

## Evaluation Monitor

Use the bundled unified Monitor when the user asks for a website, dashboard, log workbench, visual sample browser, local training monitor, or SSH-accessible deployment. On the server, run `python "$MONITOR_TOOL" deploy`; this creates a self-contained runtime at `~/.local/share/streamlake-eval-monitor`, including `config/`, `data/`, the deployed app, PID, and service log, starts a detached server on `127.0.0.1:18280`, and prints the SSH tunnel command. It must remain loopback-only. Do not put runtime credentials or generated data in the installed skill source directory.

Tell the user to run the tunnel command on the computer where their browser runs, not inside the remote SSH shell: `ssh -N -L 18280:127.0.0.1:18280 USER@SERVER`. Replace `USER@SERVER` with the real SSH login, such as `alice@203.0.113.10`; add `-p SSH_PORT` when required. A user-specific SSH alias may be used only when it is already defined on that computer. Keep the tunnel terminal open, then open `http://127.0.0.1:18280/` locally for the training Monitor or `http://127.0.0.1:18280/eval/` for evaluation logs. Read [references/eval-monitor.md](references/eval-monitor.md) for exact Windows, alias, custom-port, and troubleshooting commands.

Open the settings dialog to paste complete StreamLake request headers. The app extracts and persists the Cookie and project ID without echoing the Cookie. Use the Sync button to update metadata and complete logs. After each successful sync, the Monitor analyzes only new or changed logs in the background and persists full parsed results under `<output-dir>/analysis_cache`; page loads do not start batch analysis or show cache progress, and later page loads and service restarts reuse valid caches. Opening an uncached evaluation may analyze and persist that one log on demand. Re-run `deploy` after the skill is upgraded; never package credentials, generated caches, PID files, service logs, or screenshots.

The evaluation page includes overview, task, sample, log, pairwise compare, ranking, and tools tabs. The ranking tab sorts all synchronized evaluations by platform scores or cached local automatic metrics; it reads lightweight analysis summaries and must not trigger batch parsing during normal page load.

Every training execution has a mandatory explanation contract. Before an AI starts a training command it must run `experiment_manifest.py init` for the run output directory (unless a manifest already exists), fill every required field, run `validate`, and stop if validation fails. After changing the task or completing it, the AI must update `run_manifest.json` and run `render`, which writes the human-readable `training_task.md`. Required fields are `schema_version`, `run_id`, `title`, `purpose`, `hypothesis`, `changes`, `comparison_run`, `dataset`, `model`, `config_file`, `expected_result`, `notes`, and `created_at`; `config_file` must be a filename inside that run directory. This is a hard workflow constraint for AI-operated training, not optional UI metadata.

The evaluation toolbar accepts manually uploaded evaluation logs before the Sync button. The upload uses the same parser and is rejected when no task, sample, or evaluation metadata can be recognized. Accepted originals are permanently stored under `<output-dir>/logs/<eval-task-id>/evaluation.log`, the same path used by platform synchronization, with an `evaluation_note.md`, analysis cache, and a visible `自主上传` source badge; platform sync never removes them. The evaluation detail header can bind a synchronized or manually uploaded evaluation to a discovered training run. Binding is intentionally rejected unless that run has a valid `run_manifest.json`; the binding stores only the validated training task identity and sanitized manifest/document content in the local data directory.

Task binding in the Monitor falls back to task-family prefixes, so unfamiliar `challenge_itemic_*`, `challenge_evolution_*`, `challenge_recommendation_*`, and `challenge_common_sense` aliases still land in the right family instead of collapsing into “other”.

The unified Monitor exposes `/` (canonical training entry, redirecting to `/train/`) for the local LLaMA-Factory training monitor and `/eval/` for StreamLake evaluation-log analysis. The training tab reuses the original training Monitor behavior: it reads local `trainer_log.jsonl`, GPU state, explicit PIDs, checkpoints, run manifests, and recent logs. Its default configuration is `<target-dir>/training_monitor_config.json`; a new deployment discovers only existing machine-local training roots or accepts explicit `--training-output-root` values. Users may edit that runtime file for their own output roots, explicit targets, and optional Hugging Face checkpoint-upload settings. Do not copy personal training config, upload registry, PID, or log files into the skill source.

The training settings dialog only binds or replaces the Hugging Face Write token. It stores that token separately under `<target-dir>/config/huggingface_token` with mode `600`; it does not overwrite the existing non-secret upload configuration. Repository naming, privacy, base model, and optional index settings come from category profiles in the training runtime configuration. An automatic checkpoint upload is eligible only when its path is `<output-root>/<category>/<run-id>/` or `<output-root>/<category>/<subcategory>/<run-id>/`, the exact category has a profile, and the run directory contains `trainer_log.jsonl` plus `training_config.yaml`; `run_id` must equal the directory name. Categories without profiles remain monitor-only. A configured checkpoint upload creates the destination model repository and uploads the staged evaluation package. Never commit or print the Hugging Face token.

In the Log tab, use **Download Complete Log** to stream the selected evaluation's full original log to the browser device. The route accepts only synchronized evaluation IDs and imposes no file-size limit; the browser controls the final local download path.

When handing off a deployment, report both the server lifecycle command and the local tunnel/browser command. If the SSH host, user, or port is unknown, leave explicit placeholders and ask the user to substitute their normal SSH connection values; do not embed a personal SSH alias.

## Workflow

1. Run `status`. If the cache is missing or stale, run `probe`, then `sync`.
2. Confirm both `error_count` and `log_downloads.error_count` are zero before claiming complete metadata and log coverage. If either is nonzero, inspect the recorded errors and qualify conclusions.
3. Resolve experiments by exact ID. Names are acceptable only when unique; ambiguous names must produce candidate IDs.
4. Compare only compatible evaluation protocols. Separate task, dataset version, split, candidate set, metric definition, cutoff, and inference settings when those fields differ.
5. Lead with the best observed result, then report metric deltas, regressions, missing metrics, configuration differences, and data-quality limitations.
6. Describe configuration/result relationships as correlations. Claim causation only for controlled repeated experiments with adequate variance evidence.
7. For Monitor requests, run `status` first. Run `deploy` when absent or upgrading, and `start` only for an existing stopped deployment.

## Safety contract

- Read credentials from `STREAMLAKE_COOKIE` or a mode-`600` Cookie file. Set `STREAMLAKE_COOKIE_FILE` to override the default `~/.local/share/streamlake-eval-monitor/config/cookie`. Never pass a Cookie as a CLI argument or print it.
- Use only the query endpoints documented in [references/api-contract.md](references/api-contract.md). The guarded `download-log` command may follow the evaluation output URL returned by the documented query endpoint, without forwarding the StreamLake Cookie and without imposing a file-size limit. Never call create, publish, terminate, submit, delete, upload, model/checkpoint download, or inference-result endpoints.
- Preserve unknown lightweight fields in sanitized raw JSON. Strip authentication fields, signed query parameters, binary bodies, and large content fields.
- Treat login redirects, `401`, and `403` as expired authentication. Stop and request a refreshed local Cookie.
- Bind the Monitor only to loopback and expose it through SSH port forwarding. Never bind its credential-bearing configuration endpoint to a public interface.

## Local outputs

Use `experiments.sqlite` for joins, `exports/*.csv` for tabular tools, `raw/` for sanitized evidence, `catalog.md` for discovery, `context.md` for bounded LLM context, and `analysis_cache/` for Monitor-derived persistent results. Do not commit this generated directory unless the user explicitly requests it.

## Common mistakes

- Comparing every numeric value in one leaderboard even when protocols differ.
- Treating a missing metric as zero.
- Selecting the first experiment when names collide.
- Reporting a single best run as a stable improvement.
- Downloading an evaluation log outside the guarded `download-log` workflow or forwarding the StreamLake Cookie to the log CDN.
