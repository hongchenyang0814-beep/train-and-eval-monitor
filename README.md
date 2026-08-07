# Train and Eval Monitor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文说明](README.zh-CN.md)

Train and Eval Monitor is a Codex skill and dependency-free Python toolkit for two related workflows:

- synchronizing StreamLake/Wanqing fine-tuning and competition-evaluation records;
- operating one SSH-accessible web Monitor for local LLaMA-Factory training and evaluation-log analysis.

It downloads complete evaluation logs when the guarded workflow allows them. It never downloads model weights, checkpoints, datasets, or prediction files.

## Mandatory Path Rule

Every machine has its own home directory, repository checkout, training output root, run directories, and SSH settings. The files in this repository are templates, not a copy of the author's machine.

Before using the Monitor, each user must:

1. choose the local skill checkout directory;
2. deploy with an explicit `--target-dir` and `--output-dir` when the defaults are not suitable;
3. pass one or more **existing absolute** training output roots with `--training-output-root`, or set one of the supported environment variables;
4. open `<target-dir>/training_monitor_config.json` and verify every `outputs_roots` and `targets.*` path belongs to the current machine;
5. replace every copied `output_dir`, `metrics_path`, `log_path`, and `config_path` before using an explicit target.
6. after loading an existing runtime, run `python "$MONITOR_TOOL" repair-paths` or deploy with `--repair-paths` so stale paths from another machine are removed automatically.

Never copy another machine's `/root/...`, `/home/...`, Windows drive path, personal SSH alias, Cookie, project ID, Hugging Face token, PID, cache, or service log. A path that does not exist on the current machine must be corrected or removed. This is a hard project constraint.

For a new runtime, deployment conservatively checks `--training-output-root`, `STREAMLAKE_TRAINING_OUTPUT_ROOT`, `LLAMA_FACTORY_OUTPUT_ROOT`, `LLAMA_FACTORY_OUTPUT_DIR`, `TRAINING_OUTPUT_ROOT`, `TRAINING_OUTPUT_DIR`, the current user's `~/output`, `~/LLaMA-Factory/output`, and `./output`. It only records directories that already exist. It never scans the whole filesystem and never overwrites an existing runtime training configuration.

The fixed runtime template is machine-relative: `$HOME/.local/share/streamlake-eval-monitor` with `config/` and `data/` below it. The training directory is represented by `<TRAINING_OUTPUT_ROOT>` until deployment discovers or receives a real directory. `repair-paths` resolves the template on the current computer, removes invalid `outputs_roots` and `targets`, and keeps valid local entries.

## Install

Requirements: Git, Python 3.10+, and Codex.

```bash
SKILL_DIR="$HOME/.codex/skills/streamlake-experiment-analyst"
mkdir -p "$(dirname "$SKILL_DIR")"
git clone "https://github.com/<OWNER>/train-and-eval-monitor.git" "$SKILL_DIR"
python "$SKILL_DIR/scripts/streamlake_experiments.py" --help
```

Replace `<OWNER>` with the repository owner supplied by the project maintainer, or use the clone URL you received separately.

Restart Codex or start a new task so the skill is rediscovered. The GitHub repository is named `Train and Eval Monitor`; the Codex skill identifier remains `streamlake-experiment-analyst` for compatibility.

Update an existing checkout:

```bash
git -C "$HOME/.codex/skills/streamlake-experiment-analyst" pull --ff-only
```

## Deploy With Correct Paths

Run this on the training/evaluation server. Replace the training root with an existing absolute path on that server:

```bash
SKILL_DIR="$HOME/.codex/skills/streamlake-experiment-analyst"
MONITOR_TOOL="$SKILL_DIR/scripts/eval_monitor.py"
TRAINING_ROOT="/absolute/path/to/your/training/output"
TARGET_DIR="$HOME/.local/share/train-and-eval-monitor"

python "$MONITOR_TOOL" deploy \
  --target-dir "$TARGET_DIR" \
  --output-dir "$TARGET_DIR/data" \
  --training-output-root "$TRAINING_ROOT"
```

Repeat `--training-output-root` for multiple roots. If the command reports `Training output roots: none detected`, edit `<target-dir>/training_monitor_config.json` before using the training page. The deployment is loopback-only and starts on port `18280` by default.

Manage the service:

```bash
python "$MONITOR_TOOL" status --target-dir "$TARGET_DIR"
python "$MONITOR_TOOL" stop --target-dir "$TARGET_DIR"
python "$MONITOR_TOOL" start --target-dir "$TARGET_DIR"
```

Re-running `deploy` upgrades managed application files and preserves runtime configuration, credentials, caches, upload registry, PID, and logs. Review paths after an upgrade; the skill never silently rewrites existing user configuration.

## Configure Training Uploads

The training settings gear stores only the Hugging Face Write Access Token at `<target-dir>/config/huggingface_token` with mode `600`. It does not change repository settings.

Repository creation and checkpoint upload use category profiles in the runtime configuration. A profile key matches the real directory category: use `<output-root>/<category>/<run-id>/` for one level, or `<output-root>/<category>/<subcategory>/<run-id>/` for two levels. Each eligible run must contain both `trainer_log.jsonl` and `training_config.yaml`; `run_id` must equal its directory name. Categories without an exact profile remain visible in the Monitor but cannot upload. Configure a distinct repository prefix for each category that should upload:

```json
{
  "outputs_roots": ["/absolute/path/to/your/training/output"],
  "targets": [],
  "auto_upload": {
    "config_file": "training_config.yaml",
    "profiles": {
      "lora_sft": {
        "hub_endpoint": "https://huggingface.co",
        "hub_owner": "YOUR_NAMESPACE",
        "hub_repo_prefix": "lora-sft",
        "hub_private_repo": true,
        "hub_index_repo_id": "YOUR_NAMESPACE/lora_sft",
        "base_model_id": "ORG/BASE_MODEL"
      },
      "ai_infra/benchmark": {
        "hub_endpoint": "https://huggingface.co",
        "hub_owner": "YOUR_NAMESPACE",
        "hub_repo_prefix": "benchmark",
        "hub_private_repo": true,
        "base_model_id": "ORG/BASE_MODEL"
      }
    }
  }
}
```

Add, remove, or rename profile keys whenever the output hierarchy changes; profile keys are relative paths below an entry in `outputs_roots`, excluding the final run directory. The same layout requirement also applies to `targets`: `config_path` must be the run-local `training_config.yaml`, and `run_id` must equal the directory name. The Monitor derives `<owner>/<prefix>-<run-id>-step-<five-digit-step>`, stages the evaluation package, creates the repository, uploads it, and updates the optional index. A successful run/step cannot be uploaded twice.

## Configure StreamLake

Open the evaluation page settings and paste complete request headers from the logged-in browser Network panel. Use a request whose Referer contains `/wanqing/proj-.../`; the Monitor extracts Cookie and project ID and never echoes the Cookie.

The CLI defaults to `<target-dir>/config/cookie` and `<target-dir>/config/project_id`. The project ID may also be supplied through `STREAMLAKE_PROJECT_ID` or `--project-id`. Credentials never belong in Git, command-line history, README files, or screenshots.

## Use The Monitor

On the server:

```bash
python "$MONITOR_TOOL" deploy --target-dir "$TARGET_DIR" --training-output-root "$TRAINING_ROOT"
```

On the computer where the browser runs, keep this tunnel open:

```bash
ssh -N -L 18280:127.0.0.1:18280 USER@SERVER
```

Replace `USER@SERVER` with the real SSH login and add `-p SSH_PORT` when needed. A personal SSH alias is valid only if it is already configured on that computer. Open `http://127.0.0.1:18280/` for training or `http://127.0.0.1:18280/eval/` for evaluation logs.

Training provides metrics, progress, GPU state, run manifests, checkpoints, recent logs, and the original checkpoint upload flow. Evaluation provides synchronization, task grouping, Think/No-think sample browsing, complete-log download, comparisons, ranking, and persistent incremental analysis. Normal synchronization is restricted to records created on or after `2026-08-01T00:00:00Z`; failed evaluations without output are skipped, running evaluations wait for output, and complete logs are downloaded without a file-size limit.

## Required Training Explanation Contract

Before an AI starts each training run, it must create and validate the explanation files in that run's output directory:

```bash
MANIFEST_TOOL="$HOME/.codex/skills/streamlake-experiment-analyst/scripts/experiment_manifest.py"
python "$MANIFEST_TOOL" init --output-dir "/absolute/path/to/run-output"
# Fill every required field in run_manifest.json
python "$MANIFEST_TOOL" validate --output-dir "/absolute/path/to/run-output"
python "$MANIFEST_TOOL" render --output-dir "/absolute/path/to/run-output"
```

Required fields are `schema_version`, `run_id`, `title`, `purpose`, `hypothesis`, `changes`, `comparison_run`, `dataset`, `model`, `config_file`, `expected_result`, `notes`, and `created_at`. `config_file` must be a filename inside the run directory; it must not be an absolute path from another machine. An AI must not launch training when validation fails. When parameters or the comparison baseline change, it must update and render the manifest again. `run_manifest.json` is machine-readable; `training_task.md` is the readable document displayed by the evaluation page.

The evaluation toolbar accepts manual evaluation-log uploads before the Sync button. The same parser validates tasks, samples, or evaluation metadata, then permanently stores the original under `<output-dir>/logs/<eval-task-id>/evaluation.log`, the same path used by synchronized logs, with an analysis cache, `evaluation_note.md`, and a visible `自主上传` label. The evaluation detail header can bind a synchronized or manually uploaded evaluation to a discovered training task. Binding does not require a `run_manifest.json` or `training_task.md`; the page shows the run date/ID and manifest title when available, and explicitly reports when the optional training explanation is unavailable. This keeps the relationship explicit instead of inferring it from a model name.

## CLI Commands

```bash
TOOL="$HOME/.codex/skills/streamlake-experiment-analyst/scripts/streamlake_experiments.py"
python "$TOOL" probe
python "$TOOL" sync
python "$TOOL" status
python "$TOOL" list
python "$TOOL" compare EVAL_ID_A EVAL_ID_B --baseline EVAL_ID_A --primary-metric score
python "$TOOL" download-log EVAL_ID
python "$TOOL" context
```

Use `sync --log-dir ABSOLUTE_PATH` only when the log root must differ from the runtime data directory. Treat a nonzero metadata `error_count` or `log_downloads.error_count` as partial coverage.

## Tests

```bash
python tests/test_streamlake_experiments.py
python tests/test_eval_monitor.py
python -m py_compile scripts/streamlake_experiments.py scripts/eval_monitor.py scripts/experiment_manifest.py scripts/release_check.py
python scripts/release_check.py
```

`release_check.py` is the final portability and privacy gate. Run it from the
checkout before sharing or publishing; it rejects missing Monitor files,
machine-specific runtime roots/targets, generated runtime artifacts, and
non-placeholder personal paths, account URLs, or credentials.

## Safety

The StreamLake client uses documented read-only endpoints and a guarded log-download route. It never sends the StreamLake Cookie to the log CDN. The Monitor binds to loopback and should be exposed through SSH forwarding, not a public interface. Do not commit runtime credentials, generated data, checkpoints, model files, or personal path configuration.

## License

MIT. See [LICENSE](LICENSE).
