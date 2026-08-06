#!/usr/bin/env python3
"""Local LLaMA-Factory monitor with explicit checkpoint archiving to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "monitor_config.json"
HTML_PATH = HERE / "dashboard.html"
UPLOAD_REGISTRY_PATH = HERE / "upload_registry.json"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
STEP_KEYS = ("current_steps", "macro_step", "step", "global_step", "logical_microbatch_step")
EXCLUDED_METRICS = {
    "current_steps",
    "total_steps",
    "macro_step",
    "step",
    "global_step",
    "logical_microbatch_step",
    "epoch",
    "percentage",
    "timestamp",
    "total_tokens",
}
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ERROR_RE = re.compile(
    r"traceback|out of memory|cuda error|nccl.*(?:error|failed)|exception|fatal|segmentation fault",
    re.IGNORECASE,
)
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
UPLOAD_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
GROUP_LABELS = {
    "ai_infra": "AI Infra",
    "lora_sft": "LoRA SFT",
}
SUBGROUP_LABELS = {
    "ai_infra": {
        "benchmark": "Benchmark",
        "profiler": "Profiler",
    },
}
MANIFEST_MAX_BYTES = 256 * 1024
MANIFEST_TEXT_LIMIT = 4000


class UploadConflictError(ValueError):
    """Raised when an immutable evaluation checkpoint was already uploaded."""


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    host = str(config.get("bind_host", "127.0.0.1"))
    if host not in LOOPBACK_HOSTS:
        raise ValueError(f"bind_host must be a loopback address, got {host!r}")
    config["bind_host"] = host
    config["port"] = int(config.get("port", 18180))
    config["refresh_seconds"] = max(10, int(config.get("refresh_seconds", 30)))
    config["stale_after_seconds"] = max(60, int(config.get("stale_after_seconds", 600)))
    config["max_scan_depth"] = max(1, min(6, int(config.get("max_scan_depth", 3))))
    config.setdefault("outputs_roots", ["/root/output"])
    config.setdefault("targets", [])
    return config


def stable_id(path: Path) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", path.name).strip("-").lower() or "run"
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def experiment_group(output_dir: Path, output_roots: list[str]) -> dict[str, str]:
    resolved = output_dir.expanduser().resolve()
    for root_value in output_roots:
        root = Path(root_value).expanduser().resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) >= 2:
            group_id = relative.parts[0]
            label = GROUP_LABELS.get(group_id, group_id.replace("_", " ").replace("-", " ").title())
            return {"id": group_id, "label": label}
    return {"id": "other", "label": "Other"}


def experiment_subgroup(output_dir: Path, output_roots: list[str]) -> dict[str, str] | None:
    resolved = output_dir.expanduser().resolve()
    for root_value in output_roots:
        root = Path(root_value).expanduser().resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) < 3:
            return None
        group_id, subgroup_id = relative.parts[:2]
        labels = SUBGROUP_LABELS.get(group_id)
        if labels is None:
            return None
        label = labels.get(subgroup_id, subgroup_id.replace("_", " ").replace("-", " ").title())
        return {"id": subgroup_id, "label": label}
    return None


def apply_automatic_upload_config(target: dict[str, Any], config: dict[str, Any]) -> None:
    defaults = config.get("auto_upload")
    if not isinstance(defaults, dict) or target.get("enable_upload") is False:
        return
    if (target.get("group") or {}).get("id") != defaults.get("group_id"):
        return
    config_file = str(defaults.get("config_file", "training_config.yaml"))
    if Path(config_file).name != config_file:
        return
    config_path = Path(target["output_dir"]).resolve() / config_file
    if not config_path.is_file():
        return
    for field in ("hub_endpoint", "hub_owner", "hub_repo_prefix", "hub_private_repo", "hub_index_repo_id", "base_model_id"):
        if target.get(field) in (None, "") and defaults.get(field) not in (None, ""):
            target[field] = defaults[field]
    target.setdefault("config_path", str(config_path))
    target.setdefault("run_id", Path(target["output_dir"]).name)
    target.setdefault("enable_upload", True)


def discover_subgroups(config: dict[str, Any]) -> list[dict[str, str]]:
    discovered: dict[tuple[str, str], dict[str, str]] = {}
    for root_value in config["outputs_roots"]:
        root = Path(root_value).expanduser().resolve()
        for group_id, labels in SUBGROUP_LABELS.items():
            group_dir = root / group_id
            if not group_dir.is_dir():
                continue
            try:
                children = list(group_dir.iterdir())
            except OSError:
                continue
            for child in children:
                if not child.is_dir() or child.name.startswith("."):
                    continue
                subgroup_id = child.name
                label = labels.get(subgroup_id, subgroup_id.replace("_", " ").replace("-", " ").title())
                discovered[(group_id, subgroup_id)] = {"group_id": group_id, "id": subgroup_id, "label": label}
    return sorted(discovered.values(), key=lambda item: (item["group_id"], item["label"]))


def iter_log_files(root: Path, max_depth: int):
    if not root.is_dir():
        return
    root_depth = len(root.parts)
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.parts) - root_depth
        if depth >= max_depth:
            dirs[:] = []
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        if "trainer_log.jsonl" in files:
            yield current_path / "trainer_log.jsonl"


def discover_targets(config: dict[str, Any]) -> list[dict[str, Any]]:
    by_dir: dict[str, dict[str, Any]] = {}
    for root_value in config["outputs_roots"]:
        root = Path(root_value).expanduser().resolve()
        for log_path in iter_log_files(root, config["max_scan_depth"]):
            output_dir = log_path.parent.resolve()
            key = str(output_dir)
            by_dir[key] = {
                "id": stable_id(output_dir),
                "label": output_dir.name,
                "output_dir": key,
                "metrics_path": str(log_path.resolve()),
                "group": experiment_group(output_dir, config["outputs_roots"]),
                "subgroup": experiment_subgroup(output_dir, config["outputs_roots"]),
            }

    for raw in config["targets"]:
        output_dir = Path(raw["output_dir"]).expanduser().resolve()
        key = str(output_dir)
        target = by_dir.get(
            key,
            {
                "id": stable_id(output_dir),
                "label": output_dir.name,
                "output_dir": key,
                "metrics_path": str(output_dir / "trainer_log.jsonl"),
                "group": experiment_group(output_dir, config["outputs_roots"]),
                "subgroup": experiment_subgroup(output_dir, config["outputs_roots"]),
            },
        )
        for field in (
            "id", "label", "group", "subgroup", "metrics_path", "log_path", "pid", "config_path",
            "hub_endpoint", "hub_owner", "hub_repo_prefix", "hub_private_repo", "hub_index_repo_id",
            "base_model_id", "run_id", "enable_upload",
        ):
            if raw.get(field) not in (None, ""):
                target[field] = raw[field]
        by_dir[key] = target
    for target in by_dir.values():
        apply_automatic_upload_config(target, config)
    return list(by_dir.values())


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    malformed = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(item, dict):
                    records.append(item)
                else:
                    malformed += 1
    except OSError:
        return [], 0
    return records, malformed


def manifest_text(value: Any, limit: int = MANIFEST_TEXT_LIMIT) -> str:
    if not isinstance(value, (str, int, float, bool)):
        return ""
    return str(value).strip()[:limit]


def read_run_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "run_manifest.json"
    try:
        if manifest_path.stat().st_size > MANIFEST_MAX_BYTES:
            return {"available": False, "error": "run_manifest.json 超过 256 KB"}
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"available": False, "error": None}
    except json.JSONDecodeError:
        return {"available": False, "error": "run_manifest.json 不是有效 JSON"}
    except (OSError, UnicodeError):
        return {"available": False, "error": "run_manifest.json 无法读取"}
    if not isinstance(raw, dict):
        return {"available": False, "error": "run_manifest.json 顶层必须是对象"}
    if raw.get("schema_version") != 1:
        return {"available": False, "error": "不支持的 run manifest 版本"}

    raw_changes = raw.get("changes") if isinstance(raw.get("changes"), list) else []
    changes = [manifest_text(item, 1000) for item in raw_changes if manifest_text(item, 1000)][:20]
    summary = []
    for item in raw.get("config_summary", [])[:40] if isinstance(raw.get("config_summary"), list) else []:
        if not isinstance(item, dict):
            continue
        label = manifest_text(item.get("label"), 100)
        value = manifest_text(item.get("value"), 500)
        if label and value:
            summary.append({"label": label, "value": value})

    dataset = raw.get("dataset") if isinstance(raw.get("dataset"), dict) else {}
    model = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    raw_sources = dataset.get("sources") if isinstance(dataset.get("sources"), list) else []
    sources = [manifest_text(item, 300) for item in raw_sources if manifest_text(item, 300)][:20]
    config_file = manifest_text(raw.get("config_file"), 200)
    config_available = False
    if config_file and Path(config_file).name == config_file:
        config_path = (output_dir / config_file).resolve()
        config_available = config_path.parent == output_dir.resolve() and config_path.is_file()

    return {
        "available": True,
        "error": None,
        "schema_version": 1,
        "run_id": manifest_text(raw.get("run_id"), 100),
        "category": manifest_text(raw.get("category"), 100),
        "title": manifest_text(raw.get("title"), 300),
        "purpose": manifest_text(raw.get("purpose")),
        "hypothesis": manifest_text(raw.get("hypothesis")),
        "changes": changes,
        "comparison_run": manifest_text(raw.get("comparison_run"), 100),
        "dataset": {
            "summary": manifest_text(dataset.get("summary")),
            "sources": sources,
        },
        "model": {
            "base_model": manifest_text(model.get("base_model"), 500),
            "training_method": manifest_text(model.get("training_method"), 300),
        },
        "config_file": config_file if Path(config_file).name == config_file else "",
        "config_available": config_available,
        "config_summary": summary,
        "expected_result": manifest_text(raw.get("expected_result")),
        "notes": manifest_text(raw.get("notes")),
        "created_at": manifest_text(raw.get("created_at"), 100),
    }


def step_value(record: dict[str, Any]) -> float | None:
    for key in STEP_KEYS:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


def metric_series(records: list[dict[str, Any]]) -> dict[str, list[list[float]]]:
    series: dict[str, list[list[float]]] = {}
    for record in records:
        step = step_value(record)
        if step is None:
            continue
        for key, value in record.items():
            if key in EXCLUDED_METRICS or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                series.setdefault(key, []).append([step, numeric])
    return series


def summarize(series: dict[str, list[list[float]]]) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for key, points in series.items():
        values = [point[1] for point in points]
        first = values[0]
        last = values[-1]
        summaries[key] = {
            "first": first,
            "last": last,
            "min": min(values),
            "max": max(values),
            "delta": last - first,
            "count": len(values),
        }
    return summaries


def pid_is_alive(pid: Any) -> bool:
    try:
        numeric_pid = int(pid)
    except (TypeError, ValueError):
        return False
    if numeric_pid <= 0:
        return False
    try:
        os.kill(numeric_pid, 0)
        return True
    except OSError:
        return False


def tail_text(path: Path, max_bytes: int = 96 * 1024, max_lines: int = 60) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return []
    text = ANSI_RE.sub("", data.decode("utf-8", errors="replace")).replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-max_lines:]


def checkpoint_list(output_dir: Path) -> list[dict[str, Any]]:
    checkpoints = []
    try:
        children = list(output_dir.iterdir())
    except OSError:
        return checkpoints
    for child in children:
        match = CHECKPOINT_RE.fullmatch(child.name)
        if not match or not child.is_dir():
            continue
        adapter = child / "adapter_model.safetensors"
        checkpoints.append(
            {
                "name": child.name,
                "step": int(match.group(1)),
                "modified_at": child.stat().st_mtime,
                "adapter_bytes": adapter.stat().st_size if adapter.is_file() else None,
            }
        )
    return sorted(checkpoints, key=lambda item: item["step"])


def evaluation_repo_id(target: dict[str, Any], run_id: str, step: int) -> str:
    owner = str(target.get("hub_owner", "")).strip()
    prefix = str(target.get("hub_repo_prefix", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,38}", owner):
        raise ValueError("hub_owner must be a valid Hugging Face namespace")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,70}", prefix):
        raise ValueError("hub_repo_prefix must be a valid Hugging Face repository prefix")
    return f"{owner}/{prefix}-{run_id}-step-{step:05d}"


def render_evaluation_index(entries: list[dict[str, Any]]) -> str:
    lines = [
        "---",
        "tags:",
        "- lora",
        "- sft",
        "- onerec",
        "---",
        "",
        "# OneRec LoRA SFT Index",
        "",
        "> This is an archive and version index. Do not submit this repository directly to the evaluation platform.",
        "",
        "Each evaluation-ready checkpoint has its own repository, with `adapter_model.safetensors` and "
        "`adapter_config.json` at the repository root.",
        "",
        "## Evaluation Repositories",
        "",
    ]
    if entries:
        lines.extend(["| Run | Step | Evaluation repository |", "| --- | ---: | --- |"])
        for entry in sorted(entries, key=lambda item: (str(item["run_id"]), int(item["step"]))):
            repo_id = str(entry["repo_id"])
            lines.append(f"| `{entry['run_id']}` | `{int(entry['step']):05d}` | [{repo_id}](https://huggingface.co/{repo_id}) |")
    else:
        lines.append("No dedicated evaluation repository has been uploaded yet.")
    lines.extend(
        [
            "",
            "## Optional Legacy Archive",
            "",
            "If you keep an older shared adapter archive, document it here manually. "
            "This generated index only records dedicated evaluation repositories uploaded by this monitor.",
            "",
        ]
    )
    return "\n".join(lines)


def update_evaluation_index(api: Any, index_repo_id: str, entry: dict[str, Any]) -> None:
    index_filename = "evaluation_index.json"
    files = api.list_repo_files(index_repo_id, repo_type="model")
    entries: list[dict[str, Any]] = []
    if index_filename in files:
        downloaded = api.hf_hub_download(index_repo_id, index_filename, repo_type="model")
        try:
            loaded = json.loads(Path(downloaded).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("evaluation index is not valid JSON") from exc
        if not isinstance(loaded, list):
            raise RuntimeError("evaluation index must contain a JSON list")
        entries = [item for item in loaded if isinstance(item, dict)]
    entries = [item for item in entries if item.get("repo_id") != entry["repo_id"]]
    entries.append(entry)
    entries.sort(key=lambda item: (str(item.get("run_id", "")), int(item.get("step", 0))))
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is not installed in the monitor environment") from exc
    api.create_commit(
        repo_id=index_repo_id,
        repo_type="model",
        commit_message=f"Index evaluation checkpoint {entry['run_id']} step-{int(entry['step']):05d}",
        operations=[
            CommitOperationAdd(path_in_repo=index_filename, path_or_fileobj=json.dumps(entries, indent=2).encode()),
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=render_evaluation_index(entries).encode()),
        ],
    )


def load_upload_registry(path: Path) -> dict[str, Any]:
    empty = {"version": 1, "uploads": {}}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(loaded, dict) or not isinstance(loaded.get("uploads"), dict):
        return empty
    return {"version": 1, "uploads": loaded["uploads"]}


def save_upload_registry(path: Path, registry: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def classify_status(latest: dict[str, Any], age_seconds: float | None, process_alive: bool, stale_after: int) -> str:
    current = latest.get("current_steps")
    total = latest.get("total_steps")
    finished_steps = isinstance(current, (int, float)) and isinstance(total, (int, float)) and total > 0 and current >= total
    if process_alive and finished_steps:
        return "finalizing"
    if process_alive and age_seconds is not None and age_seconds > stale_after:
        return "stalled"
    if process_alive:
        return "running"
    if finished_steps:
        return "completed"
    if age_seconds is not None and age_seconds <= stale_after:
        return "recent"
    return "inactive"


def build_experiment(target: dict[str, Any], config: dict[str, Any], now: float) -> dict[str, Any]:
    output_dir = Path(target["output_dir"])
    metrics_path = Path(target["metrics_path"])
    records, malformed = read_jsonl(metrics_path)
    series = metric_series(records)
    latest = records[-1] if records else {}
    try:
        metrics_stat = metrics_path.stat()
        modified_at = metrics_stat.st_mtime
        metrics_bytes = metrics_stat.st_size
        age_seconds = max(0.0, now - modified_at)
    except OSError:
        modified_at = None
        metrics_bytes = 0
        age_seconds = None
    process_alive = pid_is_alive(target.get("pid"))
    log_path = Path(target["log_path"]) if target.get("log_path") else output_dir / "train.log"
    log_lines = tail_text(log_path)
    errors = [line for line in log_lines if ERROR_RE.search(line)][-10:]
    uploads = config.get("_upload_registry", {}).get("uploads", {}).get(target["id"], {})
    if not isinstance(uploads, dict):
        uploads = {}
    upload_enabled = bool(
        target.get("enable_upload")
        and target.get("hub_endpoint")
        and target.get("hub_owner")
        and target.get("hub_repo_prefix")
        and target.get("base_model_id")
        and target.get("config_path")
    )
    return {
        "id": target["id"],
        "label": target["label"],
        "group": target.get("group") or experiment_group(output_dir, config["outputs_roots"]),
        "subgroup": target.get("subgroup") if target.get("subgroup") is not None else experiment_subgroup(output_dir, config["outputs_roots"]),
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "config_path": target.get("config_path"),
        "hub_owner": target.get("hub_owner"),
        "hub_repo_prefix": target.get("hub_repo_prefix"),
        "run_id": target.get("run_id") or output_dir.name,
        "upload_enabled": upload_enabled,
        "pid": target.get("pid"),
        "process_alive": process_alive,
        "status": classify_status(latest, age_seconds, process_alive, config["stale_after_seconds"]),
        "latest": latest,
        "series": series,
        "summaries": summarize(series),
        "metrics": sorted(series),
        "record_count": len(records),
        "malformed_lines": malformed,
        "modified_at": modified_at,
        "metrics_bytes": metrics_bytes,
        "age_seconds": age_seconds,
        "run_manifest": read_run_manifest(output_dir),
        "checkpoints": checkpoint_list(output_dir),
        "uploads": uploads,
        "log_tail": log_lines[-30:],
        "errors": errors,
    }


def gpu_snapshot() -> dict[str, Any]:
    fields = "index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw"
    command = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "gpus": [], "error": str(exc)}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        try:
            memory_used = float(parts[4])
            memory_total = float(parts[5])
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization": float(parts[2]),
                    "memory_utilization": float(parts[3]),
                    "memory_used": memory_used,
                    "memory_total": memory_total,
                    "memory_percent": memory_used / memory_total * 100 if memory_total else 0.0,
                    "temperature": float(parts[6]),
                    "power_draw": float(parts[7]),
                }
            )
        except ValueError:
            continue
    average = sum(gpu["utilization"] for gpu in gpus) / len(gpus) if gpus else 0.0
    average_memory = sum(gpu["memory_percent"] for gpu in gpus) / len(gpus) if gpus else 0.0
    return {
        "available": bool(gpus),
        "gpus": gpus,
        "average_utilization": average,
        "average_memory_percent": average_memory,
        "sampled_at": time.time(),
        "error": None,
    }


def build_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    now = time.time()
    errors = []
    experiments = []
    for target in discover_targets(config):
        try:
            experiments.append(build_experiment(target, config, now))
        except Exception as exc:  # Keep one bad run from hiding all other runs.
            errors.append({"target": target.get("label", target.get("output_dir")), "error": str(exc)})
    experiments.sort(key=lambda item: item["modified_at"] or 0, reverse=True)
    return {
        "generated_at": now,
        "refresh_seconds": config["refresh_seconds"],
        "upload_nonce": config.get("_upload_nonce"),
        "stale_after_seconds": config["stale_after_seconds"],
        "server": {"hostname": os.uname().nodename, "outputs_roots": config["outputs_roots"]},
        "gpu": gpu_snapshot(),
        "experiments": experiments,
        "subgroups": discover_subgroups(config),
        "errors": errors,
    }


def last_eval_loss(trainer_state: dict[str, Any]) -> float | None:
    history = trainer_state.get("log_history")
    if not isinstance(history, list):
        return None
    values = [
        item.get("eval_loss")
        for item in history
        if isinstance(item, dict) and isinstance(item.get("eval_loss"), (int, float))
    ]
    return float(values[-1]) if values else None


def stage_checkpoint_upload(
    checkpoint_dir: Path,
    config_path: Path,
    staged_dir: Path,
    base_model_id: str,
    run_id: str,
    step: int,
) -> tuple[list[str], int]:
    """Create an evaluation-oriented LoRA package without changing the local checkpoint."""
    required = ("adapter_model.safetensors", "adapter_config.json")
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if missing:
        raise ValueError(f"checkpoint is missing required files: {', '.join(missing)}")
    staged_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    total_bytes = 0
    for name in UPLOAD_FILES:
        source = checkpoint_dir / name
        if not source.is_file():
            continue
        destination = staged_dir / name
        if name == "adapter_config.json":
            try:
                adapter_config = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("adapter_config.json is not valid JSON") from exc
            if not isinstance(adapter_config, dict):
                raise ValueError("adapter_config.json must contain a JSON object")
            adapter_config["base_model_name_or_path"] = base_model_id
            destination.write_text(json.dumps(adapter_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            shutil.copy2(source, destination)
        files.append(name)
        total_bytes += destination.stat().st_size

    training_config = staged_dir / "training_config.yaml"
    shutil.copy2(config_path, training_config)
    files.append(training_config.name)
    total_bytes += training_config.stat().st_size

    state_path = checkpoint_dir / "trainer_state.json"
    trainer_state: dict[str, Any] = {}
    if state_path.is_file():
        try:
            loaded_state = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded_state, dict):
                trainer_state = loaded_state
        except (OSError, json.JSONDecodeError):
            pass
    metrics = {
        "run_id": run_id,
        "checkpoint": checkpoint_dir.name,
        "global_step": trainer_state.get("global_step", step),
        "epoch": trainer_state.get("epoch"),
        "eval_loss": last_eval_loss(trainer_state),
        "base_model": base_model_id,
    }
    metrics_path = staged_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files.append(metrics_path.name)
    total_bytes += metrics_path.stat().st_size

    readme = staged_dir / "README.md"
    readme.write_text(
        "---\n"
        "base_model: " + base_model_id + "\n"
        "library_name: peft\n"
        "tags:\n- lora\n- sft\n- onerec\n"
        "---\n\n"
        f"# OneRec LoRA SFT - {run_id} step {step:05d}\n\n"
        f"This directory contains the evaluation package for `{checkpoint_dir.name}`. "
        "Load it as a PEFT adapter on the base model declared in `adapter_config.json`.\n",
        encoding="utf-8",
    )
    files.append(readme.name)
    total_bytes += readme.stat().st_size
    return files, total_bytes


class MonitorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: dict[str, Any]):
        self.config = config
        self.upload_lock = threading.Lock()
        self.upload_registry_path = UPLOAD_REGISTRY_PATH
        self.config["_upload_registry"] = load_upload_registry(self.upload_registry_path)
        self.config["_upload_nonce"] = secrets.token_urlsafe(24)
        super().__init__(address, MonitorHandler)

    def record_upload(self, experiment_id: str, checkpoint_name: str, record: dict[str, Any]) -> None:
        registry = self.config["_upload_registry"]
        registry["uploads"].setdefault(experiment_id, {})[checkpoint_name] = record
        save_upload_registry(self.upload_registry_path, registry)

    def rotate_upload_nonce(self) -> None:
        self.config["_upload_nonce"] = secrets.token_urlsafe(24)


class MonitorHandler(BaseHTTPRequestHandler):
    server_version = "TrainMonitor/1.0"

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _send_headers(self, status: int, content_type: str, content_length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode())

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0 or length > 4096:
            raise ValueError("request body must be between 1 and 4096 bytes")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _upload_checkpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = {"experiment_id", "checkpoint_name"}
        if set(payload) - allowed_keys:
            raise ValueError("unsupported upload parameters")
        experiment_id = payload.get("experiment_id")
        checkpoint_name = payload.get("checkpoint_name")
        if not isinstance(experiment_id, str) or not isinstance(checkpoint_name, str):
            raise ValueError("experiment_id and checkpoint_name are required")
        target = next((item for item in discover_targets(self.server.config) if item["id"] == experiment_id), None)  # type: ignore[attr-defined]
        if target is None:
            raise ValueError("selected experiment is not available")
        if not target.get("enable_upload"):
            raise ValueError("upload is disabled for this experiment")
        existing = self.server.config.get("_upload_registry", {}).get("uploads", {}).get(experiment_id, {}).get(checkpoint_name)  # type: ignore[attr-defined]
        if isinstance(existing, dict):
            raise UploadConflictError(f"checkpoint was already uploaded: {existing.get('repo_url', existing.get('repo_id'))}")
        required = ("hub_endpoint", "hub_owner", "hub_repo_prefix", "base_model_id", "config_path")
        if any(not target.get(field) for field in required):
            raise ValueError("upload configuration is incomplete")
        output_dir = Path(target["output_dir"]).resolve()
        checkpoint_dir = (output_dir / checkpoint_name).resolve()
        match = CHECKPOINT_RE.fullmatch(checkpoint_name)
        if not match or checkpoint_dir.parent != output_dir or not checkpoint_dir.is_dir():
            raise ValueError("selected checkpoint is not available")
        run_id = str(target.get("run_id") or output_dir.name)
        if not RUN_ID_RE.fullmatch(run_id):
            raise ValueError("run_id must contain only letters, numbers, underscores, and hyphens")
        config_path = Path(str(target["config_path"])).resolve()
        if not config_path.is_file():
            raise ValueError("training config file is not available")
        step = int(match.group(1))
        repo_id = evaluation_repo_id(target, run_id, step)
        with tempfile.TemporaryDirectory(prefix="train-monitor-upload-") as temporary:
            staged_dir = Path(temporary)
            files, total_bytes = stage_checkpoint_upload(
                checkpoint_dir=checkpoint_dir,
                config_path=config_path,
                staged_dir=staged_dir,
                base_model_id=str(target["base_model_id"]),
                run_id=run_id,
                step=step,
            )
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise RuntimeError("huggingface_hub is not installed in the monitor environment") from exc
            api = HfApi(endpoint=str(target["hub_endpoint"]))
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=bool(target.get("hub_private_repo", True)),
                exist_ok=True,
            )
            result = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(staged_dir),
                commit_message=f"Upload LoRA SFT {run_id} step-{step:05d}",
            )
            index_warning = None
            if target.get("hub_index_repo_id"):
                try:
                    update_evaluation_index(
                        api,
                        str(target["hub_index_repo_id"]),
                        {"run_id": run_id, "step": step, "repo_id": repo_id},
                    )
                except Exception as exc:
                    index_warning = "evaluation repository was uploaded, but the archive index was not updated"
                    print(f"Evaluation index update failed: {exc!r}", flush=True)
        commit_id = getattr(result, "oid", None) or getattr(result, "commit_id", None)
        upload_record = {
            "repo_id": repo_id,
            "repo_url": f"https://huggingface.co/{repo_id}",
            "run_id": run_id,
            "step": step,
            "uploaded_at": time.time(),
        }
        self.server.record_upload(experiment_id, checkpoint_name, upload_record)  # type: ignore[attr-defined]
        return {
            "ok": True,
            "repo_id": repo_id,
            "repo_url": f"https://huggingface.co/{repo_id}",
            "remote_path": "/",
            "commit_id": commit_id,
            "commit_url": getattr(result, "commit_url", None),
            "files": files,
            "bytes": total_bytes,
            "index_warning": index_warning,
        }

    def _handle_upload_payload(self, payload: dict[str, Any]) -> None:
        upload_lock = self.server.upload_lock  # type: ignore[attr-defined]
        if not upload_lock.acquire(blocking=False):
            self._send_json(409, {"error": "another upload is already in progress"})
            return
        try:
            result = self._upload_checkpoint(payload)
            self.server.rotate_upload_nonce()  # type: ignore[attr-defined]
            self._send_json(200, result)
        except UploadConflictError as exc:
            self._send_json(409, {"error": str(exc)})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # Do not return provider exceptions, which may contain sensitive request details.
            print(f"Upload failed: {exc!r}", flush=True)
            self._send_json(500, {"error": "upload failed; inspect monitor_server.log"})
        finally:
            upload_lock.release()

    def _upload_from_query(self, query: dict[str, list[str]]) -> None:
        allowed = {"experiment_id", "checkpoint_name", "nonce"}
        if set(query) != allowed or any(len(values) != 1 for values in query.values()):
            self._send_json(400, {"error": "invalid upload query"})
            return
        nonce = query["nonce"][0]
        expected = self.server.config.get("_upload_nonce", "")  # type: ignore[attr-defined]
        if not isinstance(expected, str) or not secrets.compare_digest(nonce, expected):
            self._send_json(403, {"error": "invalid or expired upload nonce"})
            return
        self._handle_upload_payload(
            {"experiment_id": query["experiment_id"][0], "checkpoint_name": query["checkpoint_name"][0]}
        )

    def do_HEAD(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                size = HTML_PATH.stat().st_size
            except OSError:
                self._send_headers(500, "text/plain; charset=utf-8", 0)
                return
            self._send_headers(200, "text/html; charset=utf-8", size)
        elif path in ("/api/health", "/api/snapshot"):
            self._send_headers(200, "application/json; charset=utf-8", 0)
        else:
            self._send_headers(404, "application/json; charset=utf-8", 0)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            try:
                body = HTML_PATH.read_bytes()
            except OSError as exc:
                self._send(500, "text/plain; charset=utf-8", str(exc).encode())
                return
            self._send(200, "text/html; charset=utf-8", body)
        elif path == "/api/snapshot":
            payload = build_snapshot(self.server.config)  # type: ignore[attr-defined]
            self._send(200, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode())
        elif path == "/api/health":
            self._send(200, "application/json; charset=utf-8", b'{"status":"ok"}')
        elif path == "/api/upload":
            try:
                query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=3)
            except ValueError:
                self._send_json(400, {"error": "invalid upload query"})
                return
            self._upload_from_query(query)
        else:
            self._send(404, "application/json; charset=utf-8", b'{"error":"not found"}')

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/upload":
            self._send_json(404, {"error": "not found"})
            return
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._handle_upload_payload(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        message = re.sub(r"(nonce=)[^&\s\"]+", r"\1[redacted]", fmt % args)
        print(f"[{self.log_date_time_string()}] {self.client_address[0]} {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    server = MonitorServer((config["bind_host"], config["port"]), config)
    print(f"Training monitor: http://{config['bind_host']}:{config['port']}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
