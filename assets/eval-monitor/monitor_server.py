#!/usr/bin/env python3
"""Loopback-only StreamLake evaluation log monitor."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlparse


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "monitor_config.json"
HTML_PATH = HERE / "dashboard.html"
TRAINING_HTML_PATH = HERE / "training_dashboard.html"
TRAINING_SERVER_PATH = HERE / "training_monitor_server.py"
TRAINING_CONFIG_PATH = HERE / "training_monitor_config.json"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
EVALUATION_ID_RE = re.compile(r"eval-task-[A-Za-z0-9-]+")
PROJECT_ID_RE = re.compile(r"(?:^|/)(proj-[A-Za-z0-9-]+)(?:/|$)")
MAX_CONFIG_BODY = 128 * 1024
MAX_SYNC_MESSAGE = 8000
ANALYSIS_CACHE_VERSION = 1
MANUAL_INDEX_VERSION = 1
BINDINGS_VERSION = 1
MANUAL_FILENAME_RE = re.compile(r"[\x00-\x1f]+")


def load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    host = str(raw.get("bind_host", "127.0.0.1"))
    if host not in LOOPBACK_HOSTS:
        raise ValueError("Evaluation monitor must bind to a loopback address")
    config = {
        "bind_host": host,
        "port": int(raw.get("port", 18280)),
        "output_dir": str(Path(raw["output_dir"]).expanduser().resolve()),
        "skill_script": str(Path(raw["skill_script"]).expanduser().resolve()),
        "parser_dir": str(Path(raw["parser_dir"]).expanduser().resolve()),
        "cookie_file": str(Path(raw["cookie_file"]).expanduser().resolve()),
        "project_id_file": str(Path(raw["project_id_file"]).expanduser().resolve()),
        "training_config_file": str(
            Path(raw.get("training_config_file", TRAINING_CONFIG_PATH)).expanduser().resolve()
        ),
        "training_upload_registry_file": str(
            Path(raw.get("training_upload_registry_file", HERE / "training_upload_registry.json")).expanduser().resolve()
        ),
    }
    if not 1024 <= config["port"] <= 65535:
        raise ValueError("Invalid monitor port")
    return config


def parse_request_headers(raw: str) -> tuple[str, str]:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("请粘贴完整请求头")
    if len(raw.encode("utf-8")) > MAX_CONFIG_BODY:
        raise ValueError("请求头内容过大")
    headers: dict[str, str] = {}
    current = ""
    for original in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not original.strip():
            continue
        if original[:1].isspace() and current:
            headers[current] = f"{headers[current]} {original.strip()}"
            continue
        if ":" not in original:
            continue
        key, value = original.split(":", 1)
        normalized = key.strip().lower()
        if not re.fullmatch(r"[a-z0-9-]+", normalized):
            continue
        headers[normalized] = value.strip()
        current = normalized

    cookie = headers.get("cookie", "").strip()
    if not cookie or "=" not in cookie:
        raise ValueError("没有找到有效的 Cookie 请求头")
    referer = headers.get("referer", "")
    project_match = PROJECT_ID_RE.search(urlparse(referer).path)
    if project_match is None:
        project_match = re.search(r"projectId\s*[=:]\s*[\"']?(proj-[A-Za-z0-9-]+)", raw, re.IGNORECASE)
    if project_match is None:
        raise ValueError("没有从 Referer 找到 project ID")
    project_id = project_match.group(1)
    return cookie, project_id


def atomic_write_secret(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value.strip() + "\n")
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_parser_module(parser_dir: Path):
    module_path = parser_dir / "eval_log_parser.py"
    spec = importlib.util.spec_from_file_location("eval_monitor_log_parser", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载评测日志解析器")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_training_module(app_dir: Path):
    module_path = app_dir / "training_monitor_server.py"
    if not module_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("streamlake_training_monitor", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载训练 Monitor 模块")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_training_config(module: Any, config_path: Path, registry_path: Path) -> dict[str, Any] | None:
    if module is None or not config_path.is_file():
        return None
    config = module.load_config(config_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    config["_upload_registry"] = module.load_upload_registry(registry_path)
    config["_upload_nonce"] = secrets.token_urlsafe(24)
    return config


def nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return default if current is None else current


def iso_to_timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


class EvaluationRepository:
    def __init__(self, output_dir: Path, parser_module: Any) -> None:
        self.output_dir = output_dir
        self.db_path = output_dir / "experiments.sqlite"
        self.logs_dir = output_dir / "logs"
        self.legacy_manual_dir = output_dir / "manual_evaluations"
        self.manual_index_path = output_dir / "manual_evaluations.json"
        self.bindings_path = output_dir / "evaluation_training_bindings.json"
        self.analysis_cache_dir = output_dir / "analysis_cache"
        self.parser = parser_module
        self.cache: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.analysis_locks: dict[str, threading.Lock] = {}

    def _read_json_file(self, path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return default

    def _manual_records(self) -> dict[str, dict[str, Any]]:
        payload = self._read_json_file(self.manual_index_path, {})
        if not isinstance(payload, dict):
            return {}
        records = payload.get("records", payload)
        if not isinstance(records, dict):
            return {}
        return {str(key): value for key, value in records.items() if isinstance(value, dict)}

    def _bindings(self) -> dict[str, dict[str, Any]]:
        payload = self._read_json_file(self.bindings_path, {})
        if not isinstance(payload, dict):
            return {}
        bindings = payload.get("bindings", payload)
        if not isinstance(bindings, dict):
            return {}
        return {str(key): value for key, value in bindings.items() if isinstance(value, dict)}

    def _write_manual_records(self, records: Mapping[str, Any]) -> None:
        atomic_write_secret(
            self.manual_index_path,
            json.dumps({"version": MANUAL_INDEX_VERSION, "records": records}, ensure_ascii=False, indent=2),
        )

    def _write_bindings(self, bindings: Mapping[str, Any]) -> None:
        atomic_write_secret(
            self.bindings_path,
            json.dumps({"version": BINDINGS_VERSION, "bindings": bindings}, ensure_ascii=False, indent=2),
        )

    def _manual_record(self, evaluation_id: str) -> dict[str, Any] | None:
        return self._manual_records().get(evaluation_id)

    def _manual_log_path(self, evaluation_id: str) -> Path:
        return self.logs_dir / evaluation_id / "evaluation.log"

    def _legacy_manual_log_path(self, evaluation_id: str) -> Path:
        return self.legacy_manual_dir / evaluation_id / "evaluation.log"

    def _record_exists(self, evaluation_id: str) -> bool:
        if self._manual_record(evaluation_id) is not None:
            return True
        if EVALUATION_ID_RE.fullmatch(evaluation_id) is None or not self.db_path.is_file():
            return False
        try:
            self._row(evaluation_id)
        except (FileNotFoundError, ValueError):
            return False
        return True

    def _binding_for(self, evaluation_id: str) -> dict[str, Any] | None:
        binding = self._bindings().get(evaluation_id)
        return dict(binding) if isinstance(binding, dict) else None

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.is_file():
            raise FileNotFoundError("尚未同步 StreamLake 评测数据")
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _row(self, evaluation_id: str) -> sqlite3.Row:
        if EVALUATION_ID_RE.fullmatch(evaluation_id) is None:
            raise ValueError("评测 ID 无效")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id,name,status,created_at,updated_at,raw_json FROM experiments "
                "WHERE experiment_type='evaluation' AND id=?",
                (evaluation_id,),
            ).fetchone()
        if row is None:
            raise FileNotFoundError("评测不存在")
        return row

    def _log_path(self, evaluation_id: str) -> Path:
        if self._manual_record(evaluation_id) is not None:
            current_path = self._manual_log_path(evaluation_id)
            if current_path.is_file():
                return current_path
            legacy_path = self._legacy_manual_log_path(evaluation_id)
            if legacy_path.is_file():
                return legacy_path
            return current_path
        return self.logs_dir / evaluation_id / "evaluation.log"

    def download_log_path(self, evaluation_id: str) -> Path:
        if EVALUATION_ID_RE.fullmatch(evaluation_id) is None:
            raise ValueError("评测 ID 无效")
        path = self._log_path(evaluation_id)
        if not path.is_file():
            raise FileNotFoundError("该评测尚无可下载的本地日志")
        return path

    def _cache_paths(self, evaluation_id: str) -> tuple[Path, Path]:
        return (
            self.analysis_cache_dir / f"{evaluation_id}.json.gz",
            self.analysis_cache_dir / f"{evaluation_id}.meta.json",
        )

    def _summary_cache_path(self, evaluation_id: str) -> Path:
        return self.analysis_cache_dir / f"{evaluation_id}.summary.json"

    def _signature(self, path: Path) -> dict[str, int]:
        stat = path.stat()
        return {
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "parser_version": int(getattr(self.parser, "PARSER_VERSION", 1)),
        }

    def analysis_inventory(self) -> list[tuple[str, dict[str, int]]]:
        result: dict[str, dict[str, int]] = {}
        for directory in (self.legacy_manual_dir, self.logs_dir):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*/evaluation.log")):
                evaluation_id = path.parent.name
                if EVALUATION_ID_RE.fullmatch(evaluation_id):
                    result[evaluation_id] = self._signature(path)
        return sorted(result.items())

    def _analysis_lock(self, evaluation_id: str) -> threading.Lock:
        with self.lock:
            return self.analysis_locks.setdefault(evaluation_id, threading.Lock())

    def _read_persistent_cache(self, evaluation_id: str, signature: Mapping[str, int]) -> dict[str, Any] | None:
        payload_path, metadata_path = self._cache_paths(evaluation_id)
        if not payload_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("cache_version") != ANALYSIS_CACHE_VERSION or metadata.get("signature") != signature:
                return None
            with gzip.open(payload_path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("cache_version") != ANALYSIS_CACHE_VERSION or payload.get("signature") != signature:
                return None
            parsed = payload.get("parsed")
            return parsed if isinstance(parsed, dict) else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _persistent_cache_valid(self, evaluation_id: str, signature: Mapping[str, int]) -> bool:
        payload_path, metadata_path = self._cache_paths(evaluation_id)
        if not payload_path.is_file() or not metadata_path.is_file():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return metadata.get("cache_version") == ANALYSIS_CACHE_VERSION and metadata.get("signature") == signature
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def _analysis_summary_payload(self, parsed: Mapping[str, Any]) -> dict[str, Any]:
        tasks: dict[str, dict[str, Any]] = {}
        for task in parsed.get("tasks", []):
            if not isinstance(task, Mapping):
                continue
            key = str(task.get("key", ""))
            if not key:
                continue
            tasks[key] = {
                "label": task.get("label", key),
                "group": task.get("group", ""),
                "sample_count": task.get("sample_count"),
                "example_count": task.get("example_count"),
                "generation_seconds": task.get("generation_seconds"),
                "automatic_metrics": task.get("automatic_metrics", {}),
            }
        return {
            "summary": parsed.get("summary", {}),
            "automatic_metrics": parsed.get("automatic_metrics", {}),
            "tasks": tasks,
            "sample_count": len(parsed.get("samples", [])),
        }

    def _write_analysis_summary(
        self,
        evaluation_id: str,
        signature: Mapping[str, int],
        parsed: Mapping[str, Any],
    ) -> None:
        self.analysis_cache_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.analysis_cache_dir, 0o700)
        payload = {
            "cache_version": ANALYSIS_CACHE_VERSION,
            "signature": dict(signature),
            "analysis": self._analysis_summary_payload(parsed),
        }
        atomic_write_secret(
            self._summary_cache_path(evaluation_id),
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )

    def _read_analysis_summary(
        self,
        evaluation_id: str,
        signature: Mapping[str, int],
        *,
        migrate_full_cache: bool = False,
    ) -> dict[str, Any] | None:
        summary_path = self._summary_cache_path(evaluation_id)
        if summary_path.is_file():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                if payload.get("cache_version") == ANALYSIS_CACHE_VERSION and payload.get("signature") == signature:
                    analysis = payload.get("analysis")
                    return analysis if isinstance(analysis, dict) else None
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        if not migrate_full_cache:
            return None
        parsed = self._read_persistent_cache(evaluation_id, signature)
        if parsed is None:
            return None
        self._write_analysis_summary(evaluation_id, signature, parsed)
        return self._analysis_summary_payload(parsed)

    def _write_persistent_cache(
        self,
        evaluation_id: str,
        signature: Mapping[str, int],
        parsed: Mapping[str, Any],
    ) -> None:
        self.analysis_cache_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.analysis_cache_dir, 0o700)
        payload_path, metadata_path = self._cache_paths(evaluation_id)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{evaluation_id}.", suffix=".json.gz", dir=self.analysis_cache_dir)
        temporary = Path(temporary_name)
        os.close(fd)
        try:
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
                json.dump(
                    {"cache_version": ANALYSIS_CACHE_VERSION, "signature": dict(signature), "parsed": parsed},
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            os.chmod(temporary, 0o600)
            os.replace(temporary, payload_path)
            atomic_write_secret(
                metadata_path,
                json.dumps(
                    {"cache_version": ANALYSIS_CACHE_VERSION, "signature": dict(signature)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
            self._write_analysis_summary(evaluation_id, signature, parsed)
        finally:
            temporary.unlink(missing_ok=True)

    def list_evaluations(self) -> list[dict[str, Any]]:
        result = []
        if self.db_path.is_file():
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT id,name,status,created_at,updated_at,raw_json FROM experiments "
                    "WHERE experiment_type='evaluation'"
                ).fetchall()
            for row in rows:
                raw = json.loads(row["raw_json"])
                detail = raw.get("detail") if isinstance(raw.get("detail"), dict) else {}
                summary = nested(detail, "metrics", "metrics", "summary", default={})
                log_path = self._log_path(row["id"])
                result.append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "model_name": detail.get("modelName") or raw.get("modelName") or "",
                        "base_model": detail.get("baseModel") or detail.get("sourceModel") or raw.get("baseModel") or raw.get("sourceModel") or "",
                        "status": detail.get("taskStatus") or raw.get("taskStatus") or row["status"] or "UNKNOWN",
                        "created_at": row["created_at"] or "",
                        "end_time": detail.get("endTime") or raw.get("endTime") or "",
                        "duration": detail.get("duration") or raw.get("duration"),
                        "score": summary.get("totalScore", detail.get("score", raw.get("score"))),
                        "groups": {key: summary.get(key) for key in ("r0", "r1", "r2", "r3")},
                        "has_output": bool(detail.get("hasOutput", raw.get("hasOutput", False))),
                        "origin": "platform_sync",
                        "manual": False,
                        "source_label": "平台同步",
                        "training_binding": self._binding_for(row["id"]),
                        "log": {
                            "available": log_path.is_file(),
                            "size_bytes": log_path.stat().st_size if log_path.is_file() else 0,
                        },
                    }
                )

        for evaluation_id, record in self._manual_records().items():
            evaluation = record.get("evaluation") if isinstance(record.get("evaluation"), dict) else record
            log_path = self._log_path(evaluation_id)
            result.append(
                {
                    "id": evaluation_id,
                    "name": evaluation.get("name") or record.get("filename") or evaluation_id,
                    "model_name": evaluation.get("model_name", ""),
                    "base_model": evaluation.get("base_model", ""),
                    "status": evaluation.get("status", "UPLOADED"),
                    "created_at": evaluation.get("created_at", ""),
                    "end_time": evaluation.get("end_time", ""),
                    "duration": evaluation.get("duration"),
                    "score": evaluation.get("score"),
                    "groups": evaluation.get("groups", {}),
                    "has_output": bool(evaluation.get("has_output", True)),
                    "origin": "manual_upload",
                    "manual": True,
                    "source_label": "自主上传",
                    "filename": record.get("filename", ""),
                    "training_binding": self._binding_for(evaluation_id),
                    "log": {
                        "available": log_path.is_file(),
                        "size_bytes": log_path.stat().st_size if log_path.is_file() else 0,
                    },
                }
            )
        result.sort(key=lambda item: (iso_to_timestamp(str(item.get("created_at", ""))), item["id"]), reverse=True)
        return result

    def _write_manual_note(self, evaluation_id: str, record: Mapping[str, Any]) -> None:
        evaluation = record.get("evaluation") if isinstance(record.get("evaluation"), Mapping) else {}
        validation = record.get("validation") if isinstance(record.get("validation"), Mapping) else {}
        binding = self._binding_for(evaluation_id)
        lines = [
            "# 评测日志说明",
            "",
            f"- 来源：自主上传",
            f"- 文件：{record.get('filename', '')}",
            f"- 评测 ID：{evaluation_id}",
            f"- 上传时间：{evaluation.get('created_at', '')}",
            f"- SHA-256：{record.get('sha256', '')}",
            f"- 解析器版本：{validation.get('parser_version', '')}",
            f"- 识别任务数：{validation.get('recognized_task_count', 0)}",
            f"- 展示样本数：{validation.get('sample_count', 0)}",
            f"- 完成任务数：{validation.get('completed_count', 0)}",
            f"- 解析异常数：{validation.get('issue_count', 0)}",
        ]
        if binding:
            lines.extend(["", "## 绑定训练任务", f"- 任务：{binding.get('label', binding.get('id', ''))}", f"- 任务 ID：{binding.get('id', '')}"])
        else:
            lines.extend(["", "## 绑定训练任务", "尚未绑定训练任务。"])
        note_path = self._log_path(evaluation_id).parent / "evaluation_note.md"
        atomic_write_secret(note_path, "\n".join(lines) + "\n",)

    def add_manual_log(self, source_path: Path, filename: str, training_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
        safe_name = str(filename or "evaluation.log").replace("\\", "/").split("/")[-1]
        safe_name = MANUAL_FILENAME_RE.sub("_", safe_name).strip("._")[:180] or "evaluation.log"
        raw = source_path.read_bytes()
        if not raw:
            raise ValueError("评测日志为空")
        try:
            text, encoding = self.parser.decode_log_bytes(raw)
            parsed = self.parser.parse_log(text, safe_name, len(raw))
        except (UnicodeError, ValueError, TypeError) as error:
            raise ValueError(f"评测日志无法解析：{error}") from error
        if not isinstance(parsed, dict):
            raise ValueError("评测日志解析结果无效")
        tasks = parsed.get("tasks") if isinstance(parsed.get("tasks"), list) else []
        recognized = [task for task in tasks if isinstance(task, Mapping) and task.get("status") != "未发现"]
        samples = parsed.get("samples") if isinstance(parsed.get("samples"), list) else []
        metadata = parsed.get("metadata") if isinstance(parsed.get("metadata"), Mapping) else {}
        if not recognized and not samples and not metadata:
            raise ValueError("文件不是可识别的评测日志：未发现任务、样本或评测元数据")
        digest = hashlib.sha256(raw).hexdigest()
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        evaluation_id = f"eval-task-manual-{datetime.now().astimezone().strftime('%Y%m%d%H%M%S')}-{digest[:10]}-{secrets.token_hex(3)}"
        destination = self._manual_log_path(evaluation_id)
        destination.parent.mkdir(parents=True, exist_ok=False)
        os.chmod(destination.parent, 0o700)
        temporary = destination.with_suffix(".uploading")
        temporary.write_bytes(raw)
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        source = parsed.get("source") if isinstance(parsed.get("source"), Mapping) else {}
        evaluation = {
            "id": evaluation_id,
            "name": safe_name,
            "status": "UPLOADED",
            "created_at": now,
            "start_time": source.get("started_at", ""),
            "end_time": source.get("ended_at", ""),
            "duration": source.get("duration_seconds"),
            "model_name": metadata.get("model_path", ""),
            "base_model": "",
            "score": None,
            "groups": {},
            "has_output": True,
        }
        record = {
            "version": MANUAL_INDEX_VERSION,
            "filename": safe_name,
            "sha256": digest,
            "encoding": encoding,
            "validation": {
                "parser_version": parsed.get("parser_version", getattr(self.parser, "PARSER_VERSION", 1)),
                "recognized_task_count": len(recognized),
                "sample_count": len(samples),
                "completed_count": (parsed.get("summary") or {}).get("completed_count", 0),
                "issue_count": len(parsed.get("issues") or []),
            },
            "evaluation": evaluation,
        }
        with self.lock:
            records = self._manual_records()
            records[evaluation_id] = record
            self._write_manual_records(records)
        signature = self._signature(destination)
        parsed["encoding"] = encoding
        self._write_persistent_cache(evaluation_id, signature, parsed)
        with self.lock:
            self.cache[evaluation_id] = {"signature": signature, "parsed": parsed}
        if training_binding:
            self.bind_training(evaluation_id, training_binding)
        self._write_manual_note(evaluation_id, record)
        return next(item for item in self.list_evaluations() if item["id"] == evaluation_id)

    def bind_training(self, evaluation_id: str, binding: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not self._record_exists(evaluation_id):
            raise FileNotFoundError("评测不存在")
        bindings = self._bindings()
        if binding is None:
            bindings.pop(evaluation_id, None)
        else:
            allowed = {"id", "label", "run_id", "output_dir", "manifest"}
            if set(binding) - allowed or not str(binding.get("id", "")).strip():
                raise ValueError("训练任务绑定格式无效")
            bindings[evaluation_id] = {key: binding[key] for key in allowed if key in binding}
        self._write_bindings(bindings)
        record = self._manual_record(evaluation_id)
        if record:
            self._write_manual_note(evaluation_id, record)
        return bindings.get(evaluation_id)

    def ranking_rows(self) -> list[dict[str, Any]]:
        rows = self.list_evaluations()
        for row in rows:
            log_path = self._log_path(row["id"])
            row["analysis_summary"] = None
            if not log_path.is_file():
                continue
            signature = self._signature(log_path)
            with self.lock:
                cached = self.cache.get(row["id"])
                parsed = cached["parsed"] if cached and cached["signature"] == signature else None
            if parsed is not None:
                row["analysis_summary"] = self._analysis_summary_payload(parsed)
                continue
            row["analysis_summary"] = self._read_analysis_summary(
                row["id"],
                signature,
                migrate_full_cache=True,
            )
        return rows

    def _analysis(self, evaluation_id: str) -> dict[str, Any]:
        path = self._log_path(evaluation_id)
        if not path.is_file():
            raise FileNotFoundError("该评测尚无本地日志")
        signature = self._signature(path)
        with self._analysis_lock(evaluation_id):
            with self.lock:
                cached = self.cache.get(evaluation_id)
                if cached and cached["signature"] == signature:
                    return cached["parsed"]
            parsed = self._read_persistent_cache(evaluation_id, signature)
            if parsed is None:
                raw = path.read_bytes()
                text, encoding = self.parser.decode_log_bytes(raw)
                parsed = self.parser.parse_log(text, path.name, len(raw))
                parsed["encoding"] = encoding
                self._write_persistent_cache(evaluation_id, signature, parsed)
            with self.lock:
                self.cache[evaluation_id] = {"signature": signature, "parsed": parsed}
            return parsed

    def detail(self, evaluation_id: str) -> dict[str, Any]:
        manual = self._manual_record(evaluation_id)
        if manual is not None:
            evaluation = manual.get("evaluation") if isinstance(manual.get("evaluation"), dict) else {}
            response: dict[str, Any] = {
                "evaluation": {
                    **evaluation,
                    "origin": "manual_upload",
                    "manual": True,
                    "source_label": "自主上传",
                    "training_binding": self._binding_for(evaluation_id),
                },
                "analysis": None,
                "training_binding": self._binding_for(evaluation_id),
                "manual_note": str(self._log_path(evaluation_id).parent / "evaluation_note.md"),
            }
            try:
                parsed = self._analysis(evaluation_id)
            except FileNotFoundError:
                return response
            response["analysis"] = {
                "source": parsed.get("source", {}),
                "encoding": parsed.get("encoding", ""),
                "metadata": parsed.get("metadata", {}),
                "summary": parsed.get("summary", {}),
                "noise": parsed.get("noise", {}),
                "issues": parsed.get("issues", []),
                "tasks": parsed.get("tasks", []),
                "automatic_metrics": parsed.get("automatic_metrics", {}),
                "sample_count": len(parsed.get("samples", [])),
            }
            return response
        row = self._row(evaluation_id)
        raw = json.loads(row["raw_json"])
        detail = raw.get("detail") if isinstance(raw.get("detail"), dict) else {}
        metrics = nested(detail, "metrics", "metrics", default={})
        summary_value = metrics.get("summary") if isinstance(metrics, dict) else None
        contributions_value = metrics.get("contributions") if isinstance(metrics, dict) else None
        summary = summary_value if isinstance(summary_value, dict) else {}
        contributions = contributions_value if isinstance(contributions_value, dict) else {}
        response: dict[str, Any] = {
            "evaluation": {
                "id": row["id"],
                "name": row["name"],
                "status": detail.get("taskStatus") or raw.get("taskStatus") or "UNKNOWN",
                "created_at": row["created_at"] or "",
                "start_time": detail.get("startTime") or raw.get("startTime") or "",
                "end_time": detail.get("endTime") or raw.get("endTime") or "",
                "duration": detail.get("duration") or raw.get("duration"),
                "model_name": detail.get("modelName") or raw.get("modelName") or "",
                "base_model": detail.get("baseModel") or detail.get("sourceModel") or raw.get("baseModel") or raw.get("sourceModel") or "",
                "model_id": detail.get("modelId") or raw.get("modelId") or "",
                "train_task_id": detail.get("trainTaskId") or raw.get("trainTaskId") or "",
                "score": summary.get("totalScore", detail.get("score", raw.get("score"))),
                "groups": {key: summary.get(key) for key in ("r0", "r1", "r2", "r3")},
                "contributions": contributions,
                "origin": "platform_sync",
                "manual": False,
                "source_label": "平台同步",
                "training_binding": self._binding_for(evaluation_id),
            },
            "analysis": None,
            "training_binding": self._binding_for(evaluation_id),
        }
        try:
            parsed = self._analysis(evaluation_id)
        except FileNotFoundError:
            return response
        response["analysis"] = {
            "source": parsed.get("source", {}),
            "encoding": parsed.get("encoding", ""),
            "metadata": parsed.get("metadata", {}),
            "summary": parsed.get("summary", {}),
            "noise": parsed.get("noise", {}),
            "issues": parsed.get("issues", []),
            "tasks": parsed.get("tasks", []),
            "automatic_metrics": parsed.get("automatic_metrics", {}),
            "sample_count": len(parsed.get("samples", [])),
        }
        return response

    def sample_catalog(self, evaluation_id: str, task: str = "", query: str = "") -> list[dict[str, Any]]:
        samples = self._analysis(evaluation_id).get("samples", [])
        needle = query.strip().lower()
        result = []
        for index, sample in enumerate(samples):
            if task and sample.get("task") != task:
                continue
            variants = sample.get("variants") if isinstance(sample.get("variants"), dict) else {}
            searchable = " ".join(
                [str(sample.get("id", "")), str(sample.get("task", "")), str(sample.get("prompt", ""))]
                + [str(value.get("input", "")) for value in variants.values() if isinstance(value, dict)]
            ).lower()
            if needle and needle not in searchable:
                continue
            result.append(
                {
                    "index": index,
                    "id": str(sample.get("id", "")),
                    "task": sample.get("task", ""),
                    "modes": sorted(variants),
                    "output_count": sum(
                        len(value.get("outputs", [])) for value in variants.values() if isinstance(value, dict)
                    ),
                    "preview": str(sample.get("prompt") or next(
                        (value.get("input", "") for value in variants.values() if isinstance(value, dict)), ""
                    ))[:180],
                }
            )
        return result

    def sample(self, evaluation_id: str, index: int) -> dict[str, Any]:
        samples = self._analysis(evaluation_id).get("samples", [])
        if index < 0 or index >= len(samples):
            raise FileNotFoundError("样本不存在")
        return samples[index]

    def log_lines(self, evaluation_id: str, offset: int, limit: int, query: str = "") -> dict[str, Any]:
        text = str(self._analysis(evaluation_id).get("filtered_text", ""))
        lines = text.splitlines()
        if query.strip():
            needle = query.strip().lower()
            lines = [line for line in lines if needle in line.lower()]
        offset = max(0, offset)
        limit = max(1, min(1000, limit))
        return {"offset": offset, "limit": limit, "total": len(lines), "lines": lines[offset:offset + limit]}


class AnalysisManager:
    def __init__(self, repository: EvaluationRepository) -> None:
        self.repository = repository
        self.lock = threading.RLock()
        self.fingerprint: tuple[tuple[str, int, int, int], ...] | None = None
        self.state: dict[str, Any] = {
            "running": False,
            "status": "idle",
            "total": 0,
            "completed": 0,
            "cached": 0,
            "parsed": 0,
            "current_id": "",
            "started_at": "",
            "finished_at": "",
            "errors": [],
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.state, ensure_ascii=False))

    def ensure_started(self) -> bool:
        inventory = self.repository.analysis_inventory()
        pending = [
            (evaluation_id, signature)
            for evaluation_id, signature in inventory
            if not self.repository._persistent_cache_valid(evaluation_id, signature)
        ]
        pending_ids = {evaluation_id for evaluation_id, _ in pending}
        with self.repository.lock:
            work = [
                (evaluation_id, signature)
                for evaluation_id, signature in inventory
                if not (
                    (cached := self.repository.cache.get(evaluation_id))
                    and cached.get("signature") == signature
                )
            ]
        cached_count = len(inventory) - len(pending)
        fingerprint = tuple(
            (evaluation_id, signature["mtime_ns"], signature["size"], signature["parser_version"])
            for evaluation_id, signature in inventory
        )
        with self.lock:
            if self.state["running"] or fingerprint == self.fingerprint:
                return False
            self.fingerprint = fingerprint
            self.state = {
                "running": bool(work),
                "status": "running" if work else "ready",
                "total": len(inventory),
                "completed": len(inventory) - len(work),
                "cached": cached_count,
                "parsed": 0,
                "loaded": 0,
                "current_id": "",
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "finished_at": "" if work else datetime.now().astimezone().isoformat(timespec="seconds"),
                "errors": [],
            }
        if not work:
            return False
        threading.Thread(
            target=self._worker,
            args=(work, pending_ids),
            name="evaluation-analysis-cache",
            daemon=True,
        ).start()
        return True

    def _worker(
        self,
        inventory: list[tuple[str, dict[str, int]]],
        pending_ids: set[str],
    ) -> None:
        for evaluation_id, signature in inventory:
            with self.lock:
                self.state["current_id"] = evaluation_id
            try:
                self.repository._analysis(evaluation_id)
                with self.lock:
                    field = "parsed" if evaluation_id in pending_ids else "loaded"
                    self.state[field] += 1
            except Exception as error:
                with self.lock:
                    self.state["errors"].append({"id": evaluation_id, "error": str(error)[:1000]})
            finally:
                with self.lock:
                    self.state["completed"] += 1
        with self.lock:
            if self.state["errors"]:
                self.fingerprint = None
            self.state.update(
                running=False,
                status="error" if self.state["errors"] else "ready",
                current_id="",
                finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )


class SyncManager:
    def __init__(self, config: Mapping[str, Any], on_success: Any | None = None) -> None:
        self.config = config
        self.on_success = on_success
        self.lock = threading.RLock()
        self.state: dict[str, Any] = {
            "running": False,
            "started_at": "",
            "finished_at": "",
            "status": "idle",
            "message": "",
            "result": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.state, ensure_ascii=False))

    def start(self) -> bool:
        with self.lock:
            if self.state["running"]:
                return False
            self.state = {
                "running": True,
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "finished_at": "",
                "status": "running",
                "message": "正在同步平台元数据和评测日志",
                "result": None,
            }
        threading.Thread(target=self._worker, name="streamlake-sync", daemon=True).start()
        return True

    def _worker(self) -> None:
        command = [
            sys.executable,
            self.config["skill_script"],
            "--output-dir",
            self.config["output_dir"],
            "sync",
        ]
        try:
            environment = os.environ.copy()
            environment.update(
                STREAMLAKE_COOKIE_FILE=self.config["cookie_file"],
                STREAMLAKE_PROJECT_ID_FILE=self.config["project_id_file"],
            )
            completed = subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
            if completed.returncode != 0:
                message = (completed.stderr or completed.stdout or "同步失败").strip()[-MAX_SYNC_MESSAGE:]
                raise RuntimeError(message)
            result = json.loads(completed.stdout)
            with self.lock:
                self.state.update(status="success", message="同步完成", result=result)
            if self.on_success is not None:
                self.on_success()
        except Exception as error:
            with self.lock:
                self.state.update(status="error", message=str(error)[:MAX_SYNC_MESSAGE], result=None)
        finally:
            with self.lock:
                self.state.update(
                    running=False,
                    finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                )


class EvalMonitorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, config: dict[str, Any]) -> None:
        super().__init__(address, handler)
        self.config = config
        self.token = secrets.token_urlsafe(32)
        parser = load_parser_module(Path(config["parser_dir"]))
        self.repository = EvaluationRepository(Path(config["output_dir"]), parser)
        self.analysis_manager = AnalysisManager(self.repository)
        self.sync_manager = SyncManager(config, self.analysis_manager.ensure_started)
        self.training_module = load_training_module(HERE)
        self.training_upload_lock = threading.Lock()
        self.training_upload_registry_path = Path(config["training_upload_registry_file"])
        self.training_config = load_training_config(
            self.training_module,
            Path(config["training_config_file"]),
            self.training_upload_registry_path,
        )
        if self.training_config is not None:
            self.training_config["_monitor_config_file"] = str(Path(config["training_config_file"]).resolve())
            self.training_config["_monitor_token"] = self.token
        self.analysis_manager.ensure_started()

    def record_training_upload(self, experiment_id: str, checkpoint_name: str, record: dict[str, Any]) -> None:
        if self.training_module is None or self.training_config is None:
            raise RuntimeError("训练 Monitor 未启用")
        registry = self.training_config["_upload_registry"]
        registry["uploads"].setdefault(experiment_id, {})[checkpoint_name] = record
        self.training_module.save_upload_registry(self.training_upload_registry_path, registry)

    def rotate_training_upload_nonce(self) -> None:
        if self.training_config is not None:
            self.training_config["_upload_nonce"] = secrets.token_urlsafe(24)

    def training_tasks(self) -> list[dict[str, Any]]:
        if self.training_module is None or self.training_config is None:
            return []
        tasks: list[dict[str, Any]] = []
        now = time.time()
        for target in self.training_module.discover_targets(self.training_config):
            try:
                experiment = self.training_module.build_experiment(target, self.training_config, now)
            except Exception:
                continue
            tasks.append(
                {
                    "id": experiment.get("id", target.get("id", "")),
                    "label": experiment.get("label", target.get("label", "")),
                    "run_id": experiment.get("run_id", ""),
                    "output_dir": experiment.get("output_dir", ""),
                    "status": experiment.get("status", ""),
                    "modified_at": experiment.get("modified_at"),
                    "run_manifest": experiment.get("run_manifest", {"available": False}),
                }
            )
        tasks.sort(key=lambda item: item.get("modified_at") or 0, reverse=True)
        return tasks

    @staticmethod
    def training_binding_for_task(task: Mapping[str, Any]) -> dict[str, Any]:
        task_id = str(task.get("id", "")).strip()
        if not task_id:
            raise ValueError("训练任务缺少有效 ID")
        manifest = task.get("run_manifest") if isinstance(task.get("run_manifest"), dict) else {"available": False}
        return {
            "id": task_id,
            "label": task.get("label", ""),
            "run_id": task.get("run_id", ""),
            "output_dir": task.get("output_dir", ""),
            "manifest": manifest,
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "EvalLogMonitor/1.0"

    @property
    def app(self) -> EvalMonitorServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"{self.log_date_time_string()} {self.address_string()} {fmt % args}\n")

    def _headers(self, content_type: str, length: int, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()

    def _redirect(self, location: str, status: int = 302) -> None:
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _json(self, value: Any, status: int = 200) -> None:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._headers("application/json; charset=utf-8", len(raw), status)
        self.wfile.write(raw)

    def _attachment(self, path: Path, filename: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("请求长度无效") from error
        if length <= 0 or length > MAX_CONFIG_BODY:
            raise ValueError("请求内容为空或过大")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("请求必须是 JSON 对象")
        return value

    def _read_small_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("请求长度无效") from error
        if length <= 0 or length > 4096:
            raise ValueError("请求内容为空或过大")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("请求必须是 JSON 对象")
        return value

    def _read_manual_upload(self) -> tuple[Path, str]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("上传长度无效") from error
        if length <= 0:
            raise ValueError("上传文件为空")
        filename = Path(self.headers.get("X-Log-Filename", "evaluation.log")).name
        if not filename or filename in {".", ".."}:
            filename = "evaluation.log"
        descriptor, temporary_name = tempfile.mkstemp(prefix="eval-upload-", suffix=".tmp")
        temporary = Path(temporary_name)
        try:
            remaining = length
            with os.fdopen(descriptor, "wb") as handle:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("上传文件提前结束")
                    handle.write(chunk)
                    remaining -= len(chunk)
            os.chmod(temporary, 0o600)
            return temporary, filename
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _require_token(self) -> bool:
        if secrets.compare_digest(self.headers.get("X-Eval-Monitor-Token", ""), self.app.token):
            return True
        self._error(HTTPStatus.FORBIDDEN, "页面令牌无效，请刷新后重试")
        return False

    def _training_ready(self) -> bool:
        if self.app.training_module is not None and self.app.training_config is not None:
            return True
        self._error(HTTPStatus.NOT_FOUND, "训练 Monitor 未部署或配置缺失")
        return False

    def _training_upload_checkpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        module = self.app.training_module
        config = self.app.training_config
        if module is None or config is None:
            raise FileNotFoundError("训练 Monitor 未部署或配置缺失")
        allowed_keys = {"experiment_id", "checkpoint_name"}
        if set(payload) - allowed_keys:
            raise ValueError("unsupported upload parameters")
        experiment_id = payload.get("experiment_id")
        checkpoint_name = payload.get("checkpoint_name")
        if not isinstance(experiment_id, str) or not isinstance(checkpoint_name, str):
            raise ValueError("experiment_id and checkpoint_name are required")
        target = next((item for item in module.discover_targets(config) if item["id"] == experiment_id), None)
        if target is None:
            raise ValueError("selected experiment is not available")
        if not target.get("enable_upload"):
            raise ValueError("upload is disabled for this experiment")
        existing = config.get("_upload_registry", {}).get("uploads", {}).get(experiment_id, {}).get(checkpoint_name)
        if isinstance(existing, dict):
            raise module.UploadConflictError(
                f"checkpoint was already uploaded: {existing.get('repo_url', existing.get('repo_id'))}"
            )
        required = ("hub_endpoint", "hub_owner", "hub_repo_prefix", "base_model_id", "config_path")
        if any(not target.get(field) for field in required):
            raise ValueError("upload configuration is incomplete")
        output_dir = Path(target["output_dir"]).resolve()
        checkpoint_dir = (output_dir / checkpoint_name).resolve()
        match = module.CHECKPOINT_RE.fullmatch(checkpoint_name)
        if not match or checkpoint_dir.parent != output_dir or not checkpoint_dir.is_dir():
            raise ValueError("selected checkpoint is not available")
        run_id = str(target.get("run_id") or output_dir.name)
        if not module.RUN_ID_RE.fullmatch(run_id):
            raise ValueError("run_id must contain only letters, numbers, underscores, and hyphens")
        config_path = Path(str(target["config_path"])).resolve()
        if not config_path.is_file():
            raise ValueError("training config file is not available")
        step = int(match.group(1))
        repo_id = module.evaluation_repo_id(target, run_id, step)
        with tempfile.TemporaryDirectory(prefix="train-monitor-upload-") as temporary:
            staged_dir = Path(temporary)
            files, total_bytes = module.stage_checkpoint_upload(
                checkpoint_dir=checkpoint_dir,
                config_path=config_path,
                staged_dir=staged_dir,
                base_model_id=str(target["base_model_id"]),
                run_id=run_id,
                step=step,
            )
            try:
                from huggingface_hub import HfApi
            except ImportError as error:
                raise RuntimeError("huggingface_hub is not installed in the monitor environment") from error
            api = HfApi(endpoint=str(target["hub_endpoint"]), token=module.huggingface_token(config))
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
                    module.update_evaluation_index(
                        api,
                        str(target["hub_index_repo_id"]),
                        {"run_id": run_id, "step": step, "repo_id": repo_id},
                    )
                except Exception as error:
                    index_warning = "evaluation repository was uploaded, but the archive index was not updated"
                    sys.stderr.write(f"Evaluation index update failed: {error!r}\n")
        commit_id = getattr(result, "oid", None) or getattr(result, "commit_id", None)
        upload_record = {
            "repo_id": repo_id,
            "repo_url": f"https://huggingface.co/{repo_id}",
            "run_id": run_id,
            "step": step,
            "uploaded_at": datetime.now().timestamp(),
        }
        self.app.record_training_upload(experiment_id, checkpoint_name, upload_record)
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

    def _handle_training_upload_payload(self, payload: dict[str, Any]) -> None:
        if not self._training_ready():
            return
        if not self.app.training_upload_lock.acquire(blocking=False):
            self._error(HTTPStatus.CONFLICT, "another upload is already in progress")
            return
        conflict_error = self.app.training_module.UploadConflictError  # type: ignore[union-attr]
        try:
            result = self._training_upload_checkpoint(payload)
            self.app.rotate_training_upload_nonce()
            self._json(result)
        except Exception as error:
            if isinstance(error, conflict_error):
                self._error(HTTPStatus.CONFLICT, str(error))
            elif isinstance(error, ValueError):
                self._error(HTTPStatus.BAD_REQUEST, str(error))
            else:
                sys.stderr.write(f"Upload failed: {error!r}\n")
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "upload failed; inspect monitor_server.log")
        finally:
            self.app.training_upload_lock.release()

    def _training_upload_from_query(self, query: dict[str, list[str]]) -> None:
        if not self._training_ready():
            return
        allowed = {"experiment_id", "checkpoint_name", "nonce"}
        if set(query) != allowed or any(len(values) != 1 for values in query.values()):
            self._error(HTTPStatus.BAD_REQUEST, "invalid upload query")
            return
        expected = self.app.training_config.get("_upload_nonce", "") if self.app.training_config else ""
        if not isinstance(expected, str) or not secrets.compare_digest(query["nonce"][0], expected):
            self._error(HTTPStatus.FORBIDDEN, "invalid or expired upload nonce")
            return
        self._handle_training_upload_payload(
            {"experiment_id": query["experiment_id"][0], "checkpoint_name": query["checkpoint_name"][0]}
        )

    def _config_status(self) -> dict[str, Any]:
        cookie_path = Path(self.app.config["cookie_file"])
        project_path = Path(self.app.config["project_id_file"])
        project_id = ""
        if project_path.is_file():
            project_id = project_path.read_text(encoding="utf-8").strip()
        configured = cookie_path.is_file() and bool(project_id)
        sync = self.app.sync_manager.snapshot()
        sync_message = str(sync.get("message", ""))
        expired = bool(configured and sync.get("status") == "error" and re.search(r"(?:401|403|cookie|auth|登录|认证|expired)", sync_message, re.IGNORECASE))
        return {
            "configured": configured,
            "cookie_configured": cookie_path.is_file(),
            "cookie_status": "expired" if expired else "configured" if configured else "missing",
            "cookie_updated_at": datetime.fromtimestamp(cookie_path.stat().st_mtime).astimezone().isoformat(timespec="seconds") if cookie_path.is_file() else "",
            "project_id": project_id,
        }

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._redirect("/train/")
                return
            if path in ("/eval", "/eval/", "/eval/index.html"):
                raw = HTML_PATH.read_bytes()
                self._headers("text/html; charset=utf-8", len(raw))
                self.wfile.write(raw)
                return
            if path in ("/train", "/train/", "/train/index.html"):
                raw = TRAINING_HTML_PATH.read_bytes()
                self._headers("text/html; charset=utf-8", len(raw))
                self.wfile.write(raw)
                return
            if path == "/api/health":
                self._json({"status": "ok"})
                return
            if path == "/train/api/health":
                self._json({
                    "status": "ok" if self.app.training_config is not None else "unconfigured",
                    "monitor": "training",
                })
                return
            if path == "/train/api/snapshot":
                if not self._training_ready():
                    return
                include_hidden = query.get("include_hidden", ["0"])[0].lower() in {"1", "true", "yes"}
                self._json(self.app.training_module.build_snapshot(self.app.training_config, include_hidden=include_hidden))  # type: ignore[union-attr,arg-type]
                return
            if path == "/train/api/upload":
                try:
                    upload_query = parse_qs(
                        parsed.query,
                        keep_blank_values=True,
                        strict_parsing=True,
                        max_num_fields=3,
                    )
                except ValueError:
                    self._error(HTTPStatus.BAD_REQUEST, "invalid upload query")
                    return
                self._training_upload_from_query(upload_query)
                return
            if path == "/api/snapshot":
                state_path = Path(self.app.config["output_dir"]) / "sync_state.json"
                sync_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
                self._json({
                    "token": self.app.token,
                    "config": self._config_status(),
                    "sync": self.app.sync_manager.snapshot(),
                    "analysis_cache": self.app.analysis_manager.snapshot(),
                    "sync_state": sync_state,
                    "runtime": {
                        "output_dir": self.app.config["output_dir"],
                        "training_config_file": self.app.config.get("training_config_file", ""),
                        "training_monitor": self.app.training_config is not None,
                    },
                    "evaluations": self.app.repository.list_evaluations(),
                })
                return
            if path == "/api/rankings":
                self._json({"rows": self.app.repository.ranking_rows()})
                return
            if path == "/api/training-tasks":
                self._json({"tasks": self.app.training_tasks()})
                return
            if path == "/api/manual-evaluations":
                self._json({"rows": [row for row in self.app.repository.list_evaluations() if row.get("manual")]})
                return
            match = re.fullmatch(r"/api/evaluations/([^/]+)", path)
            if match:
                self._json(self.app.repository.detail(unquote(match.group(1))))
                return
            match = re.fullmatch(r"/api/evaluations/([^/]+)/samples", path)
            if match:
                self._json({"samples": self.app.repository.sample_catalog(
                    unquote(match.group(1)),
                    query.get("task", [""])[0],
                    query.get("q", [""])[0],
                )})
                return
            match = re.fullmatch(r"/api/evaluations/([^/]+)/samples/(\d+)", path)
            if match:
                self._json({"sample": self.app.repository.sample(unquote(match.group(1)), int(match.group(2)))})
                return
            match = re.fullmatch(r"/api/evaluations/([^/]+)/download-log", path)
            if match:
                evaluation_id = unquote(match.group(1))
                log_path = self.app.repository.download_log_path(evaluation_id)
                self._attachment(log_path, f"{evaluation_id}.log")
                return
            match = re.fullmatch(r"/api/evaluations/([^/]+)/log", path)
            if match:
                self._json(self.app.repository.log_lines(
                    unquote(match.group(1)),
                    int(query.get("offset", ["0"])[0]),
                    int(query.get("limit", ["400"])[0]),
                    query.get("q", [""])[0],
                ))
                return
            self._error(HTTPStatus.NOT_FOUND, "接口不存在")
        except FileNotFoundError as error:
            self._error(HTTPStatus.NOT_FOUND, str(error))
        except (ValueError, json.JSONDecodeError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"服务器错误：{error}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/train/api/upload":
            try:
                self._handle_training_upload_payload(self._read_small_json())
            except (ValueError, json.JSONDecodeError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        if not self._require_token():
            return
        try:
            if parsed.path == "/api/config":
                payload = self._read_json()
                cookie, project_id = parse_request_headers(str(payload.get("headers", "")))
                atomic_write_secret(Path(self.app.config["cookie_file"]), cookie)
                atomic_write_secret(Path(self.app.config["project_id_file"]), project_id)
                self._json({"status": "saved", "config": self._config_status()})
                return
            if parsed.path == "/api/manual-logs":
                temporary: Path | None = None
                try:
                    temporary, filename = self._read_manual_upload()
                    task_id = self.headers.get("X-Training-Task-Id", "").strip()
                    binding = None
                    if task_id:
                        task = next((item for item in self.app.training_tasks() if item.get("id") == task_id), None)
                        if task is None:
                            raise ValueError("指定的训练任务不存在或尚未被训练 Monitor 发现")
                        binding = self.app.training_binding_for_task(task)
                    result = self.app.repository.add_manual_log(temporary, filename, binding)
                    self.app.analysis_manager.ensure_started()
                    self._json({"status": "saved", "evaluation": result}, HTTPStatus.CREATED)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                return
            if parsed.path == "/api/evaluation-bindings":
                payload = self._read_small_json()
                allowed = {"evaluation_id", "training_task_id"}
                if set(payload) - allowed:
                    raise ValueError("包含不支持的绑定字段")
                evaluation_id = str(payload.get("evaluation_id", "")).strip()
                if not evaluation_id:
                    raise ValueError("缺少 evaluation_id")
                task_id = str(payload.get("training_task_id", "")).strip()
                binding = None
                if task_id:
                    task = next((item for item in self.app.training_tasks() if item.get("id") == task_id), None)
                    if task is None:
                        raise ValueError("指定的训练任务不存在或尚未被训练 Monitor 发现")
                    binding = self.app.training_binding_for_task(task)
                saved = self.app.repository.bind_training(evaluation_id, binding)
                self._json({"status": "saved", "training_binding": saved})
                return
            if parsed.path == "/train/api/huggingface":
                if not self._training_ready():
                    return
                status = self.app.training_module.save_huggingface_binding(  # type: ignore[union-attr]
                    self.app.training_config,
                    self._read_small_json(),
                )
                self._json({"status": "saved", "huggingface": status})
                return
            if parsed.path == "/train/api/experiment-visibility":
                if not self._training_ready():
                    return
                payload = self._read_small_json()
                allowed = {"experiment_id", "hidden"}
                if set(payload) - allowed:
                    raise ValueError("包含不支持的训练记录可见性字段")
                experiment_id = str(payload.get("experiment_id", "")).strip()
                hidden = payload.get("hidden")
                if not experiment_id:
                    raise ValueError("缺少 experiment_id")
                if not isinstance(hidden, bool):
                    raise ValueError("hidden 必须是布尔值")
                visibility = self.app.training_module.set_experiment_visibility(  # type: ignore[union-attr]
                    self.app.training_config,
                    experiment_id,
                    hidden,
                )
                self._json({"status": "saved", "visibility": visibility})
                return
            if parsed.path == "/api/sync":
                if not self._config_status()["configured"]:
                    self._error(HTTPStatus.CONFLICT, "请先配置 Cookie 和 project ID")
                    return
                if not self.app.sync_manager.start():
                    self._error(HTTPStatus.CONFLICT, "同步正在进行")
                    return
                self._json({"status": "started"}, HTTPStatus.ACCEPTED)
                return
            self._error(HTTPStatus.NOT_FOUND, "接口不存在")
        except (ValueError, json.JSONDecodeError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"服务器错误：{error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    server = EvalMonitorServer((config["bind_host"], config["port"]), Handler, config)
    print(f"Evaluation monitor listening on http://{config['bind_host']}:{config['port']}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
