#!/usr/bin/env python3
"""Loopback-only StreamLake evaluation log monitor."""

from __future__ import annotations

import argparse
import gzip
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
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlparse


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "monitor_config.json"
HTML_PATH = HERE / "dashboard.html"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
EVALUATION_ID_RE = re.compile(r"eval-task-[A-Za-z0-9-]+")
PROJECT_ID_RE = re.compile(r"(?:^|/)(proj-[A-Za-z0-9-]+)(?:/|$)")
MAX_CONFIG_BODY = 128 * 1024
MAX_SYNC_MESSAGE = 8000
ANALYSIS_CACHE_VERSION = 1


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
        self.analysis_cache_dir = output_dir / "analysis_cache"
        self.parser = parser_module
        self.cache: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.analysis_locks: dict[str, threading.Lock] = {}

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
        return self.logs_dir / evaluation_id / "evaluation.log"

    def _cache_paths(self, evaluation_id: str) -> tuple[Path, Path]:
        return (
            self.analysis_cache_dir / f"{evaluation_id}.json.gz",
            self.analysis_cache_dir / f"{evaluation_id}.meta.json",
        )

    def _signature(self, path: Path) -> dict[str, int]:
        stat = path.stat()
        return {
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "parser_version": int(getattr(self.parser, "PARSER_VERSION", 1)),
        }

    def analysis_inventory(self) -> list[tuple[str, dict[str, int]]]:
        if not self.logs_dir.is_dir():
            return []
        result = []
        for path in sorted(self.logs_dir.glob("*/evaluation.log")):
            evaluation_id = path.parent.name
            if EVALUATION_ID_RE.fullmatch(evaluation_id):
                result.append((evaluation_id, self._signature(path)))
        return result

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
        finally:
            temporary.unlink(missing_ok=True)

    def list_evaluations(self) -> list[dict[str, Any]]:
        if not self.db_path.is_file():
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,name,status,created_at,updated_at,raw_json FROM experiments "
                "WHERE experiment_type='evaluation' ORDER BY created_at DESC,id"
            ).fetchall()
        result = []
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
                    "log": {
                        "available": log_path.is_file(),
                        "size_bytes": log_path.stat().st_size if log_path.is_file() else 0,
                    },
                }
            )
        return result

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
            },
            "analysis": None,
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
                "running": bool(pending),
                "status": "running" if pending else "ready",
                "total": len(inventory),
                "completed": cached_count,
                "cached": cached_count,
                "parsed": 0,
                "current_id": "",
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "finished_at": "" if pending else datetime.now().astimezone().isoformat(timespec="seconds"),
                "errors": [],
            }
        if not pending:
            return False
        threading.Thread(target=self._worker, args=(pending,), name="evaluation-analysis-cache", daemon=True).start()
        return True

    def _worker(self, inventory: list[tuple[str, dict[str, int]]]) -> None:
        for evaluation_id, signature in inventory:
            with self.lock:
                self.state["current_id"] = evaluation_id
            try:
                self.repository._analysis(evaluation_id)
                with self.lock:
                    self.state["parsed"] += 1
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

    def _json(self, value: Any, status: int = 200) -> None:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._headers("application/json; charset=utf-8", len(raw), status)
        self.wfile.write(raw)

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

    def _require_token(self) -> bool:
        if secrets.compare_digest(self.headers.get("X-Eval-Monitor-Token", ""), self.app.token):
            return True
        self._error(HTTPStatus.FORBIDDEN, "页面令牌无效，请刷新后重试")
        return False

    def _config_status(self) -> dict[str, Any]:
        cookie_path = Path(self.app.config["cookie_file"])
        project_path = Path(self.app.config["project_id_file"])
        project_id = ""
        if project_path.is_file():
            project_id = project_path.read_text(encoding="utf-8").strip()
        return {
            "configured": cookie_path.is_file() and bool(project_id),
            "cookie_configured": cookie_path.is_file(),
            "cookie_updated_at": datetime.fromtimestamp(cookie_path.stat().st_mtime).astimezone().isoformat(timespec="seconds") if cookie_path.is_file() else "",
            "project_id": project_id,
        }

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/":
                raw = HTML_PATH.read_bytes()
                self._headers("text/html; charset=utf-8", len(raw))
                self.wfile.write(raw)
                return
            if path == "/api/health":
                self._json({"status": "ok"})
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
                    "evaluations": self.app.repository.list_evaluations(),
                })
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
        if not self._require_token():
            return
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/config":
                payload = self._read_json()
                cookie, project_id = parse_request_headers(str(payload.get("headers", "")))
                atomic_write_secret(Path(self.app.config["cookie_file"]), cookie)
                atomic_write_secret(Path(self.app.config["project_id_file"]), project_id)
                self._json({"status": "saved", "config": self._config_status()})
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
