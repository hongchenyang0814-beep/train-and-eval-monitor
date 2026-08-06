#!/usr/bin/env python3
"""Read-only StreamLake experiment synchronization and local comparison CLI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_ORIGIN = "https://console.streamlake.com/api/console/open-api"
# User-required retention boundary. Synchronization must never fetch or retain older records.
MIN_SYNC_CREATED_AT = datetime(2026, 8, 1, tzinfo=timezone.utc)
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_STORED_STRING_BYTES = 64 * 1024
METRIC_QUERY_BATCH_SIZE = 16
ALLOWED_STREAMLAKE_HOST = "console.streamlake.com"
ALLOWED_EVALUATION_LOG_DOMAIN = "safetyimg.com"
ALLOWED_EVALUATION_LOG_DOMAINS = ("safetyimg.com", "yximgs.com")
READ_ONLY_POST_PATHS = {
    "/api/customized/commercial/v1/train-task/list",
    "/api/customized/commercial/v1/train-task/metric-query",
    "/api/customized/commercial/v1/competition-eval-task/list",
}
READ_ONLY_GET_PATTERNS = (
    re.compile(r"^/api/customized/commercial/v1/train-task/[^/]+$"),
    re.compile(r"^/api/customized/commercial/v1/train-task/analysis/dashboard$"),
    re.compile(r"^/api/customized/commercial/v1/competition-eval-task/[^/]+(?:/output)?$"),
)
JWT_RE = re.compile(r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
SECRET_KEY_RE = re.compile(
    r"(^|[-_])(authorization|cookie|token|secret|password|signature|credential|access[-_]?key|api[-_]?key)($|[-_])",
    re.IGNORECASE,
)
INLINE_SECRET_RE = re.compile(
    r"(?i)(authorization|cookie|access[_ -]?token|refresh[_ -]?token|api[_ -]?key)\s*[:=]\s*[^\s,;]+"
)
SIGNED_QUERY_KEYS = {
    "authorization",
    "credential",
    "expires",
    "policy",
    "signature",
    "token",
    "x-amz-algorithm",
    "x-amz-credential",
    "x-amz-date",
    "x-amz-expires",
    "x-amz-security-token",
    "x-amz-signature",
    "x-amz-signedheaders",
}
LARGE_BODY_RE = re.compile(
    r"(^|[-_])(bytes|binary|blob|body|content|prediction|predictions|samples|checkpoint[-_]?data|dataset[-_]?data|file[-_]?content|weights)($|[-_])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EndpointSpec:
    path: str
    items_path: str
    method: str = "GET"
    pagination: str = "page"
    pagination_location: str = "query"
    page_param: str = "page"
    page_size_param: str = "pageSize"
    total_path: str | None = None
    cursor_param: str = "cursor"
    next_cursor_path: str | None = None
    static_query: Mapping[str, Any] | None = None
    static_body: Mapping[str, Any] | None = None
    id_fields: tuple[str, ...] = ("id", "experimentId", "experiment_id")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EndpointSpec":
        allowed = {field for field in cls.__dataclass_fields__}
        return cls(**{key: val for key, val in value.items() if key in allowed})


@dataclass(frozen=True)
class SyncResult:
    experiment_count: int
    metric_count: int
    parameter_count: int
    source_counts: Mapping[str, int]
    error_count: int = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_datetime(value: Any) -> datetime | None:
    """Parse common StreamLake timestamps as timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        try:
            parsed = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            for format_string in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    parsed = datetime.strptime(text, format_string)
                    break
                except ValueError:
                    continue
            else:
                return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_since(value: str | None) -> datetime:
    if value is None:
        return MIN_SYNC_CREATED_AT
    parsed = parse_datetime(value)
    if parsed is None:
        raise ValueError(f"Invalid --since timestamp: {value!r}")
    if parsed < MIN_SYNC_CREATED_AT:
        raise ValueError("StreamLake sync is restricted to records created on or after 2026-08-01T00:00:00Z")
    return parsed


def default_cookie_file(environ: Mapping[str, str] | None = None) -> Path:
    environment = os.environ if environ is None else environ
    override = environment.get("STREAMLAKE_COOKIE_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    home = environment.get("HOME", "").strip()
    return (Path(home) if home else Path.home()) / ".local" / "share" / "streamlake-eval-monitor" / "config" / "cookie"


def default_project_id_file(environ: Mapping[str, str] | None = None) -> Path:
    environment = os.environ if environ is None else environ
    override = environment.get("STREAMLAKE_PROJECT_ID_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    home = environment.get("HOME", "").strip()
    return (Path(home) if home else Path.home()) / ".local" / "share" / "streamlake-eval-monitor" / "config" / "project_id"


def resolve_project_id(cli_value: str | None, environ: Mapping[str, str] | None = None) -> str:
    environment = os.environ if environ is None else environ
    value = (cli_value or environment.get("STREAMLAKE_PROJECT_ID", "")).strip()
    if not value:
        project_id_file = default_project_id_file(environment)
        if project_id_file.is_file():
            value = project_id_file.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(
            "StreamLake project ID is required; use --project-id, STREAMLAKE_PROJECT_ID, or the Monitor config/project_id file"
        )
    return value


def load_cookie(cookie_file: Path | str | None = None, environ: Mapping[str, str] | None = None) -> str:
    environment = os.environ if environ is None else environ
    from_environment = environment.get("STREAMLAKE_COOKIE", "").strip()
    if from_environment:
        return from_environment
    path = Path(cookie_file) if cookie_file is not None else default_cookie_file(environment)
    if not path.is_file():
        raise FileNotFoundError(f"Cookie file not found: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(f"Cookie file is too permissive; run chmod 600 {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Cookie file is empty: {path}")
    return value


def _sanitize_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return value
    if parsed.scheme not in {"http", "https"} or not parsed.query:
        return value
    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    cleaned = [(key, val) for key, val in pairs if key.lower() not in SIGNED_QUERY_KEYS]
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(cleaned), parsed.fragment)
    )


def sanitize(value: Any, key: str = "") -> Any:
    """Return a JSON-safe value with credentials and large bodies removed."""
    normalized_key = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
    if SECRET_KEY_RE.search(normalized_key):
        return "[REDACTED]"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"_omitted": "binary", "size": len(value)}
    if LARGE_BODY_RE.search(key) and isinstance(value, (str, list, dict)):
        return {"_omitted": "large-body"}
    if isinstance(value, Mapping):
        return {str(child_key): sanitize(child_value, str(child_key)) for child_key, child_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item, key) for item in value]
    if isinstance(value, str):
        if JWT_RE.fullmatch(value.strip()):
            return "[REDACTED]"
        if len(value.encode("utf-8")) > MAX_STORED_STRING_BYTES:
            return {"_omitted": "large-string", "size": len(value.encode("utf-8"))}
        return _sanitize_url(INLINE_SECRET_RE.sub(r"\1: [REDACTED]", value))
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def get_path(value: Any, dotted_path: str | None, default: Any = None) -> Any:
    if not dotted_path:
        return default
    current = value
    for part in dotted_path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return default
    return current


class StreamLakeClient:
    def __init__(
        self,
        cookie: str,
        origin: str = DEFAULT_ORIGIN,
        timeout: float = 30.0,
        max_bytes: int = MAX_JSON_BYTES,
        retries: int = 4,
    ) -> None:
        parsed_origin = urllib.parse.urlsplit(origin)
        if parsed_origin.scheme != "https" or parsed_origin.hostname != ALLOWED_STREAMLAKE_HOST:
            raise ValueError(f"Origin must use the StreamLake host {ALLOWED_STREAMLAKE_HOST} over HTTPS")
        self._cookie = cookie
        self.origin = origin.rstrip("/")
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.retries = retries

    def request_json(
        self,
        method: str,
        path: str,
        query: Mapping[str, Any] | None = None,
        body: Any = None,
    ) -> Any:
        normalized_method = method.upper()
        if normalized_method not in {"GET", "POST"}:
            raise ValueError("StreamLake client permits read-only GET and query-only POST requests")
        if normalized_method == "GET" and body is not None:
            raise ValueError("GET requests cannot contain a JSON body")
        parsed_path = urllib.parse.urlsplit(path)
        if parsed_path.scheme or parsed_path.netloc:
            raise ValueError("StreamLake API paths must be relative")
        requested_path = parsed_path.path
        if normalized_method == "POST" and requested_path not in READ_ONLY_POST_PATHS:
            raise ValueError(f"POST endpoint is not in the read-only allowlist: {requested_path}")
        if normalized_method == "GET" and not any(pattern.fullmatch(requested_path) for pattern in READ_ONLY_GET_PATTERNS):
            raise ValueError(f"GET endpoint is not in the read-only allowlist: {requested_path}")
        url = f"{self.origin}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query, doseq=True)}"
        headers = {
            "Accept": "application/json",
            "Cookie": self._cookie,
            "Referer": f"{self.origin}/",
            "User-Agent": "streamlake-experiment-analyst/1.0",
            "X-Requested-With": "XMLHttpRequest",
            "open-api-product": "WANQING",
            "Accept-Language": "zh-CN",
        }
        encoded_body = None
        if normalized_method == "POST":
            headers["Content-Type"] = "application/json"
            encoded_body = json.dumps(body or {}, ensure_ascii=False).encode("utf-8")
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(url, data=encoded_body, headers=headers, method=normalized_method)
            try:
                opener = urllib.request.build_opener(_SameHostRedirectHandler(ALLOWED_STREAMLAKE_HOST))
                with opener.open(request, timeout=self.timeout) as response:
                    final_url = response.geturl()
                    if urllib.parse.urlsplit(final_url).hostname != ALLOWED_STREAMLAKE_HOST:
                        raise PermissionError("Refused to send credentials outside StreamLake")
                    content_type = response.headers.get_content_type()
                    if "login" in urllib.parse.urlsplit(final_url).path.lower():
                        raise PermissionError("StreamLake authentication expired; refresh the local Cookie file")
                    if content_type not in {"application/json", "text/json"} and not content_type.endswith("+json"):
                        raise ValueError(f"Expected JSON response, received {content_type}")
                    raw = response.read(self.max_bytes + 1)
                    if len(raw) > self.max_bytes:
                        raise ValueError("JSON response exceeded the safe size limit")
                    return json.loads(raw.decode(response.headers.get_content_charset() or "utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 403}:
                    raise PermissionError("StreamLake authentication failed; refresh the local Cookie file") from None
                if exc.code not in {408, 429} and exc.code < 500:
                    raise RuntimeError(f"StreamLake HTTP {exc.code}") from None
                if attempt >= self.retries:
                    raise RuntimeError(f"StreamLake HTTP {exc.code} after retries") from None
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = min(float(retry_after), 30.0) if retry_after and retry_after.isdigit() else min(2**attempt, 16)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self.retries:
                    raise RuntimeError("StreamLake request failed after retries") from exc
                time.sleep(min(2**attempt, 16))
        raise RuntimeError("unreachable")


class _SameHostRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_host: str) -> None:
        self.allowed_host = allowed_host

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname != self.allowed_host:
            raise PermissionError("Refused cross-origin redirect carrying StreamLake credentials")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_evaluation_log_url(value: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(value)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        raise ValueError("Evaluation log URL must use HTTPS")
    if not any(hostname == domain or hostname.endswith(f".{domain}") for domain in ALLOWED_EVALUATION_LOG_DOMAINS):
        raise ValueError("Evaluation log URL must use safetyimg.com or yximgs.com")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Evaluation log URL contains unsupported credentials or fragment")
    if not urllib.parse.unquote(parsed.path).lower().endswith(".log"):
        raise ValueError("Evaluation output is not a .log file")
    return parsed


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
    return size_bytes, digest.hexdigest()


def download_evaluation_log_url(
    evaluation_id: str,
    log_url: str,
    log_dir: Path | str,
    overwrite: bool = False,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    parsed_url = validate_evaluation_log_url(log_url)
    destination_dir = Path(log_dir).expanduser() / evaluation_id
    destination = destination_dir / "evaluation.log"
    if destination.exists() and not overwrite:
        if not reuse_existing:
            raise ValueError(f"Evaluation log already exists: {destination}; use --force to replace it")
        size_bytes, sha256 = _hash_file(destination)
        if size_bytes == 0:
            raise ValueError(f"Existing evaluation log is empty: {destination}")
        return {
            "evaluation_id": evaluation_id,
            "path": str(destination),
            "size_bytes": size_bytes,
            "sha256": sha256,
            "content_type": "",
            "downloaded_at": "",
            "status": "existing",
        }
    destination_dir.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        log_url,
        headers={
            "Accept": "text/plain, application/octet-stream;q=0.9, */*;q=0.1",
            "User-Agent": "streamlake-experiment-analyst/1.0",
        },
        method="GET",
    )
    opener = urllib.request.build_opener(_SameHostRedirectHandler(parsed_url.hostname or ""))
    temp_path: Path | None = None
    digest = hashlib.sha256()
    size_bytes = 0
    content_type = ""
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".evaluation-log-", suffix=".partial", dir=destination_dir, delete=False
        ) as temp:
            temp_path = Path(temp.name)
            with opener.open(request, timeout=60.0) as response:
                validate_evaluation_log_url(response.geturl())
                content_type = response.headers.get_content_type()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temp.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
        if size_bytes == 0:
            raise ValueError("Downloaded evaluation log is empty")
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return {
        "evaluation_id": evaluation_id,
        "path": str(destination),
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
        "content_type": content_type,
        "downloaded_at": utc_now(),
        "status": "downloaded",
    }


def download_evaluation_log(
    client: Any,
    project_id: str,
    evaluation_id: str,
    log_dir: Path | str,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not re.fullmatch(r"eval-task-[A-Za-z0-9-]+", evaluation_id):
        raise ValueError("Evaluation ID must be an exact eval-task ID")

    detail_payload = client.request_json(
        "GET",
        f"/api/customized/commercial/v1/competition-eval-task/{urllib.parse.quote(evaluation_id, safe='')}",
    )
    detail = _response_data(detail_payload)
    if not isinstance(detail, Mapping) or str(detail.get("evalTaskId", "")) != evaluation_id:
        raise ValueError("Evaluation detail does not match the requested ID")
    if str(detail.get("projectId", "")) != project_id:
        raise PermissionError("Evaluation does not belong to the configured project")
    created_at = parse_datetime(detail.get("createTime"))
    if created_at is None or created_at < MIN_SYNC_CREATED_AT:
        raise ValueError("Evaluation log download is restricted to tasks created on or after 2026-08-01T00:00:00Z")
    if str(detail.get("taskStatus", "")).upper() != "SUCCEEDED" or detail.get("hasOutput") is not True:
        raise ValueError("Evaluation must be SUCCEEDED with hasOutput=true before its log can be downloaded")

    output_payload = client.request_json(
        "GET",
        f"/api/customized/commercial/v1/competition-eval-task/{urllib.parse.quote(evaluation_id, safe='')}/output",
    )
    log_url = _response_data(output_payload)
    if not isinstance(log_url, str) or not log_url.strip():
        raise ValueError("Evaluation output endpoint did not return a log URL")
    return download_evaluation_log_url(evaluation_id, log_url, log_dir, overwrite=overwrite)


def download_synced_evaluation_logs(
    records: Iterable[Mapping[str, Any]], log_dir: Path | str
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for record in records:
        if record.get("experiment_type") != "evaluation":
            continue
        detail = record.get("detail") if isinstance(record.get("detail"), Mapping) else {}
        status = str(detail.get("taskStatus", record.get("taskStatus", ""))).upper()
        has_output = detail.get("hasOutput", record.get("hasOutput"))
        log_url = record.get("evaluation_output")
        evaluation_id = str(record.get("id", ""))
        if status != "SUCCEEDED" or has_output is not True or not isinstance(log_url, str):
            continue
        try:
            results.append(
                download_evaluation_log_url(
                    evaluation_id,
                    log_url,
                    log_dir,
                    reuse_existing=True,
                )
            )
        except Exception as error:
            errors.append({"evaluation_id": evaluation_id, "error": str(error)})
    return {
        "eligible": len(results) + len(errors),
        "downloaded": sum(result.get("status") == "downloaded" for result in results),
        "existing": sum(result.get("status") == "existing" for result in results),
        "error_count": len(errors),
        "errors": errors,
        "items": results,
    }


def fetch_all_pages(client: Any, endpoint: EndpointSpec, page_size: int = 100) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    static_query = dict(endpoint.static_query or {})
    static_body = dict(endpoint.static_body or {})
    if endpoint.pagination == "cursor":
        cursor = ""
        used_cursors: set[str] = set()
        while True:
            if cursor in used_cursors:
                raise RuntimeError(f"repeated cursor: {cursor}")
            used_cursors.add(cursor)
            query = {**static_query, endpoint.cursor_param: cursor, endpoint.page_size_param: page_size}
            payload = client.request_json(endpoint.method, endpoint.path, query=query)
            items = get_path(payload, endpoint.items_path)
            if not isinstance(items, list):
                raise ValueError(f"API contract drift: {endpoint.items_path} is not a list")
            _append_unique(rows, seen_ids, items, endpoint.id_fields)
            next_cursor = get_path(payload, endpoint.next_cursor_path)
            if not items or next_cursor in {None, ""}:
                break
            if str(next_cursor) in used_cursors:
                raise RuntimeError(f"repeated cursor: {next_cursor}")
            cursor = str(next_cursor)
        return rows
    if endpoint.pagination != "page":
        raise ValueError(f"Unsupported pagination mode: {endpoint.pagination}")
    page = 1
    while True:
        page_values = {endpoint.page_param: page, endpoint.page_size_param: page_size}
        if endpoint.pagination_location == "body":
            query = static_query or None
            body = {**static_body, **page_values}
        elif endpoint.pagination_location == "query":
            query = {**static_query, **page_values}
            body = None
        else:
            raise ValueError(f"Unsupported pagination location: {endpoint.pagination_location}")
        payload = client.request_json(endpoint.method, endpoint.path, query=query, body=body)
        items = get_path(payload, endpoint.items_path)
        if not isinstance(items, list):
            raise ValueError(f"API contract drift: {endpoint.items_path} is not a list")
        _append_unique(rows, seen_ids, items, endpoint.id_fields)
        total = get_path(payload, endpoint.total_path)
        if not items or (isinstance(total, (int, float)) and len(rows) >= int(total)):
            break
        page += 1
        if page > 1_000_000:
            raise RuntimeError("pagination safety limit exceeded")
    return rows


def _append_unique(
    target: list[dict[str, Any]],
    seen_ids: set[str],
    items: Iterable[Any],
    id_fields: Iterable[str] = ("id", "experimentId", "experiment_id"),
) -> None:
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("API contract drift: experiment item is not an object")
        item_id = next((item.get(field) for field in id_fields if item.get(field) is not None), None)
        key = str(item_id) if item_id is not None else json.dumps(sanitize(item), sort_keys=True, ensure_ascii=False)
        if key not in seen_ids:
            seen_ids.add(key)
            target.append(dict(item))


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE experiments (
    experiment_type TEXT NOT NULL,
    id TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT,
    created_at TEXT,
    updated_at TEXT,
    raw_path TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (experiment_type, id)
);
CREATE TABLE parameters (
    experiment_type TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    name TEXT NOT NULL,
    json_value TEXT NOT NULL,
    display_value TEXT,
    PRIMARY KEY (experiment_type, experiment_id, namespace, name)
);
CREATE TABLE metrics (
    experiment_type TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    numeric_value REAL,
    display_value TEXT,
    unit TEXT,
    direction TEXT,
    PRIMARY KEY (experiment_type, experiment_id, name, category)
);
CREATE TABLE artifacts (
    experiment_type TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    name TEXT NOT NULL,
    artifact_type TEXT,
    uri TEXT,
    size INTEGER,
    metadata_json TEXT NOT NULL
);
CREATE TABLE relations (
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    provenance TEXT NOT NULL,
    confidence REAL NOT NULL
);
CREATE TABLE sync_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    outcome TEXT NOT NULL,
    source_counts_json TEXT NOT NULL
);
CREATE TABLE sync_errors (
    id INTEGER PRIMARY KEY,
    sync_run_id INTEGER,
    experiment_type TEXT,
    experiment_id TEXT,
    endpoint_class TEXT,
    error TEXT NOT NULL,
    retry_state TEXT NOT NULL
);
"""


def _record_id(record: Mapping[str, Any]) -> tuple[str, str]:
    experiment_type = str(record.get("experiment_type") or "").strip()
    experiment_id = str(record.get("id") or record.get("experimentId") or record.get("experiment_id") or "").strip()
    if experiment_type not in {"finetune", "evaluation"}:
        raise ValueError(f"Invalid experiment_type: {experiment_type or '<missing>'}")
    if not experiment_id:
        raise ValueError("Experiment record is missing id")
    return experiment_type, experiment_id


def _metric_direction(name: str) -> str:
    lowered = name.lower()
    return "min" if any(part in lowered for part in ("loss", "error", "latency", "duration", "time")) else "max"


def _display(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def write_repository(
    output_dir: Path | str,
    records: Iterable[Mapping[str, Any]],
    source_counts: Mapping[str, int],
    sync_errors: Iterable[Mapping[str, Any]] = (),
) -> SyncResult:
    root = Path(output_dir)
    materialized = [dict(record) for record in records]
    materialized_errors = [dict(error) for error in sync_errors]
    identities = [_record_id(record) for record in materialized]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate experiment identity")
    root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".streamlake-stage-", dir=root))
    db_path = staging_root / "experiments.sqlite"
    raw_root = staging_root / "raw"
    started = utc_now()
    metric_count = 0
    parameter_count = 0
    try:
        connection = sqlite3.connect(db_path)
        try:
            connection.executescript(SCHEMA)
            for record, identity in zip(materialized, identities):
                experiment_type, experiment_id = identity
                cleaned = sanitize(record)
                relative_raw = Path("raw") / experiment_type / f"{_safe_filename(experiment_id)}.json"
                raw_path = staging_root / relative_raw
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                connection.execute(
                    "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        experiment_type,
                        experiment_id,
                        str(record.get("name") or record.get("experimentName") or experiment_id),
                        _display(record.get("status")),
                        _display(record.get("created_at") or record.get("createTime")),
                        _display(record.get("updated_at") or record.get("updateTime")),
                        str(relative_raw),
                        json.dumps(cleaned, ensure_ascii=False, sort_keys=True),
                    ),
                )
                parameters = record.get("parameters") or record.get("params") or {}
                if isinstance(parameters, Mapping):
                    for name, value in _flatten(parameters):
                        connection.execute(
                            "INSERT INTO parameters VALUES (?, ?, ?, ?, ?, ?)",
                            (experiment_type, experiment_id, "training", name, json.dumps(sanitize(value), ensure_ascii=False), _display(value)),
                        )
                        parameter_count += 1
                metrics = record.get("metrics") or record.get("metric") or {}
                if isinstance(metrics, Mapping):
                    for name, value in _flatten(metrics):
                        numeric = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
                        connection.execute(
                            "INSERT INTO metrics VALUES (?, ?, ?, '', ?, ?, NULL, ?)",
                            (experiment_type, experiment_id, name, numeric, _display(value), _metric_direction(name)),
                        )
                        metric_count += 1
                artifacts = record.get("artifacts") or []
                if isinstance(artifacts, list):
                    for index, artifact in enumerate(artifacts):
                        if not isinstance(artifact, Mapping):
                            continue
                        cleaned_artifact = sanitize(artifact)
                        uri = cleaned_artifact.get("uri") or cleaned_artifact.get("url") or cleaned_artifact.get("path")
                        connection.execute(
                            "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                experiment_type,
                                experiment_id,
                                str(artifact.get("name") or f"artifact-{index + 1}"),
                                _display(artifact.get("type")),
                                _display(uri),
                                artifact.get("size") if isinstance(artifact.get("size"), int) else None,
                                json.dumps(cleaned_artifact, ensure_ascii=False, sort_keys=True),
                            ),
                        )
                train_id = record.get("train_experiment_id") or record.get("trainExperimentId")
                if experiment_type == "evaluation" and train_id:
                    connection.execute(
                        "INSERT INTO relations VALUES ('evaluation', ?, 'finetune', ?, 'explicit_train_experiment_id', 1.0)",
                        (experiment_id, str(train_id)),
                    )
            finished = utc_now()
            outcome = "partial" if materialized_errors else "success"
            connection.execute(
                "INSERT INTO sync_runs(started_at, finished_at, mode, outcome, source_counts_json) VALUES (?, ?, 'snapshot', ?, ?)",
                (started, finished, outcome, json.dumps(dict(source_counts), sort_keys=True)),
            )
            run_id = connection.execute("SELECT last_insert_rowid()").fetchone()[0]
            for error in materialized_errors:
                connection.execute(
                    "INSERT INTO sync_errors(sync_run_id, experiment_type, experiment_id, endpoint_class, error, retry_state) VALUES (?, ?, ?, ?, ?, 'pending')",
                    (
                        run_id,
                        _display(error.get("experiment_type")),
                        _display(error.get("id")),
                        _display(error.get("endpoint")),
                        _display(sanitize(error.get("error"))),
                    ),
                )
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"SQLite integrity failure: {integrity}")
        finally:
            connection.close()
        _write_exports(db_path, staging_root)
        _install_staged_repository(root, staging_root)
        _write_state(root, source_counts, len(materialized), metric_count, parameter_count, len(materialized_errors))
        return SyncResult(len(materialized), metric_count, parameter_count, dict(source_counts), len(materialized_errors))
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def _flatten(value: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    for key in sorted(value):
        child = value[key]
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            yield from _flatten(child, name)
        else:
            yield name, child


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:180] or "experiment"


def _install_staged_repository(root: Path, staging_root: Path) -> None:
    os.replace(staging_root / "experiments.sqlite", root / "experiments.sqlite")
    for name in ("raw", "exports"):
        source = staging_root / name
        target = root / name
        backup = root / f".{name}.previous"
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        os.replace(source, target)
        shutil.rmtree(backup, ignore_errors=True)
    for name in ("catalog.md", "context.md"):
        os.replace(staging_root / name, root / name)
    shutil.rmtree(staging_root, ignore_errors=True)


def _write_state(
    root: Path,
    source_counts: Mapping[str, int],
    experiments: int,
    metrics: int,
    parameters: int,
    errors: int = 0,
) -> None:
    state = {
        "updated_at": utc_now(),
        "source_counts": dict(source_counts),
        "experiment_count": experiments,
        "metric_count": metrics,
        "parameter_count": parameters,
        "error_count": errors,
    }
    temporary = root / ".sync_state.json.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, root / "sync_state.json")


def _write_log_download_state(root: Path | str, summary: Mapping[str, Any]) -> None:
    repository = Path(root)
    state_path = repository / "sync_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["log_downloads"] = sanitize(summary)
    temporary = repository / ".sync_state.json.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, state_path)


def _write_exports(db_path: Path, target_root: Path) -> None:
    exports = target_root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        for experiment_type, filename in (("finetune", "finetune.csv"), ("evaluation", "evaluation.csv")):
            rows = connection.execute(
                "SELECT id, name, status, created_at, updated_at, raw_path FROM experiments WHERE experiment_type=? ORDER BY created_at, id",
                (experiment_type,),
            ).fetchall()
            _write_csv(exports / filename, rows, ["id", "name", "status", "created_at", "updated_at", "raw_path"])
        metric_rows = connection.execute(
            "SELECT experiment_type, experiment_id, name, category, numeric_value, display_value, unit, direction FROM metrics ORDER BY experiment_type, experiment_id, name"
        ).fetchall()
        _write_csv(
            exports / "metrics.csv",
            metric_rows,
            ["experiment_type", "experiment_id", "name", "category", "numeric_value", "display_value", "unit", "direction"],
        )
        counts = dict(connection.execute("SELECT experiment_type, COUNT(*) FROM experiments GROUP BY experiment_type").fetchall())
        metrics = connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        recent = connection.execute(
            "SELECT experiment_type, id, name, status, created_at FROM experiments ORDER BY created_at DESC, id LIMIT 100"
        ).fetchall()
        lines = [
            "# StreamLake Experiment Catalog",
            "",
            f"Generated: {utc_now()}",
            f"Fine-tuning experiments: {counts.get('finetune', 0)}",
            f"Evaluation experiments: {counts.get('evaluation', 0)}",
            f"Metrics: {metrics}",
            "",
            "| Type | ID | Name | Status | Created |",
            "|---|---|---|---|---|",
        ]
        lines.extend(
            f"| {row['experiment_type']} | {row['id']} | {_md(row['name'])} | {_md(row['status'])} | {_md(row['created_at'])} |"
            for row in recent
        )
        (target_root / "catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (target_root / "context.md").write_text("\n".join(lines[:9] + lines[9:59]) + "\n", encoding="utf-8")
    finally:
        connection.close()


def _write_csv(path: Path, rows: Iterable[sqlite3.Row], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def _md(value: Any) -> str:
    return _display(value).replace("|", "\\|").replace("\n", " ")


def resolve_experiments(connection: sqlite3.Connection, selectors: Iterable[str]) -> list[sqlite3.Row]:
    connection.row_factory = sqlite3.Row
    resolved: list[sqlite3.Row] = []
    seen: set[tuple[str, str]] = set()
    for selector in selectors:
        exact_id = connection.execute("SELECT * FROM experiments WHERE id=? ORDER BY experiment_type", (selector,)).fetchall()
        candidates = exact_id or connection.execute("SELECT * FROM experiments WHERE name=? ORDER BY id", (selector,)).fetchall()
        if not candidates:
            raise ValueError(f"No experiment matches: {selector}")
        if len(candidates) > 1:
            ids = ", ".join(str(row["id"]) for row in candidates)
            raise ValueError(f"Ambiguous experiment selector {selector!r}; candidates: {ids}")
        row = candidates[0]
        key = (row["experiment_type"], row["id"])
        if key not in seen:
            seen.add(key)
            resolved.append(row)
    return resolved


def compare_experiments(
    connection: sqlite3.Connection,
    selectors: Iterable[str],
    baseline: str | None = None,
    primary_metric: str | None = None,
) -> str:
    rows = resolve_experiments(connection, selectors)
    if not rows:
        raise ValueError("Select at least one experiment")
    by_id = {str(row["id"]): row for row in rows}
    baseline_id = baseline or str(rows[0]["id"])
    if baseline_id not in by_id:
        baseline_row = resolve_experiments(connection, [baseline_id])[0]
        rows.insert(0, baseline_row)
        by_id[str(baseline_row["id"])] = baseline_row
    metric_values: dict[str, dict[str, float]] = {}
    for row in rows:
        values = connection.execute(
            "SELECT name, numeric_value FROM metrics WHERE experiment_type=? AND experiment_id=? AND numeric_value IS NOT NULL",
            (row["experiment_type"], row["id"]),
        ).fetchall()
        metric_values[str(row["id"])] = {str(name): float(value) for name, value in values}
    all_metrics = sorted({name for values in metric_values.values() for name in values})
    selected_metric = primary_metric or (all_metrics[0] if all_metrics else None)
    lines = ["# StreamLake Experiment Comparison", "", f"Baseline: {baseline_id}"]
    if selected_metric:
        direction = _metric_direction(selected_metric)
        ranked = sorted(
            ((experiment_id, values[selected_metric]) for experiment_id, values in metric_values.items() if selected_metric in values),
            key=lambda item: item[1],
            reverse=direction == "max",
        )
        if ranked:
            lines.extend(["", f"Primary metric: {selected_metric} ({direction})", f"Best observed experiment: {ranked[0][0]} ({ranked[0][1]:.6f})"])
    lines.extend(["", "## Metric deltas", ""])
    base_values = metric_values[baseline_id]
    for experiment_id in [str(row["id"]) for row in rows]:
        lines.append(f"### {experiment_id}")
        for metric in all_metrics:
            value = metric_values[experiment_id].get(metric)
            base_value = base_values.get(metric)
            if value is None:
                lines.append(f"- {metric}: missing")
            elif base_value is None:
                lines.append(f"- {metric}: {value:.6f}; baseline missing")
            else:
                lines.append(f"- {metric}: {value:.6f}; delta {value - base_value:+.6f}")
        lines.append("")
    parameter_values: dict[str, dict[str, str]] = {}
    for row in rows:
        values = connection.execute(
            "SELECT name, display_value FROM parameters WHERE experiment_type=? AND experiment_id=? ORDER BY name",
            (row["experiment_type"], row["id"]),
        ).fetchall()
        parameter_values[str(row["id"])] = {str(name): str(value) for name, value in values}
    all_parameters = sorted({name for values in parameter_values.values() for name in values})
    lines.extend(["## Configuration differences", ""])
    if not all_parameters:
        lines.append("No normalized training parameters are available for these experiments.")
    else:
        for parameter in all_parameters:
            rendered = ", ".join(
                f"{experiment_id}={parameter_values[experiment_id].get(parameter, 'missing')}"
                for experiment_id in [str(row["id"]) for row in rows]
            )
            if len({parameter_values[str(row["id"])].get(parameter) for row in rows}) > 1:
                lines.append(f"- {parameter}: {rendered}")
    lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "Configuration and metric changes are correlation evidence only; this report does not claim causation without controlled repeated experiments.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def load_contract(path: Path | str) -> dict[str, EndpointSpec]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    endpoints = raw.get("endpoints")
    if not isinstance(endpoints, Mapping):
        raise ValueError("Contract must contain an endpoints object")
    return {str(name): EndpointSpec.from_dict(value) for name, value in endpoints.items()}


def normalize_remote_record(record: Mapping[str, Any], experiment_type: str) -> dict[str, Any]:
    normalized = dict(record)
    normalized["experiment_type"] = experiment_type
    if "id" not in normalized:
        normalized["id"] = record.get("experimentId") or record.get("experiment_id")
    if "name" not in normalized:
        normalized["name"] = record.get("experimentName") or record.get("taskName") or normalized.get("id")
    for source, target in (
        ("createTime", "created_at"),
        ("createdAt", "created_at"),
        ("create_time", "created_at"),
        ("updateTime", "updated_at"),
        ("updatedAt", "updated_at"),
        ("update_time", "updated_at"),
    ):
        if source in record and target not in normalized:
            normalized[target] = record[source]
    return normalized


def filter_records_since(
    records: Iterable[Mapping[str, Any]], since: datetime = MIN_SYNC_CREATED_AT
) -> list[dict[str, Any]]:
    """Keep only records within the non-overridable synchronization retention window."""
    cutoff = parse_datetime(since)
    if cutoff is None:
        raise ValueError("since must be a valid timestamp")
    if cutoff < MIN_SYNC_CREATED_AT:
        raise ValueError("StreamLake sync is restricted to records created on or after 2026-08-01T00:00:00Z")
    filtered: list[dict[str, Any]] = []
    for record in records:
        created_at = record.get("created_at")
        if created_at is None:
            created_at = record.get("createTime") or record.get("createdAt") or record.get("create_time")
        parsed = parse_datetime(created_at)
        if parsed is not None and parsed >= cutoff:
            filtered.append(dict(record))
    return filtered


def is_unavailable_evaluation(record: Mapping[str, Any]) -> bool:
    """Identify failed evaluations that the platform explicitly says have no output."""
    return (
        record.get("experiment_type") == "evaluation"
        and str(record.get("taskStatus", "")).upper() == "FAILED"
        and record.get("hasOutput") is False
    )


def filter_unavailable_evaluations(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Exclude evaluations which cannot provide a result before any detail request."""
    return [dict(record) for record in records if not is_unavailable_evaluation(record)]


def _response_data(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return payload.get("responseData", payload.get("data", payload))
    return payload


def _metric_definitions(value: Any) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            metrics = node.get("metrics")
            if isinstance(metrics, list):
                for metric in metrics:
                    if isinstance(metric, Mapping) and metric.get("name"):
                        definition = {
                            "name": str(metric["name"]),
                            "seriesNameFormat": str(metric.get("seriesNameFormat") or metric["name"]),
                        }
                        key = (definition["name"], definition["seriesNameFormat"])
                        if key not in seen:
                            seen.add(key)
                            found.append(definition)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return found


METRIC_NAME_RE = re.compile(
    r"(^r[123]$|score|metric|accuracy|acc$|loss|recall|precision|f1|ndcg|hit|auc|mrr|rouge|bleu|pass@|perplex|reward)",
    re.IGNORECASE,
)


def _numeric_metrics(value: Any, prefix: str = "", include_all: bool = False) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not isinstance(value, Mapping):
        return metrics
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            metrics.update(_numeric_metrics(child, path, include_all))
        elif isinstance(child, (int, float)) and not isinstance(child, bool):
            if include_all or METRIC_NAME_RE.search(str(key)):
                metrics[path] = float(child)
    return metrics


def _latest_series_metrics(payload: Any) -> dict[str, float]:
    data = _response_data(payload)
    series = data.get("series") if isinstance(data, Mapping) else None
    if not isinstance(series, list):
        return {}
    output: dict[str, float] = {}
    for index, item in enumerate(series):
        if not isinstance(item, Mapping):
            continue
        points = item.get("points")
        if not isinstance(points, list) or not points:
            continue
        latest = points[-1]
        value = latest.get("value") if isinstance(latest, Mapping) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        name = str(item.get("name") or item.get("metricName") or f"series_{index + 1}")
        labels = item.get("labels")
        if isinstance(labels, Mapping) and labels.get("taskId"):
            name = f"{name}.{labels['taskId']}"
        output[name] = float(value)
    return output


def collect_streamlake_records(
    client: Any,
    project_id: str,
    page_size: int = 100,
    limit: int | None = None,
    since: datetime = MIN_SYNC_CREATED_AT,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, str]]]:
    """Collect post-cutoff list records plus lightweight detail and metric payloads."""
    finetune_endpoint = EndpointSpec(
        path="/api/customized/commercial/v1/train-task/list",
        method="POST",
        items_path="responseData.list",
        pagination="page",
        pagination_location="body",
        total_path="responseData.total",
        static_body={
            "projectId": project_id,
            "keyword": "",
            "tags": [],
            "taskStatus": [],
            "creator": [],
            "taskType": [],
            "fineTuningType": [],
            "baseModelName": "",
            "sortBy": "createTime",
            "sortOrder": "DESC",
        },
        id_fields=("taskId",),
    )
    evaluation_endpoint = EndpointSpec(
        path="/api/customized/commercial/v1/competition-eval-task/list",
        method="POST",
        items_path="responseData.items",
        pagination="page",
        pagination_location="body",
        total_path="responseData.total",
        static_body={"projectId": project_id, "sortBy": "createTime", "sortOrder": "DESC"},
        id_fields=("evalTaskId",),
    )
    finetune_items = fetch_all_pages(client, finetune_endpoint, page_size)
    evaluation_items = fetch_all_pages(client, evaluation_endpoint, page_size)
    finetune_items = filter_records_since(
        (normalize_remote_record(item, "finetune") for item in finetune_items), since
    )
    evaluation_items = filter_records_since(
        (normalize_remote_record(item, "evaluation") for item in evaluation_items), since
    )
    evaluation_items = filter_unavailable_evaluations(evaluation_items)
    counts = {"finetune": len(finetune_items), "evaluation": len(evaluation_items)}
    if limit is not None:
        finetune_items = finetune_items[:limit]
        evaluation_items = evaluation_items[:limit]
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for item in finetune_items:
        task_id = str(item.get("taskId") or "")
        record = normalize_remote_record({**item, "id": task_id}, "finetune")
        detail_data: Any = {}
        try:
            detail_payload = client.request_json(
                "GET",
                f"/api/customized/commercial/v1/train-task/{urllib.parse.quote(task_id, safe='')}",
                query={"projectId": project_id},
            )
            detail_data = _response_data(detail_payload)
            if isinstance(detail_data, Mapping):
                record.update(detail_data)
                record["id"] = task_id
                record["experiment_type"] = "finetune"
                hyperparameters = detail_data.get("hyperParams") or detail_data.get("hyper_params")
                if isinstance(hyperparameters, Mapping):
                    record["parameters"] = dict(hyperparameters)
        except PermissionError:
            raise
        except Exception as error:
            errors.append({"experiment_type": "finetune", "id": task_id, "endpoint": "detail", "error": str(error)})
        try:
            dashboard = client.request_json(
                "GET",
                "/api/customized/commercial/v1/train-task/analysis/dashboard",
                query={"taskId": task_id, "fineTuningType": record.get("fineTuningType", "")},
            )
            record["analysis_dashboard"] = _response_data(dashboard)
            metric_definitions = _metric_definitions(_response_data(dashboard))
            if metric_definitions:
                combined_series: list[Any] = []
                latest_metrics: dict[str, float] = {}
                pending_batches = [
                    metric_definitions[start : start + METRIC_QUERY_BATCH_SIZE]
                    for start in range(0, len(metric_definitions), METRIC_QUERY_BATCH_SIZE)
                ]
                while pending_batches:
                    batch = pending_batches.pop()
                    try:
                        metric_payload = client.request_json(
                            "POST",
                            "/api/customized/commercial/v1/train-task/metric-query",
                            body={"projectId": project_id, "taskIds": [task_id], "metrics": batch},
                        )
                    except PermissionError:
                        if len(batch) == 1:
                            raise
                        midpoint = len(batch) // 2
                        pending_batches.extend((batch[:midpoint], batch[midpoint:]))
                        continue
                    metric_data = _response_data(metric_payload)
                    if isinstance(metric_data, Mapping) and isinstance(metric_data.get("series"), list):
                        combined_series.extend(metric_data["series"])
                    latest_metrics.update(_latest_series_metrics(metric_payload))
                record["training_metric_series"] = {"series": combined_series}
                record["metrics"] = {
                    **_numeric_metrics(item),
                    **latest_metrics,
                }
            else:
                record["metrics"] = _numeric_metrics(item)
        except PermissionError as error:
            # Some legacy tasks deny only metric-query while list/detail auth remains valid.
            # Re-probe both list sources before treating this as an expired session.
            probe_streamlake(client, project_id)
            errors.append({"experiment_type": "finetune", "id": task_id, "endpoint": "metrics", "error": str(error)})
            record.setdefault("metrics", _numeric_metrics(item))
        except Exception as error:
            errors.append({"experiment_type": "finetune", "id": task_id, "endpoint": "metrics", "error": str(error)})
            record.setdefault("metrics", _numeric_metrics(item))
        records.append(record)

    for item in evaluation_items:
        task_id = str(item.get("evalTaskId") or "")
        record = normalize_remote_record({**item, "id": task_id}, "evaluation")
        record["metrics"] = _numeric_metrics(item)
        try:
            detail_payload = client.request_json(
                "GET",
                f"/api/customized/commercial/v1/competition-eval-task/{urllib.parse.quote(task_id, safe='')}",
            )
            detail_data = _response_data(detail_payload)
            record["detail"] = detail_data
            if isinstance(detail_data, Mapping):
                for key in (
                    "modelId",
                    "modelName",
                    "baseModel",
                    "description",
                    "creator",
                    "createTime",
                    "taskStatus",
                    "hasOutput",
                ):
                    if key in detail_data:
                        record[key] = detail_data[key]
                record["metrics"].update(_numeric_metrics(detail_data, "detail"))
        except PermissionError:
            raise
        except Exception as error:
            errors.append({"experiment_type": "evaluation", "id": task_id, "endpoint": "detail", "error": str(error)})
        if record.get("hasOutput") is True:
            try:
                output_payload = client.request_json(
                    "GET",
                    f"/api/customized/commercial/v1/competition-eval-task/{urllib.parse.quote(task_id, safe='')}/output",
                )
                output_data = _response_data(output_payload)
                record["evaluation_output"] = output_data
                record["metrics"].update(_numeric_metrics(output_data, "output"))
            except PermissionError:
                raise
            except Exception as error:
                errors.append(
                    {"experiment_type": "evaluation", "id": task_id, "endpoint": "output", "error": str(error)}
                )
        else:
            record["evaluation_output_status"] = "pending"
        records.append(record)
    return records, counts, errors


def sync_from_contract(
    client: StreamLakeClient,
    contract: Mapping[str, EndpointSpec],
    output_dir: Path | str,
    page_size: int = 100,
    since: datetime = MIN_SYNC_CREATED_AT,
) -> SyncResult:
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for experiment_type in ("finetune", "evaluation"):
        endpoint = contract.get(experiment_type)
        if endpoint is None:
            raise ValueError(f"Contract missing endpoint: {experiment_type}")
        items = fetch_all_pages(client, endpoint, page_size=page_size)
        items = filter_records_since(
            (normalize_remote_record(item, experiment_type) for item in items), since
        )
        if experiment_type == "evaluation":
            items = filter_unavailable_evaluations(items)
        counts[experiment_type] = len(items)
        records.extend(items)
    return write_repository(output_dir, records, counts)


def probe_streamlake(client: Any, project_id: str) -> dict[str, Any]:
    requests = {
        "finetune": (
            "/api/customized/commercial/v1/train-task/list",
            {"projectId": project_id, "page": 1, "pageSize": 1, "sortBy": "createTime", "sortOrder": "DESC"},
            "responseData.list",
            "responseData.total",
        ),
        "evaluation": (
            "/api/customized/commercial/v1/competition-eval-task/list",
            {"projectId": project_id, "page": 1, "pageSize": 1, "sortBy": "createTime", "sortOrder": "DESC"},
            "responseData.items",
            "responseData.total",
        ),
    }
    summary: dict[str, Any] = {}
    for name, (path, body, items_path, total_path) in requests.items():
        payload = client.request_json("POST", path, body=body)
        items = get_path(payload, items_path)
        if not isinstance(items, list):
            raise ValueError(f"API contract drift: {items_path} is not a list")
        summary[name] = {"authenticated": True, "sample_count": len(items), "total": get_path(payload, total_path)}
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.local/share/streamlake-eval-monitor/data").expanduser(),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "sync", "download-log"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--cookie-file", type=Path)
        subparser.add_argument("--origin", default=DEFAULT_ORIGIN)
        subparser.add_argument("--project-id")
    for name in ("probe", "sync"):
        subparser = subparsers.choices[name]
        subparser.add_argument("--contract", type=Path)
        subparser.add_argument("--page-size", type=int, default=100)
    subparsers.choices["sync"].add_argument(
        "--since",
        default=None,
        help="later minimum creation timestamp (ISO-8601, inclusive; cannot be earlier than 2026-08-01T00:00:00Z)",
    )
    subparsers.choices["sync"].add_argument("--limit", type=int)
    subparsers.choices["sync"].add_argument("--log-dir", type=Path)
    downloader = subparsers.choices["download-log"]
    downloader.add_argument("evaluation_id")
    downloader.add_argument("--log-dir", type=Path)
    downloader.add_argument("--force", action="store_true")
    subparsers.add_parser("status")
    listing = subparsers.add_parser("list")
    listing.add_argument("--type", choices=["finetune", "evaluation"])
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("selectors", nargs="+")
    comparison.add_argument("--baseline")
    comparison.add_argument("--primary-metric")
    subparsers.add_parser("context")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir
    if args.command in {"probe", "sync", "download-log"}:
        cookie = load_cookie(args.cookie_file)
        project_id = resolve_project_id(args.project_id)
        client = StreamLakeClient(cookie, origin=args.origin)
        if args.command == "download-log":
            result = download_evaluation_log(
                client,
                project_id,
                args.evaluation_id,
                args.log_dir or (output_dir / "logs"),
                overwrite=args.force,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "probe":
            if args.contract:
                contract = load_contract(args.contract)
                summary = {}
                for name, endpoint in contract.items():
                    rows = fetch_all_pages(client, endpoint, page_size=1)
                    summary[name] = {"authenticated": True, "sample_count": min(len(rows), 1)}
            else:
                summary = probe_streamlake(client, project_id)
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        log_summary: dict[str, Any] | None = None
        if args.contract:
            contract = load_contract(args.contract)
            result = sync_from_contract(
                client,
                contract,
                output_dir,
                page_size=args.page_size,
                since=parse_since(args.since),
            )
        else:
            records, counts, errors = collect_streamlake_records(
                client,
                project_id,
                page_size=args.page_size,
                limit=args.limit,
                since=parse_since(args.since),
            )
            if errors:
                sample = "; ".join(
                    f"{item.get('experiment_type')}:{item.get('id')}:{item.get('endpoint')}={item.get('error')}"
                    for item in errors[:5]
                )
                raise RuntimeError(
                    f"Synchronization incomplete ({len(errors)} endpoint errors); existing repository was preserved. {sample}"
                )
            result = write_repository(output_dir, records, counts, errors)
            log_summary = download_synced_evaluation_logs(records, args.log_dir or (output_dir / "logs"))
            _write_log_download_state(output_dir, log_summary)
        payload = dict(result.__dict__)
        if log_summary is not None:
            payload["log_downloads"] = log_summary
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    db_path = output_dir / "experiments.sqlite"
    if args.command == "status":
        state = output_dir / "sync_state.json"
        if not state.is_file():
            raise FileNotFoundError(f"No synchronized repository at {output_dir}")
        print(state.read_text(encoding="utf-8"), end="")
        return 0
    if args.command == "context":
        print((output_dir / "context.md").read_text(encoding="utf-8"), end="")
        return 0
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        if args.command == "list":
            query = "SELECT experiment_type, id, name, status, created_at FROM experiments"
            params: tuple[Any, ...] = ()
            if args.type:
                query += " WHERE experiment_type=?"
                params = (args.type,)
            query += " ORDER BY created_at DESC, id"
            print(json.dumps([dict(row) for row in connection.execute(query, params)], ensure_ascii=False, indent=2))
            return 0
        if args.command == "compare":
            print(compare_experiments(connection, args.selectors, args.baseline, args.primary_metric), end="")
            return 0
    finally:
        connection.close()
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
