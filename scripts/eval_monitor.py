#!/usr/bin/env python3
"""Deploy and manage the bundled StreamLake evaluation monitor."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = SKILL_ROOT / "assets" / "eval-monitor"
SYNC_SCRIPT = SKILL_ROOT / "scripts" / "streamlake_experiments.py"
APP_FILES = (
    "dashboard.html",
    "monitor_server.py",
    "eval_log_parser.py",
    "training_dashboard.html",
    "training_monitor_server.py",
)
TRAINING_CONFIG_TEMPLATE = "training_monitor_config.json"
TEMPLATE_FILES = ("open_monitor_windows.ps1.template",)
DEFAULT_TARGET = "~/.local/share/streamlake-eval-monitor"
DEFAULT_PORT = 18280
TRAINING_OUTPUT_ENV_VARS = (
    "STREAMLAKE_TRAINING_OUTPUT_ROOT",
    "LLAMA_FACTORY_OUTPUT_ROOT",
    "LLAMA_FACTORY_OUTPUT_DIR",
    "TRAINING_OUTPUT_ROOT",
    "TRAINING_OUTPUT_DIR",
)


def expand(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def discover_training_output_roots(explicit: list[str] | None = None) -> list[str]:
    """Return existing, machine-local training roots without assuming /root or a repo path."""
    candidates: list[str] = list(explicit or [])
    if not explicit:
        environment_values: list[str] = []
        for name in TRAINING_OUTPUT_ENV_VARS:
            value = os.environ.get(name, "").strip()
            if value:
                environment_values.extend(part.strip() for part in value.split(os.pathsep) if part.strip())
        if environment_values:
            candidates.extend(environment_values)
        else:
            home = Path.home()
            candidates.extend(
                [
                    str(home / "output"),
                    str(home / "LLaMA-Factory" / "output"),
                    str(Path.cwd() / "output"),
                ]
            )
    roots: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        path = expand(value)
        key = str(path)
        if key in seen or not path.is_dir():
            continue
        seen.add(key)
        roots.append(key)
    return roots


def training_config_payload(roots: list[str]) -> dict[str, Any]:
    template = json.loads((ASSET_DIR / TRAINING_CONFIG_TEMPLATE).read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise RuntimeError("Invalid training monitor configuration template")
    template["outputs_roots"] = roots
    return template


def repair_training_paths(target: Path, explicit_roots: list[str] | None = None, write: bool = True) -> dict[str, Any]:
    """Normalize a deployed training config to paths that exist on this machine."""
    path = target / TRAINING_CONFIG_TEMPLATE
    if not path.is_file():
        raise RuntimeError(f"Training monitor configuration is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid training monitor configuration: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid training monitor configuration: {path}")

    before_roots = [str(item).strip() for item in value.get("outputs_roots", []) if str(item).strip()]
    candidates = list(explicit_roots or []) + before_roots
    roots = discover_training_output_roots(candidates) if candidates else discover_training_output_roots()
    if not roots:
        roots = discover_training_output_roots()

    before_targets = value.get("targets", [])
    targets: list[dict[str, Any]] = []
    removed_targets = 0
    removed_target_paths = 0
    if isinstance(before_targets, list):
        for item in before_targets:
            if not isinstance(item, dict):
                removed_targets += 1
                continue
            output_value = str(item.get("output_dir", "")).strip()
            if not output_value or not expand(output_value).is_dir():
                removed_targets += 1
                continue
            target_copy = dict(item)
            output_dir = expand(output_value)
            target_copy["output_dir"] = str(output_dir)
            for field in ("metrics_path", "log_path", "config_path"):
                if field not in target_copy or target_copy[field] in (None, ""):
                    continue
                candidate = Path(str(target_copy[field])).expanduser()
                if not candidate.is_absolute():
                    candidate = output_dir / candidate
                candidate = candidate.resolve()
                if candidate.is_file():
                    target_copy[field] = str(candidate)
                else:
                    target_copy.pop(field, None)
                    removed_target_paths += 1
            if "pid" in target_copy:
                target_copy.pop("pid", None)
                removed_target_paths += 1
            targets.append(target_copy)
    else:
        removed_targets = 1

    changed = before_roots != roots or before_targets != targets
    value["outputs_roots"] = roots
    value["targets"] = targets
    if write and changed:
        atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return {
        "config_file": str(path),
        "written": bool(write and changed),
        "outputs_roots": roots,
        "removed_roots": [item for item in before_roots if item not in roots],
        "targets": len(targets),
        "removed_targets": removed_targets,
        "removed_target_paths": removed_target_paths,
    }


def atomic_write(path: Path, text: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def config_path(target: Path) -> Path:
    return target / "monitor_config.json"


def pid_path(target: Path) -> Path:
    return target / "monitor_server.pid"


def log_path(target: Path) -> Path:
    return target / "monitor_server.log"


def read_config(target: Path) -> dict[str, Any]:
    path = config_path(target)
    if not path.is_file():
        raise RuntimeError(f"Monitor is not deployed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid monitor configuration: {path}")
    return value


def read_pid(target: Path) -> int | None:
    path = pid_path(target)
    if not path.is_file():
        return None
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except (ValueError, OSError):
        return None
    return value if value > 1 else None


def process_exists(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.is_file():
        try:
            if stat_path.read_text(encoding="ascii").split()[2] == "Z":
                return False
        except (OSError, IndexError):
            pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def process_is_monitor(pid: int, target: Path) -> bool:
    command_path = Path(f"/proc/{pid}/cmdline")
    if not command_path.is_file():
        return False
    try:
        command = command_path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
    except OSError:
        return False
    return str(target / "monitor_server.py") in command and str(config_path(target)) in command


def health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1.5) as response:
            value = json.load(response)
        return response.status == 200 and value.get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def stop_monitor(target: Path, quiet: bool = False) -> bool:
    pid = read_pid(target)
    if pid is None or not process_exists(pid):
        pid_path(target).unlink(missing_ok=True)
        if not quiet:
            print("Evaluation monitor is not running.")
        return False
    if not process_is_monitor(pid, target):
        raise RuntimeError(f"Refusing to stop PID {pid}: it is not this deployed monitor")
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not process_exists(pid):
            pid_path(target).unlink(missing_ok=True)
            if not quiet:
                print(f"Evaluation monitor stopped (PID {pid}).")
            return True
        time.sleep(0.1)
    raise RuntimeError(f"Monitor PID {pid} did not stop after SIGTERM")


def start_monitor(target: Path) -> int:
    config = read_config(target)
    port = int(config["port"])
    existing = read_pid(target)
    if existing and process_exists(existing):
        if not process_is_monitor(existing, target):
            raise RuntimeError(f"PID file points to an unrelated process: {existing}")
        if health(port):
            print(f"Evaluation monitor is already running (PID {existing}).")
            print_access(port)
            return existing
        raise RuntimeError(f"Monitor PID {existing} is running but its health endpoint is unavailable")
    pid_path(target).unlink(missing_ok=True)

    log_handle = log_path(target).open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            [sys.executable, str(target / "monitor_server.py"), "--config", str(config_path(target))],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_handle.close()
    atomic_write(pid_path(target), f"{process.pid}\n", 0o600)

    for _ in range(50):
        if process.poll() is not None:
            tail = ""
            try:
                tail = log_path(target).read_text(encoding="utf-8", errors="replace")[-3000:]
            except OSError:
                pass
            raise RuntimeError(f"Monitor exited during startup. Log tail:\n{tail}")
        if health(port):
            print(f"Evaluation monitor started (PID {process.pid}).")
            print_access(port)
            return process.pid
        time.sleep(0.1)
    raise RuntimeError(f"Monitor did not become healthy; inspect {log_path(target)}")


def print_access(port: int) -> None:
    print(f"Training Monitor URL: http://127.0.0.1:{port}/")
    print(f"Evaluation Monitor URL: http://127.0.0.1:{port}/eval/")
    print("On the computer where the browser runs, keep this command open:")
    print(f"  ssh -N -L {port}:127.0.0.1:{port} USER@SERVER")
    print("A personal SSH alias works only when that computer's SSH config defines it.")
    print(f"Then open locally for training: http://127.0.0.1:{port}/")
    print(f"Then open locally for evaluation: http://127.0.0.1:{port}/eval/")


def deploy(args: argparse.Namespace) -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required")
    target = expand(args.target_dir)
    output_dir = expand(args.output_dir) if args.output_dir else target / "data"
    cookie_dir = target / "config"
    port = int(args.port)
    if not 1024 <= port <= 65535:
        raise RuntimeError("Port must be between 1024 and 65535")
    missing = [
        name
        for name in (*APP_FILES, TRAINING_CONFIG_TEMPLATE, *TEMPLATE_FILES)
        if not (ASSET_DIR / name).is_file()
    ]
    if missing or not SYNC_SCRIPT.is_file():
        raise RuntimeError(f"Skill package is incomplete: {', '.join(missing) or SYNC_SCRIPT}")

    if config_path(target).is_file():
        existing_pid = read_pid(target)
        if existing_pid and process_exists(existing_pid):
            stop_monitor(target, quiet=True)

    target.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cookie_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(target, 0o700)
    os.chmod(cookie_dir, 0o700)
    for name in APP_FILES:
        shutil.copy2(ASSET_DIR / name, target / name)
        os.chmod(target / name, 0o644)
    training_config = target / TRAINING_CONFIG_TEMPLATE
    if not training_config.is_file():
        roots = discover_training_output_roots(args.training_output_root)
        atomic_write(training_config, json.dumps(training_config_payload(roots), ensure_ascii=False, indent=2) + "\n", 0o600)
    else:
        roots = []
        try:
            current_training_config = json.loads(training_config.read_text(encoding="utf-8"))
            if isinstance(current_training_config, dict) and isinstance(current_training_config.get("outputs_roots"), list):
                roots = [str(item) for item in current_training_config["outputs_roots"] if str(item).strip()]
        except (OSError, json.JSONDecodeError):
            roots = []
        if args.repair_paths:
            repaired = repair_training_paths(target, args.training_output_root)
            roots = repaired["outputs_roots"]
            print(
                "Path repair: kept {} training root(s), removed {} invalid target(s).".format(
                    len(roots), repaired["removed_targets"]
                )
            )

    config = {
        "bind_host": "127.0.0.1",
        "port": port,
        "output_dir": str(output_dir),
        "skill_script": str(SYNC_SCRIPT.resolve()),
        "parser_dir": str(target),
        "cookie_file": str(cookie_dir / "cookie"),
        "project_id_file": str(cookie_dir / "project_id"),
        "huggingface_token_file": str(cookie_dir / "huggingface_token"),
        "training_config_file": str(training_config),
        "training_upload_registry_file": str(target / "training_upload_registry.json"),
    }
    atomic_write(config_path(target), json.dumps(config, ensure_ascii=False, indent=2) + "\n", 0o600)
    powershell = (ASSET_DIR / "open_monitor_windows.ps1.template").read_text(encoding="utf-8")
    atomic_write(target / "open_monitor_windows.ps1", powershell.replace("__REMOTE_PORT__", str(port)), 0o600)
    print(f"Evaluation monitor deployed to {target}")
    print(f"Runtime config: {cookie_dir}")
    print(f"Experiment data: {output_dir}")
    if roots:
        print("Training output roots: " + ", ".join(roots))
    else:
        print("Training output roots: none detected; edit training_monitor_config.json before using the training tab.")
    if args.no_start:
        print(f"Start it with: {sys.executable} {Path(__file__).resolve()} start --target-dir {target}")
    else:
        start_monitor(target)


def show_status(target: Path) -> None:
    config = read_config(target)
    pid = read_pid(target)
    owned = bool(pid and process_exists(pid) and process_is_monitor(pid, target))
    value = {
        "target_dir": str(target),
        "port": int(config["port"]),
        "pid": pid,
        "running": owned and health(int(config["port"])),
        "output_dir": config["output_dir"],
        "configured": Path(config["cookie_file"]).is_file() and Path(config["project_id_file"]).is_file(),
        "training_monitor": Path(config.get("training_config_file", "")).is_file(),
    }
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    deploy_parser = subparsers.add_parser("deploy", help="Install or upgrade the monitor and start it")
    deploy_parser.add_argument("--target-dir", default=DEFAULT_TARGET)
    deploy_parser.add_argument("--output-dir", help="defaults to <target-dir>/data")
    deploy_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    deploy_parser.add_argument(
        "--training-output-root",
        action="append",
        help="existing local training output root; repeat for multiple roots (otherwise auto-detected)",
    )
    deploy_parser.add_argument(
        "--repair-paths",
        action="store_true",
        help="normalize an existing training config to current-machine paths before starting",
    )
    deploy_parser.add_argument("--no-start", action="store_true")

    repair_parser = subparsers.add_parser(
        "repair-paths",
        help="remove invalid training roots/targets and keep paths that exist on this machine",
    )
    repair_parser.add_argument("--target-dir", default=DEFAULT_TARGET)
    repair_parser.add_argument(
        "--training-output-root",
        action="append",
        help="existing local training output root; repeat for multiple roots",
    )
    repair_parser.add_argument("--no-write", action="store_true", help="report repairs without changing the config")

    for command in ("start", "stop", "status"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--target-dir", default=DEFAULT_TARGET)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        target = expand(getattr(args, "target_dir", DEFAULT_TARGET))
        if args.command == "deploy":
            deploy(args)
        elif args.command == "repair-paths":
            print(json.dumps(repair_training_paths(target, args.training_output_root, not args.no_write), ensure_ascii=False, indent=2))
        elif args.command == "start":
            start_monitor(target)
        elif args.command == "stop":
            stop_monitor(target)
        elif args.command == "status":
            show_status(target)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
