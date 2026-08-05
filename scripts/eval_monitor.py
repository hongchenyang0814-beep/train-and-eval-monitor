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
APP_FILES = ("dashboard.html", "monitor_server.py", "eval_log_parser.py")
TEMPLATE_FILES = ("open_monitor_windows.ps1.template",)
DEFAULT_TARGET = "~/.local/share/streamlake-eval-monitor"
DEFAULT_PORT = 18280


def expand(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


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
    print(f"Server-local URL: http://127.0.0.1:{port}/")
    print("On the computer where the browser runs, keep this command open:")
    print(f"  ssh -N -L {port}:127.0.0.1:{port} USER@SERVER")
    print("Replace USER@SERVER with the real SSH login; add -p SSH_PORT when needed.")
    print("A personal SSH alias works only when that computer's SSH config defines it.")
    print(f"Then open locally: http://127.0.0.1:{port}/")


def deploy(args: argparse.Namespace) -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required")
    target = expand(args.target_dir)
    output_dir = expand(args.output_dir) if args.output_dir else target / "data"
    cookie_dir = target / "config"
    port = int(args.port)
    if not 1024 <= port <= 65535:
        raise RuntimeError("Port must be between 1024 and 65535")
    missing = [name for name in (*APP_FILES, *TEMPLATE_FILES) if not (ASSET_DIR / name).is_file()]
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

    config = {
        "bind_host": "127.0.0.1",
        "port": port,
        "output_dir": str(output_dir),
        "skill_script": str(SYNC_SCRIPT.resolve()),
        "parser_dir": str(target),
        "cookie_file": str(cookie_dir / "cookie"),
        "project_id_file": str(cookie_dir / "project_id"),
    }
    atomic_write(config_path(target), json.dumps(config, ensure_ascii=False, indent=2) + "\n", 0o600)
    powershell = (ASSET_DIR / "open_monitor_windows.ps1.template").read_text(encoding="utf-8")
    atomic_write(target / "open_monitor_windows.ps1", powershell.replace("__REMOTE_PORT__", str(port)), 0o600)
    print(f"Evaluation monitor deployed to {target}")
    print(f"Runtime config: {cookie_dir}")
    print(f"Experiment data: {output_dir}")
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
    }
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    deploy_parser = subparsers.add_parser("deploy", help="Install or upgrade the monitor and start it")
    deploy_parser.add_argument("--target-dir", default=DEFAULT_TARGET)
    deploy_parser.add_argument("--output-dir", help="defaults to <target-dir>/data")
    deploy_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    deploy_parser.add_argument("--no-start", action="store_true")

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
