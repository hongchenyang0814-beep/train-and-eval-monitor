import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts" / "eval_monitor.py"
MONITOR_MODULE_PATH = ROOT / "assets" / "eval-monitor" / "monitor_server.py"
MONITOR_SPEC = importlib.util.spec_from_file_location("bundled_eval_monitor_tests", MONITOR_MODULE_PATH)
monitor = importlib.util.module_from_spec(MONITOR_SPEC)
sys.modules[MONITOR_SPEC.name] = monitor
MONITOR_SPEC.loader.exec_module(monitor)


class FakeParser:
    PARSER_VERSION = 5

    def __init__(self):
        self.calls = 0

    def decode_log_bytes(self, raw):
        return raw.decode("utf-8"), "utf-8"

    def parse_log(self, text, filename, size):
        self.calls += 1
        return {"source": {"filename": filename, "file_size": size}, "samples": [], "filtered_text": text}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class EvalMonitorDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.target = self.home / "app"
        self.output = self.home / "cache"
        self.port = free_port()
        self.environment = {**os.environ, "HOME": str(self.home)}

    def tearDown(self):
        if (self.target / "monitor_config.json").is_file():
            subprocess.run(
                [sys.executable, str(TOOL), "stop", "--target-dir", str(self.target)],
                env=self.environment,
                text=True,
                capture_output=True,
                check=False,
            )
        self.temp.cleanup()

    def run_tool(self, *arguments):
        return subprocess.run(
            [sys.executable, str(TOOL), *arguments],
            env=self.environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_fresh_deploy_starts_unconfigured_loopback_monitor(self):
        result = self.run_tool(
            "deploy",
            "--target-dir", str(self.target),
            "--output-dir", str(self.output),
            "--port", str(self.port),
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/snapshot") as response:
            snapshot = json.load(response)
        self.assertFalse(snapshot["config"]["configured"])
        self.assertEqual(snapshot["evaluations"], [])

        status = self.run_tool("status", "--target-dir", str(self.target))
        self.assertEqual(status.returncode, 0, status.stderr)
        value = json.loads(status.stdout)
        self.assertTrue(value["running"])
        self.assertFalse(value["configured"])
        self.assertEqual(value["port"], self.port)

        config = json.loads((self.target / "monitor_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["bind_host"], "127.0.0.1")
        self.assertEqual(config["output_dir"], str(self.output))
        self.assertEqual(config["cookie_file"], str(self.target / "config" / "cookie"))
        self.assertEqual(config["project_id_file"], str(self.target / "config" / "project_id"))
        self.assertTrue((self.target / "config").is_dir())
        self.assertTrue((self.target / "dashboard.html").is_file())
        self.assertTrue((self.target / "eval_log_parser.py").is_file())
        self.assertIn(str(self.port), (self.target / "open_monitor_windows.ps1").read_text(encoding="utf-8"))
        self.assertFalse((self.home / ".config" / "streamlake" / "cookie").exists())

    def test_default_deploy_keeps_runtime_under_target(self):
        result = self.run_tool(
            "deploy",
            "--target-dir", str(self.target),
            "--port", str(self.port),
            "--no-start",
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        config = json.loads((self.target / "monitor_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["output_dir"], str(self.target / "data"))
        self.assertEqual(config["cookie_file"], str(self.target / "config" / "cookie"))
        self.assertEqual(config["project_id_file"], str(self.target / "config" / "project_id"))


class EvalMonitorAnalysisCacheTests(unittest.TestCase):
    def test_bundled_monitor_reuses_persistent_analysis_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation_id = "eval-task-persistent-1"
            log_path = root / "logs" / evaluation_id / "evaluation.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_text("persistent analysis", encoding="utf-8")
            parser = FakeParser()

            first = monitor.EvaluationRepository(root, parser)
            self.assertEqual(first._analysis(evaluation_id)["filtered_text"], "persistent analysis")
            self.assertEqual(parser.calls, 1)

            restarted = monitor.EvaluationRepository(root, parser)
            manager = monitor.AnalysisManager(restarted)
            self.assertFalse(manager.ensure_started())
            state = manager.snapshot()
            self.assertEqual(state["status"], "ready")
            self.assertEqual(state["cached"], 1)
            self.assertEqual(state["parsed"], 0)
            self.assertEqual(parser.calls, 1)


if __name__ == "__main__":
    unittest.main()
