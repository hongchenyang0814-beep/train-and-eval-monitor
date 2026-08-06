import importlib.util
import json
import os
import socket
import sqlite3
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
PARSER_MODULE_PATH = ROOT / "assets" / "eval-monitor" / "eval_log_parser.py"
PARSER_SPEC = importlib.util.spec_from_file_location("bundled_eval_log_parser_tests", PARSER_MODULE_PATH)
parser_module = importlib.util.module_from_spec(PARSER_SPEC)
sys.modules[PARSER_SPEC.name] = parser_module
PARSER_SPEC.loader.exec_module(parser_module)


class FakeParser:
    PARSER_VERSION = 5

    def __init__(self):
        self.calls = 0

    def decode_log_bytes(self, raw):
        return raw.decode("utf-8"), "utf-8"

    def parse_log(self, text, filename, size):
        self.calls += 1
        return {"source": {"filename": filename, "file_size": size}, "samples": [], "filtered_text": text}


class RankingParser(FakeParser):
    def parse_log(self, text, filename, size):
        self.calls += 1
        auto = {
            "sample_count": 1,
            "copy_answer": {
                "think": {"direct_copy_rate": 0.2},
                "no_think": {"direct_copy_rate": 0.1},
            },
            "think_no_think_overlap": {"overlap_rate": 0.4},
        }
        return {
            "source": {"filename": filename, "file_size": size},
            "summary": {"sample_detail_count": 1},
            "samples": [{"id": "1", "task": "challenge_recommendation_video"}],
            "tasks": [{"key": "challenge_recommendation_video", "automatic_metrics": auto}],
            "automatic_metrics": {"challenge_recommendation_video": auto},
            "filtered_text": text,
        }


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
        self.assertTrue(value["training_monitor"])

        config = json.loads((self.target / "monitor_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["bind_host"], "127.0.0.1")
        self.assertEqual(config["output_dir"], str(self.output))
        self.assertEqual(config["cookie_file"], str(self.target / "config" / "cookie"))
        self.assertEqual(config["project_id_file"], str(self.target / "config" / "project_id"))
        self.assertEqual(config["training_config_file"], str(self.target / "training_monitor_config.json"))
        self.assertEqual(config["training_upload_registry_file"], str(self.target / "training_upload_registry.json"))
        self.assertTrue((self.target / "config").is_dir())
        self.assertTrue((self.target / "dashboard.html").is_file())
        self.assertTrue((self.target / "eval_log_parser.py").is_file())
        self.assertTrue((self.target / "training_dashboard.html").is_file())
        self.assertTrue((self.target / "training_monitor_server.py").is_file())
        self.assertTrue((self.target / "training_monitor_config.json").is_file())
        self.assertIn(str(self.port), (self.target / "open_monitor_windows.ps1").read_text(encoding="utf-8"))
        self.assertFalse((self.home / ".config" / "streamlake" / "cookie").exists())

        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/train/") as response:
            training_html = response.read().decode("utf-8")
        self.assertIn("OneRec Training Monitor", training_html)
        self.assertIn('href="/"', training_html)
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/train/api/health") as response:
            training_health = json.load(response)
        self.assertEqual(training_health["monitor"], "training")

    def test_monitor_streams_complete_log_as_browser_attachment(self):
        result = self.run_tool(
            "deploy",
            "--target-dir", str(self.target),
            "--output-dir", str(self.output),
            "--port", str(self.port),
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        evaluation_id = "eval-task-download-1"
        payload = b"complete evaluation log\n" * 1024
        log_path = self.output / "logs" / evaluation_id / "evaluation.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_bytes(payload)

        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/api/evaluations/{evaluation_id}/download-log"
        ) as response:
            self.assertEqual(response.read(), payload)
            self.assertEqual(response.headers["Content-Type"], "application/octet-stream")
            self.assertEqual(
                response.headers["Content-Disposition"],
                f'attachment; filename="{evaluation_id}.log"',
            )

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
        self.assertEqual(config["training_config_file"], str(self.target / "training_monitor_config.json"))


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
            self.assertTrue(manager.ensure_started())
            for _ in range(100):
                if not manager.snapshot()["running"]:
                    break
                time.sleep(0.01)
            state = manager.snapshot()
            self.assertEqual(state["status"], "ready")
            self.assertEqual(state["cached"], 1)
            self.assertEqual(state["parsed"], 0)
            self.assertEqual(state["loaded"], 1)
            self.assertEqual(parser.calls, 1)
            self.assertIn(evaluation_id, restarted.cache)

    def test_ranking_rows_reuse_analysis_summary_without_reparsing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation_id = "eval-task-ranking-1"
            log_path = root / "logs" / evaluation_id / "evaluation.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_text("ranking analysis", encoding="utf-8")
            raw_json = {
                "detail": {
                    "taskStatus": "SUCCEEDED",
                    "modelName": "ranking-model",
                    "metrics": {
                        "metrics": {
                            "summary": {"totalScore": 1.23, "r0": 0.1, "r1": 0.2, "r2": 0.3, "r3": 0.4}
                        }
                    },
                }
            }
            with sqlite3.connect(root / "experiments.sqlite") as connection:
                connection.execute(
                    "CREATE TABLE experiments (experiment_type TEXT NOT NULL,id TEXT NOT NULL,name TEXT NOT NULL,status TEXT,created_at TEXT,updated_at TEXT,raw_path TEXT NOT NULL,raw_json TEXT NOT NULL,PRIMARY KEY (experiment_type,id))"
                )
                connection.execute(
                    "INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?)",
                    (
                        "evaluation",
                        evaluation_id,
                        "ranking run",
                        "SUCCEEDED",
                        "2026-08-05T00:00:00Z",
                        "2026-08-05T00:10:00Z",
                        "raw/ranking.json",
                        json.dumps(raw_json),
                    ),
                )
            parser = RankingParser()
            first = monitor.EvaluationRepository(root, parser)
            self.assertEqual(first._analysis(evaluation_id)["summary"]["sample_detail_count"], 1)
            self.assertTrue((root / "analysis_cache" / f"{evaluation_id}.summary.json").is_file())
            restarted = monitor.EvaluationRepository(root, parser)
            rows = restarted.ranking_rows()
            self.assertEqual(parser.calls, 1)
            self.assertEqual(rows[0]["score"], 1.23)
            self.assertEqual(rows[0]["analysis_summary"]["sample_count"], 1)
            self.assertIn("challenge_recommendation_video", rows[0]["analysis_summary"]["automatic_metrics"])


class EvalLogParserTests(unittest.TestCase):
    def test_loaded_sample_count_is_used_when_progress_bar_is_absent(self):
        text = """
[2026-08-05 00:00:00,000] [INFO] tasks      : ['challenge_evolution_action_select']
Task [1/1]: challenge_evolution_action_select | Split: test | Sample Size: N/A
[green]Loaded 574 samples for challenge_evolution_action_select[/green]
Using EvolutionSelectEvaluator for challenge_evolution_action_select
Updated sample metrics to:
/output/merged/challenge_evolution_action_select/test_generated.json
"""

        parsed = parser_module.parse_log(text, "evaluation.log", len(text.encode("utf-8")))
        task = next(item for item in parsed["tasks"] if item["key"] == "challenge_evolution_action_select")

        self.assertEqual(task["sample_count"], 574)
        self.assertEqual(task["status"], "已完成")
        self.assertEqual(task["metrics"], {})

    def test_unknown_itemic_tasks_fall_back_to_the_material_family(self):
        text = """
[2026-08-05 00:00:00,000] [INFO] tasks      : ['challenge_itemic_topk_video']
Task [1/1]: challenge_itemic_topk_video | Split: test | Sample Size: N/A
[green]Loaded 5 samples for challenge_itemic_topk_video[/green]
Using SomeEvaluator for challenge_itemic_topk_video
Updated sample metrics to:
/output/merged/challenge_itemic_topk_video/test_generated.json
"""

        parsed = parser_module.parse_log(text, "evaluation.log", len(text.encode("utf-8")))
        task = next(item for item in parsed["tasks"] if item["key"] == "challenge_itemic_topk_video")

        self.assertEqual(task["group"], "懂物料")
        self.assertEqual(task["label"], "challenge_itemic_topk_video")
        self.assertEqual(task["sample_count"], 5)
        self.assertEqual(task["status"], "已完成")

    def test_dashboard_explains_missing_task_summary_metrics(self):
        dashboard = (ROOT / "assets" / "eval-monitor" / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn("未提供任务级指标", dashboard)

    def test_dashboard_renders_recommendation_automatic_metrics(self):
        dashboard = (ROOT / "assets" / "eval-monitor" / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn("Think 抄答", dashboard)
        self.assertIn("No-think 抄答", dashboard)
        self.assertIn("两路 SID 重复", dashboard)
        self.assertIn("hallucination.hallucination_rate", dashboard)
        self.assertIn("repeat.repeated_sid_count", dashboard)
        self.assertIn("challenge_evolution_topic_gen'&&auto.sid_candidate_count", dashboard)
        self.assertIn("metric-chip platform", dashboard)
        self.assertIn("平台正式指标 · 权重", dashboard)
        self.assertIn("下载完整日志", dashboard)
        self.assertIn("/download-log", dashboard)
        self.assertIn("state.tab!=='log'", dashboard)
        self.assertIn("正在加载评测数据", dashboard)
        self.assertIn('data-tab="tools"', dashboard)
        self.assertIn('data-tab="rankings"', dashboard)
        self.assertIn("Skill CLI 能力", dashboard)
        self.assertIn("推荐平均 · Think 抄答率", dashboard)
        self.assertIn("/api/rankings", dashboard)
        self.assertIn('href="/train/"', dashboard)

    def test_dashboard_groups_samples_and_shows_task_workload(self):
        dashboard = (ROOT / "assets" / "eval-monitor" / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn("样本/耗时", dashboard)
        self.assertIn("展示样本", dashboard)
        self.assertIn("生成请求", dashboard)
        self.assertIn("generation_seconds", dashboard)
        self.assertIn("sample-group-title", dashboard)
        self.assertIn("sample-task-title", dashboard)
        for group in ("懂物料", "懂用户", "懂推荐", "懂世界"):
            self.assertIn(group, dashboard)

    def test_training_dashboard_has_workspace_switcher(self):
        dashboard = (ROOT / "assets" / "eval-monitor" / "training_dashboard.html").read_text(encoding="utf-8")
        self.assertIn("OneRec Training Monitor", dashboard)
        self.assertIn('href="/"', dashboard)
        self.assertIn('href="/train/"', dashboard)


if __name__ == "__main__":
    unittest.main()
