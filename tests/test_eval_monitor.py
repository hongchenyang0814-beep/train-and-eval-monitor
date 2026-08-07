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
MANIFEST_TOOL = ROOT / "scripts" / "experiment_manifest.py"
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
TRAINING_MODULE_PATH = ROOT / "assets" / "eval-monitor" / "training_monitor_server.py"
TRAINING_SPEC = importlib.util.spec_from_file_location("bundled_training_monitor_tests", TRAINING_MODULE_PATH)
training_module = importlib.util.module_from_spec(TRAINING_SPEC)
sys.modules[TRAINING_SPEC.name] = training_module
TRAINING_SPEC.loader.exec_module(training_module)


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


class ManualParser(FakeParser):
    def parse_log(self, text, filename, size):
        self.calls += 1
        return {
            "parser_version": self.PARSER_VERSION,
            "source": {"filename": filename, "file_size": size, "started_at": "", "ended_at": ""},
            "metadata": {"tasks": "['challenge_common_sense']"},
            "summary": {"task_count": 1},
            "tasks": [{"key": "challenge_common_sense", "label": "通用常识", "status": "已完成", "example_count": 1}],
            "samples": [{"id": "1", "task": "challenge_common_sense"}],
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
        self.assertEqual(config["huggingface_token_file"], str(self.target / "config" / "huggingface_token"))
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

        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/") as response:
            training_html = response.read().decode("utf-8")
        self.assertEqual(response.url, f"http://127.0.0.1:{self.port}/train/")
        self.assertIn("Training Monitor", training_html)
        self.assertIn('href="/"', training_html)
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/train/api/health") as response:
            training_health = json.load(response)
        self.assertEqual(training_health["monitor"], "training")
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/eval/") as response:
            evaluation_html = response.read().decode("utf-8")
        self.assertIn("Evaluation Monitor", evaluation_html)
        self.assertIn('href="/eval/"', evaluation_html)

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
        self.assertEqual(config["huggingface_token_file"], str(self.target / "config" / "huggingface_token"))
        self.assertEqual(config["training_config_file"], str(self.target / "training_monitor_config.json"))

    def test_deploy_records_only_existing_explicit_training_root(self):
        training_root = self.home / "friend-training-output"
        training_root.mkdir()
        result = self.run_tool(
            "deploy",
            "--target-dir", str(self.target),
            "--port", str(self.port),
            "--training-output-root", str(training_root),
            "--no-start",
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        training_config = json.loads((self.target / "training_monitor_config.json").read_text(encoding="utf-8"))
        self.assertEqual(training_config["outputs_roots"], [str(training_root.resolve())])
        self.assertIn("Training output roots:", result.stdout)

    def test_repair_paths_removes_stale_roots_and_targets(self):
        training_root = self.home / "friend-training-output"
        valid_target = training_root / "lora_sft" / "run-1"
        training_root.mkdir()
        valid_target.mkdir(parents=True)
        self.target.mkdir(parents=True)
        config = {
            "outputs_roots": [str(self.home / "stale-root"), str(training_root)],
            "targets": [
                {
                    "output_dir": str(valid_target),
                    "label": "valid",
                    "metrics_path": str(self.home / "stale-metrics.jsonl"),
                    "log_path": str(self.home / "stale-train.log"),
                    "pid": 999999,
                },
                {"output_dir": str(self.home / "old-run"), "label": "stale"},
            ],
        }
        (self.target / "training_monitor_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        result = self.run_tool("repair-paths", "--target-dir", str(self.target))
        self.assertEqual(result.returncode, 0, result.stderr)
        repaired = json.loads((self.target / "training_monitor_config.json").read_text(encoding="utf-8"))
        self.assertEqual(repaired["outputs_roots"], [str(training_root.resolve())])
        self.assertEqual(len(repaired["targets"]), 1)
        self.assertEqual(repaired["targets"][0]["output_dir"], str(valid_target.resolve()))
        self.assertNotIn("metrics_path", repaired["targets"][0])
        self.assertNotIn("log_path", repaired["targets"][0])
        self.assertNotIn("pid", repaired["targets"][0])
        self.assertIn('"removed_targets": 1', result.stdout)


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

    def test_manual_log_is_permanent_validated_and_bindable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "friend-eval.log"
            source.write_text("challenge_common_sense\n", encoding="utf-8")
            repository = monitor.EvaluationRepository(root, ManualParser())
            row = repository.add_manual_log(source, source.name)
            self.assertTrue(row["manual"])
            self.assertEqual(row["source_label"], "自主上传")
            evaluation_id = row["id"]
            self.assertTrue((root / "logs" / evaluation_id / "evaluation.log").is_file())
            self.assertTrue((root / "logs" / evaluation_id / "evaluation_note.md").is_file())
            self.assertTrue((root / "analysis_cache" / f"{evaluation_id}.json.gz").is_file())
            repository.bind_training(
                evaluation_id,
                {"id": "run-1", "label": "训练 1", "run_id": "run-1", "manifest": {"available": True}},
            )
            detail = repository.detail(evaluation_id)
            self.assertEqual(detail["evaluation"]["origin"], "manual_upload")
            self.assertEqual(detail["training_binding"]["id"], "run-1")
            restarted = monitor.EvaluationRepository(root, ManualParser())
            self.assertEqual(restarted.list_evaluations()[0]["id"], evaluation_id)

    def test_manifest_tool_creates_and_validates_required_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run-1"
            output.mkdir()
            (output / "training_config.yaml").write_text("model: test\n", encoding="utf-8")
            created = subprocess.run(
                [sys.executable, str(MANIFEST_TOOL), "init", "--output-dir", str(output)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            value = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            value.update({"title": "test", "purpose": "test", "hypothesis": "test", "changes": ["test"], "expected_result": "test", "notes": "test"})
            (output / "run_manifest.json").write_text(json.dumps(value), encoding="utf-8")
            validated = subprocess.run(
                [sys.executable, str(MANIFEST_TOOL), "validate", "--output-dir", str(output)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            rendered = subprocess.run(
                [sys.executable, str(MANIFEST_TOOL), "render", "--output-dir", str(output)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(rendered.returncode, 0, rendered.stderr)
            self.assertTrue((output / "training_task.md").is_file())


class HuggingFaceBindingTests(unittest.TestCase):
    def test_binding_only_stores_token_and_preserves_upload_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "training_monitor_config.json"
            upload_config = {
                "config_file": "training_config.yaml",
                "profiles": {"lora_sft": {
                    "hub_endpoint": "https://huggingface.co",
                    "hub_owner": "test-owner",
                    "hub_repo_prefix": "lora-sft",
                    "hub_private_repo": True,
                    "hub_index_repo_id": "",
                    "base_model_id": "org/base-model",
                }},
            }
            config_path.write_text(
                json.dumps({"outputs_roots": [str(root / "output")], "targets": [], "auto_upload": upload_config}),
                encoding="utf-8",
            )
            config = training_module.load_config(config_path)
            self.assertFalse(training_module.huggingface_status(config)["configured"])
            original_validator = training_module.validate_huggingface_write_access
            training_module.validate_huggingface_write_access = lambda _config, _token: None
            try:
                saved = training_module.save_huggingface_binding(
                    config,
                    {"token": "hf_first_test_token"},
                )
            finally:
                training_module.validate_huggingface_write_access = original_validator
            self.assertTrue(saved["configured"])
            self.assertTrue(saved["token_configured"])
            self.assertTrue(saved["upload_configured"])
            self.assertTrue(saved["upload_ready"])
            self.assertEqual(saved["owner"], "test-owner")
            self.assertEqual(json.loads(config_path.read_text(encoding="utf-8"))["auto_upload"]["profiles"]["lora_sft"]["hub_owner"], "test-owner")
            self.assertNotIn("hf_first_test_token", config_path.read_text(encoding="utf-8"))
            token_path = Path(config["huggingface_token_file"])
            self.assertEqual(token_path.read_text(encoding="utf-8").strip(), "hf_first_test_token")
            self.assertEqual(token_path.stat().st_mode & 0o777, 0o600)

            training_module.validate_huggingface_write_access = lambda _config, _token: None
            try:
                overwritten = training_module.save_huggingface_binding(
                    config,
                    {"token": "hf_second_test_token"},
                )
            finally:
                training_module.validate_huggingface_write_access = original_validator
            self.assertTrue(overwritten["configured"])
            self.assertEqual(overwritten["owner"], "test-owner")
            self.assertEqual(token_path.read_text(encoding="utf-8").strip(), "hf_second_test_token")

    def test_failed_write_validation_does_not_replace_the_existing_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "training_monitor_config.json"
            config_path.write_text(json.dumps({"outputs_roots": [], "targets": [], "auto_upload": {"profiles": {"group": {"hub_owner": "owner"}}}}), encoding="utf-8")
            config = training_module.load_config(config_path)
            token_path = Path(config["huggingface_token_file"])
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text("hf_previous\n", encoding="utf-8")
            original_validator = training_module.validate_huggingface_write_access
            training_module.validate_huggingface_write_access = lambda _config, _token: (_ for _ in ()).throw(ValueError("无 Write 权限"))
            try:
                with self.assertRaisesRegex(ValueError, "Write"):
                    training_module.save_huggingface_binding(config, {"token": "hf_rejected"})
            finally:
                training_module.validate_huggingface_write_access = original_validator
            self.assertEqual(token_path.read_text(encoding="utf-8").strip(), "hf_previous")

    def test_binding_rejects_upload_configuration_from_the_token_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "training_monitor_config.json"
            config_path.write_text(json.dumps({"outputs_roots": [str(root / "output")], "targets": []}), encoding="utf-8")
            config = training_module.load_config(config_path)
            with self.assertRaisesRegex(ValueError, "不支持"):
                training_module.save_huggingface_binding(config, {"token": "hf_test", "owner": "should-not-change"})


class EvaluationTrainingBindingTests(unittest.TestCase):
    def test_binding_keeps_task_identity_when_manifest_is_missing(self):
        binding = monitor.EvalMonitorServer.training_binding_for_task(
            {
                "id": "run-1-id",
                "label": "08051644",
                "run_id": "08051644",
                "output_dir": "/tmp/output/run-1",
                "run_manifest": {"available": False, "error": None},
            }
        )
        self.assertEqual(binding["label"], "08051644")
        self.assertEqual(binding["run_id"], "08051644")
        self.assertFalse(binding["manifest"]["available"])

    def test_discovered_task_exposes_manifest_title_without_requiring_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output" / "lora_sft" / "08051644"
            output.mkdir(parents=True)
            (output / "trainer_log.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
            (output / "training_config.yaml").write_text("model: test\n", encoding="utf-8")
            (output / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "08051644",
                        "title": "8K LoRA Linear kernel profiling",
                        "purpose": "test",
                        "hypothesis": "test",
                        "changes": ["test"],
                        "comparison_run": "baseline",
                        "dataset": {"summary": "test"},
                        "model": {"base_model": "test"},
                        "config_file": "training_config.yaml",
                        "expected_result": "test",
                        "notes": "test",
                        "created_at": "2026-08-07T00:00:00+08:00",
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "training_monitor_config.json"
            config_path.write_text(json.dumps({"outputs_roots": [str(root / "output")], "targets": []}), encoding="utf-8")
            config = training_module.load_config(config_path)
            target = training_module.discover_targets(config)[0]
            task = training_module.build_experiment(target, config, time.time())
            binding = monitor.EvalMonitorServer.training_binding_for_task(task)
            self.assertEqual(task["label"], "08051644")
            self.assertEqual(task["run_manifest"]["title"], "8K LoRA Linear kernel profiling")
            self.assertFalse(task["run_manifest"]["documentation"]["available"])
            self.assertEqual(binding["manifest"]["title"], "8K LoRA Linear kernel profiling")


class TrainingUploadLayoutTests(unittest.TestCase):
    def test_training_record_visibility_is_persisted_without_deleting_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "output"
            run = output_root / "lora_sft" / "run-001"
            run.mkdir(parents=True)
            log_path = run / "trainer_log.jsonl"
            log_path.write_text('{"step": 1, "loss": 0.4}\n', encoding="utf-8")
            config_path = root / "training_monitor_config.json"
            config_path.write_text(json.dumps({"outputs_roots": [str(output_root)], "targets": []}), encoding="utf-8")
            config = training_module.load_config(config_path)
            target = training_module.discover_targets(config)[0]

            saved = training_module.set_experiment_visibility(config, target["id"], True)
            self.assertTrue(saved["hidden"])
            self.assertEqual(json.loads(config_path.read_text(encoding="utf-8"))["hidden_experiment_ids"], [target["id"]])
            self.assertTrue(run.is_dir())
            self.assertTrue(log_path.is_file())
            self.assertEqual(training_module.build_snapshot(config)["experiments"], [])
            shown = training_module.build_snapshot(config, include_hidden=True)
            self.assertEqual(len(shown["experiments"]), 1)
            self.assertTrue(shown["experiments"][0]["hidden"])

            restored = training_module.set_experiment_visibility(config, target["id"], False)
            self.assertFalse(restored["hidden"])
            self.assertEqual(len(training_module.build_snapshot(config)["experiments"]), 1)

    def test_remote_upload_match_requires_the_adapter_file(self):
        target = {
            "output_dir": "/tmp/output/lora_sft/run-001",
            "enable_upload": True,
            "hub_endpoint": "https://hub.example",
            "hub_owner": "owner",
            "hub_repo_prefix": "lora-sft",
            "run_id": "run-001",
        }
        repo_id = "owner/lora-sft-run-001-step-00042"
        config = {
            "_hf_repo_cache": {"https://hub.example/owner": {"checked_at": time.time(), "repo_ids": [repo_id]}},
            "_hf_repo_file_cache": {repo_id: {"checked_at": time.time(), "has_adapter": True}},
        }
        checkpoints = [{"name": "checkpoint-42", "step": 42}]
        matched = training_module.remote_uploads_for_checkpoints(target, checkpoints, config)
        self.assertEqual(matched["checkpoint-42"]["repo_id"], repo_id)
        config["_hf_repo_file_cache"][repo_id]["has_adapter"] = False
        self.assertEqual(training_module.remote_uploads_for_checkpoints(target, checkpoints, config), {})

    def test_automatic_upload_matches_configured_one_or_two_level_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "output"
            valid_run = output_root / "lora_sft" / "run-001"
            nested_run = output_root / "ai_infra" / "benchmark" / "run-002"
            unconfigured_run = output_root / "other" / "run-003"
            invalid_run = output_root / "too" / "deep" / "for" / "run-004"
            for run in (valid_run, nested_run, unconfigured_run, invalid_run):
                run.mkdir(parents=True)
                (run / "trainer_log.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
                (run / "training_config.yaml").write_text("model: test\n", encoding="utf-8")
            upload_config = {
                "config_file": "training_config.yaml",
                "profiles": {
                    "lora_sft": {"hub_endpoint": "https://huggingface.co", "hub_owner": "test-owner", "hub_repo_prefix": "lora-sft", "base_model_id": "org/base-model"},
                    "ai_infra/benchmark": {"hub_endpoint": "https://huggingface.co", "hub_owner": "test-owner", "hub_repo_prefix": "benchmark", "base_model_id": "org/base-model"},
                },
            }
            config_path = root / "training_monitor_config.json"
            config_path.write_text(
                json.dumps({"outputs_roots": [str(output_root)], "targets": [], "auto_upload": upload_config}),
                encoding="utf-8",
            )
            config = training_module.load_config(config_path)
            discovered = training_module.discover_targets(config)
            by_dir = {item["output_dir"]: item for item in discovered}
            valid = by_dir[str(valid_run.resolve())]
            nested = by_dir[str(nested_run.resolve())]
            unconfigured = by_dir[str(unconfigured_run.resolve())]
            self.assertEqual(valid["run_id"], "run-001")
            self.assertEqual(valid["config_path"], str(valid_run / "training_config.yaml"))
            self.assertTrue(valid["enable_upload"])
            self.assertEqual(nested["run_id"], "run-002")
            self.assertTrue(nested["enable_upload"])
            self.assertFalse(unconfigured.get("enable_upload", False))
            invalid = {"output_dir": str(invalid_run)}
            self.assertIsNone(training_module.automatic_upload_layout(invalid, config))

    def test_explicit_target_cannot_override_the_required_run_local_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "output"
            run = output_root / "lora_sft" / "run-001"
            run.mkdir(parents=True)
            (run / "trainer_log.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
            (run / "training_config.yaml").write_text("model: test\n", encoding="utf-8")
            external_config = root / "external.yaml"
            external_config.write_text("model: wrong\n", encoding="utf-8")
            config_path = root / "training_monitor_config.json"
            config_path.write_text(
                json.dumps({
                    "outputs_roots": [str(output_root)],
                    "auto_upload": {"config_file": "training_config.yaml", "profiles": {"lora_sft": {"hub_owner": "test-owner"}}},
                    "targets": [{"output_dir": str(run), "config_path": str(external_config), "run_id": "run-001", "enable_upload": True}],
                }),
                encoding="utf-8",
            )
            config = training_module.load_config(config_path)
            target = training_module.discover_targets(config)[0]
            self.assertIsNone(training_module.automatic_upload_layout(target, config))
            self.assertFalse(training_module.build_experiment(target, config, 0)["upload_enabled"])


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


class TrainingDurationTests(unittest.TestCase):
    def test_manifest_duration_uses_wall_clock_start_and_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "lora_sft" / "run-001"
            output_dir.mkdir(parents=True)
            (output_dir / "trainer_log.jsonl").write_text('{"current_steps": 1}\n', encoding="utf-8")
            (output_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "started_at": "2026-08-04T03:16:50+08:00",
                        "finished_at": "2026-08-04T09:54:29+08:00",
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "outputs_roots": [str(root)],
                "stale_after_seconds": 600,
                "targets": [],
            }
            target = training_module.discover_targets({**config, "max_scan_depth": 3})[0]
            experiment = training_module.build_experiment(target, config, time.time())

        self.assertAlmostEqual(experiment["duration_seconds"], 23859.0)
        self.assertEqual(experiment["duration_source"], "manifest")

    def test_duration_falls_back_to_trainer_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "train_results.json").write_text('{"train_runtime": 12.5}\n', encoding="utf-8")

            duration, source = training_module.run_duration(output_dir, {}, 0)

        self.assertEqual(duration, 12.5)
        self.assertEqual(source, "trainer")


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

    def test_dashboard_places_log_download_in_detail_header(self):
        dashboard = (ROOT / "assets" / "eval-monitor" / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn("detail-actions", dashboard)
        self.assertIn("downloadSelectedLog", dashboard)
        self.assertIn("summary.log?.available", dashboard)
        self.assertIn("state.detail.analysis?.source?.filename", dashboard)

    def test_dashboard_places_upload_before_sync_in_toolbar(self):
        dashboard = (ROOT / "assets" / "eval-monitor" / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn('id="uploadBtn" class="upload"', dashboard)
        self.assertIn('id="syncBtn" class="primary"', dashboard)
        self.assertLess(dashboard.index('id="uploadBtn"'), dashboard.index('id="syncBtn"'))
        self.assertNotIn('id="manualUploadButton"', dashboard)

    def test_dashboard_surfaces_persistent_sync_health_and_last_sync(self):
        dashboard = (ROOT / "assets" / "eval-monitor" / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn('id="syncHealth"', dashboard)
        self.assertIn("sync_state?.updated_at", dashboard)
        self.assertIn("日志下载", dashboard)
        self.assertIn("数据健康需关注", dashboard)

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
        self.assertIn('href="/eval/"', dashboard)
        self.assertIn('href="/"', dashboard)

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
        self.assertIn("Training Monitor", dashboard)
        self.assertIn('href="/eval/"', dashboard)
        self.assertIn('href="/"', dashboard)
        self.assertIn('class="summary-strip"', dashboard)
        self.assertIn('class="summary-item"', dashboard)
        self.assertIn("Hugging Face Access Token", dashboard)
        self.assertIn("保存并覆盖 Access Token", dashboard)
        self.assertIn("Settings → Access Tokens", dashboard)
        self.assertIn("继续使用原训练监控配置", dashboard)
        self.assertIn("当前监控未配置HuggingFace 账号，请点击右上角齿轮配置", dashboard)
        self.assertIn("此任务未配置上传参数，请检查原训练监控配置", dashboard)
        self.assertIn('id="showHidden"', dashboard)
        self.assertIn('id="recordVisibilityButton"', dashboard)
        self.assertIn("experiment-visibility", dashboard)
        self.assertIn("不会删除原始目录或日志", dashboard)
        self.assertIn("function toast(message)", dashboard)
        self.assertNotIn("window.confirm", dashboard)

    def test_training_dashboard_formats_age_as_hours_then_days(self):
        dashboard = (ROOT / "assets" / "eval-monitor" / "training_dashboard.html").read_text(encoding="utf-8")
        self.assertIn("seconds < 86400", dashboard)
        self.assertIn("formatNumber(seconds / 3600, 1)", dashboard)
        self.assertIn("Math.floor(seconds / 86400)", dashboard)

        nav = dashboard.split('<nav class="workspace-tabs"', 1)[1].split("</nav>", 1)[0]
        self.assertLess(nav.index('href="/"'), nav.index('href="/eval/"'))

    def test_evaluation_dashboard_places_training_before_evaluation(self):
        dashboard = (ROOT / "assets" / "eval-monitor" / "dashboard.html").read_text(encoding="utf-8")
        nav = dashboard.split('<nav class="workspace-tabs"', 1)[1].split("</nav>", 1)[0]
        self.assertLess(nav.index('href="/"'), nav.index('href="/eval/"'))

    def test_connection_status_reports_credential_health(self):
        training = (ROOT / "assets" / "eval-monitor" / "training_dashboard.html").read_text(encoding="utf-8")
        evaluation = (ROOT / "assets" / "eval-monitor" / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn("Hugging Face 账号未绑定", training)
        self.assertIn("Token 无 Write 权限", training)
        self.assertIn("Token 具备 Write 权限", training)
        self.assertIn("Cookie 已配置", evaluation)
        self.assertIn("Cookie 已过期", evaluation)
        self.assertIn("本地服务连接失败", training)
        self.assertIn("本地服务连接失败", evaluation)
        style = "font-size: 11px; line-height: 1.4; font-weight: 400;"
        self.assertIn(style, training)
        self.assertIn(style, evaluation)


if __name__ == "__main__":
    unittest.main()
