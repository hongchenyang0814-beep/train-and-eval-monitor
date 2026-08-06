import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "streamlake_experiments.py"


def load_module():
    spec = importlib.util.spec_from_file_location("streamlake_experiments", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def request_json(self, method, path, query=None, body=None):
        key = (method, path, tuple(sorted((query or {}).items())))
        self.calls.append(key)
        response = self.responses[key]
        if isinstance(response, Exception):
            raise response
        return response


class FakePostClient:
    def __init__(self):
        self.calls = []

    def request_json(self, method, path, query=None, body=None):
        self.calls.append((method, path, query, body))
        page = body["page"]
        if page == 1:
            return {"responseData": {"list": [{"taskId": "a"}], "total": 2}}
        return {"responseData": {"list": [{"taskId": "b"}], "total": 2}}


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_environment_cookie_has_priority(self):
        cookie_file = self.root / "cookie"
        cookie_file.write_text("file=value", encoding="utf-8")
        cookie_file.chmod(0o600)
        actual = self.module.load_cookie(cookie_file, {"STREAMLAKE_COOKIE": "env=value"})
        self.assertEqual(actual, "env=value")

    def test_default_cookie_file_uses_current_home(self):
        self.assertEqual(
            self.module.default_cookie_file({"HOME": "/tmp/alice"}),
            Path("/tmp/alice/.local/share/streamlake-eval-monitor/config/cookie"),
        )

    def test_cookie_file_environment_override(self):
        self.assertEqual(
            self.module.default_cookie_file({"STREAMLAKE_COOKIE_FILE": "/tmp/private-cookie"}),
            Path("/tmp/private-cookie"),
        )

    def test_default_project_id_file_uses_current_home(self):
        self.assertEqual(
            self.module.default_project_id_file({"HOME": "/tmp/alice"}),
            Path("/tmp/alice/.local/share/streamlake-eval-monitor/config/project_id"),
        )

    def test_project_id_resolves_from_environment(self):
        self.assertEqual(
            self.module.resolve_project_id(None, {"STREAMLAKE_PROJECT_ID": "proj-demo"}),
            "proj-demo",
        )

    def test_project_id_resolves_from_default_file(self):
        config_dir = self.root / ".local" / "share" / "streamlake-eval-monitor" / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "project_id").write_text("proj-file\n", encoding="utf-8")
        self.assertEqual(self.module.resolve_project_id(None, {"HOME": str(self.root)}), "proj-file")

    def test_project_id_cli_value_has_priority(self):
        self.assertEqual(
            self.module.resolve_project_id("proj-cli", {"STREAMLAKE_PROJECT_ID": "proj-env"}),
            "proj-cli",
        )

    def test_project_id_is_required(self):
        with self.assertRaisesRegex(ValueError, "project ID"):
            self.module.resolve_project_id(None, {"HOME": str(self.root)})

    def test_rejects_world_readable_cookie(self):
        cookie_file = self.root / "cookie"
        cookie_file.write_text("secret=value", encoding="utf-8")
        cookie_file.chmod(0o644)
        with self.assertRaisesRegex(PermissionError, "chmod 600"):
            self.module.load_cookie(cookie_file, {})

    def test_rejects_blank_cookie(self):
        cookie_file = self.root / "cookie"
        cookie_file.write_text("\n", encoding="utf-8")
        cookie_file.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "empty"):
            self.module.load_cookie(cookie_file, {})

    def test_redacts_credentials_and_signed_urls_recursively(self):
        value = {
            "sl-token": "secret-token",
            "nested": {
                "authorization": "Bearer secret",
                "artifactUrl": "https://files.example/x?X-Amz-Signature=secret&safe=1",
                "opaque": "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxIn0.signature",
                "score": 0.9,
            },
        }
        cleaned = self.module.sanitize(value)
        rendered = json.dumps(cleaned, ensure_ascii=False)
        self.assertNotIn("secret-token", rendered)
        self.assertNotIn("Bearer secret", rendered)
        self.assertNotIn("X-Amz-Signature", rendered)
        self.assertNotIn("eyJhbGci", rendered)
        self.assertEqual(cleaned["nested"]["score"], 0.9)

    def test_client_rejects_mutating_post_endpoint_before_network(self):
        client = self.module.StreamLakeClient("cookie=value")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            client.request_json("POST", "/api/customized/commercial/v1/train-task/terminate/t1")

    def test_client_rejects_download_get_endpoint_before_network(self):
        client = self.module.StreamLakeClient("cookie=value")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            client.request_json("GET", "/api/customized/commercial/v1/train-task/t1/model-package/download")

    def test_client_rejects_non_streamlake_origin_and_absolute_url(self):
        with self.assertRaisesRegex(ValueError, "StreamLake host"):
            self.module.StreamLakeClient("cookie=value", origin="https://attacker.invalid/api")
        client = self.module.StreamLakeClient("cookie=value")
        with self.assertRaisesRegex(ValueError, "relative"):
            client.request_json(
                "GET",
                "https://attacker.invalid/api/customized/commercial/v1/train-task/t1",
            )

    def test_redacts_camelcase_secrets_cookie_text_and_large_content(self):
        cleaned = self.module.sanitize({
            "accessToken": "a-secret",
            "apiKey": "k-secret",
            "message": "request failed Cookie: session=private-value",
            "content": "x" * (128 * 1024),
            "predictions": [{"text": "sample", "score": 0.5}] * 10000,
        })
        rendered = json.dumps(cleaned)
        self.assertNotIn("a-secret", rendered)
        self.assertNotIn("k-secret", rendered)
        self.assertNotIn("private-value", rendered)
        self.assertNotIn("x" * 1000, rendered)
        self.assertEqual(cleaned["predictions"]["_omitted"], "large-body")


class EvaluationLogDownloadTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_log_url_validation_rejects_non_https_and_unapproved_host(self):
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            self.module.validate_evaluation_log_url("http://p2-pro.safetyimg.com/example.log")
        self.module.validate_evaluation_log_url("https://p4-infra-fdl.yximgs.com/example.log")
        with self.assertRaisesRegex(ValueError, "safetyimg.com or yximgs.com"):
            self.module.validate_evaluation_log_url("https://attacker.invalid/example.log")
        with self.assertRaisesRegex(ValueError, "not a .log"):
            self.module.validate_evaluation_log_url("https://p2-pro.safetyimg.com/model.bin")

    def test_downloads_complete_log_without_forwarding_cookie(self):
        evaluation_id = "eval-task-demo-1"
        log_url = "https://p2-pro.safetyimg.com/example.log?signature=temporary"
        content = (b"evaluation-line\n" * 131072) + b"final-line\n"

        class Client:
            def request_json(self, method, path, query=None, body=None):
                if path.endswith(f"/{evaluation_id}"):
                    return {"responseData": {
                        "evalTaskId": evaluation_id,
                        "projectId": "proj-1",
                        "createTime": "2026-08-01T00:00:00Z",
                        "taskStatus": "SUCCEEDED",
                        "hasOutput": True,
                    }}
                if path.endswith(f"/{evaluation_id}/output"):
                    return {"responseData": log_url}
                raise AssertionError(path)

        class Headers:
            @staticmethod
            def get_content_type():
                return "text/plain"

        class Response:
            headers = Headers()

            def __init__(self):
                self.position = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def geturl(self):
                return log_url

            def read(self, size):
                chunk = content[self.position:self.position + size]
                self.position += len(chunk)
                return chunk

        class Opener:
            request = None

            def open(self, request, timeout):
                self.request = request
                return Response()

        opener = Opener()
        with mock.patch.object(self.module.urllib.request, "build_opener", return_value=opener):
            result = self.module.download_evaluation_log(Client(), "proj-1", evaluation_id, self.root)

        destination = self.root / evaluation_id / "evaluation.log"
        self.assertEqual(destination.read_bytes(), content)
        self.assertEqual(result["size_bytes"], len(content))
        self.assertNotIn("Cookie", dict(opener.request.header_items()))

    def test_sync_log_download_selects_only_succeeded_evaluations(self):
        records = [
            {
                "experiment_type": "evaluation",
                "id": "eval-task-success",
                "taskStatus": "SUCCEEDED",
                "hasOutput": True,
                "evaluation_output": "https://p2-pro.safetyimg.com/success.log",
            },
            {
                "experiment_type": "evaluation",
                "id": "eval-task-running",
                "taskStatus": "RUNNING",
                "hasOutput": False,
            },
            {"experiment_type": "finetune", "id": "train-task-1"},
        ]
        downloaded = {
            "evaluation_id": "eval-task-success",
            "path": "/logs/eval-task-success/evaluation.log",
            "size_bytes": 10,
            "sha256": "abc",
            "status": "existing",
        }
        with mock.patch.object(self.module, "download_evaluation_log_url", return_value=downloaded) as download:
            summary = self.module.download_synced_evaluation_logs(records, self.root)

        download.assert_called_once_with(
            "eval-task-success",
            "https://p2-pro.safetyimg.com/success.log",
            self.root,
            reuse_existing=True,
        )
        self.assertEqual(summary["eligible"], 1)
        self.assertEqual(summary["downloaded"], 0)
        self.assertEqual(summary["existing"], 1)
        self.assertEqual(summary["error_count"], 0)

    def test_log_download_summary_is_added_to_sync_state(self):
        self.module.write_repository(
            self.root,
            [{"experiment_type": "evaluation", "id": "eval-task-1"}],
            {"finetune": 0, "evaluation": 1},
        )
        summary = {"eligible": 1, "downloaded": 1, "existing": 0, "error_count": 0, "errors": [], "items": []}
        self.module._write_log_download_state(self.root, summary)
        state = json.loads((self.root / "sync_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["log_downloads"], summary)


class PaginationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_paginates_and_deduplicates_by_id(self):
        client = FakeClient(
            {
                ("GET", "/list", (("page", 1), ("pageSize", 2))): {
                    "data": {"list": [{"id": "a"}, {"id": "b"}], "total": 3}
                },
                ("GET", "/list", (("page", 2), ("pageSize", 2))): {
                    "data": {"list": [{"id": "b"}, {"id": "c"}], "total": 3}
                },
            }
        )
        endpoint = self.module.EndpointSpec(
            path="/list",
            items_path="data.list",
            pagination="page",
            page_param="page",
            page_size_param="pageSize",
            total_path="data.total",
        )
        rows = self.module.fetch_all_pages(client, endpoint, page_size=2)
        self.assertEqual([row["id"] for row in rows], ["a", "b", "c"])

    def test_stops_on_repeated_cursor(self):
        client = FakeClient(
            {
                ("GET", "/list", (("cursor", ""), ("limit", 1))): {
                    "items": [{"id": "a"}], "next": "same"
                },
                ("GET", "/list", (("cursor", "same"), ("limit", 1))): {
                    "items": [{"id": "b"}], "next": "same"
                },
            }
        )
        endpoint = self.module.EndpointSpec(
            path="/list",
            items_path="items",
            pagination="cursor",
            cursor_param="cursor",
            page_size_param="limit",
            next_cursor_path="next",
        )
        with self.assertRaisesRegex(RuntimeError, "repeated cursor"):
            self.module.fetch_all_pages(client, endpoint, page_size=1)

    def test_read_only_post_paginates_in_json_body(self):
        client = FakePostClient()
        endpoint = self.module.EndpointSpec(
            path="/list",
            method="POST",
            items_path="responseData.list",
            pagination="page",
            pagination_location="body",
            page_param="page",
            page_size_param="pageSize",
            total_path="responseData.total",
            static_body={"projectId": "proj-1", "keyword": ""},
            id_fields=("taskId",),
        )
        rows = self.module.fetch_all_pages(client, endpoint, page_size=1)
        self.assertEqual([row["taskId"] for row in rows], ["a", "b"])
        self.assertEqual(client.calls[0][3]["projectId"], "proj-1")
        self.assertEqual(client.calls[1][3]["page"], 2)


class FilteringTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_default_cutoff_keeps_boundary_and_newer_records_only(self):
        records = [
            {"id": "before", "created_at": "2026-07-31T23:59:59Z"},
            {"id": "boundary", "created_at": "2026-08-01T00:00:00Z"},
            {"id": "after", "created_at": "2026-08-01T00:00:01+00:00"},
            {"id": "missing"},
        ]
        filtered = self.module.filter_records_since(records)
        self.assertEqual([record["id"] for record in filtered], ["boundary", "after"])

    def test_cutoff_parses_millisecond_timestamps(self):
        since = self.module.parse_since("2026-08-01T00:00:00Z")
        self.assertEqual(
            [record["id"] for record in self.module.filter_records_since(
                [{"id": "old", "created_at": 1785542399000}], since
            )],
            [],
        )

    def test_invalid_cutoff_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid --since"):
            self.module.parse_since("not-a-date")

    def test_cutoff_before_august_first_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "restricted"):
            self.module.parse_since("2026-07-31T23:59:59Z")

    def test_internal_filter_rejects_cutoff_before_august_first(self):
        with self.assertRaisesRegex(ValueError, "restricted"):
            self.module.filter_records_since([], self.module.datetime(2026, 7, 31, tzinfo=self.module.timezone.utc))

    def test_failed_evaluation_without_output_is_excluded(self):
        records = [
            {"experiment_type": "evaluation", "id": "failed", "taskStatus": "FAILED", "hasOutput": False},
            {"experiment_type": "evaluation", "id": "failed-with-output", "taskStatus": "FAILED", "hasOutput": True},
            {"experiment_type": "evaluation", "id": "completed", "taskStatus": "COMPLETED", "hasOutput": False},
        ]
        filtered = self.module.filter_unavailable_evaluations(records)
        self.assertEqual([record["id"] for record in filtered], ["failed-with-output", "completed"])


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_sync_normalizes_both_experiment_types_and_skips_large_bodies(self):
        records = [
            {
                "experiment_type": "finetune",
                "id": "train-1",
                "name": "baseline",
                "status": "SUCCESS",
                "created_at": "2026-07-01T00:00:00Z",
                "parameters": {"learning_rate": 0.0001, "epochs": 2},
                "artifacts": [{"name": "checkpoint", "url": "https://x/signed?token=secret", "size": 10**9}],
                "checkpoint_bytes": b"do-not-store",
            },
            {
                "experiment_type": "evaluation",
                "id": "eval-1",
                "name": "baseline-eval",
                "status": "SUCCESS",
                "created_at": "2026-07-02T00:00:00Z",
                "train_experiment_id": "train-1",
                "metrics": {"overall": 0.75, "recall@20": 0.8},
            },
        ]
        result = self.module.write_repository(self.root, records, source_counts={"finetune": 1, "evaluation": 1})
        self.assertEqual(result.experiment_count, 2)
        connection = sqlite3.connect(self.root / "experiments.sqlite")
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0], 1)
        finally:
            connection.close()
        raw_text = "\n".join(path.read_text(encoding="utf-8") for path in (self.root / "raw").rglob("*.json"))
        self.assertNotIn("do-not-store", raw_text)
        self.assertNotIn("token=secret", raw_text)

    def test_failed_rewrite_preserves_existing_database(self):
        self.module.write_repository(
            self.root,
            [{"experiment_type": "finetune", "id": "old", "name": "old"}],
            source_counts={"finetune": 1, "evaluation": 0},
        )
        original = (self.root / "experiments.sqlite").read_bytes()
        with self.assertRaises(ValueError):
            self.module.write_repository(
                self.root,
                [{"experiment_type": "finetune", "name": "missing-id"}],
                source_counts={"finetune": 1, "evaluation": 0},
            )
        self.assertEqual((self.root / "experiments.sqlite").read_bytes(), original)

    def test_collects_details_and_metrics_for_both_streamlake_sources(self):
        class Client:
            def request_json(self, method, path, query=None, body=None):
                if path.endswith("/train-task/list"):
                    return {"responseData": {"list": [{"taskId": "t1", "taskName": "train", "fineTuningType": "SFT", "createTime": "2026-08-01T00:00:00Z"}], "total": 1}}
                if path.endswith("/train-task/t1"):
                    return {"responseData": {"taskId": "t1", "hyperParams": {"learningRate": 0.001}}}
                if path.endswith("/analysis/dashboard"):
                    return {"responseData": {"panels": [{"metrics": [{"name": "train_loss"}]}]}}
                if path.endswith("/metric-query"):
                    return {"responseData": {"series": [{"name": "train_loss", "points": [{"step": 1, "value": 2.0}, {"step": 2, "value": 1.0}]}]}}
                if path.endswith("/competition-eval-task/list"):
                    return {"responseData": {"items": [
                        {
                            "evalTaskId": "e1",
                            "taskName": "eval",
                            "r1": 0.8,
                            "taskStatus": "SUCCEEDED",
                            "hasOutput": True,
                            "createTime": "2026-08-01T00:00:00Z",
                        },
                        {"evalTaskId": "failed", "taskStatus": "FAILED", "hasOutput": False, "createTime": "2026-08-01T00:00:00Z"},
                    ], "total": 2}}
                if path.endswith("/competition-eval-task/e1/output"):
                    return {"responseData": {"overallScore": 0.9}}
                if path.endswith("/competition-eval-task/e1"):
                    return {"responseData": {"evalTaskId": "e1", "modelName": "model"}}
                raise AssertionError((method, path, query, body))

        records, counts, errors = self.module.collect_streamlake_records(Client(), "proj-1", page_size=100)
        self.assertEqual(counts, {"finetune": 1, "evaluation": 1})
        self.assertEqual(errors, [])
        by_id = {record["id"]: record for record in records}
        self.assertEqual(by_id["t1"]["metrics"]["train_loss"], 1.0)
        self.assertEqual(by_id["e1"]["metrics"]["r1"], 0.8)
        self.assertEqual(by_id["e1"]["metrics"]["output.overallScore"], 0.9)

    def test_authentication_failure_during_details_aborts_collection(self):
        class Client:
            def request_json(self, method, path, query=None, body=None):
                if path.endswith("/train-task/list"):
                    return {"responseData": {"list": [{"taskId": "t1", "createTime": "2026-08-01T00:00:00Z"}], "total": 1}}
                if path.endswith("/competition-eval-task/list"):
                    return {"responseData": {"items": [], "total": 0}}
                raise PermissionError("authentication expired")

        with self.assertRaisesRegex(PermissionError, "authentication expired"):
            self.module.collect_streamlake_records(Client(), "proj-1")

    def test_running_evaluation_is_synced_without_requesting_output(self):
        class Client:
            def __init__(self):
                self.paths = []

            def request_json(self, method, path, query=None, body=None):
                self.paths.append(path)
                if path.endswith("/train-task/list"):
                    return {"responseData": {"list": [], "total": 0}}
                if path.endswith("/competition-eval-task/list"):
                    return {"responseData": {"items": [{
                        "evalTaskId": "eval-task-running",
                        "taskStatus": "RUNNING",
                        "hasOutput": False,
                        "createTime": "2026-08-01T00:00:00Z",
                    }], "total": 1}}
                if path.endswith("/competition-eval-task/eval-task-running"):
                    return {"responseData": {
                        "evalTaskId": "eval-task-running",
                        "taskStatus": "RUNNING",
                        "hasOutput": False,
                    }}
                raise AssertionError(path)

        client = Client()
        records, counts, errors = self.module.collect_streamlake_records(client, "proj-1")
        self.assertEqual(counts, {"finetune": 0, "evaluation": 1})
        self.assertEqual(errors, [])
        self.assertEqual(records[0]["evaluation_output_status"], "pending")
        self.assertFalse(any(path.endswith("/output") for path in client.paths))

    def test_metric_permission_error_continues_when_auth_probe_succeeds(self):
        class Client:
            def request_json(self, method, path, query=None, body=None):
                if path.endswith("/train-task/list"):
                    return {"responseData": {"list": [{"taskId": "t1", "createTime": "2026-08-01T00:00:00Z"}], "total": 1}}
                if path.endswith("/competition-eval-task/list"):
                    return {"responseData": {"items": [], "total": 0}}
                if path.endswith("/train-task/t1"):
                    return {"responseData": {"taskId": "t1", "fineTuningType": "SFT"}}
                if path.endswith("/analysis/dashboard"):
                    return {"responseData": {"metrics": [{"name": "loss"}]}}
                if path.endswith("/metric-query"):
                    raise PermissionError("endpoint forbidden")
                raise AssertionError(path)

        records, counts, errors = self.module.collect_streamlake_records(Client(), "proj-1")
        self.assertEqual(len(records), 1)
        self.assertEqual(counts["finetune"], 1)
        self.assertEqual(errors[0]["endpoint"], "metrics")

    def test_sync_result_reports_partial_errors(self):
        result = self.module.write_repository(
            self.root,
            [{"experiment_type": "finetune", "id": "t1"}],
            {"finetune": 1, "evaluation": 0},
            [{"experiment_type": "finetune", "id": "t1", "endpoint": "metric", "error": "timeout"}],
        )
        self.assertEqual(result.error_count, 1)
        connection = sqlite3.connect(self.root / "experiments.sqlite")
        try:
            self.assertEqual(connection.execute("SELECT outcome FROM sync_runs").fetchone()[0], "partial")
        finally:
            connection.close()


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.module.write_repository(
            self.root,
            [
                {"experiment_type": "evaluation", "id": "e1", "name": "same", "metrics": {"score": 0.5, "loss": 2.0}},
                {"experiment_type": "evaluation", "id": "e2", "name": "same", "metrics": {"score": 0.6, "loss": 1.5}, "parameters": {"learning_rate": 0.001}},
                {"experiment_type": "evaluation", "id": "e3", "name": "third", "metrics": {"score": 0.55}},
            ],
            source_counts={"finetune": 0, "evaluation": 3},
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_ambiguous_name_lists_candidate_ids(self):
        connection = sqlite3.connect(self.root / "experiments.sqlite")
        try:
            with self.assertRaisesRegex(ValueError, "e1.*e2"):
                self.module.resolve_experiments(connection, ["same"])
        finally:
            connection.close()

    def test_compare_reports_metric_deltas_and_missing_values(self):
        connection = sqlite3.connect(self.root / "experiments.sqlite")
        try:
            report = self.module.compare_experiments(connection, ["e1", "e2", "e3"], baseline="e1", primary_metric="score")
        finally:
            connection.close()
        self.assertIn("e2", report)
        self.assertIn("+0.100000", report)
        self.assertIn("loss", report)
        self.assertIn("missing", report.lower())
        self.assertIn("correlation", report.lower())
        self.assertIn("learning_rate", report)


if __name__ == "__main__":
    unittest.main()
