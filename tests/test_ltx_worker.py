from __future__ import annotations

import base64
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from ltx_worker.api import create_server
from ltx_worker.config import WorkerSettings
from ltx_worker.errors import AmbiguousSubmissionError
from ltx_worker.service import JobConflict, JobRequest, VideoWorkerService
from ltx_worker.store import JobStore


PNG = b"\x89PNG\r\n\x1a\n" + b"offline-test-image"
MP4 = b"\x00\x00\x00\x18ftypisom" + b"offline-test-video"


class InlineExecutor:
    def submit(self, function, *args):
        function(*args)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        return None


class FakeRunner:
    def __init__(self) -> None:
        self.calls = 0

    def readiness(self) -> tuple[bool, str]:
        return True, "ready"

    def run(self, request: JobRequest, input_path: Path, output_path: Path) -> dict[str, object]:
        self.calls += 1
        self.last_request = request
        self.last_input = input_path.read_bytes()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(MP4)
        return {
            "duration_seconds": 5.0,
            "width": 1024,
            "height": 576,
            "has_video": True,
            "has_audio": True,
        }


def payload(job_id: str = "cf-test-v01", prompt: str = "A paper plane moves once.") -> dict[str, object]:
    return {
        "job_id": job_id,
        "model": "ltx-2.3",
        "workflow": "distilled",
        "prompt": prompt,
        "seed": 42,
        "duration_seconds": 5,
        "aspect_ratio": "16:9",
        "width": 1024,
        "height": 576,
        "fps": 24,
        "audio": True,
        "image_base64": base64.b64encode(PNG).decode("ascii"),
    }


class AmbiguousRunner(FakeRunner):
    def run(self, request: JobRequest, input_path: Path, output_path: Path) -> dict[str, object]:
        self.calls += 1
        raise AmbiguousSubmissionError("ComfyUI response was lost after POST")


class VideoWorkerCoreTest(unittest.TestCase):
    def test_same_job_and_fingerprint_runs_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runner = FakeRunner()
            service = VideoWorkerService(
                WorkerSettings.for_test(root),
                JobStore(root / "jobs.sqlite3"),
                runner,
                executor=InlineExecutor(),
            )

            first, first_created = service.submit(payload())
            second, second_created = service.submit(payload())

            self.assertTrue(first_created)
            self.assertFalse(second_created)
            self.assertEqual(first["job_id"], second["job_id"])
            self.assertEqual(second["status"], "completed")
            self.assertEqual(runner.calls, 1)
            self.assertEqual(runner.last_input, PNG)

    def test_reusing_job_id_with_changed_request_is_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = VideoWorkerService(
                WorkerSettings.for_test(root),
                JobStore(root / "jobs.sqlite3"),
                FakeRunner(),
                executor=InlineExecutor(),
            )
            service.submit(payload())

            with self.assertRaises(JobConflict):
                service.submit(payload(prompt="A different paid request."))

    def test_restart_never_requeues_a_job_marked_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3")
            request = JobRequest.from_payload(payload())
            store.reserve(request, root / "jobs" / request.job_id / "input.png")
            store.mark_running(request.job_id)
            runner = FakeRunner()

            VideoWorkerService(
                WorkerSettings.for_test(root),
                store,
                runner,
                executor=InlineExecutor(),
            )

            recovered = store.require(request.job_id)
            self.assertEqual(recovered["status"], "reconciliation_required")
            self.assertEqual(runner.calls, 0)

    def test_lost_submit_response_requires_reconciliation_and_never_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.sqlite3")
            runner = AmbiguousRunner()
            service = VideoWorkerService(
                WorkerSettings.for_test(root),
                store,
                runner,
                executor=InlineExecutor(),
            )

            first, created = service.submit(payload())
            second, created_again = service.submit(payload())

            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(first["status"], "reconciliation_required")
            self.assertEqual(second["status"], "reconciliation_required")
            self.assertEqual(runner.calls, 1)


class WorkerApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.settings = WorkerSettings.for_test(root, api_token="unit-test-token")
        self.service = VideoWorkerService(
            self.settings,
            JobStore(root / "jobs.sqlite3"),
            FakeRunner(),
            executor=InlineExecutor(),
        )
        self.server = create_server(("127.0.0.1", 0), self.settings, self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.service.close()
        self.temp.cleanup()

    def request(self, path: str, *, method: str = "GET", body=None, token: str = "unit-test-token"):
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Authorization": f"Bearer {token}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        return urllib.request.urlopen(
            urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method),
            timeout=3,
        )

    def test_video_routes_require_bearer_authentication(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/video/jobs/missing", token="wrong")
        self.assertEqual(caught.exception.code, 401)

    def test_ready_route_requires_bearer_authentication(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/ready", token="wrong")
        self.assertEqual(caught.exception.code, 401)

        with self.request("/ready") as response:
            body = json.load(response)
        self.assertEqual(response.status, 200)
        self.assertTrue(body["ready"])

    def test_async_contract_returns_status_and_authenticated_result(self) -> None:
        with self.request("/video/jobs", method="POST", body=payload()) as response:
            created = json.load(response)
        self.assertEqual(response.status, 201)
        self.assertEqual(created["status"], "completed")

        with self.request("/video/jobs/cf-test-v01") as response:
            status = json.load(response)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["result_url"], "/video/jobs/cf-test-v01/result")

        with self.request(status["result_url"]) as response:
            self.assertEqual(response.read(), MP4)
            self.assertEqual(response.headers.get_content_type(), "video/mp4")


if __name__ == "__main__":
    unittest.main()
