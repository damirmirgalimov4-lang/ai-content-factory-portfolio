from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.error
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent_platform.config import Settings
from agent_platform.video_provider import VideoGenerationRequest
from agent_platform.viktor import ViktorClient, ViktorError


def settings(root: Path) -> Settings:
    return Settings(
        telegram_bot_token="test",
        telegram_allowed_user_ids={1},
        vault_path=root,
        openai_api_key="",
        openai_base_url="https://api.openai.com/v1",
        openai_image_model="gpt-image-2",
        openai_image_size="1024x1536",
        openai_image_quality="low",
        deepgram_api_key="",
        deepgram_model="nova-3",
        deepgram_language="ru",
        viktor_api_key="example-viktor-key",
    )


class FakeResponse:
    def __init__(self, payload: dict | bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


class ViktorClientTest(unittest.TestCase):
    def _request(self, root: Path) -> VideoGenerationRequest:
        return VideoGenerationRequest(
            model="bytedance/seedance-2",
            prompt=(
                "A red paper airplane glides once through a bright white studio. "
                "The camera slowly tracks from left to right."
            ),
            image_path=None,
            duration_seconds=15,
            aspect_ratio="9:16",
            mode="",
            sound_enabled=False,
            user="VIKTOR-SMOKE-001",
            resolution="720p",
            provider="viktor",
        )

    def test_settings_loads_viktor_variables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        "TELEGRAM_BOT_TOKEN=test",
                        "TELEGRAM_ALLOWED_USER_IDS=1",
                        f"VAULT_PATH={root / 'vault'}",
                        "VIKTOR_API_KEY=example-viktor-key",
                        "VIKTOR_BASE_URL=https://api.viktor.example/",
                        "VIKTOR_POLL_INTERVAL_SECONDS=9",
                        "VIKTOR_TIMEOUT_SECONDS=321",
                        "VIKTOR_MAX_STATUS_RETRIES=4",
                    )
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                loaded = Settings.load(env_path)

            self.assertEqual(loaded.viktor_api_key, "example-viktor-key")
            self.assertEqual(loaded.viktor_base_url, "https://api.viktor.example")
            self.assertEqual(loaded.viktor_poll_interval_seconds, 9)
            self.assertEqual(loaded.viktor_timeout_seconds, 321)
            self.assertEqual(loaded.viktor_max_status_retries, 4)

    def test_create_uses_official_thread_endpoint_and_idempotency_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = ViktorClient(settings(root), sleep=lambda _: None)
            captured = {}

            def fake_urlopen(request, timeout):
                captured["url"] = request.full_url
                captured["payload"] = json.loads(request.data)
                captured["headers"] = request.headers
                return FakeResponse(
                    {
                        "thread": {"id": "thread-1"},
                        "message": {"id": "message-1"},
                        "run": {"id": "run-1", "status": "queued"},
                    }
                )

            with patch(
                "agent_platform.viktor.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ):
                task = client.create_video_task(self._request(root))

            self.assertEqual(task.task_id, "run-1")
            self.assertEqual(task.status, "pending")
            self.assertEqual(
                captured["url"],
                "https://api.viktor.com/api/public/v1/threads",
            )
            self.assertEqual(captured["payload"]["speed"], "smarter")
            self.assertEqual(
                captured["payload"]["metadata"]["duration_seconds"], 15
            )
            self.assertIn("Seedance 2", captured["payload"]["message"])
            self.assertIn("ровно ОДНО", captured["payload"]["message"])
            self.assertIn(
                "не передавай генератору аргумент resolution",
                captured["payload"]["message"],
            )
            self.assertIn(
                "После принятия media task не создавай новый",
                captured["payload"]["message"],
            )
            self.assertIn(
                "продолжай безопасный polling того же task ID",
                captured["payload"]["message"],
            )
            self.assertIn(
                "Не завершай текущий run текстовым отчётом",
                captured["payload"]["message"],
            )
            self.assertTrue(
                captured["headers"]["Idempotency-key"].startswith(
                    "content-factory-"
                )
            )
            self.assertEqual(
                captured["headers"]["Authorization"],
                "Bearer example-viktor-key",
            )
            self.assertNotIn(
                "example-viktor-key",
                json.dumps(captured["payload"], ensure_ascii=False),
            )

    def test_same_request_produces_same_idempotency_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = ViktorClient(settings(root))
            request = self._request(root)
            self.assertEqual(
                client._idempotency_key(request),
                client._idempotency_key(request),
            )
            self.assertNotEqual(
                client._idempotency_key(request),
                client._idempotency_key(replace(request, user="another-run")),
            )

    def test_create_does_not_retry_ambiguous_network_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ViktorClient(settings(Path(temp_dir)), sleep=lambda _: None)
            with patch(
                "agent_platform.viktor.urllib.request.urlopen",
                side_effect=urllib.error.URLError("lost response"),
            ) as mocked:
                with self.assertRaises(ViktorError) as caught:
                    client.create_video_task(self._request(Path(temp_dir)))

            self.assertEqual(mocked.call_count, 1)
            self.assertTrue(caught.exception.ambiguous_submission)
            self.assertFalse(caught.exception.retryable)

    def test_missing_run_id_blocks_automatic_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ViktorClient(settings(Path(temp_dir)))
            response = FakeResponse(
                {
                    "thread": {"id": "thread-1"},
                    "message": {"id": "message-1"},
                    "run": {"status": "queued"},
                }
            )
            with patch(
                "agent_platform.viktor.urllib.request.urlopen",
                return_value=response,
            ):
                with self.assertRaises(ViktorError) as caught:
                    client.create_video_task(self._request(Path(temp_dir)))
            self.assertTrue(caught.exception.ambiguous_submission)

    def test_local_reference_is_rejected_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "frame.png"
            image.write_bytes(b"frame")
            request = replace(self._request(root), image_path=image)
            client = ViktorClient(settings(root))
            with patch(
                "agent_platform.viktor.urllib.request.urlopen"
            ) as mocked:
                with self.assertRaisesRegex(
                    ViktorError,
                    "не принимает локальные reference images",
                ):
                    client.create_video_task(request)
            mocked.assert_not_called()

    def test_non_default_seedance_resolution_is_rejected_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request = replace(self._request(root), resolution="480p")
            client = ViktorClient(settings(root))
            with patch(
                "agent_platform.viktor.urllib.request.urlopen"
            ) as mocked:
                with self.assertRaisesRegex(
                    ViktorError,
                    "возвращает 720p по умолчанию",
                ):
                    client.create_video_task(request)
            mocked.assert_not_called()

    def test_sound_request_is_explicit_in_agent_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request = replace(
                self._request(Path(temp_dir)),
                sound_enabled=True,
            )
            message = ViktorClient._build_generation_message(request)
            self.assertIn("со звуком", message)

    def test_completed_run_reads_video_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ViktorClient(settings(Path(temp_dir)), sleep=lambda _: None)
            responses = [
                FakeResponse(
                    {
                        "id": "run-1",
                        "thread_id": "thread-1",
                        "status": "completed",
                        "created_at": "2026-08-02T00:00:00Z",
                        "started_at": "2026-08-02T00:00:01Z",
                        "completed_at": "2026-08-02T00:01:00Z",
                        "failed_at": None,
                        "cancelled_at": None,
                        "result": {},
                    }
                ),
                FakeResponse(
                    {
                        "run_id": "run-1",
                        "status": "completed",
                        "artifacts": [
                            {
                                "id": "file-1",
                                "download_url": "https://cdn.example/result.mp4",
                                "display_name": "result.mp4",
                                "content_type": "video/mp4",
                            }
                        ],
                    }
                ),
                FakeResponse(
                    {
                        "run_id": "run-1",
                        "url": "https://cdn.example/result.mp4",
                        "expires_at": "2026-08-02T00:16:00Z",
                    }
                ),
            ]
            with patch(
                "agent_platform.viktor.urllib.request.urlopen",
                side_effect=responses,
            ) as mocked:
                task = client.get_task("run-1")

            self.assertEqual(mocked.call_count, 3)
            self.assertEqual(
                mocked.call_args_list[2].args[0].full_url,
                "https://api.viktor.com/api/public/v1/files/file-1/download-url",
            )
            self.assertEqual(task.status, "completed")
            self.assertEqual(task.result_url, "https://cdn.example/result.mp4")

    def test_completed_run_without_video_artifact_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ViktorClient(settings(Path(temp_dir)), sleep=lambda _: None)
            responses = [
                FakeResponse({"id": "run-1", "status": "completed"}),
                FakeResponse(
                    {
                        "run_id": "run-1",
                        "status": "completed",
                        "artifacts": [
                            {
                                "id": "file-1",
                                "download_url": "https://cdn.example/report.txt",
                                "display_name": "report.txt",
                                "content_type": "text/plain",
                            }
                        ],
                    }
                ),
            ]
            with patch(
                "agent_platform.viktor.urllib.request.urlopen",
                side_effect=responses,
            ):
                task = client.get_task("run-1")
            self.assertEqual(task.status, "failed")
            self.assertIn("не вернул MP4/video artifact", task.error)

    def test_requires_action_is_not_approved_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ViktorClient(settings(Path(temp_dir)))
            with patch(
                "agent_platform.viktor.urllib.request.urlopen",
                return_value=FakeResponse(
                    {"id": "run-1", "status": "requires_action"}
                ),
            ):
                task = client.get_task("run-1")
            self.assertEqual(task.status, "failed")
            self.assertIn("ручное действие", task.error)

    def test_safe_status_request_retries_but_submission_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ViktorClient(settings(Path(temp_dir)), sleep=lambda _: None)
            with patch(
                "agent_platform.viktor.urllib.request.urlopen",
                side_effect=[
                    urllib.error.URLError("temporary"),
                    FakeResponse({"id": "run-1", "status": "queued"}),
                ],
            ) as mocked:
                task = client.get_task("run-1")
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(task.status, "pending")

    def test_http_errors_are_normalized_and_secret_is_redacted(self) -> None:
        cases = {
            401: (False, "API-ключ отклонён"),
            403: (False, "не хватает прав"),
            429: (True, "частоту запросов"),
            500: (True, "серверная ошибка"),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            for status_code, (retryable, expected) in cases.items():
                with self.subTest(status_code=status_code):
                    client = ViktorClient(settings(Path(temp_dir)))
                    body = json.dumps(
                        {
                            "detail": {
                                "error": "test",
                                "message": "example-viktor-key",
                            }
                        }
                    ).encode()
                    error = urllib.error.HTTPError(
                        "https://api.viktor.com/api/public/v1/test",
                        status_code,
                        "error",
                        {},
                        io.BytesIO(body),
                    )
                    with patch(
                        "agent_platform.viktor.urllib.request.urlopen",
                        side_effect=error,
                    ):
                        with self.assertRaises(ViktorError) as caught:
                            client.test_connection()
                    self.assertEqual(caught.exception.retryable, retryable)
                    self.assertIn(expected, str(caught.exception))
                    self.assertNotIn(
                        "example-viktor-key",
                        str(caught.exception),
                    )

    def test_download_verifies_video_container(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = ViktorClient(settings(root))
            content = b"\x00\x00\x00\x18ftypisomvideo"
            with patch(
                "agent_platform.viktor.urllib.request.urlopen",
                return_value=FakeResponse(content),
            ):
                target = client.download_video(
                    "https://cdn.example/result.mp4",
                    root / "result.mp4",
                )
            self.assertEqual(target.read_bytes(), content)

    def test_invalid_download_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = ViktorClient(settings(root))
            with patch(
                "agent_platform.viktor.urllib.request.urlopen",
                return_value=FakeResponse(b"not-video-content"),
            ):
                with self.assertRaisesRegex(ViktorError, "не похож"):
                    client.download_video(
                        "https://cdn.example/result.mp4",
                        root / "result.mp4",
                    )


if __name__ == "__main__":
    unittest.main()
