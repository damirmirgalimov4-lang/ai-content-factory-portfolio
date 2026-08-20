from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent_platform.config import Settings
from agent_platform.polza import PolzaClient, PolzaError, PolzaVideoRequest


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
        polza_api_key="super-secret-test-key",
        polza_base_url="https://polza.ai/api",
        polza_max_status_retries=2,
    )


class FakeResponse:
    def __init__(self, payload: dict | bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.payload if isinstance(self.payload, bytes) else json.dumps(self.payload).encode()


class PolzaClientTest(unittest.TestCase):
    def _request(self, root: Path) -> PolzaVideoRequest:
        image = root / "frame.png"
        image.write_bytes(b"image")
        return PolzaVideoRequest(
            "bytedance/seedance-2",
            "move",
            image,
            5,
            "9:16",
            "std",
            False,
            resolution="720p",
        )

    def _kling_request(self, root: Path) -> PolzaVideoRequest:
        image = root / "frame.png"
        image.write_bytes(b"image")
        return PolzaVideoRequest(
            "kling/v3",
            "move",
            image,
            5,
            "9:16",
            "std",
            False,
            resolution="720p",
        )

    def test_create_uses_async_media_endpoint_and_returns_task_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = PolzaClient(settings(root), sleep=lambda _: None)
            captured = {}

            def fake_urlopen(request, timeout):
                captured["url"] = request.full_url
                captured["payload"] = json.loads(request.data)
                captured["authorization"] = request.headers["Authorization"]
                return FakeResponse({"id": "media-123", "status": "pending"})

            with patch("agent_platform.polza.urllib.request.urlopen", side_effect=fake_urlopen):
                task = client.create_video_task(self._request(root))
            self.assertEqual(task.task_id, "media-123")
            self.assertEqual(captured["url"], "https://polza.ai/api/v1/media")
            self.assertTrue(captured["payload"]["async"])
            self.assertEqual(captured["payload"]["model"], "bytedance/seedance-2")
            self.assertEqual(len(captured["payload"]["input"]["images"]), 1)
            self.assertEqual(captured["payload"]["input"]["duration"], "5")
            self.assertEqual(captured["payload"]["input"]["resolution"], "720p")
            self.assertEqual(captured["payload"]["input"]["generate_audio"], "false")
            self.assertEqual(captured["payload"]["input"]["multi_shots"], "false")
            self.assertNotIn("mode", captured["payload"]["input"])
            self.assertNotIn("sound", captured["payload"]["input"])

    def test_seedance_rejects_unsupported_duration_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request = replace(self._request(root), duration_seconds=6)
            with self.assertRaises(PolzaError) as caught:
                request.payload()
            self.assertIn("5, 10 или 15", str(caught.exception))

    def test_kling_standard_without_sound_uses_official_input_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = self._kling_request(root).payload()

            self.assertEqual(payload["model"], "kling/v3")
            self.assertEqual(payload["input"]["mode"], "std")
            self.assertEqual(payload["input"]["sound"], "false")
            self.assertEqual(payload["input"]["duration"], "5")
            self.assertEqual(payload["input"]["aspect_ratio"], "9:16")
            self.assertEqual(len(payload["input"]["images"]), 1)
            self.assertNotIn("resolution", payload["input"])
            self.assertNotIn("multi_shots", payload["input"])

    def test_missing_key_is_reported_without_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = PolzaClient(replace(settings(root), polza_api_key=""))
            with self.assertRaises(PolzaError):
                client.create_video_task(self._request(root))

    def test_http_errors_are_classified_and_secret_is_redacted(self) -> None:
        for code, retryable in ((401, False), (403, False), (429, True), (500, True)):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                client = PolzaClient(settings(root), sleep=lambda _: None)
                error = urllib.error.HTTPError(
                    "url", code, "error", {}, io.BytesIO(
                        b'{"error":{"message":"super-secret-test-key failed"}}'
                    )
                )
                with patch("agent_platform.polza.urllib.request.urlopen", side_effect=error):
                    with self.assertRaises(PolzaError) as caught:
                        client._request_json("GET", "/v1/media/id")
                self.assertEqual(caught.exception.retryable, retryable)
                self.assertEqual(caught.exception.status_code, code)
                self.assertNotIn("super-secret-test-key", str(caught.exception))
                if code in {401, 403}:
                    self.assertIn(f"HTTP {code}", str(caught.exception))
                    self.assertIn("Ответ сервиса", str(caught.exception))

    def test_ambiguous_create_network_error_is_not_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = PolzaClient(settings(root), sleep=lambda _: None)
            with patch(
                "agent_platform.polza.urllib.request.urlopen",
                side_effect=urllib.error.URLError("timeout"),
            ) as mocked:
                with self.assertRaises(PolzaError) as caught:
                    client.create_video_task(self._request(root))
            self.assertEqual(mocked.call_count, 1)
            self.assertTrue(caught.exception.ambiguous_submission)
            self.assertFalse(caught.exception.retryable)

    def test_wait_timeout_and_download_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = PolzaClient(replace(settings(root), polza_timeout_seconds=0), sleep=lambda _: None)
            with self.assertRaises(PolzaError) as caught:
                client.wait_for_task("task")
            self.assertIn("timeout", str(caught.exception))

            valid_mp4 = b"\x00\x00\x00\x18ftypisom" + b"data"
            with patch("agent_platform.polza.urllib.request.urlopen", return_value=FakeResponse(valid_mp4)):
                target = client.download_video("https://cdn.example/video.mp4", root / "video.mp4")
            self.assertTrue(target.is_file())


if __name__ == "__main__":
    unittest.main()
