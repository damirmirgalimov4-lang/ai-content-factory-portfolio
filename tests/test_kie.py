from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from agent_platform.config import Settings
from agent_platform.kie import KieClient, KieError
from agent_platform.video_provider import VideoGenerationRequest


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
        kie_api_key="super-secret-kie-key",
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


class KieClientTest(unittest.TestCase):
    def _request(
        self,
        root: Path,
        *,
        with_frame: bool = True,
        sound_enabled: bool = False,
    ) -> VideoGenerationRequest:
        image = root / "frame.png"
        if with_frame:
            image.write_bytes(b"frame")
        return VideoGenerationRequest(
            model="bytedance/seedance-2",
            prompt="A red ball jumps once while the camera remains still.",
            image_path=image if with_frame else None,
            duration_seconds=5,
            aspect_ratio="9:16",
            mode="",
            sound_enabled=sound_enabled,
            resolution="480p",
            provider="kie",
        )

    def test_create_uploads_frame_and_uses_official_seedance_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = KieClient(settings(root), sleep=lambda _: None)
            requests = []

            def fake_urlopen(request, timeout):
                payload = json.loads(request.data) if request.data else None
                requests.append((request.full_url, payload, request.headers))
                if request.full_url.endswith("/api/file-base64-upload"):
                    return FakeResponse(
                        {
                            "success": True,
                            "code": 200,
                            "data": {
                                "fileUrl": "https://tempfile.example/frame.png"
                            },
                        }
                    )
                return FakeResponse(
                    {
                        "code": 200,
                        "msg": "success",
                        "data": {"taskId": "task-seedance-1"},
                    }
                )

            with patch(
                "agent_platform.kie.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ):
                task = client.create_video_task(self._request(root))

            self.assertEqual(task.task_id, "task-seedance-1")
            self.assertEqual(len(requests), 2)
            create_url, payload, headers = requests[1]
            self.assertEqual(
                create_url,
                "https://api.kie.ai/api/v1/jobs/createTask",
            )
            self.assertEqual(payload["model"], "bytedance/seedance-2")
            self.assertEqual(payload["input"]["resolution"], "480p")
            self.assertEqual(payload["input"]["duration"], 5)
            self.assertEqual(payload["input"]["aspect_ratio"], "9:16")
            self.assertFalse(payload["input"]["generate_audio"])
            self.assertEqual(
                payload["input"]["first_frame_url"],
                "https://tempfile.example/frame.png",
            )
            self.assertNotIn("super-secret-kie-key", json.dumps(payload))
            self.assertTrue(headers["Authorization"].startswith("Bearer "))
            upload_headers = requests[0][2]
            self.assertIn("Mozilla/5.0", upload_headers["User-agent"])
            self.assertIn("application/json", upload_headers["Accept"])

    def test_create_enables_seedance_audio_when_profile_requests_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = KieClient(settings(root), sleep=lambda _: None)
            create_payload = {}

            def fake_urlopen(request, timeout):
                payload = json.loads(request.data) if request.data else None
                if request.full_url.endswith("/api/file-base64-upload"):
                    return FakeResponse(
                        {
                            "success": True,
                            "code": 200,
                            "data": {"fileUrl": "https://tempfile.example/frame.png"},
                        }
                    )
                create_payload.update(payload)
                return FakeResponse(
                    {"code": 200, "msg": "success", "data": {"taskId": "task-audio"}}
                )

            with patch(
                "agent_platform.kie.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ):
                task = client.create_video_task(
                    self._request(root, sound_enabled=True)
                )

            self.assertEqual(task.task_id, "task-audio")
            self.assertTrue(create_payload["input"]["generate_audio"])
            self.assertEqual(create_payload["input"]["resolution"], "480p")

    def test_text_to_video_does_not_upload_a_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = KieClient(settings(root), sleep=lambda _: None)
            captured = {}

            def fake_urlopen(request, timeout):
                captured["payload"] = json.loads(request.data)
                return FakeResponse(
                    {"code": 200, "data": {"taskId": "task-text"}}
                )

            with patch(
                "agent_platform.kie.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ) as mocked:
                task = client.create_video_task(
                    self._request(root, with_frame=False)
                )

            self.assertEqual(task.task_id, "task-text")
            self.assertEqual(mocked.call_count, 1)
            self.assertNotIn("first_frame_url", captured["payload"]["input"])

    def test_multireference_mode_preserves_image_order_and_omits_first_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            references = tuple(root / f"ref-{index}.png" for index in range(1, 4))
            for index, path in enumerate(references, 1):
                path.write_bytes(f"reference-{index}".encode())
            client = KieClient(settings(root), sleep=lambda _: None)
            captured: dict[str, dict] = {}

            def fake_urlopen(request, timeout):
                payload = json.loads(request.data) if request.data else None
                if request.full_url.endswith("/api/file-base64-upload"):
                    file_name = payload["fileName"]
                    return FakeResponse(
                        {
                            "success": True,
                            "code": 200,
                            "data": {
                                "fileUrl": f"https://tempfile.example/{file_name}"
                            },
                        }
                    )
                captured["payload"] = payload
                return FakeResponse(
                    {"code": 200, "data": {"taskId": "task-multiref"}}
                )

            request = VideoGenerationRequest(
                model="bytedance/seedance-2",
                prompt=(
                    "@Image1 是首帧，@Image2 是人物，@Image3 是物体。"
                    "[0:00–5s] @Image1 @Image2 打开 @Image3。"
                ),
                image_path=None,
                duration_seconds=5,
                aspect_ratio="9:16",
                mode="",
                sound_enabled=False,
                resolution="480p",
                provider="kie",
                reference_image_paths=references,
            )

            with patch(
                "agent_platform.kie.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ):
                task = client.create_video_task(request)

            model_input = captured["payload"]["input"]
            self.assertEqual(task.task_id, "task-multiref")
            self.assertEqual(len(model_input["reference_image_urls"]), 3)
            self.assertNotIn("first_frame_url", model_input)
            self.assertEqual(
                [
                    Path(url).name
                    for url in model_input["reference_image_urls"]
                ],
                [
                    hashlib.sha256(path.read_bytes()).hexdigest()[:20] + ".png"
                    for path in references
                ],
            )

    def test_status_maps_result_json_to_completed_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = KieClient(settings(Path(temp_dir)), sleep=lambda _: None)
            response = {
                "code": 200,
                "data": {
                    "taskId": "task-1",
                    "state": "success",
                    "resultJson": json.dumps(
                        {"resultUrls": ["https://cdn.example/result.mp4"]}
                    ),
                },
            }
            with patch(
                "agent_platform.kie.urllib.request.urlopen",
                return_value=FakeResponse(response),
            ):
                task = client.get_task("task-1")
            self.assertEqual(task.status, "completed")
            self.assertEqual(task.result_url, "https://cdn.example/result.mp4")

    def test_documented_nonstandard_status_code_is_accepted_with_task_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = KieClient(settings(Path(temp_dir)), sleep=lambda _: None)
            response = {
                "code": 505,
                "msg": "success",
                "data": {"taskId": "task-1", "state": "generating"},
            }
            with patch(
                "agent_platform.kie.urllib.request.urlopen",
                return_value=FakeResponse(response),
            ):
                task = client.get_task("task-1")
            self.assertEqual(task.status, "processing")

    def test_http_errors_are_classified_and_secret_is_redacted(self) -> None:
        for code, retryable in (
            (401, False),
            (402, False),
            (403, False),
            (429, True),
            (500, True),
        ):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temp_dir:
                client = KieClient(settings(Path(temp_dir)), sleep=lambda _: None)
                error = urllib.error.HTTPError(
                    "url",
                    code,
                    "error",
                    {},
                    io.BytesIO(
                        b'{"msg":"super-secret-kie-key rejected"}'
                    ),
                )
                with patch(
                    "agent_platform.kie.urllib.request.urlopen",
                    side_effect=error,
                ):
                    with self.assertRaises(KieError) as caught:
                        client.get_task("task")
                self.assertEqual(caught.exception.status_code, code)
                self.assertEqual(caught.exception.retryable, retryable)
                self.assertNotIn(
                    "super-secret-kie-key", str(caught.exception)
                )

    def test_paid_create_network_error_is_ambiguous_and_never_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = KieClient(settings(root), sleep=lambda _: None)
            with patch.object(
                client,
                "upload_file",
                return_value="https://tempfile.example/frame.png",
            ), patch(
                "agent_platform.kie.urllib.request.urlopen",
                side_effect=urllib.error.URLError("timeout"),
            ) as mocked:
                with self.assertRaises(KieError) as caught:
                    client.create_video_task(self._request(root))
            self.assertEqual(mocked.call_count, 1)
            self.assertTrue(caught.exception.ambiguous_submission)
            self.assertFalse(caught.exception.retryable)

    def test_download_validates_video_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = KieClient(settings(root), sleep=lambda _: None)
            valid_mp4 = b"\x00\x00\x00\x18ftypisom" + b"data"
            captured = {}

            def fake_urlopen(request, timeout):
                captured["user_agent"] = request.get_header("User-agent")
                captured["accept"] = request.get_header("Accept")
                return FakeResponse(valid_mp4)

            with patch(
                "agent_platform.kie.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ):
                target = client.download_video(
                    "https://cdn.example/video.mp4",
                    root / "video.mp4",
                )
            self.assertTrue(target.is_file())
            self.assertIn("Mozilla/5.0", captured["user_agent"])
            self.assertIn("video/mp4", captured["accept"])


if __name__ == "__main__":
    unittest.main()
