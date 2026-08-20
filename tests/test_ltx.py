from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from agent_platform.config import Settings
from agent_platform.content_factory import ContentFactoryStore
from agent_platform.ltx import LtxClient, LtxError
from agent_platform.production import ProductionContractError, ProductionStore, SceneSpec
from agent_platform.telegram_bot import AgentTelegramBot
from agent_platform.video_jobs import VideoJobManager
from agent_platform.video_profiles import profiles_for_model, video_profile
from agent_platform.video_provider import VideoGenerationRequest, VideoTask


class FakeResponse:
    def __init__(self, payload: dict[str, object] | bytes, *, content_type: str = "application/json"):
        self.body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.stream = io.BytesIO(self.body)
        self.headers = {"Content-Type": content_type}
        self.status = 200

    def read(self, amount: int = -1) -> bytes:
        return self.stream.read(amount)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def settings(root: Path, **overrides) -> Settings:
    values = {
        "telegram_bot_token": "test",
        "telegram_allowed_user_ids": {1},
        "vault_path": root,
        "openai_api_key": "",
        "openai_base_url": "https://api.openai.com/v1",
        "openai_image_model": "gpt-image-2",
        "openai_image_size": "1024x1536",
        "openai_image_quality": "low",
        "deepgram_api_key": "",
        "deepgram_model": "nova-3",
        "deepgram_language": "ru",
        "ltx_video_enabled": True,
        "ltx_api_token": "secret-token",
        "ltx_base_url": "https://ltx.example",
    }
    values.update(overrides)
    return Settings(**values)


def request(root: Path) -> VideoGenerationRequest:
    frame = root / "frame.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\nframe")
    return VideoGenerationRequest(
        model="ltx-2.3",
        prompt="A red paper plane glides once.",
        image_path=frame,
        duration_seconds=5,
        aspect_ratio="16:9",
        mode="distilled",
        sound_enabled=True,
        user="CF-TEST:V01",
        resolution="1024x576",
        provider="ltx",
        seed=42,
        idempotency_key="a" * 64,
    )


def run_with_frame(root: Path, *, project_path: Path | None = None):
    project = project_path or root
    run = ContentFactoryStore(project).create_run("LTX video")
    store = ProductionStore(project)
    store.save_scene_contract(
        run.run_id,
        [
            SceneSpec(
                "S01",
                1,
                5,
                "purpose",
                "visual",
                "action",
                "static",
                "",
                "ambient sound",
                "",
                "cut",
                {},
                "image prompt",
            )
        ],
    )
    attempt = store.start_frame(run.run_id, "S01")
    store.complete_frame(
        run.run_id,
        "S01",
        attempt,
        b"\x89PNG\r\n\x1a\nframe",
        ".png",
    )
    return run, store


class FakeLtxProvider:
    provider_name = "ltx"
    is_configured = True

    def __init__(self, store: ProductionStore, run_id: str):
        self.store = store
        self.run_id = run_id
        self.requests = []

    def create_video_task(self, video_request):
        self.requests.append(video_request)
        committed = self.store.load(self.run_id)["video_jobs"]["S01"]
        expected = f"cf-{video_request.idempotency_key}"
        if committed.get("external_task_id") != expected:
            raise AssertionError("external LTX job identity was not committed before POST")
        return VideoTask(expected, "pending")

    def get_task(self, task_id: str):
        return VideoTask(task_id, "pending")

    def download_video(self, url: str, target: Path):
        raise AssertionError("not used")


class LtxSettingsAndProfileTest(unittest.TestCase):
    def test_feature_flag_is_off_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("TELEGRAM_BOT_TOKEN=test\n", encoding="utf-8")
            loaded = Settings.load(env_file)
        self.assertFalse(loaded.ltx_video_enabled)
        self.assertEqual(loaded.ltx_base_url, "")
        self.assertEqual(loaded.ltx_api_token, "")

    def test_ltx_profile_is_fixed_to_first_smoke_contract(self) -> None:
        profile = video_profile("l23")
        self.assertEqual(profile.provider, "ltx")
        self.assertEqual(profile.model, "ltx-2.3")
        self.assertEqual(profile.duration_seconds, 5)
        self.assertEqual(profile.aspect_ratio, "16:9")
        self.assertEqual(profile.resolution, "1024x576")
        self.assertTrue(profile.sound_enabled)
        self.assertEqual([item.code for item in profiles_for_model("ltx")], ["l23"])

    def test_disabled_flag_hides_ltx_and_rejects_stale_profile_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(settings(root, ltx_video_enabled=False))
            run, store = run_with_frame(root, project_path=bot._content_store().project_path)

            menu = bot.video_setup(run.run_id)
            labels = [label for row in menu.keyboard or [] for label, _ in row]
            response = bot.select_video_profile(run.run_id, "l23")

            self.assertFalse(any("LTX" in label for label in labels))
            self.assertIn("выключен", response.text)
            self.assertNotEqual(
                store.load(run.run_id).get("video_settings", {}).get("provider"),
                "ltx",
            )

    def test_enabled_flag_exposes_fixed_ltx_duration_and_persists_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(settings(root))
            run, store = run_with_frame(root, project_path=bot._content_store().project_path)

            menu = bot.video_setup(run.run_id)
            labels = [label for row in menu.keyboard or [] for label, _ in row]
            duration_menu = bot.video_duration_menu(run.run_id, "l23")
            callbacks = [callback for row in duration_menu.keyboard or [] for _, callback in row]
            response = bot.select_video_profile(run.run_id, "l23", 5)
            persisted = store.load(run.run_id)["video_settings"]

            self.assertTrue(any("LTX-2.3" in label for label in labels))
            self.assertEqual(
                callbacks,
                [
                    f"cf_video_duration:{run.run_id}:l23-5",
                    f"cf_video_model:{run.run_id}:ltx",
                ],
            )
            self.assertIn("LTX-2.3", response.text)
            self.assertEqual(persisted["provider"], "ltx")
            self.assertEqual(persisted["seed"], 42)

    def test_ltx_external_job_identity_is_committed_before_single_post(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run, store = run_with_frame(root)
            store.set_video_settings(run.run_id, video_profile("l23").to_dict())
            store.save_video_prompts(
                run.run_id,
                [
                    {
                        "scene_id": "S01",
                        "model_prompt": "LTX prompt",
                        "universal_prompt": "LTX prompt",
                        "model_id": "ltx-2.3",
                    }
                ],
            )
            provider = FakeLtxProvider(store, run.run_id)
            manager = VideoJobManager(store, {"ltx": provider})
            preview = manager.prepare(
                run.run_id,
                provider="ltx",
                model="ltx-2.3",
                mode="distilled",
                duration_seconds=5,
                aspect_ratio="16:9",
                sound_enabled=True,
                resolution="1024x576",
                seed=42,
            )
            manager.approve(run.run_id, preview.approval_id)

            first = manager.submit_approved(run.run_id)
            second = manager.submit_approved(run.run_id)

            self.assertEqual(len(provider.requests), 1)
            job = first["video_jobs"]["S01"]
            self.assertEqual(
                job["external_task_id"],
                f"cf-{job['request_fingerprint']}",
            )
            self.assertEqual(
                second["video_jobs"]["S01"]["external_task_id"],
                job["external_task_id"],
            )
            self.assertEqual(provider.requests[0].seed, 42)

    def test_unconfigured_ltx_provider_cannot_create_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run, store = run_with_frame(root)
            provider = FakeLtxProvider(store, run.run_id)
            provider.is_configured = False
            manager = VideoJobManager(store, {"ltx": provider})

            with self.assertRaises(ProductionContractError) as ctx:
                manager.prepare(
                    run.run_id,
                    provider="ltx",
                    model="ltx-2.3",
                    mode="distilled",
                    duration_seconds=5,
                    aspect_ratio="16:9",
                    sound_enabled=True,
                    resolution="1024x576",
                    seed=42,
                )

            self.assertIn("feature flag", str(ctx.exception))
            self.assertEqual(provider.requests, [])
            approval = store.load(run.run_id).get("video_approval") or {}
            self.assertEqual(approval.get("status"), "not_requested")


class LtxClientTest(unittest.TestCase):
    def test_create_uses_stable_job_id_bearer_auth_and_exact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            captured = {}

            def fake_urlopen(http_request, timeout=0):
                captured["request"] = http_request
                captured["timeout"] = timeout
                return FakeResponse(
                    {
                        "job_id": "cf-" + "a" * 64,
                        "status": "queued",
                        "result_url": "",
                    }
                )

            client = LtxClient(settings(root))
            with patch("agent_platform.ltx.urllib.request.urlopen", side_effect=fake_urlopen):
                task = client.create_video_task(request(root))

            payload = json.loads(captured["request"].data)
            self.assertEqual(task.task_id, "cf-" + "a" * 64)
            self.assertEqual(payload["job_id"], task.task_id)
            self.assertEqual(payload["model"], "ltx-2.3")
            self.assertEqual(payload["workflow"], "distilled")
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["duration_seconds"], 5)
            self.assertEqual(payload["width"], 1024)
            self.assertEqual(payload["height"], 576)
            self.assertEqual(payload["fps"], 24)
            self.assertTrue(payload["audio"])
            self.assertTrue(payload["image_base64"])
            self.assertEqual(
                captured["request"].headers["Authorization"],
                "Bearer secret-token",
            )

    def test_create_is_never_retried_after_lost_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = LtxClient(settings(Path(temp_dir)))
            with patch(
                "agent_platform.ltx.urllib.request.urlopen",
                side_effect=urllib.error.URLError("lost response"),
            ) as mocked:
                with self.assertRaises(LtxError) as caught:
                    client.create_video_task(request(Path(temp_dir)))
            self.assertEqual(mocked.call_count, 1)
            self.assertTrue(caught.exception.ambiguous_submission)

    def test_status_and_download_reuse_auth_without_create_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            responses = [
                FakeResponse(
                    {
                        "job_id": "cf-job",
                        "status": "completed",
                        "result_url": "/video/jobs/cf-job/result",
                    }
                ),
                FakeResponse(b"\x00\x00\x00\x18ftypisomvideo", content_type="video/mp4"),
            ]
            captured = []

            def fake_urlopen(http_request, timeout=0):
                captured.append(http_request)
                return responses.pop(0)

            client = LtxClient(settings(root))
            with patch("agent_platform.ltx.urllib.request.urlopen", side_effect=fake_urlopen):
                task = client.get_task("cf-job")
                target = client.download_video(task.result_url, root / "result.mp4")

            self.assertEqual(task.status, "completed")
            self.assertEqual(target.read_bytes(), b"\x00\x00\x00\x18ftypisomvideo")
            self.assertTrue(all(req.headers["Authorization"] == "Bearer secret-token" for req in captured))


if __name__ == "__main__":
    unittest.main()
