from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_platform.content_factory import ContentFactoryStore
from agent_platform.polza import PolzaError, PolzaTask
from agent_platform.production import ProductionContractError, ProductionStore, SceneSpec
from agent_platform.video_jobs import VideoJobManager
from agent_platform.video_profiles import video_profile


class FakePolzaClient:
    is_configured = True

    def __init__(self, ambiguous: bool = False, denied_status: int | None = None):
        self.creates = 0
        self.statuses = 0
        self.requests = []
        self.ambiguous = ambiguous
        self.denied_status = denied_status

    def create_video_task(self, request):
        self.creates += 1
        self.requests.append(request)
        if self.ambiguous:
            raise PolzaError("lost response", ambiguous_submission=True)
        if self.denied_status:
            raise PolzaError(
                "model unavailable",
                status_code=self.denied_status,
            )
        return PolzaTask(f"task-{self.creates}", "pending")

    def get_task(self, task_id: str):
        self.statuses += 1
        return PolzaTask(task_id, "completed", "https://cdn.example/result.mp4")

    def download_video(self, url: str, target: Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x00\x00\x00\x18ftypisomdata")
        return target


class FakeKieClient(FakePolzaClient):
    provider_name = "kie"


def one_scene() -> SceneSpec:
    return SceneSpec(
        "S01", 1, 5, "purpose", "visual", "action", "static", "", "", "", "cut",
        {"wardrobe": "black"}, "image prompt"
    )


def prepared_store(root: Path):
    run = ContentFactoryStore(root).create_run("Video job")
    store = ProductionStore(root)
    store.save_scene_contract(run.run_id, [one_scene()])
    attempt = store.start_frame(run.run_id, "S01")
    store.complete_frame(run.run_id, "S01", attempt, b"frame", ".png")
    store.set_video_settings(run.run_id, video_profile("s720").to_dict())
    store.save_video_prompts(
        run.run_id,
        [{
            "scene_id": "S01",
            "model_prompt": "animate",
            "universal_prompt": "animate",
            "model_id": "bytedance/seedance-2",
        }],
    )
    return run, store


def two_scene_store(root: Path):
    run = ContentFactoryStore(root).create_run("Two video jobs")
    store = ProductionStore(root)
    second = SceneSpec(
        "S02", 2, 5, "purpose 2", "visual 2", "action 2", "static", "", "", "", "cut",
        {"wardrobe": "black"}, "image prompt 2"
    )
    store.save_scene_contract(run.run_id, [one_scene(), second])
    for scene_id in ("S01", "S02"):
        attempt = store.start_frame(run.run_id, scene_id)
        store.complete_frame(run.run_id, scene_id, attempt, b"frame", ".png")
    store.set_video_settings(run.run_id, video_profile("s720").to_dict())
    store.save_video_prompts(
        run.run_id,
        [
            {"scene_id": "S01", "model_prompt": "one", "universal_prompt": "one", "model_id": "bytedance/seedance-2"},
            {"scene_id": "S02", "model_prompt": "two", "universal_prompt": "two", "model_id": "bytedance/seedance-2"},
        ],
    )
    return run, store


class VideoJobManagerTest(unittest.TestCase):
    def test_seedance_480_routes_to_kie_and_persists_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run, store = prepared_store(Path(temp_dir))
            store.set_video_settings(run.run_id, video_profile("s480").to_dict())
            store.save_video_prompts(
                run.run_id,
                [{
                    "scene_id": "S01",
                    "model_prompt": "A red ball jumps once.",
                    "universal_prompt": "A red ball jumps once.",
                    "model_id": "bytedance/seedance-2",
                }],
            )
            polza = FakePolzaClient()
            kie = FakeKieClient()
            manager = VideoJobManager(
                store,
                {"polza": polza, "kie": kie},
            )
            preview = manager.prepare(
                run.run_id,
                provider="kie",
                model="bytedance/seedance-2",
                mode="",
                duration_seconds=5,
                aspect_ratio="9:16",
                sound_enabled=False,
                resolution="480p",
            )

            self.assertEqual(preview.provider, "kie")
            manager.approve(run.run_id, preview.approval_id)
            state = manager.submit_approved(run.run_id)

            self.assertEqual(kie.creates, 1)
            self.assertEqual(polza.creates, 0)
            self.assertEqual(state["video_approval"]["provider"], "kie")
            self.assertEqual(state["video_jobs"]["S01"]["provider"], "kie")

    def test_kie_seedance_submits_exact_ordered_multireference_pack(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run, store = prepared_store(root)
            frame_path = store.frame_path(run.run_id, "S01")
            self.assertIsNotNone(frame_path)
            character_path = root / "character.png"
            object_path = root / "object.png"
            character_path.write_bytes(b"character")
            object_path.write_bytes(b"object")
            ordered_references = [
                str(frame_path),
                str(character_path),
                str(object_path),
            ]
            store.set_video_settings(run.run_id, video_profile("s480").to_dict())
            store.save_video_prompts(
                run.run_id,
                [{
                    "clip_id": "V01",
                    "scene_id": "V01",
                    "source_scene_ids": ["S01"],
                    "start_scene_id": "S01",
                    "model_prompt": (
                        "@Image1 是首帧，@Image2 是人物，@Image3 是物体。"
                        "[0:00–5s] @Image1 @Image2 打开 @Image3。"
                    ),
                    "universal_prompt": "one continuous clip",
                    "model_id": "bytedance/seedance-2",
                    "provider_reference_files": ordered_references,
                }],
            )
            kie = FakeKieClient()
            manager = VideoJobManager(store, {"kie": kie})
            preview = manager.prepare(
                run.run_id,
                provider="kie",
                model="bytedance/seedance-2",
                mode="",
                duration_seconds=5,
                aspect_ratio="9:16",
                sound_enabled=False,
                resolution="480p",
            )

            manager.approve(run.run_id, preview.approval_id)
            state = manager.submit_approved(run.run_id)

            self.assertEqual(kie.creates, 1)
            request = kie.requests[0]
            self.assertIsNone(request.image_path)
            self.assertEqual(
                request.reference_image_paths,
                tuple(Path(value) for value in ordered_references),
            )
            self.assertEqual(
                state["video_jobs"]["V01"]["reference_image_count"],
                3,
            )

    def test_logical_clip_creates_one_paid_task_from_its_start_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run, store = two_scene_store(root)
            store.save_video_prompts(
                run.run_id,
                [{
                    "clip_id": "V01",
                    "scene_id": "V01",
                    "source_scene_ids": ["S01", "S02"],
                    "start_scene_id": "S01",
                    "model_prompt": "one continuous clip",
                    "universal_prompt": "one continuous clip",
                    "model_id": "bytedance/seedance-2",
                }],
            )
            client = FakePolzaClient()
            manager = VideoJobManager(store, client)
            preview = manager.prepare(
                run.run_id, model="bytedance/seedance-2", mode="std", duration_seconds=5,
                aspect_ratio="9:16", sound_enabled=False, resolution="720p",
            )

            self.assertEqual(preview.video_count, 1)
            manager.approve(run.run_id, preview.approval_id)
            state = manager.submit_approved(run.run_id)

            self.assertEqual(client.creates, 1)
            self.assertEqual(client.requests[0].image_path.parent.name, "S01")
            self.assertEqual(state["video_jobs"]["V01"]["source_scene_ids"], ["S01", "S02"])

    def test_regenerating_prompts_replaces_only_failed_jobs_without_task_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run, store = prepared_store(Path(temp_dir))
            state = store.load(run.run_id)
            state["video_jobs"] = {
                "S01": {
                    "scene_id": "S01",
                    "submission_state": "failed",
                    "status": "failed",
                    "external_task_id": "",
                    "retry_allowed": True,
                }
            }
            state["video_approval"] = {"status": "partially_submitted"}
            store.write_state(state)

            store.save_video_prompts(
                run.run_id,
                [{
                    "scene_id": "S01",
                    "model_prompt": "new Seedance prompt",
                    "universal_prompt": "new universal prompt",
                    "model_id": "bytedance/seedance-2",
                }],
            )

            state = store.load(run.run_id)
            self.assertEqual(state["video_jobs"], {})
            self.assertEqual(state["video_approval"]["status"], "not_requested")
            self.assertEqual(
                state["video_prompts"]["S01"]["model_prompt"],
                "new Seedance prompt",
            )

    def test_regenerating_prompts_never_replaces_submitted_external_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run, store = prepared_store(Path(temp_dir))
            state = store.load(run.run_id)
            state["video_jobs"] = {
                "S01": {
                    "scene_id": "S01",
                    "submission_state": "submitted",
                    "status": "pending",
                    "external_task_id": "paid-task-123",
                }
            }
            store.write_state(state)

            with self.assertRaisesRegex(
                ProductionContractError, "Нельзя заменить видеопромпты"
            ):
                store.save_video_prompts(
                    run.run_id,
                    [{
                        "scene_id": "S01",
                        "model_prompt": "must not replace",
                        "universal_prompt": "must not replace",
                        "model_id": "bytedance/seedance-2",
                    }],
                )

            state = store.load(run.run_id)
            self.assertEqual(
                state["video_jobs"]["S01"]["external_task_id"],
                "paid-task-123",
            )

    def test_prepare_is_dry_run_and_submission_requires_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run, store = prepared_store(Path(temp_dir))
            client = FakePolzaClient()
            manager = VideoJobManager(store, client)
            preview = manager.prepare(
                run.run_id, model="bytedance/seedance-2", mode="std", duration_seconds=5,
                aspect_ratio="9:16", sound_enabled=False, resolution="720p",
            )
            self.assertEqual(client.creates, 0)
            self.assertEqual(preview.video_count, 1)
            with self.assertRaises(ValueError):
                manager.submit_approved(run.run_id)

            manager.approve(run.run_id, preview.approval_id)
            state = manager.submit_approved(run.run_id)
            self.assertEqual(client.creates, 1)
            self.assertEqual(state["video_jobs"]["S01"]["external_task_id"], "task-1")

    def test_restart_with_task_id_never_submits_again_and_downloads_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run, store = prepared_store(Path(temp_dir))
            first_client = FakePolzaClient()
            first = VideoJobManager(store, first_client)
            preview = first.prepare(
                run.run_id, model="bytedance/seedance-2", mode="std", duration_seconds=5,
                aspect_ratio="9:16", sound_enabled=False, resolution="720p",
            )
            first.approve(run.run_id, preview.approval_id)
            first.submit_approved(run.run_id)

            restarted_client = FakePolzaClient()
            restarted = VideoJobManager(ProductionStore(Path(temp_dir)), restarted_client)
            restarted.submit_approved(run.run_id)
            self.assertEqual(restarted_client.creates, 0)
            state = restarted.poll_existing(run.run_id)
            self.assertEqual(restarted_client.statuses, 1)
            self.assertTrue(state["video_jobs"]["S01"]["video_file"])

    def test_ambiguous_submission_is_not_retried_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run, store = prepared_store(Path(temp_dir))
            client = FakePolzaClient(ambiguous=True)
            manager = VideoJobManager(store, client)
            preview = manager.prepare(
                run.run_id, model="bytedance/seedance-2", mode="std", duration_seconds=5,
                aspect_ratio="9:16", sound_enabled=False, resolution="720p",
            )
            manager.approve(run.run_id, preview.approval_id)
            manager.submit_approved(run.run_id)
            manager.submit_approved(run.run_id)
            self.assertEqual(client.creates, 1)
            state = store.load(run.run_id)
            self.assertEqual(state["video_jobs"]["S01"]["submission_state"], "unknown")

    def test_single_retry_submits_only_the_explicit_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run, store = two_scene_store(Path(temp_dir))
            client = FakePolzaClient()
            manager = VideoJobManager(store, client)
            preview = manager.prepare(
                run.run_id, model="bytedance/seedance-2", mode="std", duration_seconds=5,
                aspect_ratio="9:16", sound_enabled=False, resolution="720p",
            )
            manager.approve(run.run_id, preview.approval_id)
            state = store.load(run.run_id)
            state["video_approval"]["status"] = "partially_submitted"
            state["video_jobs"] = {
                scene_id: {
                    "scene_id": scene_id,
                    "submission_state": "failed",
                    "status": "failed",
                    "external_task_id": "",
                    "retry_allowed": True,
                }
                for scene_id in ("S01", "S02")
            }
            store.write_state(state)

            manager.retry_failed_submission(run.run_id, "S01")
            manager.submit_approved(run.run_id, ["S01"])

            state = store.load(run.run_id)
            self.assertEqual(client.creates, 1)
            self.assertTrue(state["video_jobs"]["S01"]["external_task_id"])
            self.assertEqual(state["video_jobs"]["S02"]["status"], "failed")

    def test_provider_denial_stops_batch_after_first_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run, store = two_scene_store(Path(temp_dir))
            client = FakePolzaClient(denied_status=403)
            manager = VideoJobManager(store, client)
            preview = manager.prepare(
                run.run_id, model="bytedance/seedance-2", mode="std", duration_seconds=5,
                aspect_ratio="9:16", sound_enabled=False, resolution="720p",
            )
            manager.approve(run.run_id, preview.approval_id)

            state = manager.submit_approved(run.run_id)

            self.assertEqual(client.creates, 1)
            self.assertEqual(state["video_jobs"]["S01"]["error_status_code"], 403)
            self.assertEqual(state["video_jobs"]["S02"]["error_status_code"], 403)
            self.assertTrue(
                state["video_jobs"]["S02"]["skipped_after_provider_denial"]
            )
            self.assertFalse(state["video_jobs"]["S01"].get("external_task_id"))
            self.assertFalse(state["video_jobs"]["S02"].get("external_task_id"))


if __name__ == "__main__":
    unittest.main()
