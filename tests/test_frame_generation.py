from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent_platform.content_factory import ContentFactoryStore
from agent_platform.frame_generation import FrameBatchGenerator
from agent_platform.image_generation import GeneratedImage, ImageGenerationError, build_single_frame_prompt
from agent_platform.production import (
    ProductionContractError,
    ProductionStore,
    ReferenceSpec,
    SceneSpec,
    merge_image_prompt_contract,
    parse_scene_contract,
    validate_script_scene_plan,
)


def scenes_payload(count: int) -> str:
    scenes = []
    for index in range(1, count + 1):
        scenes.append(
            {
                "scene_id": f"S{index:02d}",
                "order": index,
                "duration_seconds": 3,
                "purpose": f"Функция {index}",
                "visual": f"Один отдельный кадр сцены {index}",
                "physical_action": "Персонаж делает один шаг",
                "camera_movement": "slow dolly in",
                "voiceover": "",
                "on_screen_text": "",
                "sound": "",
                "transition": "cut",
                "continuity": {"wardrobe": "чёрная куртка"},
                "image_prompt": f"Single frame prompt {index}",
            }
        )
    return "SCENE_CONTRACT\n```json\n" + json.dumps(
        {"schema_version": 1, "scenes": scenes}, ensure_ascii=False
    ) + "\n```"


class CountingImageClient:
    is_configured = True

    def __init__(self, fail_calls: set[int] | None = None):
        self.prompts: list[str] = []
        self.reference_batches: list[tuple] = []
        self.fail_calls = fail_calls or set()

    def generate(self, prompt: str, references=()) -> GeneratedImage:
        self.prompts.append(prompt)
        self.reference_batches.append(tuple(references))
        if len(self.prompts) in self.fail_calls:
            raise ImageGenerationError("test provider failure")
        return GeneratedImage(content=f"image-{len(self.prompts)}".encode())


class SceneContractTest(unittest.TestCase):
    def test_dynamic_scene_counts_parse_without_six_assumption(self) -> None:
        for count in (1, 2, 6, 8):
            with self.subTest(count=count):
                scenes = parse_scene_contract(scenes_payload(count))
                self.assertEqual(len(scenes), count)
                self.assertEqual(len({scene.image_prompt for scene in scenes}), count)

    def test_script_plan_allows_variable_durations_and_matches_total(self) -> None:
        payload = json.loads(
            scenes_payload(2).split("```json\n", 1)[1].rsplit("\n```", 1)[0]
        )
        payload["scenes"][0]["duration_seconds"] = 2
        payload["scenes"][1]["duration_seconds"] = 8
        artifact = (
            "SCENE_CONTRACT\n```json\n"
            + json.dumps(payload, ensure_ascii=False)
            + "\n```"
        )
        scenes = parse_scene_contract(artifact)

        validate_script_scene_plan(scenes, "Ролик на 10 секунд")

    def test_script_plan_rejects_fixed_count_and_wrong_total(self) -> None:
        with self.assertRaisesRegex(ProductionContractError, "от 2 до 10"):
            validate_script_scene_plan(
                parse_scene_contract(scenes_payload(1)),
                "Ролик на 3 секунды",
            )
        with self.assertRaisesRegex(ProductionContractError, "хронометражем"):
            validate_script_scene_plan(
                parse_scene_contract(scenes_payload(2)),
                "Длительность: 10 секунд",
            )

    def test_prompt_contract_must_match_scene_ids_exactly(self) -> None:
        scenes = parse_scene_contract(scenes_payload(2))
        valid = "IMAGE_PROMPT_CONTRACT\n```json\n" + json.dumps(
            {
                "schema_version": 1,
                "scenes": [
                    {"scene_id": "S01", "image_prompt": "Updated one"},
                    {"scene_id": "S02", "image_prompt": "Updated two"},
                ],
            }
        ) + "\n```"
        merged = merge_image_prompt_contract(scenes, valid)
        self.assertEqual([item.image_prompt for item in merged], ["Updated one", "Updated two"])

        invalid = "IMAGE_PROMPT_CONTRACT\n```json\n" + json.dumps(
            {"schema_version": 1, "scenes": [{"scene_id": "S01", "image_prompt": "Only"}]}
        ) + "\n```"
        with self.assertRaises(ProductionContractError):
            merge_image_prompt_contract(scenes, invalid)

    def test_single_frame_prompt_forbids_collage(self) -> None:
        prompt = build_single_frame_prompt(
            scene_id="S01",
            visual="Один человек за столом",
            image_prompt="Cinematic portrait",
            continuity={"wardrobe": "black jacket"},
        )
        self.assertIn("exactly one production reference image", prompt)
        self.assertIn("Do not create a grid", prompt)
        self.assertIn("black jacket", prompt)


class FrameBatchGeneratorTest(unittest.TestCase):
    def _create(self, root: Path, count: int):
        content = ContentFactoryStore(root)
        run = content.create_run("Dynamic frame test")
        production = ProductionStore(root)
        production.save_scene_contract(run.run_id, parse_scene_contract(scenes_payload(count)))
        return run, production

    def test_generates_one_separate_file_per_prompt_for_dynamic_counts(self) -> None:
        for count in (1, 2, 6, 8):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temp_dir:
                run, store = self._create(Path(temp_dir), count)
                client = CountingImageClient()
                result = FrameBatchGenerator(store, client).generate(run.run_id)

                self.assertEqual(result.ready_count, count)
                self.assertEqual(len(client.prompts), count)
                state = store.load(run.run_id)
                files = [store.frame_path(run.run_id, f"S{index:02d}") for index in range(1, count + 1)]
                self.assertTrue(all(path and path.is_file() for path in files))
                self.assertEqual(len(set(files)), count)
                self.assertEqual(len(state["selected_frame_ids"]), count)

    def test_one_failure_does_not_abort_and_only_failed_scene_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run, store = self._create(Path(temp_dir), 3)
            first_client = CountingImageClient(fail_calls={2})
            progress: list[tuple[int, int, str]] = []
            result = FrameBatchGenerator(store, first_client).generate(
                run.run_id,
                on_progress=lambda done, total, item: progress.append((done, total, item.status)),
            )

            self.assertEqual(result.ready_count, 2)
            self.assertEqual(result.failed_count, 1)
            self.assertEqual(progress[-1], (3, 3, "ready"))
            self.assertTrue(store.frame_path(run.run_id, "S01"))
            self.assertIsNone(store.frame_path(run.run_id, "S02"))
            self.assertTrue(store.frame_path(run.run_id, "S03"))

            retry_client = CountingImageClient()
            retry = FrameBatchGenerator(store, retry_client).generate(
                run.run_id, scene_ids=["S02"]
            )
            self.assertEqual(retry.ready_count, 1)
            self.assertEqual(len(retry_client.prompts), 1)
            state = store.load(run.run_id)
            self.assertEqual(len(state["frames"]["S02"]["attempts"]), 2)
            self.assertEqual(len(state["frames"]["S01"]["attempts"]), 1)

    def test_cancellation_stops_before_next_frame_and_preserves_completed_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run, store = self._create(Path(temp_dir), 3)
            client = CountingImageClient()
            stop = False

            def progress(done, total, result):
                nonlocal stop
                stop = done == 1

            result = FrameBatchGenerator(store, client).generate(
                run.run_id,
                on_progress=progress,
                should_cancel=lambda: stop,
            )

            self.assertTrue(result.was_cancelled)
            self.assertEqual(result.ready_count, 1)
            self.assertIsNotNone(store.frame_path(run.run_id, "S01"))
            self.assertEqual(store.load(run.run_id)["frames"]["S02"]["status"], "pending")

    def test_more_than_five_references_are_prioritized_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run, store = self._create(Path(temp_dir), 1)
            reference_ids = tuple(f"REF-CHAR-{index:02d}" for index in range(1, 7))
            scene = replace(store.scenes(run.run_id)[0], reference_ids=reference_ids)
            store.save_scene_contract(run.run_id, [scene])
            references = [
                ReferenceSpec(
                    reference_id=reference_id,
                    kind="character",
                    name=f"Character {index}",
                    prompt=f"Canonical character {index}",
                    scene_ids=("S01",),
                )
                for index, reference_id in enumerate(reference_ids, 1)
            ]
            store.save_reference_plan(run.run_id, references, [])
            for reference_id in reference_ids:
                attempt = store.start_reference(run.run_id, reference_id)
                store.complete_reference(
                    run.run_id,
                    reference_id,
                    attempt,
                    b"reference",
                    ".png",
                )
            client = CountingImageClient()

            result = FrameBatchGenerator(store, client).generate(run.run_id)

            self.assertEqual(result.ready_count, 1)
            self.assertEqual(len(client.reference_batches[0]), 5)
            attempt = store.load(run.run_id)["frames"]["S01"]["attempts"][-1]
            self.assertEqual(len(attempt["reference_inputs"]), 5)
            self.assertEqual(len(attempt["omitted_reference_inputs"]), 1)
            self.assertIn("TEXT-ONLY REFERENCE REQUIREMENTS", client.prompts[0])


if __name__ == "__main__":
    unittest.main()
