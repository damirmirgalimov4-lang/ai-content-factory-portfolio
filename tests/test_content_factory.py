from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_platform.content_factory import (
    PIPELINE,
    ContentFactoryStore,
    build_stage_messages,
)
from agent_platform.production import (
    ProductionContractError,
    ReferenceSpec,
    SceneSpec,
    generation_plan_summary,
    parse_visual_bible_contract,
    validate_english_image_prompts,
)


class ContentFactoryStoreTest(unittest.TestCase):
    def test_run_survives_each_pipeline_stage_with_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ContentFactoryStore(Path(temp_dir))
            run = store.create_run("Ролик о контент-заводе")

            self.assertRegex(run.run_id, r"^CF-\d{8}-001$")
            self.assertEqual(run.status, "queued")
            self.assertTrue((Path(temp_dir) / "runs" / run.run_id / "00-IDEA.md").exists())

            for index, stage in enumerate(PIPELINE):
                running = store.mark_running(run.run_id)
                self.assertEqual(running.current_stage, stage.key)
                run = store.save_stage(run.run_id, stage.key, f"Материал этапа {stage.key}")
                if index < len(PIPELINE) - 1:
                    self.assertEqual(run.status, "waiting_approval")
                    run = store.advance(run.run_id)

            self.assertEqual(run.status, "ready_for_production")
            self.assertEqual(run.completed_stages, tuple(stage.key for stage in PIPELINE))

            manifest_path = Path(temp_dir) / "runs" / run.run_id / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["artifacts"]), len(PIPELINE) + 1)
            self.assertTrue(all(item["sha256"] for item in manifest["artifacts"]))

            restored = store.get_run(run.run_id)
            self.assertEqual(restored, run)
            self.assertEqual(store.list_runs()[0].run_id, run.run_id)

            first_name = store.next_visual_draft_filename(run.run_id)
            first_path = store.save_visual_artifact(
                run.run_id,
                artifact_key="visual-draft-v001",
                filename=first_name,
                content=b"image-bytes",
            )
            self.assertEqual(first_path.name, "visual-draft-v001.png")
            self.assertEqual(
                store.next_visual_draft_filename(run.run_id),
                "visual-draft-v002.png",
            )
            self.assertEqual(store.list_visual_drafts(run.run_id), [first_path])

    def test_failed_run_can_be_retried_without_losing_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ContentFactoryStore(Path(temp_dir))
            run = store.create_run("Тест ошибки")
            failed = store.mark_failed(run.run_id, "Временная ошибка API")

            self.assertEqual(failed.status, "failed")
            self.assertEqual(failed.current_stage, "brief")

            store.mark_running(run.run_id)
            recovered = store.save_stage(run.run_id, "brief", "Исправленный бриф")
            self.assertEqual(recovered.status, "waiting_approval")
            self.assertIn("Исправленный бриф", store.read_artifact(run.run_id, "brief"))

    def test_prompt_is_scoped_to_one_stage_and_supports_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ContentFactoryStore(Path(temp_dir))
            run = store.create_run("Ролик про автоматизацию")
            stage = run.current_stage_spec

            messages = build_stage_messages(
                run=run,
                stage=stage,
                project_context="Цель проекта: рост аудитории",
                agent_profile="Продюсер отвечает за пайплайн",
                previous_artifacts={},
                revision_request="Сделай хук смелее",
            )

            self.assertIn("producer", messages[0]["content"])
            self.assertIn("Ролик про автоматизацию", messages[1]["content"])
            self.assertIn("Сделай хук смелее", messages[1]["content"])
            self.assertIn("Не перескакивай", messages[0]["content"])

    def test_visual_bible_maps_every_asset_location_and_frame(self) -> None:
        scenes = [
            SceneSpec(
                scene_id=f"S0{index}", order=index, duration_seconds=5,
                purpose="purpose", visual="visual", physical_action="action",
                camera_movement="static", voiceover="", on_screen_text="", sound="",
                transition="cut", continuity={}, image_prompt="English prompt",
            )
            for index in (1, 2)
        ]
        payload = {
            "schema_version": 1,
            "visual_basis": "Warm cinematic miniature world",
            "assets": [{
                "reference_id": "REF-CHAR-KING", "kind": "character", "name": "King",
                "description": "The same king in both scenes", "scene_ids": ["S01", "S02"],
            }],
            "locations": [{
                "location_id": "LOC-CASTLE", "name": "Castle", "description": "Hall",
                "scene_ids": ["S01", "S02"], "canonical_scene_id": "S01",
            }],
            "frames": [
                {"scene_id": scene_id, "location_id": "LOC-CASTLE",
                 "reference_ids": ["REF-CHAR-KING"], "task": "task",
                 "composition": "composition", "must_show": "king",
                 "constraints": "same identity", "transition": "cut"}
                for scene_id in ("S01", "S02")
            ],
        }
        artifact = "VISUAL_BIBLE_CONTRACT\n```json\n" + json.dumps(payload) + "\n```"
        bible = parse_visual_bible_contract(artifact, scenes)
        self.assertEqual(len(bible.frames), 2)

        payload["assets"][0]["scene_ids"] = ["S01"]
        invalid = "VISUAL_BIBLE_CONTRACT\n```json\n" + json.dumps(payload) + "\n```"
        with self.assertRaises(ProductionContractError):
            parse_visual_bible_contract(invalid, scenes)

    def test_generation_plan_is_computed_and_prompts_must_be_english(self) -> None:
        scene = SceneSpec(
            scene_id="S01", order=1, duration_seconds=5, purpose="purpose",
            visual="visual", physical_action="action", camera_movement="static",
            voiceover="", on_screen_text="", sound="", transition="cut",
            continuity={}, image_prompt="A king in a castle",
        )
        references = [
            ReferenceSpec("REF-CHAR-KING", "character", "King", "A king on white", ("S01",)),
            ReferenceSpec("REF-OBJ-CROWN", "object", "Crown", "A gold crown", ("S01",)),
        ]
        summary = generation_plan_summary([scene], references, [])
        self.assertEqual(summary["total_images"], 3)
        validate_english_image_prompts([scene], references)
        bad_scene = SceneSpec(**{**scene.to_dict(), "image_prompt": "Король в замке"})
        with self.assertRaises(ProductionContractError):
            validate_english_image_prompts([bad_scene], references)


if __name__ == "__main__":
    unittest.main()
