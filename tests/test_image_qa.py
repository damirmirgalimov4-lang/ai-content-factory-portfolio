from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_platform.content_factory import ContentFactoryStore
from agent_platform.image_qa import PostImageQa
from agent_platform.production import ProductionStore, ReferenceSpec, SceneSpec


class FakeInspector:
    is_configured = True

    def inspect(self, path: Path) -> str:
        return f"Visible stable subject in {path.parent.name}"


class FakeQaLlm:
    is_configured = True

    def chat(self, messages: list[dict[str, str]]) -> str:
        return json.dumps(
            {
                "verdict": "pass",
                "summary": "Персонаж и сцена согласованы.",
                "ready_for_video_prompts": True,
                "issues": [],
            },
            ensure_ascii=False,
        )


class PostImageQaTest(unittest.TestCase):
    def _store(self, root: Path) -> tuple[str, ProductionStore]:
        run = ContentFactoryStore(root).create_run("Image QA")
        store = ProductionStore(root)
        scene = SceneSpec(
            "S01", 1, 5, "purpose", "hero in studio", "raises hand", "static",
            "", "", "", "cut", {}, "A hero in a studio",
            ("REF-HERO",), "LOC-STUDIO", "",
        )
        store.save_scene_contract(run.run_id, [scene])
        store.save_reference_plan(
            run.run_id,
            [ReferenceSpec("REF-HERO", "character", "Hero", "Hero on white", ("S01",))],
            [],
        )
        return run.run_id, store

    def test_real_files_are_inspected_and_report_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_id, store = self._store(Path(temp_dir))
            ref_attempt = store.start_reference(run_id, "REF-HERO")
            store.complete_reference(run_id, "REF-HERO", ref_attempt, b"ref", ".png")
            frame_attempt = store.start_frame(run_id, "S01")
            store.complete_frame(run_id, "S01", frame_attempt, b"frame", ".png")

            result = PostImageQa(store, FakeInspector(), FakeQaLlm()).run(run_id)

            self.assertEqual(result.verdict, "pass")
            self.assertTrue(result.ready_for_video_prompts)
            self.assertEqual(result.checked_items, 2)
            self.assertTrue((Path(temp_dir) / "runs" / run_id / "06-IMAGE-QA.json").exists())
            self.assertEqual(store.load(run_id)["image_qa"]["verdict"], "pass")

    def test_missing_images_fail_without_claiming_visual_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_id, store = self._store(Path(temp_dir))

            result = PostImageQa(store, FakeInspector(), FakeQaLlm()).run(run_id)

            self.assertEqual(result.verdict, "fail")
            self.assertFalse(result.ready_for_video_prompts)
            self.assertEqual(result.checked_items, 0)


if __name__ == "__main__":
    unittest.main()
