from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_platform.storyboard import (
    STORYBOARD_PROMPT_METADATA_PATH,
    STORYBOARD_PROMPT_SOURCE_LABEL,
    STORYBOARD_STAGES,
    StoryboardStore,
    load_storyboard_prompt_template,
)


def guided_plan_json() -> str:
    return """{
  "schema_version": 2,
  "title": "Умный свет встречает героя",
  "logline": "Герой возвращается в тёмную квартиру, и свет оживляет пространство.",
  "duration_seconds": 15,
  "aspect_ratio": "16:9",
  "layout": {"columns": 3, "rows": 1},
  "references": [
    {
      "reference_id": "REF-CHAR-01",
      "kind": "character",
      "label": "Герой",
      "description": "Один и тот же молодой герой в тёмной куртке.",
      "usage": "Сохранять внешность и одежду во всех панелях."
    },
    {
      "reference_id": "REF-LOC-01",
      "kind": "environment",
      "label": "Квартира",
      "description": "Современная квартира с прихожей и гостиной.",
      "usage": "Сохранять планировку и направление света."
    }
  ],
  "panels": [
    {
      "panel_id": "P01",
      "order": 1,
      "timecode": "00:00-00:05",
      "shot_type": "wide",
      "visual": "Тёмная прихожая и закрытая дверь.",
      "action": "Герой открывает дверь и входит.",
      "camera": "Статичный общий план.",
      "caption": "Возвращение домой",
      "reference_ids": ["REF-CHAR-01", "REF-LOC-01"]
    },
    {
      "panel_id": "P02",
      "order": 2,
      "timecode": "00:05-00:10",
      "shot_type": "medium",
      "visual": "Герой делает шаг в темноту.",
      "action": "Свет последовательно включается по пути героя.",
      "camera": "Плавное сопровождение.",
      "caption": "Дом узнаёт героя",
      "reference_ids": ["REF-CHAR-01", "REF-LOC-01"]
    },
    {
      "panel_id": "P03",
      "order": 3,
      "timecode": "00:10-00:15",
      "shot_type": "wide",
      "visual": "Тёплая освещённая гостиная.",
      "action": "Герой останавливается и улыбается.",
      "camera": "Мягкий отъезд.",
      "caption": "Свет встречает дома",
      "reference_ids": ["REF-CHAR-01", "REF-LOC-01"]
    }
  ],
  "sheet_prompt": "Create one single 16:9 storyboard sheet with exactly three numbered panels in chronological order, using the attached character and apartment references consistently. Put a short timecode and action caption below every panel. Do not create separate images."
}"""


class StoryboardStoreTest(unittest.TestCase):
    def test_raster_dimensions_are_read_for_png_jpeg_and_webp(self) -> None:
        png = (
            b"\x89PNG\r\n\x1a\n"
            + (13).to_bytes(4, "big")
            + b"IHDR"
            + (1536).to_bytes(4, "big")
            + (1024).to_bytes(4, "big")
            + b"\x08\x02\x00\x00\x00"
        )
        jpeg = (
            b"\xff\xd8\xff\xc0"
            + (17).to_bytes(2, "big")
            + b"\x08"
            + (1024).to_bytes(2, "big")
            + (1536).to_bytes(2, "big")
            + b"\x03"
            + b"\x00" * 9
        )
        vp8x = (
            b"\x00\x00\x00\x00"
            + (1535).to_bytes(3, "little")
            + (1023).to_bytes(3, "little")
        )
        webp = (
            b"RIFF"
            + (4 + 8 + len(vp8x)).to_bytes(4, "little")
            + b"WEBP"
            + b"VP8X"
            + len(vp8x).to_bytes(4, "little")
            + vp8x
        )

        for content in (png, jpeg, webp):
            self.assertEqual(StoryboardStore._image_dimensions(content), (1536, 1024))
        with self.assertRaisesRegex(ValueError, "определить размер"):
            StoryboardStore._image_dimensions(b"RIFF\x00\x00\x00\x00WEBP")

    def test_one_input_becomes_a_guided_storyboard_plan_instead_of_empty_manual_stages(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = StoryboardStore(root)
            project = store.create_guided_project(
                "Ролик о герое, которого дома встречает умный свет."
            )
            self.assertEqual(project.workflow, "guided-v2")
            self.assertEqual(project.status, "planning")

            planned = store.save_generated_plan(project.project_id, guided_plan_json())
            restored = StoryboardStore(root).get_project(project.project_id)
            plan = StoryboardStore(root).read_plan(project.project_id)

            self.assertEqual(planned.status, "plan_review")
            self.assertEqual(planned.current_stage, "plan_review")
            self.assertEqual(restored, planned)
            self.assertEqual([item["panel_id"] for item in plan["panels"]], ["P01", "P02", "P03"])
            self.assertEqual([item["order"] for item in plan["panels"]], [1, 2, 3])
            self.assertEqual(
                [item["reference_id"] for item in plan["references"]],
                ["REF-CHAR-01", "REF-LOC-01"],
            )
            self.assertIn("one single 16:9 storyboard sheet", plan["sheet_prompt"])
            self.assertNotIn("video_prompt", planned.completed_stages)
            self.assertFalse(planned.storyboard_approved_at)

    def test_guided_plan_rejects_broken_timeline_excess_panels_and_corrupt_storage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = StoryboardStore(root)

            gap_project = store.create_guided_project("План с дырой в timeline")
            gap_plan = json.loads(guided_plan_json())
            gap_plan["panels"][1]["timecode"] = "00:06-00:10"
            with self.assertRaisesRegex(ValueError, "таймлайн"):
                store.save_generated_plan(
                    gap_project.project_id,
                    json.dumps(gap_plan, ensure_ascii=False),
                )

            excess_project = store.create_guided_project("Слишком много панелей")
            excess_plan = json.loads(guided_plan_json())
            excess_plan["layout"] = {"columns": 7, "rows": 3}
            excess_plan["panels"] = [
                {
                    **gap_plan["panels"][0],
                    "panel_id": f"P{index:02d}",
                    "order": index,
                    "timecode": f"00:{index - 1:02d}-00:{index:02d}",
                }
                for index in range(1, 22)
            ]
            excess_plan["duration_seconds"] = 21
            with self.assertRaisesRegex(ValueError, "не более 20"):
                store.save_generated_plan(
                    excess_project.project_id,
                    json.dumps(excess_plan, ensure_ascii=False),
                )

            valid_project = store.create_guided_project("Повреждение после записи")
            store.save_generated_plan(valid_project.project_id, guided_plan_json())
            plan_path = (
                root
                / "storyboards"
                / valid_project.project_id
                / "02-GUIDED-STORYBOARD-PLAN.json"
            )
            plan_path.write_text('{"schema_version": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "повреждён"):
                StoryboardStore(root).read_plan(valid_project.project_id)

    def test_guided_plan_decisions_are_durable_and_do_not_approve_phase_two(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = StoryboardStore(root)

            approved = store.create_guided_project("Идея для approval")
            approved = store.save_generated_plan(approved.project_id, guided_plan_json())
            approved = store.approve_plan(approved.project_id)
            self.assertEqual(approved.status, "plan_approved")
            self.assertEqual(approved.current_stage, "plan_approved")
            self.assertEqual(approved.storyboard_approved_at, "")
            self.assertEqual(StoryboardStore(root).get_project(approved.project_id), approved)
            with self.assertRaisesRegex(ValueError, "Storyboard ещё не готов"):
                store.approve_storyboard(approved.project_id)

            rejected = store.create_guided_project("Идея для reject")
            rejected = store.save_generated_plan(rejected.project_id, guided_plan_json())
            rejected = store.reject_plan(rejected.project_id, "История не работает")
            self.assertEqual(rejected.status, "rejected")
            self.assertEqual(rejected.storyboard_approved_at, "")
            rejection_path = (
                root
                / "storyboards"
                / rejected.project_id
                / "PLAN-REJECTION.md"
            )
            self.assertIn("История не работает", rejection_path.read_text(encoding="utf-8"))

    def test_sheet_generation_commit_point_survives_restart_and_blocks_resubmission(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = StoryboardStore(root)
            project = store.create_guided_project("Durable gate для одного sheet")
            project = store.save_generated_plan(project.project_id, guided_plan_json())
            project = store.approve_plan(project.project_id)
            quote = {
                "provider_key": "codex",
                "provider_label": "Codex Image",
                "model": "Codex agent gpt-5.6-sol; image backend tool-selected",
                "size": "1536x1024",
                "quality": "provider default",
                "cost_display": "0 ₽ отдельного API-платежа",
                "billing_note": "Один запрос из лимита подписки.",
                "result_display": "ровно один общий storyboard sheet",
                "expected_requests": 1,
                "input_sha256": "a" * 64,
            }

            project = store.prepare_sheet_quote(project.project_id, quote)
            self.assertEqual(project.status, "sheet_awaiting_confirmation")
            project = store.begin_sheet_generation(project.project_id, quote)
            self.assertEqual(project.status, "sheet_generating")
            wrong_size_png = (
                b"\x89PNG\r\n\x1a\n"
                + (13).to_bytes(4, "big")
                + b"IHDR"
                + (1024).to_bytes(4, "big")
                + (1024).to_bytes(4, "big")
                + b"\x08\x02\x00\x00\x00"
            )
            with self.assertRaisesRegex(ValueError, "1024x1024.*1536x1024"):
                store.save_generated_sheet(project.project_id, wrong_size_png)
            self.assertFalse(
                (root / "storyboards" / project.project_id / "sheets").exists()
            )

            for index in range(20):
                store.create_project(f"Более новый Storyboard {index + 1}")
            self.assertNotIn(
                project.project_id,
                [item.project_id for item in store.list_projects()],
            )
            self.assertIn(
                project.project_id,
                [item.project_id for item in store.list_projects(limit=None)],
            )

            restarted = StoryboardStore(root)
            restored = restarted.get_project(project.project_id)
            assert restored is not None
            self.assertEqual(restored.status, "sheet_generating")
            self.assertEqual(restored.pending_operation, "sheet_generation")
            with self.assertRaisesRegex(ValueError, "не ожидает подтверждения"):
                restarted.begin_sheet_generation(project.project_id, quote)

            recovered = restarted.recover_interrupted_sheet_generations()
            self.assertEqual([item.project_id for item in recovered], [project.project_id])
            reconciled = restarted.get_project(project.project_id)
            assert reconciled is not None
            self.assertEqual(reconciled.status, "sheet_reconciliation_required")
            self.assertEqual(reconciled.pending_operation, "sheet_reconciliation")
            self.assertEqual(restarted.recover_interrupted_sheet_generations(), [])
            with self.assertRaisesRegex(ValueError, "не ожидает подтверждения"):
                StoryboardStore(root).begin_sheet_generation(project.project_id, quote)

    def test_guided_plan_revision_archives_the_prior_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = StoryboardStore(root)
            project = store.create_guided_project("Идея для revision")
            project = store.save_generated_plan(project.project_id, guided_plan_json())

            project = store.prepare_plan_revision(project.project_id, "Ускорить финал")
            context = StoryboardStore(root).read_pending_plan_revision(project.project_id)

            self.assertEqual(project.status, "planning")
            self.assertEqual(project.storyboard_revision_count, 1)
            self.assertEqual(context["request"], "Ускорить финал")
            self.assertEqual(context["prior_plan"]["schema_version"], 2)
            archive = (
                root
                / "storyboards"
                / project.project_id
                / "revisions"
                / "guided-plan-001.json"
            )
            self.assertTrue(archive.is_file())
            self.assertIn("Ускорить финал", archive.read_text(encoding="utf-8"))
            self.assertEqual(StoryboardStore(root).get_project(project.project_id), project)

            project = store.save_generated_plan(project.project_id, guided_plan_json())
            project = store.begin_reference_update(project.project_id)
            self.assertEqual(project.pending_operation, "references")
            with self.assertRaisesRegex(ValueError, "нет незавершённой правки"):
                store.read_pending_plan_revision(project.project_id)

    def test_uploaded_reference_is_validated_deduplicated_and_required_in_next_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = StoryboardStore(root)
            project = store.create_guided_project("Идея с пользовательским референсом")
            project = store.save_generated_plan(project.project_id, guided_plan_json())
            png = b"\x89PNG\r\n\x1a\n" + b"reference-pixels"
            project = store.begin_reference_update(project.project_id)
            self.assertEqual(project.status, "planning")
            self.assertEqual(project.pending_operation, "references")
            with self.assertRaisesRegex(ValueError, "ещё не готов"):
                store.approve_plan(project.project_id)

            reference, created = store.save_uploaded_reference(
                project.project_id,
                png,
                "Главный герой: сохранить лицо, куртку и пропорции.",
            )
            duplicate, duplicate_created = store.save_uploaded_reference(
                project.project_id,
                png,
                "Та же картинка повторно",
            )

            self.assertTrue(created)
            self.assertFalse(duplicate_created)
            self.assertEqual(reference["reference_id"], "REF-UPLOAD-001")
            self.assertEqual(duplicate["reference_id"], reference["reference_id"])
            self.assertTrue(
                (
                    root
                    / "storyboards"
                    / project.project_id
                    / "references"
                    / reference["filename"]
                ).is_file()
            )
            with self.assertRaisesRegex(ValueError, "пользовательские референсы"):
                store.save_generated_plan(project.project_id, guided_plan_json())
            with self.assertRaisesRegex(ValueError, "JPEG, PNG или WebP"):
                store.save_uploaded_reference(project.project_id, b"not-an-image", "Файл")

    def test_public_template_is_mit_licensed_and_keeps_the_approval_gate(self) -> None:
        template = load_storyboard_prompt_template()
        metadata = json.loads(STORYBOARD_PROMPT_METADATA_PATH.read_text(encoding="utf-8"))

        self.assertEqual("AI Content Factory Storyboard Template", STORYBOARD_PROMPT_SOURCE_LABEL)
        self.assertEqual("MIT", metadata["license"])
        self.assertEqual("clean_room", metadata["source_type"])
        self.assertIn("Phase 2 requires explicit human approval", template)
        self.assertIn("Do not proceed to Phase 2 without approval", template)
        self.assertIn("Audio: Diegetic sound only", template)

    def test_project_survives_full_manual_cycle_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = StoryboardStore(root)
            self.assertEqual(
                [stage.key for stage in STORYBOARD_STAGES],
                [
                    "idea",
                    "references",
                    "beats",
                    "image_prompt",
                    "storyboard_result",
                    "video_prompt",
                    "results",
                    "review",
                ],
            )

            project = store.create_project(
                "Реклама умной лампы: герой возвращается домой и свет встречает его."
            )

            self.assertRegex(project.project_id, r"^SB-\d{8}-001$")
            self.assertEqual(project.current_stage, "references")
            self.assertEqual(project.completed_stages, ("idea",))
            self.assertEqual(project.status, "in_progress")

            for stage in STORYBOARD_STAGES[1:]:
                project = store.save_current_stage(
                    project.project_id,
                    stage.key,
                    f"Материал этапа {stage.key}",
                )
                if stage.key == "storyboard_result":
                    self.assertEqual(project.status, "awaiting_storyboard_approval")
                    with self.assertRaisesRegex(ValueError, "подтверждения storyboard"):
                        store.save_current_stage(
                            project.project_id,
                            "video_prompt",
                            "Видео-промпт нельзя создавать раньше approval",
                        )
                    project = store.approve_storyboard(project.project_id)
                    self.assertEqual(project.status, "in_progress")
                    self.assertEqual(project.current_stage, "video_prompt")
                    self.assertTrue(project.storyboard_approved_at)

            self.assertEqual(project.status, "review_ready")
            self.assertEqual(
                project.completed_stages,
                tuple(stage.key for stage in STORYBOARD_STAGES),
            )

            completed = store.complete_project(project.project_id)
            self.assertEqual(completed.status, "completed")

            restored_store = StoryboardStore(root)
            restored = restored_store.get_project(project.project_id)
            self.assertEqual(restored, completed)
            self.assertIn(
                "Материал этапа image_prompt",
                restored_store.read_stage(project.project_id, "image_prompt"),
            )
            self.assertEqual(
                restored_store.list_projects()[0].project_id,
                project.project_id,
            )

    def test_storyboard_can_be_revised_or_rejected_without_unlocking_phase_two(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = StoryboardStore(root)
            project = store.create_project("Тест веток решения")
            for stage in STORYBOARD_STAGES[1:5]:
                project = store.save_current_stage(
                    project.project_id,
                    stage.key,
                    f"Первая версия {stage.key}",
                )

            project = store.request_storyboard_revision(
                project.project_id,
                "Исправить направление взгляда персонажа.",
            )
            self.assertEqual(project.status, "in_progress")
            self.assertEqual(project.current_stage, "storyboard_result")
            self.assertEqual(project.storyboard_revision_count, 1)
            self.assertNotIn("storyboard_result", project.completed_stages)
            revision = (
                root
                / "storyboards"
                / project.project_id
                / "revisions"
                / "storyboard-result-001.md"
            ).read_text(encoding="utf-8")
            self.assertIn("Первая версия storyboard_result", revision)
            self.assertIn("Исправить направление взгляда", revision)

            project = store.save_current_stage(
                project.project_id,
                "storyboard_result",
                "Исправленная версия storyboard_result",
            )
            project = store.reject_storyboard(
                project.project_id,
                "Вариант не соответствует истории.",
            )
            self.assertEqual(project.status, "rejected")
            self.assertFalse(project.storyboard_approved_at)
            with self.assertRaisesRegex(ValueError, "не готов"):
                store.approve_storyboard(project.project_id)

    def test_stage_order_and_completion_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = StoryboardStore(Path(temp_dir))
            project = store.create_project("Тестовая идея")

            with self.assertRaisesRegex(ValueError, "Ожидался этап references"):
                store.save_current_stage(project.project_id, "beats", "Кадр 1")
            with self.assertRaisesRegex(ValueError, "пустой"):
                store.save_current_stage(project.project_id, "references", "   ")
            with self.assertRaisesRegex(ValueError, "не завершены"):
                store.complete_project(project.project_id)


if __name__ == "__main__":
    unittest.main()
