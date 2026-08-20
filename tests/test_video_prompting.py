from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_platform.config import Settings
from agent_platform.content_factory import ContentFactoryStore
from agent_platform.production import ProductionStore, ReferenceSpec, SceneSpec
from agent_platform.video_prompting import CodexImageInspector, VideoPromptBuilder
from agent_platform.video_profiles import video_profile


class FakeInspector:
    is_configured = True

    def __init__(self):
        self.paths: list[Path] = []

    def inspect(self, path: Path) -> str:
        self.paths.append(path)
        return f"Фактически виден отдельный кадр {path.parent.name}"


class FakeSeedanceLlm:
    is_configured = True

    def chat(self, messages: list[dict[str, str]]) -> str:
        tags = sorted(
            {
                f"@Image{value}"
                for value in re.findall(r"@Image([1-9]\d*)", messages[-1]["content"])
            },
            key=lambda value: int(value.removeprefix("@Image")),
        )
        tags = tags or ["@Image1"]
        reference_map = "，".join(
            f"{tag} 具有唯一且固定的视觉作用" for tag in tags
        )
        timeline_links = "，".join(
            (
                f"从{tag}的首帧开始"
                if index == 0
                else f"保持与{tag}对应的角色或物体外观"
            )
            for index, tag in enumerate(tags)
        )
        return json.dumps(
            {
                "summary_ru": "Персонаж плавно поднимает карточку, камера медленно приближается.",
                "prompt_zh": (
                    f"REFERENCE MAP: {reference_map}。FORMAT: 9:16，5秒连续镜头。 "
                    "STYLE: 自然真实。 COLOR: 保持原图色彩与光线。 ENVIRONMENT: 背景布局不变。 "
                    f"[0:00–5s] {timeline_links}，人物缓慢举起一张卡片，镜头只做缓慢推进。"
                ),
            },
            ensure_ascii=False,
        )


def scene(index: int) -> SceneSpec:
    return SceneSpec(
        scene_id=f"S{index:02d}",
        order=index,
        duration_seconds=4,
        purpose=f"Функция {index}",
        visual=f"Визуал {index}",
        physical_action="Герой поднимает одну карточку",
        camera_movement="slow dolly in",
        voiceover="",
        on_screen_text="",
        sound="тихий звук комнаты",
        transition="cut",
        continuity={"wardrobe": "black jacket"},
        image_prompt=f"Prompt {index}",
    )


class VideoPromptBuilderTest(unittest.TestCase):
    @staticmethod
    def _build_manifest_fixture(
        root: Path,
        references: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        frame_path = root / "S01.png"
        frame_path.write_bytes(b"frame")
        inputs: list[dict[str, object]] = []
        for index, raw in enumerate(references, 1):
            item = dict(raw)
            path = root / f"reference-{index}.png"
            path.write_bytes(f"reference-{index}".encode())
            item.setdefault("reference_id", f"REF-{index:02d}")
            item.setdefault("file", str(path))
            item.setdefault("scene_ids", ["S01"])
            inputs.append(item)
        return VideoPromptBuilder._build_reference_manifest(
            [
                {
                    "scene_id": "S01",
                    "file": str(frame_path),
                    "description": "A person stands beside a table.",
                    "inspection_status": "visual",
                }
            ],
            inputs,
            use_multireference=True,
        )

    def test_codex_inspector_streams_prompt_through_stdin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "codex.exe"
            executable.write_bytes(b"test")
            image_path = root / "frame.png"
            image_path.write_bytes(b"image")
            settings = Settings(
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
                codex_cli_path=str(executable),
                codex_workdir=root,
            )
            stdout = json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "Виден герой."},
                }
            )
            completed = subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

            with patch.dict(os.environ, {"OPENAI_API_KEY": "do-not-pass"}, clear=False):
                with patch("agent_platform.video_prompting.subprocess.run", return_value=completed) as run:
                    answer = CodexImageInspector(settings).inspect(image_path)

            self.assertEqual(answer, "Виден герой.")
            command = run.call_args.args[0]
            self.assertEqual(command[-1], "-")
            self.assertIn(str(image_path.resolve()), run.call_args.kwargs["input"])
            self.assertNotIn(str(image_path.resolve()), command)
            self.assertNotIn("OPENAI_API_KEY", run.call_args.kwargs["env"])

    def test_kling_adapter_names_image_to_video_and_preserves_start_frame(self) -> None:
        adapted = VideoPromptBuilder._adapt_to_model("kling/v3", "Move one hand slowly.")

        self.assertIn("Kling 3 image-to-video", adapted)
        self.assertIn("supplied start frame", adapted)
        self.assertIn("Move one hand slowly", adapted)

    def test_prompt_count_matches_selected_frames_and_files_are_saved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            content = ContentFactoryStore(root)
            run = content.create_run("Video prompts")
            production = ProductionStore(root)
            production.save_scene_contract(run.run_id, [scene(1), scene(2), scene(3)])
            for index in range(1, 4):
                attempt = production.start_frame(run.run_id, f"S{index:02d}")
                production.complete_frame(run.run_id, f"S{index:02d}", attempt, b"image", ".png")
            production.set_selected(run.run_id, "S02", False)
            production.set_video_settings(run.run_id, video_profile("s720").to_dict())

            inspector = FakeInspector()
            prompts = VideoPromptBuilder(production, inspector, FakeSeedanceLlm()).build(
                run.run_id,
                model_id="bytedance/seedance-2",
                duration_seconds=5,
                aspect_ratio="9:16",
                sound_enabled=False,
            )

            self.assertEqual([item["clip_id"] for item in prompts], ["V01", "V02"])
            self.assertEqual(
                [item["source_scene_ids"] for item in prompts],
                [["S01"], ["S03"]],
            )
            self.assertEqual(len(inspector.paths), 2)
            self.assertTrue(all("Visual evidence:" in item["universal_prompt"] for item in prompts))
            self.assertTrue(all("@Image1" in item["model_prompt"] for item in prompts))
            self.assertTrue(all("固定的视觉作用" in item["model_prompt"] for item in prompts))
            self.assertTrue(all(item["prompt_guide"] == "seedance_guide (2)" for item in prompts))
            prompt_dir = root / "runs" / run.run_id / "video-prompts"
            self.assertTrue((prompt_dir / "V01.json").exists())
            self.assertFalse((prompt_dir / "S02.json").exists())
            self.assertTrue((prompt_dir / "ALL-VIDEO-PROMPTS.json").exists())

    def test_codex_inspector_error_text_is_not_accepted_as_visual_evidence(self) -> None:
        runtime_error = (
            "Codex CLI failed: CreateProcessWithLogonW failed: 2 "
            "(The system cannot find the file specified)"
        )

        self.assertTrue(CodexImageInspector.is_error_response(runtime_error))
        self.assertFalse(
            CodexImageInspector.is_error_response(
                "На изображении виден мужчина в чёрной куртке у панели."
            )
        )

    def test_unsupported_confirmed_parameter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = ContentFactoryStore(root).create_run("Invalid duration")
            production = ProductionStore(root)
            production.save_scene_contract(run.run_id, [scene(1)])
            attempt = production.start_frame(run.run_id, "S01")
            production.complete_frame(run.run_id, "S01", attempt, b"image", ".png")
            with self.assertRaises(ValueError):
                VideoPromptBuilder(production, FakeInspector()).build(
                    run.run_id,
                    model_id="bytedance/seedance-2",
                    duration_seconds=6,
                    aspect_ratio="9:16",
                    sound_enabled=False,
                )

    def test_video_prompter_analyzes_character_card_used_by_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = ContentFactoryStore(root).create_run("Character evidence")
            production = ProductionStore(root)
            linked_scene = SceneSpec(
                **{
                    **scene(1).to_dict(),
                    "reference_ids": ("REF-HERO",),
                }
            )
            production.save_scene_contract(run.run_id, [linked_scene])
            production.save_reference_plan(
                run.run_id,
                [ReferenceSpec("REF-HERO", "character", "Hero", "hero", ("S01",))],
                [],
            )
            ref_attempt = production.start_reference(run.run_id, "REF-HERO")
            ref_path = production.complete_reference(
                run.run_id, "REF-HERO", ref_attempt, b"character-card", ".png"
            )
            frame_attempt = production.start_frame(
                run.run_id,
                "S01",
                reference_inputs=[
                    {
                        "reference_id": "REF-HERO",
                        "role": "character",
                        "file": str(ref_path),
                    }
                ],
            )
            production.complete_frame(
                run.run_id, "S01", frame_attempt, b"scene-frame", ".png"
            )
            production.set_video_settings(run.run_id, video_profile("ks").to_dict())

            inspector = FakeInspector()
            prompts = VideoPromptBuilder(production, inspector).build(
                run.run_id,
                model_id="kling/v3",
                duration_seconds=5,
                aspect_ratio="9:16",
                sound_enabled=False,
            )

            self.assertEqual(len(inspector.paths), 2)
            self.assertEqual(prompts[0]["reference_inputs"][0]["reference_id"], "REF-HERO")
            self.assertIn("REF-HERO", prompts[0]["universal_prompt"])
            self.assertIn("exact first frame", prompts[0]["model_prompt"])
            self.assertIn("Temporal, visual and physical continuity contract", prompts[0]["model_prompt"])

    def test_cut_scenes_in_same_location_remain_separate_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = ContentFactoryStore(root).create_run("Logical clips")
            production = ProductionStore(root)
            contract = [
                SceneSpec(
                    **{
                        **scene(index).to_dict(),
                        "location_id": "LOC-ROOM",
                    }
                )
                for index in range(1, 4)
            ]
            production.save_scene_contract(run.run_id, contract)
            for scene_id in ("S01", "S02", "S03"):
                attempt = production.start_frame(run.run_id, scene_id)
                production.complete_frame(run.run_id, scene_id, attempt, b"image", ".png")
            settings = video_profile("ks").to_dict()
            settings["duration_seconds"] = 10
            production.set_video_settings(run.run_id, settings)

            prompts = VideoPromptBuilder(production, FakeInspector()).build(
                run.run_id,
                model_id="kling/v3",
                duration_seconds=10,
                aspect_ratio="9:16",
                sound_enabled=False,
            )

            self.assertEqual(len(prompts), 3)
            self.assertEqual(prompts[0]["source_scene_ids"], ["S01"])
            self.assertEqual(prompts[0]["start_scene_id"], "S01")
            self.assertEqual(prompts[0]["provider_input_frame_ids"], ["S01"])
            self.assertEqual(prompts[0]["planning_only_frame_ids"], [])
            self.assertEqual(prompts[0]["timeline"][0]["start_seconds"], 0)
            self.assertEqual(prompts[0]["timeline"][-1]["end_seconds"], 10)
            self.assertEqual(prompts[1]["source_scene_ids"], ["S02"])
            self.assertEqual(prompts[2]["source_scene_ids"], ["S03"])
            state = production.load(run.run_id)
            self.assertEqual(state["video_clip_ids"], ["V01", "V02", "V03"])
            self.assertEqual(state["video_prompt_qa"]["verdict"], "pass")

    def test_explicit_continuous_scenes_with_shared_anchor_can_merge(self) -> None:
        first = SceneSpec(
            **{
                **scene(1).to_dict(),
                "location_id": "LOC-ROOM",
                "reference_ids": ("REF-HERO",),
                "transition": "Continuous movement in one shot without a cut",
            }
        )
        second = SceneSpec(
            **{
                **scene(2).to_dict(),
                "location_id": "LOC-ROOM",
                "reference_ids": ("REF-HERO",),
            }
        )

        clips = VideoPromptBuilder._plan_clips(
            ["S01", "S02"], {"S01": first, "S02": second}, 10
        )

        self.assertEqual(
            [[item.scene_id for item in clip] for clip in clips],
            [["S01", "S02"]],
        )

    def test_new_character_not_visible_in_start_frame_begins_new_clip(self) -> None:
        first = SceneSpec(
            **{
                **scene(1).to_dict(),
                "location_id": "LOC-ROOM",
                "reference_ids": ("REF-HERO",),
            }
        )
        second = SceneSpec(
            **{
                **scene(2).to_dict(),
                "location_id": "LOC-ROOM",
                "reference_ids": ("REF-HERO", "REF-DRAGON"),
            }
        )

        clips = VideoPromptBuilder._plan_clips(
            ["S01", "S02"], {"S01": first, "S02": second}, 10
        )

        self.assertEqual([[item.scene_id for item in clip] for clip in clips], [["S01"], ["S02"]])

    def test_kie_seedance_uses_multiple_scene_and_canonical_asset_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = ContentFactoryStore(root).create_run("Kie multireference")
            production = ProductionStore(root)
            first = SceneSpec(
                **{
                        **scene(1).to_dict(),
                        "location_id": "LOC-ROOM",
                        "reference_ids": ("REF-HERO", "REF-DOME"),
                        "transition": "Continuous movement in one shot without a cut",
                }
            )
            second = SceneSpec(
                **{
                    **scene(2).to_dict(),
                    "location_id": "LOC-ROOM",
                    "reference_ids": ("REF-HERO", "REF-DOME-OPEN"),
                }
            )
            production.save_scene_contract(run.run_id, [first, second])
            production.save_reference_plan(
                run.run_id,
                [
                    ReferenceSpec(
                        "REF-HERO",
                        "character",
                        "Hero",
                        "same adult man in a black jacket",
                        ("S01", "S02"),
                    ),
                    ReferenceSpec(
                        "REF-DOME",
                        "object",
                        "Dome",
                        "canonical transparent dome",
                        ("S01",),
                        "ENTITY-DOME-01",
                        "canonical",
                        "",
                    ),
                    # This simulates a legacy run created before transient states
                    # were rejected at contract validation.
                    ReferenceSpec(
                        "REF-DOME-OPEN",
                        "object",
                        "Open dome",
                        "the same dome in an open state",
                        ("S02",),
                        "ENTITY-DOME-01",
                        "open",
                        "REF-DOME",
                    ),
                ],
                [],
            )
            for reference_id in ("REF-HERO", "REF-DOME", "REF-DOME-OPEN"):
                attempt = production.start_reference(run.run_id, reference_id)
                production.complete_reference(
                    run.run_id,
                    reference_id,
                    attempt,
                    reference_id.encode(),
                    ".png",
                )
            for scene_id in ("S01", "S02"):
                attempt = production.start_frame(run.run_id, scene_id)
                production.complete_frame(
                    run.run_id,
                    scene_id,
                    attempt,
                    scene_id.encode(),
                    ".png",
                )
            settings = video_profile("s480").to_dict()
            settings["duration_seconds"] = 10
            production.set_video_settings(run.run_id, settings)

            prompts = VideoPromptBuilder(
                production,
                FakeInspector(),
                FakeSeedanceLlm(),
            ).build(
                run.run_id,
                model_id="bytedance/seedance-2",
                duration_seconds=10,
                aspect_ratio="9:16",
                sound_enabled=False,
                provider_name="kie",
            )

            self.assertEqual(len(prompts), 1)
            manifest = prompts[0]["reference_manifest"]
            self.assertEqual(
                [item["reference_id"] for item in manifest],
                ["FRAME-S01", "REF-HERO", "REF-DOME", "FRAME-S02"],
            )
            self.assertNotIn(
                "REF-DOME-OPEN",
                [item["reference_id"] for item in manifest],
            )
            dome = next(item for item in manifest if item["reference_id"] == "REF-DOME")
            self.assertEqual(dome["aliases"], ["REF-DOME-OPEN"])
            self.assertEqual(dome["role"], "prop")
            self.assertEqual(
                {tag for beat in prompts[0]["timeline"] for tag in beat["reference_tags"]},
                {"@Image1", "@Image2", "@Image3", "@Image4"},
            )
            self.assertEqual(
                production.load(run.run_id)["video_prompt_qa"]["verdict"],
                "pass",
            )

    def test_manifest_maps_one_character_and_one_prop_to_concrete_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest, omitted = self._build_manifest_fixture(
                Path(temp_dir),
                [
                    {
                        "reference_id": "REF-HERO",
                        "kind": "character",
                        "label": "Главный герой",
                        "description": "Same adult man in a black jacket.",
                    },
                    {
                        "reference_id": "REF-SHOVEL",
                        "kind": "object",
                        "label": "Металлическая лопата",
                        "description": "Metal shovel with a wooden handle.",
                    },
                ],
            )

        self.assertFalse(omitted)
        self.assertEqual(
            [(item["tag"], item["role"], item["label"]) for item in manifest],
            [
                ("@Image1", "start_frame", "Стартовый кадр S01"),
                ("@Image2", "character", "Главный герой"),
                ("@Image3", "prop", "Металлическая лопата"),
            ],
        )

    def test_manifest_keeps_one_character_and_several_props_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest, _ = self._build_manifest_fixture(
                Path(temp_dir),
                [
                    {"reference_id": "REF-HERO", "kind": "character", "label": "Герой"},
                    {"reference_id": "REF-CUP", "kind": "object", "label": "Чашка"},
                    {"reference_id": "REF-BOOK", "kind": "object", "label": "Книга"},
                ],
            )

        self.assertEqual(
            [item["role"] for item in manifest],
            ["start_frame", "character", "prop", "prop"],
        )
        self.assertEqual(len({item["file"] for item in manifest}), 4)

    def test_manifest_maps_character_environment_and_camera_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest, _ = self._build_manifest_fixture(
                Path(temp_dir),
                [
                    {"reference_id": "REF-HERO", "kind": "character", "label": "Герой"},
                    {
                        "reference_id": "REF-ROOM",
                        "kind": "environment",
                        "label": "Лесная мастерская",
                    },
                    {
                        "reference_id": "REF-ANGLE",
                        "kind": "style",
                        "label": "Низкий ракурс камеры",
                    },
                ],
            )

        self.assertEqual(
            [item["role"] for item in manifest],
            ["start_frame", "character", "environment", "camera"],
        )

    def test_manifest_omits_image_not_used_by_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest, omitted = self._build_manifest_fixture(
                Path(temp_dir),
                [
                    {
                        "reference_id": "REF-UNUSED",
                        "kind": "object",
                        "label": "Лишний предмет",
                        "scene_ids": ["S99"],
                    }
                ],
            )

        self.assertEqual([item["reference_id"] for item in manifest], ["FRAME-S01"])
        self.assertEqual(omitted[0]["reference_id"], "REF-UNUSED")
        self.assertEqual(omitted[0]["omission_reason"], "not_used_by_clip")

    def test_unknown_role_uses_safe_additional_visual_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest, _ = self._build_manifest_fixture(
                Path(temp_dir),
                [
                    {
                        "reference_id": "REF-MISC",
                        "kind": "legacy_unknown",
                        "label": "Неподписанный визуальный материал",
                    }
                ],
            )

        item = manifest[1]
        self.assertEqual(item["role"], "additional_visual_reference")
        self.assertIn("do not infer", item["usage"])

    def test_missing_description_gets_safe_noninvented_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest, _ = self._build_manifest_fixture(
                Path(temp_dir),
                [
                    {
                        "reference_id": "REF-PROP",
                        "kind": "object",
                        "label": "Красная коробка",
                        "description": "",
                    }
                ],
            )

        self.assertEqual(manifest[1]["description"], "Additional visual reference: Красная коробка")
        self.assertEqual(manifest[1]["role"], "prop")

    def test_structural_qa_blocks_reference_count_file_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frame = Path(temp_dir) / "frame.png"
            frame.write_bytes(b"frame")
            contract = scene(1)
            prompt = {
                "clip_id": "V01",
                "source_scene_ids": ["S01"],
                "reference_manifest": [
                    {
                        "tag": "@Image1",
                        "reference_id": "FRAME-S01",
                        "kind": "start_frame",
                        "role": "start_frame",
                        "label": "Стартовый кадр S01",
                        "usage": "set the exact opening composition and first video frame",
                        "description": "Hero beside a table.",
                        "file": str(frame),
                        "scene_ids": ["S01"],
                    }
                ],
                "provider_reference_files": [str(frame)],
                "provider_reference_count": 2,
                "model_id": "kling/v3",
                "model_prompt": "@Image1 exact prompt",
                "universal_prompt": (
                    "Temporal, visual and physical continuity contract: objects persist; "
                    "the final frame holds the reached final state"
                ),
                "timeline": [
                    {
                        "start_seconds": 0,
                        "end_seconds": 5,
                        "scene_id": "S01",
                        "reference_tags": ["@Image1"],
                        "reference_context": "begin exactly from @Image1 (S01)",
                        "reference_uses": [
                            {
                                "tag": "@Image1",
                                "role": "start_frame",
                                "label": "Стартовый кадр S01",
                                "usage": "set exact first frame",
                            }
                        ],
                    }
                ],
                "frame_evidence": [],
                "reference_inputs": [],
                "omitted_reference_inputs": [],
                "planning_only_frame_ids": [],
            }

            qa = VideoPromptBuilder._build_structural_qa(
                [prompt], ["S01"], {"S01": contract}, 5
            )

        self.assertEqual(qa["verdict"], "fail")
        self.assertTrue(any("число provider reference" in error for error in qa["errors"]))

    def test_timeline_semantically_links_references_and_persists_prop(self) -> None:
        timeline = VideoPromptBuilder._build_timeline([scene(1)], 5)
        manifest = [
            {
                "tag": "@Image1",
                "reference_id": "FRAME-S01",
                "role": "start_frame",
                "label": "Стартовый кадр S01",
                "usage": "set exact first frame",
                "scene_ids": ["S01"],
            },
            {
                "tag": "@Image2",
                "reference_id": "REF-HERO",
                "role": "character",
                "label": "Герой",
                "usage": "preserve exact identity",
                "scene_ids": ["S01"],
            },
            {
                "tag": "@Image3",
                "reference_id": "REF-CARD",
                "role": "prop",
                "label": "Карточка",
                "usage": "preserve exact prop",
                "scene_ids": ["S01"],
            },
        ]

        VideoPromptBuilder._attach_timeline_reference_tags(timeline, manifest)

        context = timeline[0]["reference_context"]
        self.assertIn("@Image2 (Герой)", context)
        self.assertIn("@Image3 (Карточка) as the exact prop", context)
        self.assertIn("keep it persistent", context)
        self.assertNotRegex(context, r"@Image\d+\s+@Image\d+")
        self.assertIn("every visible character and object remains", timeline[0]["final_state"])


if __name__ == "__main__":
    unittest.main()
