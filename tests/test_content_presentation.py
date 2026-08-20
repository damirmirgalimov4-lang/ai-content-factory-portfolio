from __future__ import annotations

import json
import unittest

from agent_platform.content_presentation import present_stage, present_video_prompts
from agent_platform.telegram_bot import TelegramClient


def contract(marker: str, scenes: list[dict]) -> str:
    return f"{marker}\n```json\n{json.dumps({'schema_version': 1, 'scenes': scenes}, ensure_ascii=False)}\n```"


class ContentPresentationTest(unittest.TestCase):
    def test_script_hides_machine_json_and_keeps_each_scene_readable(self) -> None:
        scenes = [
            {
                "scene_id": "S01",
                "order": 1,
                "duration_seconds": 3,
                "purpose": "Хук",
                "visual": "Герой входит",
                "physical_action": "Открывает дверь",
                "camera_movement": "push-in",
                "voiceover": "Смотри",
                "on_screen_text": "Начинаем",
                "sound": "шаги",
                "transition": "cut",
                "continuity": {},
                "image_prompt": "one frame",
            },
            {
                "scene_id": "S02",
                "order": 2,
                "duration_seconds": 4,
                "purpose": "Демонстрация",
                "visual": "Система работает",
                "continuity": {},
                "image_prompt": "second frame",
            },
        ]
        rendered = present_stage("script", "Короткий сценарий.\n\n" + contract("SCENE_CONTRACT", scenes))
        self.assertIn("Короткий сценарий", rendered)
        self.assertIn("Сцен в производстве:** 2", rendered)
        self.assertNotIn("SCENE_CONTRACT", rendered)
        self.assertNotIn('"scene_id"', rendered)

    def test_image_prompts_are_cards_not_json(self) -> None:
        rendered = present_stage(
            "prompts",
            contract(
                "IMAGE_PROMPT_CONTRACT",
                [
                    {"scene_id": "S01", "image_prompt": "first exact prompt"},
                    {"scene_id": "S02", "image_prompt": "second exact prompt"},
                ],
            ),
        )
        self.assertIn("Кадр S01", rendered)
        self.assertIn("first exact prompt", rendered)
        self.assertNotIn("IMAGE_PROMPT_CONTRACT", rendered)
        self.assertNotIn('"image_prompt"', rendered)

    def test_video_prompts_show_exact_provider_prompt(self) -> None:
        rendered = present_video_prompts(
            "CF-20260714-002",
            {
                "S01": {
                    "model_id": "kling/v3",
                    "duration_seconds": 5,
                    "aspect_ratio": "9:16",
                    "sound_enabled": False,
                    "model_prompt": "exact video prompt",
                }
            },
        )
        self.assertIn("exact video prompt", rendered)
        self.assertIn("kling/v3", rendered)

    def test_video_prompt_references_are_presented_by_role_not_as_tag_pile(self) -> None:
        rendered = present_video_prompts(
            "CF-20260805-001",
            {
                "V01": {
                    "model_id": "bytedance/seedance-2",
                    "duration_seconds": 5,
                    "aspect_ratio": "9:16",
                    "sound_enabled": False,
                    "source_scene_ids": ["S01"],
                    "model_prompt": "точный prompt",
                    "reference_manifest": [
                        {
                            "tag": "@Image1",
                            "reference_id": "FRAME-S01",
                            "role": "start_frame",
                            "label": "Стартовый кадр S01",
                            "scene_ids": ["S01"],
                        },
                        {
                            "tag": "@Image2",
                            "reference_id": "REF-HERO",
                            "role": "character",
                            "label": "Главный герой",
                            "scene_ids": ["S01"],
                        },
                    ],
                    "timeline": [
                        {
                            "start_seconds": 0,
                            "end_seconds": 5,
                            "action": "Герой берёт со стола предмет.",
                            "camera": "медленное приближение",
                            "reference_uses": [
                                {
                                    "tag": "@Image1",
                                    "role": "start_frame",
                                    "label": "Стартовый кадр S01",
                                },
                                {
                                    "tag": "@Image2",
                                    "role": "character",
                                    "label": "Главный герой",
                                },
                            ],
                        }
                    ],
                }
            },
        )

        self.assertIn("@Image2 — Главный герой", rendered)
        self.assertIn("персонаж", rendered)
        self.assertIn("Референсы:", rendered)
        self.assertNotIn("@Image1 @Image2", rendered)

    def test_telegram_chunks_keep_scene_paragraphs_intact(self) -> None:
        client = TelegramClient("test")
        scene_one = "Кадр 1\n" + "a" * 900
        scene_two = "Кадр 2\n" + "b" * 900
        scene_three = "Кадр 3\n" + "c" * 900
        chunks = client._chunks("\n\n".join([scene_one, scene_two, scene_three]), 2000)
        self.assertEqual(len(chunks), 2)
        self.assertIn("Кадр 1", chunks[0])
        self.assertIn("Кадр 2", chunks[0])
        self.assertNotIn("Кадр 3", chunks[0])
        self.assertTrue(chunks[1].startswith("Кадр 3"))

    def test_telegram_chunks_detect_scene_boundaries_without_blank_lines(self) -> None:
        client = TelegramClient("test")
        text = "\n".join(
            [
                "Кадр 1",
                "Задача: " + "a" * 700,
                "Кадр 2",
                "Задача: " + "b" * 700,
                "Кадр 3",
                "Задача: " + "c" * 700,
            ]
        )

        chunks = client._chunks(text, 1600)

        self.assertEqual(len(chunks), 2)
        self.assertIn("Кадр 1", chunks[0])
        self.assertIn("Кадр 2", chunks[0])
        self.assertNotIn("Кадр 3", chunks[0])
        self.assertTrue(chunks[1].startswith("Кадр 3"))


if __name__ == "__main__":
    unittest.main()
