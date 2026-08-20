from __future__ import annotations

import json
import unittest

from agent_platform.seedance_prompting import (
    SEEDANCE_FINAL_LOCK,
    parse_seedance_prompt,
)


class SeedancePromptingTest(unittest.TestCase):
    def test_valid_native_chinese_prompt_gets_required_final_lock(self) -> None:
        raw = json.dumps(
            {
                "summary_ru": "Герой поднимает карточку, камера медленно приближается.",
                "prompt_zh": (
                    "@Image1 — 精确保留人物的脸、身体、服装和场景。 FORMAT: 9:16，5秒连续镜头。 "
                    "STYLE: 真实手机质感。 COLOR: 保持原图颜色。 ENVIRONMENT: 背景不变。 "
                    "[0:00–5s] 人物缓慢举起卡片，镜头只做平稳推进。"
                ),
            },
            ensure_ascii=False,
        )

        parsed = parse_seedance_prompt(f"```json\n{raw}\n```")

        self.assertIn("@Image1", parsed.prompt_zh)
        self.assertRegex(parsed.prompt_zh, r"[\u4e00-\u9fff]")
        self.assertTrue(parsed.prompt_zh.endswith(SEEDANCE_FINAL_LOCK))
        self.assertLessEqual(len(parsed.prompt_zh), 1800)

    def test_non_chinese_prompt_is_rejected(self) -> None:
        raw = json.dumps(
            {"summary_ru": "Описание.", "prompt_zh": "@Image1 Move slowly."},
            ensure_ascii=False,
        )

        with self.assertRaisesRegex(ValueError, "китайского текста"):
            parse_seedance_prompt(raw)

    def test_prompt_longer_than_guide_limit_is_rejected_not_cut(self) -> None:
        raw = json.dumps(
            {
                "summary_ru": "Описание.",
                "prompt_zh": "@Image1 " + ("场" * 1800),
            },
            ensure_ascii=False,
        )

        with self.assertRaisesRegex(ValueError, "1800"):
            parse_seedance_prompt(raw)

    def test_multireference_prompt_requires_every_attached_tag_in_timeline(self) -> None:
        raw = json.dumps(
            {
                "summary_ru": "Герой открывает канонический купол.",
                "prompt_zh": (
                    "@Image1 是首帧，@Image2 是同一位主角，@Image3 是关闭的标准圆顶。 "
                    "FORMAT: 9:16，5秒连续镜头。 STYLE: 真实。 COLOR: 保持原图。 "
                    "ENVIRONMENT: 背景不变。 [0:00–5s] 从@Image1的首帧开始，"
                    "@Image2中的主角打开与@Image3一致的同一个圆顶，镜头缓慢推进。"
                ),
            },
            ensure_ascii=False,
        )

        parsed = parse_seedance_prompt(
            raw,
            expected_tags=("@Image1", "@Image2", "@Image3"),
            timeline_tags=("@Image1", "@Image2", "@Image3"),
        )

        self.assertIn("@Image3", parsed.prompt_zh)

    def test_bare_reference_tag_list_in_timeline_is_rejected(self) -> None:
        raw = json.dumps(
            {
                "summary_ru": "Герой берёт предмет.",
                "prompt_zh": (
                    "@Image1 是首帧，@Image2 是主角，@Image3 是道具。"
                    "[0:00–5s] @Image1 @Image2 @Image3 人物拿起道具。"
                ),
            },
            ensure_ascii=False,
        )

        with self.assertRaisesRegex(ValueError, "бессмысленный список"):
            parse_seedance_prompt(
                raw,
                expected_tags=("@Image1", "@Image2", "@Image3"),
                timeline_tags=("@Image1", "@Image2", "@Image3"),
            )

    def test_reference_mentioned_only_before_timeline_is_rejected(self) -> None:
        raw = json.dumps(
            {
                "summary_ru": "Герой открывает купол.",
                "prompt_zh": (
                    "@Image1 是首帧，@Image2 是圆顶。 FORMAT: 9:16，5秒连续镜头。 "
                    "STYLE: 真实。 COLOR: 保持原图。 ENVIRONMENT: 背景不变。 "
                    "[0:00–5s] @Image1 人物打开圆顶，镜头缓慢推进。"
                ),
            },
            ensure_ascii=False,
        )

        with self.assertRaisesRegex(ValueError, "@Image2"):
            parse_seedance_prompt(
                raw,
                expected_tags=("@Image1", "@Image2"),
                timeline_tags=("@Image1", "@Image2"),
            )


if __name__ == "__main__":
    unittest.main()
