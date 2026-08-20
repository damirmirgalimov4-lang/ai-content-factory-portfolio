from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_platform.content_factory import ContentFactoryStore
from agent_platform.frame_generation import FrameBatchGenerator
from agent_platform.image_generation import GeneratedImage
from agent_platform.polza import PolzaTask
from agent_platform.production import (
    ProductionStore,
    SceneSpec,
    format_image_prompt_contract,
    format_scene_contract,
)
from agent_platform.video_jobs import VideoJobManager
from agent_platform.video_prompting import VideoPromptBuilder
from agent_platform.video_profiles import video_profile


class FlowImageClient:
    is_configured = True

    def generate(self, prompt: str) -> GeneratedImage:
        return GeneratedImage(content=("frame:" + prompt).encode("utf-8"), extension=".png")


class FlowInspector:
    is_configured = True

    def inspect(self, path: Path) -> str:
        return f"standalone frame {path.parent.name} with stable character and environment"


class FlowPromptLlm:
    is_configured = True

    def chat(self, messages: list[dict[str, str]]) -> str:
        return (
            '{"summary_ru":"Герой делает одно понятное движение.",'
            '"prompt_zh":"@Image1 — 精确保留人物和环境。 FORMAT: 9:16，5秒连续镜头。 '
            'STYLE: 自然。 COLOR: 保持原图。 ENVIRONMENT: 背景不变。 '
            '[0:00–5s] @Image1 人物完成一个动作，镜头缓慢推进。"}'
        )


class FlowPolzaClient:
    is_configured = True

    def __init__(self) -> None:
        self.creates = 0

    def create_video_task(self, request) -> PolzaTask:
        self.creates += 1
        return PolzaTask(f"task-{self.creates}", "pending")

    def get_task(self, task_id: str) -> PolzaTask:
        return PolzaTask(task_id, "completed", f"https://cdn.example/{task_id}.mp4")

    def download_video(self, url: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x00\x00\x00\x18ftypisomvideo")
        return target


def scenes(count: int = 3) -> list[SceneSpec]:
    return [
        SceneSpec(
            scene_id=f"S{index:02d}",
            order=index,
            duration_seconds=5,
            purpose=f"Функция сцены {index}",
            visual=f"Один цельный кадр {index}",
            physical_action="Персонаж делает одно понятное движение",
            camera_movement="slow push-in",
            voiceover="",
            on_screen_text="",
            sound="",
            transition="cut",
            continuity={"character": "same character", "environment": "same room"},
            image_prompt=f"single standalone vertical frame {index}, no collage",
        )
        for index in range(1, count + 1)
    ]


class FullContentFactoryFlowTest(unittest.TestCase):
    def test_complete_durable_flow_survives_restart_without_duplicate_paid_posts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            content = ContentFactoryStore(root)
            run = content.create_run("Полный пробный ролик")
            contract_scenes = scenes(3)
            artifacts = {
                "brief": "Полный бриф",
                "script": "Читаемый сценарий\n\n" + format_scene_contract(contract_scenes),
                "storyboard": "Три самостоятельных кадра с continuity",
                "prompts": format_image_prompt_contract(contract_scenes),
                "qa": "Verdict: pass",
            }
            for index, (stage, artifact) in enumerate(artifacts.items()):
                content.mark_running(run.run_id)
                run = content.save_stage(run.run_id, stage, artifact)
                if index < len(artifacts) - 1:
                    run = content.advance(run.run_id)
            self.assertEqual(run.status, "ready_for_production")

            production = ProductionStore(root)
            production.save_scene_contract(run.run_id, contract_scenes)
            frames = FrameBatchGenerator(production, FlowImageClient()).generate(
                run.run_id,
                aspect_ratio="9:16",
            )
            self.assertEqual(frames.ready_count, 3)
            production.select_all_ready(run.run_id)
            production.set_video_settings(run.run_id, video_profile("s720").to_dict())

            prompts = VideoPromptBuilder(
                production, FlowInspector(), FlowPromptLlm()
            ).build(
                run.run_id,
                model_id="bytedance/seedance-2",
                duration_seconds=5,
                aspect_ratio="9:16",
                sound_enabled=False,
            )
            self.assertEqual(len(prompts), 3)

            provider = FlowPolzaClient()
            manager = VideoJobManager(production, provider)
            preview = manager.prepare(
                run.run_id,
                model="bytedance/seedance-2",
                mode="std",
                resolution="720p",
                duration_seconds=5,
                aspect_ratio="9:16",
                sound_enabled=False,
            )
            manager.approve(run.run_id, preview.approval_id)
            submitted = manager.submit_approved(run.run_id)
            self.assertEqual(provider.creates, 3)
            self.assertTrue(
                all(job.get("external_task_id") for job in submitted["video_jobs"].values())
            )

            restarted = VideoJobManager(ProductionStore(root), provider)
            restarted.submit_approved(run.run_id)
            self.assertEqual(provider.creates, 3)
            completed = restarted.poll_existing(run.run_id)
            self.assertTrue(
                all(job.get("status") == "completed" for job in completed["video_jobs"].values())
            )
            self.assertTrue(
                all(
                    (production.runs_path / run.run_id / job["video_file"]).is_file()
                    for job in completed["video_jobs"].values()
                )
            )


if __name__ == "__main__":
    unittest.main()
