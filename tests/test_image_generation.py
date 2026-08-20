from __future__ import annotations

import base64
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent_platform.config import Settings
from agent_platform.image_generation import (
    CodexImageClient,
    ImageGenerationError,
    ImageReference,
    OpenAIImageClient,
    build_visual_draft_prompt,
    create_image_client,
)


def image_settings(vault_path: Path) -> Settings:
    return Settings(
        telegram_bot_token="test",
        telegram_allowed_user_ids={1},
        vault_path=vault_path,
        openai_api_key="test-openai-key",
        openai_base_url="https://api.openai.com/v1",
        openai_image_model="gpt-image-2",
        openai_image_size="1024x1536",
        openai_image_quality="low",
        deepgram_api_key="",
        deepgram_model="nova-3",
        deepgram_language="ru",
    )


class OpenAIImageClientTest(unittest.TestCase):
    def test_generate_decodes_base64_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = OpenAIImageClient(image_settings(Path(temp_dir)))
            expected = b"fake-png-content"
            captured: dict = {}

            def fake_request(path: str, payload: dict) -> dict:
                captured.update({"path": path, "payload": payload})
                return {"data": [{"b64_json": base64.b64encode(expected).decode("ascii")}]}

            client._request = fake_request  # type: ignore[method-assign]
            result = client.generate("Storyboard prompt")

            self.assertEqual(result.content, expected)
            self.assertEqual(captured["path"], "/images/generations")
            self.assertEqual(captured["payload"]["model"], "gpt-image-2")
            self.assertEqual(captured["payload"]["size"], "1024x1536")
            self.assertEqual(captured["payload"]["quality"], "low")

    def test_openai_adapter_preserves_provider_specific_configured_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = replace(
                image_settings(Path(temp_dir)),
                openai_image_model="dall-e-3",
                openai_image_size="1792x1024",
            )
            client = OpenAIImageClient(settings)
            captured: dict = {}

            def fake_request(path: str, payload: dict) -> dict:
                captured.update({"path": path, "payload": payload})
                return {
                    "data": [
                        {"b64_json": base64.b64encode(b"legacy-size-image").decode("ascii")}
                    ]
                }

            client._request = fake_request  # type: ignore[method-assign]
            result = client.generate("Legacy configured size")

            self.assertEqual(result.content, b"legacy-size-image")
            self.assertEqual(captured["payload"]["model"], "dall-e-3")
            self.assertEqual(captured["payload"]["size"], "1792x1024")

    def test_visual_draft_prompt_uses_complete_package(self) -> None:
        prompt = build_visual_draft_prompt(
            "Идея",
            "Шесть сцен",
            "Промпты кадров",
            "Проверка пройдена",
        )
        self.assertIn("Choose rows and columns dynamically", prompt)
        self.assertNotIn("panel numbers 1-6", prompt)
        self.assertIn("Preserve continuity", prompt)
        self.assertIn("Шесть сцен", prompt)
        self.assertIn("Промпты кадров", prompt)
        self.assertIn("Проверка пройдена", prompt)

    def test_codex_image_client_reads_generated_file_without_auth_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "codex.exe"
            executable.write_bytes(b"test")
            settings = replace(
                image_settings(root),
                image_provider="codex",
                codex_cli_path=str(executable),
                codex_workdir=root,
            )
            codex_home = root / "codex-home"
            with patch.dict("os.environ", {"CODEX_HOME": str(codex_home)}):
                client = create_image_client(settings)
            self.assertIsInstance(client, CodexImageClient)

            generated = codex_home / "generated_images" / "test-session" / "result.png"
            reference_path = root / "reference.png"
            reference_path.write_bytes(b"reference-pixels")
            captured_command: list[str] = []
            captured_instruction = ""
            captured_environment: dict[str, str] = {}

            class FakeStdin:
                def write(inner_self, value: str) -> None:
                    nonlocal captured_instruction
                    captured_instruction = value
                    generated.parent.mkdir(parents=True, exist_ok=True)
                    generated.write_bytes(b"oauth-image")

                def close(inner_self) -> None:
                    return None

            class FakeProcess:
                def __init__(inner_self, command: list[str], **kwargs) -> None:
                    captured_command.extend(command)
                    captured_environment.update(kwargs.get("env") or {})
                    inner_self.stdin = FakeStdin()
                    inner_self.returncode = None

                def poll(inner_self):
                    return inner_self.returncode

                def terminate(inner_self) -> None:
                    inner_self.returncode = 0

                def wait(inner_self, timeout=None):
                    inner_self.returncode = 0
                    return 0

                def kill(inner_self) -> None:
                    inner_self.returncode = -9

            with patch.dict(
                "os.environ",
                {
                    "PATH": "/usr/bin",
                    "HOME": str(root),
                    "OPENAI_API_KEY": "must-not-reach-codex",
                    "TELEGRAM_BOT_TOKEN": "must-not-reach-codex",
                },
                clear=True,
            ):
                with patch(
                    "agent_platform.image_generation.subprocess.Popen", FakeProcess
                ), patch("agent_platform.image_generation.time.sleep", return_value=None):
                    result = client.generate(
                        "Visual prompt",
                        [ImageReference("REF-HERO", "character", reference_path)],
                        size="1536x1024",
                    )

            self.assertEqual(result.content, b"oauth-image")
            self.assertIn("workspace-write", captured_command)
            self.assertIn("--add-dir", captured_command)
            self.assertIn("--image", captured_command)
            self.assertIn(str(reference_path), captured_command)
            self.assertNotIn("login", captured_command)
            self.assertNotIn("logout", captured_command)
            self.assertNotIn("Visual prompt", captured_command)
            self.assertEqual(captured_environment["HOME"], str(root))
            self.assertNotIn("OPENAI_API_KEY", captured_environment)
            self.assertNotIn("TELEGRAM_BOT_TOKEN", captured_environment)
            self.assertIn("Visual prompt", captured_instruction)
            self.assertIn("required final raster dimensions are 1536x1024", captured_instruction)
            self.assertIn("do not pass an unsupported size field", captured_instruction)
            self.assertIn("managed generated-images storage", captured_instruction)
            self.assertIn("num_last_images_to_include=1", captured_instruction)
            self.assertNotIn(str(reference_path), captured_instruction)

    def test_codex_image_client_can_terminate_only_its_active_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "codex.exe"
            executable.write_bytes(b"test")
            settings = replace(
                image_settings(root),
                image_provider="codex",
                codex_cli_path=str(executable),
                codex_workdir=root,
            )
            client = CodexImageClient(settings)

            class ActiveProcess:
                def poll(self):
                    return None

            process = ActiveProcess()
            client._active_process = process  # type: ignore[assignment]
            with patch.object(client, "_stop_process") as stop:
                stopped = client.cancel_active()

            self.assertTrue(stopped)
            stop.assert_called_once_with(process)

    def test_generation_lock_rejects_concurrent_holder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "codex.exe"
            executable.write_bytes(b"test")
            settings = replace(
                image_settings(root),
                image_provider="codex",
                codex_cli_path=str(executable),
                codex_workdir=root,
            )
            first = CodexImageClient(settings)
            second = CodexImageClient(settings)

            with first._generation_lock():
                with self.assertRaises(ImageGenerationError) as raised:
                    with second._generation_lock():
                        self.fail("the second client acquired an active generation lock")

            self.assertEqual(raised.exception.code, "codex_image_busy")


if __name__ == "__main__":
    unittest.main()
