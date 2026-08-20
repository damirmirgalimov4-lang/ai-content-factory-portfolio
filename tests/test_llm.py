from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_platform.config import Settings
from agent_platform.llm import (
    CodexExecClient,
    build_agent_messages,
    build_intent_messages,
    create_chat_llm_client,
    create_llm_client,
    parse_intent_response,
)


def settings_for(vault_path: Path, codex_path: Path) -> Settings:
    return Settings(
        telegram_bot_token="test",
        telegram_allowed_user_ids={1},
        vault_path=vault_path,
        openai_api_key="",
        openai_base_url="https://api.openai.com/v1",
        openai_image_model="gpt-image-2",
        openai_image_size="1024x1536",
        openai_image_quality="low",
        deepgram_api_key="",
        deepgram_model="nova-3",
        deepgram_language="ru",
        codex_cli_path=str(codex_path),
        codex_chat_model="gpt-5.6-sol",
        codex_timeout_seconds=30,
        codex_production_timeout_seconds=600,
        codex_workdir=vault_path,
    )


class LlmPromptTest(unittest.TestCase):
    def test_build_agent_messages_includes_context_and_user_text(self) -> None:
        messages = build_agent_messages(
            context="Активный проект: content-factory",
            user_text="Составь план ролика",
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("content-factory", messages[1]["content"])
        self.assertIn("Составь план ролика", messages[1]["content"])

    def test_build_intent_messages_includes_context_and_user_text(self) -> None:
        messages = build_intent_messages(
            context="Активный проект: content-factory",
            user_text="Надо собрать референсы",
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("валидный JSON", messages[0]["content"])
        self.assertIn("content-factory", messages[1]["content"])
        self.assertIn("Надо собрать референсы", messages[1]["content"])

    def test_parse_intent_response_normalizes_action(self) -> None:
        intent = parse_intent_response(
            '{"reply":"Ок","action":{"type":"create_task","text":"Собрать референсы","reason":"это задача"}}'
        )

        self.assertEqual(intent["reply"], "Ок")
        self.assertEqual(intent["action"]["type"], "create_task")
        self.assertEqual(intent["action"]["text"], "Собрать референсы")

    def test_parse_intent_response_rejects_empty_action_text(self) -> None:
        intent = parse_intent_response(
            '{"reply":"Ок","action":{"type":"create_task","text":"","reason":"пусто"}}'
        )

        self.assertEqual(intent["action"]["type"], "reply_only")

    def test_codex_exec_parses_final_agent_message_without_managing_auth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "codex.exe"
            executable.write_bytes(b"test")
            client = CodexExecClient(settings_for(root, executable))
            stdout = "\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "test"}),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {"type": "agent_message", "text": "Ответ"},
                        }
                    ),
                ]
            )
            completed = subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

            with patch("agent_platform.llm.subprocess.run", return_value=completed) as run:
                answer = client.chat([{"role": "user", "content": "Вопрос"}])

            self.assertEqual(answer, "Ответ")
            command = run.call_args.args[0]
            self.assertIn("--ephemeral", command)
            self.assertIn("read-only", command)
            self.assertIn("gpt-5.6-sol", command)
            self.assertNotIn("login", command)
            self.assertNotIn("logout", command)
            self.assertEqual(command[-1], "-")
            self.assertEqual(run.call_args.kwargs["input"], client._build_prompt([{"role": "user", "content": "Вопрос"}]))
            self.assertNotIn("Вопрос", command)

    def test_codex_exec_allows_runtime_workdir_outside_git_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "codex.exe"
            executable.write_bytes(b"test")
            client = CodexExecClient(settings_for(root, executable))
            completed = subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "Ответ"},
                    }
                ),
                stderr="",
            )

            with patch("agent_platform.llm.subprocess.run", return_value=completed) as run:
                client.chat([{"role": "user", "content": "Вопрос"}])

            command = run.call_args.args[0]
            self.assertIn("--skip-git-repo-check", command)

    def test_large_prompt_is_streamed_through_stdin_instead_of_windows_command_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "codex.exe"
            executable.write_bytes(b"test")
            client = CodexExecClient(settings_for(root, executable))
            completed = subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "Готово"},
                    }
                ),
                stderr="",
            )
            large_text = ("длинный сценарий " * 10000).strip()

            with patch("agent_platform.llm.subprocess.run", return_value=completed) as run:
                answer = client.chat([{"role": "user", "content": large_text}])

            self.assertEqual(answer, "Готово")
            command = run.call_args.args[0]
            self.assertEqual(command[-1], "-")
            self.assertNotIn(large_text, command)
            self.assertIn(large_text, run.call_args.kwargs["input"])

    def test_images_are_attached_and_secret_environment_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "codex.exe"
            executable.write_bytes(b"test")
            image = root / "frame.jpg"
            image.write_bytes(b"image")
            client = CodexExecClient(settings_for(root, executable))
            completed = subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "Вижу кадр"},
                    }
                ),
                stderr="",
            )

            environment = {
                "PATH": os.environ.get("PATH", ""),
                "NORMAL_SETTING": "visible",
                "OPENAI_API_KEY": "hidden",
                "TELEGRAM_BOT_TOKEN": "hidden",
                "SESSION_SECRET": "hidden",
            }
            with patch.dict(os.environ, environment, clear=True):
                with patch("agent_platform.llm.subprocess.run", return_value=completed) as run:
                    answer = client.chat_with_images(
                        [{"role": "user", "content": "Что на фото?"}],
                        [image],
                    )

            self.assertEqual(answer, "Вижу кадр")
            command = run.call_args.args[0]
            self.assertIn("--image", command)
            self.assertIn(str(image.resolve()), command)
            child_env = run.call_args.kwargs["env"]
            self.assertEqual(child_env["NORMAL_SETTING"], "visible")
            self.assertNotIn("OPENAI_API_KEY", child_env)
            self.assertNotIn("TELEGRAM_BOT_TOKEN", child_env)
            self.assertNotIn("SESSION_SECRET", child_env)

    def test_chat_client_uses_codex_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "codex.exe"
            executable.write_bytes(b"test")

            client = create_chat_llm_client(settings_for(root, executable))

            self.assertIsInstance(client, CodexExecClient)
            self.assertTrue(client.is_configured)

    def test_production_client_uses_same_codex_model_with_longer_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "codex.exe"
            executable.write_bytes(b"test")
            settings = settings_for(root, executable)

            client = create_llm_client(settings)

            self.assertIsInstance(client, CodexExecClient)
            self.assertEqual(client.model, "gpt-5.6-sol")
            self.assertEqual(client.timeout_seconds, 600)


if __name__ == "__main__":
    unittest.main()
