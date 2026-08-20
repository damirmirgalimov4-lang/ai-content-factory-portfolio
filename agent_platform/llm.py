from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import Settings


class LlmError(RuntimeError):
    pass


class LlmClient(Protocol):
    @property
    def is_configured(self) -> bool:
        raise NotImplementedError

    def chat(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    def chat_with_images(
        self,
        messages: list[dict[str, str]],
        image_paths: list[Path],
    ) -> str:
        raise NotImplementedError


SYSTEM_PROMPT = """Ты рабочий агент в Telegram-интерфейсе личной платформы.

Отвечай по-русски, конкретно и без лишней воды. Используй контекст памяти и активного проекта, но не выдумывай факты, которых там нет.

В обычном режиме чата ты не можешь сам вызывать инструменты и произвольно изменять файлы. Контент-завод отдельно сохраняет только артефакты своих запусков. Если в обычном разговоре нужно что-то сохранить явно, предложи пользователю команды /remember или /note. Если задача похожа на разработку, сформулируй понятный следующий шаг для Codex/code-worker.
"""

INTENT_SYSTEM_PROMPT = """Ты агент-роутер для Telegram-first платформы с памятью.

Твоя задача: ответить пользователю и определить, нужно ли предложить безопасное действие с памятью проекта.

Верни только валидный JSON без markdown. Схема:
{
  "reply": "короткий ответ пользователю",
  "action": {
    "type": "reply_only | remember_global | add_project_note | create_task",
    "text": "текст для записи или пустая строка",
    "reason": "почему это действие уместно"
  }
}

Правила:
- reply_only: если пользователь просто спрашивает, рассуждает или просит совет.
- remember_global: устойчивый факт о пользователе, его предпочтениях, целях или принципах работы.
- add_project_note: мысль, материал, решение или контекст для активного проекта.
- create_task: формулировка будущего действия, которое нужно выполнить.
- Не предлагай действие, если текст слишком неопределённый.
- Не записывай секреты, токены, пароли, ключи.
- Не выдумывай детали, которых нет в сообщении или контексте.
- Ответ должен быть на русском.
"""

VALID_ACTION_TYPES = {
    "reply_only",
    "remember_global",
    "add_project_note",
    "create_task",
}


def build_agent_messages(context: str, user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Контекст из памяти и активного проекта:\n"
                f"{context}\n\n"
                "Сообщение пользователя:\n"
                f"{user_text}"
            ),
        },
    ]


def build_intent_messages(context: str, user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Контекст из памяти и активного проекта:\n"
                f"{context}\n\n"
                "Сообщение пользователя:\n"
                f"{user_text}"
            ),
        },
    ]


def parse_intent_response(raw_text: str) -> dict[str, Any]:
    """Parse strict JSON intent output and normalize invalid actions to reply_only."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()

    decoded = json.loads(cleaned)
    if not isinstance(decoded, dict):
        raise ValueError("Intent response must be a JSON object.")

    reply = str(decoded.get("reply", "")).strip()
    action = decoded.get("action") if isinstance(decoded.get("action"), dict) else {}
    action_type = str(action.get("type", "reply_only")).strip()

    if action_type not in VALID_ACTION_TYPES:
        action_type = "reply_only"

    action_text = str(action.get("text", "")).strip()
    action_reason = str(action.get("reason", "")).strip()

    if action_type != "reply_only" and not action_text:
        action_type = "reply_only"

    return {
        "reply": reply,
        "action": {
            "type": action_type,
            "text": action_text,
            "reason": action_reason,
        },
    }


@dataclass
class NoLlmClient:
    reason: str = "Codex CLI не найден."

    @property
    def is_configured(self) -> bool:
        return False

    def chat(self, messages: list[dict[str, str]]) -> str:
        raise LlmError(self.reason)

    def chat_with_images(
        self,
        messages: list[dict[str, str]],
        image_paths: list[Path],
    ) -> str:
        raise LlmError(self.reason)


class CodexExecClient:
    """Use the installed Codex CLI without reading or managing its OAuth credentials."""

    def __init__(
        self,
        settings: Settings,
        timeout_seconds: int | None = None,
        *,
        ignore_user_config: bool = False,
    ):
        self.executable = find_codex_cli(settings.codex_cli_path)
        self.model = settings.codex_chat_model
        self.timeout_seconds = (
            settings.codex_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self.workdir = settings.codex_workdir.resolve()
        self.ignore_user_config = ignore_user_config

    @property
    def is_configured(self) -> bool:
        return self.executable is not None

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self._chat(messages, image_paths=[])

    def chat_with_images(
        self,
        messages: list[dict[str, str]],
        image_paths: list[Path],
    ) -> str:
        """Ask the same isolated Codex engine while attaching local images."""
        if not image_paths:
            return self.chat(messages)
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise LlmError(f"Файл изображения не найден: {missing[0]}")
        return self._chat(messages, image_paths=image_paths)

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        image_paths: list[Path],
    ) -> str:
        if self.executable is None:
            raise LlmError(
                "Codex CLI не найден. Укажи CODEX_CLI_PATH или установи расширение Codex."
            )

        prompt = self._build_prompt(messages)
        command = [
            str(self.executable),
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--cd",
            str(self.workdir),
        ]
        if self.ignore_user_config:
            command.extend(["--ignore-user-config", "--ignore-rules"])
        for image_path in image_paths:
            command.extend(["--image", str(image_path.resolve())])
        command.extend(
            [
                "--model",
                self.model,
                "--color",
                "never",
                "--json",
                "-",
            ]
        )
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=safe_subprocess_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            raise LlmError(
                f"Codex не завершил ответ за {self.timeout_seconds} секунд."
            ) from exc
        except OSError as exc:
            raise LlmError(f"Не удалось запустить Codex CLI: {exc}") from exc

        answer, error = self._parse_events(completed.stdout)
        if answer:
            return answer
        if completed.returncode != 0:
            error = error or f"процесс завершился с кодом {completed.returncode}"
        raise LlmError(f"Codex не вернул ответ: {error or 'неизвестная ошибка'}")

    @staticmethod
    def _build_prompt(messages: list[dict[str, str]]) -> str:
        parts = [
            "Ты работаешь как LLM-движок Telegram-бота. Не используй инструменты, "
            "не открывай и не изменяй файлы. Ответь только на основе переданной ниже переписки."
        ]
        for message in messages:
            role = str(message.get("role", "user")).upper()
            content = str(message.get("content", "")).strip()
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_events(output: str) -> tuple[str, str]:
        answers: list[str] = []
        errors: list[str] = []
        for raw_line in output.splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    text = str(item.get("text", "")).strip()
                    if text:
                        answers.append(text)
                elif item.get("type") == "error":
                    errors.append(str(item.get("message", "")).strip())
            elif event.get("type") in {"turn.failed", "error"}:
                message = str(event.get("message", "")).strip()
                if message and not message.startswith("Reconnecting"):
                    errors.append(message)
        return (answers[-1] if answers else "", errors[-1][:1000] if errors else "")


_SECRET_ENV_PATTERN = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|AUTH|COOKIE|PRIVATE[_-]?KEY)",
    re.IGNORECASE,
)


def safe_subprocess_environment() -> dict[str, str]:
    """Keep normal process settings while withholding credentials from Codex tools."""
    return {
        key: value
        for key, value in os.environ.items()
        if not _SECRET_ENV_PATTERN.search(key)
    }


def find_codex_cli(configured_path: str = "") -> Path | None:
    if configured_path:
        configured = Path(configured_path).expanduser()
        return configured if configured.is_file() else None

    discovered = shutil.which("codex")
    if discovered:
        return Path(discovered)

    extension_root = Path.home() / ".vscode" / "extensions"
    candidates = list(
        extension_root.glob(
            "openai.chatgpt-*-win32-x64/bin/windows-x86_64/codex.exe"
        )
    )
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def create_llm_client(settings: Settings) -> LlmClient:
    client = CodexExecClient(
        settings,
        timeout_seconds=settings.codex_production_timeout_seconds,
    )
    return client if client.is_configured else NoLlmClient()


def create_chat_llm_client(settings: Settings) -> LlmClient:
    client = CodexExecClient(settings)
    return client if client.is_configured else NoLlmClient()
