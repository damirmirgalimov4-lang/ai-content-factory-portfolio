from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import Settings
from .content_factory import (
    PIPELINE,
    ContentFactoryStore,
    ContentRun,
    build_stage_messages,
)
from .content_presentation import present_stage, present_video_prompts
from .diagnostics import ReadOnlyRunDiagnostic
from .partner_research import (
    IDEA_SIMILARITY_THRESHOLD,
    LONG_FORM_YOUTUBE_SECONDS,
    canonical_source_key,
    idea_similarity,
)
from .frame_generation import FrameBatchGenerator, FrameResult
from .llm import (
    LlmError,
    build_agent_messages,
    build_intent_messages,
    create_chat_llm_client,
    create_llm_client,
    parse_intent_response,
)
from .maintenance import GitRepairManager, RepairRecord
from .image_generation import (
    ImageReference,
    ImageGenerationError,
    build_visual_draft_prompt,
    create_image_client,
)
from .image_qa import PostImageQa
from .kie import KieClient
from .ltx import LtxClient
from .polza import PolzaClient
from .viktor import ViktorClient
from .production import (
    ProductionContractError,
    ProductionStore,
    build_image_prompt_contract_messages,
    build_scene_contract_messages,
    build_visual_bible_contract_messages,
    format_image_prompt_contract,
    format_scene_contract,
    format_visual_bible_contract,
    generation_plan_summary,
    merge_image_prompt_contract,
    parse_reference_plan,
    parse_plain_image_prompt_json,
    parse_plain_scene_json,
    parse_plain_visual_bible_json,
    parse_scene_contract,
    parse_visual_bible_contract,
    strip_json_contract,
    validate_english_image_prompts,
    validate_image_plan_against_visual_bible,
    validate_script_scene_plan,
)
from .reference_generation import ReferenceBatchGenerator, ReferenceResult
from .shared_content import SharedContentItem, SharedContentStore
from .storyboard import (
    GUIDED_STORYBOARD_MAX_REFERENCE_BYTES,
    GUIDED_STORYBOARD_SHEET_SIZE,
    GUIDED_STORYBOARD_WORKFLOW,
    STORYBOARD_PROMPT_SOURCE_LABEL,
    STORYBOARD_PROMPT_SOURCE_STATUS,
    STORYBOARD_STAGE_MAP,
    STORYBOARD_STAGES,
    StoryboardProject,
    StoryboardStore,
    build_guided_storyboard_plan_messages,
)
from .telegram_formatting import markdown_to_telegram_html
from .vault import VaultStore
from .video_jobs import VideoJobManager
from .video_prompting import (
    ImageInspectionError,
    VideoPromptBuilder,
    create_image_inspector,
)
from .video_profiles import profile_label, profiles_for_model, video_profile
from .video_provider import VideoProviderError


class TelegramApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class TelegramUser:
    user_id: int
    chat_id: int
    first_name: str


InlineKeyboard = list[list[tuple[str, str]]]
BotCommand = tuple[str, str]

CONTENT_FACTORY_COMMANDS: tuple[BotCommand, ...] = (
    ("start", "Открыть главное меню"),
    ("new_video", "Создать новый ролик"),
    ("runs", "Открыть список роликов"),
    ("radar", "Запустить радар идей"),
    ("ideas", "Открыть сохранённые идеи радара"),
    ("inbox", "Материалы от партнёра"),
    ("factory", "Открыть контент-завод"),
    ("status", "Проверить состояние системы"),
    ("help", "Показать доступные команды"),
)


@dataclass(frozen=True)
class BotResponse:
    text: str
    keyboard: InlineKeyboard | None = None
    replace_message: bool = False
    render_markdown: bool = False


class TelegramClient:
    """Small Telegram Bot API client built on stdlib to keep the MVP dependency-free."""

    def __init__(self, token: str, request_timeout_seconds: int = 60):
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.file_base_url = f"https://api.telegram.org/file/bot{token}"
        self.request_timeout_seconds = request_timeout_seconds

    def get_me(self) -> dict[str, Any]:
        return dict(self._request("getMe", {}).get("result", {}))

    def download_file(
        self,
        file_id: str,
        destination: Path,
        max_bytes: int | None = None,
    ) -> Path:
        """Download one Telegram attachment into a caller-controlled local path."""
        result = self._request("getFile", {"file_id": file_id}).get("result", {})
        file_path = str(result.get("file_path", "")).strip()
        if not file_path:
            raise TelegramApiError("Telegram не вернул путь к файлу.")
        file_size = result.get("file_size")
        if (
            max_bytes is not None
            and isinstance(file_size, int)
            and file_size > max_bytes
        ):
            raise TelegramApiError("Файл Telegram превышает допустимый размер.")

        request = urllib.request.Request(
            f"{self.file_base_url}/{file_path}",
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout_seconds,
            ) as response:
                if max_bytes is None:
                    content = response.read()
                else:
                    content = response.read(max_bytes + 1)
        except urllib.error.URLError as exc:
            raise TelegramApiError(str(exc)) from exc
        if max_bytes is not None and len(content) > max_bytes:
            raise TelegramApiError("Файл Telegram превышает допустимый размер.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return destination

    def get_updates(self, offset: int | None, timeout_seconds: int) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": timeout_seconds,
            "allowed_updates": ["message", "callback_query"],
        }
        if offset is not None:
            payload["offset"] = offset

        response = self._request("getUpdates", payload)
        return response.get("result", [])

    def send_message(
        self,
        chat_id: int,
        text: str,
        keyboard: InlineKeyboard | None = None,
        render_markdown: bool = False,
    ) -> None:
        self._ensure_readable(text)
        chunks = self._chunks(text, limit=3200 if render_markdown else 3900)
        for index, chunk in enumerate(chunks):
            rendered = markdown_to_telegram_html(chunk) if render_markdown else chunk
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": rendered,
                "disable_web_page_preview": True,
            }
            if render_markdown:
                payload["parse_mode"] = "HTML"
            if keyboard and index == len(chunks) - 1:
                payload["reply_markup"] = self._reply_markup(keyboard)
            self._request(
                "sendMessage",
                payload,
            )

    def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        keyboard: InlineKeyboard | None = None,
        render_markdown: bool = False,
    ) -> None:
        self._ensure_readable(text)
        rendered = markdown_to_telegram_html(text) if render_markdown else text
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": rendered,
            "disable_web_page_preview": True,
        }
        if render_markdown:
            payload["parse_mode"] = "HTML"
        if keyboard:
            payload["reply_markup"] = self._reply_markup(keyboard)
        self._request("editMessageText", payload)

    def answer_callback_query(self, callback_query_id: str) -> None:
        self._request("answerCallbackQuery", {"callback_query_id": callback_query_id})

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self._request("sendChatAction", {"chat_id": chat_id, "action": action})

    def set_commands(self, commands: tuple[BotCommand, ...]) -> None:
        """Publish the bot's primary commands to Telegram's native command menu."""

        payload_commands: list[dict[str, str]] = []
        for command, description in commands:
            if not re.fullmatch(r"[a-z0-9_]{1,32}", command):
                raise ValueError(f"Некорректная Telegram-команда: {command}")
            clean_description = description.strip()
            if not clean_description:
                raise ValueError(f"У команды /{command} отсутствует описание.")
            self._ensure_readable(clean_description)
            payload_commands.append(
                {"command": command, "description": clean_description[:256]}
            )
        self._request("setMyCommands", {"commands": payload_commands})

    def set_commands_menu_button(self) -> None:
        """Show Telegram's native menu button instead of a web-app menu."""

        self._request(
            "setChatMenuButton",
            {"menu_button": {"type": "commands"}},
        )

    def send_photo(
        self,
        chat_id: int,
        path: Path,
        caption: str,
        keyboard: InlineKeyboard | None = None,
    ) -> None:
        self._ensure_readable(caption)
        fields: dict[str, str] = {
            "chat_id": str(chat_id),
            "caption": caption[:1000],
        }
        if keyboard:
            fields["reply_markup"] = json.dumps(
                self._reply_markup(keyboard),
                ensure_ascii=False,
            )
        self._request_multipart(
            "sendPhoto",
            fields=fields,
            file_field="photo",
            filename=path.name,
            content=path.read_bytes(),
            content_type="image/png",
        )

    def send_photo_album(
        self,
        chat_id: int,
        items: list[tuple[Path, str]],
    ) -> None:
        """Send ordered frames as Telegram albums while respecting the 10-item API limit."""

        for batch_start in range(0, len(items), 10):
            batch = items[batch_start : batch_start + 10]
            if len(batch) == 1:
                path, caption = batch[0]
                self.send_photo(chat_id, path, caption)
                continue

            media: list[dict[str, str]] = []
            files: list[tuple[str, str, bytes, str]] = []
            for index, (path, caption) in enumerate(batch):
                self._ensure_readable(caption)
                attachment_name = f"photo{index}"
                item = {
                    "type": "photo",
                    "media": f"attach://{attachment_name}",
                }
                if caption:
                    item["caption"] = caption[:1000]
                media.append(item)
                files.append(
                    (
                        attachment_name,
                        path.name,
                        path.read_bytes(),
                        self._image_content_type(path),
                    )
                )

            self._request_multipart_files(
                "sendMediaGroup",
                fields={
                    "chat_id": str(chat_id),
                    "media": json.dumps(media, ensure_ascii=False),
                },
                files=files,
            )

    def send_video(
        self,
        chat_id: int,
        path: Path,
        caption: str,
        keyboard: InlineKeyboard | None = None,
    ) -> None:
        self._ensure_readable(caption)
        fields: dict[str, str] = {"chat_id": str(chat_id), "caption": caption[:1000]}
        if keyboard:
            fields["reply_markup"] = json.dumps(self._reply_markup(keyboard), ensure_ascii=False)
        self._request_multipart(
            "sendVideo",
            fields=fields,
            file_field="video",
            filename=path.name,
            content=path.read_bytes(),
            content_type="video/mp4",
        )

    def send_document(self, chat_id: int, path: Path, caption: str = "") -> None:
        self._ensure_readable(caption)
        self._request_multipart(
            "sendDocument",
            fields={"chat_id": str(chat_id), "caption": caption[:1000]},
            file_field="document",
            filename=path.name,
            content=path.read_bytes(),
            content_type="application/octet-stream",
        )

    def _request(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/{method}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise TelegramApiError(str(exc)) from exc

        decoded = json.loads(body)
        if not decoded.get("ok"):
            raise TelegramApiError(decoded.get("description", "Telegram API error"))
        return decoded

    def _request_multipart(
        self,
        method: str,
        *,
        fields: dict[str, str],
        file_field: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> dict[str, Any]:
        return self._request_multipart_files(
            method,
            fields=fields,
            files=[(file_field, filename, content, content_type)],
        )

    def _request_multipart_files(
        self,
        method: str,
        *,
        fields: dict[str, str],
        files: list[tuple[str, str, bytes, str]],
    ) -> dict[str, Any]:
        boundary = f"----AgentPlatform{uuid.uuid4().hex}"
        body = bytearray()
        for name, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode("ascii"))
            body.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii")
            )
            body.extend(value.encode("utf-8"))
            body.extend(b"\r\n")
        for file_field, filename, content, content_type in files:
            body.extend(f"--{boundary}\r\n".encode("ascii"))
            body.extend(
                (
                    f'Content-Disposition: form-data; name="{file_field}"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8")
            )
            body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("ascii"))
            body.extend(content)
            body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("ascii"))

        request = urllib.request.Request(
            f"{self.base_url}/{method}",
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout_seconds,
            ) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise TelegramApiError(str(exc)) from exc
        decoded = json.loads(response_body)
        if not decoded.get("ok"):
            raise TelegramApiError(decoded.get("description", "Telegram API error"))
        return decoded

    @staticmethod
    def _image_content_type(path: Path) -> str:
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "application/octet-stream")

    def _chunks(self, text: str, limit: int) -> list[str]:
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        current = ""
        blocks = self._semantic_blocks(text)
        for block in blocks:
            if len(block) > limit:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_large_block(block, limit))
                continue
            candidate = f"{current}\n\n{block}" if current else block
            if len(candidate) > limit:
                chunks.append(current)
                current = block
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _semantic_blocks(text: str) -> list[str]:
        """Keep complete sections, frames, and numbered ideas together when possible."""

        raw_blocks = [
            item.strip("\n")
            for item in re.split(r"\n\s*\n", text)
            if item.strip()
        ]
        blocks: list[str] = []
        section_pattern = re.compile(
            r"^(?:#{1,6}\s+\S|"
            r"(?:Кадр|Сцена|Шаг|Этап|Промпт|Идея|Вариант)\s+\d+\b|"
            r"\d+[.)]\s+\S)",
            re.IGNORECASE,
        )
        for raw_block in raw_blocks:
            current: list[str] = []
            for line in raw_block.splitlines():
                if section_pattern.match(line.strip()) and current:
                    blocks.append("\n".join(current).strip())
                    current = [line]
                else:
                    current.append(line)
            if current:
                blocks.append("\n".join(current).strip())

        grouped: list[str] = []
        index = 0
        while index < len(blocks):
            block = blocks[index]
            if (
                index + 1 < len(blocks)
                and len(block.splitlines()) == 1
                and re.match(r"^#{1,6}\s+\S", block.strip())
            ):
                grouped.append(f"{block}\n\n{blocks[index + 1]}")
                index += 2
                continue
            grouped.append(block)
            index += 1
        return grouped

    @staticmethod
    def _split_large_block(block: str, limit: int) -> list[str]:
        """Split an oversized semantic block at lines, sentences, then words."""

        parts: list[str] = []
        current = ""
        for line in block.splitlines():
            units = TelegramClient._logical_line_units(line, limit)
            for unit in units:
                candidate = f"{current}\n{unit}" if current else unit
                if len(candidate) > limit and current:
                    parts.append(current)
                    current = unit
                else:
                    current = candidate
        if current:
            parts.append(current)
        return parts

    @staticmethod
    def _logical_line_units(line: str, limit: int) -> list[str]:
        """Prefer complete sentences; split by words only when one sentence is oversized."""

        if len(line) <= limit:
            return [line]
        sentences = [
            item.strip()
            for item in re.findall(r".+?(?:[.!?](?=\s|$)|$)", line)
            if item.strip()
        ]
        if not sentences:
            sentences = [line]

        units: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > limit:
                if current:
                    units.append(current)
                    current = ""
                units.extend(TelegramClient._word_chunks(sentence, limit))
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > limit:
                units.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            units.append(current)
        return units

    @staticmethod
    def _word_chunks(text: str, limit: int) -> list[str]:
        chunks: list[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) > limit and current:
                chunks.append(current)
                current = word
            elif len(word) > limit:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(word[index : index + limit] for index in range(0, len(word), limit))
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _reply_markup(self, keyboard: InlineKeyboard) -> dict[str, Any]:
        for row in keyboard:
            for label, _ in row:
                self._ensure_readable(label)
        return {
            "inline_keyboard": [
                [
                    (
                        {"text": text, "url": target[4:]}
                        if target.startswith("url:")
                        else {"text": text, "callback_data": target}
                    )
                    for text, target in row
                ]
                for row in keyboard
            ]
        }

    @staticmethod
    def _ensure_readable(text: str) -> None:
        """Reject text whose Cyrillic was already destroyed before Telegram sees it."""

        question_marks = text.count("?")
        cyrillic_letters = len(re.findall(r"[А-Яа-яЁё]", text))
        if question_marks >= 8 and question_marks > cyrillic_letters:
            raise TelegramApiError(
                "Исходящее Telegram-сообщение повреждено кодировкой и не было отправлено."
            )


class AgentTelegramBot:
    """Command router that exposes the file-backed agent memory through Telegram."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.vault = VaultStore(settings.vault_path)
        shared_root = (
            settings.shared_content_path
            or settings.vault_path.parent / "shared-content"
        )
        self.shared_content = SharedContentStore(shared_root)
        test_shared_root = (
            settings.test_shared_content_path
            or shared_root.parent / "shared-content-test"
        )
        self.test_shared_content = (
            self.shared_content
            if test_shared_root.resolve() == shared_root.resolve()
            else SharedContentStore(test_shared_root)
        )
        self.telegram = TelegramClient(settings.telegram_bot_token)
        self.chat_llm = create_chat_llm_client(settings)
        self.llm = create_llm_client(settings)
        self.image_client = create_image_client(settings)
        self.image_inspector = create_image_inspector(settings)
        self.polza_client = PolzaClient(settings)
        self.kie_client = KieClient(settings)
        self.viktor_client = ViktorClient(settings)
        self.ltx_client = LtxClient(settings)
        self.offset: int | None = None
        self.content_factory: ContentFactoryStore | None = None
        self.storyboards: StoryboardStore | None = None
        self.production_store: ProductionStore | None = None
        self.last_video_poll_at = 0.0
        self._active_image_jobs: dict[str, threading.Event] = {}
        self._active_image_jobs_lock = threading.Lock()
        self._frame_gallery_lock = threading.Lock()
        self._frame_gallery_in_flight: set[tuple[int, str]] = set()
        self._frame_gallery_not_before: dict[tuple[int, str], float] = {}
        self._frame_gallery_cooldown_seconds = 30.0
        self._storyboard_sheet_generation_lock = threading.Lock()
        self.repair_manager: GitRepairManager | None = None
        self._active_repair_jobs: set[str] = set()
        self._active_repair_jobs_lock = threading.Lock()
        self.radar = None
        if settings.radar_vault_path is not None:
            # Imported lazily because the Partner controller reuses the Telegram
            # transport types defined in this module.
            from .radar_bot import ContentFactoryRadarBot

            radar_root = settings.radar_vault_path.expanduser().resolve()
            self._validate_radar_root(
                radar_root,
                settings,
                test_shared_root,
            )
            radar_settings = replace(
                settings,
                vault_path=radar_root,
                codex_workdir=radar_root,
                shared_content_path=test_shared_root,
                telegram_tester_user_ids=set(),
                test_vault_path=None,
                test_shared_content_path=None,
                radar_redirect_to_content_factory=False,
            )
            self.radar = ContentFactoryRadarBot(radar_settings)
            self.radar.attach_transport(self.telegram, self.test_shared_content)

    @staticmethod
    def _validate_radar_root(
        radar_root: Path,
        settings: Settings,
        test_shared_root: Path,
    ) -> None:
        data_root = settings.vault_path.expanduser().resolve().parent
        forbidden = {
            settings.vault_path.expanduser().resolve(),
            (data_root / "vault-partner").resolve(),
            (data_root / "vault-partner-test").resolve(),
            test_shared_root.expanduser().resolve(),
        }
        if settings.test_vault_path is not None:
            forbidden.add(settings.test_vault_path.expanduser().resolve())
        if settings.shared_content_path is not None:
            forbidden.add(settings.shared_content_path.expanduser().resolve())
        if settings.test_shared_content_path is not None:
            forbidden.add(settings.test_shared_content_path.expanduser().resolve())
        if any(
            radar_root == path
            or radar_root.is_relative_to(path)
            or path.is_relative_to(radar_root)
            for path in forbidden
        ):
            raise ValueError(
                "RADAR_VAULT_PATH must be a dedicated runtime and must not overlap "
                "main, Partner, tester, or shared-content paths."
            )

    def run_forever(self) -> None:
        self.vault.ensure_bootstrap()
        self.shared_content.ensure()
        self.test_shared_content.ensure()
        if self.radar is not None:
            self.radar.attach_transport(self.telegram, self.test_shared_content)
            self.radar._prepare_runtime()
        self._recover_interrupted_image_attempts()
        self._recover_interrupted_storyboard_sheets()
        self._recover_interrupted_repairs()
        self._configure_telegram_command_menu()
        print(f"Agent Telegram bot is running. Vault: {self.settings.vault_path}")

        while True:
            try:
                updates = self.telegram.get_updates(
                    offset=self.offset,
                    timeout_seconds=self.settings.poll_timeout_seconds,
                )
                for update in updates:
                    self.offset = update["update_id"] + 1
                    self.handle_update(update)
                video_poll_interval = min(
                    self.settings.polza_poll_interval_seconds,
                    self.settings.kie_poll_interval_seconds,
                )
                if time.monotonic() - self.last_video_poll_at >= max(
                    video_poll_interval, 5
                ):
                    self._resume_pending_video_jobs()
                    self.last_video_poll_at = time.monotonic()
            except KeyboardInterrupt:
                print("Stopping bot.")
                return
            except Exception as exc:
                print(f"Bot loop error: {exc}", file=sys.stderr)
                time.sleep(5)

    def _recover_interrupted_image_attempts(self) -> None:
        """A new bot process cannot own image calls left in generating state by the old one."""

        content = self._content_store()
        production = self._production_store()
        for run in content.list_runs(limit=1000):
            production.recover_interrupted_images(run.run_id)

    def _recover_interrupted_storyboard_sheets(self) -> None:
        """Never retry a sheet request after losing its in-memory provider outcome."""

        self._storyboard_store().recover_interrupted_sheet_generations()

    def _recover_interrupted_repairs(self) -> None:
        """Never resume a write operation whose exact Git state is no longer known."""

        try:
            self._repair_manager().recover_interrupted()
        except Exception as exc:
            print(f"Repair recovery error: {exc}", file=sys.stderr)

    def _configure_telegram_command_menu(self) -> None:
        """Configure native Telegram controls without making bot startup depend on them."""

        try:
            self.telegram.set_commands(CONTENT_FACTORY_COMMANDS)
            self.telegram.set_commands_menu_button()
        except TelegramApiError as exc:
            print(f"Telegram command menu setup error: {exc}", file=sys.stderr)

    def _clear_main_input_state_if_present(self, user_id: int) -> None:
        if self.vault.get_input_state(user_id) is not None:
            self.vault.clear_input_state(user_id)

    def _clear_radar_input_state_if_present(self, user_id: int) -> None:
        if self.radar is None:
            return
        state = self.radar.vault.get_input_state(user_id)
        if self.radar.handles_input_state(state):
            self.radar.vault.clear_input_state(user_id)

    def handle_update(self, update: dict[str, Any]) -> None:
        callback_query = update.get("callback_query")
        if callback_query:
            self.handle_callback_query(callback_query)
            return

        message = update.get("message")
        if not message:
            return

        user = self._extract_user(message)
        if user is None:
            return

        if not self._is_authorized(user.user_id):
            self.telegram.send_message(
                user.chat_id,
                "Доступ закрыт. Твой user id: "
                f"{user.user_id}. Добавь его в TELEGRAM_ALLOWED_USER_IDS.",
            )
            return

        text = str(message.get("text", "")).strip()

        try:
            radar_state = None
            if self.radar is not None:
                self.radar.attach_transport(self.telegram, self.test_shared_content)
                radar_state = self.radar.vault.get_input_state(user.user_id)
            if (
                self.radar is not None
                and self.radar.handles_input_state(radar_state)
                and text.casefold() == "/cancel"
            ):
                self._clear_main_input_state_if_present(user.user_id)
                self.radar.vault.clear_input_state(user.user_id)
                response = self.radar.home(user)
            elif (
                self.radar is not None
                and (radar_state or {}).get("kind") == "research_account_import"
                and message.get("document")
            ):
                self._clear_main_input_state_if_present(user.user_id)
                response = self.radar.handle_account_document(user, message)
            elif (
                (self.vault.get_input_state(user.user_id) or {}).get("kind")
                == "storyboard_reference"
            ):
                if message.get("photo"):
                    response = self.handle_storyboard_reference_photo(user, message)
                else:
                    response = BotResponse(
                        "Для референса пришли именно фото отдельным сообщением. "
                        "Описание можно добавить в подписи к изображению.",
                        keyboard=[[('Отмена', 'input:cancel')]],
                    )
            elif not text:
                response = BotResponse("Пока обрабатываю только текстовые сообщения.")
            elif (
                self.radar is not None
                and self.radar.handles_input_state(radar_state)
                and not text.startswith("/")
            ):
                assert radar_state is not None
                self._clear_main_input_state_if_present(user.user_id)
                response = self.radar.dispatch_radar_input(user, text, radar_state)
            else:
                if (
                    self.radar is not None
                    and self.radar.handles_input_state(radar_state)
                    and text.startswith("/")
                ):
                    self.radar.vault.clear_input_state(user.user_id)
                input_state = self.vault.get_input_state(user.user_id)
                if input_state and not text.startswith("/"):
                    response = self.handle_input_state(user, text, input_state)
                else:
                    if not text.startswith("/"):
                        self._show_typing(user.chat_id)
                    response = self.dispatch(user, text)
        except Exception as exc:
            response = BotResponse(
                text=f"Ошибка: {exc}\n\nСостояние сохранено. Можно повторить действие.",
                keyboard=self.main_menu_keyboard(),
            )

        self.send_response(user.chat_id, response)

    def handle_callback_query(self, callback_query: dict[str, Any]) -> None:
        callback_id = str(callback_query.get("id", ""))
        message = callback_query.get("message") or {}
        from_user = callback_query.get("from") or {}
        chat = message.get("chat") or {}
        if not from_user or not chat:
            return
        if not self._is_private_chat(from_user, chat):
            return

        user = TelegramUser(
            user_id=int(from_user["id"]),
            chat_id=int(chat["id"]),
            first_name=str(from_user.get("first_name", "")),
        )
        if not self._is_authorized(user.user_id):
            return

        try:
            if callback_id:
                self.telegram.answer_callback_query(callback_id)
        except TelegramApiError:
            pass

        try:
            response = self.dispatch_callback(user, str(callback_query.get("data", "")))
        except Exception as exc:
            response = BotResponse(
                text=f"Ошибка: {exc}\n\nЗапуск не потерян. Попробуй ещё раз.",
                keyboard=self.main_menu_keyboard(),
            )

        if response is None:
            return

        message_id = int(message.get("message_id", 0))
        if response.replace_message and message_id and len(response.text) <= 3900:
            try:
                self.telegram.edit_message(
                    user.chat_id,
                    message_id,
                    response.text,
                    response.keyboard,
                    response.render_markdown,
                )
                return
            except TelegramApiError:
                pass
        self.send_response(user.chat_id, response)

    def send_response(self, chat_id: int, response: str | BotResponse) -> None:
        if isinstance(response, BotResponse):
            self.telegram.send_message(
                chat_id,
                response.text,
                response.keyboard,
                render_markdown=response.render_markdown,
            )
            return
        self.telegram.send_message(chat_id, response)

    def dispatch(self, user: TelegramUser, text: str) -> str | BotResponse:
        command, arg = self._split_command(text)
        if command:
            self._clear_radar_input_state_if_present(user.user_id)

        if command == "/start":
            if arg.strip().casefold() == "radar" and self.radar is not None:
                self._clear_main_input_state_if_present(user.user_id)
                return self.radar.home(user)
            if arg.strip().casefold() == "ideas":
                return self.ideas_inbox()
            idea_link = re.fullmatch(
                r"idea_(prod|test)_(CR-\d{8}-\d{3})",
                arg.strip(),
                re.IGNORECASE,
            )
            if idea_link:
                return self.idea_item(
                    f"{idea_link.group(1).lower()}:{idea_link.group(2).upper()}"
                )
            return self.main_menu(user)
        if command == "/help":
            return self.help_text(user)
        if command == "/factory":
            return self.factory_home(user)
        if command == "/new_video":
            return self.start_new_content(user)
        if command == "/runs":
            return self.runs_menu(user)
        if command in {"/radar", "/discover"} and self.radar is not None:
            self._clear_main_input_state_if_present(user.user_id)
            self.radar.attach_transport(self.telegram, self.test_shared_content)
            return self.radar.home(user)
        if command == "/research" and self.radar is not None:
            self._clear_main_input_state_if_present(user.user_id)
            self.radar.attach_transport(self.telegram, self.test_shared_content)
            return self.radar.research_search_home()
        if command == "/results" and self.radar is not None:
            self._clear_main_input_state_if_present(user.user_id)
            self.radar.attach_transport(self.telegram, self.test_shared_content)
            return self.radar.results_home()
        if command == "/ideas" or command == "/radar":
            return self.ideas_inbox()
        if command == "/inbox":
            return self.shared_inbox()
        if command == "/whoami":
            return f"user_id: {user.user_id}\nchat_id: {user.chat_id}"
        if command == "/status":
            return self.status_text(user)
        if command == "/work_status":
            return self.work_status()
        if command == "/model":
            return self.model_text()
        if command == "/approve":
            return self.approve_action(user)
        if command == "/cancel":
            return self.cancel_action(user)
        if command == "/projects":
            return self.projects_text(user)
        if command == "/new_project":
            return self.create_project(user, arg)
        if command == "/use":
            return self.use_project(user, arg)
        if command == "/context":
            return self.vault.context_summary(user.user_id)
        if command == "/remember":
            return self.remember(arg)
        if command == "/note":
            return self.note(user, arg)
        if command == "/task":
            return self.create_task(user, arg)
        if command == "/tasks":
            return self.tasks_text(user, arg)
        if command == "/done":
            return self.complete_task(user, arg)
        if command == "/agents":
            return self.agents_text()
        if command == "/new_agent":
            return self.create_agent(arg)
        if command == "/agent":
            return self.read_agent(arg)
        if command == "/storyboard":
            return self.storyboard_home(user)

        if self.chat_llm.is_configured:
            return self.answer_with_llm(user, text)

        path = self.vault.log_conversation(user.user_id, text)
        return (
            "Я ещё не подключён к LLM, поэтому пока не отвечаю как полноценный агент.\n"
            "Но сообщение сохранено в журнал, чтобы не потерять контекст:\n"
            f"{path}\n\n"
            "Для явной записи используй /note или /remember."
        )

    def dispatch_callback(self, user: TelegramUser, data: str) -> BotResponse | None:
        if self.radar is not None and data == "input:cancel":
            radar_state = self.radar.vault.get_input_state(user.user_id)
            if self.radar.handles_input_state(radar_state):
                self._clear_main_input_state_if_present(user.user_id)
                self.radar.vault.clear_input_state(user.user_id)
                return self.radar.home(user, replace_message=True)

        radar_callback = self.radar is not None and (
            data == "radar:home" or self.radar.handles_callback(data)
        )
        if self.radar is not None and radar_callback:
            self._clear_main_input_state_if_present(user.user_id)
            self.radar.attach_transport(self.telegram, self.test_shared_content)
            return self.radar.dispatch_radar_callback(user, data)
        if self.radar is not None:
            radar_state = self.radar.vault.get_input_state(user.user_id)
            if self.radar.handles_input_state(radar_state):
                self.radar.vault.clear_input_state(user.user_id)
        main_input_state = self.vault.get_input_state(user.user_id) or {}
        if (
            main_input_state.get("kind", "").startswith("storyboard")
            and data != "input:cancel"
            and not data.startswith("sb:")
        ):
            self.vault.clear_input_state(user.user_id)
        if data == "menu:main":
            self._clear_main_input_state_if_present(user.user_id)
            return self.main_menu(user, replace_message=True)
        if data == "menu:factory":
            self._clear_main_input_state_if_present(user.user_id)
            return self.factory_home(user, replace_message=True)
        if data == "menu:tasks":
            return BotResponse(
                text=self.tasks_text(user, ""),
                keyboard=[[('« Главное меню', 'menu:main')]],
                replace_message=True,
            )
        if data == "menu:context":
            return BotResponse(
                text=self.vault.context_summary(user.user_id),
                keyboard=[[('« Главное меню', 'menu:main')]],
            )
        if data == "menu:status":
            return BotResponse(
                text=self.status_text(user),
                keyboard=[
                    [('🛠 Технический отчёт', 'menu:work_status')],
                    [('« Главное меню', 'menu:main')],
                ],
                replace_message=True,
            )
        if data == "menu:work_status":
            return self.work_status(replace_message=True)
        if data == "input:cancel":
            input_state = self.vault.get_input_state(user.user_id) or {}
            self.vault.clear_input_state(user.user_id)
            if input_state.get("kind", "").startswith("storyboard"):
                project_id = input_state.get("project_id", "")
                if project_id and self._storyboard_store().get_project(project_id):
                    return self.storyboard_project_detail(
                        project_id,
                        replace_message=True,
                    )
                return self.storyboard_home(user, replace_message=True)
            return self.factory_home(user, replace_message=True)
        if data == "cf:new":
            return self.start_new_content(user, replace_message=True)
        if data == "cf:list":
            return self.runs_menu(user, replace_message=True)
        if data == "cf:videos":
            return self.video_jobs_menu(replace_message=True)
        if data == "shared:list":
            return self.shared_inbox(replace_message=True)
        if data == "ideas:list":
            return self.ideas_inbox(replace_message=True)
        if data == "cf:storyboards":
            self._clear_main_input_state_if_present(user.user_id)
            return self.storyboard_home(user, replace_message=True)
        if data == "cf:storyboard_new":
            return self.start_new_storyboard(user, replace_message=True)
        if data.startswith("sb:"):
            return self.dispatch_storyboard_callback(user, data)

        action, separator, action_payload = data.partition(":")
        if not separator or not action_payload:
            return BotResponse("Неизвестная кнопка.", self.main_menu_keyboard())
        if action.startswith("cf_storyboard") or action == "cf_legacy_visual":
            return BotResponse(
                "Storyboard временно отключён. Используй обычный путь через «Новый ролик».",
                self.main_menu_keyboard(),
                replace_message=True,
            )
        if action == "cf_run":
            run_id = action_payload
            return self.run_detail(run_id, replace_message=True)
        if action == "shared_item":
            return self.shared_item(action_payload, replace_message=True)
        if action == "idea_item":
            return self.idea_item(action_payload, replace_message=True)
        if action == "idea_script":
            return self.idea_script(action_payload)
        if action == "idea_accept":
            return self.accept_idea(user, action_payload)
        if action == "shared_accept":
            return self.accept_shared_item(user, action_payload)
        if action == "shared_return":
            self.vault.set_input_state(
                user.user_id,
                {"kind": "shared_return", "item_id": action_payload},
            )
            return BotResponse(
                text=(
                    f"↩️ Возврат {action_payload}\n\n"
                    "Напиши коротко, что партнёру нужно уточнить или изменить."
                ),
                keyboard=[[('Отмена', f'shared_return_cancel:{action_payload}')]],
                replace_message=True,
            )
        if action == "shared_return_cancel":
            state = self.vault.get_input_state(user.user_id)
            if state and state.get("kind") == "shared_return":
                self.vault.clear_input_state(user.user_id)
            return self.shared_item(action_payload, replace_message=True)
        if action == "shared_reject":
            return self.reject_shared_item(action_payload)
        if action == "cf_view":
            run_id = action_payload
            return self.view_current_artifact(run_id)
        if action == "cf_generate_visual":
            run_id = action_payload
            return self._start_image_job(
                user,
                run_id,
                "генерация кадров",
                lambda event: self.generate_frames(user, run_id, cancel_event=event),
            )
        if action == "cf_regenerate_frames":
            scenes = self._production_store().scenes(action_payload)
            return self._start_image_job(
                user,
                action_payload,
                "повторная генерация кадров",
                lambda event: self.generate_frames(
                    user,
                    action_payload,
                    scene_ids=[scene.scene_id for scene in scenes],
                    cancel_event=event,
                ),
            )
        if action == "cf_generate_refs":
            return self._start_image_job(
                user,
                action_payload,
                "генерация референсов",
                lambda event: self.generate_references(
                    user, action_payload, cancel_event=event
                ),
            )
        if action == "cf_show_refs":
            return self.show_references(user, action_payload)
        if action == "cf_retry_ref":
            run_id, reference_id = self._parse_run_scene(action_payload)
            return self._start_image_job(
                user,
                run_id,
                f"повтор референса {reference_id}",
                lambda event: self.generate_references(
                    user, run_id, reference_ids=[reference_id], cancel_event=event
                ),
            )
        if action == "cf_show_visual":
            run_id = action_payload
            return self.show_frames(user, run_id)
        if action == "cf_retry_frame":
            run_id, scene_id = self._parse_run_scene(action_payload)
            return self._start_image_job(
                user,
                run_id,
                f"повтор кадра {scene_id}",
                lambda event: self.generate_frames(
                    user, run_id, scene_ids=[scene_id], cancel_event=event
                ),
            )
        if action == "cf_stop_images":
            return self.stop_image_generation(action_payload)
        if action == "cf_toggle_frame":
            run_id, scene_id = self._parse_run_scene(action_payload)
            return self.toggle_frame(run_id, scene_id)
        if action == "cf_select_all":
            self._production_store().select_all_ready(action_payload)
            return self.frames_detail(action_payload, replace_message=True)
        if action == "cf_image_qa":
            return self.run_image_qa(user, action_payload)
        if action == "cf_video_prompts":
            return self.generate_video_prompts(user, action_payload)
        if action == "cf_video_setup":
            return self.video_setup(action_payload, replace_message=True)
        if action == "cf_video_model":
            run_id, model_key = self._parse_run_token(action_payload)
            return self.video_quality_menu(run_id, model_key, replace_message=True)
        if action == "cf_video_quality":
            run_id, profile_code = self._parse_run_token(action_payload)
            return self.video_duration_menu(run_id, profile_code, replace_message=True)
        if action == "cf_video_duration":
            run_id, selection = self._parse_run_token(action_payload)
            profile_code, raw_duration = selection.rsplit("-", 1)
            return self.select_video_profile(run_id, profile_code, int(raw_duration))
        if action == "cf_show_video_prompts":
            return self.show_video_prompts(action_payload)
        if action == "cf_video_prepare":
            return self.prepare_video_generation(action_payload)
        if action == "cf_video_confirm":
            run_id, approval_id = self._parse_run_token(action_payload)
            return self.confirm_video_generation(user, run_id, approval_id)
        if action == "cf_video_cancel":
            run_id, approval_id = self._parse_run_token(action_payload)
            self._video_job_manager().cancel(run_id, approval_id)
            return BotResponse(
                f"Платная генерация {run_id} отменена. Кадры и prompts сохранены.",
                [[('« К запуску', f'cf_run:{run_id}')]],
            )
        if action == "cf_video_status":
            return self.video_status(user, action_payload)
        if action == "cf_retry_video":
            run_id, scene_id = self._parse_run_scene(action_payload)
            return self.retry_video_submission(user, run_id, scene_id)
        if action == "cf_retry_video_review":
            return self.review_all_video_retries(action_payload)
        if action == "cf_retry_all_video":
            return self.retry_all_video_submissions(user, action_payload)
        if action == "cf_next":
            run_id, expected_stage = self._parse_stage_action(action_payload)
            stale = self._stale_stage_response(run_id, expected_stage)
            if stale:
                return stale
            store = self._content_store()
            run = store.advance(run_id)
            return self.generate_stage(user, run)
        if action == "cf_revise":
            run_id, expected_stage = self._parse_stage_action(action_payload)
            stale = self._stale_stage_response(run_id, expected_stage)
            if stale:
                return stale
            run = self._require_content_run(run_id)
            self.vault.set_input_state(
                user.user_id,
                {"kind": "content_revision", "run_id": run.run_id},
            )
            return BotResponse(
                text=(
                    f"✏️ Доработка {run.run_id}\n\n"
                    f"Напиши, что изменить в этапе «{run.current_stage_spec.title}». "
                    "Я перепишу текущий артефакт целиком."
                ),
                keyboard=[[('Отмена', 'input:cancel')]],
                replace_message=True,
            )
        if action == "cf_retry":
            run_id, expected_stage = self._parse_stage_action(action_payload)
            stale = self._stale_stage_response(run_id, expected_stage)
            if stale:
                return stale
            run = self._require_content_run(run_id)
            return self.generate_stage(user, run)
        if action == "cf_diagnose":
            return self.diagnose_failed_run(action_payload)
        if action == "repair_review":
            return self.review_repair(action_payload)
        if action == "repair_start":
            return self.start_repair(user, action_payload)
        if action == "repair_status":
            return self.repair_status(action_payload, replace_message=True)
        if action == "repair_apply_review":
            return self.review_repair_apply(action_payload)
        if action == "repair_apply":
            return self.start_repair_apply(user, action_payload)
        if action == "repair_discard_review":
            return self.review_repair_discard(action_payload)
        if action == "repair_discard":
            return self.discard_repair(action_payload)
        if action == "cf_cancel":
            run_id = action_payload
            run = self._require_content_run(run_id)
            return BotResponse(
                text=f"Отменить запуск {run.run_id}? Созданные файлы останутся в истории.",
                keyboard=[
                    [('Да, отменить', f'cf_confirm_cancel:{run.run_id}')],
                    [('« К запуску', f'cf_run:{run.run_id}')],
                ],
                replace_message=True,
            )
        if action == "cf_confirm_cancel":
            run_id = action_payload
            run = self._content_store().cancel(run_id)
            return BotResponse(
                text=f"Запуск {run.run_id} отменён. Файлы сохранены.",
                keyboard=[
                    [('📋 Все запуски', 'cf:list')],
                    [('« Главное меню', 'menu:main')],
                ],
                replace_message=True,
            )
        return BotResponse("Неизвестная кнопка.", self.main_menu_keyboard())

    def handle_input_state(
        self,
        user: TelegramUser,
        text: str,
        input_state: dict[str, str],
    ) -> BotResponse:
        kind = input_state.get("kind", "")
        if kind == "storyboard_create":
            self._activate_content_project(user.user_id)
            project = self._storyboard_store().create_guided_project(text)
            self.vault.set_input_state(
                user.user_id,
                {
                    "kind": "storyboard_planning",
                    "project_id": project.project_id,
                },
            )
            self.telegram.send_message(
                user.chat_id,
                f"🧩 Создан {project.project_id}. Сейчас сам разбираю историю и готовлю последовательность панелей.",
            )
            return self._run_guided_storyboard_planning(user, project.project_id)
        if kind == "storyboard_stage":
            project = self._storyboard_store().save_current_stage(
                input_state.get("project_id", ""),
                input_state.get("stage_key", ""),
                text,
            )
            self.vault.clear_input_state(user.user_id)
            return self.storyboard_project_detail(
                project.project_id,
                notice="✅ Этап сохранён.",
            )
        if kind == "storyboard_plan_revision":
            store = self._storyboard_store()
            project_id = input_state.get("project_id", "")
            previous_plan = store.read_plan(project_id)
            project = store.prepare_plan_revision(project_id, text)
            self.vault.set_input_state(
                user.user_id,
                {
                    "kind": "storyboard_planning",
                    "project_id": project.project_id,
                },
            )
            self.telegram.send_message(
                user.chat_id,
                f"✏️ Правка сохранена для {project.project_id}. Перестраиваю весь план панелей.",
            )
            return self._run_guided_storyboard_planning(
                user,
                project.project_id,
                previous_plan=previous_plan,
                revision_request=text,
                notice="✅ План автоматически обновлён; предыдущая версия сохранена.",
            )
        if kind == "storyboard_plan_rejection":
            project = self._storyboard_store().reject_plan(
                input_state.get("project_id", ""),
                text,
            )
            self.vault.clear_input_state(user.user_id)
            return self.storyboard_project_detail(
                project.project_id,
                notice="🛑 План Storyboard отклонён. Phase 1 и Phase 2 не запускались.",
            )
        if kind == "storyboard_sheet_revision":
            project = self._storyboard_store().request_generated_sheet_revision(
                input_state.get("project_id", ""),
                text,
            )
            self.vault.clear_input_state(user.user_id)
            return self.storyboard_project_detail(
                project.project_id,
                notice=(
                    "Правка сохранена. Предыдущий sheet остаётся в истории. "
                    "Новая генерация не начнётся без нового расчёта и подтверждения."
                ),
            )
        if kind == "storyboard_sheet_rejection":
            project = self._storyboard_store().reject_generated_sheet(
                input_state.get("project_id", ""),
                text,
            )
            self.vault.clear_input_state(user.user_id)
            return self.storyboard_project_detail(
                project.project_id,
                notice="Готовый storyboard sheet отклонён. Phase 2 не запускалась.",
            )
        if kind == "storyboard_revision_request":
            project = self._storyboard_store().request_storyboard_revision(
                input_state.get("project_id", ""),
                text,
            )
            self.vault.clear_input_state(user.user_id)
            return self.storyboard_project_detail(
                project.project_id,
                notice=(
                    "✏️ Правки записаны. Предыдущая версия сохранена; "
                    "добавь исправленный storyboard result."
                ),
            )
        if kind == "storyboard_rejection":
            project = self._storyboard_store().reject_storyboard(
                input_state.get("project_id", ""),
                text,
            )
            self.vault.clear_input_state(user.user_id)
            return self.storyboard_project_detail(
                project.project_id,
                notice="🛑 Storyboard отклонён. Phase 2 осталась заблокирована.",
            )
        if kind == "content_idea":
            self.vault.clear_input_state(user.user_id)
            self._activate_content_project(user.user_id)
            run = self._content_store().create_run(text)
            self.telegram.send_message(
                user.chat_id,
                f"🎬 Создан запуск {run.run_id}.\nСейчас готовлю первый этап: концепт и бриф.",
            )
            return self.generate_stage(user, run)
        if kind == "content_revision":
            self.vault.clear_input_state(user.user_id)
            run = self._require_content_run(input_state.get("run_id", ""))
            return self.generate_stage(user, run, revision_request=text)
        if kind == "shared_return":
            self.vault.clear_input_state(user.user_id)
            item = self.shared_content.return_to_partner(
                "owner",
                input_state.get("item_id", ""),
                text,
            )
            return BotResponse(
                text=f"↩️ {item.item_id} возвращён партнёру с комментарием.",
                keyboard=[
                    [('📄 Открыть карточку', f'shared_item:{item.item_id}')],
                    [('« Входящие', 'shared:list')],
                ],
            )

        self.vault.clear_input_state(user.user_id)
        return BotResponse(
            "Форма устарела и была сброшена. Выбери действие заново.",
            self.main_menu_keyboard(),
        )

    def _run_guided_storyboard_planning(
        self,
        user: TelegramUser,
        project_id: str,
        *,
        previous_plan: dict[str, Any] | None = None,
        revision_request: str = "",
        notice: str = "✅ Автоматический план панелей готов.",
    ) -> BotResponse:
        store = self._storyboard_store()
        project = store.get_project(project_id)
        if (
            project is None
            or project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status not in {"planning", "plan_review"}
        ):
            if project is not None:
                return self.storyboard_project_detail(
                    project.project_id,
                    notice="Повторный разбор уже не нужен — показано актуальное состояние.",
                    replace_message=True,
                )
            raise ValueError("Storyboard-проект не найден.")
        idea = store.read_stage(project.project_id, "idea")
        references = store.list_uploaded_references(project.project_id)
        if previous_plan is None:
            try:
                previous_plan = store.read_plan(project.project_id)
            except ValueError:
                previous_plan = None
        if not revision_request and project.status == "planning":
            try:
                pending_revision = store.read_pending_plan_revision(project.project_id)
            except ValueError:
                pending_revision = None
            if pending_revision is not None:
                previous_plan = dict(pending_revision["prior_plan"])
                revision_request = str(pending_revision["request"])
            elif references and previous_plan is not None:
                revision_request = (
                    "Учти все пользовательские референсы и сохрани их visual identity "
                    "во всех связанных панелях."
                )
        self._show_typing(user.chat_id)
        raw_plan = self.llm.chat(
            build_guided_storyboard_plan_messages(
                idea,
                references=references,
                previous_plan=previous_plan,
                revision_request=revision_request,
            )
        )
        project = store.save_generated_plan(project.project_id, raw_plan)
        self.vault.clear_input_state(user.user_id)
        return self.storyboard_project_detail(
            project.project_id,
            notice=notice,
        )

    def handle_storyboard_reference_photo(
        self,
        user: TelegramUser,
        message: dict[str, Any],
    ) -> BotResponse:
        """Save one Telegram photo as a project asset and rebuild the guided plan."""

        state = self.vault.get_input_state(user.user_id) or {}
        project_id = state.get("project_id", "")
        if state.get("kind") != "storyboard_reference" or not project_id:
            return BotResponse("Форма референса устарела. Открой Storyboard заново.")
        photos = message.get("photo") or []
        if not isinstance(photos, list) or not photos:
            return BotResponse("Telegram не передал изображение. Отправь фото ещё раз.")
        largest = photos[-1] if isinstance(photos[-1], dict) else {}
        file_id = str(largest.get("file_id", "")).strip()
        if not file_id:
            return BotResponse("У изображения отсутствует Telegram file_id.")

        store = self._storyboard_store()
        project = store.get_project(project_id)
        if (
            project is None
            or project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status != "plan_review"
        ):
            self.vault.clear_input_state(user.user_id)
            if project is not None:
                return self.storyboard_project_detail(
                    project.project_id,
                    notice="Форма референса устарела — показано актуальное состояние.",
                )
            return BotResponse("Storyboard-проект не найден.")

        incoming = store.root / project.project_id / "references" / ".incoming-reference"
        project = store.begin_reference_update(project.project_id)
        self.vault.set_input_state(
            user.user_id,
            {
                "kind": "storyboard_planning",
                "project_id": project.project_id,
            },
        )
        try:
            self.telegram.download_file(
                file_id,
                incoming,
                max_bytes=GUIDED_STORYBOARD_MAX_REFERENCE_BYTES,
            )
            content = incoming.read_bytes()
        finally:
            incoming.unlink(missing_ok=True)
        caption = str(message.get("caption", "")).strip()
        reference, created = store.save_uploaded_reference(
            project.project_id,
            content,
            caption,
        )
        previous_plan = store.read_plan(project.project_id)
        self.telegram.send_message(
            user.chat_id,
            (
                f"🖼 {reference['reference_id']} сохранён. "
                "Перестраиваю панели с учётом референса."
                if created
                else (
                    f"🖼 {reference['reference_id']} уже был сохранён. "
                    "Повторно файл не создаю; обновляю план."
                )
            ),
        )
        return self._run_guided_storyboard_planning(
            user,
            project.project_id,
            previous_plan=previous_plan,
            revision_request=(
                "Учти все пользовательские референсы и сохрани их visual identity "
                "во всех связанных панелях."
            ),
            notice=(
                f"✅ Референс сохранён как {reference['reference_id']}; "
                "автоматический план обновлён."
            ),
        )

    def shared_inbox(self, replace_message: bool = False) -> BotResponse:
        items = self.shared_content.list_items(
            statuses={"handoff_requested", "accepted", "in_production", "ready"},
            item_kinds={"material"},
            limit=20,
        )
        if not items:
            return BotResponse(
                "📥 Общая очередь пока пуста.\n\n"
                "партнёр может добавить первые ссылки через /reference в своём боте.",
                keyboard=[[('« Главное меню', 'menu:main')]],
                replace_message=replace_message,
            )
        waiting = sum(item.status == "handoff_requested" for item in items)
        production = sum(
            item.status in {"accepted", "in_production"} for item in items
        )
        lines = [
            "📥 Материалы партнёра",
            f"Ожидают решения: {waiting} · в производстве: {production}",
            "",
        ]
        keyboard: InlineKeyboard = []
        for item in items:
            lines.append(
                f"{item.item_id} · {item.status_label}\n{self._shared_preview(item)}"
            )
            keyboard.append(
                [(f"{item.item_id} · {item.status_label}", f"shared_item:{item.item_id}")]
            )
        keyboard.append([('« Главное меню', 'menu:main')])
        return BotResponse("\n\n".join(lines), keyboard, replace_message)

    def ideas_inbox(self, replace_message: bool = False) -> BotResponse:
        """Show evidence-backed radar ideas from production and tester storage."""

        statuses = {"handoff_requested", "accepted", "in_production", "ready"}
        entries: list[tuple[str, SharedContentItem]] = [
            ("prod", item)
            for item in self.shared_content.list_items(
                statuses=statuses,
                item_kinds={"production_idea"},
                limit=50,
            )
        ]
        if self.test_shared_content is not self.shared_content:
            entries.extend(
                ("test", item)
                for item in self.test_shared_content.list_items(
                    statuses=statuses,
                    item_kinds={"production_idea"},
                    limit=50,
                )
            )
        entries = self._deduplicate_idea_entries(entries)[:30]
        if not entries:
            guidance = (
                "Запусти «Радар» в этом боте. После сбора каналов сюда придут "
                "идея, реальные ссылки, метрики и производственный сценарий."
                if self.radar is not None
                else (
                    "Запусти «Радар идей» в боте партнёра. После сбора каналов сюда "
                    "придут идея, реальные ссылки, метрики и производственный сценарий."
                )
            )
            keyboard: InlineKeyboard = []
            if self.radar is not None:
                keyboard.append([('📡 Открыть Радар', 'radar:home')])
            keyboard.extend(
                [
                    [('📥 Обычные материалы', 'shared:list')],
                    [('« Главное меню', 'menu:main')],
                ]
            )
            return BotResponse(
                f"💡 Идей радара пока нет.\n\n{guidance}",
                keyboard=keyboard,
                replace_message=replace_message,
            )

        lines = [
            "💡 Идеи радара",
            "Каждая карточка привязана к реальным роликам и метрикам.",
            "",
        ]
        keyboard: InlineKeyboard = []
        for origin, item in entries:
            metadata = item.metadata
            analytics = metadata.get("analytics", {})
            origin_label = "ТЕСТ" if origin == "test" else "РАБОЧАЯ"
            evidence_count = int(analytics.get("evidence_count", 0) or 0)
            total_views = int(analytics.get("total_views", 0) or 0)
            title = str(metadata.get("idea", "")).strip() or item.title or item.item_id
            lines.append(
                f"[{origin_label}] {title}\n"
                f"{evidence_count} источн. · {self._compact_metric(total_views)} просмотров · "
                f"{item.status_label}"
            )
            keyboard.append(
                [
                    (
                        f"{'🧪' if origin == 'test' else '💡'} {title[:42]}",
                        f"idea_item:{origin}:{item.item_id}",
                    )
                ]
            )
        keyboard.extend(
            [
                [('📥 Обычные материалы', 'shared:list')],
                [('« Главное меню', 'menu:main')],
            ]
        )
        return BotResponse("\n\n".join(lines), keyboard, replace_message)

    @staticmethod
    def _deduplicate_idea_entries(
        entries: list[tuple[str, SharedContentItem]],
    ) -> list[tuple[str, SharedContentItem]]:
        """Hide historical duplicates without deleting their audit records."""

        retained: list[tuple[str, SharedContentItem]] = []
        retained_ideas: list[str] = []
        consumed_sources: set[str] = set()
        for entry in sorted(
            entries,
            key=lambda value: (value[1].created_at, value[1].item_id),
        ):
            metadata = entry[1].metadata
            idea = str(metadata.get("idea", "")).strip()
            if idea and any(
                idea_similarity(idea, existing) >= IDEA_SIMILARITY_THRESHOLD
                for existing in retained_ideas
            ):
                continue

            evidence = metadata.get("evidence", [])
            source_keys: set[str] = set()
            if isinstance(evidence, list):
                for raw in evidence:
                    if not isinstance(raw, dict):
                        continue
                    platform = str(raw.get("platform", "")).strip().lower()
                    external_id = str(raw.get("external_id", "")).strip()
                    source_url = str(
                        raw.get("url", raw.get("source_url", ""))
                    ).strip()
                    if not platform or (not external_id and not source_url):
                        continue
                    try:
                        duration_seconds = int(
                            raw.get("duration_seconds", 0) or 0
                        )
                    except (TypeError, ValueError):
                        duration_seconds = 0
                    reusable_long_video = (
                        platform == "youtube"
                        and duration_seconds >= LONG_FORM_YOUTUBE_SECONDS
                    )
                    if reusable_long_video:
                        continue
                    source_keys.add(
                        str(raw.get("source_key", "")).strip()
                        or canonical_source_key(
                            platform,
                            external_id,
                            source_url,
                        )
                    )
            if source_keys & consumed_sources:
                continue
            retained.append(entry)
            if idea:
                retained_ideas.append(idea)
            consumed_sources.update(source_keys)
        retained.sort(
            key=lambda value: (value[1].updated_at, value[1].item_id),
            reverse=True,
        )
        return retained

    def idea_item(self, locator: str, replace_message: bool = False) -> BotResponse:
        origin, item, store = self._require_idea(locator)
        item = self._sync_shared_item_from_store(store, item)
        metadata = item.metadata
        analytics = metadata.get("analytics", {})
        evidence = metadata.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        idea = str(metadata.get("idea", "")).strip() or item.title
        content_format = {
            "ai": "AI-видео",
            "hybrid": "Смешанный",
            "live": "Живой",
        }.get(str(metadata.get("format", "")).lower(), "Не указан")
        lines = [
            f"💡 {idea or item.item_id}",
            f"Режим: {'тест владельца' if origin == 'test' else 'рабочая идея'}",
            f"Статус: {item.status_label}",
            f"Формат: {content_format}",
            "",
            "Почему стоит сделать:",
            str(metadata.get("reason", "")).strip() or "Обоснование не сохранено.",
            "",
            "Аналитика:",
            f"- роликов-доказательств: {int(analytics.get('evidence_count', 0) or 0)}",
            f"- опубликованы за 14 дней: {int(analytics.get('recent_evidence_count_14d', 0) or 0)}",
            f"- суммарные просмотры: {self._compact_metric(int(analytics.get('total_views', 0) or 0))}",
            f"- суммарные лайки: {self._compact_metric(int(analytics.get('total_likes', 0) or 0))}",
            f"- скорость: {self._compact_metric(int(float(analytics.get('combined_views_per_day', 0) or 0)))} просмотров/день",
            "",
            "Источники:",
        ]
        if (
            str(metadata.get("production_target", "")).strip()
            == "ai_video_content_factory"
            and metadata.get("requires_live_shoot") is False
        ):
            lines.insert(
                4,
                "Получатель: AI-video content factory · живая съёмка не требуется",
            )
        for index, raw in enumerate(evidence[:5], start=1):
            if not isinstance(raw, dict):
                continue
            lines.extend(
                [
                    f"{index}. {raw.get('creator') or 'Автор не указан'} · "
                    f"{raw.get('title') or 'Без названия'}",
                    f"   {self._compact_metric(int(raw.get('views', 0) or 0))} просмотров · "
                    f"{self._compact_metric(int(raw.get('likes', 0) or 0))} лайков · "
                    f"{float(raw.get('age_days', 0) or 0):.1f} дн.",
                    f"   {raw.get('url', '')}",
                ]
            )
        keyboard: InlineKeyboard = []
        if str(metadata.get("script", "")).strip():
            keyboard.append([('📝 Открыть сценарий', f'idea_script:{origin}:{item.item_id}')])
        if item.status in {"handoff_requested", "accepted"} and not item.linked_run_id:
            label = (
                "🧪 Создать тестовый запуск"
                if origin == "test"
                else "✅ Принять в производство"
            )
            keyboard.append([(label, f'idea_accept:{origin}:{item.item_id}')])
        if item.linked_run_id:
            keyboard.append([('🎬 Открыть запуск', f'cf_run:{item.linked_run_id}')])
        keyboard.extend(
            [
                [('« К идеям', 'ideas:list')],
                [('« Главное меню', 'menu:main')],
            ]
        )
        return BotResponse("\n".join(lines), keyboard, replace_message)

    def idea_script(self, locator: str) -> BotResponse:
        origin, item, _ = self._require_idea(locator)
        script = str(item.metadata.get("script", "")).strip()
        if not script:
            script = "Сценарий в этой карточке не сохранён."
        return BotResponse(
            text=f"📝 Сценарий · {item.item_id}\n\n{script}",
            keyboard=[[('« К идее', f'idea_item:{origin}:{item.item_id}')]],
            render_markdown=True,
        )

    def accept_idea(self, user: TelegramUser, locator: str) -> BotResponse:
        origin, item, store = self._require_idea(locator)
        if item.linked_run_id:
            return BotResponse(
                f"✅ Идея уже связана с запуском {item.linked_run_id}.",
                keyboard=[
                    [('🎬 Открыть запуск', f'cf_run:{item.linked_run_id}')],
                    [('« К идее', f'idea_item:{origin}:{item.item_id}')],
                ],
            )
        if item.status == "handoff_requested":
            item = store.accept("owner", item.item_id)
        if item.status != "accepted":
            return self.idea_item(f"{origin}:{item.item_id}", replace_message=True)

        self._activate_content_project(user.user_id)
        marker = f"[Идея радара: {origin}:{item.item_id}]"
        run = next(
            (
                candidate
                for candidate in self._content_store().list_runs(limit=1000)
                if marker in candidate.idea
            ),
            None,
        )
        if run is None:
            metadata = item.metadata
            evidence_lines = [
                f"- {raw.get('url', '')}"
                for raw in metadata.get("evidence", [])
                if isinstance(raw, dict) and str(raw.get("url", "")).strip()
            ]
            idea_parts = [marker, item.source_text]
            if evidence_lines:
                idea_parts.append(
                    "Реальные источники радара:\n" + "\n".join(evidence_lines)
                )
            run = self._content_store().create_run("\n\n".join(idea_parts))
        item = store.link_run("owner", item.item_id, run.run_id)
        return BotResponse(
            text=(
                f"✅ Идея {item.item_id} принята.\n"
                f"Создан запуск: {run.run_id}\n\n"
                "Ссылки, метрики и сценарий добавлены во входные данные. "
                "Платная генерация не запускалась."
            ),
            keyboard=[
                [('🎬 Открыть запуск', f'cf_run:{run.run_id}')],
                [('« К идеям', 'ideas:list')],
            ],
        )

    def shared_item(self, item_id: str, replace_message: bool = False) -> BotResponse:
        item = self._sync_shared_item(item_id)
        lines = [
            f"📄 {item.item_id}",
            f"Статус: {item.status_label}",
            f"Тип: {item.source_type}",
            f"Создал: {item.created_by_role}",
        ]
        if item.source_url:
            lines.append(f"Ссылка: {item.source_url}")
        if item.media_path:
            lines.append("Исходный файл: сохранён в общем хранилище")
        lines.extend(["", item.source_text[:1800]])
        if item.notes:
            lines.extend(["", "Комментарии:", item.notes[-1200:]])
        if item.linked_run_id:
            lines.extend(["", f"Запуск контент-завода: {item.linked_run_id}"])

        keyboard: InlineKeyboard = []
        if item.status == "handoff_requested":
            keyboard.extend(
                [
                    [('✅ Принять в производство', f'shared_accept:{item.item_id}')],
                    [('↩️ Вернуть с комментарием', f'shared_return:{item.item_id}')],
                    [('Отклонить', f'shared_reject:{item.item_id}')],
                ]
            )
        elif item.status == "accepted" and not item.linked_run_id:
            keyboard.extend(
                [
                    [('▶️ Завершить создание запуска', f'shared_accept:{item.item_id}')],
                    [('↩️ Вернуть с комментарием', f'shared_return:{item.item_id}')],
                    [('Отклонить', f'shared_reject:{item.item_id}')],
                ]
            )
        if item.linked_run_id:
            keyboard.append(
                [('🎬 Открыть запуск', f'cf_run:{item.linked_run_id}')]
            )
        keyboard.extend(
            [
                [('« Входящие', 'shared:list')],
                [('« Главное меню', 'menu:main')],
            ]
        )
        return BotResponse("\n".join(lines), keyboard, replace_message)

    def accept_shared_item(self, user: TelegramUser, item_id: str) -> BotResponse:
        item = self.shared_content.require(item_id)
        if item.linked_run_id:
            return BotResponse(
                f"✅ {item.item_id} уже связан с запуском {item.linked_run_id}.",
                keyboard=[
                    [('🎬 Открыть запуск', f'cf_run:{item.linked_run_id}')],
                    [('« Входящие', 'shared:list')],
                ],
            )
        if item.status == "handoff_requested":
            item = self.shared_content.accept("owner", item.item_id)
        if item.status != "accepted":
            return self.shared_item(item.item_id, replace_message=True)

        self._activate_content_project(user.user_id)
        marker = f"[Общий материал: {item.item_id}]"
        run = next(
            (
                candidate
                for candidate in self._content_store().list_runs(limit=1000)
                if marker in candidate.idea
            ),
            None,
        )
        if run is None:
            idea_parts = [marker, item.source_text]
            if item.source_url and item.source_url not in item.source_text:
                idea_parts.append(f"Источник: {item.source_url}")
            if item.notes:
                idea_parts.append(f"Комментарии команды:\n{item.notes}")
            run = self._content_store().create_run("\n\n".join(idea_parts))
        item = self.shared_content.link_run("owner", item.item_id, run.run_id)
        return BotResponse(
            text=(
                f"✅ {item.item_id} принят в производство.\n"
                f"Создан запуск: {run.run_id}\n\n"
                "Платные операции и генерация не запускались. Открой запуск и начни с брифа."
            ),
            keyboard=[
                [('🎬 Открыть запуск', f'cf_run:{run.run_id}')],
                [('« Входящие', 'shared:list')],
            ],
        )

    def reject_shared_item(self, item_id: str) -> BotResponse:
        item = self.shared_content.reject("owner", item_id)
        return BotResponse(
            f"Материал {item.item_id} отклонён. Он не удалён и остаётся в истории.",
            keyboard=[
                [('📄 Открыть карточку', f'shared_item:{item.item_id}')],
                [('« Входящие', 'shared:list')],
            ],
        )

    def _sync_shared_item(self, item_id: str) -> SharedContentItem:
        item = self.shared_content.require(item_id)
        if item.status != "in_production" or not item.linked_run_id:
            return item
        run = self._content_store().get_run(item.linked_run_id)
        if run and run.status == "ready_for_production":
            return self.shared_content.mark_ready(item.item_id, run.run_id)
        return item

    def _sync_shared_item_from_store(
        self,
        store: SharedContentStore,
        item: SharedContentItem,
    ) -> SharedContentItem:
        if item.status != "in_production" or not item.linked_run_id:
            return item
        run = self._content_store().get_run(item.linked_run_id)
        if run and run.status == "ready_for_production":
            return store.mark_ready(item.item_id, run.run_id)
        return item

    def _require_idea(
        self,
        locator: str,
    ) -> tuple[str, SharedContentItem, SharedContentStore]:
        origin, separator, item_id = locator.partition(":")
        if not separator or origin not in {"prod", "test"}:
            raise ValueError("Некорректная ссылка на идею.")
        store = self.shared_content if origin == "prod" else self.test_shared_content
        item = store.require(item_id)
        if item.item_kind != "production_idea":
            raise ValueError("Карточка не является идеей радара.")
        return origin, item, store

    @staticmethod
    def _compact_metric(value: int) -> str:
        number = max(0, int(value))
        if number >= 1_000_000:
            return f"{number / 1_000_000:.1f}M".replace(".0M", "M")
        if number >= 1_000:
            return f"{number / 1_000:.1f}K".replace(".0K", "K")
        return str(number)

    @staticmethod
    def _shared_preview(item: SharedContentItem) -> str:
        value = item.title or item.source_url or item.source_text
        clean = " ".join(value.split())
        return clean if len(clean) <= 110 else clean[:107].rstrip() + "..."

    def main_menu(self, user: TelegramUser, replace_message: bool = False) -> BotResponse:
        runs = self._content_store().list_runs(limit=100)
        active = sum(run.status not in {"cancelled", "ready_for_production"} for run in runs)
        ready = sum(run.status == "ready_for_production" for run in runs)
        return BotResponse(
            text=(
                "🏭 AI Content Factory\n\n"
                "Управление производством роликов с телефона.\n"
                f"Запусков: {len(runs)} · активных: {active} · пакетов готово: {ready}\n\n"
                "С чего начнём?"
            ),
            keyboard=self.main_menu_keyboard(),
            replace_message=replace_message,
        )

    def storyboard_home(
        self,
        user: TelegramUser,
        replace_message: bool = False,
    ) -> BotResponse:
        projects = self._storyboard_store().list_projects(limit=100)
        active = sum(
            project.status not in {"completed", "rejected"} for project in projects
        )
        return BotResponse(
            text=(
                "🧩 Storyboard\n\n"
                "Идея или сценарий → автоматический разбор истории → последовательный "
                "план панелей → режиссёрский preview.\n\n"
                "Можно отдельно добавить фото-референсы персонажей, локаций, предметов "
                "или стиля. Система сама перестроит план; техническую анкету заполнять "
                "не нужно.\n\n"
                "Результат Phase 1: один общий storyboard sheet с пронумерованными "
                "панелями. Перед его будущей генерацией будут отдельно показаны "
                "provider/model, точный результат и ожидаемая стоимость.\n\n"
                f"Prompt-asset: {STORYBOARD_PROMPT_SOURCE_LABEL}\n"
                f"Статус: {STORYBOARD_PROMPT_SOURCE_STATUS}.\n\n"
                f"Проектов: {len(projects)} · активных: {active}\n"
                "Сейчас работает текстовый planning и локальное сохранение референсов; "
                "image/video providers и платные запросы из Storyboard не запускаются."
            ),
            keyboard=[
                [('➕ Новый Storyboard', 'sb:new')],
                [('📂 Мои Storyboard', 'sb:list')],
                [('ℹ️ Как работает', 'sb:about')],
                [('« Главное меню', 'menu:main')],
            ],
            replace_message=replace_message,
        )

    def start_new_storyboard(
        self,
        user: TelegramUser,
        replace_message: bool = False,
    ) -> BotResponse:
        self._clear_main_input_state_if_present(user.user_id)
        self._activate_content_project(user.user_id)
        self.vault.set_input_state(user.user_id, {"kind": "storyboard_create"})
        return BotResponse(
            text=(
                "🧩 Новый Storyboard\n\n"
                "Пришли одним сообщением идею, логлайн или готовый сценарий. "
                "Можно писать свободно — технические поля не нужны.\n\n"
                "Я автоматически разберу историю, подготовлю последовательность панелей "
                "и покажу preview. После этого можно добавить отдельные фото-референсы, "
                "попросить правку или подтвердить план."
            ),
            keyboard=[[('Отмена', 'input:cancel')]],
            replace_message=replace_message,
        )

    def dispatch_storyboard_callback(
        self,
        user: TelegramUser,
        data: str,
    ) -> BotResponse:
        parts = data.split(":")
        action = parts[1] if len(parts) > 1 else ""
        if action == "new" and len(parts) == 2:
            return self.start_new_storyboard(user, replace_message=True)

        self._clear_main_input_state_if_present(user.user_id)
        if action == "home" and len(parts) == 2:
            return self.storyboard_home(user, replace_message=True)
        if action == "list" and len(parts) == 2:
            return self.storyboard_projects_menu(replace_message=True)
        if action == "about" and len(parts) == 2:
            return self.storyboard_about(replace_message=True)
        if action == "open" and len(parts) == 3:
            return self.storyboard_project_detail(parts[2], replace_message=True)
        if action == "plan_retry" and len(parts) == 3:
            project = self._storyboard_store().get_project(parts[2])
            if (
                project is None
                or project.workflow != GUIDED_STORYBOARD_WORKFLOW
                or project.status != "planning"
            ):
                if project is not None:
                    return self.storyboard_project_detail(
                        project.project_id,
                        notice="Повторный разбор уже не нужен — показано актуальное состояние.",
                        replace_message=True,
                    )
                return BotResponse(
                    "Storyboard-проект не найден.",
                    [[('« К проектам', 'sb:list')]],
                    replace_message=True,
                )
            self.vault.set_input_state(
                user.user_id,
                {
                    "kind": "storyboard_planning",
                    "project_id": project.project_id,
                },
            )
            return self._run_guided_storyboard_planning(user, project.project_id)
        if action == "plan_approve" and len(parts) == 3:
            try:
                project = self._storyboard_store().approve_plan(parts[2])
            except ValueError as exc:
                project = self._storyboard_store().get_project(parts[2])
                if project is not None:
                    return self.storyboard_project_detail(
                        project.project_id,
                        notice=f"Эта кнопка устарела: {exc}",
                        replace_message=True,
                    )
                return BotResponse(
                    str(exc),
                    [[('« К проектам', 'sb:list')]],
                    replace_message=True,
                )
            return self.storyboard_project_detail(
                project.project_id,
                notice=(
                    "✅ План панелей подтверждён. Это не approval готового storyboard: "
                    "Phase 2 остаётся заблокирована."
                ),
                replace_message=True,
            )
        if action == "sheet_prepare" and len(parts) == 3:
            store = self._storyboard_store()
            project = store.get_project(parts[2])
            if (
                project is None
                or project.workflow != GUIDED_STORYBOARD_WORKFLOW
                or project.status != "plan_approved"
            ):
                if project is not None:
                    return self.storyboard_project_detail(
                        project.project_id,
                        notice="Эта кнопка устарела — показано актуальное состояние.",
                        replace_message=True,
                    )
                return BotResponse(
                    "Storyboard-проект не найден.",
                    [[('« К проектам', 'sb:list')]],
                    replace_message=True,
                )
            try:
                quote = self._storyboard_sheet_quote(project.project_id)
                project = store.prepare_sheet_quote(project.project_id, quote)
            except ValueError as exc:
                return self.storyboard_project_detail(
                    project.project_id,
                    notice=f"Генерация пока недоступна: {exc}",
                    replace_message=True,
                )
            return self.storyboard_project_detail(
                project.project_id,
                notice="Условия сохранены. Проверь их перед отдельным подтверждением.",
                replace_message=True,
            )
        if action == "sheet_cancel" and len(parts) == 3:
            try:
                project = self._storyboard_store().cancel_sheet_quote(parts[2])
            except ValueError:
                project = self._storyboard_store().get_project(parts[2])
                if project is None:
                    return BotResponse(
                        "Storyboard-проект не найден.",
                        [[('« К проектам', 'sb:list')]],
                        replace_message=True,
                    )
                return self.storyboard_project_detail(
                    project.project_id,
                    notice="Эта кнопка устарела — показано актуальное состояние.",
                    replace_message=True,
                )
            return self.storyboard_project_detail(
                project.project_id,
                notice="Генерация не запускалась; подтверждение отменено.",
                replace_message=True,
            )
        if action == "sheet_confirm" and len(parts) == 3:
            return self._confirm_storyboard_sheet_generation(
                user,
                parts[2],
                replace_message=True,
            )
        if action == "sheet_show" and len(parts) == 3:
            project = self._storyboard_store().get_project(parts[2])
            if project is None or project.status not in {"sheet_review", "sheet_approved"}:
                if project is not None:
                    return self.storyboard_project_detail(
                        project.project_id,
                        notice="Эта кнопка устарела — показано актуальное состояние.",
                        replace_message=True,
                    )
                return BotResponse(
                    "Storyboard-проект не найден.",
                    [[('« К проектам', 'sb:list')]],
                    replace_message=True,
                )
            try:
                path, result = self._storyboard_store().latest_generated_sheet(
                    project.project_id
                )
                self.telegram.send_photo(
                    user.chat_id,
                    path,
                    caption=(
                        f"🧩 {project.project_id} · {result['version']}\n"
                        + (
                            "Готовый общий storyboard sheet подтверждён."
                            if project.status == "sheet_approved"
                            else "Готовый общий storyboard sheet. Проверь все панели."
                        )
                    ),
                    keyboard=(
                        self._storyboard_sheet_review_keyboard(project.project_id)
                        if project.status == "sheet_review"
                        else [[('« К проекту', f'sb:open:{project.project_id}')]]
                    ),
                )
            except (TelegramApiError, ValueError) as exc:
                return self.storyboard_project_detail(
                    project.project_id,
                    notice=f"Не удалось отправить сохранённый sheet: {exc}",
                    replace_message=True,
                )
            return BotResponse(
                "Готовый storyboard sheet отправлен отдельным изображением.",
                [[('« К проекту', f'sb:open:{project.project_id}')]],
                replace_message=True,
            )
        if action == "sheet_approve" and len(parts) == 3:
            try:
                project = self._storyboard_store().approve_generated_sheet(parts[2])
            except ValueError:
                project = self._storyboard_store().get_project(parts[2])
                if project is None:
                    return BotResponse(
                        "Storyboard-проект не найден.",
                        [[('« К проектам', 'sb:list')]],
                        replace_message=True,
                    )
                return self.storyboard_project_detail(
                    project.project_id,
                    notice="Эта кнопка устарела — показано актуальное состояние.",
                    replace_message=True,
                )
            return self.storyboard_project_detail(
                project.project_id,
                notice=(
                    "Готовый storyboard sheet подтверждён. Phase 1 завершена; "
                    "Phase 2 не запускалась."
                ),
                replace_message=True,
            )
        if action in {"sheet_revise", "sheet_reject"} and len(parts) == 3:
            project = self._storyboard_store().get_project(parts[2])
            if project is None or project.status != "sheet_review":
                if project is not None:
                    return self.storyboard_project_detail(
                        project.project_id,
                        notice="Эта кнопка устарела — показано актуальное состояние.",
                        replace_message=True,
                    )
                return BotResponse(
                    "Storyboard-проект не найден.",
                    [[('« К проектам', 'sb:list')]],
                    replace_message=True,
                )
            revision = action == "sheet_revise"
            self.vault.set_input_state(
                user.user_id,
                {
                    "kind": (
                        "storyboard_sheet_revision"
                        if revision
                        else "storyboard_sheet_rejection"
                    ),
                    "project_id": project.project_id,
                },
            )
            return BotResponse(
                (
                    "Опиши одним сообщением, что исправить в готовом sheet. "
                    "Новая генерация потребует нового расчёта и отдельного подтверждения."
                    if revision
                    else "Напиши причину отклонения готового storyboard sheet."
                ),
                [[('Отмена', 'input:cancel')]],
                replace_message=True,
            )
        if action in {"plan_revise", "plan_reject"} and len(parts) == 3:
            project = self._storyboard_store().get_project(parts[2])
            if (
                project is None
                or project.workflow != GUIDED_STORYBOARD_WORKFLOW
                or project.status != "plan_review"
            ):
                if project is not None:
                    return self.storyboard_project_detail(
                        project.project_id,
                        notice="Эта кнопка устарела — показано актуальное состояние.",
                        replace_message=True,
                    )
                return BotResponse(
                    "Storyboard-проект не найден.",
                    [[('« К проектам', 'sb:list')]],
                    replace_message=True,
                )
            is_revision = action == "plan_revise"
            self.vault.set_input_state(
                user.user_id,
                {
                    "kind": (
                        "storyboard_plan_revision"
                        if is_revision
                        else "storyboard_plan_rejection"
                    ),
                    "project_id": project.project_id,
                },
            )
            return BotResponse(
                text=(
                    "✏️ Опиши одним сообщением, что изменить в истории, порядке или композиции панелей. "
                    "Система сама перестроит весь план; предыдущая версия сохранится."
                    if is_revision
                    else (
                        "🛑 Напиши причину отклонения плана. Изображение не будет "
                        "генерироваться, а Phase 2 останется заблокирована."
                    )
                ),
                keyboard=[[('Отмена', 'input:cancel')]],
                replace_message=True,
            )
        if action == "refs" and len(parts) == 3:
            project = self._storyboard_store().get_project(parts[2])
            if (
                project is None
                or project.workflow != GUIDED_STORYBOARD_WORKFLOW
                or project.status != "plan_review"
            ):
                if project is not None:
                    return self.storyboard_project_detail(
                        project.project_id,
                        notice="Добавлять референсы сейчас нельзя — показано актуальное состояние.",
                        replace_message=True,
                    )
                return BotResponse(
                    "Storyboard-проект не найден.",
                    [[('« К проектам', 'sb:list')]],
                    replace_message=True,
                )
            self.vault.set_input_state(
                user.user_id,
                {
                    "kind": "storyboard_reference",
                    "project_id": project.project_id,
                },
            )
            return BotResponse(
                text=(
                    "🖼 Пришли изображение отдельным сообщением. В подписи напиши, "
                    "что это и что нужно сохранить: персонаж, локация, предмет или стиль."
                ),
                keyboard=[[('Готово без добавления', 'input:cancel')]],
                replace_message=True,
            )
        if action == "fill" and len(parts) == 4:
            project_id, stage_key = parts[2], parts[3]
            project = self._storyboard_store().get_project(project_id)
            stage = STORYBOARD_STAGE_MAP.get(stage_key)
            if project is None or stage is None:
                return BotResponse(
                    "Storyboard-проект или этап не найден.",
                    [[('« К проектам', 'sb:list')]],
                    replace_message=True,
                )
            if project.status != "in_progress" or project.current_stage != stage_key:
                return self.storyboard_project_detail(
                    project.project_id,
                    notice="Эта кнопка устарела — показано актуальное состояние.",
                    replace_message=True,
                )
            self.vault.set_input_state(
                user.user_id,
                {
                    "kind": "storyboard_stage",
                    "project_id": project.project_id,
                    "stage_key": stage.key,
                },
            )
            return BotResponse(
                text=(
                    f"✍️ {stage.title}\n\n{stage.prompt}\n\n"
                    "Пришли материал одним текстовым сообщением. В MVP можно "
                    "вставлять ссылки; загрузку файлов добавим отдельно."
                ),
                keyboard=[[('Отмена', 'input:cancel')]],
                replace_message=True,
            )
        if action in {"revise", "reject"} and len(parts) == 3:
            project = self._storyboard_store().get_project(parts[2])
            if project is None or project.status != "awaiting_storyboard_approval":
                if project is not None:
                    return self.storyboard_project_detail(
                        project.project_id,
                        notice="Эта кнопка устарела — показано актуальное состояние.",
                        replace_message=True,
                    )
                return BotResponse(
                    "Storyboard-проект не найден.",
                    [[('« К проектам', 'sb:list')]],
                    replace_message=True,
                )
            is_revision = action == "revise"
            self.vault.set_input_state(
                user.user_id,
                {
                    "kind": (
                        "storyboard_revision_request"
                        if is_revision
                        else "storyboard_rejection"
                    ),
                    "project_id": project.project_id,
                },
            )
            return BotResponse(
                text=(
                    "✏️ Напиши одним сообщением, что нужно исправить. "
                    "Текущая версия будет сохранена в истории."
                    if is_revision
                    else (
                        "🛑 Напиши причину отклонения. Phase 2 не будет "
                        "разблокирована."
                    )
                ),
                keyboard=[[('Отмена', 'input:cancel')]],
                replace_message=True,
            )
        if action == "approve" and len(parts) == 3:
            try:
                project = self._storyboard_store().approve_storyboard(parts[2])
            except ValueError as exc:
                return BotResponse(
                    str(exc),
                    [[('« К проектам', 'sb:list')]],
                    replace_message=True,
                )
            return self.storyboard_project_detail(
                project.project_id,
                notice=(
                    "✅ Storyboard подтверждён. Phase 2 разблокирована; "
                    "теперь можно заполнить Cinematic video prompt."
                ),
                replace_message=True,
            )
        if action == "complete" and len(parts) == 3:
            try:
                project = self._storyboard_store().complete_project(parts[2])
            except ValueError as exc:
                return BotResponse(
                    str(exc),
                    [[('« К проектам', 'sb:list')]],
                    replace_message=True,
                )
            return self.storyboard_project_detail(
                project.project_id,
                notice="✅ Эксперимент завершён. Все текстовые этапы сохранены.",
                replace_message=True,
            )
        return BotResponse(
            "Неизвестная кнопка Storyboard.",
            [[('« Storyboard', 'sb:home')]],
            replace_message=True,
        )

    def storyboard_projects_menu(self, replace_message: bool = False) -> BotResponse:
        projects = self._storyboard_store().list_projects(limit=20)
        if not projects:
            return BotResponse(
                "📂 Storyboard-проектов пока нет.",
                [
                    [('➕ Новый Storyboard', 'sb:new')],
                    [('« Storyboard', 'sb:home')],
                ],
                replace_message=replace_message,
            )
        lines = ["📂 Мои Storyboard", ""]
        keyboard: InlineKeyboard = []
        for project in projects:
            status = self._storyboard_status_label(project.status)
            lines.append(f"{project.project_id} · {status}\n{project.title}")
            keyboard.append(
                [(f"{project.project_id} · {status}", f"sb:open:{project.project_id}")]
            )
        keyboard.extend(
            [
                [('➕ Новый Storyboard', 'sb:new')],
                [('« Storyboard', 'sb:home')],
            ]
        )
        return BotResponse("\n\n".join(lines), keyboard, replace_message)

    def storyboard_project_detail(
        self,
        project_id: str,
        notice: str = "",
        replace_message: bool = False,
    ) -> BotResponse:
        project = self._storyboard_store().get_project(project_id)
        if project is None:
            return BotResponse(
                "Storyboard-проект не найден.",
                [[('« К проектам', 'sb:list')]],
                replace_message=replace_message,
            )
        if project.workflow == GUIDED_STORYBOARD_WORKFLOW:
            return self._guided_storyboard_project_detail(
                project,
                notice=notice,
                replace_message=replace_message,
            )
        lines = [
            f"🧩 {project.project_id}",
            project.title,
            "",
            f"Статус: {self._storyboard_status_label(project.status)}",
            f"Прогресс: {len(project.completed_stages)}/{len(STORYBOARD_STAGES)}",
            f"Шаблон: {STORYBOARD_PROMPT_SOURCE_LABEL}",
            "",
            "Этапы:",
        ]
        for stage in STORYBOARD_STAGES:
            if stage.key in project.completed_stages:
                marker = "✅"
            elif stage.key == project.current_stage:
                marker = "▶️"
            else:
                marker = "▫️"
            lines.append(f"{marker} {stage.title}")

        keyboard: InlineKeyboard = []
        if project.status == "in_progress":
            stage = project.current_stage_spec
            lines.extend(["", f"Следующий этап: {stage.title}"])
            keyboard.append(
                [(f"✍️ Заполнить: {stage.title}", f"sb:fill:{project.project_id}:{stage.key}")]
            )
        elif project.status == "awaiting_storyboard_approval":
            lines.extend(
                [
                    "",
                    "Storyboard sheet ждёт твоего подтверждения.",
                    "Phase 2 заблокирована: эта кнопка ничего не генерирует и "
                    "только записывает approval.",
                ]
            )
            keyboard.append(
                [('✅ Подтвердить storyboard', f'sb:approve:{project.project_id}')]
            )
            keyboard.append(
                [('✏️ Нужны правки', f'sb:revise:{project.project_id}')]
            )
            keyboard.append(
                [('🛑 Отклонить storyboard', f'sb:reject:{project.project_id}')]
            )
        elif project.status == "review_ready":
            lines.extend(["", "Проект готов к завершению после финального разбора."])
            keyboard.append(
                [('✅ Завершить эксперимент', f'sb:complete:{project.project_id}')]
            )
        elif project.status == "completed":
            lines.extend(["", "Эксперимент завершён; материалы сохранены."])
        else:
            lines.extend(
                [
                    "",
                    "Storyboard отклонён; Phase 2 не разблокирована.",
                ]
            )

        keyboard.extend(
            [
                [('« К проектам', 'sb:list')],
                [('« Storyboard', 'sb:home')],
            ]
        )
        text = "\n".join(lines)
        if notice:
            text = f"{notice}\n\n{text}"
        return BotResponse(text, keyboard, replace_message)

    @staticmethod
    def _storyboard_sheet_review_keyboard(project_id: str) -> InlineKeyboard:
        return [
            [('✅ Подтвердить готовый sheet', f'sb:sheet_approve:{project_id}')],
            [('✏️ Нужны правки', f'sb:sheet_revise:{project_id}')],
            [('🛑 Отклонить sheet', f'sb:sheet_reject:{project_id}')],
        ]

    def _storyboard_sheet_materials(
        self,
        project_id: str,
    ) -> tuple[str, tuple[ImageReference, ...], str]:
        """Capture the exact prompt/settings/reference digests bound to confirmation."""

        store = self._storyboard_store()
        plan = store.read_plan(project_id)
        if str(plan["aspect_ratio"]).strip() != "16:9":
            raise ValueError(
                "Phase 1 sheet generation сейчас поддерживает только план 16:9; "
                "он размещается в центре provider canvas 1536x1024"
            )
        panel_lines = []
        for panel in plan["panels"]:
            panel_lines.append(
                " | ".join(
                    [
                        str(panel["panel_id"]),
                        str(panel["timecode"]),
                        str(panel["shot_type"]),
                        str(panel["visual"]),
                        str(panel["action"]),
                        str(panel["camera"]),
                        str(panel["caption"]),
                        "references=" + ",".join(panel["reference_ids"]),
                    ]
                )
            )
        reference_lines = [
            " | ".join(
                [
                    str(item["reference_id"]),
                    str(item["kind"]),
                    str(item["label"]),
                    str(item["description"]),
                    str(item["usage"]),
                ]
            )
            for item in plan["references"]
        ]
        revision_request = store.read_sheet_revision_request(project_id)
        prompt = (
            str(plan["sheet_prompt"]).strip()
            + "\n\nHARD OUTPUT CONTRACT:\n"
            + f"- Deliver one single {plan['aspect_ratio']} storyboard sheet composition.\n"
            + "- Create exactly one raster image, never separate panel images.\n"
            + f"- Use exactly {len(plan['panels'])} numbered panels in this order.\n"
            + f"- Storyboard content aspect ratio: {plan['aspect_ratio']}.\n"
            + "- The provider raster canvas is 1536x1024. Keep the complete storyboard "
            + "inside the centered 1536x864 (exact 16:9) safe area; keep the 80 px top "
            + "and 80 px bottom margins neutral and outside every panel, caption and gutter.\n"
            + (
                f"- Apply this director revision: {revision_request}\n"
                if revision_request
                else ""
            )
            + "- Preserve character identity, wardrobe, environment geometry, props, "
            + "lighting and story state across related panels.\n\n"
            + "REFERENCE REQUIREMENTS:\n"
            + ("\n".join(reference_lines) or "None")
            + "\n\nORDERED PANELS:\n"
            + "\n".join(panel_lines)
        )
        if len(prompt) > 23000:
            raise ValueError(
                "План слишком объёмный для одного безопасного image-запроса; сократи описания."
            )

        uploaded = store.list_uploaded_references(project_id)
        if len(uploaded) > 5:
            raise ValueError(
                "Codex Image принимает не более 5 загруженных референсов за один запрос."
            )
        project_root = (store.root / project_id).resolve()
        image_references: list[ImageReference] = []
        digest_references: list[dict[str, Any]] = []
        for item in uploaded:
            reference_id = str(item.get("reference_id", "")).strip()
            filename = str(item.get("filename", "")).strip()
            expected_digest = str(item.get("sha256", "")).strip()
            if (
                not reference_id
                or not filename
                or Path(filename).name != filename
                or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
            ):
                raise ValueError("Manifest загруженных референсов повреждён.")
            path = (project_root / "references" / filename).resolve()
            if project_root not in path.parents:
                raise ValueError("Путь референса выходит за границы Storyboard-проекта.")
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"Референс {reference_id} недоступен.") from exc
            actual_digest = hashlib.sha256(content).hexdigest()
            if actual_digest != expected_digest or len(content) != int(item.get("bytes", -1)):
                raise ValueError(f"Референс {reference_id} изменился или повреждён.")
            role = str(item.get("description") or item.get("kind") or "visual reference")
            image_references.append(ImageReference(reference_id, role[:500], path))
            digest_references.append(
                {
                    "reference_id": reference_id,
                    "sha256": actual_digest,
                    "bytes": len(content),
                }
            )

        binding = {
            "provider": self.settings.image_provider,
            "agent_model": self.settings.codex_chat_model,
            "image_backend": "selected-and-not-disclosed-by-built-in-tool",
            "size": GUIDED_STORYBOARD_SHEET_SIZE,
            "prompt": prompt,
            "references": digest_references,
        }
        input_sha256 = hashlib.sha256(
            json.dumps(
                binding,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return prompt, tuple(image_references), input_sha256

    def _storyboard_sheet_quote(
        self,
        project_id: str,
        *,
        input_sha256: str = "",
    ) -> dict[str, Any]:
        if self.settings.image_provider != "codex":
            raise ValueError(
                "для текущего image provider нет проверенного точного расчёта стоимости"
            )
        if not self.image_client.is_configured:
            raise ValueError("Codex Image сейчас не подключён")
        plan = self._storyboard_store().read_plan(project_id)
        if not input_sha256:
            _, _, input_sha256 = self._storyboard_sheet_materials(project_id)
        return {
            "provider_key": "codex",
            "provider_label": "Codex Image (текущая subscription OAuth)",
            "model": (
                f"Codex agent {self.settings.codex_chat_model}; image backend "
                "выбирается built-in tool и точным именем не раскрывается"
            ),
            "size": GUIDED_STORYBOARD_SHEET_SIZE,
            "quality": "управляется built-in image tool; отдельное значение не раскрывается",
            "cost_display": "0 ₽ дополнительного API-списания",
            "billing_note": (
                "Используется лимит текущей Codex-подписки; отдельный OPENAI_API_KEY "
                "и API-биллинг не используются. Внутренний расход subscription-квоты "
                "Codex CLI в деньгах не показывает."
            ),
            "result_display": (
                f"ровно один raster 1536x1024: центрированная storyboard-зона "
                f"1536x864 (точно 16:9), поля 80 px сверху/снизу, "
                f"{len(plan['panels'])} пронумерованными панелями; не отдельные изображения"
            ),
            "expected_requests": 1,
            "input_sha256": input_sha256,
        }

    def _confirm_storyboard_sheet_generation(
        self,
        user: TelegramUser,
        project_id: str,
        *,
        replace_message: bool,
    ) -> BotResponse:
        store = self._storyboard_store()
        with self._storyboard_sheet_generation_lock:
            project = store.get_project(project_id)
            if (
                project is None
                or project.workflow != GUIDED_STORYBOARD_WORKFLOW
                or project.status != "sheet_awaiting_confirmation"
                or project.pending_operation != "sheet_confirmation"
            ):
                if project is not None:
                    return self.storyboard_project_detail(
                        project.project_id,
                        notice="Эта кнопка устарела — показано актуальное состояние.",
                        replace_message=replace_message,
                    )
                return BotResponse(
                    "Storyboard-проект не найден.",
                    [[('« К проектам', 'sb:list')]],
                    replace_message=replace_message,
                )
            try:
                prompt, references, input_sha256 = self._storyboard_sheet_materials(
                    project.project_id
                )
                current_quote = self._storyboard_sheet_quote(
                    project.project_id,
                    input_sha256=input_sha256,
                )
                store.begin_sheet_generation(project.project_id, current_quote)
            except ValueError as exc:
                return self.storyboard_project_detail(
                    project.project_id,
                    notice=f"Генерация не запущена: {exc}",
                    replace_message=replace_message,
                )

            try:
                image = self.image_client.generate(
                    prompt,
                    references=references,
                    size=GUIDED_STORYBOARD_SHEET_SIZE,
                )
                project, _, version = store.save_generated_sheet(
                    project.project_id,
                    image.content,
                )
                verified_path, result = store.latest_generated_sheet(project.project_id)
            except (ImageGenerationError, ValueError, OSError) as exc:
                try:
                    store.mark_sheet_generation_failed(project.project_id, str(exc))
                except (ValueError, OSError):
                    pass
                explanation = (
                    self._friendly_image_error(exc)
                    if isinstance(exc, ImageGenerationError)
                    else f"Локальная проверка результата не завершилась: {exc}"
                )
                return self.storyboard_project_detail(
                    project.project_id,
                    notice=(
                        f"Результат генерации требует ручной сверки: {explanation} "
                        "Автоматический повтор заблокирован, чтобы исключить повторное списание."
                    ),
                    replace_message=replace_message,
                )

            caption = (
                f"🧩 {project.project_id} · {version}\n"
                "Готов один общий storyboard sheet. Проверь панели и выбери решение ниже."
            )
            try:
                self.telegram.send_photo(
                    user.chat_id,
                    verified_path,
                    caption=caption,
                    keyboard=self._storyboard_sheet_review_keyboard(project.project_id),
                )
            except TelegramApiError as exc:
                return self.storyboard_project_detail(
                    project.project_id,
                    notice=(
                        "Sheet сгенерирован, проверен и сохранён, но Telegram не принял "
                        f"файл: {exc}. Повторная генерация не нужна."
                    ),
                    replace_message=replace_message,
                )
            return BotResponse(
                text=(
                    f"Готово: {result['version']} сохранён и отправлен. "
                    "Phase 2 остаётся закрыта до approval готового sheet."
                ),
                keyboard=[[('« К проекту', f'sb:open:{project.project_id}')]],
                replace_message=replace_message,
            )

    def _guided_storyboard_project_detail(
        self,
        project: StoryboardProject,
        *,
        notice: str = "",
        replace_message: bool = False,
    ) -> BotResponse:
        """Render the automatic workflow without exposing the legacy stage form."""

        try:
            plan = self._storyboard_store().read_plan(project.project_id)
        except ValueError:
            plan = None

        lines = [
            f"🧩 {project.project_id}",
            project.title,
            "",
            f"Статус: {self._storyboard_status_label(project.status)}",
        ]
        keyboard: InlineKeyboard = []
        if plan is None:
            lines.extend(
                [
                    "",
                    "Идея сохранена. Автоматический план ещё не готов.",
                    "Phase 1 и Phase 2 не запускались.",
                ]
            )
        else:
            panels = list(plan["panels"])
            references = list(plan["references"])
            layout = dict(plan["layout"])
            lines.extend(
                [
                    "",
                    str(plan["logline"]),
                    "",
                    (
                        f"План: {len(panels)} пан. · {plan['duration_seconds']} сек. · "
                        f"{plan['aspect_ratio']} · сетка {layout['columns']}×{layout['rows']}"
                    ),
                    f"Референсы: {len(references)}",
                    "Результат Phase 1: один общий storyboard sheet с пронумерованными панелями.",
                    "",
                    "Последовательность:",
                ]
            )
            for panel in panels:
                caption = str(panel["caption"])
                visual = str(panel["visual"])
                action = str(panel["action"])
                lines.append(
                    f"{panel['panel_id']} · {panel['timecode']} · {caption}\n"
                    f"{visual} {action}"
                )

        if project.status == "plan_review" and plan is not None:
            lines.extend(
                [
                    "",
                    "Проверь логику и последовательность. Изображение пока не генерировалось.",
                    "Подтверждение плана не открывает Phase 2: сначала будет отдельный расчёт стоимости общего sheet.",
                ]
            )
            keyboard.extend(
                [
                    [('✅ План подходит', f'sb:plan_approve:{project.project_id}')],
                    [('🖼 Добавить референсы', f'sb:refs:{project.project_id}')],
                    [('✏️ Изменить план', f'sb:plan_revise:{project.project_id}')],
                    [('🛑 Отклонить', f'sb:plan_reject:{project.project_id}')],
                ]
            )
        elif project.status == "plan_approved":
            lines.extend(
                [
                    "",
                    "✅ План панелей подтверждён.",
                    "Следующий шаг только подготовит точные условия одной генерации: "
                    "provider, model, стоимость и ожидаемый результат.",
                    "Image provider не вызывается до отдельного подтверждения.",
                    "Платная генерация и Phase 2 пока заблокированы.",
                ]
            )
            keyboard.append(
                [('🧩 Создать storyboard sheet', f'sb:sheet_prepare:{project.project_id}')]
            )
        elif project.status == "sheet_awaiting_confirmation":
            try:
                quote = self._storyboard_store().read_sheet_quote(project.project_id)
            except ValueError as exc:
                lines.extend(
                    [
                        "",
                        f"⚠️ Сохранённые условия недоступны: {exc}",
                        "Генерация заблокирована; image provider не вызывался.",
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        "💳 Условия одной генерации",
                        f"Provider: {quote['provider_label']}",
                        f"Model: {quote['model']}",
                        f"Размер: {quote['size']}",
                        f"Качество: {quote['quality']}",
                        f"Стоимость: {quote['cost_display']}",
                        f"Биллинг: {quote['billing_note']}",
                        f"Точный результат: {quote['result_display']}",
                        "Запросов к image provider после подтверждения: ровно 1.",
                        "",
                        "Подтверждение плана выше не является подтверждением этой генерации.",
                    ]
                )
                keyboard.extend(
                    [
                        [
                            (
                                '✅ Подтверждаю одну генерацию',
                                f'sb:sheet_confirm:{project.project_id}',
                            )
                        ],
                        [('Отмена', f'sb:sheet_cancel:{project.project_id}')],
                    ]
                )
        elif project.status == "sheet_generating":
            lines.extend(
                [
                    "",
                    "⏳ Подтверждение уже принято, запрос зарегистрирован.",
                    "Повторный запуск заблокирован. После restart это состояние "
                    "нельзя автоматически повторить — сначала нужна ручная сверка.",
                ]
            )
        elif project.status == "sheet_reconciliation_required":
            lines.extend(
                [
                    "",
                    "⚠️ Результат внешнего вызова требует ручной сверки.",
                    "Автоматический повтор заблокирован, чтобы исключить второй вызов "
                    "и возможное повторное списание.",
                ]
            )
        elif project.status == "sheet_review":
            lines.extend(
                [
                    "",
                    "🖼 Один общий storyboard sheet сгенерирован и сохранён.",
                    "Открой сохранённое изображение и отдельно подтверди, попроси "
                    "правки или отклони его. Phase 2 пока закрыта.",
                ]
            )
            keyboard.append(
                [('🖼 Показать готовый sheet', f'sb:sheet_show:{project.project_id}')]
            )
        elif project.status == "sheet_approved":
            lines.extend(
                [
                    "",
                    "✅ Готовый storyboard sheet подтверждён.",
                    "Phase 1 завершена. Phase 2 не запускалась.",
                ]
            )
            keyboard.append(
                [('🖼 Показать готовый sheet', f'sb:sheet_show:{project.project_id}')]
            )
        elif project.status == "rejected":
            lines.extend(
                [
                    "",
                    "Storyboard-проект отклонён. Phase 1 и Phase 2 не запускаются.",
                ]
            )
        elif project.status == "planning":
            lines.extend(
                [
                    "",
                    "Автоматический разбор можно безопасно повторить: новый проект и платный provider не создаются.",
                ]
            )
            keyboard.append(
                [('🔄 Повторить автоматический разбор', f'sb:plan_retry:{project.project_id}')]
            )

        keyboard.extend(
            [
                [('« К проектам', 'sb:list')],
                [('« Storyboard', 'sb:home')],
            ]
        )
        text = "\n".join(lines)
        if notice:
            text = f"{notice}\n\n{text}"
        return BotResponse(text, keyboard, replace_message)

    def storyboard_about(self, replace_message: bool = False) -> BotResponse:
        return BotResponse(
            text=(
                "ℹ️ Как работает Storyboard\n\n"
                "1. Ты присылаешь идею или сценарий одним сообщением.\n"
                "2. Система сама строит план последовательных панелей и один prompt "
                "для общего storyboard sheet.\n"
                "3. При необходимости ты отдельно добавляешь фото-референсы; план "
                "автоматически обновляется.\n"
                "4. Ты подтверждаешь, исправляешь или отклоняешь план.\n"
                "5. После подтверждения плана система должна показать provider/model, "
                "точный результат и стоимость одной общей раскадровки. Этот платный "
                "этап пока не подключён.\n"
                "6. Только готовый storyboard sheet получает отдельное "
                "approve/revise/reject. Phase 2 с видеопромптом открывается лишь после "
                "approval готового sheet.\n\n"
                "Сейчас доступны автоматический text planning, persistence, resume, "
                "референсы и решения по плану. Image/video providers не вызываются."
            ),
            keyboard=[[('« Storyboard', 'sb:home')]],
            replace_message=replace_message,
        )

    @staticmethod
    def _storyboard_status_label(status: str) -> str:
        return {
            "planning": "Автоматический разбор",
            "plan_review": "Проверка плана",
            "plan_approved": "План подтверждён",
            "sheet_awaiting_confirmation": "Ждёт подтверждения генерации",
            "sheet_generating": "Генерация зарегистрирована",
            "sheet_reconciliation_required": "Требуется ручная сверка",
            "sheet_review": "Проверка готового sheet",
            "sheet_approved": "Storyboard sheet подтверждён",
            "in_progress": "В работе",
            "awaiting_storyboard_approval": "Ждёт подтверждения",
            "review_ready": "Готов к завершению",
            "completed": "Завершён",
            "rejected": "Отклонён",
        }.get(status, status)

    def main_menu_keyboard(self) -> InlineKeyboard:
        runs = self._content_store().list_runs(limit=1)
        first_row: list[tuple[str, str]] = [('🎬 Новый ролик', 'cf:new')]
        if runs:
            first_row.append(('▶️ Продолжить последний', f'cf_run:{runs[0].run_id}'))
        radar_row = (
            [('📡 Радар', 'radar:home'), ('💡 Сохранённые идеи', 'ideas:list')]
            if self.radar is not None
            else [('💡 Идеи радара', 'ideas:list'), ('📥 Файлы партнёра', 'shared:list')]
        )
        rows = [first_row, [('🧩 Storyboard', 'cf:storyboards')], radar_row]
        if self.radar is not None:
            rows.append([('📥 Файлы партнёра', 'shared:list')])
        rows.extend(
            [
                [('📋 Мои ролики', 'cf:list'), ('🎥 Генерации видео', 'cf:videos')],
                [('🩺 Состояние системы', 'menu:status')],
            ]
        )
        return rows

    def work_status(self, replace_message: bool = False) -> BotResponse:
        path = self.settings.codex_workdir / "reports" / "CURRENT_WORK_STATUS.md"
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            text = "# Ход работы\n\nТекущий отчёт ещё не создан."
        return BotResponse(
            text=text,
            keyboard=[
                [('🔄 Обновить', 'menu:work_status')],
                [('« Главное меню', 'menu:main')],
            ],
            replace_message=replace_message,
            render_markdown=True,
        )

    def factory_home(self, user: TelegramUser, replace_message: bool = False) -> BotResponse:
        runs = self._content_store().list_runs(limit=5)
        if runs:
            latest_lines = [
                f"• {run.run_id} · {self._run_status_label(run.status)} · "
                f"{run.current_stage_spec.title}"
                for run in runs[:3]
            ]
            latest = "\n".join(latest_lines)
        else:
            latest = "Запусков пока нет."
        return BotResponse(
            text=(
                "🏭 Контент-завод\n\n"
                "Пайплайн:\n"
                "идея → сценарий → отдельные кадры → выбор кадров → "
                "видеопромпты → подтверждение → видео\n\n"
                f"Последние запуски:\n{latest}\n\n"
                "Кадры создаются отдельными файлами. Платная генерация видео "
                "запускается только после явного подтверждения."
            ),
            keyboard=[
                [('🎬 Новый ролик', 'cf:new')],
                [('📋 Все запуски', 'cf:list')],
                [('« Главное меню', 'menu:main')],
            ],
            replace_message=replace_message,
        )

    def start_new_content(
        self,
        user: TelegramUser,
        replace_message: bool = False,
    ) -> BotResponse:
        self._activate_content_project(user.user_id)
        self.vault.set_input_state(user.user_id, {"kind": "content_idea"})
        return BotResponse(
            text=(
                "🎬 Новый ролик\n\n"
                "Пришли идею обычным сообщением. Можно коротко, например:\n\n"
                "«Вертикальный ролик на 30 секунд о том, как AI-агент собирает контент-завод. "
                "Стиль — динамичный и немного ироничный».\n\n"
                "Чем точнее укажешь цель, платформу и настроение, тем меньше понадобится правок."
            ),
            keyboard=[[('Отмена', 'input:cancel')]],
            replace_message=replace_message,
        )

    def runs_menu(self, user: TelegramUser, replace_message: bool = False) -> BotResponse:
        runs = self._content_store().list_runs(limit=10)
        if not runs:
            return BotResponse(
                text="📋 Запусков пока нет. Создай первый ролик.",
                keyboard=[
                    [('🎬 Новый ролик', 'cf:new')],
                    [('« Главное меню', 'menu:main')],
                ],
                replace_message=replace_message,
            )

        lines = ["📋 Последние запуски", ""]
        keyboard: InlineKeyboard = []
        for run in runs:
            lines.append(
                f"{run.run_id} · {self._run_status_label(run.status)}\n"
                f"{run.idea[:90]}"
            )
            keyboard.append(
                [(f"{self._run_status_icon(run.status)} {run.run_id}", f"cf_run:{run.run_id}")]
            )
        keyboard.append([('🎬 Новый ролик', 'cf:new')])
        keyboard.append([('« Главное меню', 'menu:main')])
        return BotResponse("\n\n".join(lines), keyboard, replace_message)

    def video_jobs_menu(self, replace_message: bool = False) -> BotResponse:
        lines = ["🎥 Генерации видео", ""]
        keyboard: InlineKeyboard = []
        found = 0
        for run in self._content_store().list_runs(limit=50):
            state = self._production_store().load(run.run_id)
            jobs = state.get("video_jobs", {})
            if not isinstance(jobs, dict) or not jobs:
                continue
            found += 1
            completed = sum(
                isinstance(job, dict) and job.get("status") == "completed"
                for job in jobs.values()
            )
            active = sum(
                isinstance(job, dict)
                and job.get("status") in {"submitting", "pending", "queued", "processing", "running"}
                for job in jobs.values()
            )
            lines.append(
                f"{run.run_id}: готово {completed}/{len(jobs)}, в работе {active}"
            )
            keyboard.append(
                [(f"⏳ {run.run_id}", f"cf_video_status:{run.run_id}")]
            )
        if not found:
            lines.append("Активных или завершённых видеозадач пока нет.")
        keyboard.extend(
            [
                [('📋 Все ролики', 'cf:list')],
                [('« Главное меню', 'menu:main')],
            ]
        )
        return BotResponse("\n".join(lines), keyboard, replace_message)

    def run_detail(self, run_id: str, replace_message: bool = False) -> BotResponse:
        run = self._require_content_run(run_id)
        production = self._production_store().load(run.run_id)
        frames = production.get("frames", {}) if isinstance(production.get("frames"), dict) else {}
        references = (
            production.get("references", {})
            if isinstance(production.get("references"), dict)
            else {}
        )
        ready_references = sum(
            item.get("status") == "ready" for item in references.values()
        )
        failed_references = sum(
            item.get("status") == "failed" for item in references.values()
        )
        ready_frames = sum(item.get("status") == "ready" for item in frames.values())
        failed_frames = sum(item.get("status") == "failed" for item in frames.values())
        selected_frames = len(production.get("selected_frame_ids", []))
        video_prompts = len(production.get("video_prompts", {}))
        video_settings = (
            production.get("video_settings", {})
            if isinstance(production.get("video_settings"), dict)
            else {}
        )
        video_jobs = production.get("video_jobs", {}) if isinstance(production.get("video_jobs"), dict) else {}
        ready_videos = sum(
            bool(item.get("status") == "completed" and item.get("video_file"))
            for item in video_jobs.values()
        )
        failed_videos = sum(
            item.get("status") in {"failed", "submission_unknown", "download_failed"}
            for item in video_jobs.values()
        )
        progress: list[str] = []
        for stage in PIPELINE:
            if stage.key in run.completed_stages:
                marker = "✅"
            elif stage.key == run.current_stage:
                marker = "▶️"
            else:
                marker = "▫️"
            progress.append(f"{marker} {stage.title}")

        error = f"\n\nПричина ошибки: {run.last_error}" if run.last_error else ""
        keyboard: InlineKeyboard = []
        if run.status == "waiting_approval":
            keyboard.extend(
                [
                    [('✅ Продолжить', f'cf_next:{run.run_id}:{run.current_stage}')],
                    [('✏️ Доработать', f'cf_revise:{run.run_id}:{run.current_stage}')],
                    [('📄 Показать этап', f'cf_view:{run.run_id}')],
                ]
            )
        elif run.status == "failed":
            keyboard.extend(
                [
                    [('🔍 Разобрать ошибку', f'cf_diagnose:{run.run_id}')],
                    [('🔄 Повторить', f'cf_retry:{run.run_id}:{run.current_stage}')],
                    [('✏️ Уточнить задачу', f'cf_revise:{run.run_id}:{run.current_stage}')],
                ]
            )
        elif run.status == "ready_for_production":
            keyboard.append([('📄 Показать проверку', f'cf_view:{run.run_id}')])
            keyboard.append(
                [('🧑 Подготовить референсы', f'cf_generate_refs:{run.run_id}')]
            )
            if references:
                keyboard.append(
                    [('🗂 Референсы', f'cf_show_refs:{run.run_id}')]
                )
            keyboard.append(
                [
                    (
                        '🎨 Создать отдельные кадры'
                        if not frames
                        else '🎨 Создать недостающие кадры заново',
                        f'cf_generate_visual:{run.run_id}',
                    )
                ]
            )
            if frames:
                keyboard.append(
                    [('🖼 Кадры и выбор', f'cf_show_visual:{run.run_id}')]
                )
                if references and ready_references == len(references):
                    keyboard.append(
                        [('🔄 Пересоздать все кадры с референсами', f'cf_regenerate_frames:{run.run_id}')]
                    )
            if selected_frames:
                if video_settings:
                    keyboard.append(
                        [('🎥 Создать видеопромпты', f'cf_video_prompts:{run.run_id}')]
                    )
                    keyboard.append(
                        [('⚙️ Сменить модель или качество', f'cf_video_setup:{run.run_id}')]
                    )
                else:
                    keyboard.append(
                        [('🎥 Выбрать модель и качество', f'cf_video_setup:{run.run_id}')]
                    )
            if video_prompts:
                keyboard.append(
                    [('📄 Показать видеопромпты', f'cf_show_video_prompts:{run.run_id}')]
                )
                keyboard.append(
                    [('💳 Параметры и подтверждение', f'cf_video_prepare:{run.run_id}')]
                )
            if video_jobs:
                keyboard.append([('⏳ Статус видео', f'cf_video_status:{run.run_id}')])
        elif run.status in {"queued", "running"}:
            keyboard.append(
                [('▶️ Продолжить этап', f'cf_retry:{run.run_id}:{run.current_stage}')]
            )

        has_technical_failure = bool(
            run.status == "failed"
            or failed_references
            or failed_frames
            or failed_videos
        )
        if has_technical_failure:
            latest_repair = self._repair_manager().latest_for_run(run.run_id)
            if latest_repair and latest_repair.status not in {"discarded", "rolled_back"}:
                keyboard.append(
                    [
                        (
                            f'🧰 Ремонт {latest_repair.repair_id}',
                            f'repair_status:{latest_repair.repair_id}',
                        )
                    ]
                )
            else:
                keyboard.append(
                    [('🧰 Подготовить исправление', f'repair_review:{run.run_id}')]
                )

        if run.status not in {"cancelled", "ready_for_production"}:
            keyboard.append([('Отменить запуск', f'cf_cancel:{run.run_id}')])
        keyboard.extend(
            [
                [('📋 Все запуски', 'cf:list')],
                [('« Главное меню', 'menu:main')],
            ]
        )
        return BotResponse(
            text=(
                f"🎬 {run.run_id}\n\n"
                f"Статус: {self._run_status_label(run.status)}\n"
                f"Текущий этап: {run.current_stage_spec.title}\n\n"
                f"Идея:\n{run.idea}\n\n"
                f"Референсы: {ready_references}/{len(references) or 0} готово"
                f"{f', {failed_references} с ошибкой' if failed_references else ''}\n"
                f"Отдельные кадры: {ready_frames}/{len(frames) or 0} готово"
                f"{f', {failed_frames} с ошибкой' if failed_frames else ''}\n"
                f"Выбрано кадров: {selected_frames}\n"
                f"Видеопрофиль: {profile_label(video_settings)}\n"
                f"Видеопромптов: {video_prompts}\n"
                f"Готовых видео: {ready_videos}/{len(video_jobs) or 0}\n\n"
                f"Прогресс:\n" + "\n".join(progress) + error
            ),
            keyboard=keyboard,
            replace_message=replace_message,
        )

    def view_current_artifact(self, run_id: str) -> BotResponse:
        run = self._require_content_run(run_id)
        content = self._content_store().read_artifact(run.run_id, run.current_stage)
        if not content:
            content = "Артефакт текущего этапа ещё не создан."
        else:
            content = present_stage(run.current_stage, content)
        return BotResponse(
            text=f"{run.run_id}\n\n{content}",
            keyboard=[[('« К запуску', f'cf_run:{run.run_id}')]],
            render_markdown=True,
        )

    def generate_references(
        self,
        user: TelegramUser,
        run_id: str,
        reference_ids: list[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> BotResponse:
        run = self._require_content_run(run_id)
        if run.status != "ready_for_production":
            return BotResponse(
                "Референсы доступны после подтверждения текстового production-пакета.",
                [[('« К запуску', f'cf_run:{run.run_id}')]],
            )
        if not self.image_client.is_configured:
            return BotResponse(
                "Генератор изображений не подключён. Проверь IMAGE_PROVIDER и Codex CLI.",
                [[('« К запуску', f'cf_run:{run.run_id}')]],
            )
        try:
            self._ensure_scene_contract(run)
        except Exception as exc:
            return BotResponse(
                f"Не удалось подготовить план референсов для {run.run_id}.\n\nПричина: {exc}",
                [[('« К запуску', f'cf_run:{run.run_id}')]],
            )
        state = self._production_store().load(run.run_id)
        references = state.get("references", {})
        if not isinstance(references, dict) or not references:
            return BotResponse(
                "В этом ролике не обнаружены повторяющиеся персонажи или отдельные visual references.",
                [
                    [('🎨 Создать отдельные кадры', f'cf_generate_visual:{run.run_id}')],
                    [('« К запуску', f'cf_run:{run.run_id}')],
                ],
            )
        if reference_ids is None:
            reference_ids = [
                reference_id
                for reference_id, item in references.items()
                if item.get("status") != "ready"
            ]
            if not reference_ids:
                return BotResponse(
                    f"Все {len(references)} референса уже готовы.",
                    [
                        [('🗂 Показать референсы', f'cf_show_refs:{run.run_id}')],
                        [('🎨 Создать отдельные кадры', f'cf_generate_visual:{run.run_id}')],
                    ],
                )
        try:
            self.telegram.send_message(
                user.chat_id,
                f"🧑 {run.run_id}\nСоздаются канонические референсы: 0 из {len(reference_ids)}.",
            )
        except TelegramApiError:
            pass

        def progress(done: int, total: int, result: ReferenceResult) -> None:
            if result.status == "ready":
                try:
                    self.telegram.send_photo(
                        user.chat_id,
                        Path(result.path),
                        caption=(
                            f"🧑 {run.run_id} · {result.reference_id}\n"
                            f"Тип: {self._reference_kind_label(result.kind)} · готово {done} из {total}"
                        ),
                        keyboard=[
                            [('🔄 Повторить референс', f'cf_retry_ref:{run.run_id}:{result.reference_id}')]
                        ],
                    )
                except TelegramApiError:
                    pass
            elif result.status == "cancelled":
                try:
                    self.telegram.send_message(
                        user.chat_id,
                        f"⛔ {run.run_id} · {result.reference_id}: генерация остановлена.",
                    )
                except TelegramApiError:
                    pass
            else:
                try:
                    self.telegram.send_message(
                        user.chat_id,
                        (
                            f"⚠️ {run.run_id} · {result.reference_id}: референс не создан.\n"
                            f"Причина: {result.error[:300]}\n"
                            "Остальные референсы продолжают создаваться."
                        ),
                        [[('🔄 Повторить референс', f'cf_retry_ref:{run.run_id}:{result.reference_id}')]],
                    )
                except TelegramApiError:
                    pass

        result = ReferenceBatchGenerator(
            self._production_store(), self.image_client
        ).generate(
            run.run_id,
            reference_ids=reference_ids,
            on_progress=progress,
            should_cancel=cancel_event.is_set if cancel_event else None,
        )
        return BotResponse(
            (
                f"{'⛔' if result.was_cancelled else ('✅' if result.failed_count == 0 else '⚠️')} "
                f"{run.run_id}: референсы обработаны.\n\n"
                f"Готово: {result.ready_count}\nОшибки: {result.failed_count}\n"
                + (
                    "Генерация остановлена. Готовые файлы сохранены; оставшиеся можно запустить позже."
                    if result.was_cancelled
                    else "Карточки персонажей будут передаваться в каждую связанную сцену как реальные изображения."
                )
            ),
            [
                [('🗂 Показать референсы', f'cf_show_refs:{run.run_id}')],
                [('🎨 Создать отдельные кадры', f'cf_generate_visual:{run.run_id}')],
                [('« К запуску', f'cf_run:{run.run_id}')],
            ],
        )

    def show_references(self, user: TelegramUser, run_id: str) -> BotResponse:
        state = self._production_store().load(run_id)
        references = state.get("references", {})
        sent = 0
        if isinstance(references, dict):
            for reference_id, item in references.items():
                try:
                    path = self._production_store().reference_path(run_id, reference_id)
                except ValueError:
                    path = None
                if path is None:
                    continue
                try:
                    state_label = str(item.get("state_label", "")).strip()
                    base_reference_id = str(item.get("base_reference_id", "")).strip()
                    state_note = f"\nСостояние: {state_label}" if state_label else ""
                    base_note = (
                        f"\nБазовая карточка: {base_reference_id}"
                        if base_reference_id
                        else ""
                    )
                    self.telegram.send_photo(
                        user.chat_id,
                        path,
                        caption=(
                            f"{reference_id} · {item.get('name', reference_id)}\n"
                            f"Тип: {self._reference_kind_label(str(item.get('kind', 'reference')))}\n"
                            f"{state_note}{base_note}\n"
                            f"Сцены: {', '.join(item.get('scene_ids', [])) or 'не указаны'}"
                        ),
                        keyboard=[
                            [('🔄 Повторить референс', f'cf_retry_ref:{run_id}:{reference_id}')]
                        ],
                    )
                    sent += 1
                except TelegramApiError:
                    continue
        if not sent:
            return BotResponse(
                "Готовых референсов пока нет.",
                [[('🧑 Создать референсы', f'cf_generate_refs:{run_id}')]],
            )
        return BotResponse(
            f"Показано референсов: {sent}.",
            [
                [('🎨 Создать отдельные кадры', f'cf_generate_visual:{run_id}')],
                [('« К запуску', f'cf_run:{run_id}')],
            ],
        )

    def generate_frames(
        self,
        user: TelegramUser,
        run_id: str,
        scene_ids: list[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> BotResponse:
        run = self._require_content_run(run_id)
        if run.status != "ready_for_production":
            return BotResponse(
                "Отдельные кадры доступны после подтверждения всех текстовых этапов.",
                [[('« К запуску', f'cf_run:{run.run_id}')]],
            )
        if not self.image_client.is_configured:
            return BotResponse(
                "Генератор кадров не подключён. Проверь IMAGE_PROVIDER и Codex CLI.",
                [[('« К запуску', f'cf_run:{run.run_id}')]],
            )
        try:
            scenes = self._ensure_scene_contract(run)
        except (ProductionContractError, Exception) as exc:
            return BotResponse(
                f"Не удалось подготовить структуру сцен для {run.run_id}.\n\nПричина: {exc}",
                [[('« К запуску', f'cf_run:{run.run_id}')]],
            )
        selected_scene_ids = set(scene_ids or [scene.scene_id for scene in scenes])
        required_reference_ids = {
            reference_id
            for scene in scenes
            if scene.scene_id in selected_scene_ids
            for reference_id in scene.reference_ids
        }
        reference_state = self._production_store().load(run.run_id).get("references", {})
        missing_references = sorted(
            reference_id
            for reference_id in required_reference_ids
            if not isinstance(reference_state, dict)
            or reference_state.get(reference_id, {}).get("status") != "ready"
        )
        if missing_references:
            return BotResponse(
                (
                    "Сначала нужно создать карточки и visual references.\n\n"
                    f"Не готовы: {', '.join(missing_references)}."
                ),
                [
                    [('🧑 Создать референсы', f'cf_generate_refs:{run.run_id}')],
                    [('« К запуску', f'cf_run:{run.run_id}')],
                ],
            )
        if scene_ids is None:
            state = self._production_store().load(run.run_id)
            frames = state.get("frames", {})
            scene_ids = [
                scene.scene_id
                for scene in scenes
                if frames.get(scene.scene_id, {}).get("status") != "ready"
            ]
            if not scene_ids:
                return BotResponse(
                    f"Все {len(scenes)} кадров уже готовы. Выбери нужные или повтори конкретный кадр.",
                    [[('🖼 Кадры и выбор', f'cf_show_visual:{run.run_id}')]],
                )
        requested_count = len(scene_ids)
        try:
            self.telegram.send_message(
                user.chat_id,
                f"🎨 {run.run_id}\nСоздаются отдельные кадры: 0 из {requested_count}.",
            )
        except TelegramApiError:
            pass

        def progress(done: int, total: int, result: FrameResult) -> None:
            if result.status == "ready":
                path = Path(result.path)
                try:
                    self.telegram.send_photo(
                        user.chat_id,
                        path,
                        caption=f"🖼 {run.run_id} · {result.scene_id} · кадр {done} из {total}",
                        keyboard=[
                            [('🔄 Повторить этот кадр', f'cf_retry_frame:{run.run_id}:{result.scene_id}')],
                            [('Убрать из выбранных', f'cf_toggle_frame:{run.run_id}:{result.scene_id}')],
                        ],
                    )
                except TelegramApiError:
                    pass
            elif result.status == "cancelled":
                try:
                    self.telegram.send_message(
                        user.chat_id,
                        f"⛔ {run.run_id} · {result.scene_id}: генерация кадра остановлена.",
                    )
                except TelegramApiError:
                    pass
            else:
                try:
                    self.telegram.send_message(
                        user.chat_id,
                        (
                            f"⚠️ {run.run_id} · {result.scene_id}: кадр не создан.\n"
                            f"Причина: {result.error[:300]}\n"
                            "Остальные сцены продолжают обрабатываться."
                        ),
                        [[('🔄 Повторить этот кадр', f'cf_retry_frame:{run.run_id}:{result.scene_id}')]],
                    )
                except TelegramApiError:
                    pass
            try:
                self.telegram.send_message(
                    user.chat_id,
                    f"Статус {run.run_id}: создано/обработано кадров {done} из {total}.",
                )
            except TelegramApiError:
                pass

        result = FrameBatchGenerator(self._production_store(), self.image_client).generate(
            run.run_id,
            scene_ids=scene_ids,
            aspect_ratio="9:16",
            on_progress=progress,
            should_cancel=cancel_event.is_set if cancel_event else None,
        )
        video_settings = self._production_store().load(run.run_id).get("video_settings", {})
        video_next = (
            ('🎥 Создать видеопромпты', f'cf_video_prompts:{run.run_id}')
            if video_settings
            else ('🎥 Выбрать модель и качество', f'cf_video_setup:{run.run_id}')
        )
        qa_text = ""
        if (
            not result.was_cancelled
            and result.failed_count == 0
            and self._all_production_images_are_rasters(run.run_id)
        ):
            try:
                self.telegram.send_message(
                    user.chat_id,
                    f"🔎 {run.run_id}\nПроверяю реальные изображения, references и continuity.",
                )
            except TelegramApiError:
                pass
            try:
                qa = PostImageQa(
                    self._production_store(), self.image_inspector, self.llm
                ).run(run.run_id)
                qa_text = "\n\n" + self._format_image_qa(qa.to_dict())
            except Exception as exc:
                qa_text = (
                    "\n\n⚠️ Автоматическая визуальная проверка не завершена: "
                    f"{str(exc)[:300]}. Кадры сохранены; проверку можно повторить кнопкой."
                )
        return BotResponse(
            (
                f"{'⛔' if result.was_cancelled else ('✅' if result.failed_count == 0 else '⚠️')} "
                f"{run.run_id}: обработка кадров завершена.\n\n"
                f"Готово: {result.ready_count}\nОшибки: {result.failed_count}\n"
                + (
                    "Генерация остановлена. Готовые кадры сохранены; остальные можно продолжить позже."
                    if result.was_cancelled
                    else "Каждая сцена сохранена отдельным файлом."
                )
                + qa_text
            ),
            [
                [('🖼 Кадры и выбор', f'cf_show_visual:{run.run_id}')],
                [('🔎 Проверить кадры', f'cf_image_qa:{run.run_id}')],
                [video_next],
                [('« К запуску', f'cf_run:{run.run_id}')],
            ],
        )

    def _start_image_job(
        self,
        user: TelegramUser,
        run_id: str,
        label: str,
        operation,
    ) -> BotResponse:
        """Run a cancellable image batch outside the Telegram polling thread."""

        with self._active_image_jobs_lock:
            if self._active_image_jobs:
                active_run = next(iter(self._active_image_jobs))
                return BotResponse(
                    f"Уже выполняется генерация для {active_run}. Сначала дождись её или останови.",
                    [[('⛔ Остановить генерации', f'cf_stop_images:{active_run}')]],
                )
            event = threading.Event()
            self._active_image_jobs[run_id] = event

        def worker() -> None:
            try:
                response = operation(event)
                self.telegram.send_message(
                    user.chat_id,
                    response.text,
                    response.keyboard,
                    render_markdown=response.render_markdown,
                )
            except Exception as exc:
                try:
                    self.telegram.send_message(
                        user.chat_id,
                        f"❌ {run_id}: фоновая генерация завершилась ошибкой.\nПричина: {str(exc)[:500]}",
                        [[('« К запуску', f'cf_run:{run_id}')]],
                    )
                except TelegramApiError:
                    pass
            finally:
                with self._active_image_jobs_lock:
                    self._active_image_jobs.pop(run_id, None)

        threading.Thread(
            target=worker,
            name=f"content-images-{run_id}",
            daemon=True,
        ).start()
        return BotResponse(
            f"▶️ {run_id}: {label} запущена в фоне. Бот остаётся доступен.",
            [
                [('⛔ Остановить генерации', f'cf_stop_images:{run_id}')],
                [('« К запуску', f'cf_run:{run_id}')],
            ],
        )

    def stop_image_generation(self, run_id: str) -> BotResponse:
        with self._active_image_jobs_lock:
            event = self._active_image_jobs.get(run_id)
        if event is None:
            return BotResponse(
                "Активной генерации изображений для этого запуска нет.",
                [[('« К запуску', f'cf_run:{run_id}')]],
            )
        event.set()
        cancel_active = getattr(self.image_client, "cancel_active", None)
        current_stopped = bool(cancel_active()) if callable(cancel_active) else False
        return BotResponse(
            (
                f"⛔ {run_id}: остановка запрошена. "
                + (
                    "Текущий процесс Codex завершается."
                    if current_stopped
                    else "Новые изображения запускаться не будут."
                )
                + " Уже готовые файлы останутся сохранены."
            ),
            [[('« К запуску', f'cf_run:{run_id}')]],
        )

    def run_image_qa(self, user: TelegramUser, run_id: str) -> BotResponse:
        run = self._require_content_run(run_id)
        try:
            self.telegram.send_message(
                user.chat_id,
                f"🔎 {run.run_id}\nАнализирую готовые reference-карточки и кадры.",
            )
        except TelegramApiError:
            pass
        try:
            result = PostImageQa(
                self._production_store(), self.image_inspector, self.llm
            ).run(run.run_id)
        except Exception as exc:
            return BotResponse(
                f"Визуальная проверка не выполнена.\n\nПричина: {str(exc)[:500]}",
                [
                    [('🔄 Повторить проверку', f'cf_image_qa:{run.run_id}')],
                    [('« К запуску', f'cf_run:{run.run_id}')],
                ],
            )
        keyboard: InlineKeyboard = [
            [('🖼 Кадры и выбор', f'cf_show_visual:{run.run_id}')],
        ]
        if result.ready_for_video_prompts:
            state = self._production_store().load(run.run_id)
            callback = (
                f'cf_video_prompts:{run.run_id}'
                if state.get("video_settings")
                else f'cf_video_setup:{run.run_id}'
            )
            keyboard.append([('🎥 Перейти к видео', callback)])
        keyboard.append([('« К запуску', f'cf_run:{run.run_id}')])
        return BotResponse(self._format_image_qa(result.to_dict()), keyboard)

    def _all_production_images_are_rasters(self, run_id: str) -> bool:
        state = self._production_store().load(run_id)
        paths: list[Path] = []
        for reference_id, item in state.get("references", {}).items():
            if item.get("status") != "ready":
                return False
            path = self._production_store().reference_path(run_id, reference_id)
            if path is not None:
                paths.append(path)
        for scene_id, item in state.get("frames", {}).items():
            if item.get("status") != "ready":
                return False
            path = self._production_store().frame_path(run_id, scene_id)
            if path is None:
                return False
            paths.append(path)
        signatures = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"RIFF")
        return bool(paths) and all(
            any(path.read_bytes()[:12].startswith(signature) for signature in signatures)
            for path in paths
        )

    @staticmethod
    def _format_image_qa(report: dict[str, Any]) -> str:
        verdict = str(report.get("verdict", "warning")).lower()
        icon = {"pass": "✅", "warning": "⚠️", "fail": "❌"}.get(verdict, "⚠️")
        issues = report.get("issues", [])
        lines = [
            f"{icon} Проверка изображений: {verdict.upper()}",
            str(report.get("summary", "Проверка завершена.")),
            f"Проверено файлов: {report.get('checked_items', 0)}",
            (
                "Можно создавать видеопромпты: да"
                if report.get("ready_for_video_prompts")
                else "Можно создавать видеопромпты: нет"
            ),
        ]
        for issue in issues[:10] if isinstance(issues, list) else []:
            if isinstance(issue, dict):
                lines.append(
                    f"- {issue.get('item_id', '?')}: {issue.get('message', 'Проблема')}"
                )
        return "\n".join(lines)

    def show_frames(self, user: TelegramUser, run_id: str) -> BotResponse | None:
        """Send one ordered album and absorb duplicate taps that Telegram has already queued."""

        run = self._require_content_run(run_id)
        request_key = (user.chat_id, run.run_id)
        now = time.monotonic()
        with self._frame_gallery_lock:
            self._frame_gallery_not_before = {
                key: deadline
                for key, deadline in self._frame_gallery_not_before.items()
                if deadline > now
            }
            if (
                request_key in self._frame_gallery_in_flight
                or self._frame_gallery_not_before.get(request_key, 0.0) > now
            ):
                return None
            self._frame_gallery_in_flight.add(request_key)
            # The deadline is set before network I/O, so a failed or interrupted send
            # cannot turn a queued burst of taps into another flood.
            self._frame_gallery_not_before[request_key] = (
                now + self._frame_gallery_cooldown_seconds
            )

        try:
            state = self._production_store().load(run.run_id)
            frames = state.get("frames", {})
            album: list[tuple[Path, str]] = []
            for scene in state.get("scenes", []):
                scene_id = str(scene.get("scene_id", ""))
                frame = frames.get(scene_id, {}) if isinstance(frames, dict) else {}
                path = (
                    self._production_store().frame_path(run.run_id, scene_id)
                    if frame
                    else None
                )
                if path is None:
                    continue
                selected = bool(frame.get("selected"))
                album.append(
                    (
                        path,
                        (
                            f"{len(album) + 1}. {scene_id} · "
                            f"{'выбран' if selected else 'не выбран'} · "
                            f"попыток: {len(frame.get('attempts', []))}"
                        ),
                    )
                )

            if not album:
                return BotResponse(
                    "Готовых отдельных кадров пока нет.",
                    [[('🎨 Создать кадры', f'cf_generate_visual:{run.run_id}')]],
                    replace_message=True,
                )

            self.telegram.send_photo_album(user.chat_id, album)
            return self.frames_detail(run.run_id, replace_message=True)
        finally:
            with self._frame_gallery_lock:
                self._frame_gallery_in_flight.discard(request_key)

    def frames_detail(self, run_id: str, replace_message: bool = False) -> BotResponse:
        state = self._production_store().load(run_id)
        frames = state.get("frames", {}) if isinstance(state.get("frames"), dict) else {}
        lines = [f"🖼 Кадры · {run_id}", ""]
        keyboard: InlineKeyboard = []
        for scene in state.get("scenes", []):
            scene_id = str(scene.get("scene_id", ""))
            frame = frames.get(scene_id, {})
            status = str(frame.get("status", "pending"))
            selected = bool(frame.get("selected"))
            icon = "✅" if status == "ready" and selected else "▫️" if status == "ready" else "❌" if status == "failed" else "⏳"
            lines.append(f"{icon} {scene_id}: {status}")
            if status == "ready":
                keyboard.append(
                    [
                        (
                            f"{'Убрать' if selected else '✅ Выбрать'} {scene_id}",
                            f'cf_toggle_frame:{run_id}:{scene_id}',
                        ),
                        (f'🔄 {scene_id}', f'cf_retry_frame:{run_id}:{scene_id}'),
                    ]
                )
            elif status == "failed":
                keyboard.append([(f'🔄 {scene_id}', f'cf_retry_frame:{run_id}:{scene_id}')])
        selected_count = len(state.get("selected_frame_ids", []))
        video_settings = state.get("video_settings", {})
        lines.extend(
            [
                "",
                f"Выбрано: {selected_count} из {len(frames)}",
                f"Видеопрофиль: {profile_label(video_settings)}",
            ]
        )
        video_next = (
            ('🎥 Создать видеопромпты', f'cf_video_prompts:{run_id}')
            if video_settings
            else ('🎥 Выбрать модель и качество', f'cf_video_setup:{run_id}')
        )
        keyboard.extend(
            [
                [('✅ Выбрать все готовые', f'cf_select_all:{run_id}')],
                [video_next],
                [('« К запуску', f'cf_run:{run_id}')],
            ]
        )
        return BotResponse("\n".join(lines), keyboard, replace_message)

    def toggle_frame(self, run_id: str, scene_id: str) -> BotResponse:
        state = self._production_store().load(run_id)
        frame = state.get("frames", {}).get(scene_id, {})
        selected = not bool(frame.get("selected"))
        self._production_store().set_selected(run_id, scene_id, selected)
        return self.frames_detail(run_id, replace_message=True)

    def video_setup(self, run_id: str, replace_message: bool = False) -> BotResponse:
        state = self._production_store().load(run_id)
        if not state.get("selected_frame_ids"):
            return BotResponse(
                "Сначала выбери хотя бы один готовый кадр.",
                [[('🖼 Кадры и выбор', f'cf_show_visual:{run_id}')]],
                replace_message=replace_message,
            )
        model_rows: InlineKeyboard = [
            [('🌊 Seedance 2 · основной', f'cf_video_model:{run_id}:seedance')],
            [('🎞 Kling 3', f'cf_video_model:{run_id}:kling')],
        ]
        if self.settings.ltx_video_enabled:
            model_rows.append(
                [('⚡ LTX-2.3 · отдельный GPU', f'cf_video_model:{run_id}:ltx')]
            )
        model_rows.append([('« К запуску', f'cf_run:{run_id}')])
        return BotResponse(
            (
                f"🎥 Модель видео · {run_id}\n\n"
                f"Сейчас: {profile_label(state.get('video_settings'))}\n\n"
                "Сначала выбери модель. На следующем экране бот покажет доступное качество. "
                "Выбор сохранится только для этого ролика и будет использован при написании промптов."
            ),
            model_rows,
            replace_message=replace_message,
        )

    def video_quality_menu(
        self,
        run_id: str,
        model_key: str,
        replace_message: bool = False,
    ) -> BotResponse:
        profiles = profiles_for_model(model_key)
        if not profiles:
            raise ValueError("Неизвестная видеомодель.")
        if profiles[0].provider == "ltx" and not self.settings.ltx_video_enabled:
            return BotResponse(
                "LTX-2.3 выключен feature flag и пока недоступен для этого бота.",
                [[('« Выбор модели', f'cf_video_setup:{run_id}')]],
                replace_message=replace_message,
            )
        rows: InlineKeyboard = [
            [
                (
                    (
                        f"{profile.model_label} · {profile.quality_label}"
                        f" · {self._video_provider_label(profile.provider)}"
                        f"{' · рекомендуется' if profile.code == 's480' else ''}"
                    ),
                    f"cf_video_quality:{run_id}:{profile.code}",
                )
            ]
            for profile in profiles
        ]
        rows.append([('« Выбор модели', f'cf_video_setup:{run_id}')])
        return BotResponse(
            (
                f"⚙️ Качество · {run_id}\n\n"
                f"Модель: {profiles[0].model_label}\n"
                "Формат: 9:16. Звук указан в названии каждого варианта.\n\n"
                "Выбери качество. Затем бот спросит длительность клипа. "
                "Платный запрос на этом шаге не выполняется."
            ),
            rows,
            replace_message=replace_message,
        )

    def video_duration_menu(
        self,
        run_id: str,
        profile_code: str,
        replace_message: bool = False,
    ) -> BotResponse:
        profile = video_profile(profile_code)
        if profile.provider == "ltx" and not self.settings.ltx_video_enabled:
            return BotResponse(
                "LTX-2.3 выключен feature flag и пока недоступен для этого бота.",
                [[('« Выбор модели', f'cf_video_setup:{run_id}')]],
                replace_message=replace_message,
            )
        durations = (5,) if profile.provider == "ltx" else (5, 10, 15)
        return BotResponse(
            (
                f"⏱ Длительность клипа · {run_id}\n\n"
                f"Модель: {profile.model_label}\n"
                f"Качество: {profile.quality_label}\n\n"
                "Видеопромтер объединит только соседние кадры одной локации, которые "
                "помещаются в выбранное время. Для большинства сюжетных блоков подходят "
                "10–15 секунд; 5 секунд оставлены для коротких действий."
            ),
            [
                [(f"{duration} секунд", f"cf_video_duration:{run_id}:{profile.code}-{duration}")]
                for duration in durations
            ]
            + [[('« К качеству', f'cf_video_model:{run_id}:{profile.model_key}')]],
            replace_message=replace_message,
        )

    def select_video_profile(
        self,
        run_id: str,
        profile_code: str,
        duration_seconds: int | None = None,
    ) -> BotResponse:
        profile = video_profile(profile_code)
        if profile.provider == "ltx" and not self.settings.ltx_video_enabled:
            return BotResponse(
                "LTX-2.3 выключен feature flag; профиль не сохранён.",
                [[('« Выбор модели', f'cf_video_setup:{run_id}')]],
            )
        selected_duration = profile.duration_seconds if duration_seconds is None else duration_seconds
        if profile.model.startswith("bytedance/seedance") and selected_duration not in {5, 10, 15}:
            raise ValueError("Seedance 2 поддерживает длительность 5, 10 или 15 секунд.")
        if profile.model == "kling/v3" and not 3 <= selected_duration <= 15:
            raise ValueError("Kling 3 поддерживает длительность от 3 до 15 секунд.")
        if profile.model == "ltx-2.3" and selected_duration != 5:
            raise ValueError("LTX-2.3 до benchmark поддерживает только 5 секунд.")
        settings = profile.to_dict()
        settings["duration_seconds"] = selected_duration
        try:
            self._production_store().set_video_settings(run_id, settings)
        except ProductionContractError as exc:
            return BotResponse(
                f"Профиль не изменён. Причина: {exc}",
                [[('⏳ Статус видео', f'cf_video_status:{run_id}')]],
            )
        return BotResponse(
            (
                f"✅ Видеопрофиль сохранён · {run_id}\n\n"
                f"Модель: {profile.model_label}\n"
                f"Качество: {profile.quality_label}\n"
                f"Провайдер: {self._video_provider_label(profile.provider)}\n"
                f"Длительность: {selected_duration} секунд\n"
                f"Формат: {profile.aspect_ratio}\n"
                f"Звук: {'да' if profile.sound_enabled else 'нет'}\n\n"
                "Теперь видеопромтер соберёт выбранные кадры в логические клипы."
            ),
            [
                [('🎥 Создать видеопромпты', f'cf_video_prompts:{run_id}')],
                [('⚙️ Изменить выбор', f'cf_video_setup:{run_id}')],
                [('« К запуску', f'cf_run:{run_id}')],
            ],
        )

    def generate_video_prompts(self, user: TelegramUser, run_id: str) -> BotResponse:
        run = self._require_content_run(run_id)
        state = self._production_store().load(run.run_id)
        if not state.get("selected_frame_ids"):
            return BotResponse(
                "Сначала выбери хотя бы один готовый кадр.",
                [[('🖼 Кадры и выбор', f'cf_show_visual:{run.run_id}')]],
            )
        video_settings = state.get("video_settings", {})
        if not isinstance(video_settings, dict) or not video_settings.get("model"):
            return self.video_setup(run.run_id)
        try:
            self.telegram.send_message(
                user.chat_id,
                (
                    f"🎥 {run.run_id}\n"
                    f"Видеопромтер анализирует кадры для {profile_label(video_settings)}. "
                    "Он объединяет только соседние кадры, явно описанные как одно непрерывное "
                    "действие без склейки, и показывает каждый клип отдельно."
                ),
            )
        except TelegramApiError:
            pass
        try:
            def prompt_progress(done: int, total: int, item: dict[str, Any]) -> None:
                try:
                    self.telegram.send_message(
                        user.chat_id,
                        (
                            f"✅ Видеопромтер · {run.run_id} · {item['clip_id']}\n"
                            f"Логический клип собран. Готово: {done} из {total}."
                        ),
                    )
                except TelegramApiError:
                    pass

            prompts = VideoPromptBuilder(
                self._production_store(), self.image_inspector, self.llm
            ).build(
                run.run_id,
                model_id=str(video_settings["model"]),
                duration_seconds=int(video_settings["duration_seconds"]),
                aspect_ratio=str(video_settings["aspect_ratio"]),
                sound_enabled=bool(video_settings["sound_enabled"]),
                provider_name=str(video_settings.get("provider") or "polza"),
                on_progress=prompt_progress,
            )
        except (ImageInspectionError, ProductionContractError, Exception) as exc:
            return BotResponse(
                f"❌ {run.run_id}: видеопромпты не созданы.\n\nПричина: {exc}",
                [
                    [('🔄 Повторить', f'cf_video_prompts:{run.run_id}')],
                    [('« К запуску', f'cf_run:{run.run_id}')],
                ],
            )
        return BotResponse(
            present_video_prompts(
                run.run_id,
                {str(item["clip_id"]): item for item in prompts},
                self._production_store().load(run.run_id).get("video_prompt_qa"),
            ),
            [
                [('💳 Параметры и подтверждение', f'cf_video_prepare:{run.run_id}')],
                [('« К запуску', f'cf_run:{run.run_id}')],
            ],
            render_markdown=True,
        )

    def show_video_prompts(self, run_id: str) -> BotResponse:
        state = self._production_store().load(run_id)
        prompts = state.get("video_prompts", {})
        if not isinstance(prompts, dict) or not prompts:
            return BotResponse(
                "Видеопромпты ещё не созданы.",
                [[('🎥 Создать видеопромпты', f'cf_video_prompts:{run_id}')]],
            )
        jobs = state.get("video_jobs", {})
        retryable = [
            item
            for item in jobs.values()
            if isinstance(item, dict) and item.get("retry_allowed")
        ] if isinstance(jobs, dict) else []
        model_mismatch = self._video_prompt_model_mismatch(state)
        safe_to_replace = self._video_jobs_safe_to_replace(state)
        note = ""
        if model_mismatch and safe_to_replace:
            primary = [(
                '🎥 Пересоздать под выбранную модель',
                f'cf_video_prompts:{run_id}',
            )]
            note = (
                "\n\n⚠️ Эти видеопромпты относятся к прежней модели. "
                f"Пересоздай их перед новой генерацией {profile_label(state.get('video_settings'))}."
            )
        elif model_mismatch:
            primary = [('⏳ Статус видео', f'cf_video_status:{run_id}')]
            note = (
                "\n\n⚠️ У запуска есть внешняя или неоднозначная video task. "
                "Её нельзя автоматически заменить другой моделью."
            )
        elif retryable:
            primary = [(
                f'💳 Проверить повтор всех ({len(retryable)})',
                f'cf_retry_video_review:{run_id}',
            )]
        elif jobs:
            primary = [('⏳ Статус видео', f'cf_video_status:{run_id}')]
        else:
            primary = [('💳 Параметры и подтверждение', f'cf_video_prepare:{run_id}')]
        return BotResponse(
            present_video_prompts(run_id, prompts, state.get("video_prompt_qa")) + note,
            [primary, [('« К запуску', f'cf_run:{run_id}')]],
            render_markdown=True,
        )

    def prepare_video_generation(self, run_id: str) -> BotResponse:
        state = self._production_store().load(run_id)
        video_settings = state.get("video_settings", {})
        if not isinstance(video_settings, dict) or not video_settings.get("model"):
            return self.video_setup(run_id)
        if state.get("video_jobs"):
            if self._video_prompt_model_mismatch(state) and self._video_jobs_safe_to_replace(state):
                return BotResponse(
                    (
                        "Сохранённые видеопромпты и failed jobs относятся к прежней модели. "
                        f"Сначала пересоздай видеопромпты под {profile_label(video_settings)}; платный запрос "
                        "при этом не выполняется."
                    ),
                    [
                        [('🎥 Пересоздать под выбранную модель', f'cf_video_prompts:{run_id}')],
                        [('« К запуску', f'cf_run:{run_id}')],
                    ],
                )
            jobs = state.get("video_jobs", {})
            retryable = [
                item
                for item in jobs.values()
                if isinstance(item, dict) and item.get("retry_allowed")
            ] if isinstance(jobs, dict) else []
            keyboard: InlineKeyboard = []
            if retryable:
                keyboard.append([(
                    f'💳 Проверить повтор всех ({len(retryable)})',
                    f'cf_retry_video_review:{run_id}',
                )])
            keyboard.append([('⏳ Проверить статус', f'cf_video_status:{run_id}')])
            return BotResponse(
                "Для этого запуска уже существуют video tasks. Новая платная отправка не создана.",
                keyboard,
            )
        try:
            preview = self._video_job_manager().prepare(
                run_id,
                model=str(video_settings["model"]),
                mode=str(video_settings["mode"]),
                duration_seconds=int(video_settings["duration_seconds"]),
                aspect_ratio=str(video_settings["aspect_ratio"]),
                sound_enabled=bool(video_settings["sound_enabled"]),
                resolution=str(video_settings["resolution"]),
                provider=str(video_settings.get("provider") or "polza"),
                seed=int(video_settings.get("seed", 0)),
            )
        except (ProductionContractError, VideoProviderError) as exc:
            return BotResponse(
                f"Предпросмотр генерации не подготовлен.\n\nПричина: {exc}",
                [[('« К запуску', f'cf_run:{run_id}')]],
            )
        cost = (
            "Недоступна до запуска: API провайдера не вернул надёжную предварительную цену. "
            "Списание будет по тарифу выбранной модели."
            if preview.estimated_cost == "unavailable"
            else preview.estimated_cost
        )
        if preview.model == "ltx-2.3":
            quality = (
                f"Разрешение: {preview.resolution}\n"
                f"Workflow: {preview.mode}\n"
                f"Seed: {preview.seed}\n"
                "Ограничение до benchmark: один клип, один submission, без автоматического платного retry.\n"
            )
        elif preview.model in {"bytedance/seedance-2", "bytedance/seedance-2-fast"}:
            quality = (
                f"Разрешение: {preview.resolution}\n"
                "Мульти-шот API: нет — каждый подготовленный логический клип создаётся "
                "отдельным видео из своего стартового кадра.\n"
            )
        else:
            quality = (
                f"Режим: {preview.mode}\n"
                "Разрешение: качество определяется режимом модели.\n"
            )
        return BotResponse(
            (
                f"💳 Подтверждение расхода · {run_id}\n\n"
                f"Провайдер: {self._video_provider_label(preview.provider)}\n"
                f"Модель: {preview.model}\n{quality}"
                f"Длительность каждого видео: {preview.duration_seconds} сек.\n"
                f"Формат: {preview.aspect_ratio}\nЗвук: {'да' if preview.sound_enabled else 'нет'}\n"
                f"Количество видео: {preview.video_count}\nПримерная стоимость: {cost}\n"
                f"Внутренний ID подтверждения: {preview.approval_id}\n\n"
                "Нажатие «Подтвердить» запустит платные запросы. До этого видеопровайдер не вызывается."
            ),
            [
                [('✅ Подтвердить платную генерацию', f'cf_video_confirm:{run_id}:{preview.approval_id}')],
                [('Отмена', f'cf_video_cancel:{run_id}:{preview.approval_id}')],
                [('« К запуску', f'cf_run:{run_id}')],
            ],
        )

    def _video_prompt_model_mismatch(self, state: dict[str, Any]) -> bool:
        prompts = state.get("video_prompts", {})
        if not isinstance(prompts, dict) or not prompts:
            return False
        settings = state.get("video_settings", {})
        selected_model = str(settings.get("model", "")) if isinstance(settings, dict) else ""
        if not selected_model:
            return True
        return any(
            not isinstance(prompt, dict)
            or str(prompt.get("model_id", "")) != selected_model
            for prompt in prompts.values()
        )

    @staticmethod
    def _video_jobs_safe_to_replace(state: dict[str, Any]) -> bool:
        jobs = state.get("video_jobs", {})
        if not isinstance(jobs, dict):
            return True
        return not any(
            isinstance(job, dict)
            and (
                job.get("external_task_id")
                or job.get("submission_state") in {"submitting", "unknown"}
            )
            for job in jobs.values()
        )

    def confirm_video_generation(
        self, user: TelegramUser, run_id: str, approval_id: str
    ) -> BotResponse:
        manager = self._video_job_manager()
        try:
            manager.approve(run_id, approval_id)
            approval = self._production_store().load(run_id).get(
                "video_approval", {}
            )
            provider = self._video_provider_label(
                str(approval.get("provider") or "polza")
            )
            self.telegram.send_message(
                user.chat_id,
                f"✅ Расход подтверждён · {run_id}\nОтправляю video tasks в {provider}.",
            )
            manager.submit_approved(
                run_id,
                on_progress=self._video_submission_progress(user, run_id),
            )
        except (ProductionContractError, VideoProviderError, TelegramApiError) as exc:
            return BotResponse(
                f"❌ {run_id}: video tasks не запущены полностью.\n\nПричина: {exc}\n"
                "Сохранённое состояние можно проверить кнопкой статуса.",
                [[('⏳ Проверить статус', f'cf_video_status:{run_id}')]],
            )
        return self.video_status(user, run_id, poll=False)

    def video_status(
        self, user: TelegramUser, run_id: str, *, poll: bool = True
    ) -> BotResponse:
        try:
            state = (
                self._video_job_manager().poll_existing(run_id)
                if poll
                else self._production_store().load(run_id)
            )
        except VideoProviderError as exc:
            state = self._production_store().load(run_id)
            poll_error = str(exc)
        else:
            poll_error = ""
        delivered = self._deliver_ready_videos(user.chat_id, run_id, state)
        jobs = state.get("video_jobs", {}) if isinstance(state.get("video_jobs"), dict) else {}
        lines = [f"⏳ Видео · {run_id}", ""]
        keyboard: InlineKeyboard = []
        for scene_id, job in sorted(jobs.items()):
            status = str(job.get("status", "unknown"))
            external_id = str(job.get("external_task_id", "")) or "не получен"
            lines.append(
                f"{scene_id}: {self._video_status_label(status)} · task {external_id}"
            )
            error = self._friendly_video_error(str(job.get("error", "")))
            if error and status in {"failed", "submission_unknown", "download_failed"}:
                lines.append(f"Причина: {error}")
            if job.get("retry_allowed"):
                keyboard.append([(f'💳 Повторить платный запрос {scene_id}', f'cf_retry_video:{run_id}:{scene_id}')])
        retryable_ids = [
            scene_id
            for scene_id, job in sorted(jobs.items())
            if isinstance(job, dict) and job.get("retry_allowed")
        ]
        if len(retryable_ids) > 1:
            keyboard.insert(
                0,
                [(
                    f'💳 Проверить повтор всех ({len(retryable_ids)})',
                    f'cf_retry_video_review:{run_id}',
                )],
            )
        if not jobs:
            lines.append("Video tasks ещё не созданы.")
        if delivered:
            lines.extend(["", f"Новых видео отправлено: {delivered}"])
        if poll_error:
            lines.extend(["", f"Проверка статуса временно не удалась: {poll_error}"])
        keyboard.extend(
            [
                [('🔄 Обновить статус', f'cf_video_status:{run_id}')],
                [('« К запуску', f'cf_run:{run_id}')],
            ]
        )
        return BotResponse("\n".join(lines), keyboard)

    def retry_video_submission(
        self, user: TelegramUser, run_id: str, scene_id: str
    ) -> BotResponse:
        manager = self._video_job_manager()
        try:
            manager.retry_failed_submission(run_id, scene_id)
            manager.submit_approved(
                run_id,
                [scene_id],
                on_progress=self._video_submission_progress(user, run_id),
            )
        except (ProductionContractError, VideoProviderError) as exc:
            return BotResponse(
                f"Повтор {scene_id} не выполнен. Причина: {exc}",
                [[('⏳ Статус', f'cf_video_status:{run_id}')]],
            )
        return self.video_status(user, run_id, poll=False)

    def review_all_video_retries(self, run_id: str) -> BotResponse:
        state = self._production_store().load(run_id)
        jobs = state.get("video_jobs", {})
        retryable_ids = [
            scene_id
            for scene_id, job in sorted(jobs.items())
            if isinstance(job, dict)
            and job.get("retry_allowed")
            and not job.get("external_task_id")
        ]
        if not retryable_ids:
            return BotResponse(
                "Безопасных повторов без task ID сейчас нет.",
                [[('⏳ Статус', f'cf_video_status:{run_id}')]],
            )
        approval = state.get("video_approval", {})
        return BotResponse(
            (
                f"💳 Подтверждение повторной генерации · {run_id}\n\n"
                f"Сцены: {', '.join(retryable_ids)}\n"
                f"Модель: {approval.get('model', 'не указана')}\n"
                f"Разрешение: {approval.get('resolution', 'не указано')}\n"
                f"Длительность каждого видео: {approval.get('duration_seconds', '?')} сек.\n"
                f"Количество новых платных запросов: {len(retryable_ids)}\n\n"
                "Предыдущие запросы были отклонены до получения task ID. "
                "Нажатие кнопки ниже создаст новые платные задачи."
            ),
            [
                [(
                    f'✅ Подтвердить повтор всех ({len(retryable_ids)})',
                    f'cf_retry_all_video:{run_id}',
                )],
                [('Отмена', f'cf_video_status:{run_id}')],
            ],
        )

    def retry_all_video_submissions(
        self, user: TelegramUser, run_id: str
    ) -> BotResponse:
        state = self._production_store().load(run_id)
        jobs = state.get("video_jobs", {})
        retryable_ids = [
            scene_id
            for scene_id, job in sorted(jobs.items())
            if isinstance(job, dict)
            and job.get("retry_allowed")
            and not job.get("external_task_id")
        ]
        manager = self._video_job_manager()
        try:
            manager.retry_failed_submissions(run_id, retryable_ids)
            manager.submit_approved(
                run_id,
                retryable_ids,
                on_progress=self._video_submission_progress(user, run_id),
            )
        except (ProductionContractError, VideoProviderError) as exc:
            return BotResponse(
                f"Массовый повтор не выполнен. Причина: {exc}",
                [[('⏳ Статус', f'cf_video_status:{run_id}')]],
            )
        return self.video_status(user, run_id, poll=False)

    def _video_submission_progress(self, user: TelegramUser, run_id: str):
        def notify(scene_id: str, job: dict[str, Any]) -> None:
            task_id = str(job.get("external_task_id", "")).strip()
            error = self._friendly_video_error(str(job.get("error", "")))
            provider = self._video_provider_label(
                str(job.get("provider") or "polza")
            )
            if task_id:
                text = (
                    f"✅ {provider} · {run_id} · {scene_id}\n"
                    f"Задача принята. Task ID: {task_id}. Теперь бот отслеживает генерацию."
                )
            else:
                text = (
                    f"❌ {provider} · {run_id} · {scene_id}\n"
                    f"Задача не создана. Причина: {error or 'провайдер не вернул task ID'}."
                )
            try:
                self.telegram.send_message(user.chat_id, text)
            except TelegramApiError:
                pass

        return notify

    def generate_visual_draft(
        self,
        user: TelegramUser,
        run_id: str,
    ) -> BotResponse:
        run = self._require_content_run(run_id)
        if run.status != "ready_for_production":
            return BotResponse(
                (
                    "Генерация кадров откроется после завершения всех текстовых этапов: "
                    "брифа, сценария, раскадровки, промптов и проверки."
                ),
                [[('« К запуску', f'cf_run:{run.run_id}')]],
            )
        if not self.image_client.is_configured:
            return BotResponse(
                (
                    "Генератор изображений не подключён. Проверь IMAGE_PROVIDER и "
                    "доступность Codex CLI, затем перезапусти бота."
                ),
                [[('« К запуску', f'cf_run:{run.run_id}')]],
            )

        storyboard = self._content_store().read_artifact(run.run_id, "storyboard")
        prompts = self._content_store().read_artifact(run.run_id, "prompts")
        qa = self._content_store().read_artifact(run.run_id, "qa")
        prompt = build_visual_draft_prompt(run.idea, storyboard, prompts, qa)
        production = self._production_store()
        state = production.load(run.run_id)
        raw_references = state.get("references", {})
        kind_priority = {"character": 0, "environment": 1, "object": 2, "style": 3}
        storyboard_references: list[ImageReference] = []
        if isinstance(raw_references, dict):
            candidates = sorted(
                raw_references.values(),
                key=lambda item: (
                    kind_priority.get(str(item.get("kind", "")), 9),
                    str(item.get("reference_id", "")),
                ),
            )
            for item in candidates:
                if item.get("status") != "ready":
                    continue
                reference_id = str(item.get("reference_id", "")).strip()
                path = production.reference_path(run.run_id, reference_id)
                if path is None:
                    continue
                storyboard_references.append(
                    ImageReference(
                        reference_id,
                        str(item.get("kind") or "visual reference"),
                        path,
                    )
                )
                if len(storyboard_references) == 5:
                    break
        self._show_action(user.chat_id, "upload_photo")
        try:
            self.telegram.send_message(
                user.chat_id,
                (
                    f"🧩 {run.run_id}\n"
                    f"Генерирую storyboard-лист через {self.settings.openai_image_model}.\n"
                    "Это альтернативный общий лист; стандартные отдельные кадры не изменяются."
                ),
            )

        except TelegramApiError:
            pass

        try:
            image = self.image_client.generate(prompt, storyboard_references)
            filename = self._content_store().next_visual_draft_filename(
                run.run_id,
                image.extension,
            )
            version = filename.removesuffix(image.extension)
            path = self._content_store().save_visual_artifact(
                run.run_id,
                artifact_key=version,
                filename=filename,
                content=image.content,
            )
        except ImageGenerationError as exc:
            return BotResponse(
                text=(
                    f"❌ Не удалось сгенерировать storyboard-лист для {run.run_id}.\n\n"
                    f"{self._friendly_image_error(exc)}\n\n"
                    "Текстовые этапы и состояние запуска сохранены."
                ),
                keyboard=[
                    [('🔄 Повторить', f'cf_storyboard_sheet:{run.run_id}')],
                    [('« К запуску', f'cf_run:{run.run_id}')],
                ],
            )

        keyboard: InlineKeyboard = [
            [('🔄 Создать другой вариант', f'cf_storyboard_sheet:{run.run_id}')],
        ]
        keyboard.append([('« К запуску', f'cf_run:{run.run_id}')])
        try:
            self.telegram.send_photo(
                user.chat_id,
                path,
                caption=(
                    f"🧩 Storyboard-лист · {run.run_id} · {version}\n"
                    f"{self.settings.openai_image_model} · "
                    f"{self.settings.openai_image_quality} · {self.settings.openai_image_size}"
                ),
                keyboard=keyboard,
            )
        except TelegramApiError as exc:
            return BotResponse(
                text=(
                    "Изображение создано и сохранено, но Telegram не смог его отправить.\n"
                    f"Файл: {path}\nПричина: {exc}"
                ),
                keyboard=[[('« К запуску', f'cf_run:{run.run_id}')]],
            )
        return BotResponse(
            text=(
                f"✅ Storyboard-вариант {version} сохранён в запуске {run.run_id}.\n"
                "Это экспериментальная общая раскадровка по правилам Syntx. "
                "Для основной production-цепочки продолжай использовать отдельные кадры."
            ),
            keyboard=[[('« К запуску', f'cf_run:{run.run_id}')]],
        )

    def show_latest_visual_draft(
        self,
        user: TelegramUser,
        run_id: str,
    ) -> BotResponse:
        run = self._require_content_run(run_id)
        images = self._content_store().list_visual_drafts(run.run_id)
        if not images:
            return BotResponse(
                "У этого запуска ещё нет storyboard-листа.",
                [[('🧩 Создать storyboard-лист', f'cf_storyboard_sheet:{run.run_id}')]],
            )
        latest = images[-1]
        self.telegram.send_photo(
            user.chat_id,
            latest,
            caption=f"🧩 Storyboard-лист · {run.run_id} · {latest.stem}",
            keyboard=[
                [('🔄 Новый вариант', f'cf_storyboard_sheet:{run.run_id}')],
                [('« К запуску', f'cf_run:{run.run_id}')],
            ],
        )
        return BotResponse(
            f"Показал последний визуальный вариант: {latest.name}",
            [[('« К запуску', f'cf_run:{run.run_id}')]],
        )

    def diagnose_failed_run(self, run_id: str) -> BotResponse:
        """Explain a stored stage failure without editing code, files, or authentication."""

        try:
            report, target = ReadOnlyRunDiagnostic(
                self._content_store(),
                self.vault,
                self.chat_llm,
            ).diagnose(run_id)
        except Exception as exc:
            return BotResponse(
                (
                    f"Диагностика {run_id} не выполнена.\n\n"
                    f"Причина: {exc}\n\n"
                    "Файлы проекта и авторизация не изменялись."
                ),
                [[('« К запуску', f'cf_run:{run_id}')]],
            )
        return BotResponse(
            (
                f"🔍 Диагностика · {run_id}\n\n{report}\n\n"
                f"Отчёт сохранён: diagnostics/{target.name}\n"
                "Это только разбор: Codex ничего не исправлял и не менял авторизацию."
            ),
            [
                [('🧰 Подготовить исправление', f'repair_review:{run_id}')],
                [('🔄 Повторить этап', f'cf_retry:{run_id}:{self._require_content_run(run_id).current_stage}')],
                [('✏️ Уточнить задачу', f'cf_revise:{run_id}:{self._require_content_run(run_id).current_stage}')],
                [('« К запуску', f'cf_run:{run_id}')],
            ],
        )

    def review_repair(self, run_id: str) -> BotResponse:
        """Explain the branch/test gate before Codex receives write access."""

        run = self._require_content_run(run_id)
        if not self._run_has_technical_failure(run.run_id):
            return BotResponse(
                "У запуска нет сохранённой технической ошибки для ремонта.",
                [[('« К запуску', f'cf_run:{run.run_id}')]],
            )
        manager = self._repair_manager()
        if not manager.is_configured:
            return BotResponse(
                (
                    "Автоматический ремонт недоступен: Codex CLI или Git-репозиторий не найден.\n\n"
                    "OAuth и файлы авторизации бот не изменяет."
                ),
                [[('« К запуску', f'cf_run:{run.run_id}')]],
            )
        return BotResponse(
            (
                f"🧰 Подготовка ремонта · {run.run_id}\n\n"
                "Codex создаст отдельную Git-ветку и временную рабочую папку. "
                "Основной код работающего бота на этом шаге не изменится.\n\n"
                "Разрешено менять только код контент-завода, его тесты и профили агентов. "
                ".env не копируется в worktree; OAuth, токены, память запусков и исходные "
                "материалы запрещено читать или изменять.\n\n"
                "После правки бот сам запустит компиляцию, все unit-тесты и проверку diff. "
                "Перенос в основную ветку потребует отдельного подтверждения."
            ),
            [
                [('▶️ Запустить подготовку', f'repair_start:{run.run_id}')],
                [('« К запуску', f'cf_run:{run.run_id}')],
            ],
            replace_message=True,
        )

    def start_repair(self, user: TelegramUser, run_id: str) -> BotResponse:
        if not self._run_has_technical_failure(run_id):
            return BotResponse(
                "Ремонт не запущен: у запуска нет сохранённой технической ошибки.",
                [[('« К запуску', f'cf_run:{run_id}')]],
            )
        manager = self._repair_manager()
        try:
            record = manager.create(run_id)
        except Exception as exc:
            return BotResponse(
                f"Ремонт не запущен. Причина: {exc}",
                [[('« К запуску', f'cf_run:{run_id}')]],
            )
        if record.status == "ready":
            return self.repair_status(record.repair_id)

        with self._active_repair_jobs_lock:
            if record.repair_id in self._active_repair_jobs:
                return self.repair_status(record.repair_id)
            self._active_repair_jobs.add(record.repair_id)

        worker = threading.Thread(
            target=self._prepare_repair_worker,
            args=(user.chat_id, record.repair_id),
            name=f"repair-{record.repair_id}",
            daemon=True,
        )
        worker.start()
        return BotResponse(
            (
                f"🧰 {record.repair_id}: задача принята.\n\n"
                "Создаю отдельную ветку. Бот будет присылать понятные статусы по ходу работы. "
                "Текущий контент-завод продолжает работать на прежнем коде."
            ),
            [[('🔄 Проверить статус', f'repair_status:{record.repair_id}')]],
            replace_message=True,
        )

    def _prepare_repair_worker(self, chat_id: int, repair_id: str) -> None:
        manager = self._repair_manager()

        def progress(message: str) -> None:
            try:
                self.telegram.send_message(chat_id, f"🧰 {repair_id}\n{message}")
            except TelegramApiError:
                pass

        try:
            record = manager.prepare(repair_id, progress)
            self.telegram.send_message(
                chat_id,
                manager.describe(record),
                self._repair_keyboard(record),
            )
        except Exception as exc:
            try:
                record = manager.get(repair_id)
                text = manager.describe(record)
            except Exception:
                text = f"🧰 {repair_id}\nПодготовка завершилась ошибкой: {exc}"
            try:
                self.telegram.send_message(
                    chat_id,
                    text,
                    [[('« К запуску', f'cf_run:{record.run_id}')]] if 'record' in locals() else None,
                )
            except TelegramApiError:
                pass
        finally:
            with self._active_repair_jobs_lock:
                self._active_repair_jobs.discard(repair_id)

    def repair_status(
        self,
        repair_id: str,
        replace_message: bool = False,
    ) -> BotResponse:
        try:
            record = self._repair_manager().get(repair_id)
        except Exception as exc:
            return BotResponse(f"Статус ремонта недоступен: {exc}", self.main_menu_keyboard())
        return BotResponse(
            self._repair_manager().describe(record),
            self._repair_keyboard(record),
            replace_message=replace_message,
        )

    def review_repair_apply(self, repair_id: str) -> BotResponse:
        record = self._repair_manager().get(repair_id)
        if record.status != "ready":
            return self.repair_status(record.repair_id)
        return BotResponse(
            (
                f"⚠️ Применить {record.repair_id} в основную ветку?\n\n"
                "Бот проверит, что эти файлы не имеют локальных незакоммиченных изменений, "
                "перенесёт один проверенный коммит и повторно запустит все тесты.\n\n"
                "При конфликте или провале тестов перенос будет остановлен и откачен. "
                "После успеха потребуется перезапуск Telegram-бота, чтобы он загрузил новый код."
            ),
            [
                [('✅ Да, применить', f'repair_apply:{record.repair_id}')],
                [('« К ремонту', f'repair_status:{record.repair_id}')],
            ],
            replace_message=True,
        )

    def start_repair_apply(self, user: TelegramUser, repair_id: str) -> BotResponse:
        record = self._repair_manager().get(repair_id)
        if record.status != "ready":
            return self.repair_status(record.repair_id)
        with self._active_repair_jobs_lock:
            if record.repair_id in self._active_repair_jobs:
                return self.repair_status(record.repair_id)
            self._active_repair_jobs.add(record.repair_id)
        threading.Thread(
            target=self._apply_repair_worker,
            args=(user.chat_id, record.repair_id),
            name=f"apply-{record.repair_id}",
            daemon=True,
        ).start()
        return BotResponse(
            (
                f"🧰 {record.repair_id}: применение началось.\n"
                "Сначала переносится проверенный коммит, затем тесты запускаются ещё раз."
            ),
            [[('🔄 Проверить статус', f'repair_status:{record.repair_id}')]],
            replace_message=True,
        )

    def _apply_repair_worker(self, chat_id: int, repair_id: str) -> None:
        manager = self._repair_manager()

        def progress(message: str) -> None:
            try:
                self.telegram.send_message(chat_id, f"🧰 {repair_id}\n{message}")
            except TelegramApiError:
                pass

        try:
            record = manager.apply(repair_id, progress)
            self.telegram.send_message(
                chat_id,
                manager.describe(record),
                self._repair_keyboard(record),
            )
        except Exception as exc:
            try:
                record = manager.get(repair_id)
                text = manager.describe(record)
                keyboard = self._repair_keyboard(record)
            except Exception:
                text = f"🧰 {repair_id}\nПрименение завершилось ошибкой: {exc}"
                keyboard = self.main_menu_keyboard()
            try:
                self.telegram.send_message(chat_id, text, keyboard)
            except TelegramApiError:
                pass
        finally:
            with self._active_repair_jobs_lock:
                self._active_repair_jobs.discard(repair_id)

    def review_repair_discard(self, repair_id: str) -> BotResponse:
        record = self._repair_manager().get(repair_id)
        return BotResponse(
            (
                f"Удалить изолированную ветку {record.branch}?\n\n"
                "Основная ветка и файлы контент-завода не изменятся."
            ),
            [
                [('🗑 Да, отклонить ремонт', f'repair_discard:{record.repair_id}')],
                [('« К ремонту', f'repair_status:{record.repair_id}')],
            ],
            replace_message=True,
        )

    def discard_repair(self, repair_id: str) -> BotResponse:
        try:
            record = self._repair_manager().discard(repair_id)
        except Exception as exc:
            return BotResponse(
                f"Ремонт не удалён. Причина: {exc}",
                [[('« К ремонту', f'repair_status:{repair_id}')]],
            )
        return BotResponse(
            self._repair_manager().describe(record),
            [[('« К запуску', f'cf_run:{record.run_id}')]],
            replace_message=True,
        )

    @staticmethod
    def _repair_keyboard(record: RepairRecord) -> InlineKeyboard:
        keyboard: InlineKeyboard = []
        if record.status == "ready":
            keyboard.append(
                [('✅ Проверить и применить', f'repair_apply_review:{record.repair_id}')]
            )
            keyboard.append(
                [('🗑 Отклонить исправление', f'repair_discard_review:{record.repair_id}')]
            )
        elif record.status in {"queued", "preparing", "applying"}:
            keyboard.append([('🔄 Обновить статус', f'repair_status:{record.repair_id}')])
        elif record.status == "failed":
            keyboard.append([('🔄 Подготовить заново', f'repair_start:{record.run_id}')])
            keyboard.append(
                [('🗑 Удалить ветку', f'repair_discard_review:{record.repair_id}')]
            )
        elif record.status == "rolled_back":
            keyboard.append(
                [('🧰 Подготовить новый ремонт', f'repair_start:{record.run_id}')]
            )
            keyboard.append(
                [('🗑 Удалить ветку', f'repair_discard_review:{record.repair_id}')]
            )
        keyboard.append([('« К запуску', f'cf_run:{record.run_id}')])
        return keyboard

    def generate_stage(
        self,
        user: TelegramUser,
        run: ContentRun,
        revision_request: str = "",
    ) -> BotResponse:
        if not self.llm.is_configured:
            failed = self._content_store().mark_failed(
                run.run_id,
                "Codex CLI с существующей локальной авторизацией недоступен.",
            )
            return self.run_detail(failed.run_id)

        store = self._content_store()
        running = store.mark_running(run.run_id)
        stage = running.current_stage_spec
        self._show_typing(user.chat_id)
        try:
            self.telegram.send_message(
                user.chat_id,
                (
                    f"⏳ {running.run_id}\n"
                    f"Этап: {stage.title}\n"
                    f"Работает агент: {self._agent_label(stage.agent)}\n"
                    f"Что происходит: {self._stage_activity(stage.key)}\n"
                    "Когда результат будет готов, бот пришлёт его на проверку."
                ),
            )
        except TelegramApiError:
            # A transient Telegram failure must not interrupt durable production work.
            pass

        profile = self.vault.read_agent(stage.agent) or f"Роль: {stage.agent}"
        context = self.vault.context_summary(user.user_id)
        messages = build_stage_messages(
            run=running,
            stage=stage,
            project_context=context,
            agent_profile=profile,
            previous_artifacts=store.previous_artifacts(running),
            revision_request=revision_request,
        )
        try:
            artifact = self.llm.chat(messages)
            artifact = self._normalize_stage_contract(running, stage.key, artifact)
            self._validate_stage_contract(running, stage.key, artifact)
            updated = store.save_stage(running.run_id, stage.key, artifact)
        except Exception as exc:
            failed = store.mark_failed(running.run_id, str(exc))
            return BotResponse(
                text=(
                    f"❌ {failed.run_id}: этап «{stage.title}» не выполнен.\n\n"
                    f"Причина: {failed.last_error}\n\n"
                    "Состояние и предыдущие файлы сохранены."
                ),
                keyboard=[
                    [('🔍 Разобрать ошибку', f'cf_diagnose:{failed.run_id}')],
                    [('🔄 Повторить', f'cf_retry:{failed.run_id}:{failed.current_stage}')],
                    [('✏️ Уточнить задачу', f'cf_revise:{failed.run_id}:{failed.current_stage}')],
                    [('« К запуску', f'cf_run:{failed.run_id}')],
                ],
            )

        if updated.status == "ready_for_production":
            intro = (
                f"✅ {updated.run_id}: текстовый production-пакет проверен.\n"
                "Теперь можно создать отдельный файл для каждой сцены."
            )
            keyboard = [
                [('🎨 Начать генерацию кадров', f'cf_generate_visual:{updated.run_id}')],
                [('📄 Показать проверку', f'cf_view:{updated.run_id}')],
                [('📋 Все запуски', 'cf:list')],
                [('« Главное меню', 'menu:main')],
            ]
        else:
            intro = f"✅ {updated.run_id}: этап «{stage.title}» готов к твоей проверке."
            keyboard = [
                [('✅ Продолжить', f'cf_next:{updated.run_id}:{updated.current_stage}')],
                [('✏️ Доработать', f'cf_revise:{updated.run_id}:{updated.current_stage}')],
                [('📄 Показать этап', f'cf_view:{updated.run_id}')],
                [('« К запуску', f'cf_run:{updated.run_id}')],
            ]
        return BotResponse(
            text=f"{intro}\n\n{present_stage(stage.key, artifact)}",
            keyboard=keyboard,
            render_markdown=True,
        )

    def _ensure_scene_contract(self, run: ContentRun):
        store = self._content_store()
        script = store.read_artifact(run.run_id, "script")
        storyboard = store.read_artifact(run.run_id, "storyboard")
        prompts = store.read_artifact(run.run_id, "prompts")
        production = self._production_store()
        existing = production.scenes(run.run_id)
        if existing:
            scenes = merge_image_prompt_contract(existing, prompts)
        else:
            try:
                scenes = parse_scene_contract(script)
                scenes = merge_image_prompt_contract(scenes, prompts)
            except ProductionContractError:
                if not self.llm.is_configured:
                    raise ProductionContractError(
                        "Старый запуск не содержит SCENE_CONTRACT, а production LLM не подключена."
                    )
                brief = store.read_artifact(run.run_id, "brief")
                migrated = self.llm.chat(
                    build_scene_contract_messages(
                        script,
                        storyboard,
                        prompts,
                        source_context=f"IDEA:\n{run.idea}\n\nBRIEF:\n{brief}",
                    )
                )
                scenes = parse_plain_scene_json(migrated)
        references, locations = parse_reference_plan(scenes, prompts)
        bible = self._ensure_visual_bible(run, script, storyboard, parse_scene_contract(script))
        validate_english_image_prompts(scenes, references)
        validate_image_plan_against_visual_bible(bible, scenes, references, locations)
        production.save_scene_contract(run.run_id, scenes)
        production.save_reference_plan(run.run_id, references, locations)
        return scenes

    def _ensure_visual_bible(
        self,
        run: ContentRun,
        script: str,
        storyboard: str,
        scenes,
    ):
        """Migrate a legacy storyboard once and persist the validated visual contract."""

        try:
            return parse_visual_bible_contract(storyboard, scenes)
        except ProductionContractError:
            if not self.llm.is_configured:
                raise ProductionContractError(
                    "Раскадровка не содержит VISUAL_BIBLE_CONTRACT, а production LLM недоступна."
                )
            repaired = self.llm.chat(
                build_visual_bible_contract_messages(script, storyboard, scenes)
            )
            bible = parse_plain_visual_bible_json(repaired, scenes)
            readable = strip_json_contract(storyboard, "VISUAL_BIBLE_CONTRACT")
            canonical = format_visual_bible_contract(bible)
            migrated = "\n\n".join(part for part in (readable, canonical) if part)
            self._content_store().replace_stage_artifact(
                run.run_id, "storyboard", migrated
            )
            return bible

    def _normalize_stage_contract(
        self, run: ContentRun, stage_key: str, artifact: str
    ) -> str:
        """Repair an omitted machine contract without discarding readable stage prose."""

        if stage_key == "script":
            brief = self._content_store().read_artifact(run.run_id, "brief")
            source_context = f"IDEA:\n{run.idea}\n\nBRIEF:\n{brief}"
            try:
                scenes = parse_scene_contract(artifact)
                validate_script_scene_plan(scenes, source_context)
                return artifact
            except ProductionContractError:
                repaired = self.llm.chat(
                    build_scene_contract_messages(
                        artifact,
                        brief,
                        "",
                        source_context=source_context,
                    )
                )
                scenes = parse_plain_scene_json(repaired)
                validate_script_scene_plan(scenes, source_context)
                return artifact.rstrip() + "\n\n" + format_scene_contract(scenes)

        if stage_key == "storyboard":
            script = self._content_store().read_artifact(run.run_id, "script")
            scenes = parse_scene_contract(script)
            try:
                parse_visual_bible_contract(artifact, scenes)
                return artifact
            except ProductionContractError:
                repaired = self.llm.chat(
                    build_visual_bible_contract_messages(script, artifact, scenes)
                )
                bible = parse_plain_visual_bible_json(repaired, scenes)
                readable = strip_json_contract(artifact, "VISUAL_BIBLE_CONTRACT")
                canonical = format_visual_bible_contract(bible)
                return "\n\n".join(part for part in (readable, canonical) if part)

        if stage_key == "prompts":
            script = self._content_store().read_artifact(run.run_id, "script")
            storyboard = self._content_store().read_artifact(run.run_id, "storyboard")
            scenes = parse_scene_contract(script)
            bible = self._ensure_visual_bible(run, script, storyboard, scenes)
            try:
                merged = merge_image_prompt_contract(scenes, artifact)
                references, locations = parse_reference_plan(merged, artifact)
                validate_english_image_prompts(merged, references)
                validate_image_plan_against_visual_bible(
                    bible, merged, references, locations
                )
                return artifact
            except ProductionContractError:
                repaired = self.llm.chat(
                    build_image_prompt_contract_messages(scenes, artifact, bible)
                )
                merged = parse_plain_image_prompt_json(scenes, repaired)
                wrapped = (
                    "IMAGE_PROMPT_CONTRACT\n```json\n"
                    + repaired.strip()
                    + "\n```"
                )
                references, locations = parse_reference_plan(merged, wrapped)
                validate_english_image_prompts(merged, references)
                validate_image_plan_against_visual_bible(
                    bible, merged, references, locations
                )
                readable = strip_json_contract(artifact, "IMAGE_PROMPT_CONTRACT")
                canonical = format_image_prompt_contract(merged, references, locations)
                return "\n\n".join(part for part in (readable, canonical) if part)

        return artifact

    def _validate_stage_contract(
        self, run: ContentRun, stage_key: str, artifact: str
    ) -> None:
        if stage_key == "script":
            scenes = parse_scene_contract(artifact)
            brief = self._content_store().read_artifact(run.run_id, "brief")
            validate_script_scene_plan(
                scenes,
                f"IDEA:\n{run.idea}\n\nBRIEF:\n{brief}",
            )
        elif stage_key == "storyboard":
            script = self._content_store().read_artifact(run.run_id, "script")
            parse_visual_bible_contract(artifact, parse_scene_contract(script))
        elif stage_key == "prompts":
            if "IMAGE_PROMPT_CONTRACT" not in artifact:
                raise ProductionContractError(
                    "Этап prompts не содержит обязательный IMAGE_PROMPT_CONTRACT."
                )
            script = self._content_store().read_artifact(run.run_id, "script")
            script_scenes = parse_scene_contract(script)
            scenes = merge_image_prompt_contract(script_scenes, artifact)
            references, locations = parse_reference_plan(scenes, artifact)
            storyboard = self._content_store().read_artifact(run.run_id, "storyboard")
            bible = parse_visual_bible_contract(storyboard, script_scenes)
            validate_english_image_prompts(scenes, references)
            validate_image_plan_against_visual_bible(
                bible, scenes, references, locations
            )

    def _deliver_ready_videos(
        self, chat_id: int, run_id: str, state: dict[str, Any]
    ) -> int:
        jobs = state.get("video_jobs", {})
        run_path = self._production_store().runs_path / run_id
        delivered = 0
        if not isinstance(jobs, dict):
            return 0
        for scene_id, job in sorted(jobs.items()):
            relative = str(job.get("video_file", "")).strip()
            if not relative or job.get("result_delivered"):
                continue
            path = run_path / relative
            if not path.is_file():
                continue
            try:
                self.telegram.send_video(
                    chat_id,
                    path,
                    caption=f"✅ Видео готово · {run_id} · {scene_id}",
                    keyboard=[[('⏳ Все статусы', f'cf_video_status:{run_id}')]],
                )
            except TelegramApiError:
                continue
            self._video_job_manager().mark_delivered(run_id, scene_id)
            delivered += 1
        return delivered

    def _resume_pending_video_jobs(self) -> None:
        if not (
            self.polza_client.is_configured
            or self.kie_client.is_configured
            or self.viktor_client.is_configured
            or self.ltx_client.is_configured
        ):
            return
        chat_ids = sorted(self.settings.telegram_allowed_user_ids)
        for run in self._content_store().list_runs(limit=100):
            state = self._production_store().load(run.run_id)
            jobs = state.get("video_jobs", {})
            if not isinstance(jobs, dict) or not any(
                job.get("external_task_id") and not job.get("result_delivered")
                for job in jobs.values()
            ):
                continue
            previous_statuses = {
                scene_id: str(job.get("status", ""))
                for scene_id, job in jobs.items()
                if isinstance(job, dict)
            }
            try:
                refreshed = self._video_job_manager().poll_existing(run.run_id)
            except Exception:
                continue
            for chat_id in chat_ids:
                self._notify_video_status_changes(
                    chat_id,
                    run.run_id,
                    previous_statuses,
                    refreshed,
                )
                self._deliver_ready_videos(chat_id, run.run_id, refreshed)

    def _notify_video_status_changes(
        self,
        chat_id: int,
        run_id: str,
        previous_statuses: dict[str, str],
        state: dict[str, Any],
    ) -> None:
        jobs = state.get("video_jobs", {})
        if not isinstance(jobs, dict):
            return
        for scene_id, job in sorted(jobs.items()):
            if not isinstance(job, dict):
                continue
            status = str(job.get("status", ""))
            if not status or previous_statuses.get(scene_id) == status:
                continue
            if status == "completed" and job.get("video_file"):
                # The video itself is the clearest completion notification.
                continue
            text = (
                f"⏳ Видео · {run_id} · {scene_id}\n"
                f"Новый статус: {self._video_status_label(status)}."
            )
            error = self._friendly_video_error(str(job.get("error", "")))
            if error:
                text += f"\nПричина: {error}"
            keyboard: InlineKeyboard = [[('⏳ Все статусы', f'cf_video_status:{run_id}')]]
            if job.get("retry_allowed") and not job.get("external_task_id"):
                keyboard.insert(
                    0,
                    [('💳 Подготовить повтор', f'cf_retry_video:{run_id}:{scene_id}')],
                )
            try:
                self.telegram.send_message(chat_id, text, keyboard)
            except TelegramApiError:
                continue

    def _content_store(self) -> ContentFactoryStore:
        if self.content_factory is None:
            project = self.vault.get_project("content-factory")
            if project is None:
                project = self.vault.create_project(
                    "content-factory",
                    "AI-video content factory: от идеи до проверенного production-пакета.",
                )
            self.content_factory = ContentFactoryStore(project.path)
            self.content_factory.ensure()
        return self.content_factory

    def _storyboard_store(self) -> StoryboardStore:
        if self.storyboards is None:
            project = self.vault.get_project("content-factory")
            if project is None:
                project = self.vault.create_project(
                    "content-factory",
                    "AI-video content factory: от идеи до проверенного production-пакета.",
                )
            self.storyboards = StoryboardStore(project.path)
            self.storyboards.ensure()
        return self.storyboards

    def _repair_manager(self) -> GitRepairManager:
        if self.repair_manager is None:
            self.repair_manager = GitRepairManager(
                self.settings.codex_workdir,
                self._content_store(),
                codex_cli_path=self.settings.codex_cli_path,
                model=self.settings.codex_chat_model,
                timeout_seconds=max(
                    self.settings.codex_production_timeout_seconds * 2,
                    900,
                ),
            )
        return self.repair_manager

    def _run_has_technical_failure(self, run_id: str) -> bool:
        run = self._require_content_run(run_id)
        if run.status == "failed" or run.last_error:
            return True
        production = self._production_store().load(run.run_id)
        for section in ("references", "frames", "video_jobs"):
            items = production.get(section, {})
            if not isinstance(items, dict):
                continue
            for item in items.values():
                if not isinstance(item, dict):
                    continue
                if item.get("status") in {
                    "failed",
                    "submission_unknown",
                    "download_failed",
                }:
                    return True
        return False

    def _production_store(self) -> ProductionStore:
        if self.production_store is None:
            content = self._content_store()
            self.production_store = ProductionStore(content.project_path)
        return self.production_store

    def _video_job_manager(self) -> VideoJobManager:
        return VideoJobManager(
            self._production_store(),
            {
                "polza": self.polza_client,
                "kie": self.kie_client,
                "viktor": self.viktor_client,
                "ltx": self.ltx_client,
            },
        )

    def _activate_content_project(self, user_id: int) -> None:
        self._content_store()
        self.vault.set_active_project(user_id, "content-factory")

    def _require_content_run(self, run_id: str) -> ContentRun:
        run = self._content_store().get_run(run_id)
        if run is None:
            raise ValueError(f"Запуск не найден: {run_id}")
        return run

    def _parse_stage_action(self, payload: str) -> tuple[str, str]:
        run_id, separator, expected_stage = payload.partition(":")
        return run_id, expected_stage if separator else ""

    @staticmethod
    def _parse_run_scene(payload: str) -> tuple[str, str]:
        run_id, separator, scene_id = payload.partition(":")
        if not separator or not scene_id:
            raise ValueError("Некорректная кнопка сцены.")
        return run_id, scene_id

    @staticmethod
    def _parse_run_token(payload: str) -> tuple[str, str]:
        run_id, separator, token = payload.partition(":")
        if not separator or not token:
            raise ValueError("Некорректная кнопка подтверждения.")
        return run_id, token

    def _stale_stage_response(
        self,
        run_id: str,
        expected_stage: str,
    ) -> BotResponse | None:
        run = self._require_content_run(run_id)
        if expected_stage and run.current_stage != expected_stage:
            current = self.run_detail(run.run_id)
            return BotResponse(
                text=(
                    "Эта кнопка относится к уже завершённому этапу. "
                    "Показываю актуальное состояние.\n\n"
                    f"{current.text}"
                ),
                keyboard=current.keyboard,
            )
        return None

    def _show_typing(self, chat_id: int) -> None:
        self._show_action(chat_id, "typing")

    def _show_action(self, chat_id: int, action: str) -> None:
        try:
            self.telegram.send_chat_action(chat_id, action)
        except TelegramApiError:
            pass

    @staticmethod
    def _friendly_image_error(exc: ImageGenerationError) -> str:
        if exc.code in {"billing_hard_limit_reached", "billing_limit_user_error"}:
            return (
                "OpenAI заблокировал генерацию из-за лимита API-биллинга. "
                "Нужно пополнить баланс API или увеличить месячный лимит на platform.openai.com. "
                "Подписка ChatGPT не оплачивает запросы API."
            )
        if exc.code == "invalid_api_key" or exc.status_code == 401:
            return "Ключ OpenAI отклонён. Проверь OPENAI_API_KEY в .env и перезапусти бота."
        if exc.status_code == 429:
            return "OpenAI временно ограничил число запросов. Подожди немного и повтори."
        if exc.code == "codex_image_read_denied":
            return (
                "Codex создал изображение, но Windows заблокировал боту доступ к файлу. "
                "Рабочая папка и папка выдачи разделены; повтори генерацию после перезапуска бота."
            )
        return f"Сервис генерации вернул ошибку: {exc}"

    def _run_status_label(self, status: str) -> str:
        return {
            "queued": "в очереди",
            "running": "в работе",
            "waiting_approval": "ждёт проверки",
            "failed": "ошибка",
            "ready_for_production": "пакет готов",
            "cancelled": "отменён",
        }.get(status, status)

    @staticmethod
    def _stage_activity(stage_key: str) -> str:
        return {
            "brief": "собирает цель, аудиторию, формат, ограничения и рабочие допущения",
            "script": "строит хук, структуру, сцены, озвучку и проверяемый контракт производства",
            "storyboard": "раскладывает сценарий по кадрам и фиксирует визуальную связность",
            "prompts": "создаёт отдельный точный image prompt для каждого кадра",
            "qa": "сверяет весь пакет, ищет противоречия и определяет готовность к генерации",
        }.get(stage_key, "выполняет текущий производственный этап")

    @staticmethod
    def _agent_label(agent: str) -> str:
        return {
            "producer": "Продюсер и контент-стратег",
            "content-strategist": "Контент-стратег",
            "scriptwriter": "Сценарист",
            "storyboarder": "Раскадровщик",
            "prompt-engineer": "Промптер изображений",
            "qa-delivery": "Редактор и проверяющий",
            "video-prompter": "Видеопромтер",
        }.get(agent, agent)

    @staticmethod
    def _reference_kind_label(kind: str) -> str:
        return {
            "character": "карточка персонажа",
            "environment": "локация",
            "style": "стиль и свет",
            "object": "объект",
        }.get(kind, "референс")

    @staticmethod
    def _video_status_label(status: str) -> str:
        return {
            "submitting": "отправляется провайдеру",
            "pending": "в очереди",
            "processing": "генерируется",
            "completed": "готово",
            "failed": "ошибка",
            "cancelled": "отменено",
            "submission_unknown": "неясный результат отправки",
            "download_failed": "видео готово, но скачивание не удалось",
        }.get(status, status or "неизвестно")

    @staticmethod
    def _friendly_video_error(error: str) -> str:
        clean = error.strip()
        if not clean:
            return ""
        lowered = clean.lower()
        if "недоступна для данного api-ключа" in lowered:
            return (
                "Ключ PolzaAI действителен, но выбранная видеомодель недоступна для него. "
                "Открой кабинет PolzaAI -> API-ключи, разреши эту модель или создай новый "
                "ключ с доступом к ней. Затем замени ключ в локальном .env; отправлять его "
                "в Telegram не нужно."
            )
        if "http 401" in lowered:
            if "kie" in lowered:
                return (
                    "Kie отклонил API-ключ. Проверь KIE_API_KEY в локальном .env "
                    "и перезапусти бота."
                )
            if "viktor" in lowered:
                return (
                    "Viktor отклонил API-ключ. Создай новый ключ в Settings -> API Keys "
                    "с правами threads:create, runs:create, runs:read и files:read, "
                    "затем замени VIKTOR_API_KEY в локальном .env."
                )
            return (
                "PolzaAI отклонил API-ключ. Проверь или создай ключ в кабинете PolzaAI, "
                "затем замени его в локальном .env."
            )
        if "http 403" in lowered:
            if "kie" in lowered:
                return (
                    "Kie запретил эту операцию для текущего API-ключа. "
                    "Проверь права ключа и доступ к Seedance 2 в кабинете Kie."
                )
            if "viktor" in lowered:
                return (
                    "Viktor запретил операцию для текущего ключа. Проверь scopes "
                    "threads:create, runs:create, runs:read и files:read."
                )
            return (
                "PolzaAI запретил эту операцию для текущего API-ключа. "
                "Проверь права ключа и доступ к выбранной модели в кабинете PolzaAI."
            )
        if "input.duration must be a string" in clean:
            return (
                "PolzaAI отклонил длительность из-за неверного формата. "
                "Исправление уже установлено; запрос можно безопасно подготовить повторно."
            )
        return clean[:300]

    @staticmethod
    def _video_provider_label(provider: str) -> str:
        return {
            "kie": "Kie",
            "polza": "PolzaAI",
            "viktor": "Viktor",
            "ltx": "LTX worker",
        }.get(provider.strip().lower(), provider or "не указан")

    def _run_status_icon(self, status: str) -> str:
        return {
            "queued": "🕓",
            "running": "⏳",
            "waiting_approval": "👀",
            "failed": "❌",
            "ready_for_production": "✅",
            "cancelled": "⛔",
        }.get(status, "•")

    def help_text(self, user: TelegramUser) -> str:
        auth_note = ""
        if not self.settings.telegram_allowed_user_ids:
            auth_note = (
                "\n\nДоступ заблокирован: TELEGRAM_ALLOWED_USER_IDS не задан. "
                "Добавь разрешённый user_id и перезапусти бот."
            )

        active = self.vault.get_active_project(user.user_id)
        active_text = active.slug if active else "не выбран"

        return (
            "AI Content Factory\n\n"
            f"Активный проект: {active_text}\n\n"
            "Основные команды:\n"
            "/start - открыть кнопочное меню\n"
            "/factory - открыть контент-завод\n"
            "/new_video - создать новый ролик\n"
            "/runs - показать запуски роликов\n"
            "/ideas - открыть идеи радара с аналитикой\n"
            "/inbox - открыть обычные материалы партнёра\n"
            "/status - показать состояние бота\n"
            "/model - показать активную GPT-модель\n"
            "/work_status - показать технический отчёт\n"
            "/whoami - показать твой Telegram id\n\n"
            f"Единая GPT-модель: {'подключена' if self.llm.is_configured else 'не подключена'}.\n"
            "Чат, сценарий, раскадровка, промпты и QA работают через локальный Codex CLI."
            f"{auth_note}"
        )

    def status_text(self, user: TelegramUser) -> str:
        active = self.vault.get_active_project(user.user_id)
        active_text = active.slug if active else "не выбран"
        runs = self._content_store().list_runs(limit=100)
        active_runs = sum(
            run.status not in {"cancelled", "ready_for_production"}
            for run in runs
        )
        return (
            "🩺 Состояние системы\n\n"
            f"- бот: запущен\n"
            f"- vault: {self.settings.vault_path}\n"
            f"- активный проект: {active_text}\n"
            f"- единая LLM: {'подключена' if self.llm.is_configured else 'не подключена'}\n"
            f"- модель: {self.settings.codex_chat_model if self.llm.is_configured else 'нет'}\n"
            "- запуск LLM: Codex CLI, существующая локальная авторизация\n"
            f"- генерация изображений: {'подключена' if self.image_client.is_configured else 'не подключена'}\n"
            f"- image model: {self.settings.openai_image_model if self.image_client.is_configured else 'нет'}\n"
            f"- PolzaAI: {'подключён' if self.polza_client.is_configured else 'не подключён'}\n"
            f"- Kie: {'подключён' if self.kie_client.is_configured else 'не подключён'}\n"
            f"- Viktor: {'ключ загружен, проверка доступа требуется' if self.viktor_client.is_configured else 'не подключён'}\n"
            f"- LTX-2.3: {'подключён' if self.ltx_client.is_configured else 'выключен/не подключён'}\n"
            "- video model: выбирается отдельно для каждого ролика\n"
            "- платная video generation: только после кнопки подтверждения\n"
            f"- запусков контент-завода: {len(runs)}\n"
            f"- активных запусков: {active_runs}"
        )

    def model_text(self) -> str:
        if not self.chat_llm.is_configured:
            return (
                "Чат LLM пока не подключена.\n"
                "Проверь, что установленный Codex CLI доступен боту."
            )
        return (
            "Единая LLM контент-завода:\n"
            "- движок: установленный Codex CLI\n"
            f"- модель: {self.settings.codex_chat_model}\n"
            "- авторизация: существующий локальный профиль Codex CLI\n"
            "- используется для чата и всех текстовых production-этапов"
        )

    def projects_text(self, user: TelegramUser) -> str:
        projects = self.vault.list_projects()
        if not projects:
            return "Проектов пока нет. Создай первый: /new_project content-factory"

        active = self.vault.get_active_project(user.user_id)
        lines = ["Проекты:"]
        for project in projects:
            marker = "*" if active and active.slug == project.slug else "-"
            lines.append(f"{marker} {project.slug} — {project.title}")
        return "\n".join(lines)

    def create_project(self, user: TelegramUser, arg: str) -> str:
        if not arg:
            return "Укажи название: /new_project content-factory"

        project = self.vault.create_project(arg)
        self.vault.set_active_project(user.user_id, project.slug)
        return f"Создан и выбран проект: {project.slug}\nПуть: {project.path}"

    def use_project(self, user: TelegramUser, arg: str) -> str:
        if not arg:
            return "Укажи проект: /use content-factory"

        project = self.vault.set_active_project(user.user_id, arg)
        return f"Активный проект: {project.slug}"

    def remember(self, arg: str) -> str:
        if not arg:
            return "Напиши факт: /remember Я предпочитаю короткие видео с сильным визуальным хуком."
        path = self.vault.remember_global(arg)
        return f"Запомнил в глобальной памяти:\n{path}"

    def note(self, user: TelegramUser, arg: str) -> str:
        if not arg:
            return "Напиши заметку: /note Идея ролика про автоматизацию AI-видео."
        path = self.vault.add_project_note(user.user_id, arg)
        return f"Заметку сохранил:\n{path}"

    def create_task(self, user: TelegramUser, arg: str) -> str:
        if not arg:
            return "Напиши задачу: /task Описать цель проекта content-factory"
        task = self.vault.create_task(user.user_id, arg)
        return f"Задача создана: {task.task_id}\n{task.text}"

    def tasks_text(self, user: TelegramUser, arg: str) -> str:
        include_done = arg.strip().lower() in {"all", "все"}
        tasks = self.vault.list_tasks(user.user_id, include_done=include_done)
        if not tasks:
            return "Открытых задач нет." if not include_done else "Задач пока нет."

        lines = ["Задачи:"]
        for task in tasks:
            marker = "x" if task.status == "done" else " "
            lines.append(f"[{marker}] {task.task_id} — {task.text}")
        return "\n".join(lines)

    def complete_task(self, user: TelegramUser, arg: str) -> str:
        if not arg:
            return "Укажи id задачи: /done T001"
        task = self.vault.complete_task(user.user_id, arg)
        return f"Готово: {task.task_id}\n{task.text}"

    def approve_action(self, user: TelegramUser) -> str:
        return self.vault.apply_pending_action(user.user_id)

    def cancel_action(self, user: TelegramUser) -> str:
        pending = self.vault.get_pending_action(user.user_id)
        if pending is None:
            return "Нет действия для отмены."
        self.vault.clear_pending_action(user.user_id)
        return "Ок, предложенное действие отменено."

    def agents_text(self) -> str:
        agents = self.vault.list_agents()
        if not agents:
            return "Профилей агентов пока нет."
        return "Профили агентов:\n" + "\n".join(f"- {name}" for name in agents)

    def create_agent(self, arg: str) -> str:
        if not arg:
            return "Формат: /new_agent researcher | Ищет идеи, тренды и референсы для видео."

        name, _, purpose = arg.partition("|")
        path = self.vault.create_agent(name.strip(), purpose.strip())
        return f"Профиль агента создан:\n{path}"

    def read_agent(self, arg: str) -> str:
        if not arg:
            return "Укажи имя профиля: /agent general"

        content = self.vault.read_agent(arg)
        if content is None:
            return f"Профиль не найден: {arg}"
        return content

    def answer_with_llm(self, user: TelegramUser, text: str) -> BotResponse:
        context = self.vault.context_summary(user.user_id)
        messages = build_intent_messages(context=context, user_text=text)

        try:
            raw_answer = self.chat_llm.chat(messages)
            intent = parse_intent_response(raw_answer)
        except LlmError as exc:
            path = self.vault.log_conversation(user.user_id, text)
            return BotResponse(
                text=(
                    "LLM сейчас не ответила, но сообщение сохранено в журнал:\n"
                    f"{path}\n\n"
                    f"Причина: {exc}"
                )
            )
        except Exception:
            return self.answer_with_plain_llm(user, text, context)

        answer = intent["reply"]
        action = intent["action"]
        if action["type"] != "reply_only":
            self.vault.set_pending_action(user.user_id, action)
            answer = (
                f"{answer}\n\n"
                f"Предлагаю действие: {self._describe_action(action)}\n"
                f"Причина: {action.get('reason') or 'похоже, это полезно сохранить'}\n\n"
                "Подтвердить: /approve\n"
                "Отменить: /cancel"
            )

        self.vault.log_exchange(user.user_id, user_text=text, assistant_text=answer)
        return BotResponse(answer, render_markdown=True)

    def answer_with_plain_llm(
        self,
        user: TelegramUser,
        text: str,
        context: str,
    ) -> BotResponse:
        messages = build_agent_messages(context=context, user_text=text)
        try:
            answer = self.chat_llm.chat(messages)
        except LlmError as exc:
            path = self.vault.log_conversation(user.user_id, text)
            return BotResponse(
                text=(
                    "LLM сейчас не ответила, но сообщение сохранено в журнал:\n"
                    f"{path}\n\n"
                    f"Причина: {exc}"
                )
            )

        self.vault.log_exchange(user.user_id, user_text=text, assistant_text=answer)
        return BotResponse(answer, render_markdown=True)

    def _describe_action(self, action: dict[str, str]) -> str:
        action_type = action.get("type", "reply_only")
        text = action.get("text", "")
        labels = {
            "remember_global": "сохранить в глобальную память",
            "add_project_note": "сохранить заметку в активный проект",
            "create_task": "создать задачу",
        }
        return f"{labels.get(action_type, action_type)}\n{text}"

    @staticmethod
    def _is_private_chat(
        from_user: dict[str, Any],
        chat: dict[str, Any],
    ) -> bool:
        if str(chat.get("type", "")) != "private":
            return False
        try:
            return int(chat["id"]) == int(from_user["id"])
        except (KeyError, TypeError, ValueError):
            return False

    def _extract_user(self, message: dict[str, Any]) -> TelegramUser | None:
        from_user = message.get("from")
        chat = message.get("chat")
        if not from_user or not chat:
            return None
        if not self._is_private_chat(from_user, chat):
            return None

        return TelegramUser(
            user_id=int(from_user["id"]),
            chat_id=int(chat["id"]),
            first_name=from_user.get("first_name", ""),
        )

    def _is_authorized(self, user_id: int) -> bool:
        return user_id in self.settings.telegram_allowed_user_ids

    def _split_command(self, text: str) -> tuple[str, str]:
        if not text.startswith("/"):
            return "", text

        first, _, rest = text.partition(" ")
        command = first.split("@", 1)[0].lower()
        return command, rest.strip()


def main() -> None:
    settings = Settings.load()
    if not settings.telegram_bot_token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN не задан. Создай .env на основе .env.example "
            "и вставь токен от BotFather."
        )
    if not settings.telegram_allowed_user_ids:
        raise SystemExit(
            "TELEGRAM_ALLOWED_USER_IDS не задан. Основной бот работает "
            "fail-closed: добавь хотя бы один разрешённый Telegram user_id."
        )

    AgentTelegramBot(settings).run_forever()
