from __future__ import annotations

import tempfile
import threading
import unittest
import json
import re
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent_platform.config import Settings
from agent_platform.partner_bot import PartnerTelegramBot
from agent_platform.partner_research import ResearchStore
from agent_platform.image_generation import GeneratedImage
from agent_platform.production import (
    ReferenceSpec,
    SceneSpec,
    merge_image_prompt_contract,
    parse_scene_contract,
)
from agent_platform.storyboard import STORYBOARD_STAGES
from agent_platform.telegram_bot import (
    CONTENT_FACTORY_COMMANDS,
    AgentTelegramBot,
    BotResponse,
    TelegramClient,
    TelegramUser,
)
from agent_platform.video_profiles import video_profile


class FakeLlm:
    is_configured = True

    def chat(self, messages: list[dict[str, str]]) -> str:
        if "Convert the approved production package" in messages[0]["content"]:
            return json.dumps(
                {
                    "schema_version": 1,
                    "scenes": [
                        {
                            "scene_id": f"S{index:02d}",
                            "order": index,
                            "duration_seconds": 3,
                            "purpose": f"Функция {index}",
                            "visual": f"Один кадр {index}",
                            "physical_action": "Один жест",
                            "camera_movement": "static",
                            "voiceover": "",
                            "on_screen_text": "",
                            "sound": "",
                            "transition": "cut",
                            "continuity": {},
                            "image_prompt": f"Prompt {index}",
                        }
                        for index in range(1, 4)
                    ],
                }
            )
        if "Build a visual bible before framing" in messages[0]["content"]:
            return json.dumps(self._visual_payload(), ensure_ascii=False)
        if "Текущая роль: scriptwriter" in messages[0]["content"]:
            return "Сценарий\n\n" + self._contract("SCENE_CONTRACT", full=True)
        if "Текущая роль: storyboarder" in messages[0]["content"]:
            return (
                "Раскадровка\n\nVISUAL_BIBLE_CONTRACT\n```json\n"
                + json.dumps(self._visual_payload(), ensure_ascii=False)
                + "\n```"
            )
        if "Текущая роль: prompt-engineer" in messages[0]["content"]:
            return "Промпты\n\n" + self._contract("IMAGE_PROMPT_CONTRACT", full=False)
        return "Сгенерированный рабочий артефакт"

    @staticmethod
    def _contract(marker: str, *, full: bool) -> str:
        scenes = []
        for index in range(1, 4):
            item = {
                "scene_id": f"S{index:02d}",
                "image_prompt": f"One cinematic frame number {index}",
            }
            if full:
                item.update(
                    order=index,
                    duration_seconds=3,
                    purpose=f"Функция {index}",
                    visual=f"Один кадр {index}",
                    physical_action="Один жест",
                    camera_movement="static",
                    voiceover="",
                    on_screen_text="",
                    sound="",
                    transition="cut",
                    continuity={},
                )
            scenes.append(item)
        payload = {"schema_version": 1, "scenes": scenes}
        if marker == "IMAGE_PROMPT_CONTRACT":
            payload = {
                "schema_version": 2,
                "references": [],
                "locations": [{
                    "location_id": "LOC-STUDIO", "name": "Studio",
                    "description": "One continuous studio",
                    "scene_ids": ["S01", "S02", "S03"],
                    "canonical_scene_id": "S01",
                }],
                "scenes": [
                    {
                        **item,
                        "reference_ids": [],
                        "location_id": "LOC-STUDIO",
                        "location_reference_scene_id": "" if index == 1 else "S01",
                    }
                    for index, item in enumerate(scenes, 1)
                ],
            }
        return marker + "\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"

    @staticmethod
    def _visual_payload() -> dict:
        return {
            "schema_version": 1,
            "visual_basis": "One continuous cinematic studio",
            "assets": [],
            "locations": [{
                "location_id": "LOC-STUDIO", "name": "Studio",
                "description": "One continuous studio",
                "scene_ids": ["S01", "S02", "S03"],
                "canonical_scene_id": "S01",
            }],
            "frames": [
                {
                    "scene_id": f"S{index:02d}", "location_id": "LOC-STUDIO",
                    "reference_ids": [], "task": f"Task {index}",
                    "composition": "Single cinematic composition", "must_show": "Main action",
                    "constraints": "Preserve continuity", "transition": "cut",
                }
                for index in range(1, 4)
            ],
        }


class RepairingLlm(FakeLlm):
    def chat(self, messages: list[dict[str, str]]) -> str:
        if "scene_id,image_prompt" in messages[0]["content"]:
            contract = self._contract("IMAGE_PROMPT_CONTRACT", full=False)
            json_text = contract.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
            return json_text
        return super().chat(messages)


class GuidedStoryboardLlm:
    is_configured = True

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        uploaded_ids = sorted(
            set(re.findall(r'REF-UPLOAD-\d{3}', json.dumps(messages, ensure_ascii=False)))
        )
        references = [
            {
                "reference_id": "REF-CHAR-01",
                "kind": "character",
                "label": "Герой",
                "description": "Молодой герой в тёмной куртке.",
                "usage": "Сохранять лицо и одежду во всех панелях.",
            },
            {
                "reference_id": "REF-LOC-01",
                "kind": "environment",
                "label": "Квартира",
                "description": "Современная квартира с прихожей и гостиной.",
                "usage": "Сохранять планировку и направление света.",
            },
            *[
                {
                    "reference_id": reference_id,
                    "kind": "user_upload",
                    "label": "Загруженный референс",
                    "description": "Пользовательское изображение.",
                    "usage": "Сохранять визуальные признаки.",
                }
                for reference_id in uploaded_ids
            ],
        ]
        return json.dumps(
            {
                "schema_version": 2,
                "title": "Умный свет встречает героя",
                "logline": "Герой возвращается домой, и свет оживляет квартиру.",
                "duration_seconds": 15,
                "aspect_ratio": "16:9",
                "layout": {"columns": 3, "rows": 1},
                "references": references,
                "panels": [
                    {
                        "panel_id": f"P{index:02d}",
                        "order": index,
                        "timecode": f"00:{(index - 1) * 5:02d}-00:{index * 5:02d}",
                        "shot_type": ("wide", "medium", "wide")[index - 1],
                        "visual": f"Визуальное развитие истории {index}.",
                        "action": f"Действие героя {index}.",
                        "camera": "Плавное кинематографическое движение.",
                        "caption": f"Бит истории {index}",
                        "reference_ids": [
                            "REF-CHAR-01",
                            "REF-LOC-01",
                            *uploaded_ids,
                        ],
                    }
                    for index in range(1, 4)
                ],
                "sheet_prompt": (
                    "Create one single 16:9 professional storyboard sheet with exactly "
                    "three numbered panels in chronological order. Keep the same character "
                    "and apartment across all panels. Do not create separate images."
                ),
            },
            ensure_ascii=False,
        )

    def chat_with_images(
        self,
        messages: list[dict[str, str]],
        image_paths: list[Path],
    ) -> str:
        raise AssertionError("Guided Storyboard planning must use the text-only LLM path.")


class ForbiddenImageClient:
    is_configured = True

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, references=()) -> GeneratedImage:
        self.calls += 1
        raise AssertionError("Storyboard planning must not call an image provider.")


class FailingStoryboardLlm(GuidedStoryboardLlm):
    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        raise RuntimeError("temporary storyboard planning failure")


class FakeTelegram:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []
        self.keyboards: list[object] = []
        self.actions: list[tuple[int, str]] = []
        self.photos: list[Path] = []
        self.photo_keyboards: list[object] = []
        self.photo_albums: list[list[tuple[Path, str]]] = []
        self.commands: tuple[tuple[str, str], ...] = ()
        self.commands_menu_enabled = False
        self.download_content = b"\x89PNG\r\n\x1a\n" + b"telegram-reference"
        self.downloads: list[tuple[str, Path]] = []

    def download_file(
        self,
        file_id: str,
        destination: Path,
        max_bytes: int | None = None,
    ) -> Path:
        if max_bytes is not None and len(self.download_content) > max_bytes:
            raise ValueError("Telegram file exceeds the configured limit.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.download_content)
        self.downloads.append((file_id, destination))
        return destination

    def set_commands(self, commands: tuple[tuple[str, str], ...]) -> None:
        self.commands = commands

    def set_commands_menu_button(self) -> None:
        self.commands_menu_enabled = True

    def send_message(
        self,
        chat_id: int,
        text: str,
        keyboard=None,
        render_markdown: bool = False,
    ) -> None:
        self.messages.append((chat_id, text))
        self.keyboards.append(keyboard)

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self.actions.append((chat_id, action))

    def send_photo(self, chat_id: int, path: Path, caption: str, keyboard=None) -> None:
        self.photos.append(path)
        self.photo_keyboards.append(keyboard)

    def send_photo_album(
        self,
        chat_id: int,
        items: list[tuple[Path, str]],
    ) -> None:
        self.photo_albums.append(list(items))

    def send_video(self, chat_id: int, path: Path, caption: str, keyboard=None) -> None:
        pass


class FakeImageClient:
    is_configured = True

    def generate(self, prompt: str, references=()) -> GeneratedImage:
        return GeneratedImage(content=b"generated-image")


class RecordingStoryboardImageClient:
    is_configured = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], str | None]] = []

    def generate(self, prompt: str, references=(), *, size=None) -> GeneratedImage:
        self.calls.append((prompt, tuple(references), size))
        return GeneratedImage(
            content=(
                b"\x89PNG\r\n\x1a\n"
                + (13).to_bytes(4, "big")
                + b"IHDR"
                + (1536).to_bytes(4, "big")
                + (1024).to_bytes(4, "big")
                + b"\x08\x02\x00\x00\x00"
            ),
            extension=".png",
        )


def settings_for(vault_path: Path) -> Settings:
    return Settings(
        telegram_bot_token="test-token",
        telegram_allowed_user_ids={123},
        vault_path=vault_path,
        openai_api_key="",
        openai_base_url="https://api.openai.com/v1",
        openai_image_model="gpt-image-2",
        openai_image_size="1024x1536",
        openai_image_quality="low",
        deepgram_api_key="",
        deepgram_model="nova-3",
        deepgram_language="ru",
        poll_timeout_seconds=1,
        shared_content_path=vault_path / "shared-content",
    )


class ContentFactoryBotTest(unittest.TestCase):
    def test_empty_allowlist_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = replace(
                settings_for(Path(temp_dir)),
                telegram_allowed_user_ids=set(),
            )
            bot = AgentTelegramBot(settings)

            self.assertFalse(bot._is_authorized(123))

    def test_non_private_messages_are_ignored(self) -> None:
        for chat_type in ("group", "supergroup", "channel"):
            with self.subTest(chat_type=chat_type), tempfile.TemporaryDirectory() as temp_dir:
                bot = AgentTelegramBot(settings_for(Path(temp_dir)))
                telegram = FakeTelegram()
                bot.telegram = telegram

                bot.handle_update(
                    {
                        "message": {
                            "from": {"id": 123, "first_name": "Owner"},
                            "chat": {"id": -10001, "type": chat_type},
                            "text": "/start",
                        }
                    }
                )

                self.assertEqual([], telegram.messages)
                self.assertEqual([], telegram.actions)

    def test_non_private_callback_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            telegram = FakeTelegram()
            answered: list[str] = []
            telegram.answer_callback_query = answered.append  # type: ignore[attr-defined]
            bot.telegram = telegram

            bot.handle_callback_query(
                {
                    "id": "callback-1",
                    "from": {"id": 123, "first_name": "Owner"},
                    "message": {
                        "chat": {"id": -10001, "type": "supergroup"},
                    },
                    "data": "input:cancel",
                }
            )

            self.assertEqual([], answered)
            self.assertEqual([], telegram.messages)

    def test_ready_run_does_not_offer_experimental_storyboard_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            store = bot._content_store()
            run = store.create_run("Вертикальный ролик на 10 секунд")
            for stage_key in ("brief", "script", "storyboard", "prompts", "qa"):
                store.mark_running(run.run_id)
                run = store.save_stage(
                    run.run_id,
                    stage_key,
                    f"Approved {stage_key} artifact",
                )
                if run.status == "waiting_approval":
                    run = store.advance(run.run_id)

            detail = bot.run_detail(run.run_id)
            callbacks = [
                callback
                for row in detail.keyboard
                for _, callback in row
            ]
            self.assertNotIn(f"cf_storyboard_sheet:{run.run_id}", callbacks)
            self.assertFalse(any("storyboard" in callback for callback in callbacks))

    def test_storyboard_entry_opens_separate_workflow_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot = AgentTelegramBot(settings_for(Path(tmp)))
            user = TelegramUser(user_id=42, chat_id=42, first_name="Owner")

            command_response = bot.dispatch(user, "/storyboard")
            menu_response = bot.dispatch_callback(user, "cf:storyboards")
            self.assertIsInstance(command_response, BotResponse)
            self.assertIsInstance(menu_response, BotResponse)
            assert isinstance(command_response, BotResponse)
            assert isinstance(menu_response, BotResponse)

            self.assertIn("🧩 Storyboard", command_response.text)
            self.assertIn("автоматический разбор истории", command_response.text)
            self.assertIn("техническую анкету заполнять не нужно", command_response.text)
            self.assertIn("платные запросы из Storyboard не запускаются", command_response.text)
            self.assertIn("AI Content Factory Storyboard Template", command_response.text)
            self.assertEqual(command_response.text, menu_response.text)
            callbacks = [
                callback
                for row in menu_response.keyboard or []
                for _, callback in row
            ]
            self.assertIn("sb:new", callbacks)
            self.assertIn("sb:list", callbacks)
            self.assertTrue(menu_response.replace_message)

    def test_storyboard_one_input_builds_guided_plan_preview_without_provider_calls(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bot = AgentTelegramBot(settings_for(root))
            planning_llm = GuidedStoryboardLlm()
            forbidden_image_client = ForbiddenImageClient()
            bot.telegram = FakeTelegram()
            bot.llm = planning_llm
            bot.image_client = forbidden_image_client
            user = TelegramUser(user_id=42, chat_id=42, first_name="Owner")

            create_prompt = bot.dispatch_callback(user, "sb:new")
            self.assertIsInstance(create_prompt, BotResponse)
            state = bot.vault.get_input_state(user.user_id) or {}
            self.assertEqual(state.get("kind"), "storyboard_create")

            preview = bot.handle_input_state(
                user,
                "Ролик о герое, которого дома встречает умный свет.",
                state,
            )

            projects = bot._storyboard_store().list_projects()
            self.assertEqual(len(projects), 1)
            project = projects[0]
            self.assertEqual(project.workflow, "guided-v2")
            self.assertEqual(project.status, "plan_review")
            self.assertEqual(len(planning_llm.calls), 1)
            self.assertEqual(forbidden_image_client.calls, 0)
            self.assertIsNone(bot.production_store)
            self.assertIsNone(bot.vault.get_input_state(user.user_id))

            plan = bot._storyboard_store().read_plan(project.project_id)
            self.assertEqual([panel["panel_id"] for panel in plan["panels"]], ["P01", "P02", "P03"])
            self.assertIn("один общий storyboard sheet", preview.text)
            self.assertIn("P01", preview.text)
            self.assertIn("P03", preview.text)
            self.assertNotIn("Заполнить этап", preview.text)
            callbacks = [
                callback
                for row in preview.keyboard or []
                for _, callback in row
            ]
            self.assertIn(f"sb:plan_approve:{project.project_id}", callbacks)
            self.assertIn(f"sb:refs:{project.project_id}", callbacks)
            self.assertIn(f"sb:plan_revise:{project.project_id}", callbacks)
            self.assertIn(f"sb:plan_reject:{project.project_id}", callbacks)
            self.assertFalse(any(callback.startswith("sb:fill:") for callback in callbacks))

    def test_storyboard_planning_failure_resumes_same_project_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user = TelegramUser(user_id=123, chat_id=123, first_name="Owner")
            forbidden_image_client = ForbiddenImageClient()
            first = AgentTelegramBot(settings_for(root))
            first.telegram = FakeTelegram()  # type: ignore[assignment]
            first.llm = FailingStoryboardLlm()
            first.image_client = forbidden_image_client
            first.dispatch_callback(user, "sb:new")
            state = first.vault.get_input_state(user.user_id) or {}

            with self.assertRaisesRegex(RuntimeError, "temporary storyboard"):
                first.handle_input_state(user, "Идея, которая переживёт restart", state)

            projects = first._storyboard_store().list_projects()
            self.assertEqual(len(projects), 1)
            project_id = projects[0].project_id
            self.assertEqual(projects[0].status, "planning")
            self.assertEqual(projects[0].pending_operation, "initial")
            recovery_state = first.vault.get_input_state(user.user_id) or {}
            self.assertEqual(recovery_state.get("kind"), "storyboard_planning")
            self.assertEqual(recovery_state.get("project_id"), project_id)

            restarted = AgentTelegramBot(settings_for(root))
            restarted.telegram = FakeTelegram()  # type: ignore[assignment]
            restarted.llm = GuidedStoryboardLlm()
            restarted.image_client = forbidden_image_client
            resumed = restarted.dispatch_callback(user, f"sb:plan_retry:{project_id}")

            self.assertIsInstance(resumed, BotResponse)
            projects = restarted._storyboard_store().list_projects()
            self.assertEqual(len(projects), 1)
            self.assertEqual(projects[0].project_id, project_id)
            self.assertEqual(projects[0].status, "plan_review")
            self.assertIsNone(restarted.vault.get_input_state(user.user_id))
            self.assertEqual(forbidden_image_client.calls, 0)

    def test_storyboard_revision_failure_restores_director_note_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user = TelegramUser(user_id=123, chat_id=123, first_name="Owner")
            first = AgentTelegramBot(settings_for(root))
            first.telegram = FakeTelegram()  # type: ignore[assignment]
            first.llm = FailingStoryboardLlm()
            store = first._storyboard_store()
            project = store.create_guided_project("История для recovery правки")
            project = store.save_generated_plan(project.project_id, GuidedStoryboardLlm().chat([]))
            first.dispatch_callback(user, f"sb:plan_revise:{project.project_id}")
            state = first.vault.get_input_state(user.user_id) or {}
            director_note = "Ускорить финал и закончить крупным планом героя."

            with self.assertRaisesRegex(RuntimeError, "temporary storyboard"):
                first.handle_input_state(user, director_note, state)

            failed = store.get_project(project.project_id)
            assert failed is not None
            self.assertEqual(failed.status, "planning")
            self.assertEqual(failed.pending_operation, "revision")
            self.assertEqual(failed.storyboard_revision_count, 1)

            restarted = AgentTelegramBot(settings_for(root))
            telegram = FakeTelegram()
            planning_llm = GuidedStoryboardLlm()
            restarted.telegram = telegram  # type: ignore[assignment]
            restarted.llm = planning_llm
            resumed = restarted.dispatch_callback(
                user,
                f"sb:plan_retry:{project.project_id}",
            )

            self.assertIsInstance(resumed, BotResponse)
            self.assertEqual(len(planning_llm.calls), 1)
            prompt_text = json.dumps(planning_llm.calls[0], ensure_ascii=False)
            self.assertIn(director_note, prompt_text)
            restored = restarted._storyboard_store().get_project(project.project_id)
            assert restored is not None
            self.assertEqual(restored.status, "plan_review")
            self.assertEqual(restored.storyboard_revision_count, 1)
            self.assertIsNone(restarted.vault.get_input_state(user.user_id))

    def test_guided_plan_review_actions_are_real_and_keep_phase_two_locked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bot = AgentTelegramBot(settings_for(root))
            bot.telegram = FakeTelegram()  # type: ignore[assignment]
            bot.llm = GuidedStoryboardLlm()
            forbidden_image_client = ForbiddenImageClient()
            bot.image_client = forbidden_image_client
            user = TelegramUser(user_id=42, chat_id=42, first_name="Owner")
            store = bot._storyboard_store()

            approved = store.create_guided_project("План для подтверждения")
            approved = store.save_generated_plan(approved.project_id, GuidedStoryboardLlm().chat([]))
            response = bot.dispatch_callback(user, f"sb:plan_approve:{approved.project_id}")
            self.assertIsInstance(response, BotResponse)
            approved = store.get_project(approved.project_id)
            assert approved is not None
            self.assertEqual(approved.status, "plan_approved")
            self.assertEqual(approved.storyboard_approved_at, "")
            self.assertIn("Платная генерация и Phase 2 пока заблокированы", response.text)

            revised = store.create_guided_project("План для правки")
            revised = store.save_generated_plan(revised.project_id, GuidedStoryboardLlm().chat([]))
            request_prompt = bot.dispatch_callback(user, f"sb:plan_revise:{revised.project_id}")
            self.assertIsInstance(request_prompt, BotResponse)
            state = bot.vault.get_input_state(user.user_id) or {}
            self.assertEqual(state.get("kind"), "storyboard_plan_revision")
            revised_response = bot.handle_input_state(
                user,
                "Сделай финальную панель крупнее и динамичнее.",
                state,
            )
            revised = store.get_project(revised.project_id)
            assert revised is not None
            self.assertEqual(revised.status, "plan_review")
            self.assertEqual(revised.storyboard_revision_count, 1)
            self.assertEqual(revised.storyboard_approved_at, "")
            self.assertIn("обновлён", revised_response.text)

            rejected = store.create_guided_project("План для отклонения")
            rejected = store.save_generated_plan(rejected.project_id, GuidedStoryboardLlm().chat([]))
            rejection_prompt = bot.dispatch_callback(user, f"sb:plan_reject:{rejected.project_id}")
            self.assertIsInstance(rejection_prompt, BotResponse)
            state = bot.vault.get_input_state(user.user_id) or {}
            self.assertEqual(state.get("kind"), "storyboard_plan_rejection")
            rejected_response = bot.handle_input_state(user, "Не подходит идея.", state)
            rejected = store.get_project(rejected.project_id)
            assert rejected is not None
            self.assertEqual(rejected.status, "rejected")
            self.assertEqual(rejected.storyboard_approved_at, "")
            self.assertIn("отклонён", rejected_response.text)
            self.assertEqual(forbidden_image_client.calls, 0)

    def test_storyboard_sheet_requires_quote_confirmation_before_single_generation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user = TelegramUser(user_id=123, chat_id=123, first_name="Owner")
            image_client = RecordingStoryboardImageClient()
            first = AgentTelegramBot(settings_for(root))
            first.telegram = FakeTelegram()  # type: ignore[assignment]
            first.image_client = image_client
            store = first._storyboard_store()
            project = store.create_guided_project("История для единого storyboard sheet")
            project = store.save_generated_plan(project.project_id, GuidedStoryboardLlm().chat([]))

            approved = first.dispatch_callback(user, f"sb:plan_approve:{project.project_id}")
            assert isinstance(approved, BotResponse)
            approved_callbacks = [
                callback
                for row in approved.keyboard or []
                for _, callback in row
            ]
            self.assertIn(f"sb:sheet_prepare:{project.project_id}", approved_callbacks)
            self.assertEqual(image_client.calls, [])

            quote = first.dispatch_callback(user, f"sb:sheet_prepare:{project.project_id}")
            assert isinstance(quote, BotResponse)
            self.assertIn("Codex", quote.text)
            self.assertIn("gpt-5.6-sol", quote.text)
            self.assertNotIn("gpt-image-2", quote.text)
            self.assertIn("1536x1024", quote.text)
            self.assertIn("0 ₽", quote.text)
            quote_callbacks = [
                callback
                for row in quote.keyboard or []
                for _, callback in row
            ]
            self.assertIn(f"sb:sheet_confirm:{project.project_id}", quote_callbacks)
            self.assertEqual(image_client.calls, [])

            restarted = AgentTelegramBot(settings_for(root))
            telegram = FakeTelegram()
            restarted.telegram = telegram  # type: ignore[assignment]
            restarted.image_client = image_client
            restored_quote = restarted.storyboard_project_detail(project.project_id)
            self.assertIn("0 ₽", restored_quote.text)

            generated = restarted.dispatch_callback(
                user,
                f"sb:sheet_confirm:{project.project_id}",
            )
            assert isinstance(generated, BotResponse)
            self.assertEqual(len(image_client.calls), 1)
            self.assertIn("one single 16:9 storyboard sheet", image_client.calls[0][0])
            self.assertEqual(image_client.calls[0][2], "1536x1024")
            saved = store.get_project(project.project_id)
            assert saved is not None
            self.assertEqual(saved.status, "sheet_review")
            _, saved_result = store.latest_generated_sheet(project.project_id)
            self.assertEqual((saved_result["width"], saved_result["height"]), (1536, 1024))
            self.assertEqual(len(telegram.photos), 1)
            self.assertTrue(telegram.photos[0].is_file())
            review_callbacks = [
                callback
                for keyboard in telegram.photo_keyboards
                for row in keyboard or []
                for _, callback in row
            ]
            self.assertIn(f"sb:sheet_approve:{project.project_id}", review_callbacks)
            self.assertIn(f"sb:sheet_revise:{project.project_id}", review_callbacks)
            self.assertIn(f"sb:sheet_reject:{project.project_id}", review_callbacks)

            stale = restarted.dispatch_callback(
                user,
                f"sb:sheet_confirm:{project.project_id}",
            )
            assert isinstance(stale, BotResponse)
            self.assertEqual(len(image_client.calls), 1)
            self.assertIn("актуальное состояние", stale.text)

            approved_sheet = restarted.dispatch_callback(
                user,
                f"sb:sheet_approve:{project.project_id}",
            )
            assert isinstance(approved_sheet, BotResponse)
            self.assertIn("Phase 1 завершена", approved_sheet.text)
            saved = store.get_project(project.project_id)
            assert saved is not None
            self.assertEqual(saved.status, "sheet_approved")
            self.assertEqual(saved.storyboard_approved_at, "")
            with self.assertRaisesRegex(ValueError, "не готов"):
                store.approve_storyboard(project.project_id)

            shown = restarted.dispatch_callback(
                user,
                f"sb:sheet_show:{project.project_id}",
            )
            assert isinstance(shown, BotResponse)
            self.assertIn("отправлен отдельным изображением", shown.text)
            self.assertEqual(len(telegram.photos), 2)
            self.assertEqual(len(image_client.calls), 1)

    def test_storyboard_reference_photo_is_saved_and_replans_through_main_router(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bot = AgentTelegramBot(settings_for(root))
            telegram = FakeTelegram()
            planning_llm = GuidedStoryboardLlm()
            forbidden_image_client = ForbiddenImageClient()
            bot.telegram = telegram  # type: ignore[assignment]
            bot.llm = planning_llm
            bot.image_client = forbidden_image_client
            user = TelegramUser(user_id=123, chat_id=123, first_name="Owner")
            store = bot._storyboard_store()
            project = store.create_guided_project("История с референсом героя")
            project = store.save_generated_plan(project.project_id, planning_llm.chat([]))

            prompt = bot.dispatch_callback(user, f"sb:refs:{project.project_id}")
            self.assertIsInstance(prompt, BotResponse)
            self.assertEqual(
                (bot.vault.get_input_state(user.user_id) or {}).get("kind"),
                "storyboard_reference",
            )

            bot.handle_update(
                {
                    "message": {
                        "message_id": 101,
                        "from": {"id": 123, "first_name": "Owner"},
                        "chat": {"id": 123, "type": "private"},
                        "photo": [
                            {"file_id": "unsafe/path/photo-small"},
                            {"file_id": "unsafe/path/photo-large"},
                        ],
                        "caption": "Главный герой: сохранить лицо и тёмную куртку.",
                    }
                }
            )

            uploaded = store.list_uploaded_references(project.project_id)
            self.assertEqual(len(uploaded), 1)
            self.assertEqual(uploaded[0]["reference_id"], "REF-UPLOAD-001")
            self.assertEqual(uploaded[0]["filename"], "REF-UPLOAD-001.png")
            self.assertNotIn("unsafe", uploaded[0]["filename"])
            self.assertEqual(len(telegram.downloads), 1)
            self.assertEqual(telegram.downloads[0][0], "unsafe/path/photo-large")
            self.assertGreaterEqual(len(planning_llm.calls), 2)
            planning_payload = json.dumps(planning_llm.calls[-1], ensure_ascii=False)
            self.assertIn("REF-UPLOAD-001", planning_payload)
            self.assertNotIn("REF-UPLOAD-001.png", planning_payload)
            self.assertNotIn(uploaded[0]["sha256"], planning_payload)
            self.assertNotIn('"bytes"', planning_payload)
            plan = store.read_plan(project.project_id)
            self.assertIn(
                "REF-UPLOAD-001",
                [item["reference_id"] for item in plan["references"]],
            )
            project = store.get_project(project.project_id)
            assert project is not None
            self.assertEqual(project.status, "plan_review")
            self.assertEqual(project.pending_operation, "")
            self.assertEqual(project.storyboard_approved_at, "")
            self.assertIsNone(bot.vault.get_input_state(user.user_id))
            self.assertIn("Референс сохранён", telegram.messages[-1][1])
            self.assertNotIn("Пока обрабатываю только текст", telegram.messages[-1][1])
            self.assertEqual(forbidden_image_client.calls, 0)

    def test_storyboard_reference_form_keeps_waiting_when_text_is_sent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bot = AgentTelegramBot(settings_for(root))
            telegram = FakeTelegram()
            bot.telegram = telegram  # type: ignore[assignment]
            user = TelegramUser(user_id=123, chat_id=123, first_name="Owner")
            store = bot._storyboard_store()
            project = store.create_guided_project("История для формы референса")
            project = store.save_generated_plan(project.project_id, GuidedStoryboardLlm().chat([]))
            bot.dispatch_callback(user, f"sb:refs:{project.project_id}")

            bot.handle_update(
                {
                    "message": {
                        "message_id": 102,
                        "from": {"id": 123, "first_name": "Owner"},
                        "chat": {"id": 123, "type": "private"},
                        "text": "Это описание, но фото я забыл приложить.",
                    }
                }
            )

            state = bot.vault.get_input_state(user.user_id) or {}
            self.assertEqual(state.get("kind"), "storyboard_reference")
            self.assertEqual(state.get("project_id"), project.project_id)
            self.assertIn("пришли именно фото", telegram.messages[-1][1].casefold())
            self.assertEqual(store.list_uploaded_references(project.project_id), [])

    def test_existing_manual_storyboard_cycle_remains_compatible_without_provider_calls(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bot = AgentTelegramBot(settings_for(root))
            user = TelegramUser(user_id=42, chat_id=42, first_name="Owner")
            project = bot._storyboard_store().create_project(
                "Ролик о герое, которого дома встречает умный свет."
            )
            project_id = project.project_id
            project_response = bot.storyboard_project_detail(project_id)

            self.assertIn(project_id, project_response.text)
            self.assertEqual(project.workflow, "manual-v1")
            self.assertIsNone(bot.production_store)

            for stage in STORYBOARD_STAGES[1:]:
                fill_response = bot.dispatch_callback(
                    user,
                    f"sb:fill:{project_id}:{stage.key}",
                )
                self.assertIn(stage.title, fill_response.text)
                state = bot.vault.get_input_state(user.user_id) or {}
                self.assertEqual(state.get("kind"), "storyboard_stage")
                self.assertEqual(state.get("stage_key"), stage.key)
                project_response = bot.handle_input_state(
                    user,
                    f"Материал ручного этапа {stage.key}",
                    state,
                )
                if stage.key == "storyboard_result":
                    self.assertIn("ждёт твоего подтверждения", project_response.text)
                    gate_callbacks = [
                        callback
                        for row in project_response.keyboard or []
                        for _, callback in row
                    ]
                    self.assertIn(f"sb:approve:{project_id}", gate_callbacks)
                    self.assertIn(f"sb:revise:{project_id}", gate_callbacks)
                    self.assertIn(f"sb:reject:{project_id}", gate_callbacks)

                    revision_prompt = bot.dispatch_callback(
                        user,
                        f"sb:revise:{project_id}",
                    )
                    self.assertIn("что нужно исправить", revision_prompt.text)
                    revision_state = bot.vault.get_input_state(user.user_id) or {}
                    self.assertEqual(
                        revision_state.get("kind"),
                        "storyboard_revision_request",
                    )
                    bot.handle_input_state(
                        user,
                        "Исправить направление взгляда персонажа.",
                        revision_state,
                    )
                    revised_fill = bot.dispatch_callback(
                        user,
                        f"sb:fill:{project_id}:storyboard_result",
                    )
                    self.assertIn("Storyboard sheet", revised_fill.text)
                    revised_state = bot.vault.get_input_state(user.user_id) or {}
                    project_response = bot.handle_input_state(
                        user,
                        "Исправленный ручной storyboard result",
                        revised_state,
                    )
                    self.assertIn("ждёт твоего подтверждения", project_response.text)

                    approval_response = bot.dispatch_callback(
                        user,
                        f"sb:approve:{project_id}",
                    )
                    self.assertIn("Storyboard подтверждён", approval_response.text)
                    self.assertIn("Cinematic video prompt", approval_response.text)

            self.assertIn("готов к завершению", project_response.text)
            completed_response = bot.dispatch_callback(
                user,
                f"sb:complete:{project_id}",
            )
            self.assertIn("Эксперимент завершён", completed_response.text)
            self.assertIsNone(bot.production_store)

            restarted = AgentTelegramBot(settings_for(root))
            restored_response = restarted.dispatch_callback(user, "sb:list")
            self.assertIn(project_id, restored_response.text)
            self.assertIn("Завершён", restored_response.text)

    def test_leaving_storyboard_form_clears_hidden_input_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot = AgentTelegramBot(settings_for(Path(tmp)))
            user = TelegramUser(user_id=42, chat_id=42, first_name="Owner")

            bot.dispatch_callback(user, "sb:new")
            self.assertIsNotNone(bot.vault.get_input_state(user.user_id))

            bot.dispatch_callback(user, "menu:main")
            self.assertIsNone(bot.vault.get_input_state(user.user_id))

            bot.dispatch_callback(user, "sb:new")
            self.assertIsNotNone(bot.vault.get_input_state(user.user_id))
            bot.dispatch_callback(user, "cf:list")

            self.assertIsNone(bot.vault.get_input_state(user.user_id))

    def test_telegram_photo_album_preserves_order_and_chunks_at_ten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            items: list[tuple[Path, str]] = []
            for index in range(12):
                path = root / f"frame-{index + 1:02d}.png"
                path.write_bytes(b"image")
                items.append((path, f"Кадр {index + 1}"))

            client = TelegramClient("test-token")
            requests: list[tuple[str, dict[str, str], list[tuple[str, str, bytes, str]]]] = []

            def record_request(method, *, fields, files):
                requests.append((method, fields, files))
                return {"ok": True, "result": []}

            client._request_multipart_files = record_request  # type: ignore[method-assign]
            client.send_photo_album(456, items)

            self.assertEqual([len(request[2]) for request in requests], [10, 2])
            self.assertTrue(all(request[0] == "sendMediaGroup" for request in requests))
            self.assertEqual(
                [
                    file_info[1]
                    for _, _, files in requests
                    for file_info in files
                ],
                [path.name for path, _ in items],
            )

    def test_frame_gallery_sends_one_album_and_absorbs_repeated_taps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            telegram = FakeTelegram()
            bot.telegram = telegram
            user = TelegramUser(user_id=123, chat_id=456, first_name="Owner")
            run = bot._content_store().create_run("Album without repeated sends")
            production = bot._production_store()
            scenes = [
                SceneSpec(
                    scene_id=f"S{index:02d}",
                    order=index,
                    duration_seconds=2,
                    purpose=f"purpose {index}",
                    visual=f"visual {index}",
                    physical_action="action",
                    camera_movement="static",
                    voiceover="",
                    on_screen_text="",
                    sound="",
                    transition="cut",
                    continuity={},
                    image_prompt=f"prompt {index}",
                )
                for index in range(1, 8)
            ]
            production.save_scene_contract(run.run_id, scenes)
            for scene in scenes:
                attempt_id = production.start_frame(run.run_id, scene.scene_id)
                production.complete_frame(
                    run.run_id,
                    scene.scene_id,
                    attempt_id,
                    f"frame-{scene.scene_id}".encode("ascii"),
                    ".png",
                )

            first = bot.show_frames(user, run.run_id)
            duplicate = bot.show_frames(user, run.run_id)

            self.assertIsNotNone(first)
            self.assertTrue(first.replace_message)  # type: ignore[union-attr]
            self.assertIsNone(duplicate)
            self.assertEqual(len(telegram.photo_albums), 1)
            self.assertEqual(len(telegram.photo_albums[0]), 7)
            self.assertEqual(
                [path.parent.name for path, _ in telegram.photo_albums[0]],
                [scene.scene_id for scene in scenes],
            )
            callbacks = [
                callback
                for row in first.keyboard or []  # type: ignore[union-attr]
                for _, callback in row
            ]
            for scene in scenes:
                self.assertIn(
                    f"cf_toggle_frame:{run.run_id}:{scene.scene_id}",
                    callbacks,
                )
                self.assertIn(
                    f"cf_retry_frame:{run.run_id}:{scene.scene_id}",
                    callbacks,
                )

    def test_startup_menu_contains_only_primary_content_factory_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            telegram = FakeTelegram()
            bot.telegram = telegram

            bot._configure_telegram_command_menu()

            self.assertEqual(telegram.commands, CONTENT_FACTORY_COMMANDS)
            self.assertTrue(telegram.commands_menu_enabled)
            self.assertEqual(
                [command for command, _ in telegram.commands],
                [
                    "start",
                    "new_video",
                    "runs",
                    "radar",
                    "ideas",
                    "inbox",
                    "factory",
                    "status",
                    "help",
                ],
            )

    def test_main_menu_is_scoped_to_content_factory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))

            labels = [label for row in bot.main_menu_keyboard() for label, _ in row]

            self.assertIn("🎬 Новый ролик", labels)
            self.assertIn("🧩 Storyboard", labels)
            self.assertIn("📋 Мои ролики", labels)
            self.assertIn("🎥 Генерации видео", labels)
            self.assertIn("💡 Идеи радара", labels)
            self.assertIn("📥 Файлы партнёра", labels)
            self.assertIn("🩺 Состояние системы", labels)
            self.assertFalse(any("Задачи" in label for label in labels))
            self.assertFalse(any("Контекст" in label for label in labels))

    def test_configured_radar_is_available_from_main_bot_menu_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preexisting_store = ResearchStore(root / "radar" / "research")
            preexisting_store.ensure()
            preexisting_store.import_accounts("youtube:@saved-channel")
            settings = replace(
                settings_for(root / "vault"),
                radar_vault_path=root / "radar",
                youtube_api_key="test-youtube-key",
            )
            bot = AgentTelegramBot(settings)
            self.assertIsNotNone(bot.radar)
            assert bot.radar is not None

            menu_callbacks = [
                callback
                for row in bot.main_menu_keyboard()
                for _, callback in row
            ]
            response = bot.dispatch(
                TelegramUser(user_id=123, chat_id=123, first_name="Owner"),
                "/radar",
            )
            self.assertIn("radar:home", menu_callbacks)
            self.assertIsInstance(response, BotResponse)
            assert isinstance(response, BotResponse)
            self.assertIn("Радар идей", response.text)
            self.assertTrue(response.render_markdown)
            self.assertIn("##", response.text)
            self.assertIn("1 каналов", response.text)
            response_callbacks = [
                callback
                for row in response.keyboard
                for _, callback in row
            ]
            self.assertIn("auto:youtube", response_callbacks)
            self.assertIn("menu:main", response_callbacks)

            empty_ideas = bot.ideas_inbox()
            empty_callbacks = [
                callback
                for row in empty_ideas.keyboard or []
                for _, callback in row
            ]
            self.assertIn("в этом боте", empty_ideas.text)
            self.assertNotIn("боте партнёра", empty_ideas.text)
            self.assertIn("radar:home", empty_callbacks)

            deep_link = bot.dispatch(
                TelegramUser(user_id=123, chat_id=123, first_name="Owner"),
                "/start radar",
            )
            assert isinstance(deep_link, BotResponse)
            self.assertIn("Радар идей", deep_link.text)

            discover = bot.dispatch(
                TelegramUser(user_id=123, chat_id=123, first_name="Owner"),
                "/discover",
            )
            research = bot.dispatch(
                TelegramUser(user_id=123, chat_id=123, first_name="Owner"),
                "/research",
            )
            results = bot.dispatch(
                TelegramUser(user_id=123, chat_id=123, first_name="Owner"),
                "/results",
            )
            assert isinstance(discover, BotResponse)
            assert isinstance(research, BotResponse)
            assert isinstance(results, BotResponse)
            self.assertIn("Радар идей", discover.text)
            self.assertIn("Референсы и каналы", research.text)
            self.assertIn("поиск", results.text.casefold())

    def test_radar_rejects_legacy_and_overlapping_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = settings_for(root)
            shared_path = base.shared_content_path
            assert shared_path is not None
            forbidden = (
                base.vault_path,
                base.vault_path.parent / "vault-partner",
                base.vault_path.parent / "vault-partner-test",
                shared_path,
                base.vault_path.parent,
            )

            for radar_path in forbidden:
                with self.subTest(radar_path=radar_path):
                    with self.assertRaisesRegex(ValueError, "dedicated runtime"):
                        AgentTelegramBot(
                            replace(base, radar_vault_path=radar_path)
                        )

    def test_main_radar_startup_does_not_recover_hidden_partner_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            assert bot.radar is not None
            bot.radar.research.ensure()
            image = bot.radar.research.create_image("Hidden Partner image", 123)
            bot.radar.research.update_image(image.asset_id, "generating")
            run = bot.radar.research.create_run(
                "youtube",
                "",
                123,
                workflow="auto_content",
            )
            bot.radar.research.update_run(run.run_id, "running")
            bot.radar.research.save_results(
                run.run_id,
                [
                    {
                        "platform": "youtube",
                        "external_id": "saved-before-restart",
                        "source_url": "https://youtube.example/saved-before-restart",
                        "title": "Saved material",
                    }
                ],
            )

            bot.radar._prepare_runtime()

            self.assertEqual(
                bot.radar.research.require_image(image.asset_id).status,
                "generating",
            )
            self.assertEqual(
                bot.radar.research.require_run(run.run_id).status,
                "failed",
            )

    def test_main_bot_routes_radar_form_input_to_radar_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            telegram = FakeTelegram()
            bot.telegram = telegram
            self.assertIsNotNone(bot.radar)
            assert bot.radar is not None
            bot.radar.vault.ensure_bootstrap()
            bot.radar.vault.set_input_state(
                123,
                {"kind": "research_query", "provider": "youtube"},
            )
            user_message = {
                "message": {
                    "message_id": 1,
                    "from": {"id": 123, "first_name": "Owner"},
                    "chat": {"id": 123, "type": "private"},
                    "text": "AI animation",
                }
            }

            with patch.object(
                bot.radar,
                "prepare_research",
                return_value=BotResponse("RADAR_INPUT_HANDLED"),
            ) as prepare:
                bot.handle_update(user_message)

            prepare.assert_called_once()
            self.assertEqual(telegram.messages[-1][1], "RADAR_INPUT_HANDLED")

    def test_radar_input_clears_stale_main_input_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            assert bot.radar is not None
            telegram = FakeTelegram()
            bot.telegram = telegram  # type: ignore[assignment]
            bot.vault.ensure_bootstrap()
            bot.radar.vault.ensure_bootstrap()
            bot.vault.set_input_state(123, {"kind": "content_idea"})
            bot.radar.vault.set_input_state(
                123,
                {"kind": "research_query", "provider": "youtube"},
            )

            with patch.object(
                bot.radar,
                "prepare_research",
                return_value=BotResponse("RADAR_INPUT_HANDLED"),
            ):
                bot.handle_update(
                    {
                        "message": {
                            "message_id": 4,
                            "from": {"id": 123, "first_name": "Owner"},
                            "chat": {"id": 123, "type": "private"},
                            "text": "AI animation",
                        }
                    }
                )

            self.assertIsNone(bot.vault.get_input_state(123))

    def test_switching_to_main_callback_clears_radar_input_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            assert bot.radar is not None
            bot.radar.vault.ensure_bootstrap()
            bot.radar.vault.set_input_state(
                123,
                {"kind": "research_query", "provider": "youtube"},
            )

            bot.dispatch_callback(
                TelegramUser(user_id=123, chat_id=123, first_name="Owner"),
                "menu:main",
            )

            self.assertIsNone(bot.radar.vault.get_input_state(123))

    def test_direct_radar_command_clears_stale_radar_input_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            assert bot.radar is not None
            bot.radar.vault.ensure_bootstrap()
            bot.radar.vault.set_input_state(
                123,
                {"kind": "research_query", "provider": "youtube"},
            )
            user = TelegramUser(user_id=123, chat_id=123, first_name="Owner")

            direct_home = bot.radar.home(user)

            self.assertIsInstance(direct_home, BotResponse)
            self.assertIsNone(bot.radar.vault.get_input_state(123))

            bot.radar.vault.set_input_state(
                123,
                {"kind": "research_query", "provider": "instagram"},
            )

            response = bot.dispatch(
                user,
                "/radar",
            )

            self.assertIsInstance(response, BotResponse)
            self.assertIsNone(bot.radar.vault.get_input_state(123))

            bot.radar.vault.set_input_state(
                123,
                {"kind": "research_query", "provider": "youtube"},
            )
            callback_response = bot.dispatch_callback(user, "research:home")

            self.assertIsInstance(callback_response, BotResponse)
            self.assertIsNone(bot.radar.vault.get_input_state(123))

            telegram = FakeTelegram()
            bot.telegram = telegram  # type: ignore[assignment]
            with (
                patch.object(bot.radar, "prepare_research") as stale_prepare,
                patch.object(
                    bot,
                    "dispatch",
                    return_value=BotResponse("NORMAL_MAIN_ROUTING"),
                ),
            ):
                bot.handle_update(
                    {
                        "message": {
                            "message_id": 5,
                            "from": {"id": 123, "first_name": "Owner"},
                            "chat": {"id": 123, "type": "private"},
                            "text": "orpartnery main message",
                        }
                    }
                )

            stale_prepare.assert_not_called()
            self.assertEqual(telegram.messages[-1][1], "NORMAL_MAIN_ROUTING")

    def test_main_bot_radar_retry_reuses_saved_materials_without_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            self.assertIsNotNone(bot.radar)
            assert bot.radar is not None
            bot.radar.research.ensure()
            run = bot.radar.research.create_run(
                "youtube",
                "",
                123,
                workflow="auto_content",
            )
            bot.radar.research.update_run(run.run_id, "running")
            bot.radar.research.save_results(
                run.run_id,
                [
                    {
                        "platform": "youtube",
                        "external_id": "saved-video",
                        "source_url": "https://youtube.example/saved-video",
                        "title": "Saved material",
                        "creator": "saved-channel",
                        "views": 100_000,
                    }
                ],
                mark_completed=False,
            )
            bot.radar.research.update_run(
                run.run_id,
                "failed",
                error="temporary script error",
            )
            user = TelegramUser(user_id=123, chat_id=123, first_name="Owner")

            with (
                patch.object(bot.radar, "_start_auto_content_retry_thread") as start,
                patch.object(bot.radar.youtube, "collect") as youtube_collect,
                patch.object(bot.radar.instagram, "start") as instagram_start,
            ):
                response = bot.dispatch_callback(
                    user,
                    f"research_content_retry:{run.run_id}",
                )

            self.assertIsNotNone(response)
            assert response is not None
            self.assertIn("Новый запрос Bright Data не создаётся", response.text)
            start.assert_called_once()
            youtube_collect.assert_not_called()
            instagram_start.assert_not_called()
            self.assertEqual(
                bot.radar.research.require_run(run.run_id).status,
                "running",
            )

    def test_main_bot_radar_retry_worker_never_calls_collectors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            assert bot.radar is not None
            bot.radar.research.ensure()
            run = bot.radar.research.create_run(
                "youtube",
                "",
                123,
                workflow="auto_content",
            )
            bot.radar.research.update_run(run.run_id, "running")
            saved = bot.radar.research.save_results(
                run.run_id,
                [
                    {
                        "platform": "youtube",
                        "external_id": "persisted-video",
                        "source_url": "https://youtube.example/persisted-video",
                        "title": "Persisted material",
                        "creator": "saved-channel",
                        "views": 100_000,
                    }
                ],
                mark_completed=False,
            )
            bot.radar.research.update_run(run.run_id, "failed", error="script error")

            with (
                patch.object(
                    bot.radar.youtube,
                    "collect",
                    side_effect=AssertionError("YouTube recollection is forbidden"),
                ),
                patch.object(
                    bot.radar.instagram,
                    "start",
                    side_effect=AssertionError("Instagram recollection is forbidden"),
                ),
                patch.object(
                    bot.radar.instagram,
                    "wait_for_results",
                    side_effect=AssertionError("Instagram polling is forbidden"),
                ),
                patch.object(bot.radar, "_finish_auto_content") as finish,
            ):
                bot.radar._auto_content_retry_worker(
                    run.run_id,
                    threading.Event(),
                )

            finish.assert_called_once()
            passed_results = finish.call_args.args[1]
            self.assertEqual(
                [result.result_id for result in passed_results],
                [saved[0].result_id],
            )

    def test_main_bot_radar_account_import_uses_isolated_radar_form(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            self.assertIsNotNone(bot.radar)
            assert bot.radar is not None
            bot.radar.vault.ensure_bootstrap()
            user = TelegramUser(user_id=123, chat_id=123, first_name="Owner")

            response = bot.dispatch_callback(user, "research:import")
            state = bot.radar.vault.get_input_state(user.user_id)

            self.assertIsNotNone(response)
            self.assertEqual((state or {}).get("kind"), "research_account_import")
            self.assertIn("TXT", response.text if response else "")

    def test_main_bot_routes_radar_account_document_to_radar_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            telegram = FakeTelegram()
            bot.telegram = telegram
            self.assertIsNotNone(bot.radar)
            assert bot.radar is not None
            bot.radar.vault.ensure_bootstrap()
            bot.radar.vault.set_input_state(
                123,
                {"kind": "research_account_import"},
            )
            update = {
                "message": {
                    "message_id": 2,
                    "from": {"id": 123, "first_name": "Owner"},
                    "chat": {"id": 123, "type": "private"},
                    "document": {
                        "file_id": "accounts-file",
                        "file_name": "accounts.txt",
                    },
                }
            }

            with patch.object(
                bot.radar,
                "handle_account_document",
                return_value=BotResponse("RADAR_DOCUMENT_HANDLED"),
            ) as handle_document:
                bot.handle_update(update)

            handle_document.assert_called_once()
            self.assertEqual(telegram.messages[-1][1], "RADAR_DOCUMENT_HANDLED")

    def test_radar_account_document_response_is_markdown_prepared(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            assert bot.radar is not None
            user = TelegramUser(user_id=123, chat_id=123, first_name="Owner")

            with patch.object(
                PartnerTelegramBot,
                "handle_account_document",
                return_value=BotResponse("## DOCUMENT IMPORTED"),
            ):
                response = bot.radar.handle_account_document(user, {"document": {}})

            self.assertTrue(response.render_markdown)
            self.assertIn("## DOCUMENT IMPORTED", response.text)

    def test_main_bot_cancels_radar_form_in_radar_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            self.assertIsNotNone(bot.radar)
            assert bot.radar is not None
            bot.radar.vault.ensure_bootstrap()
            bot.radar.vault.set_input_state(
                123,
                {"kind": "research_query", "provider": "youtube"},
            )

            response = bot.dispatch_callback(
                TelegramUser(user_id=123, chat_id=123, first_name="Owner"),
                "input:cancel",
            )

            self.assertIsNone(bot.radar.vault.get_input_state(123))
            self.assertIsNotNone(response)
            callbacks = [
                callback
                for row in (response.keyboard if response else []) or []
                for _, callback in row
            ]
            self.assertIn("menu:main", callbacks)
            self.assertIn("Радар идей", response.text if response else "")

    def test_main_bot_cancel_command_clears_radar_form(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            telegram = FakeTelegram()
            bot.telegram = telegram
            self.assertIsNotNone(bot.radar)
            assert bot.radar is not None
            bot.radar.vault.ensure_bootstrap()
            bot.radar.vault.set_input_state(
                123,
                {"kind": "research_query", "provider": "youtube"},
            )

            bot.handle_update(
                {
                    "message": {
                        "message_id": 3,
                        "from": {"id": 123, "first_name": "Owner"},
                        "chat": {"id": 123, "type": "private"},
                        "text": "/cancel",
                    }
                }
            )

            self.assertIsNone(bot.radar.vault.get_input_state(123))
            self.assertIn("Радар идей", telegram.messages[-1][1])

    def test_main_bot_adapts_background_radar_callbacks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            assert bot.radar is not None
            telegram = FakeTelegram()
            bot.radar.attach_transport(telegram, bot.test_shared_content)  # type: ignore[arg-type]

            delivered = bot.radar._send_background_response(
                123,
                BotResponse("Radar completed", [[("Back", "partner:menu")]]),
            )

            self.assertTrue(delivered)
            self.assertEqual(telegram.keyboards[-1], [[("Back", "menu:main")]])

    def test_main_bot_opens_radar_shared_item_from_tester_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            self.assertIsNotNone(bot.radar)
            assert bot.radar is not None
            bot.test_shared_content.ensure()
            item = bot.test_shared_content.create_item(
                "partner",
                "Saved Radar package",
                item_kind="production_idea",
                metadata={"kind": "production_idea", "idea": "Saved idea"},
            )
            bot.test_shared_content.handoff("partner", item.item_id)

            response = bot.dispatch_callback(
                TelegramUser(user_id=123, chat_id=123, first_name="Owner"),
                f"radar_shared_item:{item.item_id}",
            )

            self.assertIsNotNone(response)
            self.assertIn(item.item_id, response.text if response else "")
            self.assertIn("Saved Radar package", response.text if response else "")
            callbacks = [
                callback
                for row in (response.keyboard if response else []) or []
                for _, callback in row
            ]
            self.assertIn("ideas:list", callbacks)

    def test_radar_refuses_persisted_script_path_outside_dedicated_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(
                replace(
                    settings_for(root / "vault"),
                    radar_vault_path=root / "radar",
                )
            )
            assert bot.radar is not None
            bot.radar.research.ensure()
            run = bot.radar.research.create_run("youtube", "", 123)
            bot.radar.research.update_run(run.run_id, "running")
            result = bot.radar.research.save_results(
                run.run_id,
                [
                    {
                        "platform": "youtube",
                        "external_id": "legacy-script",
                        "source_url": "https://youtube.example/legacy-script",
                        "title": "Legacy script",
                    }
                ],
            )[0]
            external = root / "legacy-partner" / "script.md"
            external.parent.mkdir()
            external.write_text("MUST NOT BE READ", encoding="utf-8")
            bot.radar.research.link_script(result.result_id, external)

            response = bot.radar.handoff_research_result(result.result_id)

            self.assertIn("текущего runtime не найден", response.text)
            self.assertEqual(bot.test_shared_content.list_items(limit=10), [])
            symlink = root / "radar" / "escaped-script.md"
            symlink.symlink_to(external)
            self.assertIsNone(bot.radar._script_path_in_runtime(str(symlink)))

    def test_ideas_screen_combines_working_and_tester_packages_with_analytics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(settings_for(root))
            metadata = {
                "kind": "production_idea",
                "production_target": "ai_video_content_factory",
                "requires_live_shoot": False,
                "idea": "Автоматизация вместо ручной рутины",
                "format": "ai",
                "reason": "Два свежих ролика быстро набирают просмотры.",
                "analytics": {
                    "evidence_count": 2,
                    "recent_evidence_count_14d": 2,
                    "total_views": 150000,
                    "total_likes": 7000,
                    "combined_views_per_day": 42000,
                },
                "evidence": [
                    {
                        "platform": "instagram",
                        "external_id": "prod-reel",
                        "creator": "creator",
                        "title": "Automation",
                        "url": "https://www.instagram.com/reel/prod-reel/",
                        "views": 100000,
                        "likes": 5000,
                        "age_days": 2,
                    }
                ],
                "script": "Хук: покажи результат в первую секунду.",
            }
            prod = bot.shared_content.create_item(
                "partner",
                "Рабочая идея",
                item_kind="production_idea",
                metadata=metadata,
            )
            bot.shared_content.handoff("partner", prod.item_id)
            test_metadata = dict(metadata)
            test_metadata["idea"] = "Тестовая идея владельца"
            test_metadata["evidence"] = [
                {
                    "platform": "youtube",
                    "external_id": "test-video",
                    "creator": "tester",
                    "title": "Test automation",
                    "url": "https://www.youtube.com/watch?v=test-video",
                    "duration_seconds": 45,
                    "views": 50000,
                    "likes": 2000,
                    "age_days": 1,
                }
            ]
            test = bot.test_shared_content.create_item(
                "partner",
                "Тестовая идея",
                item_kind="production_idea",
                metadata=test_metadata,
            )
            bot.test_shared_content.handoff("partner", test.item_id)

            listing = bot.ideas_inbox()
            detail = bot.idea_item(f"test:{test.item_id}")

            self.assertIn("Автоматизация вместо ручной рутины", listing.text)
            self.assertIn("Тестовая идея владельца", listing.text)
            self.assertIn("[РАБОЧАЯ]", listing.text)
            self.assertIn("[ТЕСТ]", listing.text)
            self.assertIn("150K", detail.text)
            self.assertIn("42K просмотров/день", detail.text)
            self.assertIn("живая съёмка не требуется", detail.text)
            self.assertIn("https://www.youtube.com/watch?v=test-video", detail.text)

    def test_duplicate_reel_is_hidden_but_direct_link_opens_saved_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(settings_for(root))
            first_metadata = {
                "kind": "production_idea",
                "idea": "Первая идея из одного рилса",
                "evidence": [
                    {
                        "platform": "instagram",
                        "external_id": "ABC",
                        "url": "https://www.instagram.com/reel/ABC/",
                    }
                ],
                "script": "Первый сценарий.",
            }
            duplicate_metadata = {
                "kind": "production_idea",
                "idea": "Другая формулировка из повторного рилса",
                "evidence": [
                    {
                        "platform": "instagram",
                        "external_id": "provider-other-id",
                        "url": "https://instagram.com/reel/ABC/?utm_source=test",
                    }
                ],
                "script": "Сценарий повторной карточки.",
            }
            first = bot.test_shared_content.create_item(
                "partner",
                "Первая",
                item_kind="production_idea",
                metadata=first_metadata,
            )
            bot.test_shared_content.handoff("partner", first.item_id)
            duplicate = bot.test_shared_content.create_item(
                "partner",
                "Повтор",
                item_kind="production_idea",
                metadata=duplicate_metadata,
            )
            bot.test_shared_content.handoff("partner", duplicate.item_id)

            listing = bot.ideas_inbox()
            deep_link = bot.dispatch(
                TelegramUser(123, 123, "Owner"),
                f"/start idea_test_{duplicate.item_id}",
            )
            script = bot.idea_script(f"test:{duplicate.item_id}")

            self.assertIn("Первая идея из одного рилса", listing.text)
            self.assertNotIn(
                "Другая формулировка из повторного рилса",
                listing.text,
            )
            self.assertIn(
                "Другая формулировка из повторного рилса",
                deep_link.text,
            )
            self.assertIn("Сценарий повторной карточки.", script.text)

    def test_accepting_tester_idea_creates_one_real_content_factory_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(settings_for(root))
            user = TelegramUser(123, 123, "Owner")
            idea = bot.test_shared_content.create_item(
                "partner",
                "Идея и производственный сценарий",
                item_kind="production_idea",
                metadata={
                    "kind": "production_idea",
                    "idea": "Тестовая идея",
                    "evidence": [{"url": "https://example.com/source"}],
                    "script": "Сценарий",
                },
            )
            bot.test_shared_content.handoff("partner", idea.item_id)

            first = bot.accept_idea(user, f"test:{idea.item_id}")
            second = bot.accept_idea(user, f"test:{idea.item_id}")
            linked = bot.test_shared_content.require(idea.item_id)
            runs = bot._content_store().list_runs(limit=10)

            self.assertEqual(len(runs), 1)
            self.assertEqual(linked.status, "in_production")
            self.assertEqual(linked.linked_run_id, runs[0].run_id)
            self.assertIn("Платная генерация не запускалась", first.text)
            self.assertIn("уже связана", second.text)

    def test_shared_inbox_hides_unsubmitted_partner_drafts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            draft = bot.shared_content.create_item(
                "partner", "Черновик, который партнёр ещё не передал."
            )

            response = bot.shared_inbox()

            self.assertNotIn(draft.item_id, response.text)
            self.assertIn("очередь пока пуста", response.text)

    def test_accepting_shared_item_creates_exactly_one_content_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bot = AgentTelegramBot(settings_for(root))
            user = TelegramUser(123, 123, "Owner")
            item = bot.shared_content.create_item(
                "partner",
                "Референс: просто объяснить, как автоматизация экономит время.",
                source_url="https://example.com/reference",
            )
            bot.shared_content.handoff("partner", item.item_id)

            first = bot.accept_shared_item(user, item.item_id)
            linked = bot.shared_content.require(item.item_id)
            runs_after_first = bot._content_store().list_runs(limit=10)
            second = bot.accept_shared_item(user, item.item_id)
            runs_after_second = bot._content_store().list_runs(limit=10)

            self.assertEqual(linked.status, "in_production")
            self.assertTrue(linked.linked_run_id.startswith("CF-"))
            self.assertEqual(len(runs_after_first), 1)
            self.assertEqual(len(runs_after_second), 1)
            self.assertIn(item.item_id, runs_after_first[0].idea)
            self.assertIn("Платные операции", first.text)
            self.assertIn("уже связан", second.text)
            active = bot.vault.get_active_project(user.user_id)
            self.assertIsNotNone(active)
            self.assertEqual(active.slug, "content-factory")

    def test_return_form_can_be_cancelled_or_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            user = TelegramUser(123, 123, "Owner")
            first = bot.shared_content.create_item("partner", "Нужно уточнить автора.")
            bot.shared_content.handoff("partner", first.item_id)

            bot.dispatch_callback(user, f"shared_return:{first.item_id}")
            self.assertIsNotNone(bot.vault.get_input_state(user.user_id))
            bot.dispatch_callback(user, f"shared_return_cancel:{first.item_id}")
            self.assertIsNone(bot.vault.get_input_state(user.user_id))
            self.assertEqual(bot.shared_content.require(first.item_id).status, "handoff_requested")

            bot.dispatch_callback(user, f"shared_return:{first.item_id}")
            state = bot.vault.get_input_state(user.user_id)
            response = bot.handle_input_state(user, "Добавь исходную ссылку.", state or {})
            returned = bot.shared_content.require(first.item_id)

            self.assertEqual(returned.status, "returned")
            self.assertIn("Добавь исходную ссылку", returned.notes)
            self.assertIn("возвращён партнёру", response.text)

    def test_video_profile_selection_is_persisted_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            run = bot._content_store().create_run("Seedance profile")
            store = bot._production_store()
            store.save_scene_contract(
                run.run_id,
                [SceneSpec(
                    "S01", 1, 5, "purpose", "visual", "action", "static",
                    "", "", "", "cut", {}, "image prompt",
                )],
            )
            attempt = store.start_frame(run.run_id, "S01")
            store.complete_frame(run.run_id, "S01", attempt, b"frame", ".png")

            response = bot.select_video_profile(run.run_id, "s1080")
            persisted = store.load(run.run_id)["video_settings"]

            self.assertIn("Seedance 2", response.text)
            self.assertIn("1080p", response.text)
            self.assertEqual(persisted["model"], "bytedance/seedance-2")
            self.assertEqual(persisted["resolution"], "1080p")
            self.assertFalse(persisted["sound_enabled"])

    def test_video_duration_is_selected_after_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            run = bot._content_store().create_run("Duration selection")
            store = bot._production_store()
            store.save_scene_contract(
                run.run_id,
                [SceneSpec(
                    "S01", 1, 5, "purpose", "visual", "action", "static",
                    "", "", "", "cut", {}, "image prompt",
                )],
            )
            attempt = store.start_frame(run.run_id, "S01")
            store.complete_frame(run.run_id, "S01", attempt, b"frame", ".png")

            menu = bot.video_duration_menu(run.run_id, "s720")
            callbacks = [callback for row in menu.keyboard or [] for _, callback in row]
            self.assertIn(f"cf_video_duration:{run.run_id}:s720-10", callbacks)

            response = bot.select_video_profile(run.run_id, "s720", 10)
            persisted = store.load(run.run_id)["video_settings"]

            self.assertIn("10 секунд", response.text)
            self.assertEqual(persisted["duration_seconds"], 10)

    def test_seedance_480_profile_uses_kie_and_is_recommended(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            run = bot._content_store().create_run("Kie Seedance profile")
            store = bot._production_store()
            store.save_scene_contract(
                run.run_id,
                [SceneSpec(
                    "S01", 1, 5, "purpose", "visual", "action", "static",
                    "", "", "", "cut", {}, "image prompt",
                )],
            )
            attempt = store.start_frame(run.run_id, "S01")
            store.complete_frame(run.run_id, "S01", attempt, b"frame", ".png")

            menu = bot.video_quality_menu(run.run_id, "seedance")
            labels = [label for row in menu.keyboard or [] for label, _ in row]
            response = bot.select_video_profile(run.run_id, "s480")
            persisted = store.load(run.run_id)["video_settings"]

            self.assertTrue(
                any(
                    "480p" in label
                    and "Kie" in label
                    and "рекомендуется" in label
                    for label in labels
                )
            )
            self.assertIn("Провайдер: Kie", response.text)
            self.assertEqual(persisted["provider"], "kie")
            self.assertEqual(persisted["resolution"], "480p")

    def test_seedance_480_sound_profile_is_selectable_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            run = bot._content_store().create_run("Kie Seedance sound profile")
            store = bot._production_store()
            store.save_scene_contract(
                run.run_id,
                [SceneSpec(
                    "S01", 1, 5, "purpose", "visual", "action", "static",
                    "", "ambient sound", "", "cut", {}, "image prompt",
                )],
            )
            attempt = store.start_frame(run.run_id, "S01")
            store.complete_frame(run.run_id, "S01", attempt, b"frame", ".png")

            menu = bot.video_quality_menu(run.run_id, "seedance")
            labels = [label for row in menu.keyboard or [] for label, _ in row]
            response = bot.select_video_profile(run.run_id, "s480a", 15)
            persisted = store.load(run.run_id)["video_settings"]

            self.assertTrue(any("480p · со звуком" in label for label in labels))
            self.assertIn("Звук: да", response.text)
            self.assertEqual(persisted["provider"], "kie")
            self.assertEqual(persisted["resolution"], "480p")
            self.assertEqual(persisted["duration_seconds"], 15)
            self.assertTrue(persisted["sound_enabled"])

    def test_kling_standard_preview_shows_mode_and_sound_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            run = bot._content_store().create_run("Kling preview")
            store = bot._production_store()
            store.save_scene_contract(
                run.run_id,
                [SceneSpec(
                    "S01", 1, 5, "purpose", "visual", "action", "static",
                    "", "", "", "cut", {}, "image prompt",
                )],
            )
            attempt = store.start_frame(run.run_id, "S01")
            store.complete_frame(run.run_id, "S01", attempt, b"frame", ".png")
            store.set_video_settings(run.run_id, video_profile("ks").to_dict())
            store.save_video_prompts(
                run.run_id,
                [{
                    "scene_id": "S01",
                    "model_prompt": "Kling model prompt",
                    "universal_prompt": "universal prompt",
                    "model_id": "kling/v3",
                }],
            )

            response = bot.prepare_video_generation(run.run_id)

            self.assertIn("Модель: kling/v3", response.text)
            self.assertIn("Режим: std", response.text)
            self.assertIn("Звук: нет", response.text)
            self.assertIn("качество определяется режимом модели", response.text)
            approval = store.load(run.run_id)["video_approval"]
            self.assertEqual(approval["status"], "pending")

    def test_old_model_failed_jobs_offer_safe_selected_model_prompt_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            run = bot._content_store().create_run("Проверка video retry UX")
            state = bot._production_store().load(run.run_id)
            state["video_settings"] = video_profile("ks").to_dict()
            state["video_prompts"] = {
                "S01": {
                    "scene_id": "S01",
                    "model_id": "bytedance/seedance-2",
                    "duration_seconds": 5,
                    "aspect_ratio": "9:16",
                    "sound_enabled": False,
                    "model_prompt": "move slowly",
                }
            }
            state["video_jobs"] = {
                "S01": {
                    "scene_id": "S01",
                    "status": "failed",
                    "retry_allowed": True,
                    "external_task_id": "",
                }
            }
            bot._production_store().write_state(state)

            response = bot.show_video_prompts(run.run_id)

            callbacks = [callback for row in response.keyboard or [] for _, callback in row]
            self.assertIn(f"cf_video_prompts:{run.run_id}", callbacks)
            self.assertIn("прежней модели", response.text)
            self.assertIn("Kling 3", response.text)
            self.assertNotIn(f"cf_retry_video_review:{run.run_id}", callbacks)
            self.assertNotIn(f"cf_video_prepare:{run.run_id}", callbacks)

    def test_missing_script_and_prompt_contracts_are_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            bot.llm = RepairingLlm()
            store = bot._content_store()
            run = store.create_run("Ролик о надёжной автоматизации")
            store.mark_running(run.run_id)
            store.save_stage(run.run_id, "brief", "Полный компактный бриф")
            run = store.advance(run.run_id)

            script = bot._normalize_stage_contract(
                run,
                "script",
                "Краткий читаемый сценарий без служебного JSON.",
            )
            self.assertIn("SCENE_CONTRACT", script)
            self.assertEqual(len(parse_scene_contract(script)), 3)

            store.mark_running(run.run_id)
            store.save_stage(run.run_id, "script", script)
            run = store.advance(run.run_id)
            store.mark_running(run.run_id)
            store.save_stage(run.run_id, "storyboard", "План трёх кадров")
            run = store.advance(run.run_id)

            prompts = bot._normalize_stage_contract(
                run,
                "prompts",
                "Три отдельных image prompt описаны прозой без JSON.",
            )
            self.assertIn("IMAGE_PROMPT_CONTRACT", prompts)
            scenes = parse_scene_contract(script)
            merged = merge_image_prompt_contract(scenes, prompts)
            self.assertEqual(len(merged), 3)

    def test_invalid_reference_contract_is_replaced_not_left_before_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            bot.llm = RepairingLlm()
            store = bot._content_store()
            run = store.create_run("Repair malformed references")
            store.mark_running(run.run_id)
            store.save_stage(run.run_id, "brief", "Brief")
            run = store.advance(run.run_id)
            script = "Readable script\n\n" + FakeLlm._contract("SCENE_CONTRACT", full=True)
            store.mark_running(run.run_id)
            store.save_stage(run.run_id, "script", script)
            run = store.advance(run.run_id)
            store.mark_running(run.run_id)
            store.save_stage(run.run_id, "storyboard", "Storyboard")
            run = store.advance(run.run_id)
            bad_payload = {
                "schema_version": 2,
                "references": [{
                    "reference_id": "REF-MIXED-01",
                    "kind": "character_environment_style",
                    "name": "Mixed",
                    "prompt": "ambiguous",
                    "scene_ids": ["S01"],
                }],
                "locations": [],
                "scenes": [
                    {
                        "scene_id": f"S{index:02d}",
                        "image_prompt": f"Prompt {index}",
                        "reference_ids": [],
                        "location_id": "",
                        "location_reference_scene_id": "",
                    }
                    for index in range(1, 4)
                ],
            }
            draft = (
                "Readable prompts\n\nIMAGE_PROMPT_CONTRACT\n```json\n"
                + json.dumps(bad_payload)
                + "\n```"
            )

            repaired = bot._normalize_stage_contract(run, "prompts", draft)

            bot._validate_stage_contract(run, "prompts", repaired)
            self.assertIn("Readable prompts", repaired)
            self.assertNotIn("REF-MIXED-01", repaired)
            self.assertEqual(repaired.count("IMAGE_PROMPT_CONTRACT"), 1)

    def test_button_flow_creates_run_and_advances_to_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            bot.llm = FakeLlm()
            bot.telegram = FakeTelegram()
            user = TelegramUser(user_id=123, chat_id=456, first_name="Owner")

            menu = bot.main_menu(user)
            self.assertIsInstance(menu, BotResponse)
            self.assertTrue(any(button[0] == "🎬 Новый ролик" for row in menu.keyboard or [] for button in row))

            bot.start_new_content(user)
            input_state = bot.vault.get_input_state(user.user_id)
            self.assertEqual(input_state["kind"], "content_idea")

            first_result = bot.handle_input_state(
                user,
                "Ролик о том, как агент строит контент-завод",
                input_state,
            )
            self.assertIn("готов к твоей проверке", first_result.text)
            self.assertTrue(
                any("Что происходит:" in text for _, text in bot.telegram.messages)
            )
            self.assertTrue(
                any(
                    "Работает агент: Продюсер и контент-стратег" in text
                    for _, text in bot.telegram.messages
                )
            )

            run = bot._content_store().list_runs()[0]
            self.assertEqual(run.current_stage, "brief")
            self.assertEqual(run.status, "waiting_approval")

            next_result = bot.dispatch_callback(user, f"cf_next:{run.run_id}:brief")
            self.assertIn("Сценарий", next_result.text)
            advanced = bot._content_store().get_run(run.run_id)
            self.assertEqual(advanced.current_stage, "script")
            self.assertEqual(advanced.status, "waiting_approval")
            self.assertTrue(bot.telegram.actions)

            stale_result = bot.dispatch_callback(user, f"cf_next:{run.run_id}:brief")
            self.assertIn("уже завершённому этапу", stale_result.text)
            still_script = bot._content_store().get_run(run.run_id)
            self.assertEqual(still_script.current_stage, "script")

            storyboard_result = bot.dispatch_callback(
                user,
                f"cf_next:{run.run_id}:script",
            )
            self.assertIn("Раскадровка", storyboard_result.text)
            storyboard_run = bot._content_store().get_run(run.run_id)
            self.assertEqual(storyboard_run.current_stage, "storyboard")
            self.assertFalse(
                any(
                    callback.startswith("cf_generate_visual:")
                    for row in storyboard_result.keyboard or []
                    for _, callback in row
                )
            )

            bot.image_client = FakeImageClient()
            blocked_result = bot.generate_frames(user, run.run_id)
            self.assertIn("после подтверждения всех текстовых этапов", blocked_result.text)

            qa_result = None
            for expected_stage in ("storyboard", "prompts"):
                qa_result = bot.dispatch_callback(
                    user,
                    f"cf_next:{run.run_id}:{expected_stage}",
                )
            self.assertIsNotNone(qa_result)
            self.assertTrue(
                any(
                    callback.startswith("cf_generate_visual:")
                    for row in qa_result.keyboard or []  # type: ignore[union-attr]
                    for _, callback in row
                )
            )

            image_result = bot.generate_frames(user, run.run_id)
            self.assertIn("обработка кадров завершена", image_result.text)
            self.assertEqual(len(bot.telegram.photos), 3)
            self.assertTrue(all(path.exists() for path in bot.telegram.photos))
            state = bot._production_store().load(run.run_id)
            self.assertEqual(len(state["scenes"]), 3)
            self.assertEqual(len(state["frames"]), 3)
            self.assertEqual(len(state["selected_frame_ids"]), 3)
            self.assertTrue(
                any("0 из 3" in text for _, text in bot.telegram.messages)
            )
            self.assertTrue(
                any("3 из 3" in text for _, text in bot.telegram.messages)
            )

    def test_factory_home_describes_complete_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            user = TelegramUser(user_id=123, chat_id=456, first_name="Owner")

            response = bot.factory_home(user)

            self.assertIn("отдельные кадры", response.text)
            self.assertIn("видеопромпты", response.text)
            self.assertIn("явного подтверждения", response.text)
            self.assertNotIn("подключим следующим блоком", response.text)

    def test_ready_run_exposes_reference_generation_and_safe_frame_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            content = bot._content_store()
            run = content.create_run("Reference-aware Telegram menu")
            for stage in ("brief", "script", "storyboard", "prompts", "qa"):
                content.mark_running(run.run_id)
                run = content.save_stage(run.run_id, stage, f"Artifact for {stage}")
                if stage != "qa":
                    run = content.advance(run.run_id)

            production = bot._production_store()
            production.save_scene_contract(
                run.run_id,
                [
                    SceneSpec(
                        "S01",
                        1,
                        3,
                        "purpose",
                        "visual",
                        "action",
                        "static",
                        "",
                        "",
                        "",
                        "cut",
                        {},
                        "image prompt",
                        ("REF-A",),
                    )
                ],
            )
            production.save_reference_plan(
                run.run_id,
                [ReferenceSpec("REF-A", "character", "Hero", "model sheet", ("S01",))],
                [],
            )
            reference_attempt = production.start_reference(run.run_id, "REF-A")
            reference_path = production.complete_reference(
                run.run_id, "REF-A", reference_attempt, b"reference", ".png"
            )
            frame_attempt = production.start_frame(
                run.run_id,
                "S01",
                reference_inputs=[
                    {
                        "reference_id": "REF-A",
                        "role": "character",
                        "file": str(reference_path),
                    }
                ],
            )
            production.complete_frame(run.run_id, "S01", frame_attempt, b"frame", ".png")

            response = bot.run_detail(run.run_id)
            callbacks = [callback for row in response.keyboard or [] for _, callback in row]

            self.assertIn(f"cf_generate_refs:{run.run_id}", callbacks)
            self.assertIn(f"cf_show_refs:{run.run_id}", callbacks)
            self.assertIn(f"cf_regenerate_frames:{run.run_id}", callbacks)

    def test_video_status_transition_is_visible_in_telegram(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            bot.telegram = FakeTelegram()

            bot._notify_video_status_changes(
                456,
                "CF-TEST-001",
                {"S01": "pending"},
                {
                    "video_jobs": {
                        "S01": {
                            "scene_id": "S01",
                            "status": "processing",
                            "error": "",
                        }
                    }
                },
            )

            self.assertEqual(len(bot.telegram.messages), 1)
            self.assertIn("S01", bot.telegram.messages[0][1])
            self.assertIn("генерируется", bot.telegram.messages[0][1])

    def test_video_model_access_error_is_actionable(self) -> None:
        message = AgentTelegramBot._friendly_video_error(
            'PolzaAI HTTP 403: доступ к операции запрещён. Ответ сервиса: '
            'FORBIDDEN: Модель "bytedance/seedance-2" недоступна для данного API-ключа'
        )

        self.assertIn("Ключ PolzaAI действителен", message)
        self.assertIn("API-ключи", message)
        self.assertIn(".env", message)
        self.assertNotIn("pza_", message)


if __name__ == "__main__":
    unittest.main()
