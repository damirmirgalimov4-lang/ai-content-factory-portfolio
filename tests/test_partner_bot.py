from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent_platform.config import Settings
from agent_platform.deepgram import Transcript
from agent_platform.partner_assistant import PartnerAccessStore, PartnerWorkspace
from agent_platform.image_generation import GeneratedImage
from agent_platform.partner_bot import (
    PartnerTelegramBot,
    create_partner_llm,
    load_partner_settings,
    validate_partner_settings,
)
from agent_platform.llm import CodexExecClient
from agent_platform.telegram_bot import BotResponse, TelegramApiError, TelegramUser
from agent_platform.vault import VaultStore


def settings_for(
    vault_path: Path,
    *,
    allowed_ids: set[int] | None = None,
    tester_ids: set[int] | None = None,
    codex_path: Path | None = None,
) -> Settings:
    return Settings(
        telegram_bot_token="test-token",
        telegram_allowed_user_ids={777} if allowed_ids is None else allowed_ids,
        vault_path=vault_path,
        openai_api_key="",
        openai_base_url="https://api.openai.com/v1",
        openai_image_model="gpt-image-2",
        openai_image_size="1024x1536",
        openai_image_quality="low",
        deepgram_api_key="",
        deepgram_model="nova-3",
        deepgram_language="ru",
        codex_cli_path=str(codex_path) if codex_path else "missing-codex.exe",
        codex_chat_model="gpt-5.6-sol",
        codex_timeout_seconds=30,
        codex_production_timeout_seconds=60,
        codex_workdir=vault_path,
        shared_content_path=vault_path.parent / "shared-content",
        telegram_tester_user_ids=set() if tester_ids is None else tester_ids,
        test_vault_path=vault_path.parent / "vault-partner-test",
        test_shared_content_path=vault_path.parent / "shared-content-test",
    )


class FakeTelegram:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str, object, bool]] = []
        self.actions: list[tuple[int, str]] = []
        self.photos: list[tuple[int, Path, str, object]] = []

    def send_message(self, chat_id, text, keyboard=None, render_markdown=False):
        self.messages.append((chat_id, text, keyboard, render_markdown))

    def send_chat_action(self, chat_id, action="typing"):
        self.actions.append((chat_id, action))

    def set_commands(self, commands):
        return None

    def set_commands_menu_button(self):
        return None

    def answer_callback_query(self, callback_query_id):
        return None

    def edit_message(
        self,
        chat_id,
        message_id,
        text,
        keyboard=None,
        render_markdown=False,
    ):
        self.messages.append((chat_id, text, keyboard, render_markdown))

    def download_file(self, file_id, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"telegram-media")
        return destination

    def send_photo(self, chat_id, path, caption, keyboard=None):
        self.photos.append((chat_id, Path(path), caption, keyboard))


class OfflineTelegram(FakeTelegram):
    def send_message(self, chat_id, text, keyboard=None, render_markdown=False):
        del chat_id, text, keyboard, render_markdown
        raise TelegramApiError("temporary network outage")


class FakePartnerLlm:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self.image_calls: list[list[Path]] = []

    @property
    def is_configured(self) -> bool:
        return True

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        system = messages[0]["content"]
        prompt = messages[-1]["content"]
        if "редактор-аналитик Reels" in system:
            result_ids = [
                int(value)
                for value in re.findall(r"^result_id:\s*(\d+)$", prompt, re.MULTILINE)
            ]
            return json.dumps(
                {
                    "result_id": result_ids[0],
                    "evidence_result_ids": [result_ids[0]],
                    "source_premise": (
                        "Практический процесс автоматизации выполняется без ручной рутины"
                    ),
                    "idea": "Показать реальную автоматизацию на понятном бытовом примере",
                    "adaptation_changes": [
                        "заменить конкретный интерфейс нейтральной визуализацией"
                    ],
                    "theme_changed": False,
                    "reason": "Тема совпадает с направлением, а ролик быстро набирает просмотры",
                    "format": "ai",
                },
                ensure_ascii=False,
            )
        if "производственный сценарист AI-video" in system:
            return (
                "## Название и цель\n\n"
                "«Невидимый конвейер». Показать, как AI-система превращает одну "
                "идею в готовый ролик без ручной рутины.\n\n"
                "## Хук\n\n"
                "За первые две секунды хаотичная стопка задач сама превращается "
                "в аккуратную ленту готовых видео.\n\n"
                "## Сценарная структура\n\n"
                "### Сцена 1 · 0-3 сек\n\n"
                "**Функция:** показать проблему. **Визуал:** генерируемый стеклянный "
                "конвейер и десятки карточек-задач. **Действие:** карточки падают "
                "на ленту. **Камера:** плавный push-in. **Озвучка:** «Контент всё "
                "ещё забирает часы?» **Переход:** одна карточка вспыхивает.\n\n"
                "### Сцена 2 · 3-8 сек\n\n"
                "**Функция:** показать решение. **Визуал:** та же стеклянная лента "
                "и карточки в той же студии. **Действие:** карточки складываются в готовый "
                "вертикальный ролик. **Камера:** боковой tracking. **Озвучка:** "
                "«Одна идея проходит весь конвейер автоматически». **Переход:** "
                "готовый ролик раскрывается на весь экран.\n\n"
                "## Финал\n\n"
                "Готовый ролик появляется рядом со счётчиком сэкономленного времени; "
                "обещание хука закрыто видимым результатом.\n\n"
                "## Производственные ограничения\n\n"
                "**Режим производства:** только AI-видео; живая съёмка не требуется.\n\n"
                "Один конвейер, один дизайн карточек и одна палитра во всех сценах. "
                "Не использовать логотипы и неподтверждённые цифры."
            )
        if "валидный JSON" in system:
            user_message = prompt.rsplit("Сообщение партнёра:\n", 1)[-1]
            action = {
                "type": "reply_only",
                "text": "",
                "reason": "",
            }
            if "запомни" in user_message.lower():
                action = {
                    "type": "remember_global",
                    "text": "партнёр предпочитает спокойную экспертную подачу",
                    "reason": "устойчивое предпочтение",
                }
            return json.dumps(
                {
                    "reply": "Принял.",
                    "scope": "personal" if "ужин" in user_message.lower() else "work",
                    "action": action,
                },
                ensure_ascii=False,
            )
        if "Режим: идеи для Reels" in prompt:
            return "1. Идея о разборе реальной автоматизации\n\nСделать первой: идея 1."
        if "Режим: сценарий для Reels" in prompt:
            return "Хук: показываем проблему.\n\nCTA: написать в личные сообщения."
        if "Режим: контент-план" in prompt:
            return "День 1: живой разбор процесса."
        if "Режим: адаптация найденного референса" in prompt:
            return (
                "## Хук\n\nПокажи результат автоматизации в первой секунде.\n\n"
                "## Основная часть\n\nОбъясни процесс на одном конкретном примере."
            )
        return "Обычный ответ ассистента."

    def chat_with_images(self, messages, image_paths):
        self.image_calls.append(list(image_paths))
        return self.chat(messages)


class FakeDeepgram:
    is_configured = True

    def transcribe_file(self, audio_path):
        return Transcript(
            text="Запомни идею для живого ролика про автоматизацию",
            confidence=0.99,
            duration_seconds=4.0,
            raw_response={},
        )


class FakeImageClient:
    is_configured = True

    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompt, references=()):
        self.calls.append((prompt, tuple(references)))
        return GeneratedImage(b"fake-png", ".png", "image/png")

    def cancel_active(self):
        return True


class FakeYouTubeClient:
    is_configured = True

    def collect(self, accounts, query, *, limit, cancelled):
        del accounts, query, limit
        if cancelled():
            return []
        return [
            {
                "platform": "youtube",
                "external_id": "fresh-video",
                "title": "Fresh automation workflow",
                "source_url": "https://www.youtube.com/watch?v=fresh-video",
                "creator": "Fresh creator",
                "published_at": "2026-07-26T12:00:00Z",
                "duration_seconds": 34,
                "views": 80_000,
                "likes": 5_000,
                "comments": 300,
                "description": "A fast-growing practical automation example",
            },
            {
                "platform": "youtube",
                "external_id": "old-video",
                "title": "Old broad hit",
                "source_url": "https://www.youtube.com/watch?v=old-video",
                "creator": "Old creator",
                "published_at": "2025-01-01T12:00:00Z",
                "duration_seconds": 40,
                "views": 120_000,
                "likes": 2_000,
                "comments": 50,
                "description": "An older generic video",
            },
        ]


class PartnerWorkspaceTest(unittest.TestCase):
    def test_bootstrap_creates_only_partner_workspace_and_default_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            workspace = PartnerWorkspace(VaultStore(root))

            project = workspace.ensure(user_id=777)

            self.assertEqual(project.slug, "контент-партнёра")
            self.assertTrue((root / "workspace" / "PARTNER_PROFILE.md").exists())
            self.assertTrue((project.path / "ideas").is_dir())
            self.assertTrue((project.path / "scripts").is_dir())
            self.assertTrue((project.path / "plans").is_dir())
            self.assertFalse((root / "projects" / "content-factory").exists())
            self.assertIn("партнёр", (root / "workspace" / "USER.md").read_text(encoding="utf-8"))

    def test_recent_conversation_and_artifact_are_added_to_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            vault = VaultStore(root)
            workspace = PartnerWorkspace(vault)
            workspace.ensure(user_id=777)
            vault.log_exchange(777, "Мой прошлый вопрос", "Мой прошлый ответ")
            workspace.save_artifact(777, "idea", "Тема", "Сильная идея")

            context = workspace.context_summary(777)

            self.assertIn("Мой прошлый вопрос", context)
            self.assertIn("Сильная идея", context)


class PartnerBotTest(unittest.TestCase):
    def build_bot(
        self,
        root: Path,
        allowed_ids: set[int] | None = None,
        tester_ids: set[int] | None = None,
    ):
        bot = PartnerTelegramBot(
            settings_for(
                root,
                allowed_ids=allowed_ids,
                tester_ids=tester_ids,
            )
        )
        bot.telegram = FakeTelegram()
        bot.chat_llm = FakePartnerLlm()
        if bot.tester_bot is not None:
            bot.tester_bot.telegram = bot.telegram
            bot.tester_bot.chat_llm = FakePartnerLlm()
        return bot

    def test_radar_redirect_blocks_stale_retry_callback_in_partner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = PartnerTelegramBot(
                replace(
                    settings_for(Path(temp_dir) / "vault-partner"),
                    radar_redirect_to_content_factory=True,
                ),
                enable_tester_routing=False,
            )
            user = TelegramUser(777, 777, "Partner")

            with patch.object(
                bot,
                "retry_auto_content",
                return_value=BotResponse("MUTATING_OLD_RADAR"),
            ) as retry:
                callback_response = bot.dispatch_callback(
                    user,
                    "research_content_retry:RS-20260807-005",
                )
            command_response = bot.dispatch(user, "/research")

            retry.assert_not_called()
            self.assertIn("Контент-завод", callback_response.text)
            self.assertIn("Контент-завод", command_response.text)
            callbacks = [
                callback
                for row in callback_response.keyboard or []
                for _, callback in row
            ]
            self.assertIn(
                "url:https://t.me/ContentFactoryExampleBot?start=radar",
                callbacks,
            )

    def test_radar_redirect_blocks_stale_research_input_in_partner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = PartnerTelegramBot(
                replace(
                    settings_for(Path(temp_dir) / "vault-partner"),
                    radar_redirect_to_content_factory=True,
                ),
                enable_tester_routing=False,
            )
            user = TelegramUser(777, 777, "Partner")
            bot.vault.ensure_bootstrap()
            bot.vault.set_input_state(
                user.user_id,
                {"kind": "research_query", "provider": "instagram"},
            )
            state = bot.vault.get_input_state(user.user_id)
            assert state is not None

            with patch.object(
                bot,
                "prepare_research",
                return_value=BotResponse("MUTATING_OLD_RADAR"),
            ) as prepare:
                response = bot.handle_input_state(user, "AI videos", state)

            prepare.assert_not_called()
            self.assertIsNone(bot.vault.get_input_state(user.user_id))
            self.assertIn("Контент-завод", response.text)

            bot.vault.set_input_state(
                user.user_id,
                {"kind": "research_query", "provider": "youtube"},
            )
            command_response = bot.dispatch(user, "/research")
            assert isinstance(command_response, BotResponse)
            self.assertIsNone(bot.vault.get_input_state(user.user_id))
            self.assertIn("Контент-завод", command_response.text)

            bot.vault.set_input_state(
                user.user_id,
                {"kind": "research_query", "provider": "instagram"},
            )
            callback_response = bot.dispatch_callback(user, "research:home")
            self.assertIsNone(bot.vault.get_input_state(user.user_id))
            self.assertIn("Контент-завод", callback_response.text)

    def test_radar_redirect_is_inherited_by_default_tester_routing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tester_root = root / "vault-partner-test"
            settings = replace(
                settings_for(root / "vault-partner"),
                telegram_allowed_user_ids={777},
                telegram_tester_user_ids={888},
                test_vault_path=tester_root,
                test_shared_content_path=root / "shared-content-test",
                radar_redirect_to_content_factory=True,
            )
            bot = PartnerTelegramBot(settings)
            assert bot.tester_bot is not None
            tester = TelegramUser(888, 888, "Owner")

            with patch.object(
                bot.tester_bot,
                "retry_auto_content",
                return_value=BotResponse("MUTATING_TESTER_RADAR"),
            ) as retry:
                response = bot.tester_bot.dispatch_callback(
                    tester,
                    "research_content_retry:RS-20260807-005",
                )

            retry.assert_not_called()
            self.assertTrue(bot.tester_bot.settings.radar_redirect_to_content_factory)
            self.assertIn("Контент-завод", response.text)
            self.assertFalse((tester_root / "research" / "research.sqlite3").exists())

    def test_radar_redirect_keeps_image_recovery_but_skips_radar_run_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = PartnerTelegramBot(
                replace(
                    settings_for(Path(temp_dir) / "vault-partner"),
                    radar_redirect_to_content_factory=True,
                ),
                enable_tester_routing=False,
            )
            bot.research.ensure()
            image = bot.research.create_image("Saved image prompt", 777)
            bot.research.update_image(image.asset_id, "generating")
            run = bot.research.create_run(
                "youtube",
                "saved query",
                777,
                workflow="auto_content",
            )
            bot.research.update_run(run.run_id, "running")

            bot._prepare_runtime()

            self.assertEqual(
                bot.research.require_image(image.asset_id).status,
                "failed",
            )
            self.assertEqual(bot.research.require_run(run.run_id).status, "running")

    def test_radar_redirect_blocks_stale_account_import_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = PartnerTelegramBot(
                replace(
                    settings_for(Path(temp_dir) / "vault-partner"),
                    radar_redirect_to_content_factory=True,
                ),
                enable_tester_routing=False,
            )
            telegram = FakeTelegram()
            bot.telegram = telegram  # type: ignore[assignment]
            bot.vault.ensure_bootstrap()
            bot.vault.set_input_state(777, {"kind": "research_account_import"})

            with patch.object(bot, "handle_account_document") as import_document:
                bot.handle_update(
                    {
                        "message": {
                            "from": {"id": 777, "first_name": "Partner"},
                            "chat": {"id": 777, "type": "private"},
                            "document": {
                                "file_id": "stale-file",
                                "file_name": "accounts.txt",
                            },
                        }
                    }
                )

            import_document.assert_not_called()
            self.assertIsNone(bot.vault.get_input_state(777))
            self.assertIn("Контент-завод", telegram.messages[-1][1])

    def test_radar_redirect_finishes_generic_list_without_importing_accounts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = PartnerTelegramBot(
                replace(
                    settings_for(Path(temp_dir) / "vault-partner"),
                    radar_redirect_to_content_factory=True,
                ),
                enable_tester_routing=False,
            )
            user = TelegramUser(777, 777, "Partner")
            bot.workspace.ensure(user.user_id)
            bot.workbench.ensure()
            bot.start_list_collection(user)
            state = bot.vault.get_input_state(user.user_id)
            assert state is not None
            batch_id = state["batch_id"]
            bot.append_list_collection(user, "youtube:@must-not-import", batch_id)

            response = bot.finish_list_collection(user, batch_id)
            direct_import = bot.import_accounts(user, "youtube:@also-must-not-import")

            self.assertEqual(bot.research.list_accounts(), [])
            self.assertIn("Список завершён", response.text)
            self.assertIn("Контент-завод", direct_import.text)
            callbacks = [
                callback
                for row in response.keyboard or []
                for _, callback in row
            ]
            self.assertIn(f"list_analyze:{batch_id}", callbacks)
            self.assertIn(f"list_handoff:{batch_id}", callbacks)
            self.assertIn(
                "url:https://t.me/ContentFactoryExampleBot?start=radar",
                callbacks,
            )

    def test_owner_tester_uses_same_bot_with_isolated_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            test_root = Path(temp_dir) / "vault-partner-test"
            bot = self.build_bot(root, allowed_ids={777}, tester_ids={888})

            bot.handle_update(
                {
                    "update_id": 1,
                    "message": {
                        "from": {"id": 888, "first_name": "Owner"},
                        "chat": {"id": 888, "type": "private"},
                        "text": "/start",
                    },
                }
            )
            bot.handle_update(
                {
                    "update_id": 2,
                    "message": {
                        "from": {"id": 888, "first_name": "Owner"},
                        "chat": {"id": 888, "type": "private"},
                        "text": "TESTER-ISOLATION-MARKER",
                    },
                }
            )

            self.assertIsNotNone(bot.tester_bot)
            self.assertIn("Режим тестировщика владельца", bot.telegram.messages[0][1])
            tester_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in test_root.rglob("*.md")
            )
            production_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in root.rglob("*.md")
            ) if root.exists() else ""
            self.assertIn("TESTER-ISOLATION-MARKER", tester_text)
            self.assertNotIn("TESTER-ISOLATION-MARKER", production_text)

    def test_partner_owner_is_not_routed_to_tester_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            test_root = Path(temp_dir) / "vault-partner-test"
            bot = self.build_bot(root, allowed_ids={777}, tester_ids={888})

            bot.handle_update(
                {
                    "update_id": 1,
                    "message": {
                        "from": {"id": 777, "first_name": "Partner"},
                        "chat": {"id": 777, "type": "private"},
                        "text": "PARTNER-OWNER-MARKER",
                    },
                }
            )

            production_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in root.rglob("*.md")
            )
            tester_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in test_root.rglob("*.md")
            ) if test_root.exists() else ""
            self.assertIn("PARTNER-OWNER-MARKER", production_text)
            self.assertNotIn("PARTNER-OWNER-MARKER", tester_text)

    def test_tester_callback_is_routed_to_isolated_menu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = self.build_bot(
                Path(temp_dir) / "vault-partner",
                allowed_ids={777},
                tester_ids={888},
            )

            bot.handle_update(
                {
                    "update_id": 1,
                    "callback_query": {
                        "id": "callback-1",
                        "from": {"id": 888, "first_name": "Owner"},
                        "message": {
                            "message_id": 10,
                            "chat": {"id": 888, "type": "private"},
                        },
                        "data": "partner:menu",
                    },
                }
            )

            self.assertIn("Режим тестировщика владельца", bot.telegram.messages[-1][1])

    def test_tester_can_launch_instagram_only_after_explicit_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = self.build_bot(
                Path(temp_dir) / "vault-partner",
                allowed_ids={777},
                tester_ids={888},
            )
            tester = bot.tester_bot
            self.assertIsNotNone(tester)
            assert tester is not None
            tester.research.ensure()
            run = tester.research.create_run("instagram", "AI reels", 888)

            with patch.object(tester, "_start_research_thread") as start_thread:
                response = tester.confirm_research(
                    TelegramUser(888, 888, "Owner"),
                    run.run_id,
                )

            start_thread.assert_called_once()
            self.assertIn("Bright Data действительно создаст внешний snapshot", response.text)
            self.assertIn("изолированном тестовом хранилище", response.text)
            self.assertEqual(
                tester.research.require_run(run.run_id).status,
                "running",
            )

    def test_empty_allowlist_requests_pairing_without_saving_or_calling_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root, allowed_ids=set())

            bot.handle_update(
                {
                    "update_id": 1,
                    "message": {
                        "from": {"id": 777, "first_name": "Partner"},
                        "chat": {"id": 777, "type": "private"},
                        "text": "/whoami",
                    },
                }
            )

            self.assertFalse(root.exists())
            self.assertEqual(len(bot.chat_llm.calls), 0)
            self.assertNotIn("777", bot.telegram.messages[0][1])
            self.assertIn("одноразовый код", bot.telegram.messages[0][1])
            self.assertIn("не сохраняются", bot.telegram.messages[0][1])

    def test_one_time_code_claims_bot_without_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root, allowed_ids=set())
            code = bot.access.ensure_pairing_code()

            bot.handle_update(
                {
                    "update_id": 1,
                    "message": {
                        "from": {"id": 777, "first_name": "Partner"},
                        "chat": {"id": 777, "type": "private"},
                        "text": f"/start {code}",
                    },
                }
            )

            self.assertEqual(bot.access.owner_id(), 777)
            self.assertIn("привязан к партнёру", bot.telegram.messages[-1][1])
            self.assertTrue((root / "workspace" / "PARTNER_PROFILE.md").exists())

    def test_onboarding_persists_structured_partner_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            user = TelegramUser(777, 777, "Partner")
            bot.workspace.ensure(user.user_id)
            answers = [
                "Ищу клиентов и пишу сценарии",
                "Развить Instagram и получать заявки",
                "Владельцы малого бизнеса",
                "AI-видео и автоматизация",
                "Спокойно, понятно и с лёгким юмором",
            ]

            bot.start_onboarding(user)
            response = None
            for answer in answers:
                state = bot.vault.get_input_state(user.user_id)
                self.assertIsNotNone(state)
                response = bot.handle_input_state(user, answer, state or {})

            self.assertIsNotNone(response)
            self.assertIn("Профиль партнёра сохранён", response.text)
            profile = bot.workspace.profile_text()
            for answer in answers:
                self.assertIn(answer, profile)
            self.assertTrue(bot.workspace.profile_is_complete())
            self.assertIsNone(bot.vault.get_input_state(user.user_id))

    def test_idea_mode_saves_generated_artifact_in_active_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            user = TelegramUser(777, 777, "Partner")
            project = bot.workspace.ensure(user.user_id)

            bot.start_mode(user, "idea")
            state = bot.vault.get_input_state(user.user_id)
            response = bot.handle_input_state(
                user,
                "Идеи о том, как автоматизация экономит время бизнеса",
                state or {},
            )

            artifacts = list((project.path / "ideas").glob("*.md"))
            self.assertEqual(len(artifacts), 1)
            artifact = artifacts[0].read_text(encoding="utf-8")
            self.assertIn("автоматизация экономит время", artifact)
            self.assertIn("Сделать первой", artifact)
            self.assertIn("Материал сохранён", response.text)

    def test_memory_suggestion_requires_explicit_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            user = TelegramUser(777, 777, "Partner")
            bot.workspace.ensure(user.user_id)

            response = bot.answer_with_llm(
                user,
                "Запомни, что я люблю спокойную экспертную подачу",
            )

            memory_path = root / "workspace" / "MEMORY.md"
            self.assertNotIn("спокойную экспертную", memory_path.read_text(encoding="utf-8"))
            self.assertIsNotNone(bot.vault.get_pending_action(user.user_id))
            self.assertFalse(response.keyboard)
            self.assertIn("Ответь «сохрани»", response.text)

            bot.dispatch(user, "сохрани")

            self.assertIn("спокойную экспертную", memory_path.read_text(encoding="utf-8"))
            self.assertIsNone(bot.vault.get_pending_action(user.user_id))

    def test_personal_and_work_conversations_are_stored_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            user = TelegramUser(777, 777, "Partner")
            project = bot.workspace.ensure(user.user_id)

            bot.answer_with_llm(user, "Что быстро приготовить на ужин?")
            bot.answer_with_llm(user, "Предложи тему для рабочего Reels")

            personal = list((root / "personal" / "conversations").glob("*.md"))
            work = list((project.path / "conversations").glob("*.md"))
            self.assertEqual(len(personal), 1)
            self.assertEqual(len(work), 1)
            self.assertIn("ужин", personal[0].read_text(encoding="utf-8"))
            self.assertNotIn("ужин", work[0].read_text(encoding="utf-8"))

    def test_photo_is_downloaded_and_attached_to_codex(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)

            bot.handle_update(
                {
                    "update_id": 1,
                    "message": {
                        "from": {"id": 777, "first_name": "Partner"},
                        "chat": {"id": 777, "type": "private"},
                        "caption": "Что улучшить в этом кадре?",
                        "photo": [{"file_id": "small"}, {"file_id": "large"}],
                    },
                }
            )

            self.assertEqual(len(bot.chat_llm.image_calls), 1)
            attached = bot.chat_llm.image_calls[0][0]
            self.assertTrue(attached.is_file())
            self.assertTrue(attached.is_relative_to(root))
            self.assertIn("Принял", bot.telegram.messages[-1][1])

    def test_voice_is_transcribed_and_processed_as_normal_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            bot.deepgram = FakeDeepgram()

            bot.handle_update(
                {
                    "update_id": 1,
                    "message": {
                        "from": {"id": 777, "first_name": "Partner"},
                        "chat": {"id": 777, "type": "private"},
                        "voice": {"file_id": "voice-1"},
                    },
                }
            )

            self.assertNotIn("Распознал голосовое", bot.telegram.messages[-1][1])
            self.assertNotIn(
                "Запомни идею для живого ролика",
                bot.telegram.messages[-1][1],
            )
            self.assertIn("Ответь «сохрани»", bot.telegram.messages[-1][1])
            self.assertIsNotNone(bot.vault.get_pending_action(777))

    def test_plain_start_keeps_chat_first_and_exposes_shared_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = self.build_bot(Path(temp_dir) / "vault-partner")
            user = TelegramUser(777, 777, "Partner")
            bot.workspace.ensure(user.user_id)

            response = bot.main_menu(user)

            labels = [label for row in response.keyboard or [] for label, _ in row]
            self.assertEqual(
                labels,
                [
                    "📡 Радар идей для контент-завода",
                    "🔎 Референсы и каналы",
                    "📝 Сценарии",
                    "🖼 Создать изображение",
                    "📎 Файлы и списки",
                    "📤 В контент-завод",
                    "📊 Аналитика",
                    "⚙️ Настройки",
                ],
            )
            self.assertIn("Обычный чат", response.text)
            self.assertNotIn("Генерация видео", " ".join(labels))

    def test_auto_content_collects_channels_writes_script_and_hands_off(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            bot.youtube = FakeYouTubeClient()
            user = TelegramUser(777, 777, "Partner")
            bot.workspace.ensure(user.user_id)
            bot.research.import_accounts("youtube:@freshcreator")
            run = bot.research.create_run(
                "youtube",
                "",
                user.user_id,
                workflow="auto_content",
            )
            bot.research.update_run(run.run_id, "running")

            bot._research_worker(run.run_id, threading.Event())

            results = bot.research.list_results(run.run_id, limit=10)
            scripted = [item for item in results if item.script_path]
            self.assertEqual(len(scripted), 1)
            self.assertEqual(scripted[0].external_id, "fresh-video")
            self.assertTrue(scripted[0].shared_item_id.startswith("CR-"))
            shared = bot.shared_content.require(scripted[0].shared_item_id)
            self.assertEqual(shared.status, "handoff_requested")
            self.assertEqual(shared.item_kind, "production_idea")
            self.assertEqual(shared.metadata["kind"], "production_idea")
            self.assertEqual(
                shared.metadata["production_target"],
                "ai_video_content_factory",
            )
            self.assertIs(shared.metadata["requires_live_shoot"], False)
            self.assertEqual(shared.metadata["format"], "ai")
            self.assertEqual(shared.metadata["primary_result_id"], scripted[0].result_id)
            self.assertTrue(shared.metadata["evidence"])
            self.assertIn(
                str(root / "research" / "scripts"),
                scripted[0].script_path,
            )
            project = bot.vault.get_active_project(user.user_id)
            self.assertIsNotNone(project)
            self.assertEqual(list(project.path.joinpath("scripts").glob("*.md")), [])
            messages = "\n".join(message[1] for message in bot.telegram.messages)
            self.assertIn("1 из 4", messages)
            self.assertIn("Идея и сценарий готовы", messages)
            self.assertIn("Передача завершена", messages)
            self.assertIn("только AI-видео", messages)
            selection_prompt = next(
                call[-1]["content"]
                for call in bot.chat_llm.calls
                if "редактор-аналитик Reels" in call[0]["content"]
            )
            self.assertNotIn("Профиль партнёра:", selection_prompt)
            factory_system = next(
                call[0]["content"]
                for call in bot.chat_llm.calls
                if "производственный сценарист AI-video" in call[0]["content"]
            )
            self.assertIn("не личный ассистент", factory_system)
            handoff_message = next(
                message
                for message in bot.telegram.messages
                if "Передача завершена" in message[1]
            )
            button_targets = [
                target
                for row in handoff_message[2] or []
                for _, target in row
            ]
            self.assertIn(
                "url:https://t.me/ContentFactoryExampleBot"
                f"?start=idea_prod_{shared.item_id}",
                button_targets,
            )

    def test_auto_content_requires_saved_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = self.build_bot(Path(temp_dir) / "vault-partner")
            bot.youtube = FakeYouTubeClient()

            response = bot.prepare_auto_content(
                TelegramUser(777, 777, "Partner"),
                "youtube",
            )

            self.assertIn("нет сохранённых каналов", response.text)

    def test_auto_content_does_not_handoff_repeated_or_unrelated_idea(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            bot.youtube = FakeYouTubeClient()
            bot.workspace.ensure(777)
            bot.research.import_accounts("youtube:@freshcreator")

            for _ in range(2):
                run = bot.research.create_run(
                    "youtube",
                    "",
                    777,
                    workflow="auto_content",
                )
                bot.research.update_run(run.run_id, "running")
                bot._research_worker(run.run_id, threading.Event())

            shared = bot.shared_content.list_items(
                item_kinds={"production_idea"},
                limit=10,
            )
            messages = "\n".join(message[1] for message in bot.telegram.messages)
            self.assertEqual(len(shared), 1)
            self.assertIn("нет в источнике", messages)
            failed_run = bot.research.list_runs(limit=1)[0]
            self.assertEqual(failed_run.status, "failed")
            self.assertTrue(failed_run.error)
            failure_buttons = [
                target
                for message in bot.telegram.messages
                if "сценарий не создан" in message[1]
                for row in message[2] or []
                for _, target in row
            ]
            self.assertIn(
                f"research_content_retry:{failed_run.run_id}",
                failure_buttons,
            )
            selection_prompts = [
                call[-1]["content"]
                for call in bot.chat_llm.calls
                if "редактор-аналитик Reels" in call[0]["content"]
            ]
            self.assertIn(
                "Показать реальную автоматизацию на понятном бытовом примере",
                selection_prompts[-1],
            )

    def test_auto_content_retry_reuses_saved_results_without_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            bot.youtube = FakeYouTubeClient()
            bot.workspace.ensure(777)
            bot.research.import_accounts("youtube:@freshcreator")
            run = bot.research.create_run(
                "youtube", "", 777, workflow="auto_content"
            )
            bot.research.update_run(run.run_id, "running")
            with patch.object(
                bot,
                "_complete_auto_content",
                side_effect=ValueError("временная ошибка сценариста"),
            ):
                bot._research_worker(run.run_id, threading.Event())

            self.assertEqual(bot.research.require_run(run.run_id).status, "failed")
            with patch.object(bot, "_start_auto_content_retry_thread") as start:
                response = bot.retry_auto_content(
                    TelegramUser(777, 777, "Partner"),
                    run.run_id,
                )

            self.assertIn("Новый запрос Bright Data не создаётся", response.text)
            start.assert_called_once()
            self.assertEqual(bot.research.require_run(run.run_id).status, "running")

    def test_auto_content_survives_telegram_status_delivery_outage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            bot.telegram = OfflineTelegram()
            bot.youtube = FakeYouTubeClient()
            bot.workspace.ensure(777)
            bot.research.import_accounts("youtube:@freshcreator")
            run = bot.research.create_run(
                "youtube",
                "",
                777,
                workflow="auto_content",
            )
            bot.research.update_run(run.run_id, "running")

            with patch("agent_platform.partner_bot.print"):
                bot._research_worker(run.run_id, threading.Event())

            results = bot.research.list_results(run.run_id, limit=10)
            self.assertEqual(bot.research.require_run(run.run_id).status, "completed")
            self.assertEqual(len([item for item in results if item.shared_item_id]), 1)

    def test_shared_reference_batch_is_isolated_from_personal_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            user = TelegramUser(777, 777, "Partner")
            bot.workspace.ensure(user.user_id)
            private_marker = "PRIVATE-PARTNER-MARKER-DO-NOT-SHARE"
            memory_path = root / "workspace" / "MEMORY.md"
            memory_path.write_text(private_marker, encoding="utf-8")

            bot.start_shared_reference(user)
            state = bot.vault.get_input_state(user.user_id)
            response = bot.handle_input_state(
                user,
                "https://example.com/one\nhttps://example.com/two",
                state or {},
            )

            items = bot.shared_content.list_items(limit=10)
            self.assertEqual(len(items), 2)
            self.assertIn("Добавлено референсов: 2", response.text)
            self.assertIsNone(bot.vault.get_input_state(user.user_id))
            serialized = "\n".join(
                f"{item.source_text}\n{item.notes}" for item in items
            )
            self.assertNotIn(private_marker, serialized)
            self.assertFalse(bot.shared_content.root.is_relative_to(root))

            handed_off = bot.shared_content.handoff("partner", items[0].item_id)
            self.assertEqual(handed_off.status, "handoff_requested")

    def test_shared_photo_is_copied_to_shared_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            user = TelegramUser(777, 777, "Partner")
            bot.workspace.ensure(user.user_id)
            bot.start_shared_reference(user)

            response = bot.handle_shared_attachment(
                user,
                {
                    "caption": "Референс визуальной подачи",
                    "photo": [{"file_id": "small"}, {"file_id": "large"}],
                },
                "photo",
            )

            item = bot.shared_content.list_items(limit=1)[0]
            shared_file = bot.shared_content.root / item.media_path
            self.assertTrue(shared_file.is_file())
            self.assertEqual(shared_file.read_bytes(), b"telegram-media")
            self.assertIn(item.item_id, response.text)
            self.assertIsNone(bot.vault.get_input_state(user.user_id))

    def test_account_import_and_research_result_script_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            user = TelegramUser(777, 777, "Partner")
            bot.workspace.ensure(user.user_id)

            imported = bot.import_accounts(
                user,
                "youtube:@creator\ninstagram:creator.ig",
            )
            self.assertIn("Новых аккаунтов: 2", imported.text)

            run = bot.research.create_run("youtube", "automation", user.user_id)
            bot.research.update_run(run.run_id, "running")
            result = bot.research.save_results(
                run.run_id,
                [
                    {
                        "platform": "youtube",
                        "external_id": "video-1",
                        "title": "Automation case",
                        "source_url": "https://www.youtube.com/watch?v=video-1",
                        "creator": "Creator",
                        "views": 12000,
                        "likes": 500,
                        "description": "How a real workflow was automated",
                    }
                ],
            )[0]

            script_response = bot.generate_reference_script(user, result.result_id)
            handoff_response = bot.handoff_research_result(result.result_id)

            linked = bot.research.require_result(result.result_id)
            self.assertTrue(Path(linked.script_path).is_file())
            self.assertTrue(linked.shared_item_id.startswith("CR-"))
            shared = bot.shared_content.require(linked.shared_item_id)
            self.assertEqual(shared.status, "handoff_requested")
            self.assertIn("Сценарий сохранён", script_response.text)
            self.assertIn("передан владельцу", handoff_response.text)
            self.assertNotIn("PARTNER_PROFILE", shared.source_text)

    def test_multi_message_list_waits_for_finish_and_preserves_real_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            user = TelegramUser(777, 777, "Partner")
            bot.workspace.ensure(user.user_id)

            bot.start_list_collection(user)
            state = bot.vault.get_input_state(user.user_id) or {}
            batch_id = state["batch_id"]
            first_url = "https://www.instagram.com/first.creator/?igsh=one"
            second_url = "https://www.instagram.com/second.creator/?igsh=two"

            first = bot.handle_input_state(user, first_url, state)
            state = bot.vault.get_input_state(user.user_id) or {}
            second = bot.handle_input_state(
                user,
                f"{second_url}\n{first_url}",
                state,
            )

            self.assertIn("Часть 1 принята", first.text)
            self.assertIn("Часть 2 принята", second.text)
            self.assertEqual(len(bot.chat_llm.calls), 0)
            self.assertIsNone(bot.vault.get_pending_action(user.user_id))
            batch = bot.workbench.require_list(batch_id, user.user_id)
            self.assertIn(first_url, batch.text)
            self.assertIn(second_url, batch.text)

            completed = bot.finish_list_collection(user, batch_id)

            self.assertIn("Уникальных профилей: 2", completed.text)
            self.assertIn("Повторов внутри списка: 1", completed.text)
            self.assertEqual(len(bot.research.list_accounts("instagram")), 2)
            self.assertIsNone(bot.vault.get_input_state(user.user_id))

    def test_large_url_message_automatically_enters_list_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            user = TelegramUser(777, 777, "Partner")
            bot.workspace.ensure(user.user_id)

            response = bot.dispatch(
                user,
                "\n".join(
                    [
                        "https://www.instagram.com/one/",
                        "https://www.instagram.com/two/",
                        "https://www.instagram.com/three/",
                    ]
                ),
            )

            self.assertIsInstance(response, type(bot.main_menu(user)))
            self.assertIn("Часть 1 принята", response.text)
            self.assertEqual(len(bot.chat_llm.calls), 0)
            self.assertEqual(
                (bot.vault.get_input_state(user.user_id) or {}).get("kind"),
                "partner_list_collection",
            )

    def test_text_document_is_analyzed_without_memory_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)

            def download_text(file_id, destination):
                del file_id
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("Первый вывод\nВторой вывод", encoding="utf-8")
                return destination

            bot.telegram.download_file = download_text
            bot.handle_update(
                {
                    "update_id": 1,
                    "message": {
                        "from": {"id": 777, "first_name": "Partner"},
                        "chat": {"id": 777, "type": "private"},
                        "caption": "Сделай краткое резюме",
                        "document": {
                            "file_id": "document-1",
                            "file_name": "notes.md",
                            "file_size": 50,
                        },
                    },
                }
            )

            self.assertEqual(len(bot.chat_llm.calls), 1)
            prompt = bot.chat_llm.calls[0][-1]["content"]
            self.assertIn("Первый вывод", prompt)
            self.assertIn("Сделай краткое резюме", prompt)
            self.assertIn("не добавлен в долговременную память", bot.telegram.messages[-1][1])
            self.assertIsNone(bot.vault.get_pending_action(777))

    def test_bulk_message_cannot_create_memory_suggestion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            user = TelegramUser(777, 777, "Partner")
            bot.workspace.ensure(user.user_id)

            response = bot.answer_with_llm(
                user,
                "Запомни список:\nhttps://example.com/one\nhttps://example.com/two",
            )

            self.assertNotIn("Предлагаю сохранить", response.text)
            self.assertIsNone(bot.vault.get_pending_action(user.user_id))

    def test_handoff_without_id_opens_queue_instead_of_showing_technical_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)

            response = bot.handoff_shared_item("")

            self.assertIn("Общая очередь", response.text)
            self.assertNotIn("Укажи ID", response.text)

    def test_image_prompt_is_confirmed_and_actual_file_is_delivered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            bot = self.build_bot(root)
            bot.image_client = FakeImageClient()
            user = TelegramUser(777, 777, "Partner")
            bot.workspace.ensure(user.user_id)

            preview = bot.prepare_image(user, "Portrait on a clean white background")
            asset = bot.research.get_image(preview.text.split("·", 1)[1].splitlines()[0].strip())
            self.assertIsNotNone(asset)
            bot.research.update_image(asset.asset_id, "generating")

            bot._image_worker(user.chat_id, asset.asset_id)

            completed = bot.research.require_image(asset.asset_id)
            self.assertEqual(completed.status, "completed")
            self.assertTrue(Path(completed.file_path).is_file())
            self.assertEqual(Path(completed.file_path).read_bytes(), b"fake-png")
            self.assertEqual(len(bot.telegram.photos), 1)

    def test_settings_show_only_connection_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = self.build_bot(Path(temp_dir) / "vault-partner")

            response = bot.settings_view()

            self.assertIn("YouTube Data API", response.text)
            self.assertIn("Instagram / Bright Data", response.text)
            self.assertNotIn("test-token", response.text)


class PartnerConfigurationTest(unittest.TestCase):
    def test_prefixed_env_does_not_reuse_main_bot_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env.partner"
            env_path.write_text(
                "PARTNER_TELEGRAM_BOT_TOKEN=partner-token\n"
                "PARTNER_TELEGRAM_ALLOWED_USER_IDS=777\n"
                "PARTNER_TELEGRAM_TESTER_USER_IDS=888\n"
                f"PARTNER_VAULT_PATH={Path(temp_dir) / 'vault-partner'}\n"
                f"PARTNER_CODEX_WORKDIR={Path(temp_dir) / 'vault-partner'}\n"
                f"PARTNER_RADAR_VAULT_PATH={Path(temp_dir) / 'radar'}\n"
                "PARTNER_RADAR_REDIRECT_TO_CONTENT_FACTORY=true\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"TELEGRAM_BOT_TOKEN": "main-bot-token"},
                clear=True,
            ):
                settings = Settings.load(env_path, env_prefix="PARTNER_")

            self.assertEqual(settings.telegram_bot_token, "partner-token")
            self.assertEqual(settings.telegram_allowed_user_ids, {777})
            self.assertEqual(settings.telegram_tester_user_ids, {888})
            self.assertEqual(settings.test_vault_path, Path("vault-partner-test"))
            self.assertEqual(
                settings.test_shared_content_path,
                Path("shared-content-test"),
            )
            self.assertEqual(settings.radar_vault_path, Path(temp_dir) / "radar")
            self.assertTrue(settings.radar_redirect_to_content_factory)

    def test_loader_accepts_only_the_explicit_legacy_partner_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = root / ".env"
            legacy.write_text(
                "TELEGRAM_BOT_TOKEN=must-not-use\n"
                "TELEGRAM_BOT_TOKEN_Partner=partner-token\n"
                "DEEPGRAM_API_KEY=deepgram-secret\n"
                "BRIGHTDATA_API_TOKEN=brightdata-secret\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = load_partner_settings(root / ".env.partner", legacy)

            self.assertEqual(settings.telegram_bot_token, "partner-token")
            self.assertEqual(settings.deepgram_api_key, "deepgram-secret")
            self.assertEqual(settings.brightdata_api_token, "brightdata-secret")
            self.assertEqual(settings.vault_path, Path("vault-partner"))

    def test_validation_rejects_shared_vault_and_broad_codex_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            partner_vault = workspace / "vault-partner"
            valid = settings_for(partner_vault)
            validate_partner_settings(valid, workspace)

            shared = settings_for(workspace / "vault")
            with self.assertRaisesRegex(ValueError, "общий"):
                validate_partner_settings(shared, workspace)

            broad = settings_for(partner_vault)
            broad = Settings(
                **{
                    **broad.__dict__,
                    "codex_workdir": workspace,
                }
            )
            with self.assertRaisesRegex(ValueError, "PARTNER_CODEX_WORKDIR"):
                validate_partner_settings(broad, workspace)

            shared_inside_private = Settings(
                **{
                    **valid.__dict__,
                    "shared_content_path": partner_vault / "shared-content",
                }
            )
            with self.assertRaisesRegex(ValueError, "PARTNER_SHARED_CONTENT_PATH"):
                validate_partner_settings(shared_inside_private, workspace)

            tester_overlap = Settings(
                **{
                    **valid.__dict__,
                    "telegram_tester_user_ids": {777},
                }
            )
            with self.assertRaisesRegex(ValueError, "разные Telegram user_id"):
                validate_partner_settings(tester_overlap, workspace)

            tester_shared_vault = Settings(
                **{
                    **valid.__dict__,
                    "telegram_tester_user_ids": {888},
                    "test_vault_path": partner_vault,
                }
            )
            with self.assertRaisesRegex(ValueError, "PARTNER_TEST_VAULT_PATH"):
                validate_partner_settings(tester_shared_vault, workspace)

    def test_partner_codex_client_ignores_user_config_and_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vault-partner"
            root.mkdir()
            executable = Path(temp_dir) / "codex.exe"
            executable.write_bytes(b"test")
            client = create_partner_llm(settings_for(root, codex_path=executable))
            self.assertIsInstance(client, CodexExecClient)
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
                answer = client.chat([{"role": "user", "content": "Вопрос"}])

            self.assertEqual(answer, "Ответ")
            command = run.call_args.args[0]
            self.assertIn("--ephemeral", command)
            self.assertIn("read-only", command)
            self.assertIn("--ignore-user-config", command)
            self.assertIn("--ignore-rules", command)


if __name__ == "__main__":
    unittest.main()
