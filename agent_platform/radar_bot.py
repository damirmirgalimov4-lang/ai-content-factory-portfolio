from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .config import Settings
from .partner_bot import PartnerTelegramBot
from .partner_research import ResearchError
from .radar_routes import RADAR_INPUT_KINDS, is_radar_callback
from .shared_content import SharedContentStore
from .telegram_bot import BotResponse, InlineKeyboard, TelegramClient, TelegramUser


class ContentFactoryRadarBot(PartnerTelegramBot):
    """Expose only the proven Radar surface through the main Telegram bot."""

    def __init__(self, settings: Settings):
        super().__init__(
            settings,
            test_mode=False,
            enable_tester_routing=False,
        )

    def attach_transport(
        self,
        telegram: TelegramClient,
        shared_content: SharedContentStore,
    ) -> None:
        self.telegram = telegram
        self.shared_content = shared_content

    def _prepare_runtime(self) -> None:
        self.workspace.ensure()
        self.shared_content.ensure()
        self.workbench.ensure()
        self.research.ensure()
        self._recover_research_runtime()

    def send_response(self, chat_id: int, response: str | BotResponse) -> None:
        if isinstance(response, BotResponse):
            response = self._adapt_response(response)
        super().send_response(chat_id, response)

    @staticmethod
    def handles_callback(data: str) -> bool:
        return is_radar_callback(data)

    @staticmethod
    def handles_input_state(state: dict[str, str] | None) -> bool:
        return bool(state and state.get("kind") in RADAR_INPUT_KINDS)

    def home(
        self,
        user: TelegramUser,
        *,
        replace_message: bool = False,
    ) -> BotResponse:
        state = self.vault.get_input_state(user.user_id)
        if self.handles_input_state(state):
            self.vault.clear_input_state(user.user_id)
        return self._adapt_response(
            self.auto_content_home(replace_message=replace_message)
        )

    def research_search_home(self, *, replace_message: bool = False) -> BotResponse:
        return self._adapt_response(
            self.research_home(replace_message=replace_message)
        )

    def results_home(self, *, replace_message: bool = False) -> BotResponse:
        return self._adapt_response(
            self.research_results(replace_message=replace_message)
        )

    def start_account_import(
        self,
        user: TelegramUser,
        replace_message: bool = False,
    ) -> BotResponse:
        self.vault.clear_pending_action(user.user_id)
        self.vault.set_input_state(
            user.user_id,
            {"kind": "research_account_import"},
        )
        return BotResponse(
            text=(
                "📥 Пришли список YouTube-каналов и Instagram-аккаунтов текстом "
                "или одним файлом TXT/CSV. Каждый аккаунт можно указать с новой строки."
            ),
            keyboard=[[('Отмена', 'input:cancel')]],
            replace_message=replace_message,
        )

    def handle_account_document(
        self,
        user: TelegramUser,
        message: dict[str, Any],
    ) -> BotResponse:
        return self._adapt_response(
            super().handle_account_document(user, message)
        )

    def dispatch_radar_callback(
        self,
        user: TelegramUser,
        data: str,
    ) -> BotResponse | None:
        state = self.vault.get_input_state(user.user_id)
        if self.handles_input_state(state):
            self.vault.clear_input_state(user.user_id)
        if data == "radar:home":
            return self.home(user, replace_message=True)
        if data.startswith("radar_shared_item:"):
            item_id = data.split(":", 1)[1]
            return self._adapt_response(
                super().dispatch_callback(user, f"shared_item:{item_id}")
            )
        if not self.handles_callback(data):
            return None
        return self._adapt_response(super().dispatch_callback(user, data))

    def dispatch_radar_input(
        self,
        user: TelegramUser,
        text: str,
        state: dict[str, str],
    ) -> BotResponse:
        return self._adapt_response(super().handle_input_state(user, text, state))

    def main_menu_keyboard(self) -> InlineKeyboard:
        return [[("« Главное меню", "menu:main")]]

    def _content_factory_idea_url(self, item_id: str) -> str:
        username = self.settings.content_factory_bot_username.strip().lstrip("@")
        if not re.fullmatch(r"[A-Za-z0-9_]{5,32}", username):
            raise ResearchError(
                "Username контент-завода не настроен.",
                code="content_factory_link_not_configured",
            )
        return f"https://t.me/{username}?start=idea_test_{item_id.strip().upper()}"

    @staticmethod
    def _adapt_response(response: BotResponse) -> BotResponse:
        prepared = PartnerTelegramBot._prepare_response(response)
        keyboard = None
        if prepared.keyboard is not None:
            keyboard = [
                [
                    (label, ContentFactoryRadarBot._adapt_callback(callback))
                    for label, callback in row
                ]
                for row in prepared.keyboard
            ]
        return replace(prepared, keyboard=keyboard)

    @staticmethod
    def _adapt_callback(callback: str) -> str:
        if callback.startswith("shared_item:"):
            return f"radar_shared_item:{callback.split(':', 1)[1]}"
        if callback == "shared:list":
            return "ideas:list"
        if callback.startswith("shared_"):
            return "menu:main"
        if callback.startswith("partner:"):
            return "menu:main"
        return callback