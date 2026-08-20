from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_platform.telegram_bot import TelegramApiError, TelegramClient


class TelegramClientTest(unittest.TestCase):
    def test_attachment_is_downloaded_to_requested_path(self) -> None:
        client = TelegramClient("test-token")
        client._request = lambda method, payload: {  # type: ignore[method-assign]
            "ok": True,
            "result": {"file_path": "voice/file.ogg"},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b"voice-bytes"

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "nested" / "voice.ogg"
            with patch("agent_platform.telegram_bot.urllib.request.urlopen", return_value=FakeResponse()) as urlopen:
                result = client.download_file("file-id", destination)

            self.assertEqual(result.read_bytes(), b"voice-bytes")
            self.assertIn("/file/bottest-token/voice/file.ogg", urlopen.call_args.args[0].full_url)

    def test_native_command_menu_is_registered(self) -> None:
        client = TelegramClient("test-token")
        calls: list[tuple[str, dict]] = []
        client._request = lambda method, payload: calls.append((method, payload)) or {"ok": True}  # type: ignore[method-assign]

        client.set_commands(
            (("start", "Открыть главное меню"), ("new_video", "Создать ролик"))
        )
        client.set_commands_menu_button()

        self.assertEqual(calls[0][0], "setMyCommands")
        self.assertEqual(
            calls[0][1]["commands"],
            [
                {"command": "start", "description": "Открыть главное меню"},
                {"command": "new_video", "description": "Создать ролик"},
            ],
        )
        self.assertEqual(
            calls[1],
            ("setChatMenuButton", {"menu_button": {"type": "commands"}}),
        )

    def test_encoding_corrupted_text_and_buttons_are_not_sent(self) -> None:
        client = TelegramClient("test-token")
        calls: list[tuple[str, dict]] = []
        client._request = lambda method, payload: calls.append((method, payload)) or {"ok": True}  # type: ignore[method-assign]

        with self.assertRaises(TelegramApiError):
            client.send_message(chat_id=123, text="???????? ??????")
        with self.assertRaises(TelegramApiError):
            client.send_message(
                chat_id=123,
                text="Нормальный текст",
                keyboard=[[('???? ??????', 'menu:main')]],
            )
        self.assertEqual(calls, [])

        client.send_message(
            chat_id=123,
            text="Причина сообщений `????` найдена и исправлена.",
        )
        self.assertEqual(len(calls), 1)

    def test_inline_keyboard_is_sent_only_with_last_chunk(self) -> None:
        client = TelegramClient("test-token")
        calls: list[tuple[str, dict]] = []
        client._request = lambda method, payload: calls.append((method, payload)) or {"ok": True}  # type: ignore[method-assign]

        client.send_message(
            chat_id=123,
            text="Первая часть\n" + ("x" * 4000),
            keyboard=[[('Продолжить', 'cf_next:CF-20260712-001')]],
        )

        self.assertGreater(len(calls), 1)
        self.assertNotIn("reply_markup", calls[0][1])
        markup = calls[-1][1]["reply_markup"]
        self.assertEqual(markup["inline_keyboard"][0][0]["text"], "Продолжить")
        self.assertEqual(
            markup["inline_keyboard"][0][0]["callback_data"],
            "cf_next:CF-20260712-001",
        )

    def test_inline_keyboard_supports_direct_telegram_links(self) -> None:
        client = TelegramClient("test-token")
        calls: list[tuple[str, dict]] = []
        client._request = lambda method, payload: calls.append((method, payload)) or {"ok": True}  # type: ignore[method-assign]

        client.send_message(
            chat_id=123,
            text="Пакет передан.",
            keyboard=[
                [
                    (
                        "Открыть контент-завод",
                        "url:https://t.me/ContentFactoryExampleBot?start=idea_test_CR-20260727-001",
                    )
                ]
            ],
        )

        button = calls[-1][1]["reply_markup"]["inline_keyboard"][0][0]
        self.assertEqual(
            button,
            {
                "text": "Открыть контент-завод",
                "url": "https://t.me/ContentFactoryExampleBot?start=idea_test_CR-20260727-001",
            },
        )

    def test_markdown_response_is_sent_as_safe_telegram_html(self) -> None:
        client = TelegramClient("test-token")
        calls: list[tuple[str, dict]] = []
        client._request = lambda method, payload: calls.append((method, payload)) or {"ok": True}  # type: ignore[method-assign]

        client.send_message(
            chat_id=123,
            text="# Бриф\n\n**Важно:** <не выдумывать>",
            render_markdown=True,
        )

        payload = calls[0][1]
        self.assertEqual(payload["parse_mode"], "HTML")
        self.assertIn("<b>Бриф</b>", payload["text"])
        self.assertIn("<b>Важно:</b>", payload["text"])
        self.assertIn("&lt;не выдумывать&gt;", payload["text"])


if __name__ == "__main__":
    unittest.main()
