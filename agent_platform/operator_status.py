from __future__ import annotations

from pathlib import Path

from .config import Settings
from .telegram_bot import TelegramClient


def main() -> None:
    """Send the UTF-8 operator status file without shell-encoding round trips."""

    settings = Settings.load()
    if not settings.telegram_bot_token or not settings.telegram_allowed_user_ids:
        raise SystemExit("Telegram не настроен для отправки статуса.")
    path = settings.codex_workdir / "reports" / "CURRENT_WORK_STATUS.md"
    text = path.read_text(encoding="utf-8").strip()
    client = TelegramClient(settings.telegram_bot_token)
    for chat_id in sorted(settings.telegram_allowed_user_ids):
        client.send_message(
            chat_id,
            text,
            keyboard=[[('🛠 Открыть ход работы', 'menu:work_status')]],
            render_markdown=True,
        )


if __name__ == "__main__":
    main()
