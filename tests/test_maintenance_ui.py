from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_platform.config import Settings
from agent_platform.maintenance import RepairRecord
from agent_platform.telegram_bot import AgentTelegramBot, TelegramUser


class FakeRepairManager:
    is_configured = True

    def __init__(self, record: RepairRecord | None = None) -> None:
        self.record = record

    def latest_for_run(self, run_id: str) -> RepairRecord | None:
        if self.record and self.record.run_id == run_id:
            return self.record
        return None

    def get(self, repair_id: str) -> RepairRecord:
        if self.record is None or self.record.repair_id != repair_id:
            raise ValueError("repair not found")
        return self.record

    @staticmethod
    def describe(record: RepairRecord) -> str:
        return f"Ремонт {record.repair_id}: {record.status}"


def settings_for(root: Path) -> Settings:
    return Settings(
        telegram_bot_token="test-token",
        telegram_allowed_user_ids={123},
        vault_path=root,
        openai_api_key="",
        openai_base_url="https://api.openai.com/v1",
        openai_image_model="gpt-image-2",
        openai_image_size="1024x1536",
        openai_image_quality="low",
        deepgram_api_key="",
        deepgram_model="nova-3",
        deepgram_language="ru",
        poll_timeout_seconds=1,
    )


class MaintenanceUiTest(unittest.TestCase):
    def test_failed_run_offers_isolated_repair_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            bot.repair_manager = FakeRepairManager()
            run = bot._content_store().create_run("Broken stage")
            bot._content_store().mark_failed(run.run_id, "contract failure")
            user = TelegramUser(123, 123, "Owner")

            detail = bot.run_detail(run.run_id)
            review = bot.dispatch_callback(user, f"repair_review:{run.run_id}")

            labels = [label for row in detail.keyboard or [] for label, _ in row]
            self.assertIn("🧰 Подготовить исправление", labels)
            self.assertIn("отдельную Git-ветку", review.text)
            self.assertIn("отдельного подтверждения", review.text)
            callbacks = [callback for row in review.keyboard or [] for _, callback in row]
            self.assertIn(f"repair_start:{run.run_id}", callbacks)

    def test_ready_repair_requires_explicit_apply_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = AgentTelegramBot(settings_for(Path(temp_dir)))
            run = bot._content_store().create_run("Broken stage")
            bot._content_store().mark_failed(run.run_id, "contract failure")
            record = RepairRecord(
                repair_id="RP-A1B2C3D4",
                run_id=run.run_id,
                status="ready",
                branch="bot-repair/rp-a1b2c3d4",
                base_commit="base",
                worktree_path="worktree",
                created_at="2026-07-18T00:00:00+03:00",
                updated_at="2026-07-18T00:00:00+03:00",
                commit="commit",
                changed_files=("agent_platform/content_factory.py",),
                checks=("✅ Полный набор unit-тестов",),
            )
            bot.repair_manager = FakeRepairManager(record)
            user = TelegramUser(123, 123, "Owner")

            status = bot.dispatch_callback(user, f"repair_status:{record.repair_id}")
            review = bot.dispatch_callback(
                user,
                f"repair_apply_review:{record.repair_id}",
            )

            status_callbacks = [
                callback for row in status.keyboard or [] for _, callback in row
            ]
            review_callbacks = [
                callback for row in review.keyboard or [] for _, callback in row
            ]
            self.assertIn(
                f"repair_apply_review:{record.repair_id}", status_callbacks
            )
            self.assertNotIn(f"repair_apply:{record.repair_id}", status_callbacks)
            self.assertIn(f"repair_apply:{record.repair_id}", review_callbacks)
            self.assertIn("перезапуск Telegram-бота", review.text)


if __name__ == "__main__":
    unittest.main()
