import tempfile
import unittest
from pathlib import Path

from agent_platform.config import Settings
from agent_platform.partner_bot import PartnerTelegramBot
from agent_platform.telegram_bot import AgentTelegramBot, TelegramUser


def settings_for(vault_path: Path, shared_path: Path, user_id: int) -> Settings:
    return Settings(
        telegram_bot_token="test-token",
        telegram_allowed_user_ids={user_id},
        vault_path=vault_path,
        openai_api_key="",
        openai_base_url="https://api.openai.com/v1",
        openai_image_model="gpt-image-2",
        openai_image_size="1024x1536",
        openai_image_quality="low",
        deepgram_api_key="",
        deepgram_model="nova-3",
        deepgram_language="ru",
        codex_cli_path="missing-codex.exe",
        codex_workdir=vault_path,
        shared_content_path=shared_path,
    )


class SharedBotWorkflowTest(unittest.TestCase):
    def test_reference_moves_between_isolated_bots_and_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shared_path = root / "shared-content"
            partner_vault = root / "vault-partner"
            owner_vault = root / "vault"
            partner = PartnerTelegramBot(settings_for(partner_vault, shared_path, 777))
            owner = AgentTelegramBot(settings_for(owner_vault, shared_path, 123))
            partner_user = TelegramUser(777, 777, "Partner")
            owner_user = TelegramUser(123, 123, "Owner")
            partner.workspace.ensure(partner_user.user_id)
            private_marker = "PARTNER-PRIVATE-CONTEXT"
            (partner_vault / "workspace" / "MEMORY.md").write_text(
                private_marker,
                encoding="utf-8",
            )

            created_response = partner.add_shared_references(
                "https://example.com/reel\nShow a business automation idea."
            )
            item = partner.shared_content.list_items(limit=1)[0]
            self.assertIn(item.item_id, created_response.text)
            self.assertNotIn(item.item_id, owner.shared_inbox().text)

            partner.handoff_shared_item(item.item_id)
            self.assertIn(item.item_id, owner.shared_inbox().text)
            owner.accept_shared_item(owner_user, item.item_id)

            restarted_partner = PartnerTelegramBot(
                settings_for(partner_vault, shared_path, 777)
            )
            restarted_owner = AgentTelegramBot(
                settings_for(owner_vault, shared_path, 123)
            )
            persisted = restarted_partner.shared_content.require(item.item_id)
            runs = restarted_owner._content_store().list_runs(limit=10)

            self.assertEqual(persisted.status, "in_production")
            self.assertEqual(len(runs), 1)
            self.assertEqual(persisted.linked_run_id, runs[0].run_id)
            self.assertNotIn(private_marker, persisted.source_text)
            self.assertNotIn(private_marker, persisted.notes)


if __name__ == "__main__":
    unittest.main()
