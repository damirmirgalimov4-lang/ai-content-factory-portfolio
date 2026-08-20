from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_platform.archive_import import (
    extract_messages,
    inventory_archive,
    sanitize_text,
    write_inventory,
    write_messages,
)


class ArchiveImportTest(unittest.TestCase):
    def test_sanitize_text_redacts_common_secrets(self) -> None:
        telegram_secret = "123456789" + ":" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ123"
        openai_secret = "sk-" + "ant-" + ("a" * 32)
        text = f"token {telegram_secret} and {openai_secret}"

        sanitized = sanitize_text(text)
        self.assertNotIn(telegram_secret, sanitized)
        self.assertNotIn(openai_secret, sanitized)
        self.assertIn("REDACTED", sanitized)

    def test_inventory_and_message_extraction(self) -> None:
        html = """
        <html><body>
        <div class="message default clearfix" id="message1">
          <div class="body">
            <div class="pull_right date details" title="01.01.2026 10:00:00 UTC+03:00">10:00</div>
            <div class="from_name">ProjectOwner</div>
            <div class="text">Сделай план <a href="files/brief.md">brief</a></div>
          </div>
        </div>
        </body></html>
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "archive"
            archive.mkdir()
            (archive / "messages.html").write_text(html, encoding="utf-8")
            (archive / "files").mkdir()
            (archive / "files" / "brief.md").write_text("brief", encoding="utf-8")

            inventory = inventory_archive(archive)
            self.assertEqual(inventory["total_files"], 2)

            output = Path(temp_dir) / "out"
            write_inventory(inventory, output)
            self.assertTrue((output / "inventory.md").exists())

            messages = extract_messages(archive)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].sender, "ProjectOwner")
            self.assertIn("Сделай план", messages[0].text)
            self.assertEqual(messages[0].attachments, ["files/brief.md"])

            write_messages(messages, output)
            self.assertTrue((output / "text" / "messages.md").exists())


if __name__ == "__main__":
    unittest.main()
