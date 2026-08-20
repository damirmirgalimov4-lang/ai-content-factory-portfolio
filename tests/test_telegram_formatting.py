from __future__ import annotations

import unittest

from agent_platform.telegram_formatting import (
    markdown_to_telegram_html,
    normalize_telegram_markdown,
)


class TelegramFormattingTest(unittest.TestCase):
    def test_normalizer_separates_sections_and_collapses_blank_lines(self) -> None:
        normalized = normalize_telegram_markdown(
            "Главный вывод.\n\n\nКадр 1\nОписание\nКадр 2\nОписание"
        )

        self.assertEqual(
            normalized,
            "Главный вывод.\n\nКадр 1\nОписание\n\nКадр 2\nОписание",
        )

    def test_formats_headings_bold_lists_quotes_and_code(self) -> None:
        rendered = markdown_to_telegram_html(
            "# Заголовок\n\n**Важно**\n- Пункт\n> Допущение\n`code`"
        )

        self.assertIn("<b>Заголовок</b>", rendered)
        self.assertIn("<b>Важно</b>", rendered)
        self.assertIn("- Пункт", rendered)
        self.assertIn("<blockquote>Допущение</blockquote>", rendered)
        self.assertIn("<code>code</code>", rendered)

    def test_escapes_unsafe_html(self) -> None:
        rendered = markdown_to_telegram_html("<script>bad()</script>")

        self.assertEqual(rendered, "&lt;script&gt;bad()&lt;/script&gt;")

    def test_renders_markdown_table_as_preformatted_text(self) -> None:
        rendered = markdown_to_telegram_html(
            "| Кадр | Действие |\n|---|---|\n| 1 | Старт |"
        )

        self.assertTrue(rendered.startswith("<pre>"))
        self.assertNotIn("|---|", rendered)


if __name__ == "__main__":
    unittest.main()
