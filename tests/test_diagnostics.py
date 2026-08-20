from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_platform.content_factory import ContentFactoryStore
from agent_platform.diagnostics import ReadOnlyRunDiagnostic
from agent_platform.vault import VaultStore


class DiagnosticLlm:
    is_configured = True

    def __init__(self) -> None:
        self.messages = []

    def chat(self, messages):
        self.messages = messages
        return "Причина: нарушен контракт. Безопасно уточнить данные и повторить этап."


class ReadOnlyRunDiagnosticTest(unittest.TestCase):
    def test_failed_stage_is_explained_and_report_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            content = ContentFactoryStore(root)
            run = content.create_run("Diagnostic idea")
            content.mark_failed(run.run_id, "VISUAL_BIBLE_CONTRACT отсутствует")
            vault = VaultStore(root)
            vault.ensure_bootstrap()
            (vault.agents / "producer.md").write_text(
                "# producer\nПроверяет бриф.", encoding="utf-8"
            )
            llm = DiagnosticLlm()

            report, target = ReadOnlyRunDiagnostic(content, vault, llm).diagnose(run.run_id)

            self.assertIn("нарушен контракт", report)
            self.assertTrue(target.is_file())
            self.assertIn("ничего не изменяй", llm.messages[0]["content"].lower())
            self.assertIn("VISUAL_BIBLE_CONTRACT отсутствует", llm.messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
