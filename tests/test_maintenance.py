from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_platform.content_factory import ContentFactoryStore
from agent_platform.maintenance import GitRepairManager, RepairError


class StubRepairManager(GitRepairManager):
    def __init__(self, *args, create_forbidden: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.executable = Path("fake-codex")
        self.create_forbidden = create_forbidden

    def _run_codex(self, worktree: Path, prompt: str) -> str:
        self.assert_safe_prompt(prompt)
        if self.create_forbidden:
            (worktree / ".env").write_text("SECRET=must-not-exist\n", encoding="utf-8")
        else:
            (worktree / "agent_platform" / "sample.py").write_text(
                "VALUE = 2\n", encoding="utf-8"
            )
            (worktree / "tests" / "test_sample.py").write_text(
                "import unittest\n"
                "from agent_platform.sample import VALUE\n\n"
                "class SampleTest(unittest.TestCase):\n"
                "    def test_value(self):\n"
                "        self.assertEqual(VALUE, 2)\n",
                encoding="utf-8",
            )
        return "Исправлена проверяемая ошибка и добавлен регрессионный тест."

    @staticmethod
    def assert_safe_prompt(prompt: str) -> None:
        if "codex login/logout" not in prompt or "платные" not in prompt:
            raise AssertionError("repair prompt lost its safety boundaries")


@unittest.skipUnless(shutil.which("git"), "Git is required for maintenance tests")
class GitRepairManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "agent_platform").mkdir()
        (self.root / "agent_platform" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "agent_platform" / "sample.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_sample.py").write_text(
            "import unittest\n"
            "from agent_platform.sample import VALUE\n\n"
            "class SampleTest(unittest.TestCase):\n"
            "    def test_value(self):\n"
            "        self.assertEqual(VALUE, 1)\n",
            encoding="utf-8",
        )
        (self.root / ".gitignore").write_text(
            ".tmp/\nvault/projects/*/runs/\n__pycache__/\n",
            encoding="utf-8",
        )
        self.git("init")
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Tests",
            "-c",
            "user.email=tests@local.invalid",
            "commit",
            "-m",
            "initial test repository",
        )
        project = self.root / "vault" / "projects" / "content-factory"
        self.content = ContentFactoryStore(project)
        self.run = self.content.create_run("Repair a failed content stage")
        self.content.mark_failed(self.run.run_id, "sample value is stale")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )

    def manager(self, *, forbidden: bool = False) -> StubRepairManager:
        return StubRepairManager(
            self.root,
            self.content,
            create_forbidden=forbidden,
            timeout_seconds=30,
        )

    def test_repair_is_committed_only_in_isolated_branch_then_applied(self) -> None:
        manager = self.manager()
        queued = manager.create(self.run.run_id)
        prepared = manager.prepare(queued.repair_id)

        self.assertEqual(prepared.status, "ready")
        self.assertTrue(prepared.commit)
        self.assertIn("agent_platform/sample.py", prepared.changed_files)
        self.assertIn("tests/test_sample.py", prepared.changed_files)
        self.assertEqual(
            (self.root / "agent_platform" / "sample.py").read_text(encoding="utf-8"),
            "VALUE = 1\n",
        )
        self.assertTrue(all(check.startswith("✅") for check in prepared.checks))

        applied = manager.apply(prepared.repair_id)

        self.assertEqual(applied.status, "applied")
        self.assertTrue(applied.applied_commit)
        self.assertEqual(
            (self.root / "agent_platform" / "sample.py").read_text(encoding="utf-8"),
            "VALUE = 2\n",
        )

    def test_apply_stops_when_main_has_changes_in_same_file(self) -> None:
        manager = self.manager()
        prepared = manager.prepare(manager.create(self.run.run_id).repair_id)
        (self.root / "agent_platform" / "sample.py").write_text(
            "VALUE = 99\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(RepairError, "незакоммиченные изменения"):
            manager.apply(prepared.repair_id)

        self.assertEqual(manager.get(prepared.repair_id).status, "ready")
        self.assertEqual(
            (self.root / "agent_platform" / "sample.py").read_text(encoding="utf-8"),
            "VALUE = 99\n",
        )

    def test_forbidden_secret_file_blocks_repair_even_when_git_ignores_it(self) -> None:
        manager = self.manager(forbidden=True)
        record = manager.create(self.run.run_id)

        with self.assertRaisesRegex(RepairError, "секретный файл"):
            manager.prepare(record.repair_id)

        failed = manager.get(record.repair_id)
        self.assertEqual(failed.status, "failed")
        self.assertFalse(failed.commit)

    def test_subprocess_environment_excludes_service_secrets(self) -> None:
        previous = os.environ.get("POLZA_AI_API_KEY")
        os.environ["POLZA_AI_API_KEY"] = "do-not-pass"
        try:
            safe = GitRepairManager._safe_environment()
        finally:
            if previous is None:
                os.environ.pop("POLZA_AI_API_KEY", None)
            else:
                os.environ["POLZA_AI_API_KEY"] = previous

        self.assertNotIn("POLZA_AI_API_KEY", safe)
        self.assertIn("PATH", {key.upper() for key in safe})

    def test_repair_scope_excludes_personal_bot_and_general_memory_agent(self) -> None:
        manager = self.manager()

        with self.assertRaisesRegex(RepairError, "запрещённый файл"):
            manager._validate_changed_files(
                self.root,
                ["agent_platform/partner_bot.py"],
            )
        with self.assertRaisesRegex(RepairError, "запрещённый файл"):
            manager._validate_changed_files(
                self.root,
                ["vault/agents/general.md"],
            )

    def test_evidence_redaction_masks_tokens_nested_in_payload(self) -> None:
        payload = GitRepairManager._redact_payload(
            {"idea": "token: sk_exampleabcdefghijklmnop", "nested": ["safe"]}
        )

        self.assertNotIn("sk_example", str(payload))
        self.assertIn("[REDACTED]", str(payload))


if __name__ == "__main__":
    unittest.main()
