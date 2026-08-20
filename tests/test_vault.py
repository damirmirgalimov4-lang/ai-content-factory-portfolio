from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_platform.vault import VaultStore, slugify


class VaultStoreTest(unittest.TestCase):
    def test_slugify_keeps_readable_project_names(self) -> None:
        self.assertEqual(slugify("Content Factory"), "content-factory")
        self.assertEqual(slugify("Диплом проект"), "диплом-проект")

    def test_bootstrap_creates_workspace_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = VaultStore(Path(temp_dir))
            vault.ensure_bootstrap()

            self.assertTrue((Path(temp_dir) / "workspace" / "SOUL.md").exists())
            self.assertTrue((Path(temp_dir) / "workspace" / "MEMORY.md").exists())
            self.assertTrue((Path(temp_dir) / "workspace" / "MISSION.md").exists())
            self.assertTrue((Path(temp_dir) / "workspace" / "PROJECTS.md").exists())
            self.assertTrue((Path(temp_dir) / "workspace" / "PREFERENCES.md").exists())
            self.assertTrue((Path(temp_dir) / "agents" / "general.md").exists())
            self.assertTrue((Path(temp_dir) / "agents" / "researcher.md").exists())

    def test_project_lifecycle_and_notes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = VaultStore(Path(temp_dir))
            project = vault.create_project("Content Factory")

            self.assertEqual(project.slug, "content-factory")
            self.assertTrue((project.path / "TASKS.md").exists())
            self.assertTrue((project.path / "ops" / "tasks.json").exists())

            vault.set_active_project(user_id=123, slug_or_name="content-factory")
            note_path = vault.add_project_note(user_id=123, text="Первая идея ролика")

            self.assertIn("Первая идея ролика", note_path.read_text(encoding="utf-8"))
            self.assertIn("content-factory", vault.context_summary(user_id=123))

    def test_agents_can_be_created_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = VaultStore(Path(temp_dir))
            vault.create_agent("Researcher", "Ищет идеи и референсы")

            self.assertIn("researcher", vault.list_agents())
            self.assertIn("Ищет идеи", vault.read_agent("researcher") or "")

    def test_log_exchange_writes_user_and_assistant_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = VaultStore(Path(temp_dir))
            vault.create_project("Content Factory")
            vault.set_active_project(user_id=123, slug_or_name="content-factory")

            path = vault.log_exchange(
                user_id=123,
                user_text="Составь план",
                assistant_text="План готов",
            )

            content = path.read_text(encoding="utf-8")
            self.assertIn("### User", content)
            self.assertIn("Составь план", content)
            self.assertIn("### Assistant", content)
            self.assertIn("План готов", content)

    def test_tasks_can_be_created_listed_and_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = VaultStore(Path(temp_dir))
            project = vault.create_project("Content Factory")
            vault.set_active_project(user_id=123, slug_or_name="content-factory")

            task = vault.create_task(user_id=123, text="Описать цель проекта")
            self.assertEqual(task.task_id, "T001")

            open_tasks = vault.list_tasks(user_id=123)
            self.assertEqual(len(open_tasks), 1)
            self.assertEqual(open_tasks[0].text, "Описать цель проекта")

            completed = vault.complete_task(user_id=123, task_id="1")
            self.assertEqual(completed.status, "done")
            self.assertEqual(vault.list_tasks(user_id=123), [])

            tasks_markdown = (project.path / "TASKS.md").read_text(encoding="utf-8")
            self.assertIn("- [x] T001 Описать цель проекта", tasks_markdown)

    def test_pending_memory_action_can_be_approved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = VaultStore(Path(temp_dir))
            vault.ensure_bootstrap()
            vault.set_pending_action(
                user_id=123,
                action={
                    "type": "remember_global",
                    "text": "Пользователь любит короткие AI-видео",
                    "reason": "устойчивое предпочтение",
                },
            )

            result = vault.apply_pending_action(user_id=123)

            self.assertIn("глобальную память", result)
            self.assertIsNone(vault.get_pending_action(user_id=123))
            memory = (Path(temp_dir) / "workspace" / "MEMORY.md").read_text(encoding="utf-8")
            self.assertIn("Пользователь любит короткие AI-видео", memory)

    def test_pending_task_action_can_be_approved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = VaultStore(Path(temp_dir))
            vault.create_project("Content Factory")
            vault.set_active_project(user_id=123, slug_or_name="content-factory")
            vault.set_pending_action(
                user_id=123,
                action={
                    "type": "create_task",
                    "text": "Собрать 10 референсов",
                    "reason": "будущее действие",
                },
            )

            result = vault.apply_pending_action(user_id=123)

            self.assertIn("T001", result)
            self.assertEqual(vault.list_tasks(user_id=123)[0].text, "Собрать 10 референсов")

    def test_input_state_survives_store_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vault = VaultStore(root)
            vault.set_input_state(
                user_id=123,
                input_state={"kind": "content_revision", "run_id": "CF-20260712-001"},
            )

            restored = VaultStore(root).get_input_state(user_id=123)
            self.assertIsNotNone(restored)
            self.assertEqual(restored["kind"], "content_revision")
            self.assertEqual(restored["run_id"], "CF-20260712-001")

            vault.clear_input_state(user_id=123)
            self.assertIsNone(vault.get_input_state(user_id=123))


if __name__ == "__main__":
    unittest.main()
