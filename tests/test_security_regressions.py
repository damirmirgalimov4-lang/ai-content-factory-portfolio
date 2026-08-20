from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_platform.content_factory import ContentFactoryStore
from agent_platform.production import ProductionStore, SceneSpec


class SecurityRegressionTest(unittest.TestCase):
    def test_env_is_ignored_and_not_tracked(self) -> None:
        root = Path(__file__).resolve().parents[1]
        ignored = subprocess.run(
            ["git", "check-ignore", ".env"], cwd=root, capture_output=True, text=True
        )
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(ignored.returncode, 0)
        self.assertNotEqual(tracked.returncode, 0)

    def test_openrouter_is_absent_from_active_runtime_and_env_example(self) -> None:
        root = Path(__file__).resolve().parents[1]
        active_files = list((root / "agent_platform").glob("*.py")) + [
            root / ".env.example"
        ]
        for path in active_files:
            self.assertNotIn(
                "openrouter",
                path.read_text(encoding="utf-8").lower(),
                msg=f"Устаревший провайдер остался в {path.name}",
            )

    def test_production_writes_only_inside_run_and_preserves_reference_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            izuchi = root / "изучи" / "source.txt"
            prompter = root / "Папка промтер" / "source.txt"
            izuchi.parent.mkdir()
            prompter.parent.mkdir()
            izuchi.write_text("original izuchi", encoding="utf-8")
            prompter.write_text("original prompter", encoding="utf-8")
            run = ContentFactoryStore(root).create_run("Isolation")
            store = ProductionStore(root)
            store.save_scene_contract(
                run.run_id,
                [SceneSpec(
                    "S01", 1, 3, "purpose", "visual", "action", "static", "", "", "", "cut",
                    {}, "one image",
                )],
            )
            attempt = store.start_frame(run.run_id, "S01")
            store.complete_frame(run.run_id, "S01", attempt, b"image", ".png")
            self.assertEqual(izuchi.read_text(encoding="utf-8"), "original izuchi")
            self.assertEqual(prompter.read_text(encoding="utf-8"), "original prompter")


if __name__ == "__main__":
    unittest.main()
