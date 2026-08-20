from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackagingMetadataTests(unittest.TestCase):
    def test_storyboard_prompt_assets_are_included_in_installed_package(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        package_data = pyproject["tool"]["setuptools"]["package-data"]

        self.assertEqual(
            ["prompt_templates/*.txt", "prompt_templates/*.json"],
            package_data["agent_platform"],
        )


if __name__ == "__main__":
    unittest.main()
