from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cmd",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_DIRECTORIES = {
    "agent-training-notes",
    "budget-google-sheet",
    "reports",
    "vault",
}
HIGH_CONFIDENCE_SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Telegram bot token": re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "credentialed URL": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
    "private IPv4 address": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
    "absolute private path": re.compile(r"/(?:root|home/[^/]+|etc/content-factory|opt/content-factory)/"),
    "numeric Telegram identity": re.compile(
        r"(?:chat|user)_id\s*(?:=|:)\s*[1-9]\d{7,}", re.IGNORECASE
    ),
}


def iter_public_text_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    files: list[Path] = []
    for raw_path in listed.split(b"\0"):
        if not raw_path:
            continue
        path = ROOT / raw_path.decode("utf-8")
        if not path.is_file() or path == Path(__file__).resolve():
            continue
        if (
            path.suffix.lower() in TEXT_SUFFIXES
            or path.name.startswith(".env")
            or not path.suffix
        ):
            files.append(path)
    return files


class PublicRepositorySafetyTests(unittest.TestCase):
    def test_private_directories_are_not_published(self) -> None:
        present = sorted(
            part
            for path in ROOT.rglob("*")
            for part in path.relative_to(ROOT).parts
            if part in FORBIDDEN_DIRECTORIES
        )
        self.assertEqual([], present)

    def test_extensionless_release_files_are_scanned(self) -> None:
        relative_paths = {
            path.relative_to(ROOT).as_posix() for path in iter_public_text_files()
        }

        self.assertIn(".gitignore", relative_paths)
        self.assertIn("LICENSE", relative_paths)
        self.assertIn("ltx_worker/assets/WORKFLOW-TEMPLATES-LICENSE", relative_paths)

    def test_ignored_build_artifacts_are_not_scanned(self) -> None:
        relative_paths = {
            path.relative_to(ROOT).as_posix() for path in iter_public_text_files()
        }

        self.assertFalse(any(path.startswith("build/") for path in relative_paths))
        self.assertFalse(any(".egg-info/" in path for path in relative_paths))

    def test_high_confidence_secrets_and_private_paths_are_absent(self) -> None:
        findings: list[str] = []
        for path in iter_public_text_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            for label, pattern in HIGH_CONFIDENCE_SECRET_PATTERNS.items():
                if pattern.search(text):
                    findings.append(f"{path.relative_to(ROOT)}: {label}")
        self.assertEqual([], findings)


if __name__ == "__main__":
    unittest.main()
