#!/usr/bin/env python3
"""Fail-closed repository scan for credentials and private portfolio data."""

from __future__ import annotations

import ipaddress
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cmd", ".json", ".lock", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"
}
FORBIDDEN_DIRS = {"agent-training-notes", "budget-google-sheet", "reports", "vault"}
PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{20,}\b"),
    "OpenAI-compatible key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Telegram bot token": re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Google API key": re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "credentialed URL": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
    "absolute private path": re.compile(r"/(?:root|home/[^/]+|etc/content-factory|opt/content-factory)/"),
    "numeric Telegram identity": re.compile(r"(?:chat|user)_id\s*(?:=|:)\s*[1-9]\d{7,}", re.I),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b", re.I),
    "UUID": re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I),
}
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
ALLOWED_EMAIL_DOMAINS = {"example.com", "example.org", "users.noreply.github.com"}
COMFYUI_WORKFLOW_WITH_NODE_UUIDS = Path("ltx_worker/assets/video_ltx2_3_i2v.json")
ALLOWED_IP_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in ("0.0.0.0/32", "127.0.0.0/8", "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
)
ENV_LITERAL = re.compile(
    r'(?:getenv|environ\.get|environ\[|pick)\(\s*["\']([A-Za-z][A-Za-z0-9_]*)'
)
ENV_COMPATIBILITY_ALIASES = {
    "CODEX_HOME",
    "OS_AUTH_URL",
    "OS_COMPUTE_URL",
    "OS_PASSWORD",
    "OS_PROJECT_NAME",
    "OS_USERNAME",
    "OS_USER_DOMAIN_NAME",
    "PolzaAi_API_KEY",
    "TELEGRAM_CHAT_ID",
    "kie",
}


def repository_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return sorted(
        ROOT / raw_path.decode("utf-8")
        for raw_path in listed.split(b"\0")
        if raw_path and (ROOT / raw_path.decode("utf-8")).is_file()
    )


def text_files() -> list[Path]:
    result: list[Path] = []
    for path in repository_files():
        if (
            path.suffix.lower() in TEXT_SUFFIXES
            or path.name.startswith(".env")
            or not path.suffix
        ):
            result.append(path)
    return sorted(result)


def scan() -> list[str]:
    findings: list[str] = []
    for path in repository_files():
        if any(part in FORBIDDEN_DIRS for part in path.relative_to(ROOT).parts):
            findings.append(f"forbidden path: {path.relative_to(ROOT)}")

    for path in text_files():
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                if label == "email address":
                    domain = match.group(1).casefold()
                    if domain in ALLOWED_EMAIL_DOMAINS or domain.endswith(".invalid"):
                        continue
                if label == "UUID" and relative == COMFYUI_WORKFLOW_WITH_NODE_UUIDS:
                    continue
                findings.append(f"{label}: {relative}:{text.count(chr(10), 0, match.start()) + 1}")
        for match in IPV4.finditer(text):
            # Browser versions in a User-Agent can be syntactically identical to IPv4.
            if text[max(0, match.start() - 7):match.start()] == "Chrome/":
                continue
            try:
                address = ipaddress.ip_address(match.group(0))
            except ValueError:
                continue
            if not any(address in network for network in ALLOWED_IP_NETWORKS):
                findings.append(
                    f"non-example IP address: {relative}:{text.count(chr(10), 0, match.start()) + 1}"
                )

    runtime_code = "\n".join(
        path.read_text(encoding="utf-8")
        for package in (ROOT / "agent_platform", ROOT / "ltx_worker")
        for path in package.glob("*.py")
    )
    runtime_env = set(ENV_LITERAL.findall(runtime_code)) - ENV_COMPATIBILITY_ALIASES
    documented_env: set[str] = set()
    for example in (ROOT / ".env.example", ROOT / ".env.partner.example"):
        for line in example.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "=" in line:
                documented_env.add(line.split("=", 1)[0])
    for name in sorted(runtime_env - documented_env):
        findings.append(f"undocumented environment variable: {name}")

    return sorted(set(findings))


def main() -> int:
    findings = scan()
    if findings:
        print("SECURITY_SCAN_FAILED")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"SECURITY_SCAN_OK files={len(text_files())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
