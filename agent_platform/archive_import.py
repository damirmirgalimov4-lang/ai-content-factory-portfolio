from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .config import Settings
from .deepgram import DeepgramClient, TranscriptionError
from .vault import slugify


ARCHIVE_IMPORT_SLUG = "old-bot-telegram-2026-07-09"
SECRET_PATTERNS = [
    re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-or-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}\b"),
]


@dataclass
class ArchiveMessage:
    message_id: str
    sender: str = ""
    date: str = ""
    text: str = ""
    attachments: list[str] = field(default_factory=list)


class TelegramHtmlParser(HTMLParser):
    """Extracts Telegram export messages without depending on external parsers."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.messages: list[ArchiveMessage] = []
        self.current: ArchiveMessage | None = None
        self.message_depth = 0
        self.active_fields: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        classes = set(attr.get("class", "").split())

        if tag == "div" and "message" in classes and attr.get("id"):
            self.current = ArchiveMessage(message_id=attr["id"])
            self.message_depth = 1
            self.active_fields.clear()
            return

        if self.current is None:
            return

        if tag == "div":
            self.message_depth += 1
            if "from_name" in classes:
                self.active_fields.append("sender")
            elif "text" in classes:
                self.active_fields.append("text")
            elif "date" in classes and "details" in classes and attr.get("title"):
                self.current.date = attr["title"].strip()

        if tag == "a":
            href = attr.get("href", "").strip()
            if href and self._is_attachment(href):
                self.current.attachments.append(href)

    def handle_endtag(self, tag: str) -> None:
        if self.current is None:
            return

        if tag == "div":
            if self.active_fields:
                self.active_fields.pop()
            self.message_depth -= 1
            if self.message_depth <= 0:
                self.current.sender = clean_whitespace(self.current.sender)
                self.current.text = sanitize_text(clean_whitespace(self.current.text))
                self.current.attachments = sorted(set(self.current.attachments))
                self.messages.append(self.current)
                self.current = None
                self.message_depth = 0
                self.active_fields.clear()

    def handle_data(self, data: str) -> None:
        if self.current is None or not self.active_fields:
            return

        field_name = self.active_fields[-1]
        if field_name == "sender":
            self.current.sender += data
        elif field_name == "text":
            self.current.text += data

    def _is_attachment(self, href: str) -> bool:
        return href.startswith(
            (
                "voice_messages/",
                "photos/",
                "files/",
                "video_files/",
                "round_video_messages/",
            )
        )


def clean_whitespace(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def sanitize_text(value: str) -> str:
    result = value
    for pattern in SECRET_PATTERNS:
        result = pattern.sub("REDACTED", result)
    return result


def import_root(settings: Settings, project_slug: str) -> Path:
    return (
        settings.vault_path
        / "projects"
        / slugify(project_slug)
        / "imports"
        / ARCHIVE_IMPORT_SLUG
    )


def inventory_archive(archive_path: Path) -> dict[str, Any]:
    files = sorted(path for path in archive_path.rglob("*") if path.is_file())
    extension_counts = Counter(path.suffix.lower() or "(no extension)" for path in files)
    total_bytes = sum(path.stat().st_size for path in files)
    return {
        "archive_path": str(archive_path),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_files": len(files),
        "total_bytes": total_bytes,
        "extension_counts": dict(sorted(extension_counts.items())),
        "files": [
            {
                "path": str(path.relative_to(archive_path)).replace("\\", "/"),
                "extension": path.suffix.lower(),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }


def write_inventory(inventory: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    lines = [
        "# Old Bot Telegram Export Inventory",
        "",
        f"Created: {inventory['created_at']}",
        f"Raw archive path: `{inventory['archive_path']}`",
        f"Total files: {inventory['total_files']}",
        f"Total size: {inventory['total_bytes'] / 1024 / 1024:.2f} MB",
        "",
        "## File Types",
        "",
    ]
    for extension, count in inventory["extension_counts"].items():
        lines.append(f"- `{extension}`: {count}")
    lines.extend(
        [
            "",
            "## Handling Rule",
            "",
            "The raw Telegram export is treated as read-only source material. "
            "All extracted text, transcripts, summaries, and memory updates live under this import folder.",
            "",
        ]
    )
    (output_dir / "inventory.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def extract_messages(archive_path: Path) -> list[ArchiveMessage]:
    messages: list[ArchiveMessage] = []
    for html_path in sorted(archive_path.glob("messages*.html")):
        parser = TelegramHtmlParser()
        parser.feed(html_path.read_text(encoding="utf-8", errors="replace"))
        messages.extend(parser.messages)
    return messages


def write_messages(messages: list[ArchiveMessage], output_dir: Path) -> None:
    text_dir = output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    with (text_dir / "messages.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for message in messages:
            file.write(json.dumps(message.__dict__, ensure_ascii=False) + "\n")

    lines = ["# Extracted Telegram Text", ""]
    for message in messages:
        if not message.text and not message.attachments:
            continue
        lines.extend(
            [
                f"## {message.date or message.message_id} — {message.sender or 'unknown'}",
                "",
            ]
        )
        if message.text:
            lines.extend([message.text, ""])
        if message.attachments:
            lines.append("Attachments:")
            for attachment in message.attachments:
                lines.append(f"- `{attachment}`")
            lines.append("")

    (text_dir / "messages.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )


def transcribe_voice_messages(
    archive_path: Path,
    output_dir: Path,
    client: DeepgramClient,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    voice_dir = archive_path / "voice_messages"
    transcript_dir = output_dir / "transcripts" / "voice"
    raw_dir = output_dir / "transcripts" / "deepgram_raw"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    voice_files = sorted(voice_dir.glob("*.ogg"))
    if limit is not None:
        voice_files = voice_files[:limit]

    results: list[dict[str, Any]] = []
    for audio_path in voice_files:
        safe_name = slugify(audio_path.stem)
        transcript_path = transcript_dir / f"{safe_name}.md"
        raw_path = raw_dir / f"{safe_name}.json"
        if transcript_path.exists() and raw_path.exists():
            results.append(
                {
                    "source": str(audio_path.relative_to(archive_path)).replace("\\", "/"),
                    "status": "skipped_existing",
                    "transcript": str(transcript_path),
                }
            )
            continue

        try:
            transcript = client.transcribe_file(audio_path)
        except TranscriptionError as exc:
            error_path = transcript_dir / f"{safe_name}.error.md"
            error_path.write_text(
                "\n".join(
                    [
                        f"# Voice Transcript Error: {audio_path.name}",
                        "",
                        f"Source: `{audio_path.relative_to(archive_path)}`",
                        "",
                        "## Error",
                        "",
                        sanitize_text(str(exc)),
                        "",
                    ]
                ),
                encoding="utf-8",
                newline="\n",
            )
            results.append(
                {
                    "source": str(audio_path.relative_to(archive_path)).replace("\\", "/"),
                    "status": "failed",
                    "transcript": str(error_path),
                }
            )
            continue

        raw_path.write_text(
            json.dumps(transcript.raw_response, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        transcript_path.write_text(
            "\n".join(
                [
                    f"# Voice Transcript: {audio_path.name}",
                    "",
                    f"Source: `{audio_path.relative_to(archive_path)}`",
                    f"Duration seconds: {transcript.duration_seconds if transcript.duration_seconds is not None else 'unknown'}",
                    f"Confidence: {transcript.confidence if transcript.confidence is not None else 'unknown'}",
                    "",
                    "## Transcript",
                    "",
                    sanitize_text(transcript.text),
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )
        results.append(
            {
                "source": str(audio_path.relative_to(archive_path)).replace("\\", "/"),
                "status": "transcribed",
                "transcript": str(transcript_path),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a Telegram HTML export into the agent vault.")
    parser.add_argument("--archive", required=True, help="Path to Telegram export folder.")
    parser.add_argument("--project", default="content-factory", help="Project slug/name in vault.")
    parser.add_argument("--inventory", action="store_true", help="Write inventory files.")
    parser.add_argument("--extract-text", action="store_true", help="Extract messages.html text.")
    parser.add_argument("--transcribe-voice", action="store_true", help="Transcribe voice_messages/*.ogg with Deepgram.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for voice transcription.")
    args = parser.parse_args()

    settings = Settings.load()
    archive_path = Path(args.archive)
    if not archive_path.exists():
        raise SystemExit(f"Archive folder not found: {archive_path}")

    output_dir = import_root(settings, args.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.inventory:
        inventory = inventory_archive(archive_path)
        write_inventory(inventory, output_dir)
        print(f"Inventory written to {output_dir}")

    if args.extract_text:
        messages = extract_messages(archive_path)
        write_messages(messages, output_dir)
        print(f"Extracted {len(messages)} messages to {output_dir / 'text'}")

    if args.transcribe_voice:
        client = DeepgramClient(
            api_key=settings.deepgram_api_key,
            model=settings.deepgram_model,
            language=settings.deepgram_language,
        )
        try:
            results = transcribe_voice_messages(
                archive_path=archive_path,
                output_dir=output_dir,
                client=client,
                limit=args.limit,
            )
        except TranscriptionError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
