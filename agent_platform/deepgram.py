from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Transcript:
    text: str
    confidence: float | None
    duration_seconds: float | None
    raw_response: dict[str, Any]


class DeepgramClient:
    """Transcribes local audio files through Deepgram's prerecorded audio API."""

    def __init__(
        self,
        api_key: str,
        model: str = "nova-3",
        language: str = "ru",
        request_timeout_seconds: int = 180,
    ):
        self.api_key = api_key
        self.model = model
        self.language = language
        self.request_timeout_seconds = request_timeout_seconds
        self.base_url = "https://api.deepgram.com/v1/listen"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def transcribe_file(self, audio_path: Path) -> Transcript:
        if not self.api_key:
            raise TranscriptionError("DEEPGRAM_API_KEY не задан.")
        if not audio_path.exists():
            raise TranscriptionError(f"Файл не найден: {audio_path}")

        params = urllib.parse.urlencode(
            {
                "model": self.model,
                "language": self.language,
                "smart_format": "true",
            }
        )
        content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
        if audio_path.suffix.lower() == ".ogg":
            content_type = "audio/ogg"

        request = urllib.request.Request(
            f"{self.base_url}?{params}",
            data=audio_path.read_bytes(),
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": content_type,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise TranscriptionError(f"Deepgram HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise TranscriptionError(f"Deepgram network error: {exc}") from exc

        return parse_deepgram_response(json.loads(body))


def parse_deepgram_response(response: dict[str, Any]) -> Transcript:
    channel = (
        response.get("results", {})
        .get("channels", [{}])[0]
    )
    alternative = channel.get("alternatives", [{}])[0]
    text = str(alternative.get("transcript", "")).strip()
    confidence = alternative.get("confidence")
    metadata = response.get("metadata", {})
    duration = metadata.get("duration")

    if not text:
        raise TranscriptionError("Deepgram вернул пустой transcript.")

    return Transcript(
        text=text,
        confidence=float(confidence) if isinstance(confidence, int | float) else None,
        duration_seconds=float(duration) if isinstance(duration, int | float) else None,
        raw_response=response,
    )
