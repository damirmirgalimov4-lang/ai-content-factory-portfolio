from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ACTIVE_STATUSES = {"pending", "processing"}


class VideoProviderError(RuntimeError):
    """Normalized provider error used by durable video job orchestration."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        ambiguous_submission: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.ambiguous_submission = ambiguous_submission


@dataclass(frozen=True)
class VideoTask:
    task_id: str
    status: str
    result_url: str = ""
    error: str = ""
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class VideoGenerationRequest:
    model: str
    prompt: str
    image_path: Path | None
    duration_seconds: int
    aspect_ratio: str
    mode: str
    sound_enabled: bool
    user: str = ""
    resolution: str = "720p"
    provider: str = "polza"
    reference_image_paths: tuple[Path, ...] = ()
    seed: int = 0
    idempotency_key: str = ""


class VideoProviderClient(Protocol):
    provider_name: str

    @property
    def is_configured(self) -> bool: ...

    def create_video_task(self, request: VideoGenerationRequest) -> VideoTask: ...

    def get_task(self, task_id: str) -> VideoTask: ...

    def download_video(self, url: str, target: Path) -> Path: ...
