from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import WorkerSettings
from .errors import AmbiguousSubmissionError
from .store import JobStore, ReservationConflict


_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
_ALLOWED_IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"RIFF")


class JobConflict(RuntimeError):
    pass


class JobNotFound(RuntimeError):
    pass


class WorkerNotReady(RuntimeError):
    pass


class InvalidJobRequest(ValueError):
    pass


@dataclass(frozen=True)
class JobRequest:
    job_id: str
    model: str
    workflow: str
    prompt: str
    seed: int
    duration_seconds: int
    aspect_ratio: str
    width: int
    height: int
    fps: int
    audio: bool
    image_bytes: bytes
    image_sha256: str
    fingerprint: str
    external_prompt_id: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "JobRequest":
        if not isinstance(payload, dict):
            raise InvalidJobRequest("JSON body must be an object")
        job_id = str(payload.get("job_id", "")).strip()
        if not _JOB_ID.fullmatch(job_id):
            raise InvalidJobRequest("job_id must use 3-128 safe ASCII characters")
        model = str(payload.get("model", "")).strip()
        workflow = str(payload.get("workflow", "")).strip()
        prompt = str(payload.get("prompt", "")).strip()
        if model != "ltx-2.3":
            raise InvalidJobRequest("only model=ltx-2.3 is enabled")
        if workflow != "distilled":
            raise InvalidJobRequest("only workflow=distilled is enabled")
        if not prompt or len(prompt) > 5000:
            raise InvalidJobRequest("prompt must contain 1-5000 characters")
        try:
            seed = int(payload.get("seed"))
            duration = int(payload.get("duration_seconds"))
            width = int(payload.get("width"))
            height = int(payload.get("height"))
            fps = int(payload.get("fps"))
        except (TypeError, ValueError) as exc:
            raise InvalidJobRequest("seed/duration/width/height/fps must be integers") from exc
        aspect = str(payload.get("aspect_ratio", "")).strip()
        audio = payload.get("audio")
        if not 0 <= seed <= 2**63 - 1:
            raise InvalidJobRequest("seed is outside the accepted range")
        if (duration, aspect, width, height, fps, audio) != (5, "16:9", 1024, 576, 24, True):
            raise InvalidJobRequest(
                "the pre-benchmark contract is fixed to 5s, 16:9, 1024x576, 24fps, audio=true"
            )
        encoded = payload.get("image_base64")
        if not isinstance(encoded, str) or not encoded:
            raise InvalidJobRequest("image_base64 is required")
        try:
            image = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidJobRequest("image_base64 is invalid") from exc
        if not image or len(image) > 20 * 1024 * 1024:
            raise InvalidJobRequest("input image must contain 1 byte to 20 MiB")
        if not image.startswith(_ALLOWED_IMAGE_MAGIC):
            raise InvalidJobRequest("input image must be PNG, JPEG, or WebP")
        image_hash = hashlib.sha256(image).hexdigest()
        canonical = {
            "job_id": job_id,
            "model": model,
            "workflow": workflow,
            "prompt": prompt,
            "seed": seed,
            "duration_seconds": duration,
            "aspect_ratio": aspect,
            "width": width,
            "height": height,
            "fps": fps,
            "audio": audio,
            "image_sha256": image_hash,
        }
        fingerprint = hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        external_prompt_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"content-factory:ltx-2.3:{fingerprint}"))
        return cls(
            job_id=job_id,
            model=model,
            workflow=workflow,
            prompt=prompt,
            seed=seed,
            duration_seconds=duration,
            aspect_ratio=aspect,
            width=width,
            height=height,
            fps=fps,
            audio=bool(audio),
            image_bytes=image,
            image_sha256=image_hash,
            fingerprint=fingerprint,
            external_prompt_id=external_prompt_id,
        )

    def stored_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model": self.model,
            "workflow": self.workflow,
            "prompt": self.prompt,
            "seed": self.seed,
            "duration_seconds": self.duration_seconds,
            "aspect_ratio": self.aspect_ratio,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "audio": self.audio,
            "image_sha256": self.image_sha256,
            "external_prompt_id": self.external_prompt_id,
        }


class Runner(Protocol):
    def readiness(self) -> tuple[bool, str]: ...
    def run(self, request: JobRequest, input_path: Path, output_path: Path) -> dict[str, object]: ...


class VideoWorkerService:
    def __init__(
        self,
        settings: WorkerSettings,
        store: JobStore,
        runner: Runner,
        *,
        executor: Any | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.runner = runner
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="ltx-inference")
        self.store.recover_interrupted()

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(content)
        temporary.chmod(0o600)
        temporary.replace(path)

    def ready(self) -> tuple[bool, str]:
        if not self.settings.inference_enabled:
            return False, "inference is disabled"
        return self.runner.readiness()

    def submit(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        ready, reason = self.ready()
        if not ready:
            raise WorkerNotReady(reason)
        request = JobRequest.from_payload(payload)
        existing = self.store.get(request.job_id)
        if existing is not None:
            if existing.get("fingerprint") != request.fingerprint:
                raise JobConflict(f"job_id {request.job_id!r} is already used")
            return self.public_job(existing), False

        job_root = self.settings.root / "jobs" / request.job_id
        suffix = ".png" if request.image_bytes.startswith(b"\x89PNG") else ".jpg"
        input_path = job_root / f"input-{request.fingerprint}{suffix}"
        output_path = job_root / "result.mp4"
        try:
            reserved, created = self.store.reserve(request, input_path, output_path)
        except ReservationConflict as exc:
            raise JobConflict(str(exc)) from exc
        if not created:
            return self.public_job(reserved), False
        try:
            self._atomic_write(input_path, request.image_bytes)
        except Exception as exc:
            self.store.mark_failed(
                request.job_id,
                f"input staging failed: {type(exc).__name__}: {exc}",
                from_statuses=("queued",),
            )
            raise
        self.executor.submit(self._execute, request, input_path, output_path)
        return self.public_job(self.store.require(request.job_id)), True

    def _execute(self, request: JobRequest, input_path: Path, output_path: Path) -> None:
        try:
            self.store.mark_running(request.job_id)
            result = self.runner.run(request, input_path, output_path)
            if not output_path.is_file():
                raise RuntimeError("inference returned without result.mp4")
            result = {**result, "size_bytes": output_path.stat().st_size}
            self.store.mark_completed(request.job_id, result)
        except AmbiguousSubmissionError as exc:
            current = self.store.get(request.job_id)
            if current and current.get("status") == "running":
                self.store.mark_reconciliation_required(
                    request.job_id,
                    f"{type(exc).__name__}: {exc}",
                )
        except Exception as exc:  # terminal failure is durable and never auto-retried
            current = self.store.get(request.job_id)
            if current and current.get("status") == "running":
                self.store.mark_failed(request.job_id, f"{type(exc).__name__}: {exc}")

    def get(self, job_id: str) -> dict[str, Any]:
        try:
            return self.public_job(self.store.require(job_id))
        except KeyError as exc:
            raise JobNotFound(job_id) from exc

    def result_path(self, job_id: str) -> Path:
        item = self.store.get(job_id)
        if item is None:
            raise JobNotFound(job_id)
        if item.get("status") != "completed":
            raise JobConflict("result is not ready")
        path = Path(str(item.get("output_path", "")))
        if not path.is_file():
            raise JobConflict("committed result file is missing")
        return path

    @staticmethod
    def public_job(item: dict[str, Any]) -> dict[str, Any]:
        status = str(item.get("status", ""))
        return {
            "job_id": str(item.get("job_id", "")),
            "status": status,
            "result_url": (
                f"/video/jobs/{item.get('job_id')}/result" if status == "completed" else ""
            ),
            "error": str(item.get("error", "")),
            "metrics": item.get("result", {}) if status == "completed" else {},
            "created_at": str(item.get("created_at", "")),
            "updated_at": str(item.get("updated_at", "")),
        }

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
        self.store.close()
