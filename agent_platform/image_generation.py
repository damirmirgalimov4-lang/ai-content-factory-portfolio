from __future__ import annotations

import base64
import binascii
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .config import Settings
from .llm import find_codex_cli, safe_subprocess_environment


class ImageGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class GeneratedImage:
    content: bytes
    extension: str = ".png"
    content_type: str = "image/png"


@dataclass(frozen=True)
class ImageReference:
    reference_id: str
    role: str
    path: Path


MAX_IMAGE_REFERENCE_FILES = 5


def _image_size(value: str | None, default: str) -> str:
    """Preserve provider-specific configured sizes; Storyboard validates its own quote."""

    selected = (value or default).strip()
    if not selected or len(selected) > 64:
        raise ImageGenerationError(
            f"Неподдерживаемый размер изображения: {selected or '<empty>'}.",
            code="unsupported_image_size",
        )
    return selected


class ImageClient(Protocol):
    @property
    def is_configured(self) -> bool:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        references: Sequence[ImageReference] = (),
        *,
        size: str | None = None,
    ) -> GeneratedImage:
        raise NotImplementedError


def generate_image_with_retry(
    image_client: ImageClient,
    prompt: str,
    references: Sequence[ImageReference] = (),
    *,
    max_attempts: int = 2,
    retry_delay_seconds: float = 1.0,
) -> GeneratedImage:
    """Retry one transient Codex transport miss without creating an endless loop."""

    attempts = max(1, max_attempts)
    for attempt in range(1, attempts + 1):
        try:
            return (
                image_client.generate(prompt, references=references)
                if references
                else image_client.generate(prompt)
            )
        except ImageGenerationError as exc:
            retryable = exc.code == "codex_image_missing"
            if not retryable or attempt >= attempts:
                raise
            time.sleep(max(0.0, retry_delay_seconds))
    raise AssertionError("unreachable")


@dataclass
class NoImageClient:
    reason: str = "OPENAI_API_KEY не задан."

    @property
    def is_configured(self) -> bool:
        return False

    def generate(
        self,
        prompt: str,
        references: Sequence[ImageReference] = (),
        *,
        size: str | None = None,
    ) -> GeneratedImage:
        raise ImageGenerationError(self.reason)


class OpenAIImageClient:
    """Minimal GPT Image adapter that keeps the provider replaceable and secrets in env."""

    def __init__(self, settings: Settings, request_timeout_seconds: int = 180):
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url
        self.model = settings.openai_image_model
        self.size = settings.openai_image_size
        self.quality = settings.openai_image_quality
        self.request_timeout_seconds = request_timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        references: Sequence[ImageReference] = (),
        *,
        size: str | None = None,
    ) -> GeneratedImage:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ImageGenerationError("Промпт изображения пустой.")
        if not self.is_configured:
            raise ImageGenerationError("OPENAI_API_KEY не задан.")
        if references:
            raise ImageGenerationError(
                "Текущий openai_api adapter не поддерживает reference images. "
                "Для reference-aware генерации выбери IMAGE_PROVIDER=codex.",
                code="reference_images_not_supported",
            )

        requested_size = _image_size(size, self.size)
        response = self._request(
            "/images/generations",
            {
                "model": self.model,
                "prompt": clean_prompt,
                "size": requested_size,
                "quality": self.quality,
                "n": 1,
            },
        )
        data = response.get("data", [])
        if not data or not isinstance(data[0], dict):
            raise ImageGenerationError("OpenAI вернул ответ без изображения.")
        encoded = str(data[0].get("b64_json", "")).strip()
        if not encoded:
            raise ImageGenerationError("OpenAI не вернул b64_json изображения.")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageGenerationError("OpenAI вернул повреждённые данные изображения.") from exc
        if not content:
            raise ImageGenerationError("OpenAI вернул пустой файл изображения.")
        return GeneratedImage(content=content)

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
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
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            error_code = "unknown_error"
            error_type = ""
            try:
                error = json.loads(body).get("error", {})
                error_code = str(error.get("code") or error_code)
                error_type = str(error.get("type") or "")
            except (json.JSONDecodeError, AttributeError):
                pass
            details = f"{error_type}/{error_code}".strip("/")
            raise ImageGenerationError(
                f"OpenAI Images HTTP {exc.code} ({details}).",
                code=error_code,
                status_code=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise ImageGenerationError(f"OpenAI Images network error: {exc}") from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ImageGenerationError("OpenAI Images вернул невалидный JSON.") from exc
        if not isinstance(decoded, dict):
            raise ImageGenerationError("OpenAI Images вернул неожиданный формат ответа.")
        return decoded


class CodexImageClient:
    """Generate through Codex's built-in image tool and existing local OAuth profile."""

    def __init__(self, settings: Settings):
        self.executable = find_codex_cli(settings.codex_cli_path)
        self.agent_model = settings.codex_chat_model
        self.size = settings.openai_image_size
        self.timeout_seconds = settings.codex_image_timeout_seconds
        self.runtime_root = (
            settings.codex_workdir.resolve() / ".tmp" / "codex-images"
        )
        self.work_root = self.runtime_root / "work"
        self.output_root = self.runtime_root / "output"
        configured_codex_home = os.getenv("CODEX_HOME", "").strip()
        codex_home = (
            Path(configured_codex_home).expanduser()
            if configured_codex_home
            else Path.home() / ".codex"
        )
        self.generated_root = codex_home.resolve() / "generated_images"
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        self._cancel_requested = threading.Event()

    @property
    def is_configured(self) -> bool:
        return self.executable is not None

    def generate(
        self,
        prompt: str,
        references: Sequence[ImageReference] = (),
        *,
        size: str | None = None,
    ) -> GeneratedImage:
        self._cancel_requested.clear()
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ImageGenerationError("Промпт изображения пустой.")
        if self.executable is None:
            raise ImageGenerationError(
                "Codex CLI не найден.",
                code="codex_cli_not_found",
            )
        if len(references) > MAX_IMAGE_REFERENCE_FILES:
            raise ImageGenerationError(
                "Codex Image принимает не более 5 reference-файлов за один запрос.",
                code="too_many_reference_images",
            )
        normalized_references: list[ImageReference] = []
        for reference in references:
            path = reference.path.resolve()
            if not path.is_file():
                raise ImageGenerationError(
                    f"Reference image не найден: {reference.reference_id}.",
                    code="reference_image_missing",
                )
            normalized_references.append(
                ImageReference(reference.reference_id, reference.role, path)
            )

        self.work_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(
            prefix="request-",
            dir=self.work_root,
        ))
        output_path = self.output_root / f"{workdir.name}.png"
        requested_size = _image_size(size, self.size)
        instruction = self._build_instruction(
            clean_prompt,
            output_path,
            normalized_references,
            size=requested_size,
        )
        command = [
            str(self.executable),
            "exec",
            "--ephemeral",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "--cd",
            str(workdir),
            "--add-dir",
            str(self.output_root),
        ]
        if normalized_references:
            command.append("--image")
            command.extend(str(item.path) for item in normalized_references)
        command.extend([
            "--model",
            self.agent_model,
            "--color",
            "never",
            "--json",
            "-",
        ])
        with self._generation_lock():
            generated_before = self._generated_snapshot()
            stdout_path = workdir / "codex.stdout.jsonl"
            stderr_path = workdir / "codex.stderr.log"
            result: Path | None = None
            returncode: int | None = None
            timed_out = False
            try:
                with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_file, stderr_path.open(
                    "w", encoding="utf-8", newline="\n"
                ) as stderr_file:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        env=safe_subprocess_environment(),
                    )
                    with self._active_process_lock:
                        self._active_process = process
                    if self._cancel_requested.is_set():
                        self._stop_process(process)
                    if process.stdin is None:
                        raise OSError("Codex CLI stdin is unavailable")
                    process.stdin.write(instruction)
                    process.stdin.close()
                    try:
                        result, returncode, timed_out = self._wait_for_result(
                            process,
                            workdir=workdir,
                            output_path=output_path,
                            generated_before=generated_before,
                        )
                    finally:
                        with self._active_process_lock:
                            if self._active_process is process:
                                self._active_process = None
            except OSError as exc:
                raise ImageGenerationError(
                    f"Не удалось запустить Codex CLI: {exc}",
                    code="codex_cli_start_failed",
                ) from exc

        if self._cancel_requested.is_set():
            raise ImageGenerationError(
                "Генерация остановлена пользователем.",
                code="generation_cancelled",
            )
        stdout = (
            stdout_path.read_text(encoding="utf-8", errors="replace")
            if stdout_path.is_file()
            else ""
        )
        if result is None:
            if timed_out:
                raise ImageGenerationError(
                    f"Codex не завершил генерацию за {self.timeout_seconds} секунд.",
                    code="codex_image_timeout",
                )
            error = _last_codex_error(stdout)
            if returncode not in {None, 0} and not error:
                error = f"процесс завершился с кодом {returncode}"
            raise ImageGenerationError(
                f"Codex не создал файл изображения: {error or 'причина не указана'}",
                code="codex_image_missing",
            )
        try:
            content = result.read_bytes()
        except PermissionError as exc:
            raise ImageGenerationError(
                "Windows не разрешил боту прочитать созданное изображение.",
                code="codex_image_read_denied",
            ) from exc
        if not content:
            raise ImageGenerationError(
                "Codex создал пустой файл изображения.",
                code="codex_image_empty",
            )
        extension = result.suffix.lower()
        content_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(extension, "image/png")
        return GeneratedImage(
            content=content,
            extension=extension if extension in {".png", ".jpg", ".jpeg", ".webp"} else ".png",
            content_type=content_type,
        )

    def cancel_active(self) -> bool:
        """Request cancellation and terminate the currently owned Codex subprocess."""

        self._cancel_requested.set()
        with self._active_process_lock:
            process = self._active_process
        if process is None or process.poll() is not None:
            return False
        self._stop_process(process)
        return True

    def _wait_for_result(
        self,
        process: subprocess.Popen[str],
        *,
        workdir: Path,
        output_path: Path,
        generated_before: set[Path],
    ) -> tuple[Path | None, int | None, bool]:
        """Collect the managed image even when Codex stays open after image generation."""

        deadline = time.monotonic() + self.timeout_seconds
        stable_signature: tuple[Path, int] | None = None
        stable_checks = 0
        while True:
            result = self._find_result(
                workdir,
                output_path,
                generated_root=self.generated_root,
                generated_before=generated_before,
            )
            if result is not None:
                try:
                    signature = (result.resolve(), result.stat().st_size)
                except (FileNotFoundError, OSError):
                    signature = None
                if signature is not None and signature[1] > 0:
                    if signature == stable_signature:
                        stable_checks += 1
                    else:
                        stable_signature = signature
                        stable_checks = 1
                    if stable_checks >= 2:
                        self._stop_process(process)
                        return result, process.poll(), False
            else:
                stable_signature = None
                stable_checks = 0

            returncode = process.poll()
            if returncode is not None:
                final = self._find_result(
                    workdir,
                    output_path,
                    generated_root=self.generated_root,
                    generated_before=generated_before,
                )
                return final, returncode, False
            if time.monotonic() >= deadline:
                self._stop_process(process)
                final = self._find_result(
                    workdir,
                    output_path,
                    generated_root=self.generated_root,
                    generated_before=generated_before,
                )
                return final, process.poll(), final is None
            time.sleep(1.0)

    @staticmethod
    def _stop_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    def _generated_snapshot(self) -> set[Path]:
        if not self.generated_root.is_dir():
            return set()
        return {
            path.resolve()
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp")
            for path in self.generated_root.rglob(pattern)
            if path.is_file()
        }

    @contextmanager
    def _generation_lock(self):
        """Serialize subscription-backed image calls across bot and operator processes."""

        self.runtime_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.runtime_root / "generation.lock"
        handle = lock_path.open("a+b")
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        locked = False
        try:
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError as exc:
                raise ImageGenerationError(
                    "Другая генерация изображения уже выполняется. Дождись её завершения.",
                    code="codex_image_busy",
                ) from exc
            yield
        finally:
            if locked:
                try:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            handle.close()

    def _build_instruction(
        self,
        prompt: str,
        output_path: Path,
        references: Sequence[ImageReference] = (),
        *,
        size: str | None = None,
    ) -> str:
        requested_size = _image_size(size, self.size)
        reference_block = ""
        if references:
            lines = [
                f"- attached image {index}: {item.reference_id} ({item.role})"
                for index, item in enumerate(references, 1)
            ]
            reference_block = (
                "\n\nATTACHED VISUAL REFERENCES (in attachment order):\n"
                + "\n".join(lines)
                + f"\nAll {len(references)} images are already attached to this initial request. "
                f"Pass all of them to the built-in image generation tool using "
                f"num_last_images_to_include={len(references)}. Do not use shell commands or local "
                "filesystem reads for these references. Preserve character identity from character "
                "cards and spatial design from location frames."
            )
        return (
            "Use the built-in image generation tool with the existing Codex subscription OAuth. "
            "Do not use OPENAI_API_KEY and do not use the fallback image_gen.py API script. "
            "Generate one actual raster image. The built-in tool selects its image backend. "
            f"The required final raster dimensions are {requested_size}; include this requirement "
            "in the generation request, but do not pass an unsupported size field to the tool. "
            "Call the image generation tool exactly once. Let it save the result in Codex's managed "
            "generated-images storage; the host process will collect it there. Do not attempt a "
            "filesystem copy after generation. Do not create SVG and do not draw the image with code. "
            "Finish immediately when the image tool returns."
            "\n\nPRODUCTION PROMPT:\n"
            f"{prompt[:24000]}"
            f"{reference_block}"
        )

    @staticmethod
    def _find_result(
        workdir: Path,
        output_path: Path,
        *,
        generated_root: Path | None = None,
        generated_before: set[Path] | None = None,
    ) -> Path | None:
        if output_path.is_file():
            return output_path
        legacy_result = workdir / "result.png"
        if legacy_result.is_file():
            return legacy_result
        candidates = [
            path
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp")
            for path in workdir.glob(pattern)
            if path.is_file()
        ]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
        if generated_root is None or not generated_root.is_dir():
            return None
        previous = generated_before or set()
        generated = [
            path
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp")
            for path in generated_root.rglob(pattern)
            if path.is_file() and path.resolve() not in previous
        ]
        return generated[0] if len(generated) == 1 else None


def _last_codex_error(output: str) -> str:
    errors: list[str] = []
    agent_failures: list[str] = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "error":
                errors.append(str(item.get("message", "")).strip())
            elif item.get("type") == "agent_message":
                message = str(item.get("text", "")).strip()
                lowered = message.lower()
                if "generation failed" in lowered or "no image was produced" in lowered:
                    agent_failures.append(message)
        elif event.get("type") in {"turn.failed", "error"}:
            message = str(event.get("message", "")).strip()
            if message and not message.startswith("Reconnecting"):
                errors.append(message)
    if agent_failures:
        message = agent_failures[-1]
        if "could not read the required reference image" in message.lower() or "could not read the referenced image" in message.lower():
            return "Codex не смог прочитать приложенное reference-изображение."
        return message[:500]
    if not errors:
        return ""
    message = errors[-1]
    if "403 Forbidden" in message and "chatgpt.com/backend-api/codex/responses" in message:
        return "Транспорт Codex временно отклонил подключение (HTTP 403)."
    return message[:500]


def create_image_client(settings: Settings) -> ImageClient:
    if settings.image_provider == "codex":
        return CodexImageClient(settings)
    if settings.image_provider != "openai_api":
        return NoImageClient(f"Неизвестный IMAGE_PROVIDER: {settings.image_provider}")
    if settings.openai_api_key:
        return OpenAIImageClient(settings)
    return NoImageClient()


def build_visual_draft_prompt(
    idea: str,
    storyboard: str,
    prompts: str,
    qa: str = "",
) -> str:
    """Build an optional contact sheet; individual frames are the primary artifact."""

    return (
        "Create a professional color storyboard sheet for a short vertical AI video. "
        "This is the optional Syntx-style storyboard mode, not the standard separate-frame mode.\n"
        "Use one coherent visual language across every panel. Keep the same character design, "
        "wardrobe, locations, props, lighting logic, color palette, and camera language.\n"
        "Layout: a clean contact sheet containing every source scene exactly once, in source order. "
        "Use the actual dynamic scene count from the approved contract; never force 6 or 15 panels. "
        "Choose rows and columns dynamically. Use clear gutters and small scene IDs only: no prose "
        "captions, timecodes, logos, or watermark because generated text is not a source of truth.\n"
        "Each panel should read as a useful production reference frame, not a rough stick-figure sketch. "
        "Each panel must show the scene's declared composition, physical action, camera angle and "
        "location. Preserve continuity even when the camera angle or action changes. Attached "
        "character and environment references are the pixel-level source of truth.\n"
        "The panels together must form one readable visual sequence with setup, development and payoff "
        "only when those functions exist in the approved story; do not invent a fixed three-act template.\n"
        "Never merge, omit, duplicate, or invent scenes. This contact sheet is only an optional preview; "
        "the production pipeline generates each source scene as a separate image file.\n\n"
        f"SOURCE IDEA:\n{idea[:3000]}\n\n"
        f"APPROVED STORYBOARD:\n{storyboard[:12000]}\n\n"
        f"APPROVED GENERATION PROMPTS:\n{prompts[:12000]}\n\n"
        f"QA NOTES:\n{qa[:5000]}"
    )


def build_single_frame_prompt(
    *,
    scene_id: str,
    visual: str,
    image_prompt: str,
    continuity: dict[str, str] | None = None,
    aspect_ratio: str = "9:16",
    reference_inputs: Sequence[tuple[str, str]] = (),
    text_only_references: Sequence[tuple[str, str, str]] = (),
) -> str:
    """Build one isolated image request while preserving declared continuity anchors."""

    anchors = continuity or {}
    continuity_text = "\n".join(
        f"- {key}: {value}"
        for key, value in anchors.items()
        if str(value).strip()
    ) or "- No additional continuity anchors were provided."
    reference_text = "\n".join(
        f"- {reference_id}: {role}"
        for reference_id, role in reference_inputs
    ) or "- No visual reference files are attached."
    text_only_reference_text = "\n".join(
        f"- {reference_id} ({role}): {description}"
        for reference_id, role, description in text_only_references
    ) or "- None."
    return (
        f"Create exactly one production reference image for scene {scene_id}.\n"
        f"Canvas aspect ratio: {aspect_ratio}.\n"
        "Return one full-frame raster image. Do not create a grid, collage, contact sheet, "
        "split screen, storyboard page, multiple panels, panel borders, scene labels, captions, "
        "logos, or watermark. Do not show alternate takes inside the same image.\n\n"
        f"SCENE PURPOSE AND VISIBLE CONTENT:\n{visual.strip()}\n\n"
        f"IMAGE PROMPT:\n{image_prompt.strip()}\n\n"
        f"CONTINUITY ANCHORS:\n{continuity_text}\n\n"
        f"ATTACHED VISUAL REFERENCES:\n{reference_text}\n\n"
        "TEXT-ONLY REFERENCE REQUIREMENTS (files omitted because the image tool accepts at most "
        f"{MAX_IMAGE_REFERENCE_FILES} files):\n{text_only_reference_text}\n\n"
        "When visual references are attached, use their actual pixels as the source of truth. "
        "For every character reference, preserve the exact same identity, proportions, face, fur or "
        "skin pattern, hair, wardrobe, colors and materials. If several character references are "
        "listed, include each required character exactly once unless the scene says otherwise. For a "
        "location-continuity frame, preserve the same spatial layout and environment while changing "
        "only the action and camera framing required by this scene.\n\n"
        "Preserve all declared identity, wardrobe, object, material, environment, color, and "
        "lighting details. Compose this as a usable single keyframe for image-to-video animation."
    )
