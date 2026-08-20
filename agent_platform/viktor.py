from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .video_provider import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    VideoGenerationRequest,
    VideoProviderError,
    VideoTask,
)


class ViktorError(VideoProviderError):
    pass


class ViktorClient:
    """Viktor Public API adapter for durable agent-run media tasks."""

    provider_name = "viktor"

    def __init__(
        self,
        settings: Settings,
        *,
        request_timeout_seconds: int = 90,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = settings.viktor_api_key
        self.base_url = settings.viktor_base_url
        self.poll_interval_seconds = settings.viktor_poll_interval_seconds
        self.timeout_seconds = settings.viktor_timeout_seconds
        self.max_status_retries = settings.viktor_max_status_retries
        self.request_timeout_seconds = request_timeout_seconds
        self.sleep = sleep

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def test_connection(self) -> dict[str, Any]:
        """Check the key without creating an agent run or spending media credits."""

        if not self.is_configured:
            raise ViktorError("Viktor API key не задан.")
        return self._request_json("GET", "/api/public/v1/test")

    def create_video_task(self, request: VideoGenerationRequest) -> VideoTask:
        """Create one Viktor agent run that must return exactly one video artifact."""

        if not self.is_configured:
            raise ViktorError("Viktor API key не задан.")
        self._validate_request(request)
        payload = {
            "message": self._build_generation_message(request),
            "metadata": {
                "source": "content-factory",
                "request_id": request.user,
                "target_model": "seedance-2",
                "duration_seconds": request.duration_seconds,
                "resolution": request.resolution,
                "aspect_ratio": request.aspect_ratio,
                "sound_enabled": request.sound_enabled,
            },
            "speed": "smarter",
        }
        response = self._request_json(
            "POST",
            "/api/public/v1/threads",
            payload,
            extra_headers={"Idempotency-Key": self._idempotency_key(request)},
            is_submission=True,
        )
        run = response.get("run")
        if not isinstance(run, dict):
            raise ViktorError(
                "Viktor принял запрос, но не вернул объект run; автоматический повтор заблокирован.",
                ambiguous_submission=True,
            )
        task_id = str(run.get("id") or "").strip()
        if not task_id:
            raise ViktorError(
                "Viktor принял запрос, но не вернул run ID; автоматический повтор заблокирован.",
                ambiguous_submission=True,
            )
        status = self._map_status(str(run.get("status") or "queued"))
        return VideoTask(task_id, status, raw=response)

    def get_task(self, task_id: str) -> VideoTask:
        clean_id = task_id.strip()
        if not clean_id:
            raise ViktorError("Пустой Viktor run ID.")
        response = self._safe_status_request(
            f"/api/public/v1/runs/{clean_id}"
        )
        raw_status = str(response.get("status") or "queued").strip().lower()
        status = self._map_status(raw_status)
        if raw_status == "completed":
            result = self._safe_status_request(
                f"/api/public/v1/runs/{clean_id}/result"
            )
            artifact = self._video_artifact(result)
            if artifact is None:
                return VideoTask(
                    clean_id,
                    "failed",
                    error=(
                        "Viktor завершил run, но не вернул MP4/video artifact. "
                        "Вероятно, у агента не подключён инструмент видеогенерации."
                    ),
                    raw={"status": response, "result": result},
                )
            artifact_id = str(artifact.get("id") or "").strip()
            if not artifact_id:
                return VideoTask(
                    clean_id,
                    "failed",
                    error="Viktor вернул video artifact без file token.",
                    raw={"status": response, "result": result},
                )
            token = urllib.parse.quote(artifact_id, safe="")
            download = self._safe_status_request(
                f"/api/public/v1/files/{token}/download-url"
            )
            result_url = str(download.get("url") or "").strip()
            if not result_url.startswith("https://"):
                return VideoTask(
                    clean_id,
                    "failed",
                    error="Viktor не вернул безопасную временную ссылку на video artifact.",
                    raw={"status": response, "result": result, "download": download},
                )
            return VideoTask(
                clean_id,
                "completed",
                result_url=result_url,
                raw={"status": response, "result": result, "download": download},
            )
        if raw_status == "requires_action":
            return VideoTask(
                clean_id,
                "failed",
                error=(
                    "Viktor запросил ручное действие внутри run. "
                    "Публичный video adapter не может подтвердить его автоматически."
                ),
                raw=response,
            )
        error = self._run_error(response)
        return VideoTask(clean_id, status, error=error, raw=response)

    def wait_for_task(self, task_id: str) -> VideoTask:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            task = self.get_task(task_id)
            if task.status in TERMINAL_STATUSES:
                return task
            if task.status not in ACTIVE_STATUSES:
                raise ViktorError(f"Неизвестный статус Viktor: {task.status}")
            self.sleep(self.poll_interval_seconds)
        raise ViktorError(
            "Истёк timeout ожидания Viktor video run.",
            retryable=True,
        )

    def download_video(self, url: str, target: Path) -> Path:
        if not url.startswith("https://"):
            raise ViktorError("Viktor вернул небезопасный или пустой artifact URL.")
        content = self._download_with_retries(url)
        if not self._looks_like_video(content):
            raise ViktorError("Артефакт Viktor не похож на поддерживаемое видео.")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
        return target

    @staticmethod
    def _validate_request(request: VideoGenerationRequest) -> None:
        if request.model not in {
            "seedance-2",
            "bytedance/seedance-2",
        }:
            raise ViktorError(
                "Viktor adapter сейчас проверяется только для Seedance 2."
            )
        if request.duration_seconds not in {5, 10, 15}:
            raise ViktorError(
                "Viktor Seedance smoke path поддерживает 5, 10 или 15 секунд."
            )
        if request.resolution != "720p":
            raise ViktorError(
                "Viktor Seedance 2 возвращает 720p по умолчанию; "
                "явный resolution override этой моделью не поддерживается."
            )
        if request.aspect_ratio not in {
            "16:9",
            "4:3",
            "1:1",
            "3:4",
            "9:16",
            "21:9",
        }:
            raise ViktorError("Выбрано неподдерживаемое соотношение сторон.")
        if not 3 <= len(request.prompt.strip()) <= 5000:
            raise ViktorError("Video prompt должен содержать от 3 до 5000 символов.")
        if request.image_path is not None or request.reference_image_paths:
            raise ViktorError(
                "Официальный Viktor Public API не принимает локальные reference images. "
                "До появления подтверждённого upload-механизма доступен только text-to-video smoke path."
            )

    @staticmethod
    def _build_generation_message(request: VideoGenerationRequest) -> str:
        sound = "со звуком" if request.sound_enabled else "без звука"
        return (
            "Создай ровно ОДНО тестовое видео через доступный тебе генератор Seedance 2.\n"
            "Не создавай варианты, коллажи, изображения или дополнительные видео.\n"
            f"Параметры результата: {request.duration_seconds} секунд, "
            f"формат {request.aspect_ratio}, {sound}.\n"
            "Seedance 2 возвращает 720p по умолчанию: не передавай генератору "
            "аргумент resolution и проверь качество уже готового файла.\n"
            "После принятия media task не создавай новый: только проверяй статус "
            "того же задания до терминального результата. При временной ошибке "
            "ConnectionError продолжай безопасный polling того же task ID; не "
            "считай временный сбой окончательной ошибкой генерации.\n"
            "Не завершай текущий run текстовым отчётом до получения файла. "
            "Скачай готовый результат и верни его как MP4-артефакт текущего run.\n"
            "Не публикуй результат во внешние соцсети.\n\n"
            f"Содержание видео:\n{request.prompt.strip()}"
        )

    @staticmethod
    def _map_status(status: str) -> str:
        return {
            "queued": "pending",
            "in_progress": "processing",
            "cancellation_requested": "processing",
            "requires_action": "failed",
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
            "timed_out": "failed",
        }.get(status.strip().lower(), status.strip().lower())

    @staticmethod
    def _run_error(response: dict[str, Any]) -> str:
        error = response.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or error.get("code") or "").strip()
            return message[:500]
        return ""

    @staticmethod
    def _video_artifact(result: dict[str, Any]) -> dict[str, Any] | None:
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, list):
            return None
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            content_type = str(artifact.get("content_type") or "").lower()
            display_name = str(artifact.get("display_name") or "").lower()
            is_video = content_type.startswith("video/") or display_name.endswith(
                (".mp4", ".webm", ".mov")
            )
            if is_video:
                return artifact
        return None

    def _idempotency_key(self, request: VideoGenerationRequest) -> str:
        payload = {
            "user": request.user,
            "model": request.model,
            "prompt": request.prompt,
            "duration": request.duration_seconds,
            "resolution": request.resolution,
            "aspect_ratio": request.aspect_ratio,
            "sound": request.sound_enabled,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return f"content-factory-{digest}"

    def _safe_status_request(self, path: str) -> dict[str, Any]:
        attempts = 0
        while True:
            try:
                return self._request_json("GET", path)
            except ViktorError as exc:
                attempts += 1
                if not exc.retryable or attempts > self.max_status_retries:
                    raise
                self.sleep(min(2 ** (attempts - 1), 8))

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        extra_headers: dict[str, str] | None = None,
        is_submission: bool = False,
    ) -> dict[str, Any]:
        data = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "content-factory/1.0",
        }
        if extra_headers:
            headers.update(extra_headers)
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ViktorError(
                self._http_error_message(exc.code, body),
                status_code=exc.code,
                retryable=exc.code == 429 or exc.code >= 500,
                ambiguous_submission=is_submission and exc.code >= 500,
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ViktorError(
                "Сетевая ошибка Viktor без подтверждённого ответа.",
                retryable=not is_submission,
                ambiguous_submission=is_submission,
            ) from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ViktorError(
                "Viktor вернул невалидный JSON.",
                ambiguous_submission=is_submission,
            ) from exc
        if not isinstance(decoded, dict):
            raise ViktorError("Viktor вернул неожиданный формат ответа.")
        return decoded

    def _http_error_message(self, code: int, body: str) -> str:
        provider_message = self._extract_error(body)
        if code == 401:
            message = "Viktor HTTP 401: API-ключ отклонён."
        elif code == 402:
            message = "Viktor HTTP 402: недостаточно кредитов."
        elif code == 403:
            message = "Viktor HTTP 403: ключу не хватает прав для этой операции."
        elif code == 409:
            message = "Viktor HTTP 409: у thread уже есть активный run."
        elif code == 422:
            message = "Viktor HTTP 422: сервис отклонил параметры запроса."
        elif code == 429:
            message = "Viktor временно ограничил частоту запросов."
        elif code >= 500:
            message = "Временная серверная ошибка Viktor."
        else:
            message = f"Viktor HTTP {code}."
        if provider_message and code != 429:
            message += f" Ответ сервиса: {provider_message}"
        return self._sanitize(message)

    @staticmethod
    def _extract_error(body: str) -> str:
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError:
            return ""
        if not isinstance(decoded, dict):
            return ""
        detail = decoded.get("detail")
        if isinstance(detail, dict):
            return str(detail.get("message") or detail.get("error") or "").strip()
        return str(detail or decoded.get("message") or decoded.get("error") or "").strip()

    def _download_with_retries(self, url: str) -> bytes:
        attempts = 0
        while True:
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "content-factory/1.0",
                        "Accept": "video/mp4,video/*;q=0.9,*/*;q=0.8",
                    },
                    method="GET",
                )
                with urllib.request.urlopen(
                    request,
                    timeout=self.request_timeout_seconds,
                ) as response:
                    return response.read()
            except (urllib.error.URLError, TimeoutError) as exc:
                attempts += 1
                if attempts > self.max_status_retries:
                    raise ViktorError(
                        "Не удалось скачать готовое видео Viktor.",
                        retryable=True,
                    ) from exc
                self.sleep(min(2 ** (attempts - 1), 8))

    def _sanitize(self, text: str) -> str:
        sanitized = str(text)
        if self.api_key:
            sanitized = sanitized.replace(self.api_key, "[REDACTED]")
        sanitized = re.sub(r"zt_live_sk_[A-Za-z0-9_-]+", "[REDACTED]", sanitized)
        return sanitized[:500]

    @staticmethod
    def _looks_like_video(content: bytes) -> bool:
        if len(content) < 12:
            return False
        return content[4:8] == b"ftyp" or content.startswith(b"\x1a\x45\xdf\xa3")
