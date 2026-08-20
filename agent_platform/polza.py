from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
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


class PolzaError(VideoProviderError):
    pass


@dataclass(frozen=True)
class PolzaTask(VideoTask):
    pass


@dataclass(frozen=True)
class PolzaVideoRequest(VideoGenerationRequest):

    def payload(self) -> dict[str, Any]:
        if self.model == "kling/v3":
            if not 3 <= self.duration_seconds <= 15:
                raise PolzaError("kling/v3 поддерживает длительность от 3 до 15 секунд.")
            if self.aspect_ratio not in {"16:9", "9:16", "1:1"}:
                raise PolzaError("kling/v3 не поддерживает выбранное соотношение сторон.")
            if self.mode not in {"std", "pro", "4K"}:
                raise PolzaError("kling/v3 поддерживает режимы std, pro и 4K.")
            if len(self.prompt) > 2500:
                raise PolzaError("kling/v3 принимает prompt длиной до 2500 символов.")
        elif self.model in {"bytedance/seedance-2", "bytedance/seedance-2-fast"}:
            if self.duration_seconds not in {5, 10, 15}:
                raise PolzaError("Seedance 2 поддерживает длительность 5, 10 или 15 секунд.")
            if self.aspect_ratio not in {"16:9", "4:3", "1:1", "3:4", "9:16", "21:9"}:
                raise PolzaError("Seedance 2 не поддерживает выбранное соотношение сторон.")
            if self.resolution not in {"480p", "720p", "1080p"}:
                raise PolzaError("Seedance 2 не поддерживает выбранное разрешение.")
            if not 3 <= len(self.prompt) <= 5000:
                raise PolzaError("Seedance 2 принимает prompt длиной от 3 до 5000 символов.")
        if self.image_path is None:
            raise PolzaError("PolzaAI video task требует исходный кадр.")
        content = self.image_path.read_bytes()
        if not content:
            raise PolzaError("Исходный кадр пустой.")
        mime = mimetypes.guess_type(self.image_path.name)[0] or "image/png"
        image = {
            "type": "base64",
            "data": f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}",
        }
        if self.model in {"bytedance/seedance-2", "bytedance/seedance-2-fast"}:
            model_input: dict[str, Any] = {
                "prompt": self.prompt,
                "images": [image],
                "videos": [],
                "resolution": self.resolution,
                "duration": str(self.duration_seconds),
                "aspect_ratio": self.aspect_ratio,
                "generate_audio": "true" if self.sound_enabled else "false",
                # One logical clip is one paid task; this adapter sends its supported start frame.
                "multi_shots": "false",
            }
        else:
            model_input = {
                "prompt": self.prompt,
                "images": [image],
                # PolzaAI validates Kling duration as a JSON string even though the
                # configured value remains an integer inside our domain model.
                "duration": str(self.duration_seconds),
                "aspect_ratio": self.aspect_ratio,
                "mode": self.mode,
                "sound": "true" if self.sound_enabled else "false",
            }
        payload: dict[str, Any] = {
            "model": self.model,
            "input": model_input,
            "async": True,
        }
        if self.user:
            payload["user"] = self.user
        return payload


class PolzaClient:
    """Official asynchronous PolzaAI Media API adapter with secret-safe errors."""

    provider_name = "polza"

    def __init__(
        self,
        settings: Settings,
        *,
        request_timeout_seconds: int = 90,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.api_key = settings.polza_api_key
        self.base_url = settings.polza_base_url
        self.poll_interval_seconds = settings.polza_poll_interval_seconds
        self.timeout_seconds = settings.polza_timeout_seconds
        self.max_status_retries = settings.polza_max_status_retries
        self.request_timeout_seconds = request_timeout_seconds
        self.sleep = sleep

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def create_video_task(self, request: VideoGenerationRequest) -> PolzaTask:
        if not self.is_configured:
            raise PolzaError("PolzaAI API key не задан.")
        if request.image_path is None or not request.image_path.is_file():
            name = request.image_path.name if request.image_path is not None else "не указан"
            raise PolzaError(f"Исходный кадр не найден: {name}")
        polza_request = (
            request
            if isinstance(request, PolzaVideoRequest)
            else PolzaVideoRequest(
                model=request.model,
                prompt=request.prompt,
                image_path=request.image_path,
                duration_seconds=request.duration_seconds,
                aspect_ratio=request.aspect_ratio,
                mode=request.mode,
                sound_enabled=request.sound_enabled,
                user=request.user,
                resolution=request.resolution,
                provider=request.provider,
            )
        )
        # Creation is intentionally never retried automatically: a lost response could duplicate cost.
        response = self._request_json(
            "POST",
            "/v1/media",
            polza_request.payload(),
            is_submission=True,
        )
        task = self._parse_task(response)
        if not task.task_id:
            raise PolzaError(
                "PolzaAI принял запрос, но не вернул task ID; автоматический повтор заблокирован.",
                ambiguous_submission=True,
            )
        return task

    def get_task(self, task_id: str) -> PolzaTask:
        clean_id = task_id.strip()
        if not clean_id:
            raise PolzaError("Пустой PolzaAI task ID.")
        response = self._safe_status_request(f"/v1/media/{clean_id}")
        task = self._parse_task(response)
        if not task.task_id:
            task = PolzaTask(clean_id, task.status, task.result_url, task.error, task.raw)
        return task

    def wait_for_task(self, task_id: str) -> PolzaTask:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            task = self.get_task(task_id)
            if task.status in TERMINAL_STATUSES:
                return task
            if task.status not in ACTIVE_STATUSES:
                raise PolzaError(f"Неизвестный статус PolzaAI: {task.status}")
            self.sleep(self.poll_interval_seconds)
        raise PolzaError("Истёк timeout ожидания PolzaAI video task.", retryable=True)

    def download_video(self, url: str, target: Path) -> Path:
        if not url.startswith("https://"):
            raise PolzaError("PolzaAI вернул небезопасный или пустой result URL.")
        content = self._download_with_retries(url)
        if not self._looks_like_video(content):
            raise PolzaError("Полученный файл не похож на поддерживаемое видео.")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
        return target

    def _safe_status_request(self, path: str) -> dict[str, Any]:
        attempts = 0
        while True:
            try:
                return self._request_json("GET", path)
            except PolzaError as exc:
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
        is_submission: bool = False,
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            provider_message = self._extract_error(body)
            message = provider_message or f"PolzaAI HTTP {exc.code}"
            if exc.code == 401:
                message = "PolzaAI HTTP 401: API-ключ отклонён."
                if provider_message:
                    message += f" Ответ сервиса: {provider_message}"
            elif exc.code == 403:
                message = "PolzaAI HTTP 403: доступ к операции запрещён."
                if provider_message:
                    message += f" Ответ сервиса: {provider_message}"
            elif exc.code == 429:
                message = "PolzaAI временно ограничил частоту запросов."
            elif exc.code >= 500:
                message = "Временная серверная ошибка PolzaAI."
            raise PolzaError(
                self._sanitize(message),
                status_code=exc.code,
                retryable=exc.code == 429 or exc.code >= 500,
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PolzaError(
                "Сетевая ошибка PolzaAI без подтверждённого ответа.",
                retryable=not is_submission,
                ambiguous_submission=is_submission,
            ) from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PolzaError(
                "PolzaAI вернул невалидный JSON.",
                ambiguous_submission=is_submission,
            ) from exc
        if not isinstance(decoded, dict):
            raise PolzaError("PolzaAI вернул неожиданный формат ответа.")
        return decoded

    def _download_with_retries(self, url: str) -> bytes:
        attempts = 0
        while True:
            try:
                request = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                    return response.read()
            except (urllib.error.URLError, TimeoutError) as exc:
                attempts += 1
                if attempts > self.max_status_retries:
                    raise PolzaError("Не удалось скачать готовое видео.", retryable=True) from exc
                self.sleep(min(2 ** (attempts - 1), 8))

    def _sanitize(self, text: str) -> str:
        sanitized = str(text)
        if self.api_key:
            sanitized = sanitized.replace(self.api_key, "[REDACTED]")
        return sanitized[:500]

    @staticmethod
    def _extract_error(body: str) -> str:
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError:
            return ""
        error = decoded.get("error") if isinstance(decoded, dict) else None
        if isinstance(error, dict):
            code = str(error.get("code") or "").strip()
            message = str(error.get("message") or error.get("detail") or "").strip()
            return f"{code}: {message}".strip(": ")
        return str(error or decoded.get("message") or "") if isinstance(decoded, dict) else ""

    @staticmethod
    def _parse_task(response: dict[str, Any]) -> PolzaTask:
        data = response.get("data") if isinstance(response.get("data"), dict) else response
        task_id = str(data.get("id") or response.get("id") or "").strip()
        status = str(data.get("status") or response.get("status") or "pending").strip().lower()
        result = data.get("data") if isinstance(data.get("data"), dict) else data
        result_url = str(result.get("url") or "").strip()
        error_raw = data.get("error") or response.get("error") or ""
        if isinstance(error_raw, dict):
            error = str(error_raw.get("message") or error_raw.get("detail") or error_raw)
        else:
            error = str(error_raw)
        return PolzaTask(task_id, status, result_url, error[:500], response)

    @staticmethod
    def _looks_like_video(content: bytes) -> bool:
        if len(content) < 12:
            return False
        return content[4:8] == b"ftyp" or content.startswith(b"\x1a\x45\xdf\xa3")
