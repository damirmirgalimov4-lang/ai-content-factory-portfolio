from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
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


class KieError(VideoProviderError):
    pass


class KieClient:
    """Kie Market API adapter for durable Seedance 2 video tasks."""

    provider_name = "kie"

    def __init__(
        self,
        settings: Settings,
        *,
        request_timeout_seconds: int = 90,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = settings.kie_api_key
        self.base_url = settings.kie_base_url
        self.upload_base_url = settings.kie_upload_base_url
        self.poll_interval_seconds = settings.kie_poll_interval_seconds
        self.timeout_seconds = settings.kie_timeout_seconds
        self.max_status_retries = settings.kie_max_status_retries
        self.request_timeout_seconds = request_timeout_seconds
        self.sleep = sleep

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def create_video_task(self, request: VideoGenerationRequest) -> VideoTask:
        """Upload one first frame or one ordered multimodal reference pack."""

        if not self.is_configured:
            raise KieError("Kie API key не задан.")
        self._validate_request(request)
        first_frame_url = ""
        reference_image_urls: list[str] = []
        if request.reference_image_paths:
            reference_image_urls = [
                self.upload_file(path) for path in request.reference_image_paths
            ]
        elif request.image_path is not None:
            if not request.image_path.is_file():
                raise KieError(
                    f"Исходный кадр не найден: {request.image_path.name}"
                )
            first_frame_url = self.upload_file(request.image_path)

        model_input: dict[str, Any] = {
            "prompt": request.prompt,
            "generate_audio": request.sound_enabled,
            "resolution": request.resolution,
            "aspect_ratio": request.aspect_ratio,
            "duration": request.duration_seconds,
            "web_search": False,
        }
        if reference_image_urls:
            # Kie's first-frame and multimodal-reference modes are mutually exclusive.
            # In this mode @Image1 is explicitly assigned as the first frame in the prompt.
            model_input["reference_image_urls"] = reference_image_urls
        elif first_frame_url:
            model_input["first_frame_url"] = first_frame_url

        response = self._request_json(
            "POST",
            "/api/v1/jobs/createTask",
            {"model": request.model, "input": model_input},
            is_submission=True,
        )
        self._raise_api_error(response, is_submission=True)
        data = response.get("data")
        task_id = (
            str(data.get("taskId") or "").strip()
            if isinstance(data, dict)
            else ""
        )
        if not task_id:
            raise KieError(
                "Kie принял запрос, но не вернул task ID; автоматический повтор заблокирован.",
                ambiguous_submission=True,
            )
        return VideoTask(task_id, "pending", raw=response)

    def upload_file(self, path: Path) -> str:
        """Upload a local image through Kie's free temporary Base64 upload API."""

        content = path.read_bytes()
        if not content:
            raise KieError("Исходный кадр пустой.")
        if len(content) > 10 * 1024 * 1024:
            raise KieError(
                "Исходный кадр больше 10 МБ; для Base64 upload Kie нужен меньший файл."
            )
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        suffix = path.suffix.lower() or mimetypes.guess_extension(mime) or ".png"
        digest = hashlib.sha256(content).hexdigest()[:20]
        response = self._request_json(
            "POST",
            "/api/file-base64-upload",
            {
                "base64Data": (
                    f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"
                ),
                "uploadPath": "images/content-factory",
                "fileName": f"{digest}{suffix}",
            },
            base_url=self.upload_base_url,
        )
        self._raise_api_error(response)
        data = response.get("data")
        if not isinstance(data, dict):
            raise KieError("Kie File Upload вернул неожиданный формат ответа.")
        url = str(data.get("fileUrl") or data.get("downloadUrl") or "").strip()
        if not url.startswith("https://"):
            raise KieError("Kie File Upload не вернул безопасный URL файла.")
        return url

    def get_task(self, task_id: str) -> VideoTask:
        clean_id = task_id.strip()
        if not clean_id:
            raise KieError("Пустой Kie task ID.")
        query = urllib.parse.urlencode({"taskId": clean_id})
        response = self._safe_status_request(
            f"/api/v1/jobs/recordInfo?{query}"
        )
        self._raise_api_error(response)
        return self._parse_task(response, clean_id)

    def wait_for_task(self, task_id: str) -> VideoTask:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            task = self.get_task(task_id)
            if task.status in TERMINAL_STATUSES:
                return task
            if task.status not in ACTIVE_STATUSES:
                raise KieError(f"Неизвестный статус Kie: {task.status}")
            self.sleep(self.poll_interval_seconds)
        raise KieError("Истёк timeout ожидания Kie video task.", retryable=True)

    def get_credits(self) -> float:
        response = self._request_json("GET", "/api/v1/chat/credit")
        self._raise_api_error(response)
        value = response.get("data")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise KieError("Kie вернул неожиданный формат баланса.") from exc

    def download_video(self, url: str, target: Path) -> Path:
        if not url.startswith("https://"):
            raise KieError("Kie вернул небезопасный или пустой result URL.")
        content = self._download_with_retries(url)
        if not self._looks_like_video(content):
            raise KieError("Полученный файл не похож на поддерживаемое видео.")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
        return target

    @staticmethod
    def _validate_request(request: VideoGenerationRequest) -> None:
        if request.model not in {
            "bytedance/seedance-2",
            "bytedance/seedance-2-fast",
        }:
            raise KieError("Kie adapter настроен только для Seedance 2.")
        if request.duration_seconds not in {5, 10, 15}:
            raise KieError("Seedance 2 поддерживает длительность 5, 10 или 15 секунд.")
        if request.resolution not in {"480p", "720p"}:
            raise KieError(
                "Kie Seedance 2 поддерживает подтверждённые разрешения 480p и 720p."
            )
        if request.aspect_ratio not in {
            "16:9",
            "4:3",
            "1:1",
            "3:4",
            "9:16",
            "21:9",
        }:
            raise KieError("Seedance 2 не поддерживает выбранное соотношение сторон.")
        if not 3 <= len(request.prompt.strip()) <= 5000:
            raise KieError("Seedance 2 принимает prompt длиной от 3 до 5000 символов.")
        references = tuple(request.reference_image_paths)
        if references and request.image_path is not None:
            raise KieError(
                "Kie не позволяет смешивать first_frame_url и reference_image_urls."
            )
        if len(references) > 9:
            raise KieError("Seedance 2 принимает не больше 9 изображений-референсов.")
        if references:
            resolved = [path.resolve() for path in references]
            if len(set(resolved)) != len(resolved):
                raise KieError("В Seedance-пакете повторяется один и тот же файл.")
            missing_files = [path.name for path in references if not path.is_file()]
            if missing_files:
                raise KieError(
                    "Не найдены изображения-референсы: " + ", ".join(missing_files)
                )
            found_tags = {
                int(value) for value in re.findall(r"@Image([1-9]\d*)", request.prompt)
            }
            expected_tags = set(range(1, len(references) + 1))
            if found_tags != expected_tags:
                raise KieError(
                    "Теги @ImageN в видеопромпте не совпадают с прикреплёнными "
                    "референсами."
                )

    def _safe_status_request(self, path: str) -> dict[str, Any]:
        attempts = 0
        while True:
            try:
                return self._request_json("GET", path)
            except KieError as exc:
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
        base_url: str | None = None,
        is_submission: bool = False,
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{base_url or self.base_url}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/plain, */*",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/138.0.0.0 Safari/537.36"
                ),
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.request_timeout_seconds
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = self._http_error_message(exc.code, body)
            raise KieError(
                message,
                status_code=exc.code,
                retryable=exc.code == 429 or exc.code >= 500,
                ambiguous_submission=is_submission and exc.code >= 500,
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise KieError(
                "Сетевая ошибка Kie без подтверждённого ответа.",
                retryable=not is_submission,
                ambiguous_submission=is_submission,
            ) from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise KieError(
                "Kie вернул невалидный JSON.",
                ambiguous_submission=is_submission,
            ) from exc
        if not isinstance(decoded, dict):
            raise KieError("Kie вернул неожиданный формат ответа.")
        return decoded

    def _raise_api_error(
        self,
        response: dict[str, Any],
        *,
        is_submission: bool = False,
    ) -> None:
        raw_code = response.get("code", 200)
        try:
            code = int(raw_code)
        except (TypeError, ValueError):
            code = 500
        success = response.get("success")
        data = response.get("data")
        has_task_state = (
            isinstance(data, dict)
            and bool(data.get("taskId"))
            and bool(data.get("state"))
        )
        if (code == 200 or has_task_state) and success is not False:
            return
        message = str(response.get("msg") or response.get("message") or "ошибка API")
        status_code = code if 400 <= code <= 599 else None
        raise KieError(
            self._sanitize(f"Kie API {code}: {message}"),
            status_code=status_code,
            retryable=code == 429 or code >= 500,
            ambiguous_submission=is_submission and code >= 500,
        )

    def _http_error_message(self, code: int, body: str) -> str:
        provider_message = self._extract_error(body)
        if code == 401:
            message = "Kie HTTP 401: API-ключ отклонён."
        elif code == 402:
            message = "Kie HTTP 402: недостаточно кредитов для генерации."
        elif code == 403:
            message = "Kie HTTP 403: доступ к операции запрещён."
        elif code == 429:
            message = "Kie временно ограничил частоту запросов."
        elif code >= 500:
            message = "Временная серверная ошибка Kie."
        else:
            message = f"Kie HTTP {code}."
        if provider_message and code not in {429}:
            message += f" Ответ сервиса: {provider_message}"
        return self._sanitize(message)

    @staticmethod
    def _parse_task(response: dict[str, Any], fallback_id: str) -> VideoTask:
        data = response.get("data")
        if not isinstance(data, dict):
            data = {}
        raw_state = str(data.get("state") or "waiting").strip().lower()
        status = {
            "waiting": "pending",
            "wait": "pending",
            "queuing": "pending",
            "queueing": "pending",
            "generating": "processing",
            "processing": "processing",
            "success": "completed",
            "completed": "completed",
            "fail": "failed",
            "failed": "failed",
            "cancelled": "cancelled",
            "canceled": "cancelled",
        }.get(raw_state, raw_state)
        result_urls = KieClient._result_urls(data.get("resultJson"))
        if not result_urls:
            result_urls = KieClient._result_urls(data.get("resultUrls"))
        result_url = result_urls[0] if result_urls else ""
        error = str(data.get("failMsg") or data.get("failCode") or "").strip()
        task_id = str(data.get("taskId") or fallback_id).strip()
        return VideoTask(task_id, status, result_url, error[:500], response)

    @staticmethod
    def _result_urls(value: Any) -> list[str]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return [value] if value.startswith("https://") else []
        if isinstance(value, dict):
            value = value.get("resultUrls") or value.get("urls") or value.get("url")
        if isinstance(value, str):
            return [value] if value.startswith("https://") else []
        if isinstance(value, list):
            return [
                str(item)
                for item in value
                if str(item).startswith("https://")
            ]
        return []

    def _download_with_retries(self, url: str) -> bytes:
        attempts = 0
        while True:
            try:
                # Kie's result CDN rejects urllib's default user agent with 403.
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0 Safari/537.36"
                        ),
                        "Accept": "video/mp4,video/*;q=0.9,*/*;q=0.8",
                    },
                    method="GET",
                )
                with urllib.request.urlopen(
                    request, timeout=self.request_timeout_seconds
                ) as response:
                    return response.read()
            except (urllib.error.URLError, TimeoutError) as exc:
                attempts += 1
                if attempts > self.max_status_retries:
                    raise KieError(
                        "Не удалось скачать готовое видео Kie.", retryable=True
                    ) from exc
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
        if not isinstance(decoded, dict):
            return ""
        error = decoded.get("error")
        if isinstance(error, dict):
            return str(
                error.get("message") or error.get("detail") or error.get("code") or ""
            ).strip()
        return str(error or decoded.get("msg") or decoded.get("message") or "").strip()

    @staticmethod
    def _looks_like_video(content: bytes) -> bool:
        if len(content) < 12:
            return False
        return content[4:8] == b"ftyp" or content.startswith(b"\x1a\x45\xdf\xa3")
