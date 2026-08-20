from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .config import Settings
from .video_provider import VideoGenerationRequest, VideoProviderError, VideoTask


class LtxError(VideoProviderError):
    pass


class LtxClient:
    """Narrow client for the authenticated asynchronous LTX worker.

    Creating a video is deliberately attempted exactly once. Safe status and
    artifact reads may be retried; a lost create response is always ambiguous.
    """

    provider_name = "ltx"
    _IDEMPOTENCY_KEY = re.compile(r"^[0-9a-f]{64}$")
    _MAX_RESULT_BYTES = 2 * 1024 * 1024 * 1024

    def __init__(self, settings: Settings):
        self.enabled = bool(settings.ltx_video_enabled)
        self.api_token = settings.ltx_api_token.strip()
        self.base_url = settings.ltx_base_url.strip().rstrip("/")
        self.poll_interval_seconds = settings.ltx_poll_interval_seconds
        self.timeout_seconds = settings.ltx_timeout_seconds
        self.max_status_retries = settings.ltx_max_status_retries

    @property
    def is_configured(self) -> bool:
        return bool(self.enabled and self.api_token and self.base_url)

    def create_video_task(self, request: VideoGenerationRequest) -> VideoTask:
        self._require_configured()
        self._validate_request(request)
        image_path = request.image_path
        if image_path is None or not image_path.is_file():
            raise LtxError("LTX требует существующий локальный стартовый кадр.")
        image = image_path.read_bytes()
        if not image or len(image) > 20 * 1024 * 1024:
            raise LtxError("Стартовый кадр LTX должен занимать от 1 байта до 20 МиБ.")
        job_id = f"cf-{request.idempotency_key}"
        payload = {
            "job_id": job_id,
            "model": "ltx-2.3",
            "workflow": "distilled",
            "prompt": request.prompt,
            "seed": request.seed,
            "duration_seconds": 5,
            "aspect_ratio": "16:9",
            "width": 1024,
            "height": 576,
            "fps": 24,
            "audio": True,
            "image_base64": base64.b64encode(image).decode("ascii"),
        }
        http_request = urllib.request.Request(
            self._url("/video/jobs"),
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers=self._headers(content_type=True),
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=min(self.timeout_seconds, 120)) as response:
                body = self._decode_json(response.read())
        except urllib.error.HTTPError as exc:
            ambiguous = exc.code >= 500
            raise LtxError(
                self._http_message(exc.code),
                status_code=exc.code,
                retryable=exc.code in {429, 500, 502, 503, 504},
                ambiguous_submission=ambiguous,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LtxError(
                "Ответ LTX worker потерян; повторная платная отправка автоматически заблокирована.",
                retryable=False,
                ambiguous_submission=True,
            ) from exc
        return self._task(body, expected_job_id=job_id)

    def get_task(self, task_id: str) -> VideoTask:
        self._require_configured()
        safe_id = urllib.parse.quote(task_id, safe="")
        body = self._get_json(f"/video/jobs/{safe_id}")
        return self._task(body, expected_job_id=task_id)

    def download_video(self, url: str, target: Path) -> Path:
        self._require_configured()
        resolved = self._same_origin_url(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        last_error: Exception | None = None
        for attempt in range(self.max_status_retries + 1):
            try:
                request = urllib.request.Request(
                    resolved,
                    headers=self._headers(),
                    method="GET",
                )
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    total = 0
                    first = b""
                    with temporary.open("wb") as handle:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            if not first:
                                first = chunk[:64]
                            total += len(chunk)
                            if total > self._MAX_RESULT_BYTES:
                                raise LtxError("LTX result превышает безопасный предел 2 ГиБ.")
                            handle.write(chunk)
                if total < 12 or b"ftyp" not in first[:32]:
                    raise LtxError("LTX worker вернул файл без сигнатуры MP4.")
                temporary.replace(target)
                return target
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.max_status_retries:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt >= self.max_status_retries:
                    break
            finally:
                if temporary.exists() and (last_error is not None):
                    temporary.unlink(missing_ok=True)
            time.sleep(min(2**attempt, 4))
        raise LtxError("Не удалось безопасно скачать MP4 из LTX worker.", retryable=True) from last_error

    def _get_json(self, path: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_status_retries + 1):
            request = urllib.request.Request(self._url(path), headers=self._headers(), method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return self._decode_json(response.read())
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.max_status_retries:
                    raise LtxError(
                        self._http_message(exc.code),
                        status_code=exc.code,
                        retryable=exc.code in {429, 500, 502, 503, 504},
                    ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt >= self.max_status_retries:
                    break
            time.sleep(min(2**attempt, 4))
        raise LtxError("LTX worker недоступен при безопасной проверке статуса.", retryable=True) from last_error

    def _validate_request(self, request: VideoGenerationRequest) -> None:
        if request.provider != "ltx" or request.model != "ltx-2.3":
            raise LtxError("LTX adapter принимает только provider=ltx и model=ltx-2.3.")
        if request.mode != "distilled":
            raise LtxError("LTX adapter принимает только distilled workflow.")
        if (
            request.duration_seconds != 5
            or request.aspect_ratio != "16:9"
            or request.resolution != "1024x576"
            or not request.sound_enabled
        ):
            raise LtxError("LTX pre-benchmark contract: 5 секунд, 16:9, 1024x576, со звуком.")
        if request.reference_image_paths:
            raise LtxError("LTX pre-benchmark adapter принимает ровно один стартовый кадр.")
        if not request.prompt.strip():
            raise LtxError("Пустой видеопромпт LTX запрещён.")
        if not self._IDEMPOTENCY_KEY.fullmatch(request.idempotency_key):
            raise LtxError("LTX требует durable 64-символьный idempotency key до отправки.")
        if not 0 <= request.seed <= 2**63 - 1:
            raise LtxError("Seed LTX находится вне разрешённого диапазона.")

    def _task(self, payload: dict[str, Any], *, expected_job_id: str) -> VideoTask:
        job_id = str(payload.get("job_id", ""))
        if job_id != expected_job_id:
            raise LtxError("LTX worker вернул неожиданный job_id.")
        raw_status = str(payload.get("status", "")).strip().lower()
        status = {
            "queued": "pending",
            "running": "processing",
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
            "reconciliation_required": "failed",
        }.get(raw_status)
        if status is None:
            raise LtxError(f"LTX worker вернул неизвестный статус: {raw_status or 'empty'}.")
        result_url = str(payload.get("result_url", "")).strip()
        if result_url:
            result_url = self._same_origin_url(result_url)
        error = str(payload.get("error", "")).strip()
        if raw_status == "reconciliation_required" and not error:
            error = "LTX worker требует ручной сверки; автоматический повтор заблокирован."
        return VideoTask(job_id, status, result_url, error, payload)

    def _headers(self, *, content_type: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json, video/mp4",
            "User-Agent": "ContentFactory-LTX/0.1",
        }
        if content_type:
            headers["Content-Type"] = "application/json"
        return headers

    def _url(self, path: str) -> str:
        return self.base_url + "/" + path.lstrip("/")

    def _same_origin_url(self, value: str) -> str:
        resolved = urllib.parse.urljoin(self.base_url + "/", value)
        base = urllib.parse.urlsplit(self.base_url)
        target = urllib.parse.urlsplit(resolved)
        if (target.scheme, target.netloc) != (base.scheme, base.netloc):
            raise LtxError("LTX result URL указывает за пределы настроенного worker origin.")
        return resolved

    @staticmethod
    def _decode_json(body: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LtxError("LTX worker вернул повреждённый JSON.") from exc
        if not isinstance(payload, dict):
            raise LtxError("LTX worker вернул JSON неправильного типа.")
        return payload

    @staticmethod
    def _http_message(status: int) -> str:
        if status in {401, 403}:
            return "LTX worker отклонил авторизацию."
        if status == 409:
            return "LTX worker обнаружил конфликт durable job_id; повтор заблокирован."
        if status == 422:
            return "LTX worker отклонил параметры до генерации."
        if status == 503:
            return "LTX runtime ещё не готов к inference."
        return f"LTX worker вернул HTTP {status}."

    def _require_configured(self) -> None:
        if not self.enabled:
            raise LtxError("LTX video feature flag выключен.")
        if not self.base_url or not self.api_token:
            raise LtxError("LTX worker URL или API token не настроен.")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise LtxError("LTX_BASE_URL должен быть отдельным HTTPS endpoint.")
