from __future__ import annotations

import hmac
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

from .config import WorkerSettings
from .service import (
    InvalidJobRequest,
    JobConflict,
    JobNotFound,
    VideoWorkerService,
    WorkerNotReady,
)


class WorkerHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, settings: WorkerSettings, service: VideoWorkerService):
        self.settings = settings
        self.service = service
        super().__init__(address, WorkerRequestHandler)


class WorkerRequestHandler(BaseHTTPRequestHandler):
    server: WorkerHttpServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Never log authorization headers or request bodies. The default line is safe.
        super().log_message(format, *args)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str) -> None:
        self._json(status, {"error": {"code": code, "message": message}})

    def _authorized(self) -> bool:
        expected = f"Bearer {self.server.settings.api_token}"
        supplied = self.headers.get("Authorization", "")
        if hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
            return True
        self._error(HTTPStatus.UNAUTHORIZED, "unauthorized", "valid bearer token required")
        return False

    def _path_parts(self) -> list[str]:
        return [unquote(part) for part in urlsplit(self.path).path.split("/") if part]

    def do_GET(self) -> None:  # noqa: N802
        parts = self._path_parts()
        if parts == ["health"]:
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        if not self._authorized():
            return
        if parts == ["ready"]:
            ready, reason = self.server.service.ready()
            self._json(
                HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                {"ready": ready, "reason": reason},
            )
            return
        try:
            if len(parts) == 3 and parts[:2] == ["video", "jobs"]:
                self._json(HTTPStatus.OK, self.server.service.get(parts[2]))
                return
            if len(parts) == 4 and parts[:2] == ["video", "jobs"] and parts[3] == "result":
                path = self.server.service.result_path(parts[2])
                size = path.stat().st_size
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(size))
                self.send_header("Cache-Control", "private, no-store")
                self.end_headers()
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        self.wfile.write(chunk)
                return
            self._error(HTTPStatus.NOT_FOUND, "not_found", "route not found")
        except JobNotFound:
            self._error(HTTPStatus.NOT_FOUND, "job_not_found", "video job not found")
        except JobConflict as exc:
            self._error(HTTPStatus.CONFLICT, "result_not_ready", str(exc))

    def do_POST(self) -> None:  # noqa: N802
        parts = self._path_parts()
        if parts != ["video", "jobs"]:
            self._error(HTTPStatus.NOT_FOUND, "not_found", "route not found")
            return
        if not self._authorized():
            return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "invalid_content_type", "application/json required")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length <= 0 or length > self.server.settings.max_request_bytes:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "invalid_size", "request body size is invalid")
            return
        try:
            payload = json.loads(self.rfile.read(length))
            job, created = self.server.service.submit(payload)
        except json.JSONDecodeError:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_json", "request body is not valid JSON")
            return
        except InvalidJobRequest as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request", str(exc))
            return
        except WorkerNotReady as exc:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "not_ready", str(exc))
            return
        except JobConflict as exc:
            self._error(HTTPStatus.CONFLICT, "job_conflict", str(exc))
            return
        self._json(HTTPStatus.CREATED if created else HTTPStatus.OK, job)


def create_server(
    address: tuple[str, int],
    settings: WorkerSettings,
    service: VideoWorkerService,
) -> WorkerHttpServer:
    settings.validate_server()
    return WorkerHttpServer(address, settings, service)
