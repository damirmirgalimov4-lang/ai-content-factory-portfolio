from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _bool(value: str, default: bool = False) -> bool:
    if not value:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


_PACKAGE_ASSETS = Path(__file__).resolve().parent / "assets"
_DEFAULT_STUDIO_ROOT = Path("/teamspace/studios/this_studio/ltx23")


@dataclass(frozen=True)
class WorkerSettings:
    root: Path
    api_token: str
    host: str = "127.0.0.1"
    port: int = 8080
    inference_enabled: bool = False
    max_request_bytes: int = 25 * 1024 * 1024
    inference_timeout_seconds: int = 1800
    comfy_request_timeout_seconds: int = 30
    comfy_poll_seconds: float = 2.0
    comfy_base_url: str = "http://127.0.0.1:8188"
    comfy_input_dir: Path = _DEFAULT_STUDIO_ROOT / "runtime" / "ComfyUI" / "input"
    comfy_output_dir: Path = _DEFAULT_STUDIO_ROOT / "runtime" / "ComfyUI" / "output"
    comfy_model_dir: Path = _DEFAULT_STUDIO_ROOT / "runtime" / "ComfyUI" / "models"
    api_workflow_path: Path = _PACKAGE_ASSETS / "ltx23-i2v-api.json"
    runtime_manifest_path: Path = _PACKAGE_ASSETS / "runtime-manifest.json"
    dependency_lock_path: Path = _PACKAGE_ASSETS / "comfyui-lightning-py312.lock"
    verification_marker: Path = _DEFAULT_STUDIO_ROOT / "runtime" / "verified.json"
    ffprobe_path: str = "ffprobe"

    @property
    def database_path(self) -> Path:
        return self.root / "jobs.sqlite3"

    @classmethod
    def from_env(cls) -> "WorkerSettings":
        studio_root = Path(os.getenv("LTX_STUDIO_ROOT", str(_DEFAULT_STUDIO_ROOT))).expanduser()
        comfy_root = Path(
            os.getenv("LTX_COMFY_ROOT", str(studio_root / "runtime" / "ComfyUI"))
        ).expanduser()
        root = Path(os.getenv("LTX_WORKER_ROOT", str(studio_root / "worker-data"))).expanduser()
        return cls(
            root=root,
            api_token=os.getenv("LTX_API_TOKEN", "").strip(),
            host=os.getenv("LTX_WORKER_HOST", "127.0.0.1").strip(),
            port=int(os.getenv("LTX_WORKER_PORT", "8080")),
            inference_enabled=_bool(os.getenv("LTX_INFERENCE_ENABLED", "")),
            max_request_bytes=int(os.getenv("LTX_MAX_REQUEST_BYTES", str(25 * 1024 * 1024))),
            inference_timeout_seconds=int(os.getenv("LTX_INFERENCE_TIMEOUT_SECONDS", "1800")),
            comfy_request_timeout_seconds=int(os.getenv("LTX_COMFY_REQUEST_TIMEOUT_SECONDS", "30")),
            comfy_poll_seconds=float(os.getenv("LTX_COMFY_POLL_SECONDS", "2")),
            comfy_base_url=os.getenv("LTX_COMFY_BASE_URL", "http://127.0.0.1:8188").strip(),
            comfy_input_dir=Path(
                os.getenv("LTX_COMFY_INPUT_DIR", str(comfy_root / "input"))
            ).expanduser(),
            comfy_output_dir=Path(
                os.getenv("LTX_COMFY_OUTPUT_DIR", str(comfy_root / "output"))
            ).expanduser(),
            comfy_model_dir=Path(
                os.getenv("LTX_COMFY_MODEL_DIR", str(comfy_root / "models"))
            ).expanduser(),
            api_workflow_path=Path(
                os.getenv("LTX_API_WORKFLOW_PATH", str(_PACKAGE_ASSETS / "ltx23-i2v-api.json"))
            ).expanduser(),
            runtime_manifest_path=Path(
                os.getenv("LTX_RUNTIME_MANIFEST_PATH", str(_PACKAGE_ASSETS / "runtime-manifest.json"))
            ).expanduser(),
            dependency_lock_path=Path(
                os.getenv(
                    "LTX_DEPENDENCY_LOCK_PATH",
                    str(_PACKAGE_ASSETS / "comfyui-lightning-py312.lock"),
                )
            ).expanduser(),
            verification_marker=Path(
                os.getenv("LTX_VERIFICATION_MARKER", str(studio_root / "runtime" / "verified.json"))
            ).expanduser(),
            ffprobe_path=os.getenv("LTX_FFPROBE_PATH", "ffprobe").strip(),
        )

    @classmethod
    def for_test(
        cls,
        root: Path,
        api_token: str = "test-token",
        **overrides: Any,
    ) -> "WorkerSettings":
        settings = cls(
            root=root,
            api_token=api_token,
            inference_enabled=True,
            comfy_input_dir=root / "comfy" / "input",
            comfy_output_dir=root / "comfy" / "output",
            comfy_model_dir=root / "comfy" / "models",
            verification_marker=root / "verified.json",
            comfy_poll_seconds=0.01,
        )
        return replace(settings, **overrides)

    def validate_server(self) -> None:
        if not self.api_token:
            raise ValueError("LTX_API_TOKEN is required")
        if len(self.api_token) < 16 and self.api_token not in {"test-token", "unit-test-token"}:
            raise ValueError("LTX_API_TOKEN must contain at least 16 characters")
        if not 1 <= self.port <= 65535:
            raise ValueError("LTX_WORKER_PORT is invalid")
        if self.max_request_bytes < 1024:
            raise ValueError("LTX_MAX_REQUEST_BYTES is too small")
        if self.inference_timeout_seconds < 1:
            raise ValueError("LTX_INFERENCE_TIMEOUT_SECONDS is invalid")
        if self.comfy_request_timeout_seconds < 1:
            raise ValueError("LTX_COMFY_REQUEST_TIMEOUT_SECONDS is invalid")
        if self.comfy_poll_seconds <= 0:
            raise ValueError("LTX_COMFY_POLL_SECONDS is invalid")

        parsed = urlsplit(self.comfy_base_url)
        if (
            parsed.scheme != "http"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or not parsed.hostname
        ):
            raise ValueError("LTX_COMFY_BASE_URL must be a plain loopback HTTP origin")
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError as exc:
            raise ValueError("LTX_COMFY_BASE_URL must use a numeric loopback address") from exc
        if not address.is_loopback:
            raise ValueError("LTX_COMFY_BASE_URL must use a loopback address")
        try:
            parsed_port = parsed.port
        except ValueError as exc:
            raise ValueError("LTX_COMFY_BASE_URL has an invalid port") from exc
        if parsed_port is not None and not 1 <= parsed_port <= 65535:
            raise ValueError("LTX_COMFY_BASE_URL has an invalid port")
