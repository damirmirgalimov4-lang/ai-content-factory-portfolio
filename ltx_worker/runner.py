from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any

from .config import WorkerSettings
from .errors import AmbiguousSubmissionError
from .service import JobRequest


EXPECTED_API_WORKFLOW_SHA256 = "f03b3482447444cf9c72272939a71295d80f49eeb6ed071385fb79bfb25e6c82"
EXPECTED_MANIFEST_SHA256 = "80209c0a04428c32ddef303f74e2b7d99a760699353f54b4b5c7c09fbf1d0111"
_MAX_JSON_BYTES = 4 * 1024 * 1024


class ComfyUiRunner:
    """Execute the pinned LTX-2.3 graph once against a loopback ComfyUI server.

    With ``remote`` set (immers mode), the ComfyUI lives on the GPU VM: the
    runner wakes it via the power API, tunnels its loopback port over SSH,
    pushes/pulls job files with scp and always shelves the VM afterwards.
    """

    def __init__(self, settings: WorkerSettings, remote: Any = None):
        self.settings = settings
        self.remote = remote

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_manifest(self) -> dict[str, Any]:
        try:
            manifest = json.loads(self.settings.runtime_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("runtime manifest is missing or invalid") from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("runtime manifest must be a JSON object")
        claimed = str(manifest.get("manifest_sha256", ""))
        unsigned = dict(manifest)
        unsigned.pop("manifest_sha256", None)
        actual = hashlib.sha256(
            json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if claimed != EXPECTED_MANIFEST_SHA256 or actual != EXPECTED_MANIFEST_SHA256:
            raise RuntimeError("runtime manifest hash does not match the reviewed release")
        return manifest

    def _load_workflow(self) -> dict[str, Any]:
        try:
            raw = self.settings.api_workflow_path.read_bytes()
            workflow = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("API workflow is missing or invalid") from exc
        if hashlib.sha256(raw).hexdigest() != EXPECTED_API_WORKFLOW_SHA256:
            raise RuntimeError("API workflow hash does not match the reviewed release")
        if not isinstance(workflow, dict) or len(workflow) != 51:
            raise RuntimeError("API workflow does not contain the reviewed 51-node graph")
        return workflow

    @staticmethod
    def _canonical_package_name(value: object) -> str:
        import re

        return re.sub(r"[-_.]+", "-", str(value).casefold())

    def _validate_bootstrap_evidence(
        self, manifest: dict[str, Any], marker: dict[str, Any]
    ) -> None:
        bootstrap = marker.get("bootstrap")
        if not isinstance(bootstrap, dict):
            raise RuntimeError("verification marker has no bootstrap evidence")
        runtime = manifest.get("runtime", {})
        comfyui = runtime.get("comfyui", {}) if isinstance(runtime, dict) else {}
        overlay = runtime.get("python_overlay", {}) if isinstance(runtime, dict) else {}
        packages = overlay.get("packages", {}) if isinstance(overlay, dict) else {}
        if not isinstance(comfyui, dict) or not isinstance(packages, dict) or not packages:
            raise RuntimeError("runtime manifest has invalid bootstrap inventory")
        expected_packages = {
            self._canonical_package_name(name): str(version) for name, version in packages.items()
        }
        if (
            bootstrap.get("comfyui_archive_sha256") != comfyui.get("archive_sha256")
            or bootstrap.get("dependency_lock_sha256") != overlay.get("sha256")
            or bootstrap.get("dependency_packages") != expected_packages
        ):
            raise RuntimeError("verification marker bootstrap evidence does not match the reviewed runtime")

        lock_path = self.settings.dependency_lock_path
        try:
            lock_info = lock_path.lstat()
        except OSError as exc:
            raise RuntimeError("reviewed dependency lock is missing") from exc
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_path.is_symlink()
            or lock_info.st_size != int(overlay.get("size_bytes", -1))
            or self._sha256(lock_path) != overlay.get("sha256")
        ):
            raise RuntimeError("reviewed dependency lock does not match the runtime manifest")

    def readiness(self) -> tuple[bool, str]:
        if self.remote is not None:
            try:
                self._load_manifest()
                self._load_workflow()
            except Exception as exc:
                return False, str(exc)
            return True, "immers remote runtime: reviewed assets pinned; GPU VM wakes on demand"
        try:
            manifest = self._load_manifest()
            self._load_workflow()
            marker = json.loads(self.settings.verification_marker.read_text(encoding="utf-8"))
            if not isinstance(marker, dict):
                raise RuntimeError("verification marker must be a JSON object")
            if (
                marker.get("status") != "verified"
                or marker.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
                or marker.get("api_workflow_sha256") != EXPECTED_API_WORKFLOW_SHA256
            ):
                raise RuntimeError("verification marker does not match the reviewed runtime")
            self._validate_bootstrap_evidence(manifest, marker)
            verified_models = marker.get("models")
            if not isinstance(verified_models, dict):
                raise RuntimeError("verification marker has no model inventory")
            for item in manifest.get("models", []):
                if not isinstance(item, dict):
                    raise RuntimeError("runtime manifest has an invalid model entry")
                relative = str(item.get("relative_path", ""))
                path = self._model_path(relative)
                expected_size = int(item.get("size_bytes", -1))
                expected_hash = str(item.get("sha256", ""))
                evidence = verified_models.get(relative)
                if (
                    not path.is_file()
                    or path.stat().st_size != expected_size
                    or not isinstance(evidence, dict)
                    or evidence.get("size_bytes") != expected_size
                    or evidence.get("sha256") != expected_hash
                ):
                    raise RuntimeError(f"model is not verified: {relative}")
            stats = self._get_json("/system_stats", timeout=self.settings.comfy_request_timeout_seconds)
            devices = stats.get("devices") if isinstance(stats, dict) else None
            if not isinstance(devices, list) or not devices:
                raise RuntimeError("ComfyUI did not report an inference device")
        except Exception as exc:
            return False, str(exc)
        return True, "verified LTX-2.3 runtime and private ComfyUI are ready"

    def _model_path(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            raise RuntimeError("model manifest contains an unsafe relative path")
        root = self.settings.comfy_model_dir.resolve()
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("model path escapes the configured model root") from exc
        return resolved

    def _url(self, path: str) -> str:
        return self.settings.comfy_base_url.rstrip("/") + path

    @staticmethod
    def _decode_json(response: Any) -> Any:
        raw = response.read(_MAX_JSON_BYTES + 1)
        if len(raw) > _MAX_JSON_BYTES:
            raise ValueError("ComfyUI JSON response is too large")
        return json.loads(raw)

    def _get_json(self, path: str, *, timeout: int) -> Any:
        request = urllib.request.Request(self._url(path), method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return self._decode_json(response)

    def _stage_input(self, request: JobRequest, input_path: Path) -> str:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(input_path, flags)
        try:
            mode = os.fstat(descriptor).st_mode
            if not stat.S_ISREG(mode):
                raise RuntimeError("staged input is not a regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                content = handle.read(20 * 1024 * 1024 + 1)
        finally:
            os.close(descriptor)
        if len(content) > 20 * 1024 * 1024 or hashlib.sha256(content).hexdigest() != request.image_sha256:
            raise RuntimeError("staged input bytes no longer match the approved request")

        subfolder = Path("ltx23-worker")
        suffix = ".png" if content.startswith(b"\x89PNG") else ".jpg"
        name = f"{request.external_prompt_id}-{request.image_sha256[:12]}{suffix}"
        destination = self.settings.comfy_input_dir / subfolder / name
        self._atomic_bytes(destination, content)
        return (subfolder / name).as_posix()

    @staticmethod
    def _atomic_bytes(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _build_workflow(self, request: JobRequest, image_name: str) -> dict[str, Any]:
        workflow = self._load_workflow()
        workflow["269"]["inputs"]["image"] = image_name
        workflow["320:319"]["inputs"]["value"] = request.prompt
        workflow["320:277"]["inputs"]["noise_seed"] = request.seed
        workflow["320:312"]["inputs"]["value"] = request.width
        workflow["320:299"]["inputs"]["value"] = request.height
        workflow["320:300"]["inputs"]["value"] = request.fps
        workflow["320:301"]["inputs"]["value"] = request.duration_seconds
        workflow["75"]["inputs"]["filename_prefix"] = f"ltx23-worker/{request.external_prompt_id}"
        workflow["75"]["inputs"]["format"] = "mp4"
        return workflow

    def _post_once(self, request: JobRequest, workflow: dict[str, Any]) -> None:
        body = json.dumps(
            {
                "prompt": workflow,
                "prompt_id": request.external_prompt_id,
                "client_id": request.external_prompt_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        http_request = urllib.request.Request(
            self._url("/prompt"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                http_request,
                timeout=self.settings.comfy_request_timeout_seconds,
            ) as response:
                accepted = self._decode_json(response)
        except urllib.error.HTTPError as exc:
            if exc.code >= 500:
                raise AmbiguousSubmissionError(
                    f"ComfyUI returned HTTP {exc.code} after the single submission"
                ) from exc
            raise RuntimeError(f"ComfyUI rejected the workflow with HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise AmbiguousSubmissionError(
                "the single ComfyUI submission was sent but its acceptance response is uncertain"
            ) from exc
        if not isinstance(accepted, dict) or accepted.get("prompt_id") != request.external_prompt_id:
            raise AmbiguousSubmissionError(
                "ComfyUI returned an invalid acceptance response after the single submission"
            )
        node_errors = accepted.get("node_errors", {})
        if node_errors not in ({}, None):
            raise AmbiguousSubmissionError(
                "ComfyUI acceptance contained unexpected node errors after submission"
            )

    def _history_entry(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.settings.inference_timeout_seconds
        last_error = ""
        while time.monotonic() < deadline:
            try:
                history = self._get_json(
                    f"/history/{prompt_id}",
                    timeout=self.settings.comfy_request_timeout_seconds,
                )
                if not isinstance(history, dict):
                    raise ValueError("history response is not an object")
                entry = history.get(prompt_id)
                if isinstance(entry, dict):
                    status_info = entry.get("status", {})
                    status_name = status_info.get("status_str") if isinstance(status_info, dict) else ""
                    if status_name == "error":
                        raise RuntimeError("ComfyUI reported a terminal inference error")
                    if status_info.get("completed") if isinstance(status_info, dict) else False:
                        return entry
                last_error = ""
            except RuntimeError:
                raise
            except Exception as exc:
                last_error = type(exc).__name__
            time.sleep(self.settings.comfy_poll_seconds)
        self._interrupt(prompt_id)
        suffix = f"; last status error: {last_error}" if last_error else ""
        raise AmbiguousSubmissionError(
            "ComfyUI did not commit a terminal result before the hard timeout" + suffix
        )

    def _interrupt(self, prompt_id: str) -> None:
        body = json.dumps({"prompt_id": prompt_id}).encode("utf-8")
        request = urllib.request.Request(
            self._url("/interrupt"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.settings.comfy_request_timeout_seconds,
            ) as response:
                response.read(1024)
        except Exception:
            return

    def _result_ref(self, entry: dict[str, Any], prompt_id: str) -> tuple[str, str]:
        outputs = entry.get("outputs")
        node_output = outputs.get("75") if isinstance(outputs, dict) else None
        if not isinstance(node_output, dict):
            raise RuntimeError("ComfyUI history has no output from the reviewed SaveVideo node")
        candidates: list[Any] = []
        for key in ("images", "videos", "video"):
            value = node_output.get(key)
            if isinstance(value, list):
                candidates.extend(value)
            elif isinstance(value, dict):
                candidates.append(value)
        matches = [item for item in candidates if isinstance(item, dict) and str(item.get("filename", "")).endswith(".mp4")]
        if len(matches) != 1:
            raise RuntimeError("ComfyUI history does not identify exactly one MP4 result")
        item = matches[0]
        filename = str(item.get("filename", ""))
        subfolder = str(item.get("subfolder", ""))
        if (
            item.get("type") != "output"
            or subfolder != "ltx23-worker"
            or Path(filename).name != filename
            or not filename.startswith(prompt_id + "_")
        ):
            raise RuntimeError("ComfyUI returned an output outside the reserved job namespace")
        return subfolder, filename

    @staticmethod
    def _checked_result_file(root: Path, subfolder: str, filename: str) -> Path:
        candidate = (root / subfolder / filename).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("ComfyUI result escapes the configured output root") from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError("ComfyUI result is not a regular owned file")
        return candidate

    def _result_source(self, entry: dict[str, Any], prompt_id: str) -> Path:
        subfolder, filename = self._result_ref(entry, prompt_id)
        root = self.settings.comfy_output_dir.resolve(strict=True)
        return self._checked_result_file(root, subfolder, filename)

    def _remote_result_source(self, entry: dict[str, Any], prompt_id: str) -> Path:
        subfolder, filename = self._result_ref(entry, prompt_id)
        root = self.settings.comfy_output_dir.resolve()
        (root / subfolder).mkdir(parents=True, exist_ok=True)
        target = root / subfolder / filename
        self.remote.pull_output(f"{subfolder}/{filename}", str(target))
        return self._checked_result_file(root, subfolder, filename)

    @staticmethod
    def _atomic_copy(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
        source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(source, source_flags)
        target_fd = os.open(temporary, target_flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(source_fd).st_mode):
                raise RuntimeError("ComfyUI result changed before it could be captured")
            with os.fdopen(source_fd, "rb", closefd=False) as reader, os.fdopen(
                target_fd, "wb", closefd=False
            ) as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, target)
        finally:
            os.close(source_fd)
            os.close(target_fd)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def run(self, request: JobRequest, input_path: Path, output_path: Path) -> dict[str, object]:
        if self.remote is not None:
            return self._run_remote(request, input_path, output_path)
        return self._run_inference(request, input_path, output_path, remote=False)

    def _run_remote(
        self, request: JobRequest, input_path: Path, output_path: Path
    ) -> dict[str, object]:
        # Money rule: whatever happens once remote mode starts — including a
        # failure inside up() itself — the VM goes back to the shelf in
        # `finally` so billing stops. If the work failed AND shelving failed,
        # the work error wins; if only shelving failed, that error is raised.
        work_error: BaseException | None = None
        try:
            self.remote.up()
            tunnel = self.remote.open_tunnel(self._tunnel_port())
            try:
                return self._run_inference(request, input_path, output_path, remote=True)
            finally:
                tunnel.close()
        except BaseException as exc:
            work_error = exc
            raise
        finally:
            try:
                self.remote.down()
            except Exception:
                if work_error is None:
                    raise

    def _tunnel_port(self) -> int:
        from urllib.parse import urlsplit

        return urlsplit(self.settings.comfy_base_url).port or 8188

    def _run_inference(
        self, request: JobRequest, input_path: Path, output_path: Path, *, remote: bool
    ) -> dict[str, object]:
        image_name = self._stage_input(request, input_path)
        if remote:
            self.remote.push_input(
                str(self.settings.comfy_input_dir / image_name), image_name
            )
        workflow = self._build_workflow(request, image_name)
        self._post_once(request, workflow)
        entry = self._history_entry(request.external_prompt_id)
        if remote:
            source = self._remote_result_source(entry, request.external_prompt_id)
        else:
            source = self._result_source(entry, request.external_prompt_id)
        self._atomic_copy(source, output_path)
        metrics = self._probe(output_path)
        if not metrics.get("has_video") or not metrics.get("has_audio"):
            raise RuntimeError("result MP4 does not contain both video and audio streams")
        duration = float(metrics.get("duration_seconds") or 0)
        fps = float(metrics.get("fps") or 0)
        if not 4.5 <= duration <= 5.5:
            raise RuntimeError(f"result duration is outside the 5-second contract: {duration}")
        if (metrics.get("width"), metrics.get("height")) != (request.width, request.height):
            raise RuntimeError("result resolution does not match the approved contract")
        if not 23.9 <= fps <= 24.1:
            raise RuntimeError("result frame rate does not match the approved contract")
        return {**metrics, "external_prompt_id": request.external_prompt_id}

    def _probe(self, output_path: Path) -> dict[str, Any]:
        completed = subprocess.run(
            [
                self.settings.ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,width,height,avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError("ffprobe could not validate result.mp4")
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("ffprobe returned invalid JSON") from exc
        streams = data.get("streams", [])
        video = next((item for item in streams if item.get("codec_type") == "video"), {})
        rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
        try:
            fps = float(Fraction(str(rate)))
        except (ValueError, ZeroDivisionError):
            fps = 0.0
        return {
            "duration_seconds": float(data.get("format", {}).get("duration", 0)),
            "width": int(video.get("width", 0)),
            "height": int(video.get("height", 0)),
            "fps": fps,
            "has_video": bool(video),
            "has_audio": any(item.get("codec_type") == "audio" for item in streams),
        }
