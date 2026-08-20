from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .config import WorkerSettings


EXPECTED_MANIFEST_SHA256 = "80209c0a04428c32ddef303f74e2b7d99a760699353f54b4b5c7c09fbf1d0111"
_MAX_ARCHIVE_ENTRIES = 20_000
_MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024
_PRESERVED_RUNTIME_DIRS = {"models", "input", "output", ".packages"}
_PREPARATION_RECEIPT = ".ltx-prepared.json"
_DEPENDENCY_RECEIPT = ".ltx-dependencies.json"


class RuntimeBootstrap:
    """Prepare reviewed runtime files and verify pre-positioned model bytes.

    This class intentionally has no network download method. Model acquisition is
    a separate operator-approved action; inference remains blocked until every
    expected byte is present and ``verified.json`` is committed.
    """

    def __init__(
        self,
        settings: WorkerSettings,
        *,
        expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
    ) -> None:
        self.settings = settings
        self.expected_manifest_sha256 = expected_manifest_sha256

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _manifest(self) -> dict[str, Any]:
        try:
            manifest = json.loads(self.settings.runtime_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("runtime manifest is missing or invalid") from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("runtime manifest must be a JSON object")
        claimed = manifest.get("manifest_sha256")
        unsigned = dict(manifest)
        unsigned.pop("manifest_sha256", None)
        actual = hashlib.sha256(
            json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if claimed != self.expected_manifest_sha256 or actual != self.expected_manifest_sha256:
            raise RuntimeError("runtime manifest hash does not match the approved release")
        return manifest

    @property
    def _runtime_root(self) -> Path:
        return self.settings.comfy_input_dir.parent

    @property
    def _preparation_receipt(self) -> Path:
        return self._runtime_root / _PREPARATION_RECEIPT

    @property
    def _dependency_receipt(self) -> Path:
        return self._runtime_root / _DEPENDENCY_RECEIPT

    @staticmethod
    def _canonical_package_name(value: object) -> str:
        return re.sub(r"[-_.]+", "-", str(value).casefold())

    @classmethod
    def _installed_packages(cls, root: Path) -> dict[str, str]:
        installed: dict[str, str] = {}
        for metadata in root.glob("*.dist-info/METADATA"):
            if metadata.is_symlink() or not metadata.is_file():
                raise RuntimeError("installed Python overlay contains unsafe metadata")
            name = ""
            version = ""
            for line in metadata.read_text(encoding="utf-8").splitlines():
                if line.startswith("Name: "):
                    name = line[6:].strip()
                elif line.startswith("Version: "):
                    version = line[9:].strip()
                if name and version:
                    break
            canonical = cls._canonical_package_name(name)
            if not canonical or not version or canonical in installed:
                raise RuntimeError("installed Python overlay has invalid package metadata")
            installed[canonical] = version
        return installed

    def _dependency_spec(
        self, manifest: dict[str, Any]
    ) -> tuple[dict[str, Any], Path, dict[str, str]]:
        overlay = manifest.get("runtime", {}).get("python_overlay", {})
        if not isinstance(overlay, dict):
            raise RuntimeError("runtime manifest has no approved Python overlay")
        lock_path = self.settings.dependency_lock_path.expanduser().resolve(strict=True)
        if not lock_path.is_file() or lock_path.is_symlink():
            raise RuntimeError("dependency lock must be a regular local file")
        expected_size = int(overlay.get("size_bytes", -1))
        if lock_path.stat().st_size != expected_size or self._sha256(lock_path) != overlay.get(
            "sha256"
        ):
            raise RuntimeError("dependency lock hash does not match the approved release")
        packages = overlay.get("packages", {})
        if not isinstance(packages, dict) or not packages:
            raise RuntimeError("dependency lock package inventory is invalid")
        expected = {
            self._canonical_package_name(name): str(version) for name, version in packages.items()
        }
        if len(expected) != len(packages):
            raise RuntimeError("dependency lock contains duplicate canonical package names")
        return overlay, lock_path, expected

    @staticmethod
    def _read_receipt(path: Path, label: str) -> dict[str, Any]:
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or path.is_symlink():
                raise RuntimeError
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, RuntimeError) as exc:
            raise RuntimeError(f"{label} receipt is missing or invalid") from exc
        if not isinstance(receipt, dict):
            raise RuntimeError(f"{label} receipt is missing or invalid")
        return receipt

    def _validated_preparation_receipt(self, manifest: dict[str, Any]) -> dict[str, Any]:
        receipt = self._read_receipt(self._preparation_receipt, "runtime preparation")
        comfyui = manifest.get("runtime", {}).get("comfyui", {})
        main = self._runtime_root / "main.py"
        if (
            receipt.get("status") != "prepared"
            or receipt.get("manifest_sha256") != self.expected_manifest_sha256
            or receipt.get("archive_sha256") != comfyui.get("archive_sha256")
            or receipt.get("comfyui_commit") != comfyui.get("commit")
            or main.is_symlink()
            or not main.is_file()
        ):
            raise RuntimeError("runtime preparation receipt does not match the approved release")
        return receipt

    def _validated_dependency_receipt(
        self, manifest: dict[str, Any], expected_packages: dict[str, str]
    ) -> dict[str, Any]:
        receipt = self._read_receipt(self._dependency_receipt, "dependency")
        overlay = manifest.get("runtime", {}).get("python_overlay", {})
        target = self._runtime_root / ".packages"
        if (
            receipt.get("status") != "dependencies_installed"
            or receipt.get("manifest_sha256") != self.expected_manifest_sha256
            or receipt.get("lock_sha256") != overlay.get("sha256")
            or receipt.get("packages") != expected_packages
            or target.is_symlink()
            or not target.is_dir()
            or self._installed_packages(target) != expected_packages
        ):
            raise RuntimeError("dependency receipt does not match the approved release")
        return receipt

    @staticmethod
    def _safe_member_path(name: str) -> tuple[str, ...]:
        path = PurePosixPath(name)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise RuntimeError("ComfyUI archive contains an unsafe path")
        parts = tuple(part for part in path.parts if part not in {"", "."})
        if len(parts) < 2:
            return ()
        return parts[1:]

    def _extract_reviewed_archive(self, archive: Path, staging: Path) -> int:
        count = 0
        total = 0
        seen: set[tuple[str, ...]] = set()
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                count += 1
                if count > _MAX_ARCHIVE_ENTRIES:
                    raise RuntimeError("ComfyUI archive contains too many entries")
                relative = self._safe_member_path(member.name)
                if not relative:
                    if member.isdir():
                        continue
                    raise RuntimeError("ComfyUI archive has a file outside its top-level directory")
                if relative in seen:
                    raise RuntimeError("ComfyUI archive contains duplicate paths")
                seen.add(relative)
                target = staging.joinpath(*relative)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise RuntimeError("ComfyUI archive contains links or special files")
                total += int(member.size)
                if total > _MAX_EXTRACTED_BYTES:
                    raise RuntimeError("ComfyUI archive expands beyond the safety limit")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError("ComfyUI archive entry cannot be read")
                descriptor = os.open(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                try:
                    with os.fdopen(descriptor, "wb", closefd=False) as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                finally:
                    source.close()
                    os.close(descriptor)
                os.chmod(target, 0o755 if member.mode & 0o111 else 0o644)
        if not (staging / "main.py").is_file():
            raise RuntimeError("reviewed ComfyUI archive has no main.py")
        return count

    @staticmethod
    def _remove_marker(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return

    @staticmethod
    def _prepare_runtime_root(runtime_root: Path) -> None:
        runtime_root.parent.mkdir(parents=True, exist_ok=True)
        if runtime_root.exists():
            if runtime_root.is_symlink() or not runtime_root.is_dir():
                raise RuntimeError("configured ComfyUI root is not an owned directory")
            for child in runtime_root.iterdir():
                if child.name in _PRESERVED_RUNTIME_DIRS:
                    if child.is_symlink() or not child.is_dir():
                        raise RuntimeError(f"preserved runtime path is unsafe: {child.name}")
                    continue
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
                else:
                    raise RuntimeError(f"runtime contains an unsupported path: {child.name}")
        else:
            runtime_root.mkdir(mode=0o700)

    def prepare(self, archive_path: Path) -> dict[str, Any]:
        """Install only a caller-provided, hash-pinned ComfyUI source archive."""
        self._remove_marker(self.settings.verification_marker)
        self._remove_marker(self._preparation_receipt)
        self._remove_marker(self._dependency_receipt)
        manifest = self._manifest()
        archive = archive_path.expanduser().resolve(strict=True)
        if not archive.is_file() or archive.is_symlink():
            raise RuntimeError("ComfyUI archive must be a regular local file")
        expected = str(manifest.get("runtime", {}).get("comfyui", {}).get("archive_sha256", ""))
        actual = self._sha256(archive)
        if not expected or actual != expected:
            raise RuntimeError("ComfyUI archive hash does not match the approved release")

        runtime_root = self.settings.comfy_input_dir.parent
        runtime_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{runtime_root.name}.extract-", dir=runtime_root.parent
        ) as temp_dir:
            staging = Path(temp_dir)
            entry_count = self._extract_reviewed_archive(archive, staging)
            self._prepare_runtime_root(runtime_root)
            shutil.copytree(staging, runtime_root, dirs_exist_ok=True, symlinks=False)

        for path in (
            self.settings.comfy_input_dir,
            self.settings.comfy_output_dir,
            self.settings.comfy_model_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        comfyui = manifest.get("runtime", {}).get("comfyui", {})
        self._atomic_json(
            self._preparation_receipt,
            {
                "status": "prepared",
                "manifest_sha256": self.expected_manifest_sha256,
                "archive_sha256": actual,
                "comfyui_commit": comfyui.get("commit"),
                "entries": entry_count,
            },
        )
        return {
            "status": "prepared",
            "archive_sha256": actual,
            "entries": entry_count,
            "runtime_root": str(runtime_root),
            "models_downloaded": 0,
            "inference_enabled": False,
        }

    def install_dependencies(self, *, python_executable: str = "python3") -> dict[str, Any]:
        """Install the reviewed small package overlay; never resolves dependencies or models."""
        self._remove_marker(self.settings.verification_marker)
        self._remove_marker(self._dependency_receipt)
        manifest = self._manifest()
        self._validated_preparation_receipt(manifest)
        overlay, lock_path, expected = self._dependency_spec(manifest)

        runtime_root = self._runtime_root
        runtime_root.mkdir(parents=True, exist_ok=True)
        target = runtime_root / ".packages"
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            raise RuntimeError("dependency target is not an owned directory")
        with tempfile.TemporaryDirectory(prefix=".packages-install-", dir=runtime_root) as temp_dir:
            staging = Path(temp_dir) / "site-packages"
            staging.mkdir(mode=0o700)
            command = [
                python_executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--no-deps",
                "--require-hashes",
                "--only-binary=:all:",
                "--target",
                str(staging),
                "-r",
                str(lock_path),
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=600,
                    env={**os.environ, "PIP_CONFIG_FILE": os.devnull},
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError("approved Python overlay installation failed") from exc

            if self._installed_packages(staging) != expected:
                raise RuntimeError("installed Python overlay does not match the approved package inventory")

            replacement = runtime_root / f".packages-ready-{os.getpid()}"
            if replacement.exists():
                shutil.rmtree(replacement)
            shutil.copytree(staging, replacement, symlinks=False)
            old = runtime_root / f".packages-old-{os.getpid()}"
            if old.exists():
                shutil.rmtree(old)
            try:
                if target.exists():
                    target.replace(old)
                replacement.replace(target)
            except Exception:
                if not target.exists() and old.exists():
                    old.replace(target)
                raise
            finally:
                if replacement.exists():
                    shutil.rmtree(replacement)
                if old.exists():
                    shutil.rmtree(old)

        self._atomic_json(
            self._dependency_receipt,
            {
                "status": "dependencies_installed",
                "manifest_sha256": self.expected_manifest_sha256,
                "lock_sha256": overlay.get("sha256"),
                "packages": expected,
            },
        )
        return {
            "status": "dependencies_installed",
            "package_count": len(expected),
            "target": str(target),
            "models_downloaded": 0,
            "inference_enabled": False,
        }

    def _model_path(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise RuntimeError("model manifest contains an unsafe relative path")
        root = self.settings.comfy_model_dir.resolve()
        result = (root / candidate).resolve()
        try:
            result.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("model path escapes the configured model root") from exc
        return result

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def verify_models(self) -> dict[str, Any]:
        """Hash local model files and commit readiness evidence; never downloads."""
        self._remove_marker(self.settings.verification_marker)
        manifest = self._manifest()
        workflow_hash = self._sha256(self.settings.api_workflow_path)
        expected_workflow_hash = str(manifest.get("workflow", {}).get("api_sha256", ""))
        if workflow_hash != expected_workflow_hash:
            raise RuntimeError("API workflow hash does not match the runtime manifest")

        preparation = self._validated_preparation_receipt(manifest)
        _overlay, _lock_path, expected_packages = self._dependency_spec(manifest)
        dependencies = self._validated_dependency_receipt(manifest, expected_packages)

        verified: dict[str, dict[str, Any]] = {}
        for item in manifest.get("models", []):
            if not isinstance(item, dict):
                raise RuntimeError("runtime manifest contains an invalid model entry")
            relative = str(item.get("relative_path", ""))
            path = self._model_path(relative)
            try:
                info = path.lstat()
            except FileNotFoundError as exc:
                raise RuntimeError(f"required model is missing: {relative}") from exc
            if not stat.S_ISREG(info.st_mode) or path.is_symlink():
                raise RuntimeError(f"required model is not a regular owned file: {relative}")
            expected_size = int(item.get("size_bytes", -1))
            if info.st_size != expected_size:
                raise RuntimeError(f"required model has the wrong size: {relative}")
            digest = self._sha256(path)
            expected_hash = str(item.get("sha256", ""))
            if digest != expected_hash:
                raise RuntimeError(f"required model hash mismatch: {relative}")
            verified[relative] = {"size_bytes": info.st_size, "sha256": digest}

        marker = {
            "status": "verified",
            "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest_sha256": self.expected_manifest_sha256,
            "api_workflow_sha256": workflow_hash,
            "bootstrap": {
                "comfyui_archive_sha256": preparation["archive_sha256"],
                "dependency_lock_sha256": dependencies["lock_sha256"],
                "dependency_packages": expected_packages,
            },
            "models": verified,
        }
        self._atomic_json(self.settings.verification_marker, marker)
        return {
            "status": "verified",
            "model_count": len(verified),
            "total_model_bytes": sum(item["size_bytes"] for item in verified.values()),
            "verification_marker": str(self.settings.verification_marker),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare pinned LTX runtime or verify already-present models; never downloads weights."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--archive", required=True, type=Path)
    dependencies = subparsers.add_parser("install-dependencies")
    dependencies.add_argument("--python", default="python3")
    subparsers.add_parser("verify-models")
    arguments = parser.parse_args()

    settings = WorkerSettings.from_env()
    settings.validate_server()
    bootstrap = RuntimeBootstrap(settings)
    if arguments.action == "prepare":
        report = bootstrap.prepare(arguments.archive)
    elif arguments.action == "install-dependencies":
        report = bootstrap.install_dependencies(python_executable=arguments.python)
    else:
        report = bootstrap.verify_models()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
