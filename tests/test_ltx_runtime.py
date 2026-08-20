from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
import tempfile
import threading
import unittest
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

from ltx_worker.bootstrap import RuntimeBootstrap
from ltx_worker.config import WorkerSettings
from ltx_worker.runner import ComfyUiRunner, EXPECTED_API_WORKFLOW_SHA256, EXPECTED_MANIFEST_SHA256
from ltx_worker.service import JobRequest


PNG = b"\x89PNG\r\n\x1a\n" + b"runtime-test-image"
MP4 = b"\x00\x00\x00\x18ftypisom" + b"runtime-test-video"
ASSETS = Path(__file__).resolve().parents[1] / "ltx_worker" / "assets"


def fake_overlay_install(settings: WorkerSettings, calls: list[list[str]] | None = None):
    def fake_run(command, **kwargs):
        if calls is not None:
            calls.append([str(part) for part in command])
        target = Path(command[command.index("--target") + 1])
        manifest = json.loads(settings.runtime_manifest_path.read_text(encoding="utf-8"))
        for name, version in manifest["runtime"]["python_overlay"]["packages"].items():
            dist_info = target / f"{name.replace('-', '_')}-{version}.dist-info"
            dist_info.mkdir(parents=True, exist_ok=True)
            metadata_name = name.replace("-", "_")
            (dist_info / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: {metadata_name}\nVersion: {version}\n",
                encoding="utf-8",
            )
        return MagicMock(returncode=0, stdout="installed")

    return fake_run


def prepared_bootstrap(
    root: Path, *, models: list[dict[str, object]] | None = None
) -> tuple[WorkerSettings, RuntimeBootstrap, Path, str]:
    settings = WorkerSettings.for_test(root)
    manifest = json.loads(settings.runtime_manifest_path.read_text(encoding="utf-8"))
    manifest["models"] = [] if models is None else models
    archive = root / "comfyui.tar.gz"
    payload_bytes = b"print('pinned runtime')\n"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("ComfyUI-pinned/main.py")
        info.size = len(payload_bytes)
        bundle.addfile(info, io.BytesIO(payload_bytes))
    manifest["runtime"]["comfyui"]["archive_sha256"] = hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    manifest_hash = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["manifest_sha256"] = manifest_hash
    manifest_path = root / "runtime-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    settings = WorkerSettings.for_test(root, runtime_manifest_path=manifest_path)
    bootstrap = RuntimeBootstrap(settings, expected_manifest_sha256=manifest_hash)
    return settings, bootstrap, archive, manifest_hash


def payload() -> dict[str, object]:
    return {
        "job_id": "cf-runtime-v01",
        "model": "ltx-2.3",
        "workflow": "distilled",
        "prompt": "A paper plane moves once with quiet room tone.",
        "seed": 42,
        "duration_seconds": 5,
        "aspect_ratio": "16:9",
        "width": 1024,
        "height": 576,
        "fps": 24,
        "audio": True,
        "image_base64": base64.b64encode(PNG).decode("ascii"),
    }


class _ComfyState:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.posts: list[dict[str, object]] = []
        self.prompt_id = ""
        self.filename = ""
        self.subfolder = ""


class _ComfyHandler(BaseHTTPRequestHandler):
    server: "_ComfyServer"

    def log_message(self, format: str, *args) -> None:
        return None

    def _json(self, status: int, value: object) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        self.assert_path("/prompt")
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        self.server.state.posts.append(body)
        self.server.state.prompt_id = str(body["prompt_id"])
        prefix = str(body["prompt"]["75"]["inputs"]["filename_prefix"])
        prefix_path = Path(prefix)
        self.server.state.subfolder = str(prefix_path.parent)
        self.server.state.filename = prefix_path.name + "_00001_.mp4"
        output = self.server.state.output_root / self.server.state.subfolder / self.server.state.filename
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(MP4)
        self._json(200, {"prompt_id": body["prompt_id"], "number": 1, "node_errors": {}})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == f"/history/{self.server.state.prompt_id}":
            self._json(
                200,
                {
                    self.server.state.prompt_id: {
                        "status": {"status_str": "success", "completed": True, "messages": []},
                        "outputs": {
                            "75": {
                                "images": [
                                    {
                                        "filename": self.server.state.filename,
                                        "subfolder": self.server.state.subfolder,
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                    }
                },
            )
            return
        self._json(404, {"error": "not found"})

    def assert_path(self, expected: str) -> None:
        if self.path != expected:
            raise AssertionError(f"unexpected path {self.path!r}")


class _ComfyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, state: _ComfyState):
        self.state = state
        super().__init__(address, _ComfyHandler)


class _ProbeRunner(ComfyUiRunner):
    def _probe(self, output_path: Path) -> dict[str, object]:
        self.probed = output_path.read_bytes()
        return {
            "duration_seconds": 5.0,
            "width": 1024,
            "height": 576,
            "fps": 24.0,
            "has_video": True,
            "has_audio": True,
        }


class RuntimeAssetsTest(unittest.TestCase):
    def test_pinned_manifest_and_full_api_workflow_are_internally_consistent(self) -> None:
        manifest_path = ASSETS / "runtime-manifest.json"
        workflow_path = ASSETS / "ltx23-i2v-api.json"
        source_path = ASSETS / "video_ltx2_3_i2v.json"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        source = json.loads(source_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["workflow"]["source_sha256"], "91dd8e44926fd37f6d9307789484370fa333582b14e53ed771d63ed805379ee4")
        self.assertEqual(manifest["workflow"]["api_sha256"], EXPECTED_API_WORKFLOW_SHA256)
        self.assertEqual(manifest["manifest_sha256"], EXPECTED_MANIFEST_SHA256)
        self.assertEqual(len(workflow), 51)
        self.assertEqual(len(source["definitions"]["subgraphs"][0]["nodes"]), 50)
        self.assertEqual(
            sum(1 for node in workflow.values() if node["class_type"] == "SamplerCustomAdvanced"),
            2,
        )
        self.assertIn("LTXVEmptyLatentAudio", {node["class_type"] for node in workflow.values()})
        self.assertEqual(workflow["75"]["class_type"], "SaveVideo")
        self.assertEqual(len(manifest["models"]), 5)
        self.assertEqual(sum(item["size_bytes"] for item in manifest["models"]), 42_958_104_950)
        self.assertTrue(all("revision" in item and len(item["sha256"]) == 64 for item in manifest["models"]))
        self.assertEqual(manifest["policy"]["paid_retries"], 0)
        self.assertEqual(manifest["policy"]["concurrency"], 1)

        dependency_lock = ASSETS / manifest["runtime"]["python_overlay"]["file"]
        self.assertEqual(
            hashlib.sha256(dependency_lock.read_bytes()).hexdigest(),
            manifest["runtime"]["python_overlay"]["sha256"],
        )
        requirements = [
            line.strip()
            for line in dependency_lock.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(len(requirements), 9)
        for requirement in requirements:
            self.assertIn(" @ https://files.pythonhosted.org/", requirement)
            self.assertRegex(requirement, r" --hash=sha256:[0-9a-f]{64}$")

    def test_math_expression_dependency_uses_security_fixed_simpleeval(self) -> None:
        manifest = json.loads((ASSETS / "runtime-manifest.json").read_text(encoding="utf-8"))
        dependency_lock = ASSETS / manifest["runtime"]["python_overlay"]["file"]
        lock_text = dependency_lock.read_text(encoding="utf-8")

        self.assertEqual(
            manifest["runtime"]["python_overlay"]["packages"]["simpleeval"],
            "1.0.5",
        )
        self.assertIn("simpleeval-1.0.5-py3-none-any.whl", lock_text)
        self.assertNotIn("simpleeval-1.0.3", lock_text)

    def test_job_request_has_stable_external_prompt_id(self) -> None:
        first = JobRequest.from_payload(payload())
        second = JobRequest.from_payload(payload())
        self.assertEqual(first.external_prompt_id, second.external_prompt_id)
        self.assertEqual(str(uuid.UUID(first.external_prompt_id)), first.external_prompt_id)
        self.assertEqual(first.stored_dict()["external_prompt_id"], first.external_prompt_id)

    def test_comfy_backend_must_be_loopback_http(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            good = WorkerSettings.for_test(root)
            good.validate_server()
            for url in ("https://127.0.0.1:8188", "http://example.com:8188", "http://0.0.0.0:8188"):
                bad = WorkerSettings.for_test(root, comfy_base_url=url)
                with self.subTest(url=url), self.assertRaises(ValueError):
                    bad.validate_server()


    def test_entrypoint_wires_the_reviewed_comfy_runner(self) -> None:
        from ltx_worker import __main__ as entrypoint

        settings = WorkerSettings.for_test(Path("/tmp/ltx-entrypoint-test"))
        server = MagicMock()
        server.serve_forever.side_effect = KeyboardInterrupt
        with (
            patch.object(entrypoint.WorkerSettings, "from_env", return_value=settings),
            patch.object(entrypoint, "JobStore") as store_type,
            patch.object(entrypoint, "ComfyUiRunner") as runner_type,
            patch.object(entrypoint, "VideoWorkerService") as service_type,
            patch.object(entrypoint, "create_server", return_value=server),
        ):
            service_type.return_value = MagicMock()
            with self.assertRaises(KeyboardInterrupt):
                entrypoint.main()

        runner_type.assert_called_once_with(settings, remote=None)
        service_type.assert_called_once_with(settings, store_type.return_value, runner_type.return_value)
        server.server_close.assert_called_once_with()
        service_type.return_value.close.assert_called_once_with()

    def test_entrypoint_wires_immers_remote_when_executor_is_immers(self) -> None:
        import os
        from ltx_worker import __main__ as entrypoint
        from ltx_worker.immers_exec import ImmersExec

        settings = WorkerSettings.for_test(Path("/tmp/ltx-entrypoint-test"))
        server = MagicMock()
        server.serve_forever.side_effect = KeyboardInterrupt
        env = {
            "LTX_EXECUTOR": "immers",
            "IMMERS_AUTH_URL": "https://api.immers.cloud:5000/v3",
            "IMMERS_COMPUTE_URL": "https://api.immers.cloud:8774/v2.1",
            "IMMERS_USERNAME": "Hook",
            "IMMERS_PASSWORD": "unit-test-only",
            "IMMERS_SERVER_ID": "srv-1",
            "IMMERS_SSH_HOST": "ubuntu@127.0.0.1",
            "IMMERS_SSH_KEY": "/run/secrets/ltx_ssh_key",
            "IMMERS_COMFY_ROOT": "/srv/ltx/runtime/ComfyUI",
            "IMMERS_VENV": "/srv/ltx/venv",
        }
        with (
            patch.dict(os.environ, env),
            patch.object(entrypoint.WorkerSettings, "from_env", return_value=settings),
            patch.object(entrypoint, "JobStore"),
            patch.object(entrypoint, "ComfyUiRunner") as runner_type,
            patch.object(entrypoint, "VideoWorkerService") as service_type,
            patch.object(entrypoint, "create_server", return_value=server),
        ):
            service_type.return_value = MagicMock()
            with self.assertRaises(KeyboardInterrupt):
                entrypoint.main()

        remote = runner_type.call_args.kwargs["remote"]
        self.assertIsInstance(remote, ImmersExec)
        self.assertEqual(remote.ssh_host, "ubuntu@127.0.0.1")

    def test_bootstrap_verifies_local_archive_and_models_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings, bootstrap, archive, manifest_hash = prepared_bootstrap(root)

            report = bootstrap.prepare(archive)
            self.assertEqual(report["status"], "prepared")
            self.assertEqual(
                (settings.comfy_input_dir.parent / "main.py").read_bytes(),
                b"print('pinned runtime')\n",
            )
            self.assertFalse(settings.verification_marker.exists())

            with patch(
                "ltx_worker.bootstrap.subprocess.run",
                side_effect=fake_overlay_install(settings),
            ):
                dependencies = bootstrap.install_dependencies(python_executable="python3.12")
            self.assertEqual(dependencies["status"], "dependencies_installed")
            self.assertFalse(settings.verification_marker.exists())

            verified = bootstrap.verify_models()
            self.assertEqual(verified["status"], "verified")
            marker = json.loads(settings.verification_marker.read_text(encoding="utf-8"))
            self.assertEqual(marker["manifest_sha256"], manifest_hash)
            self.assertEqual(
                marker["bootstrap"]["dependency_lock_sha256"],
                json.loads(settings.runtime_manifest_path.read_text(encoding="utf-8"))["runtime"][
                    "python_overlay"
                ]["sha256"],
            )

    def test_model_verification_requires_completed_bootstrap_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings, bootstrap, archive, _manifest_hash = prepared_bootstrap(root)

            with self.assertRaisesRegex(RuntimeError, "preparation receipt"):
                bootstrap.verify_models()
            self.assertFalse(settings.verification_marker.exists())

            bootstrap.prepare(archive)
            with self.assertRaisesRegex(RuntimeError, "dependency receipt"):
                bootstrap.verify_models()
            self.assertFalse(settings.verification_marker.exists())

    def test_dependency_install_uses_hash_locked_overlay_without_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings, bootstrap, archive, _manifest_hash = prepared_bootstrap(root)
            bootstrap.prepare(archive)
            calls: list[list[str]] = []

            with patch(
                "ltx_worker.bootstrap.subprocess.run",
                side_effect=fake_overlay_install(settings, calls),
            ):
                report = bootstrap.install_dependencies(python_executable="python3.12")

            self.assertEqual(report["status"], "dependencies_installed")
            self.assertEqual(report["package_count"], 9)
            self.assertEqual(report["models_downloaded"], 0)
            self.assertFalse(report["inference_enabled"])
            self.assertEqual(len(calls), 1)
            command = calls[0]
            self.assertEqual(command[:4], ["python3.12", "-m", "pip", "install"])
            self.assertIn("--no-deps", command)
            self.assertIn("--require-hashes", command)
            self.assertNotIn("huggingface-cli", command)
            self.assertTrue((settings.comfy_input_dir.parent / ".packages").is_dir())

    def test_model_verification_fails_closed_without_creating_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_manifest = json.loads((ASSETS / "runtime-manifest.json").read_text(encoding="utf-8"))
            item = dict(source_manifest["models"][0])
            item["relative_path"] = "checkpoints/test-model.safetensors"
            item["size_bytes"] = 4
            item["sha256"] = hashlib.sha256(b"good").hexdigest()
            settings, bootstrap, archive, _manifest_hash = prepared_bootstrap(root, models=[item])
            bootstrap.prepare(archive)
            with patch(
                "ltx_worker.bootstrap.subprocess.run",
                side_effect=fake_overlay_install(settings),
            ):
                bootstrap.install_dependencies(python_executable="python3.12")

            model = settings.comfy_model_dir / item["relative_path"]
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes(b"bad!")

            with self.assertRaisesRegex(RuntimeError, "model hash mismatch"):
                bootstrap.verify_models()

            self.assertFalse(settings.verification_marker.exists())


class ComfyUiRunnerTest(unittest.TestCase):
    def test_readiness_rejects_marker_without_bootstrap_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = WorkerSettings.for_test(root)
            manifest = json.loads(settings.runtime_manifest_path.read_text(encoding="utf-8"))
            verified_models: dict[str, dict[str, object]] = {}
            for item in manifest["models"]:
                path = settings.comfy_model_dir / item["relative_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as handle:
                    handle.truncate(item["size_bytes"])
                verified_models[item["relative_path"]] = {
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                }
            settings.verification_marker.write_text(
                json.dumps(
                    {
                        "status": "verified",
                        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                        "api_workflow_sha256": EXPECTED_API_WORKFLOW_SHA256,
                        "models": verified_models,
                    }
                ),
                encoding="utf-8",
            )
            runner = ComfyUiRunner(settings)

            with patch.object(runner, "_get_json", return_value={"devices": [{"type": "cpu"}]}):
                ready, reason = runner.readiness()

            self.assertFalse(ready)
            self.assertIn("bootstrap", reason)

    def test_builds_fixed_graph_and_submits_exactly_once_with_reserved_prompt_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "comfy" / "input"
            output_root = root / "comfy" / "output"
            state = _ComfyState(output_root)
            server = _ComfyServer(("127.0.0.1", 0), state)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                settings = WorkerSettings.for_test(
                    root,
                    comfy_base_url=f"http://{host}:{port}",
                    comfy_input_dir=input_root,
                    comfy_output_dir=output_root,
                )
                request = JobRequest.from_payload(payload())
                staged_input = root / "worker-input.png"
                staged_input.write_bytes(PNG)
                result_path = root / "result.mp4"
                runner = _ProbeRunner(settings)

                metrics = runner.run(request, staged_input, result_path)

                self.assertEqual(len(state.posts), 1)
                submitted = state.posts[0]
                self.assertEqual(submitted["prompt_id"], request.external_prompt_id)
                self.assertEqual(submitted["prompt"]["320:319"]["inputs"]["value"], request.prompt)
                self.assertEqual(submitted["prompt"]["320:277"]["inputs"]["noise_seed"], 42)
                self.assertEqual(submitted["prompt"]["320:312"]["inputs"]["value"], 1024)
                self.assertEqual(submitted["prompt"]["320:299"]["inputs"]["value"], 576)
                self.assertEqual(submitted["prompt"]["320:300"]["inputs"]["value"], 24)
                self.assertEqual(submitted["prompt"]["320:301"]["inputs"]["value"], 5)
                self.assertEqual(submitted["prompt"]["75"]["inputs"]["format"], "mp4")
                image_name = submitted["prompt"]["269"]["inputs"]["image"]
                self.assertNotIn("..", image_name)
                self.assertEqual((input_root / image_name).read_bytes(), PNG)
                self.assertEqual(result_path.read_bytes(), MP4)
                self.assertEqual(runner.probed, MP4)
                self.assertEqual(metrics["fps"], 24.0)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
