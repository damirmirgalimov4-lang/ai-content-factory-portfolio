"""Remote-mode runner tests: factory worker driving ComfyUI on the immers VM.

The stub ComfyUI from test_ltx_runtime plays the VM's server; its output
root simulates the VM disk. FakeRemote stands in for ImmersExec.
"""
from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from ltx_worker.config import WorkerSettings
from ltx_worker.runner import ComfyUiRunner
from ltx_worker.service import JobRequest

from test_ltx_runtime import _ComfyServer, _ComfyState, _ProbeRunner, MP4, PNG, payload


class FakeTunnel:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeRemote:
    """Records lifecycle; pull_output copies from the simulated VM disk."""

    def __init__(self, vm_output_root: Path | None = None):
        self.calls: list = []
        self.vm_output_root = vm_output_root
        self.tunnel = FakeTunnel()

    def up(self):
        self.calls.append("up")

    def down(self):
        self.calls.append("down")

    def open_tunnel(self, local_port: int):
        self.calls.append(("tunnel", local_port))
        return self.tunnel

    def push_input(self, local_path: str, remote_name: str):
        self.calls.append(("push", remote_name, Path(local_path).read_bytes()))

    def pull_output(self, remote_name: str, local_path: str):
        self.calls.append(("pull", remote_name))
        source = self.vm_output_root / remote_name
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


class RemoteRunnerTest(unittest.TestCase):
    def _make_settings(self, root: Path, port: int) -> WorkerSettings:
        return WorkerSettings.for_test(
            root,
            comfy_base_url=f"http://127.0.0.1:{port}",
            comfy_input_dir=root / "comfy" / "input",
            comfy_output_dir=root / "comfy" / "output",
        )

    def test_remote_run_pushes_input_pulls_output_and_shelves(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vm_output = root / "vm-disk" / "output"
            state = _ComfyState(vm_output)
            server = _ComfyServer(("127.0.0.1", 0), state)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                settings = self._make_settings(root, server.server_address[1])
                remote = FakeRemote(vm_output_root=vm_output)
                runner = _ProbeRunner(settings, remote=remote)
                staged_input = root / "worker-input.png"
                staged_input.write_bytes(PNG)

                metrics = runner.run(
                    JobRequest.from_payload(payload()), staged_input, root / "result.mp4"
                )

                self.assertEqual((root / "result.mp4").read_bytes(), MP4)
                self.assertEqual(metrics["fps"], 24.0)
                verbs = [c if isinstance(c, str) else c[0] for c in remote.calls]
                self.assertEqual(verbs[0], "up")
                self.assertEqual(verbs[-1], "down")
                self.assertIn("tunnel", verbs)
                pushes = [c for c in remote.calls if isinstance(c, tuple) and c[0] == "push"]
                pulls = [c for c in remote.calls if isinstance(c, tuple) and c[0] == "pull"]
                self.assertEqual(len(pushes), 1)
                self.assertEqual(len(pulls), 1)
                self.assertEqual(pushes[0][2], PNG)
                self.assertTrue(remote.tunnel.closed)
                self.assertEqual(len(state.posts), 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_remote_run_failure_still_shelves(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            import socket

            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            dead_port = sock.getsockname()[1]
            sock.close()
            settings = self._make_settings(root, dead_port)
            remote = FakeRemote()
            runner = ComfyUiRunner(settings, remote=remote)
            staged_input = root / "worker-input.png"
            staged_input.write_bytes(PNG)

            with self.assertRaises(Exception):
                runner.run(JobRequest.from_payload(payload()), staged_input, root / "result.mp4")

            self.assertIn("down", remote.calls)

    def test_remote_readiness_does_not_wake_the_vm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._make_settings(root, 1)
            remote = FakeRemote()
            runner = ComfyUiRunner(settings, remote=remote)

            ready, reason = runner.readiness()

            self.assertTrue(ready)
            self.assertEqual(remote.calls, [])

    def test_up_failure_still_shelves(self):
        """Regression: if up() itself fails (e.g. API timeout), down() must run."""

        class UpFails(FakeRemote):
            def up(self):
                self.calls.append("up")
                raise RuntimeError("immers API timeout")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._make_settings(root, 1)
            remote = UpFails()
            runner = ComfyUiRunner(settings, remote=remote)
            staged_input = root / "worker-input.png"
            staged_input.write_bytes(PNG)

            with self.assertRaisesRegex(RuntimeError, "immers API timeout"):
                runner.run(JobRequest.from_payload(payload()), staged_input, root / "result.mp4")

            self.assertEqual(remote.calls, ["up", "down"])

    def test_down_failure_is_not_masked_when_work_succeeded(self):
        """If the job worked but shelving failed, that must surface loudly."""

        class DownFails(FakeRemote):
            def down(self):
                self.calls.append("down")
                raise RuntimeError("shelve refused")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vm_output = root / "vm-disk" / "output"
            state = _ComfyState(vm_output)
            server = _ComfyServer(("127.0.0.1", 0), state)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                settings = self._make_settings(root, server.server_address[1])
                remote = DownFails(vm_output_root=vm_output)
                runner = _ProbeRunner(settings, remote=remote)
                staged_input = root / "worker-input.png"
                staged_input.write_bytes(PNG)

                with self.assertRaisesRegex(RuntimeError, "shelve refused"):
                    runner.run(JobRequest.from_payload(payload()), staged_input, root / "result.mp4")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
