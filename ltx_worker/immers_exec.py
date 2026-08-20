"""Remote executor: run ComfyUI jobs on the immers.cloud GPU VM over SSH.

The VM stays shelved (no GPU billing) between runs. ``up()`` wakes it via
the power manager, waits for SSH, makes sure ComfyUI is serving on its
loopback port and arms a hard-stop watchdog. ``down()`` returns the VM to
the shelf. Input/output bytes move with scp.

SSH/scp are invoked through an injectable runner so tests never touch the
network. The default runner is subprocess.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

from ltx_worker.immers_power import ImmersPowerError


class ImmersExecError(RuntimeError):
    pass

def _default_run(argv: list[str], timeout: int = 60) -> tuple[int, str]:
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


@dataclass
class ImmersExec:
    ssh_host: str
    ssh_key: str
    remote_comfy_root: str
    remote_venv: str
    power: object
    run_ssh: Callable[..., tuple[int, str]] = _default_run
    sleep: Callable[[float], None] = None
    watchdog_minutes: int = 35
    comfy_port: int = 8188
    popen: Callable[..., Any] = subprocess.Popen
    probe: Callable[[int], bool] = None  # set in __post_init__
    _ssh_probe_attempts: int = 10
    _ssh_probe_delay: float = 15.0
    _comfy_ready_attempts: int = 10
    _comfy_ready_delay: float = 10.0

    def __post_init__(self):
        if self.sleep is None:
            import time

            self.sleep = time.sleep
        if self.probe is None:
            import socket

            def probe(port: int) -> bool:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=2):
                        return True
                except OSError:
                    return False

            self.probe = probe

    @classmethod
    def from_env(cls, power) -> "ImmersExec":
        missing = [
            name
            for name in ("IMMERS_SSH_HOST", "IMMERS_SSH_KEY", "IMMERS_COMFY_ROOT", "IMMERS_VENV")
            if not os.getenv(name, "").strip()
        ]
        if missing:
            raise ImmersExecError("missing env: " + ", ".join(missing))
        return cls(
            ssh_host=os.environ["IMMERS_SSH_HOST"].strip(),
            ssh_key=os.environ["IMMERS_SSH_KEY"].strip(),
            remote_comfy_root=os.environ["IMMERS_COMFY_ROOT"].strip(),
            remote_venv=os.environ["IMMERS_VENV"].strip(),
            power=power,
            watchdog_minutes=int(os.getenv("IMMERS_WATCHDOG_MINUTES", "35")),
        )

    # -- transport helpers -------------------------------------------------

    def _ssh_argv(self, remote_cmd: str) -> list[str]:
        return [
            "ssh",
            "-i", self.ssh_key,
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=accept-new",
            self.ssh_host,
            remote_cmd,
        ]

    def _ssh(self, remote_cmd: str, timeout: int = 60) -> tuple[int, str]:
        return self.run_ssh(self._ssh_argv(remote_cmd), timeout=timeout)

    def _scp(self, src: str, dst: str) -> None:
        argv = [
            "scp",
            "-i", self.ssh_key,
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15",
            src,
            dst,
        ]
        rc, out = self.run_ssh(argv, timeout=300)
        if rc != 0:
            raise ImmersExecError(f"scp failed ({rc}): {out[:300]}")

    # -- lifecycle ---------------------------------------------------------

    def up(self) -> None:
        # Idempotent wake: if the VM is already ACTIVE (e.g. we woke it
        # manually to verify things before a paid run), skip unshelve —
        # OpenStack answers 409 for unshelve on an ACTIVE server.
        try:
            status, _task = self.power.server_status()
        except Exception:
            status = None  # flaky API: wait_active below is the real gate
        if status != "ACTIVE":
            self.power.unshelve()
        self.power.wait_active()
        self._wait_ssh()
        if not self._comfy_ready():
            self._start_comfy()
            self._wait_comfy()
        self._arm_watchdog()

    def down(self) -> None:
        # Shelving is the money stop — one shot is NOT enough: shelve during
        # spawning gets a 409 (v4: wake-out timeout fired mid-spawn, the
        # single shelve was rejected, the VM stayed ACTIVE and kept billing).
        # Retry through transient rejections, then verify the shelf state.
        last_error: Exception | None = None
        for _ in range(5):
            try:
                self.power.shelve()
                last_error = None
                break
            except ImmersPowerError as exc:
                last_error = exc
                try:
                    status, _task = self.power.server_status()
                    if status == "SHELVED_OFFLOADED":
                        return
                except Exception:
                    pass
                self.sleep(30)
        if last_error is not None:
            raise last_error
        self.power.wait_shelved()

    def _wait_ssh(self) -> None:
        for _ in range(self._ssh_probe_attempts):
            rc, _ = self._ssh("echo ssh-ok", timeout=30)
            if rc == 0:
                return
            self.sleep(self._ssh_probe_delay)
        raise ImmersExecError("VM is not reachable over SSH after wake-up")

    def _comfy_ready(self) -> bool:
        rc, _ = self._ssh(
            f"curl -s -o /dev/null --max-time 5 http://127.0.0.1:{self.comfy_port}/system_stats",
            timeout=30,
        )
        return rc == 0

    def _start_comfy(self) -> None:
        log = f"{self.remote_comfy_root}/../comfy-remote.log"
        # nohup + & alone is NOT enough over ssh: a backgrounded process that
        # inherits the session stdin keeps the ssh channel open forever.
        # And `cd X && nohup ... &` makes the WHOLE chain the background job,
        # whose subshell inherits the channel pipes — ssh hangs regardless of
        # redirects. `cd; nohup setsid ... < /dev/null &` is the verified
        # working form (proved live on the VM before the paid v3 run).
        rc, out = self._ssh(
            f"cd {self.remote_comfy_root}; "
            f"nohup setsid {self.remote_venv}/bin/python main.py "
            f"--disable-all-custom-nodes --lowvram "
            f"> {log} 2>&1 < /dev/null & echo started",
            timeout=30,
        )
        if rc != 0:
            raise ImmersExecError(f"failed to start ComfyUI: {out[:300]}")

    def _wait_comfy(self) -> None:
        for _ in range(self._comfy_ready_attempts):
            if self._comfy_ready():
                return
            self.sleep(self._comfy_ready_delay)
        raise ImmersExecError("ComfyUI did not become ready on the VM")

    def _arm_watchdog(self) -> None:
        # Failure here is non-fatal (e.g. unit already armed); the shelve path
        # is the primary stop, the watchdog is the belt-and-suspenders.
        self._ssh(
            "sudo systemd-run --unit=ltx23-hardstop "
            f"--on-active={self.watchdog_minutes}min --description=hard-stop "
            "systemctl poweroff 2>/dev/null || true",
            timeout=30,
        )

    # -- SSH tunnel ---------------------------------------------------------

    def open_tunnel(self, local_port: int):
        argv = [
            "ssh", "-N",
            "-L", f"127.0.0.1:{local_port}:127.0.0.1:{self.comfy_port}",
            "-i", self.ssh_key,
            "-o", "BatchMode=yes",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            self.ssh_host,
        ]
        proc = self.popen(argv)
        for _ in range(10):
            if self.probe(local_port):
                return _SshTunnel(proc)
            if proc.poll() is not None:
                raise ImmersExecError("SSH tunnel exited before becoming ready")
            self.sleep(1.0)
        proc.terminate()
        raise ImmersExecError("SSH tunnel did not become ready")

    # -- file transfer ------------------------------------------------------

    def push_input(self, local_path: str, remote_name: str) -> None:
        # scp cannot create directories; the runner stages into a job
        # subfolder (ltx23-worker/), so ensure it exists on the VM first.
        self._mkdir_remote(f"{self.remote_comfy_root}/input", remote_name)
        remote = f"{self.ssh_host}:{self.remote_comfy_root}/input/{remote_name}"
        self._scp(local_path, remote)

    def pull_output(self, remote_name: str, local_path: str) -> None:
        self._mkdir_remote(f"{self.remote_comfy_root}/output", remote_name)
        remote = f"{self.ssh_host}:{self.remote_comfy_root}/output/{remote_name}"
        self._scp(remote, local_path)

    def _mkdir_remote(self, base: str, remote_name: str) -> None:
        sub = remote_name.rsplit("/", 1)[0] if "/" in remote_name else ""
        self._ssh(f"mkdir -p {shlex.quote(base + ('/' + sub if sub else ''))}", timeout=30)


class _SshTunnel:
    def __init__(self, proc):
        self._proc = proc

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
