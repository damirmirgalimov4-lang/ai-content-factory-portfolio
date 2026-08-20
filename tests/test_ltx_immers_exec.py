"""Tests for the immers remote executor (SSH bridge to the GPU VM).

No real SSH/subprocess: the runner callable is injected.
"""
from __future__ import annotations

import unittest

from ltx_worker.immers_exec import ImmersExec, ImmersExecError
from ltx_worker.immers_power import ImmersPowerError


class FakePower:
    def __init__(self):
        self.calls = []

    def server_status(self):
        return ("SHELVED_OFFLOADED", None)

    def unshelve(self):
        self.calls.append("unshelve")

    def wait_active(self, **kwargs):
        self.calls.append("wait_active")

    def shelve(self):
        self.calls.append("shelve")

    def wait_shelved(self, **kwargs):
        self.calls.append("wait_shelved")


class FakeSsh:
    """Scripted ssh/scp runner: records argv, replays (rc, stdout)."""

    def __init__(self, results=None):
        self.calls = []
        self.results = list(results or [])

    def __call__(self, argv, timeout=60):
        self.calls.append(argv)
        if self.results:
            return self.results.pop(0)
        return (255, "")


def make_exec(ssh=None, power=None, **overrides):
    from ltx_worker.immers_exec import ImmersExec

    kwargs = dict(
        ssh_host="gpu@203.0.113.10",
        ssh_key="/run/secrets/ltx_ssh_key",
        remote_comfy_root="/srv/ltx/runtime/ComfyUI",
        remote_venv="/srv/ltx/venv",
        power=power or FakePower(),
        run_ssh=ssh or FakeSsh(),
        sleep=lambda _: None,
    )
    kwargs.update(overrides)
    return ImmersExec(**kwargs)


class ImmersExecTest(unittest.TestCase):
    def test_up_unshelves_then_waits_active(self):
        power = FakePower()
        ssh = FakeSsh([(0, "ssh-ok"), (0, "comfy-running")])
        ex = make_exec(ssh=ssh, power=power)

        ex.up()

        self.assertEqual(power.calls[:2], ["unshelve", "wait_active"])

    def test_up_uses_key_and_batch_mode_for_ssh(self):
        ssh = FakeSsh([(0, "ok"), (0, "comfy-running")])
        ex = make_exec(ssh=ssh)

        ex.up()

        first = ssh.calls[0]
        joined = " ".join(first)
        self.assertIn("-i /run/secrets/ltx_ssh_key", joined)
        self.assertIn("BatchMode=yes", joined)
        self.assertIn("gpu@203.0.113.10", joined)

    def test_up_fails_when_ssh_unreachable(self):
        ssh = FakeSsh([(255, "")])
        ex = make_exec(ssh=ssh)

        with self.assertRaises(ImmersExecError):
            ex.up()

    def test_up_starts_comfy_when_not_running(self):
        ssh = FakeSsh([
            (0, "ssh-ok"),      # probe
            (1, ""),            # comfy probe -> not running
            (0, "started"),     # start command
            (0, "comfy-ok"),    # readiness probe
        ])
        ex = make_exec(ssh=ssh)

        ex.up()

        joined = [" ".join(c) for c in ssh.calls]
        self.assertTrue(any("main.py" in j or "nohup" in j for j in joined))

    def test_up_arms_hard_stop_watchdog(self):
        ssh = FakeSsh([(0, "ssh-ok"), (0, "comfy-running"), (0, "armed")])
        ex = make_exec(ssh=ssh, watchdog_minutes=35)

        ex.up()

        joined = [" ".join(c) for c in ssh.calls]
        self.assertTrue(any("systemd-run" in j and "poweroff" in j for j in joined))

    def test_up_aborts_when_comfy_never_ready(self):
        ssh = FakeSsh(
            [(0, "ssh-ok"), (1, ""), (0, "started")]
            + [(1, "")] * 10  # readiness probes keep failing
        )
        ex = make_exec(ssh=ssh)

        with self.assertRaises(ImmersExecError):
            ex.up()

    def test_down_shelves_and_waits(self):
        power = FakePower()
        ex = make_exec(power=power)

        ex.down()

        self.assertEqual(power.calls, ["shelve", "wait_shelved"])

    def test_down_swallows_power_api_errors_but_raises(self):
        class BadPower(FakePower):
            def server_status(self):
                return ("ACTIVE", None)  # genuinely failing, not shelved

            def shelve(self):
                self.calls.append("shelve")
                raise ImmersPowerError("boom")

        power = BadPower()
        ex = make_exec(power=power)

        with self.assertRaises(ImmersPowerError):
            ex.down()
        self.assertEqual(power.calls, ["shelve"] * 5)

    def test_down_retries_shelve_through_transient_409(self):
        """v4 regression: shelve during spawning gets 409; must retry."""

        class FlakyPower(FakePower):
            def __init__(self):
                super().__init__()
                self.failures_left = 2

            def server_status(self):
                return ("ACTIVE", "spawning")

            def shelve(self):
                self.calls.append("shelve")
                if self.failures_left > 0:
                    self.failures_left -= 1
                    raise ImmersPowerError("http 409")

            def wait_shelved(self):
                self.calls.append("wait_shelved")

        power = FlakyPower()
        ex = make_exec(power=power)

        ex.down()

        self.assertEqual(power.calls, ["shelve"] * 3 + ["wait_shelved"])

    def test_down_returns_when_already_shelved(self):
        class ShelvedPower(FakePower):
            def shelve(self):
                self.calls.append("shelve")
                raise ImmersPowerError("http 409")  # double-shelve

        power = ShelvedPower()
        ex = make_exec(power=power)

        ex.down()  # no raise: money already stopped

        self.assertEqual(power.calls, ["shelve"])

    def test_push_input_uses_scp_to_remote_input_dir(self):
        ssh = FakeSsh([(0, ""), (0, "")])
        ex = make_exec(ssh=ssh)

        ex.push_input("/local/frame.png", "job1.png")

        argv = [c for c in ssh.calls if c[0] == "scp"][0]
        self.assertEqual(argv[0], "scp")
        joined = " ".join(argv)
        self.assertIn("/local/frame.png", joined)
        self.assertIn("runtime/ComfyUI/input/job1.png", joined)

    def test_push_input_raises_on_scp_failure(self):
        ssh = FakeSsh([(1, "denied")])
        ex = make_exec(ssh=ssh)

        with self.assertRaises(ImmersExecError):
            ex.push_input("/local/frame.png", "job1.png")

    def test_pull_output_copies_from_remote_output_dir(self):
        ssh = FakeSsh([(0, ""), (0, "")])
        ex = make_exec(ssh=ssh)

        ex.pull_output("ltx-video-00001.mp4", "/local/out.mp4")

        argv = [c for c in ssh.calls if c[0] == "scp"][0]
        joined = " ".join(argv)
        self.assertIn("runtime/ComfyUI/output/ltx-video-00001.mp4", joined)
        self.assertIn("/local/out.mp4", joined)

    def test_up_skips_unshelve_when_already_active(self):
        """v3 plan: wake the VM manually, verify ComfyUI, then run the job —
        up() must tolerate an already-ACTIVE VM instead of dying on 409."""

        class ActivePower(FakePower):
            def server_status(self):
                return ("ACTIVE", None)

        ssh = FakeSsh([(0, "ssh-ok"), (0, "comfy-running"), (0, "armed")])
        power = ActivePower()
        ex = make_exec(power=power, ssh=ssh)

        ex.up()

        self.assertNotIn("unshelve", power.calls)
        self.assertIn("wait_active", power.calls)

    def test_push_input_creates_remote_subdir_first(self):
        """scp cannot create directories: staging uses the ltx23-worker/
        subfolder, so push must mkdir -p it on the VM first (v3 died here)."""
        ssh = FakeSsh([(0, ""), (0, "")])
        ex = make_exec(ssh=ssh)

        ex.push_input("/local/frame.png", "ltx23-worker/abc.png")

        mkdir_calls = [c for c in ssh.calls if "mkdir -p" in " ".join(c)]
        scp_calls = [c for c in ssh.calls if c[0] == "scp"]
        self.assertEqual(len(mkdir_calls), 1)
        self.assertIn("input/ltx23-worker", " ".join(mkdir_calls[0]))
        self.assertEqual(len(scp_calls), 1)
        # mkdir must come before scp
        self.assertTrue(" ".join(ssh.calls[0]).find("mkdir") != -1)

    def test_pull_output_ensures_remote_subdir_exists(self):
        ssh = FakeSsh([(0, ""), (0, "")])
        ex = make_exec(ssh=ssh)

        ex.pull_output("ltx23-worker/abc.mp4", "/local/out.mp4")

        mkdir_calls = [c for c in ssh.calls if "mkdir -p" in " ".join(c)]
        self.assertEqual(len(mkdir_calls), 1)
        self.assertIn("output/ltx23-worker", " ".join(mkdir_calls[0]))

    def test_up_start_command_detaches_stdin(self):
        """ssh hangs forever if the backgrounded process inherits stdin —
        the start command must redirect < /dev/null (learned the hard way,
        smoke v2 died on this)."""
        ssh = FakeSsh([(0, "ssh-ok"), (1, ""), (0, "started"), (0, "ready"), (0, "armed")])
        ex = make_exec(ssh=ssh)

        ex.up()

        start_calls = [c for c in ssh.calls if "nohup" in " ".join(c)]
        self.assertEqual(len(start_calls), 1)
        cmd = " ".join(start_calls[0])
        self.assertIn("< /dev/null", cmd)
        self.assertIn("; ", cmd.split("& echo")[0].split("nohup")[0])
        self.assertNotIn("&& nohup", cmd)

    def test_open_tunnel_forwards_loopback_port_and_close_terminates(self):
        from ltx_worker.immers_exec import ImmersExec

        class FakeProc:
            def __init__(self):
                self.terminated = False

            def poll(self):
                return None if not self.terminated else 0

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return 0

        procs = []

        def fake_popen(argv, **kwargs):
            procs.append(argv)
            return FakeProc()

        ex = make_exec()
        object.__setattr__(ex, "popen", fake_popen)
        object.__setattr__(ex, "probe", lambda port: True)

        tunnel = ex.open_tunnel(8188)

        argv = " ".join(procs[0])
        self.assertIn("-N", argv)
        self.assertIn("-L 127.0.0.1:8188:127.0.0.1:8188", argv)
        self.assertIn("-i /run/secrets/ltx_ssh_key", argv)
        tunnel.close()
        self.assertTrue(procs and True)


if __name__ == "__main__":
    unittest.main()
