"""Tests for the immers.cloud power manager (no real HTTP, no paid actions)."""
from __future__ import annotations

import os
import unittest

from ltx_worker.immers_power import ImmersPower, ImmersPowerError


class FakeHttp:
    """Scripted HTTP stub: records calls, replays queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, payload, headers=None, token=None):
        self.calls.append(("POST", url, payload, token))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body, resp_headers = item
        if status >= 400:
            raise ImmersPowerError(f"http {status}")
        return body, resp_headers

    def get(self, url, token):
        self.calls.append(("GET", url, None, token))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body, _ = item
        if status >= 400:
            raise ImmersPowerError(f"http {status}")
        return body


def make_power(http, **overrides):
    kwargs = dict(
        auth_url="https://api.immers.cloud:5000/v3",
        compute_url="https://api.immers.cloud:8774/v2.1",
        username="example-user",
        password="example-password",
        project="example-user",
        domain="Default",
        server_id="srv-example",
        http=http,
        sleep=lambda _: None,
    )
    kwargs.update(overrides)
    return ImmersPower(**kwargs)


def status_body(status, task=None):
    return (200, {"server": {"status": status, "OS-EXT-STS:task_state": task}}, {})


class ImmersPowerTest(unittest.TestCase):
    def test_token_request_posts_password_auth_and_reads_header(self):
        http = FakeHttp([(201, {}, {"X-Subject-Token": "example-token"})])
        power = make_power(http)

        token = power.token()

        self.assertEqual(token, "example-token")
        method, url, payload, _ = http.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(url, "https://api.immers.cloud:5000/v3/auth/tokens")
        user = payload["auth"]["identity"]["password"]["user"]
        self.assertEqual(user["name"], "example-user")
        self.assertEqual(user["password"], "example-password")
        self.assertEqual(payload["auth"]["scope"]["project"]["name"], "example-user")

    def test_token_is_cached_between_calls(self):
        http = FakeHttp([(201, {}, {"X-Subject-Token": "example-token"})])
        power = make_power(http)

        power.token()
        power.token()

        self.assertEqual(len(http.calls), 1)

    def test_missing_token_header_is_an_error(self):
        http = FakeHttp([(201, {}, {})])
        power = make_power(http)

        with self.assertRaises(ImmersPowerError):
            power.token()

    def test_server_status_parses_status_and_task_state(self):
        http = FakeHttp([
            (201, {}, {"X-Subject-Token": "t"}),
            status_body("SHELVED_OFFLOADED"),
        ])
        power = make_power(http)

        status, task = power.server_status()

        self.assertEqual(status, "SHELVED_OFFLOADED")
        self.assertIsNone(task)

    def test_unshelve_posts_action_and_accepts_202(self):
        http = FakeHttp([
            (201, {}, {"X-Subject-Token": "t"}),
            (202, {}, {}),
        ])
        power = make_power(http)

        power.unshelve()

        method, url, payload, _ = http.calls[-1]
        self.assertTrue(url.endswith("/servers/srv-example/action"))
        self.assertEqual(payload, {"unshelve": None})

    def test_shelve_posts_action(self):
        http = FakeHttp([
            (201, {}, {"X-Subject-Token": "t"}),
            (202, {}, {}),
        ])
        power = make_power(http)

        power.shelve()

        self.assertEqual(http.calls[-1][2], {"shelve": None})

    def test_action_rejects_non_202(self):
        http = FakeHttp([
            (201, {}, {"X-Subject-Token": "t"}),
            (409, {"conflictingRequest": {}}, {}),
        ])
        power = make_power(http)

        with self.assertRaises(ImmersPowerError):
            power.shelve()

    def test_from_env_accepts_os_names_from_immerse_env_file(self):
        saved = {k: os.environ.get(k) for k in list(os.environ) if k.startswith(("IMMERS_", "OS_"))}
        try:
            for key in saved:
                os.environ.pop(key, None)
            os.environ.update(
                {
                    "OS_AUTH_URL": "https://api.immers.cloud:5000/v3",
                    "OS_COMPUTE_URL": "https://api.immers.cloud:8774/v2.1",
                    "OS_USERNAME": "example-user",
                    "OS_PASSWORD": "unit-test-only",
                    "OS_PROJECT_NAME": "example-user",
                    "IMMERS_SERVER_ID": "srv-example",
                }
            )
            power = ImmersPower.from_env()
            self.assertEqual(power.auth_url, "https://api.immers.cloud:5000/v3")
            self.assertEqual(power.project, "example-user")
            self.assertEqual(power.domain, "Default")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_wait_active_polls_until_active(self):
        http = FakeHttp([
            (201, {}, {"X-Subject-Token": "t"}),
            status_body("SHELVED_OFFLOADED", "spawning"),
            status_body("ACTIVE"),
        ])
        power = make_power(http)

        power.wait_active(timeout_seconds=600, poll_seconds=15)

        gets = [c for c in http.calls if c[0] == "GET"]
        self.assertEqual(len(gets), 2)

    def test_wait_survives_transient_api_errors(self):
        http = FakeHttp([
            (201, {}, {"X-Subject-Token": "t"}),
            status_body("SHELVED_OFFLOADED", "spawning"),
            ImmersPowerError("timed out"),
            ImmersPowerError("timed out"),
            status_body("ACTIVE"),
        ])
        power = make_power(http)

        power.wait_active(timeout_seconds=600, poll_seconds=15)

        gets = [c for c in http.calls if c[0] == "GET"]
        self.assertEqual(len(gets), 4)

    def test_wait_gives_up_after_many_transient_errors(self):
        http = FakeHttp(
            [(201, {}, {"X-Subject-Token": "t"})]
            + [ImmersPowerError("timed out") for _ in range(6)]
        )
        clock = {"now": 0.0}

        def fake_sleep(seconds):
            clock["now"] += seconds

        power = make_power(http, sleep=fake_sleep, monotonic=lambda: clock["now"])

        with self.assertRaises(ImmersPowerError):
            power.wait_active(timeout_seconds=600, poll_seconds=15)

    def test_wait_shelved_times_out(self):
        http = FakeHttp(
            [(201, {}, {"X-Subject-Token": "t"})]
            + [status_body("ACTIVE", "shelving_image_uploading") for _ in range(50)]
        )
        clock = {"now": 0.0}

        def fake_sleep(seconds):
            clock["now"] += seconds

        def fake_monotonic():
            return clock["now"]

        power = make_power(http, sleep=fake_sleep, monotonic=fake_monotonic)

        with self.assertRaises(ImmersPowerError):
            power.wait_shelved(timeout_seconds=120, poll_seconds=15)


if __name__ == "__main__":
    unittest.main()
