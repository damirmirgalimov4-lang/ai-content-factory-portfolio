"""immers.cloud power manager: shelve/unshelve a GPU server via OpenStack API.

Stdlib-only. Credentials come from constructor/env, never logged.
The default transport speaks Keystone v3 password auth + Nova actions.
Tests inject a scripted ``http`` object instead of touching the network.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


class ImmersPowerError(RuntimeError):
    pass


class _UrllibHttp:
    """Real transport. Returns (body_dict, headers) for POST and body_dict for GET."""

    def post(self, url, payload, headers=None, token=None):
        return self._request("POST", url, payload, headers, token, want_headers=True)

    def get(self, url, token):
        body, _ = self._request("GET", url, None, None, token, want_headers=True)
        return body

    @staticmethod
    def _request(method, url, payload, headers, token, want_headers):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("X-Auth-Token", token)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read(4 * 1024 * 1024)
                status = resp.status
                resp_headers = dict(resp.headers.items())
        except urllib.error.HTTPError as exc:
            raise ImmersPowerError(f"http {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ImmersPowerError(f"request failed for {url}: {exc}") from exc
        if status >= 400:
            raise ImmersPowerError(f"http {status} for {url}")
        parsed = json.loads(body.decode("utf-8")) if body else {}
        return (parsed, resp_headers) if want_headers else parsed


@dataclass
class ImmersPower:
    auth_url: str
    compute_url: str
    username: str
    password: str
    project: str
    domain: str
    server_id: str
    http: Any = None
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic

    def __post_init__(self):
        if self.http is None:
            self.http = _UrllibHttp()
        self._token: str | None = None

    @classmethod
    def from_env(cls) -> "ImmersPower":
        def pick(*names: str, default: str = "") -> str:
            for name in names:
                value = os.getenv(name, "").strip()
                if value:
                    return value
            return default

        required = {
            "auth_url": pick("IMMERS_AUTH_URL", "OS_AUTH_URL"),
            "compute_url": pick("IMMERS_COMPUTE_URL", "OS_COMPUTE_URL"),
            "username": pick("IMMERS_USERNAME", "OS_USERNAME"),
            "server_id": pick("IMMERS_SERVER_ID"),
        }
        password = os.getenv("IMMERS_PASSWORD") or os.getenv("OS_PASSWORD", "")
        missing = [key for key, value in required.items() if not value]
        if not password:
            missing.append("password")
        if missing:
            raise ImmersPowerError("missing env: " + ", ".join(missing))
        return cls(
            auth_url=required["auth_url"],
            compute_url=required["compute_url"],
            username=required["username"],
            password=password,
            project=pick("IMMERS_PROJECT", "OS_PROJECT_NAME") or required["username"],
            domain=pick("IMMERS_DOMAIN", "OS_USER_DOMAIN_NAME", default="Default"),
            server_id=required["server_id"],
        )

    def token(self) -> str:
        if self._token:
            return self._token
        payload = {
            "auth": {
                "identity": {
                    "methods": ["password"],
                    "password": {
                        "user": {
                            "name": self.username,
                            "domain": {"name": self.domain},
                            "password": self.password,
                        }
                    },
                },
                "scope": {
                    "project": {"name": self.project, "domain": {"name": self.domain}}
                },
            }
        }
        _, headers = self.http.post(f"{self.auth_url}/auth/tokens", payload)
        token = ""
        for key, value in headers.items():
            if key.lower() == "x-subject-token":
                token = str(value).strip()
        if not token:
            raise ImmersPowerError("keystone did not return X-Subject-Token")
        self._token = token
        return token

    def server_status(self) -> tuple[str, str | None]:
        body = self.http.get(f"{self.compute_url}/servers/{self.server_id}", self.token())
        server = body.get("server", {}) if isinstance(body, dict) else {}
        return str(server.get("status", "")), server.get("OS-EXT-STS:task_state")

    def _action(self, name: str) -> None:
        self.http.post(
            f"{self.compute_url}/servers/{self.server_id}/action",
            {name: None},
            token=self.token(),
        )

    def unshelve(self) -> None:
        self._action("unshelve")

    def shelve(self) -> None:
        self._action("shelve")

    def wait_active(self, timeout_seconds: int = 1800, poll_seconds: float = 15.0) -> None:
        self._wait_for({"ACTIVE"}, timeout_seconds, poll_seconds)

    def wait_shelved(self, timeout_seconds: int = 3600, poll_seconds: float = 20.0) -> None:
        self._wait_for({"SHELVED_OFFLOADED"}, timeout_seconds, poll_seconds)

    def _wait_for(self, wanted: set[str], timeout_seconds: int, poll_seconds: float) -> None:
        deadline = self.monotonic() + timeout_seconds
        transient_failures = 0
        while True:
            try:
                status, _task = self.server_status()
                transient_failures = 0
            except ImmersPowerError:
                # The immers API flakes (timeouts) a few times per hour of
                # polling; tolerate isolated failures, give up on a streak.
                transient_failures += 1
                if transient_failures >= 5:
                    raise
                status = None
            if status is not None:
                if status in wanted:
                    return
                if self.monotonic() >= deadline:
                    raise ImmersPowerError(f"timeout waiting for {sorted(wanted)} (last: {status})")
            self.sleep(poll_seconds)
