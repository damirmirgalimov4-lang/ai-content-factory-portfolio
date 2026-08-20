from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ReservationConflict(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    """Small SQLite store. Every state transition commits before the next side effect."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        self._lock = threading.RLock()
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS video_jobs (
                    job_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        try:
            item["request"] = json.loads(item.pop("request_json"))
        except (json.JSONDecodeError, TypeError):
            item["request"] = {}
        try:
            item["result"] = json.loads(item.pop("result_json"))
        except (json.JSONDecodeError, TypeError):
            item["result"] = {}
        return item

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM video_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._row(row)

    def require(self, job_id: str) -> dict[str, Any]:
        item = self.get(job_id)
        if item is None:
            raise KeyError(job_id)
        return item

    def reserve(self, request: Any, input_path: Path, output_path: Path | None = None) -> tuple[dict[str, Any], bool]:
        output_path = output_path or input_path.with_name("result.mp4")
        now = _now()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM video_jobs WHERE job_id = ?", (request.job_id,)
            ).fetchone()
            if row is not None:
                connection.execute("COMMIT")
                existing = self._row(row) or {}
                if existing.get("fingerprint") != request.fingerprint:
                    raise ReservationConflict(
                        f"job_id {request.job_id!r} is already bound to another request"
                    )
                return existing, False
            connection.execute(
                """
                INSERT INTO video_jobs (
                    job_id, fingerprint, status, request_json, input_path, output_path,
                    result_json, error, created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, ?, ?, '{}', '', ?, ?)
                """,
                (
                    request.job_id,
                    request.fingerprint,
                    json.dumps(request.stored_dict(), ensure_ascii=False, sort_keys=True),
                    str(input_path),
                    str(output_path),
                    now,
                    now,
                ),
            )
            connection.execute("COMMIT")
        return self.require(request.job_id), True

    def _transition(
        self,
        job_id: str,
        *,
        from_statuses: tuple[str, ...],
        to_status: str,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        placeholders = ",".join("?" for _ in from_statuses)
        values: list[Any] = [
            to_status,
            json.dumps(result or {}, ensure_ascii=False, sort_keys=True),
            error[:2000],
            _now(),
            job_id,
            *from_statuses,
        ]
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                f"""
                UPDATE video_jobs
                   SET status = ?, result_json = ?, error = ?, updated_at = ?
                 WHERE job_id = ? AND status IN ({placeholders})
                """,
                values,
            )
            if cursor.rowcount != 1:
                current = self.get(job_id)
                raise RuntimeError(
                    f"unsafe job transition for {job_id}: "
                    f"{current.get('status') if current else 'missing'} -> {to_status}"
                )
        return self.require(job_id)

    def mark_running(self, job_id: str) -> dict[str, Any]:
        return self._transition(job_id, from_statuses=("queued",), to_status="running")

    def mark_completed(self, job_id: str, result: dict[str, Any]) -> dict[str, Any]:
        return self._transition(
            job_id,
            from_statuses=("running",),
            to_status="completed",
            result=result,
        )

    def mark_failed(
        self,
        job_id: str,
        error: str,
        *,
        from_statuses: tuple[str, ...] = ("running",),
    ) -> dict[str, Any]:
        return self._transition(
            job_id,
            from_statuses=from_statuses,
            to_status="failed",
            error=error,
        )

    def mark_reconciliation_required(self, job_id: str, error: str) -> dict[str, Any]:
        return self._transition(
            job_id,
            from_statuses=("running",),
            to_status="reconciliation_required",
            error=error,
        )

    def recover_interrupted(self) -> int:
        """Never guess whether queued/running work consumed GPU after a process crash."""
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE video_jobs
                   SET status = 'reconciliation_required',
                       error = 'Worker restarted before a terminal result was committed; automatic rerun is blocked.',
                       updated_at = ?
                 WHERE status IN ('queued', 'running')
                """,
                (_now(),),
            )
            return int(cursor.rowcount)

    def close(self) -> None:
        return None
