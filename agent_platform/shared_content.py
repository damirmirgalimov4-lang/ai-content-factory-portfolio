from __future__ import annotations

import json
import re
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


SHARED_ROLES = {"partner", "owner", "system"}
SHARED_ITEM_KINDS = {"material", "production_idea"}
SHARED_STATUSES = {
    "new",
    "handoff_requested",
    "accepted",
    "returned",
    "rejected",
    "in_production",
    "ready",
    "published",
}

STATUS_LABELS = {
    "new": "Новый",
    "handoff_requested": "Передан владельцу",
    "accepted": "Принят владельцем",
    "returned": "Нужна доработка",
    "rejected": "Отклонён",
    "in_production": "В производстве",
    "ready": "Готов",
    "published": "Опубликован",
}


class SharedContentError(RuntimeError):
    pass


class SharedPermissionError(SharedContentError):
    pass


class SharedTransitionError(SharedContentError):
    pass


@dataclass(frozen=True)
class SharedContentItem:
    item_id: str
    item_kind: str
    source_type: str
    source_text: str
    source_url: str
    title: str
    metadata_json: str
    notes: str
    media_path: str
    status: str
    created_by_role: str
    assigned_to: str
    linked_run_id: str
    created_at: str
    updated_at: str

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)

    @property
    def metadata(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}


def split_manual_sources(text: str, limit: int = 50) -> list[str]:
    """Treat multiple URL-bearing lines as a batch, otherwise preserve one full note."""

    clean = text.strip()
    if not clean:
        return []
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    if len(lines) > 1 and all(_first_url(line) for line in lines):
        return lines[:limit]
    return [clean]


class SharedContentStore:
    """Durable role-gated queue shared by the two isolated Telegram runtimes."""

    def __init__(self, root: Path):
        self.root = root
        self.database_path = root / "workspace.sqlite3"
        self.media_root = root / "media"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.media_root.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS counters (
                    day TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS items (
                    item_id TEXT PRIMARY KEY,
                    item_kind TEXT NOT NULL DEFAULT 'material',
                    source_type TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    source_url TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    notes TEXT NOT NULL DEFAULT '',
                    media_path TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    created_by_role TEXT NOT NULL,
                    assigned_to TEXT NOT NULL DEFAULT '',
                    linked_run_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    actor_role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    old_status TEXT NOT NULL DEFAULT '',
                    new_status TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(item_id) REFERENCES items(item_id)
                );

                CREATE INDEX IF NOT EXISTS idx_items_status_updated
                ON items(status, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_events_item
                ON events(item_id, event_id);
                """
            )
            item_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(items)").fetchall()
            }
            if "item_kind" not in item_columns:
                connection.execute(
                    "ALTER TABLE items ADD COLUMN item_kind TEXT NOT NULL DEFAULT 'material'"
                )
            if "metadata_json" not in item_columns:
                connection.execute(
                    "ALTER TABLE items ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                )

    def create_item(
        self,
        actor_role: str,
        source_text: str,
        *,
        source_type: str = "text",
        source_url: str = "",
        title: str = "",
        item_kind: str = "material",
        metadata: Mapping[str, Any] | None = None,
    ) -> SharedContentItem:
        role = self._require_role(actor_role)
        if role not in {"partner", "owner"}:
            raise SharedPermissionError("Системная роль не создаёт пользовательские материалы.")
        clean_text = source_text.strip()
        if not clean_text:
            raise ValueError("Материал не может быть пустым.")
        clean_type = source_type.strip().lower() or "text"
        if clean_type not in {"text", "link", "photo", "video", "document"}:
            raise ValueError(f"Неподдерживаемый тип материала: {clean_type}")
        clean_kind = item_kind.strip().lower() or "material"
        if clean_kind not in SHARED_ITEM_KINDS:
            raise ValueError(f"Неподдерживаемый вид материала: {clean_kind}")
        metadata_json = json.dumps(
            dict(metadata or {}),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        clean_url = source_url.strip() or _first_url(clean_text)
        if clean_url and clean_type == "text":
            clean_type = "link"

        self.ensure()
        stamp = _now()
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value FROM counters WHERE day = ?", (day,)
            ).fetchone()
            next_value = int(row["value"]) + 1 if row else 1
            connection.execute(
                "INSERT INTO counters(day, value) VALUES(?, ?) "
                "ON CONFLICT(day) DO UPDATE SET value = excluded.value",
                (day, next_value),
            )
            item_id = f"CR-{day}-{next_value:03d}"
            connection.execute(
                """
                INSERT INTO items(
                    item_id, item_kind, source_type, source_text, source_url, title,
                    metadata_json, status, created_by_role, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, 'new', ?, ?, ?)
                """,
                (
                    item_id,
                    clean_kind,
                    clean_type,
                    clean_text,
                    clean_url,
                    title.strip(),
                    metadata_json,
                    role,
                    stamp,
                    stamp,
                ),
            )
            self._append_event(
                connection,
                item_id,
                role,
                "created",
                new_status="new",
            )
        return self.require(item_id)

    def create_batch(self, actor_role: str, text: str) -> list[SharedContentItem]:
        return [
            self.create_item(actor_role, entry, source_url=_first_url(entry))
            for entry in split_manual_sources(text)
        ]

    def get(self, item_id: str) -> SharedContentItem | None:
        normalized = self._normalize_id(item_id)
        self.ensure()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM items WHERE item_id = ?", (normalized,)
            ).fetchone()
        return self._item_from_row(row) if row else None

    def require(self, item_id: str) -> SharedContentItem:
        item = self.get(item_id)
        if item is None:
            raise SharedContentError(f"Материал не найден: {item_id}")
        return item

    def list_items(
        self,
        *,
        statuses: Iterable[str] | None = None,
        item_kinds: Iterable[str] | None = None,
        limit: int = 20,
    ) -> list[SharedContentItem]:
        self.ensure()
        clean_limit = max(1, min(int(limit), 100))
        status_values = [value for value in (statuses or []) if value in SHARED_STATUSES]
        kind_values = [value for value in (item_kinds or []) if value in SHARED_ITEM_KINDS]
        query = "SELECT * FROM items"
        parameters: list[object] = []
        conditions: list[str] = []
        if status_values:
            placeholders = ", ".join("?" for _ in status_values)
            conditions.append(f"status IN ({placeholders})")
            parameters.extend(status_values)
        if kind_values:
            placeholders = ", ".join("?" for _ in kind_values)
            conditions.append(f"item_kind IN ({placeholders})")
            parameters.extend(kind_values)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY updated_at DESC, item_id DESC LIMIT ?"
        parameters.append(clean_limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._item_from_row(row) for row in rows]

    def handoff(self, actor_role: str, item_id: str, note: str = "") -> SharedContentItem:
        if self._require_role(actor_role) != "partner":
            raise SharedPermissionError("Передавать материал владельцу может только роль partner.")
        return self._transition(
            item_id,
            actor_role="partner",
            allowed_from={"new", "returned"},
            new_status="handoff_requested",
            assigned_to="owner",
            action="handoff_requested",
            note=note,
        )

    def return_to_partner(
        self, actor_role: str, item_id: str, note: str
    ) -> SharedContentItem:
        if self._require_role(actor_role) != "owner":
            raise SharedPermissionError("Возвращать материал может только роль owner.")
        clean_note = note.strip()
        if not clean_note:
            raise ValueError("Для возврата нужен короткий комментарий.")
        return self._transition(
            item_id,
            actor_role="owner",
            allowed_from={"handoff_requested", "accepted"},
            new_status="returned",
            assigned_to="partner",
            action="returned",
            note=clean_note,
        )

    def accept(self, actor_role: str, item_id: str) -> SharedContentItem:
        if self._require_role(actor_role) != "owner":
            raise SharedPermissionError("Принимать материал может только роль owner.")
        return self._transition(
            item_id,
            actor_role="owner",
            allowed_from={"handoff_requested"},
            new_status="accepted",
            assigned_to="owner",
            action="accepted",
        )

    def reject(self, actor_role: str, item_id: str, note: str = "") -> SharedContentItem:
        if self._require_role(actor_role) != "owner":
            raise SharedPermissionError("Отклонять материал может только роль owner.")
        return self._transition(
            item_id,
            actor_role="owner",
            allowed_from={"handoff_requested", "accepted"},
            new_status="rejected",
            assigned_to="",
            action="rejected",
            note=note,
        )

    def link_run(
        self, actor_role: str, item_id: str, run_id: str
    ) -> SharedContentItem:
        if self._require_role(actor_role) != "owner":
            raise SharedPermissionError("Связывать запуск может только роль owner.")
        clean_run_id = run_id.strip().upper()
        if not re.fullmatch(r"CF-\d{8}-\d{3,}", clean_run_id):
            raise ValueError(f"Некорректный ID запуска: {run_id}")
        return self._transition(
            item_id,
            actor_role="owner",
            allowed_from={"accepted"},
            new_status="in_production",
            assigned_to="owner",
            linked_run_id=clean_run_id,
            action="linked_to_content_factory",
            note=clean_run_id,
        )

    def mark_ready(self, item_id: str, run_id: str) -> SharedContentItem:
        item = self.require(item_id)
        if item.linked_run_id != run_id.strip().upper():
            raise SharedTransitionError("Материал связан с другим запуском контент-завода.")
        return self._transition(
            item_id,
            actor_role="system",
            allowed_from={"in_production"},
            new_status="ready",
            assigned_to="owner",
            linked_run_id=item.linked_run_id,
            action="production_ready",
        )

    def store_media(
        self,
        actor_role: str,
        item_id: str,
        source_path: Path,
        *,
        original_name: str = "",
    ) -> SharedContentItem:
        role = self._require_role(actor_role)
        item = self.require(item_id)
        if role != item.created_by_role and role != "system":
            raise SharedPermissionError("Прикреплять исходный файл может только автор материала.")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        safe_name = _safe_filename(original_name or source_path.name)
        destination_dir = self.media_root / item.item_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / safe_name
        shutil.copy2(source_path, destination)
        relative_path = destination.relative_to(self.root).as_posix()
        stamp = _now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE items SET media_path = ?, updated_at = ? WHERE item_id = ?",
                (relative_path, stamp, item.item_id),
            )
            self._append_event(
                connection,
                item.item_id,
                role,
                "media_attached",
                old_status=item.status,
                new_status=item.status,
                note=relative_path,
            )
        return self.require(item.item_id)

    def events(self, item_id: str) -> list[dict[str, str]]:
        item = self.require(item_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT actor_role, action, old_status, new_status, note, created_at
                FROM events WHERE item_id = ? ORDER BY event_id
                """,
                (item.item_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _transition(
        self,
        item_id: str,
        *,
        actor_role: str,
        allowed_from: set[str],
        new_status: str,
        assigned_to: str,
        action: str,
        note: str = "",
        linked_run_id: str | None = None,
    ) -> SharedContentItem:
        role = self._require_role(actor_role)
        if new_status not in SHARED_STATUSES:
            raise ValueError(f"Неизвестный статус: {new_status}")
        item = self.require(item_id)
        if item.status not in allowed_from:
            raise SharedTransitionError(
                f"Действие недоступно для статуса «{item.status_label}»."
            )
        stamp = _now()
        run_id = item.linked_run_id if linked_run_id is None else linked_run_id
        clean_note = note.strip()
        combined_notes = item.notes
        if clean_note:
            line = f"[{stamp}] {role}: {clean_note}"
            combined_notes = f"{combined_notes}\n{line}".strip()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT status FROM items WHERE item_id = ?", (item.item_id,)
            ).fetchone()
            if current is None or current["status"] != item.status:
                raise SharedTransitionError("Материал изменился. Обнови карточку и повтори.")
            connection.execute(
                """
                UPDATE items
                SET status = ?, assigned_to = ?, linked_run_id = ?, notes = ?, updated_at = ?
                WHERE item_id = ?
                """,
                (
                    new_status,
                    assigned_to,
                    run_id,
                    combined_notes,
                    stamp,
                    item.item_id,
                ),
            )
            self._append_event(
                connection,
                item.item_id,
                role,
                action,
                old_status=item.status,
                new_status=new_status,
                note=clean_note,
            )
        return self.require(item.item_id)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        item_id: str,
        actor_role: str,
        action: str,
        *,
        old_status: str = "",
        new_status: str = "",
        note: str = "",
    ) -> None:
        connection.execute(
            """
            INSERT INTO events(
                item_id, actor_role, action, old_status, new_status, note, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (item_id, actor_role, action, old_status, new_status, note, _now()),
        )

    @staticmethod
    def _item_from_row(row: sqlite3.Row) -> SharedContentItem:
        return SharedContentItem(**dict(row))

    @staticmethod
    def _require_role(role: str) -> str:
        normalized = role.strip().lower()
        if normalized not in SHARED_ROLES:
            raise SharedPermissionError(f"Неизвестная роль: {role}")
        return normalized

    @staticmethod
    def _normalize_id(item_id: str) -> str:
        normalized = item_id.strip().upper()
        if not re.fullmatch(r"CR-\d{8}-\d{3,}", normalized):
            raise SharedContentError(f"Некорректный ID материала: {item_id}")
        return normalized


def _first_url(text: str) -> str:
    match = re.search(r"https?://[^\s<>]+", text)
    if not match:
        return ""
    return match.group(0).rstrip(".,;:!?)\"]}'")


def _safe_filename(value: str) -> str:
    name = Path(value).name.strip() or "source.bin"
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return clean[:120] or "source.bin"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
