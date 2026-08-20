from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class WorkbenchError(RuntimeError):
    """A user-facing error for Partner's durable temporary workbench."""


@dataclass(frozen=True)
class ListBatch:
    batch_id: str
    user_id: int
    status: str
    chunks: tuple[dict[str, str], ...]
    analysis_path: str
    shared_item_id: str
    created_at: str
    updated_at: str

    @property
    def text(self) -> str:
        return "\n\n".join(chunk["text"] for chunk in self.chunks if chunk.get("text", "").strip())

    @property
    def line_count(self) -> int:
        return sum(1 for line in self.text.splitlines() if line.strip())


class PartnerWorkbench:
    """Persists multi-message inputs outside long-term memory and chat context."""

    max_chunks = 200
    max_characters = 500_000
    _id_pattern = re.compile(r"^DL-\d{8}-\d{6}-\d{6}$")

    def __init__(self, root: Path):
        self.root = root
        self.lists_root = root / "lists"

    def ensure(self) -> None:
        self.lists_root.mkdir(parents=True, exist_ok=True)

    def create_list(self, user_id: int) -> ListBatch:
        self.ensure()
        now = datetime.now().astimezone()
        batch_id = f"DL-{now.strftime('%Y%m%d-%H%M%S-%f')}"
        stamp = now.isoformat(timespec="seconds")
        payload = {
            "batch_id": batch_id,
            "user_id": int(user_id),
            "status": "collecting",
            "chunks": [],
            "analysis_path": "",
            "shared_item_id": "",
            "created_at": stamp,
            "updated_at": stamp,
        }
        self._write(payload)
        return self.require_list(batch_id, user_id)

    def append_list(
        self,
        batch_id: str,
        user_id: int,
        text: str,
        *,
        source_name: str = "telegram",
    ) -> ListBatch:
        batch = self.require_list(batch_id, user_id)
        if batch.status != "collecting":
            raise WorkbenchError("Этот список уже завершён. Начни новый список.")
        clean = text.strip()
        if not clean:
            raise WorkbenchError("Пустая часть списка не добавлена.")
        if len(batch.chunks) >= self.max_chunks:
            raise WorkbenchError("Достигнут предел в 200 частей. Заверши текущий список.")
        if len(batch.text) + len(clean) > self.max_characters:
            raise WorkbenchError("Список слишком большой. Заверши текущий пакет и начни следующий.")

        payload = self._to_payload(batch)
        payload["chunks"].append(
            {
                "text": clean,
                "source_name": self._clean_source_name(source_name),
                "added_at": self._now(),
            }
        )
        payload["updated_at"] = self._now()
        self._write(payload)
        return self.require_list(batch.batch_id, user_id)

    def finalize_list(self, batch_id: str, user_id: int) -> ListBatch:
        batch = self.require_list(batch_id, user_id)
        if not batch.chunks:
            raise WorkbenchError("Список пуст. Сначала отправь хотя бы одну часть.")
        if batch.status == "cancelled":
            raise WorkbenchError("Этот список отменён.")
        if batch.status == "finalized":
            return batch
        return self._set_fields(batch, status="finalized")

    def cancel_list(self, batch_id: str, user_id: int) -> ListBatch:
        batch = self.require_list(batch_id, user_id)
        if batch.status == "finalized":
            raise WorkbenchError("Завершённый список не отменяется. Можно начать новый.")
        return self._set_fields(batch, status="cancelled")

    def link_analysis(self, batch_id: str, user_id: int, path: Path) -> ListBatch:
        batch = self.require_list(batch_id, user_id)
        return self._set_fields(batch, analysis_path=str(path.resolve()))

    def link_shared_item(self, batch_id: str, user_id: int, item_id: str) -> ListBatch:
        batch = self.require_list(batch_id, user_id)
        return self._set_fields(batch, shared_item_id=item_id.strip().upper())

    def require_list(self, batch_id: str, user_id: int | None = None) -> ListBatch:
        clean_id = batch_id.strip().upper()
        if not self._id_pattern.fullmatch(clean_id):
            raise WorkbenchError("Некорректный ID списка.")
        path = self.lists_root / f"{clean_id}.json"
        if not path.is_file():
            raise WorkbenchError(f"Список не найден: {clean_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkbenchError("Файл списка повреждён и требует проверки с компьютера.") from exc
        batch = self._from_payload(payload)
        if user_id is not None and batch.user_id != int(user_id):
            raise WorkbenchError("Этот список принадлежит другому пользователю.")
        return batch

    def _set_fields(self, batch: ListBatch, **fields: str) -> ListBatch:
        payload = self._to_payload(batch)
        payload.update(fields)
        payload["updated_at"] = self._now()
        self._write(payload)
        return self.require_list(batch.batch_id, batch.user_id)

    def _write(self, payload: dict[str, object]) -> None:
        self.ensure()
        batch_id = str(payload.get("batch_id", "")).strip().upper()
        if not self._id_pattern.fullmatch(batch_id):
            raise WorkbenchError("Некорректный ID списка.")
        path = self.lists_root / f"{batch_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)

    @staticmethod
    def _from_payload(payload: object) -> ListBatch:
        if not isinstance(payload, dict):
            raise WorkbenchError("Некорректное содержимое списка.")
        raw_chunks = payload.get("chunks", [])
        if not isinstance(raw_chunks, list):
            raise WorkbenchError("Некорректные части списка.")
        chunks: list[dict[str, str]] = []
        for raw in raw_chunks:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            chunks.append(
                {
                    "text": text,
                    "source_name": str(raw.get("source_name", "telegram")),
                    "added_at": str(raw.get("added_at", "")),
                }
            )
        status = str(payload.get("status", "collecting"))
        if status not in {"collecting", "finalized", "cancelled"}:
            raise WorkbenchError("Неизвестный статус списка.")
        return ListBatch(
            batch_id=str(payload.get("batch_id", "")).strip().upper(),
            user_id=int(payload.get("user_id", 0)),
            status=status,
            chunks=tuple(chunks),
            analysis_path=str(payload.get("analysis_path", "")),
            shared_item_id=str(payload.get("shared_item_id", "")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
        )

    @staticmethod
    def _to_payload(batch: ListBatch) -> dict[str, object]:
        return {
            "batch_id": batch.batch_id,
            "user_id": batch.user_id,
            "status": batch.status,
            "chunks": [dict(chunk) for chunk in batch.chunks],
            "analysis_path": batch.analysis_path,
            "shared_item_id": batch.shared_item_id,
            "created_at": batch.created_at,
            "updated_at": batch.updated_at,
        }

    @staticmethod
    def _clean_source_name(value: str) -> str:
        clean = "".join(char for char in value if char.isalnum() or char in " ._-()")
        return clean.strip()[:120] or "telegram"

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")
