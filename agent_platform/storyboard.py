from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .vault import now_stamp


STORYBOARD_PROMPT_SOURCE_LABEL = "AI Content Factory Storyboard Template"
STORYBOARD_PROMPT_SOURCE_STATUS = "clean-room MIT template"
STORYBOARD_PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent
    / "prompt_templates"
    / "storyboard_prompt_builder.txt"
)
STORYBOARD_PROMPT_METADATA_PATH = STORYBOARD_PROMPT_TEMPLATE_PATH.with_suffix(
    ".metadata.json"
)


def load_storyboard_prompt_template() -> str:
    """Load the public template as inert, versioned prompt data."""

    text = STORYBOARD_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    required_markers = (
        "Phase 2 requires explicit human approval",
        "Do not proceed to Phase 2 without approval",
    )
    if any(marker not in text for marker in required_markers):
        raise ValueError("Storyboard template is missing its Phase 2 approval gate.")
    return text


def build_guided_storyboard_plan_messages(
    idea_or_script: str,
    *,
    references: list[dict[str, Any]] | None = None,
    previous_plan: dict[str, Any] | None = None,
    revision_request: str = "",
) -> list[dict[str, str]]:
    """Build one bounded request for an automatic, single-sheet storyboard plan."""

    clean_source = idea_or_script.strip()
    if not clean_source:
        raise ValueError("Идея или сценарий не может быть пустой.")
    reference_items: list[dict[str, str]] = []
    for item in references or []:
        if not isinstance(item, dict):
            raise ValueError("Каждый пользовательский референс должен быть объектом.")
        reference_items.append(
            {
                "reference_id": str(item.get("reference_id", "")).strip()[:80],
                "kind": str(item.get("kind", "user_upload")).strip()[:80],
                "label": str(item.get("label", "")).strip()[:200],
                "description": str(item.get("description", "")).strip()[:1000],
                "usage": str(item.get("usage", "")).strip()[:1000],
            }
        )
    schema_example = {
        "schema_version": GUIDED_STORYBOARD_PLAN_SCHEMA_VERSION,
        "title": "Короткое название",
        "logline": "Одно предложение о развитии истории",
        "duration_seconds": 15,
        "aspect_ratio": "16:9",
        "layout": {"columns": 5, "rows": 3},
        "references": [
            {
                "reference_id": "REF-CHAR-01",
                "kind": "character",
                "label": "Персонаж",
                "description": "Внешность и неизменяемые признаки",
                "usage": "Как сохранять continuity",
            }
        ],
        "panels": [
            {
                "panel_id": "P01",
                "order": 1,
                "timecode": "00:00-00:01",
                "shot_type": "wide",
                "visual": "Что физически видно в панели",
                "action": "Одно ясное действие",
                "camera": "Композиция, ракурс и движение камеры",
                "caption": "Короткая подпись",
                "reference_ids": ["REF-CHAR-01"],
            }
        ],
        "sheet_prompt": (
            "One English prompt that requests exactly one professional storyboard sheet "
            "with all numbered panels in chronological order"
        ),
    }
    revision_block = ""
    if previous_plan is not None:
        revision_block = (
            "\n\nПредыдущий валидный план:\n"
            + json.dumps(previous_plan, ensure_ascii=False)[:18000]
        )
    if revision_request.strip():
        revision_block += (
            "\n\nРежиссёрская правка:\n"
            + revision_request.strip()[:4000]
            + "\nВерни план целиком, а не частичный patch."
        )

    return [
        {
            "role": "system",
            "content": (
                "Ты storyboard director и production planner. Верни только валидный "
                "JSON-объект без Markdown и пояснений. Пользователь даёт только "
                "творческие исходники; не задавай ему техническую анкету. Сам разбери "
                "историю на последовательные панели и подготовь план exactly one "
                "storyboard sheet, а не отдельных изображений.\n\n"
                "Правила:\n"
                "- Если параметры не заданы явно, используй 15 секунд, 15 панелей, "
                "16:9 и сетку 5x3. Если источник явно задаёт иное, выбери сетку, "
                "которая вмещает все панели.\n"
                "- P01, P02, ... идут без пропусков и показывают причинно-следственное "
                "развитие истории. Одна панель — один различимый визуальный бит.\n"
                "- Панели одной сцены сохраняют персонажей, геометрию окружения, props, "
                "свет и состояние истории. Новый ракурс меняет композицию, но не мир.\n"
                "- reference_ids панели могут ссылаться только на references из ответа. "
                "Все переданные пользовательские reference_id включи в ответ без "
                "переименования и используй там, где они релевантны.\n"
                "- sheet_prompt напиши на английском. Он должен требовать одну общую "
                "раскадровку, точное число пронумерованных панелей и visual continuity.\n"
                "- Не переходи к video prompt и не утверждай, что изображение уже "
                "сгенерировано.\n\n"
                "Точный контракт JSON:\n"
                + json.dumps(schema_example, ensure_ascii=False)
            ),
        },
        {
            "role": "user",
            "content": (
                "Идея или сценарий:\n"
                + clean_source[:20000]
                + "\n\nОтдельные пользовательские референсы:\n"
                + (
                    json.dumps(reference_items, ensure_ascii=False)[:12000]
                    if reference_items
                    else "[]"
                )
                + revision_block
            ),
        },
    ]


@dataclass(frozen=True)
class StoryboardStage:
    key: str
    title: str
    filename: str
    prompt: str


STORYBOARD_STAGES: tuple[StoryboardStage, ...] = (
    StoryboardStage(
        key="idea",
        title="Идея и сценарий",
        filename="01-IDEA-AND-SCRIPT.md",
        prompt="Опиши идею или вставь сценарий. Можно начать с одного абзаца.",
    ),
    StoryboardStage(
        key="references",
        title="Персонажи, стиль и референсы",
        filename="02-REFERENCES-AND-STYLE.md",
        prompt=(
            "Для раскадровки нужны персонажи, визуальный стиль и "
            "референсы. Добавь также длительность, формат кадра, число панелей и "
            "целевую модель, если они уже известны."
        ),
    ),
    StoryboardStage(
        key="beats",
        title="Биты и панели",
        filename="03-BEATS-AND-PANELS.md",
        prompt=(
            "Разбей историю на панели. Для каждой укажи номер, timecode, тип "
            "кадра, видимое действие и при необходимости реплику."
        ),
    ),
    StoryboardStage(
        key="image_prompt",
        title="Phase 1 — Storyboard image prompt",
        filename="04-STORYBOARD-IMAGE-PROMPT.md",
        prompt=(
            "Подготовь единый prompt для storyboard sheet на основе утверждённого "
            "clean-room шаблона AI Content Factory."
        ),
    ),
    StoryboardStage(
        key="storyboard_result",
        title="Storyboard sheet и замечания",
        filename="05-STORYBOARD-RESULT.md",
        prompt=(
            "Добавь ссылку или описание полученного storyboard sheet и перечисли "
            "замечания. После этого система отдельно попросит approval."
        ),
    ),
    StoryboardStage(
        key="video_prompt",
        title="Phase 2 — Cinematic video prompt",
        filename="06-CINEMATIC-VIDEO-PROMPT.md",
        prompt=(
            "Storyboard уже подтверждён. Вставь cinematic video prompt с "
            "timecodes, камерой, действием, репликами и SFX по каждому shot."
        ),
    ),
    StoryboardStage(
        key="results",
        title="Результаты и журнал генерации",
        filename="07-GENERATION-LOG.md",
        prompt=(
            "Запиши модель, настройки, число попыток, ссылки на результаты, что "
            "получилось и какие дефекты пришлось исправлять."
        ),
    ),
    StoryboardStage(
        key="review",
        title="Финальный разбор",
        filename="08-REVIEW.md",
        prompt=(
            "Подведи итог: что сработало, что не сработало, что стоит "
            "автоматизировать и что должно остаться под ручным контролем."
        ),
    ),
)

STORYBOARD_STAGE_MAP = {stage.key: stage for stage in STORYBOARD_STAGES}
GUIDED_STORYBOARD_WORKFLOW = "guided-v2"
GUIDED_STORYBOARD_PLAN_FILENAME = "02-GUIDED-STORYBOARD-PLAN.json"
GUIDED_STORYBOARD_PLAN_SCHEMA_VERSION = 2
GUIDED_STORYBOARD_SHEET_SIZE = "1536x1024"
GUIDED_STORYBOARD_SHEET_QUOTE_FILENAME = "03-SHEET-QUOTE.json"
GUIDED_STORYBOARD_SHEET_STATE_FILENAME = "04-SHEET-GENERATION.json"
GUIDED_STORYBOARD_REFERENCE_MANIFEST_FILENAME = "references/manifest.json"
GUIDED_STORYBOARD_MAX_REFERENCE_BYTES = 10 * 1024 * 1024
GUIDED_STORYBOARD_MAX_REFERENCES = 20
GUIDED_STORYBOARD_MAX_PANELS = 20
_PROJECT_ID_RE = re.compile(r"SB-\d{8}-\d{3}")
_PANEL_ID_RE = re.compile(r"P\d{2}")
_REFERENCE_ID_RE = re.compile(r"REF-[A-Z0-9-]+")
_TIMECODE_RE = re.compile(r"(?P<start_minutes>\d{2}):(?P<start_seconds>[0-5]\d)-(?P<end_minutes>\d{2}):(?P<end_seconds>[0-5]\d)")


@dataclass(frozen=True)
class StoryboardProject:
    project_id: str
    title: str
    status: str
    current_stage: str
    completed_stages: tuple[str, ...]
    created_at: str
    updated_at: str
    storyboard_approved_at: str = ""
    storyboard_revision_count: int = 0
    rejected_at: str = ""
    workflow: str = "manual-v1"
    pending_operation: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "StoryboardProject":
        return cls(
            project_id=str(data.get("project_id", "")),
            title=str(data.get("title", "")),
            status=str(data.get("status", "in_progress")),
            current_stage=str(data.get("current_stage", STORYBOARD_STAGES[0].key)),
            completed_stages=tuple(
                str(item) for item in data.get("completed_stages", [])
            ),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            storyboard_approved_at=str(data.get("storyboard_approved_at", "")),
            storyboard_revision_count=int(
                data.get("storyboard_revision_count", 0) or 0
            ),
            rejected_at=str(data.get("rejected_at", "")),
            workflow=str(data.get("workflow", "manual-v1")),
            pending_operation=str(data.get("pending_operation", "")),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["completed_stages"] = list(self.completed_stages)
        return data

    @property
    def current_stage_spec(self) -> StoryboardStage:
        return STORYBOARD_STAGE_MAP[self.current_stage]


class StoryboardStore:
    """Isolated storage for legacy manual and guided Storyboard projects."""

    def __init__(self, project_path: Path):
        self.root = project_path / "storyboards"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def create_project(self, idea_or_script: str) -> StoryboardProject:
        clean_content = idea_or_script.strip()
        if not clean_content:
            raise ValueError("Идея или сценарий не может быть пустой.")

        self.ensure()
        project_id = self._next_project_id()
        project_path = self.root / project_id
        project_path.mkdir(parents=False, exist_ok=False)
        stamp = now_stamp()
        project = StoryboardProject(
            project_id=project_id,
            title=self._title_from(clean_content),
            status="in_progress",
            current_stage=STORYBOARD_STAGES[1].key,
            completed_stages=(STORYBOARD_STAGES[0].key,),
            created_at=stamp,
            updated_at=stamp,
            storyboard_approved_at="",
            storyboard_revision_count=0,
            rejected_at="",
        )
        self._write_stage(project_path, STORYBOARD_STAGES[0], clean_content)
        self._write_project(project)
        return project

    def create_guided_project(self, idea_or_script: str) -> StoryboardProject:
        """Create a new automatic Storyboard project without changing legacy projects."""

        clean_content = idea_or_script.strip()
        if not clean_content:
            raise ValueError("Идея или сценарий не может быть пустой.")

        self.ensure()
        project_id = self._next_project_id()
        project_path = self.root / project_id
        project_path.mkdir(parents=False, exist_ok=False)
        stamp = now_stamp()
        project = StoryboardProject(
            project_id=project_id,
            title=self._title_from(clean_content),
            status="planning",
            current_stage="planning",
            completed_stages=("idea",),
            created_at=stamp,
            updated_at=stamp,
            workflow=GUIDED_STORYBOARD_WORKFLOW,
            pending_operation="initial",
        )
        self._write_stage(project_path, STORYBOARD_STAGES[0], clean_content)
        self._write_project(project)
        return project

    def save_generated_plan(
        self,
        project_id: str,
        raw_plan: str,
    ) -> StoryboardProject:
        """Validate and persist one structured plan for a single storyboard sheet."""

        project = self._require_project(project_id)
        if project.workflow != GUIDED_STORYBOARD_WORKFLOW:
            raise ValueError("Автоматический план доступен только для нового Storyboard flow.")
        if project.status not in {"planning", "plan_review"}:
            raise ValueError("Этот Storyboard сейчас не принимает новый план.")

        plan = self._parse_guided_plan(raw_plan)
        uploaded_references = self.list_uploaded_references(project.project_id)
        if uploaded_references:
            required_ids = {
                str(item["reference_id"]) for item in uploaded_references
            }
            planned_ids = {
                str(item["reference_id"]) for item in plan["references"]
            }
            used_ids = {
                str(reference_id)
                for panel in plan["panels"]
                for reference_id in panel["reference_ids"]
            }
            if not required_ids.issubset(planned_ids) or not required_ids.issubset(used_ids):
                raise ValueError(
                    "Новый план должен сохранить и использовать все пользовательские референсы."
                )
        project_path = self.root / project.project_id
        self._write_json_atomic(
            project_path / GUIDED_STORYBOARD_PLAN_FILENAME,
            plan,
        )
        stamp = now_stamp()
        updated = replace(
            project,
            title=str(plan["title"]),
            status="plan_review",
            current_stage="plan_review",
            completed_stages=("idea", "plan"),
            storyboard_approved_at="",
            rejected_at="",
            pending_operation="",
            updated_at=stamp,
        )
        self._write_project(updated)
        return updated

    def read_plan(self, project_id: str) -> dict[str, Any]:
        project = self._require_project(project_id)
        if project.workflow != GUIDED_STORYBOARD_WORKFLOW:
            raise ValueError("У старого ручного Storyboard нет автоматического плана.")
        path = self.root / project.project_id / GUIDED_STORYBOARD_PLAN_FILENAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError("Автоматический план Storyboard ещё не создан.") from exc
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            raise ValueError("Сохранённый план Storyboard повреждён.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Сохранённый план Storyboard имеет неверный формат.")
        try:
            return self._parse_guided_plan(
                json.dumps(payload, ensure_ascii=False)
            )
        except ValueError as exc:
            raise ValueError("Сохранённый план Storyboard повреждён.") from exc

    def list_uploaded_references(self, project_id: str) -> list[dict[str, Any]]:
        project = self._require_project(project_id)
        if project.workflow != GUIDED_STORYBOARD_WORKFLOW:
            return []
        manifest_path = (
            self.root
            / project.project_id
            / GUIDED_STORYBOARD_REFERENCE_MANIFEST_FILENAME
        )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            raise ValueError("Manifest пользовательских референсов повреждён.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("references"), list):
            raise ValueError("Manifest пользовательских референсов повреждён.")
        return [dict(item) for item in payload["references"] if isinstance(item, dict)]

    def begin_reference_update(self, project_id: str) -> StoryboardProject:
        """Close stale approval actions before downloading a new reference asset."""

        project = self._require_project(project_id)
        if (
            project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status != "plan_review"
        ):
            raise ValueError("Сейчас этот Storyboard не принимает референсы.")
        updated = replace(
            project,
            status="planning",
            current_stage="planning",
            completed_stages=("idea",),
            storyboard_approved_at="",
            pending_operation="references",
            updated_at=now_stamp(),
        )
        self._write_project(updated)
        return updated

    def save_uploaded_reference(
        self,
        project_id: str,
        content: bytes,
        description: str,
    ) -> tuple[dict[str, Any], bool]:
        """Validate and persist one user-owned visual reference in the project namespace."""

        project = self._require_project(project_id)
        if (
            project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status != "planning"
            or project.pending_operation != "references"
        ):
            raise ValueError("Сейчас этот Storyboard не принимает референсы.")
        if not content:
            raise ValueError("Файл референса пустой.")
        if len(content) > GUIDED_STORYBOARD_MAX_REFERENCE_BYTES:
            raise ValueError("Референс превышает лимит 10 МБ.")
        suffix = self._image_suffix(content)
        clean_description = " ".join(description.split()).strip()
        if not clean_description:
            clean_description = "Пользовательский визуальный референс."
        clean_description = clean_description[:1000]
        digest = hashlib.sha256(content).hexdigest()
        references = self.list_uploaded_references(project.project_id)
        for item in references:
            if item.get("sha256") == digest:
                return item, False
        if len(references) >= GUIDED_STORYBOARD_MAX_REFERENCES:
            raise ValueError("Для одного Storyboard можно сохранить не более 20 референсов.")

        reference_id = f"REF-UPLOAD-{len(references) + 1:03d}"
        filename = f"{reference_id}{suffix}"
        references_path = self.root / project.project_id / "references"
        references_path.mkdir(parents=True, exist_ok=True)
        target = references_path / filename
        temporary = references_path / f"{filename}.tmp"
        temporary.write_bytes(content)
        temporary.replace(target)
        item: dict[str, Any] = {
            "reference_id": reference_id,
            "kind": "user_upload",
            "label": clean_description[:80],
            "description": clean_description,
            "usage": "Сохранять изображённые визуальные признаки во всех связанных панелях.",
            "filename": filename,
            "sha256": digest,
            "bytes": len(content),
            "created_at": now_stamp(),
        }
        references.append(item)
        self._write_json_atomic(
            references_path / "manifest.json",
            {"schema_version": 1, "references": references},
        )
        return item, True

    @staticmethod
    def _image_suffix(content: bytes) -> str:
        if content.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if (
            len(content) >= 12
            and content.startswith(b"RIFF")
            and content[8:12] == b"WEBP"
        ):
            return ".webp"
        raise ValueError("Поддерживаются только изображения JPEG, PNG или WebP.")

    @staticmethod
    def _image_dimensions(content: bytes) -> tuple[int, int]:
        """Read raster dimensions without adding a production imaging dependency."""

        width = height = 0
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            if len(content) < 24 or content[12:16] != b"IHDR":
                raise ValueError("PNG storyboard sheet не содержит валидный IHDR.")
            width = int.from_bytes(content[16:20], "big")
            height = int.from_bytes(content[20:24], "big")
        elif content.startswith(b"\xff\xd8\xff"):
            position = 2
            start_of_frame = {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }
            while position < len(content):
                while position < len(content) and content[position] == 0xFF:
                    position += 1
                if position >= len(content):
                    break
                marker = content[position]
                position += 1
                if marker in {0x01, *range(0xD0, 0xDA)}:
                    continue
                if marker == 0xDA or position + 2 > len(content):
                    break
                segment_length = int.from_bytes(
                    content[position : position + 2], "big"
                )
                if segment_length < 2 or position + segment_length > len(content):
                    break
                if marker in start_of_frame and segment_length >= 7:
                    height = int.from_bytes(
                        content[position + 3 : position + 5], "big"
                    )
                    width = int.from_bytes(
                        content[position + 5 : position + 7], "big"
                    )
                    break
                position += segment_length
        elif (
            len(content) >= 12
            and content.startswith(b"RIFF")
            and content[8:12] == b"WEBP"
        ):
            position = 12
            while position + 8 <= len(content):
                chunk_type = content[position : position + 4]
                chunk_length = int.from_bytes(
                    content[position + 4 : position + 8], "little"
                )
                start = position + 8
                end = start + chunk_length
                if end > len(content):
                    break
                data = content[start:end]
                if chunk_type == b"VP8X" and len(data) >= 10:
                    width = int.from_bytes(data[4:7], "little") + 1
                    height = int.from_bytes(data[7:10], "little") + 1
                    break
                if chunk_type == b"VP8L" and len(data) >= 5 and data[0] == 0x2F:
                    packed = int.from_bytes(data[1:5], "little")
                    width = (packed & 0x3FFF) + 1
                    height = ((packed >> 14) & 0x3FFF) + 1
                    break
                if (
                    chunk_type == b"VP8 "
                    and len(data) >= 10
                    and data[3:6] == b"\x9d\x01\x2a"
                ):
                    width = int.from_bytes(data[6:8], "little") & 0x3FFF
                    height = int.from_bytes(data[8:10], "little") & 0x3FFF
                    break
                position = end + (chunk_length % 2)
        if width <= 0 or height <= 0:
            raise ValueError("Не удалось определить размер storyboard sheet.")
        return width, height

    @staticmethod
    def _quoted_dimensions(size: str) -> tuple[int, int]:
        match = re.fullmatch(r"([1-9]\d{0,5})x([1-9]\d{0,5})", size.strip())
        if match is None:
            raise ValueError("В условиях генерации указан неподдерживаемый размер.")
        return int(match.group(1)), int(match.group(2))

    def approve_plan(self, project_id: str) -> StoryboardProject:
        """Record approval of the panel sequence without approving the generated sheet."""

        project = self._require_guided_plan_review(project_id)
        stamp = now_stamp()
        decision_path = self.root / project.project_id / "PLAN-APPROVAL.json"
        self._write_json_atomic(
            decision_path,
            {
                "decision": "approved",
                "approved_at": stamp,
                "next_gate": "sheet_cost_confirmation",
                "phase_two_unlocked": False,
            },
        )
        updated = replace(
            project,
            status="plan_approved",
            current_stage="plan_approved",
            completed_stages=("idea", "plan", "plan_approval"),
            storyboard_approved_at="",
            pending_operation="",
            updated_at=stamp,
        )
        self._write_project(updated)
        return updated

    @staticmethod
    def _normalize_sheet_quote(quote: dict[str, Any]) -> dict[str, Any]:
        required = (
            "provider_key",
            "provider_label",
            "model",
            "size",
            "quality",
            "cost_display",
            "billing_note",
            "result_display",
            "input_sha256",
        )
        normalized: dict[str, Any] = {"schema_version": 1}
        for key in required:
            value = quote.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"В условиях генерации отсутствует поле {key}.")
            normalized[key] = value.strip()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized["input_sha256"]):
            raise ValueError("Условия генерации не привязаны к исходным материалам.")
        if quote.get("expected_requests") != 1:
            raise ValueError("Один storyboard sheet должен создавать ровно один запрос.")
        normalized["expected_requests"] = 1
        return normalized

    def prepare_sheet_quote(
        self,
        project_id: str,
        quote: dict[str, Any],
    ) -> StoryboardProject:
        """Persist the exact terms shown before any image-provider call."""

        project = self._require_project(project_id)
        if (
            project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status != "plan_approved"
        ):
            raise ValueError("План Storyboard ещё не готов к расчёту генерации.")
        self.read_plan(project.project_id)
        normalized = self._normalize_sheet_quote(quote)
        normalized["prepared_at"] = now_stamp()
        self._write_json_atomic(
            self.root / project.project_id / GUIDED_STORYBOARD_SHEET_QUOTE_FILENAME,
            normalized,
        )
        updated = replace(
            project,
            status="sheet_awaiting_confirmation",
            current_stage="sheet_awaiting_confirmation",
            pending_operation="sheet_confirmation",
            updated_at=now_stamp(),
        )
        self._write_project(updated)
        return updated

    def read_sheet_quote(self, project_id: str) -> dict[str, Any]:
        project = self._require_project(project_id)
        if project.workflow != GUIDED_STORYBOARD_WORKFLOW:
            raise ValueError("У старого Storyboard нет условий автоматической генерации.")
        path = self.root / project.project_id / GUIDED_STORYBOARD_SHEET_QUOTE_FILENAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError) as exc:
            raise ValueError("Условия генерации Storyboard не найдены или повреждены.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Условия генерации Storyboard повреждены.")
        normalized = self._normalize_sheet_quote(payload)
        prepared_at = payload.get("prepared_at")
        if not isinstance(prepared_at, str) or not prepared_at.strip():
            raise ValueError("Условия генерации Storyboard повреждены.")
        normalized["prepared_at"] = prepared_at.strip()
        return normalized

    def cancel_sheet_quote(self, project_id: str) -> StoryboardProject:
        project = self._require_project(project_id)
        if (
            project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status != "sheet_awaiting_confirmation"
            or project.pending_operation != "sheet_confirmation"
        ):
            raise ValueError("Эти условия генерации уже не ожидают решения.")
        updated = replace(
            project,
            status="plan_approved",
            current_stage="plan_approved",
            pending_operation="",
            updated_at=now_stamp(),
        )
        self._write_project(updated)
        return updated

    def begin_sheet_generation(
        self,
        project_id: str,
        current_quote: dict[str, Any],
    ) -> StoryboardProject:
        """Close the confirmation gate durably before invoking the provider."""

        project = self._require_project(project_id)
        if (
            project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status != "sheet_awaiting_confirmation"
            or project.pending_operation != "sheet_confirmation"
        ):
            raise ValueError("Генерация Storyboard сейчас не ожидает подтверждения.")
        persisted_quote = self.read_sheet_quote(project.project_id)
        current_normalized = self._normalize_sheet_quote(current_quote)
        persisted_normalized = {
            key: value
            for key, value in persisted_quote.items()
            if key != "prepared_at"
        }
        if current_normalized != persisted_normalized:
            raise ValueError(
                "План, референсы или настройки генерации изменились. Подготовь условия заново."
            )
        self.read_plan(project.project_id)
        started_at = now_stamp()
        updated = replace(
            project,
            status="sheet_generating",
            current_stage="sheet_generating",
            pending_operation="sheet_generation",
            updated_at=started_at,
        )
        # The blocking state is the commit point before the non-idempotent call.
        # A repeated old Telegram callback or restart cannot submit it again.
        self._write_project(updated)
        self._write_json_atomic(
            self.root / project.project_id / GUIDED_STORYBOARD_SHEET_STATE_FILENAME,
            {
                "schema_version": 1,
                "status": "generating",
                "started_at": started_at,
                "quote": persisted_quote,
            },
        )
        return updated

    def save_generated_sheet(
        self,
        project_id: str,
        content: bytes,
    ) -> tuple[StoryboardProject, Path, str]:
        project = self._require_project(project_id)
        if (
            project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status != "sheet_generating"
            or project.pending_operation != "sheet_generation"
        ):
            raise ValueError("Для этого Storyboard нет активной генерации sheet.")
        if not content:
            raise ValueError("Image provider вернул пустой storyboard sheet.")
        extension = self._image_suffix(content)
        quote = self.read_sheet_quote(project.project_id)
        expected_dimensions = self._quoted_dimensions(quote["size"])
        actual_dimensions = self._image_dimensions(content)
        if actual_dimensions != expected_dimensions:
            raise ValueError(
                "Image provider вернул storyboard sheet размера "
                f"{actual_dimensions[0]}x{actual_dimensions[1]}, ожидался "
                f"{expected_dimensions[0]}x{expected_dimensions[1]}."
            )

        sheets_path = self.root / project.project_id / "sheets"
        sheets_path.mkdir(parents=True, exist_ok=True)
        version_numbers: list[int] = []
        for path in sheets_path.glob("sheet-v*.*"):
            match = re.fullmatch(r"sheet-v(\d+)\.[A-Za-z0-9]+", path.name)
            if match:
                version_numbers.append(int(match.group(1)))
        version = f"sheet-v{max(version_numbers, default=0) + 1:03d}"
        target = sheets_path / f"{version}{extension}"
        temporary = sheets_path / f".{target.name}.tmp"
        temporary.write_bytes(content)
        temporary.replace(target)

        completed_at = now_stamp()
        self._write_json_atomic(
            self.root / project.project_id / GUIDED_STORYBOARD_SHEET_STATE_FILENAME,
            {
                "schema_version": 1,
                "status": "review",
                "started_at": project.updated_at,
                "completed_at": completed_at,
                "quote": quote,
                "result": {
                    "version": version,
                    "filename": str(target.relative_to(self.root / project.project_id)),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                    "width": actual_dimensions[0],
                    "height": actual_dimensions[1],
                },
            },
        )
        updated = replace(
            project,
            status="sheet_review",
            current_stage="sheet_review",
            completed_stages=tuple((*project.completed_stages, "sheet_generation")),
            pending_operation="",
            updated_at=completed_at,
        )
        self._write_project(updated)
        return updated, target, version

    def mark_sheet_generation_failed(
        self,
        project_id: str,
        error: str,
    ) -> StoryboardProject:
        project = self._require_project(project_id)
        if (
            project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status != "sheet_generating"
            or project.pending_operation != "sheet_generation"
        ):
            raise ValueError("Для этого Storyboard нет активной генерации sheet.")
        failed_at = now_stamp()
        self._write_json_atomic(
            self.root / project.project_id / GUIDED_STORYBOARD_SHEET_STATE_FILENAME,
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at": failed_at,
                "error": error.strip()[:1000] or "unknown image generation error",
                "quote": self.read_sheet_quote(project.project_id),
            },
        )
        updated = replace(
            project,
            status="sheet_reconciliation_required",
            current_stage="sheet_reconciliation_required",
            pending_operation="sheet_reconciliation",
            updated_at=failed_at,
        )
        self._write_project(updated)
        return updated

    def recover_interrupted_sheet_generations(self) -> list[StoryboardProject]:
        """Quarantine requests whose external outcome became unknown after restart."""

        recovered: list[StoryboardProject] = []
        for project in self.list_projects(limit=None):
            if (
                project.workflow != GUIDED_STORYBOARD_WORKFLOW
                or project.status != "sheet_generating"
                or project.pending_operation != "sheet_generation"
            ):
                continue
            try:
                self.latest_generated_sheet(project.project_id)
            except ValueError:
                try:
                    updated = self.mark_sheet_generation_failed(
                        project.project_id,
                        (
                            "Bot restarted after the non-idempotent image request was "
                            "registered; provider outcome is unknown. Automatic retry is blocked."
                        ),
                    )
                except (OSError, ValueError):
                    # Corrupt auxiliary files must not prevent the bot from starting or
                    # reopen the non-idempotent confirmation gate.
                    updated = replace(
                        project,
                        status="sheet_reconciliation_required",
                        current_stage="sheet_reconciliation_required",
                        pending_operation="sheet_reconciliation",
                        updated_at=now_stamp(),
                    )
                    self._write_project(updated)
            else:
                updated = replace(
                    project,
                    status="sheet_review",
                    current_stage="sheet_review",
                    completed_stages=tuple(
                        dict.fromkeys((*project.completed_stages, "sheet_generation"))
                    ),
                    pending_operation="",
                    updated_at=now_stamp(),
                )
                self._write_project(updated)
            recovered.append(updated)
        return recovered

    def latest_generated_sheet(self, project_id: str) -> tuple[Path, dict[str, Any]]:
        project = self._require_project(project_id)
        if project.workflow != GUIDED_STORYBOARD_WORKFLOW:
            raise ValueError("У старого Storyboard нет автоматического sheet.")
        state_path = self.root / project.project_id / GUIDED_STORYBOARD_SHEET_STATE_FILENAME
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError) as exc:
            raise ValueError("Сохранённый storyboard sheet не найден или повреждён.") from exc
        result = state.get("result") if isinstance(state, dict) else None
        if not isinstance(result, dict):
            raise ValueError("Сохранённый storyboard sheet не найден или повреждён.")
        filename = result.get("filename")
        expected_digest = result.get("sha256")
        width = result.get("width")
        height = result.get("height")
        if (
            not isinstance(filename, str)
            or not filename.strip()
            or not isinstance(expected_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
            or not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
        ):
            raise ValueError("Сохранённый storyboard sheet повреждён.")
        project_path = (self.root / project.project_id).resolve()
        target = (project_path / filename).resolve()
        if project_path not in target.parents:
            raise ValueError("Путь storyboard sheet выходит за границы проекта.")
        try:
            content = target.read_bytes()
        except OSError as exc:
            raise ValueError("Сохранённый storyboard sheet недоступен.") from exc
        if hashlib.sha256(content).hexdigest() != expected_digest:
            raise ValueError("Сохранённый storyboard sheet повреждён.")
        actual_dimensions = self._image_dimensions(content)
        if actual_dimensions != (width, height):
            raise ValueError("Размер сохранённого storyboard sheet не совпадает с manifest.")
        quote = state.get("quote") if isinstance(state, dict) else None
        if not isinstance(quote, dict) or actual_dimensions != self._quoted_dimensions(
            str(quote.get("size", ""))
        ):
            raise ValueError("Размер сохранённого storyboard sheet не совпадает с условиями.")
        return target, dict(result)

    def read_sheet_revision_request(self, project_id: str) -> str:
        project = self._require_project(project_id)
        revisions_path = self.root / project.project_id / "sheet-revisions"
        candidates = sorted(revisions_path.glob("request-*.json"))
        if not candidates:
            return ""
        try:
            payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            raise ValueError("Последняя правка storyboard sheet повреждена.") from exc
        request = payload.get("request") if isinstance(payload, dict) else None
        if not isinstance(request, str) or not request.strip():
            raise ValueError("Последняя правка storyboard sheet повреждена.")
        return request.strip()

    def approve_generated_sheet(self, project_id: str) -> StoryboardProject:
        project = self._require_project(project_id)
        if (
            project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status != "sheet_review"
        ):
            raise ValueError("Готовый storyboard sheet сейчас не ожидает подтверждения.")
        _, result = self.latest_generated_sheet(project.project_id)
        stamp = now_stamp()
        self._write_json_atomic(
            self.root / project.project_id / "SHEET-APPROVAL.json",
            {
                "decision": "approved",
                "approved_at": stamp,
                "result": result,
                "phase_two_started": False,
            },
        )
        updated = replace(
            project,
            status="sheet_approved",
            current_stage="sheet_approved",
            completed_stages=tuple((*project.completed_stages, "sheet_approval")),
            # This legacy field means that manual-v1 Phase 2 is unlocked.
            # Guided Phase 1 approval is persisted separately above and must not set it.
            storyboard_approved_at="",
            pending_operation="",
            updated_at=stamp,
        )
        self._write_project(updated)
        return updated

    def request_generated_sheet_revision(
        self,
        project_id: str,
        revision_request: str,
    ) -> StoryboardProject:
        project = self._require_project(project_id)
        if (
            project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status != "sheet_review"
        ):
            raise ValueError("Готовый storyboard sheet сейчас не ожидает правок.")
        clean_request = revision_request.strip()
        if not clean_request:
            raise ValueError("Комментарий к storyboard sheet не может быть пустым.")
        _, result = self.latest_generated_sheet(project.project_id)
        revisions_path = self.root / project.project_id / "sheet-revisions"
        revisions_path.mkdir(parents=True, exist_ok=True)
        number = len(list(revisions_path.glob("request-*.json"))) + 1
        self._write_json_atomic(
            revisions_path / f"request-{number:03d}.json",
            {
                "request": clean_request[:4000],
                "requested_at": now_stamp(),
                "prior_result": result,
            },
        )
        updated = replace(
            project,
            status="plan_approved",
            current_stage="plan_approved",
            storyboard_approved_at="",
            pending_operation="",
            updated_at=now_stamp(),
        )
        self._write_project(updated)
        return updated

    def reject_generated_sheet(
        self,
        project_id: str,
        reason: str,
    ) -> StoryboardProject:
        project = self._require_project(project_id)
        if (
            project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status != "sheet_review"
        ):
            raise ValueError("Готовый storyboard sheet сейчас не ожидает решения.")
        clean_reason = reason.strip()
        if not clean_reason:
            raise ValueError("Причина отклонения storyboard sheet не может быть пустой.")
        _, result = self.latest_generated_sheet(project.project_id)
        stamp = now_stamp()
        self._write_json_atomic(
            self.root / project.project_id / "SHEET-REJECTION.json",
            {
                "decision": "rejected",
                "rejected_at": stamp,
                "reason": clean_reason[:4000],
                "result": result,
                "phase_two_started": False,
            },
        )
        updated = replace(
            project,
            status="rejected",
            current_stage="sheet_review",
            rejected_at=stamp,
            pending_operation="",
            updated_at=stamp,
        )
        self._write_project(updated)
        return updated

    def prepare_plan_revision(
        self,
        project_id: str,
        revision_request: str,
    ) -> StoryboardProject:
        """Archive the accepted JSON plan before asking the planner for a replacement."""

        project = self._require_guided_plan_review(project_id)
        clean_request = revision_request.strip()
        if not clean_request:
            raise ValueError("Комментарий к правкам плана не может быть пустым.")
        plan = self.read_plan(project.project_id)
        revision_number = project.storyboard_revision_count + 1
        revisions_path = self.root / project.project_id / "revisions"
        revisions_path.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(
            revisions_path / f"guided-plan-{revision_number:03d}.json",
            {
                "revision": revision_number,
                "requested_at": now_stamp(),
                "request": clean_request,
                "prior_plan": plan,
            },
        )
        updated = replace(
            project,
            status="planning",
            current_stage="planning",
            completed_stages=("idea",),
            storyboard_approved_at="",
            storyboard_revision_count=revision_number,
            pending_operation="revision",
            updated_at=now_stamp(),
        )
        self._write_project(updated)
        return updated

    def read_pending_plan_revision(self, project_id: str) -> dict[str, Any]:
        project = self._require_project(project_id)
        if (
            project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status != "planning"
            or project.pending_operation != "revision"
            or project.storyboard_revision_count <= 0
        ):
            raise ValueError("Для этого Storyboard нет незавершённой правки плана.")
        path = (
            self.root
            / project.project_id
            / "revisions"
            / f"guided-plan-{project.storyboard_revision_count:03d}.json"
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError) as exc:
            raise ValueError("Контекст незавершённой правки Storyboard повреждён.") from exc
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("request"), str)
            or not str(payload["request"]).strip()
            or not isinstance(payload.get("prior_plan"), dict)
        ):
            raise ValueError("Контекст незавершённой правки Storyboard повреждён.")
        return payload

    def reject_plan(self, project_id: str, reason: str) -> StoryboardProject:
        """Close the guided project before any image or video generation is allowed."""

        project = self._require_guided_plan_review(project_id)
        clean_reason = reason.strip()
        if not clean_reason:
            raise ValueError("Причина отклонения плана не может быть пустой.")
        project_path = self.root / project.project_id
        (project_path / "PLAN-REJECTION.md").write_text(
            f"# План Storyboard отклонён\n\n{clean_reason}\n",
            encoding="utf-8",
            newline="\n",
        )
        stamp = now_stamp()
        updated = replace(
            project,
            status="rejected",
            current_stage="plan_review",
            storyboard_approved_at="",
            rejected_at=stamp,
            pending_operation="",
            updated_at=stamp,
        )
        self._write_project(updated)
        return updated

    def _require_guided_plan_review(self, project_id: str) -> StoryboardProject:
        project = self._require_project(project_id)
        if (
            project.workflow != GUIDED_STORYBOARD_WORKFLOW
            or project.status != "plan_review"
        ):
            raise ValueError("План Storyboard ещё не готов к этому решению.")
        self.read_plan(project.project_id)
        return project

    def list_projects(self, limit: int | None = 20) -> list[StoryboardProject]:
        self.ensure()
        projects: list[StoryboardProject] = []
        for path in self.root.glob("*/project.json"):
            try:
                project = StoryboardProject.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (json.JSONDecodeError, OSError, TypeError):
                continue
            if _PROJECT_ID_RE.fullmatch(project.project_id):
                projects.append(project)
        projects.sort(key=lambda item: (item.created_at, item.project_id), reverse=True)
        return projects if limit is None else projects[: max(0, limit)]

    def get_project(self, project_id: str) -> StoryboardProject | None:
        normalized = project_id.strip().upper()
        if not _PROJECT_ID_RE.fullmatch(normalized):
            return None
        path = self.root / normalized / "project.json"
        try:
            project = StoryboardProject.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
            return None
        if project.project_id != normalized:
            return None
        return project

    def save_current_stage(
        self,
        project_id: str,
        stage_key: str,
        content: str,
    ) -> StoryboardProject:
        project = self._require_project(project_id)
        if project.status == "awaiting_storyboard_approval":
            raise ValueError(
                "Нельзя продолжить до отдельного подтверждения storyboard."
            )
        if project.status != "in_progress":
            raise ValueError("Этот Storyboard уже ожидает завершения или закрыт.")
        if stage_key != project.current_stage:
            raise ValueError(
                f"Ожидался этап {project.current_stage}, получен {stage_key}."
            )
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Нельзя сохранить пустой этап.")

        stage = STORYBOARD_STAGE_MAP[stage_key]
        self._write_stage(self.root / project.project_id, stage, clean_content)
        completed = tuple((*project.completed_stages, stage_key))
        stage_index = self._stage_index(stage_key)
        is_storyboard_gate = stage_key == "storyboard_result"
        is_final = stage_index == len(STORYBOARD_STAGES) - 1
        if is_storyboard_gate:
            status = "awaiting_storyboard_approval"
            current_stage = stage_key
        elif is_final:
            status = "review_ready"
            current_stage = stage_key
        else:
            status = "in_progress"
            current_stage = STORYBOARD_STAGES[stage_index + 1].key
        updated = replace(
            project,
            status=status,
            current_stage=current_stage,
            completed_stages=completed,
            updated_at=now_stamp(),
        )
        self._write_project(updated)
        return updated

    def approve_storyboard(self, project_id: str) -> StoryboardProject:
        project = self._require_project(project_id)
        if (
            project.status != "awaiting_storyboard_approval"
            or "storyboard_result" not in project.completed_stages
        ):
            raise ValueError("Storyboard ещё не готов к подтверждению.")
        approved_at = now_stamp()
        updated = replace(
            project,
            status="in_progress",
            current_stage="video_prompt",
            storyboard_approved_at=approved_at,
            updated_at=approved_at,
        )
        self._write_project(updated)
        return updated

    def request_storyboard_revision(
        self,
        project_id: str,
        revision_request: str,
    ) -> StoryboardProject:
        project = self._require_project(project_id)
        if (
            project.status != "awaiting_storyboard_approval"
            or "storyboard_result" not in project.completed_stages
        ):
            raise ValueError("Storyboard ещё не готов к запросу правок.")
        clean_request = revision_request.strip()
        if not clean_request:
            raise ValueError("Комментарий к правкам не может быть пустым.")

        revision_number = project.storyboard_revision_count + 1
        prior_result = self.read_stage(project.project_id, "storyboard_result")
        revisions_path = self.root / project.project_id / "revisions"
        revisions_path.mkdir(parents=True, exist_ok=True)
        revision_path = revisions_path / f"storyboard-result-{revision_number:03d}.md"
        revision_path.write_text(
            (
                f"# Storyboard revision {revision_number}\n\n"
                "## Сохранённая версия\n\n"
                f"{prior_result}\n\n"
                "## Запрошенные правки\n\n"
                f"{clean_request}\n"
            ),
            encoding="utf-8",
            newline="\n",
        )
        completed = tuple(
            key for key in project.completed_stages if key != "storyboard_result"
        )
        updated = replace(
            project,
            status="in_progress",
            current_stage="storyboard_result",
            completed_stages=completed,
            storyboard_approved_at="",
            storyboard_revision_count=revision_number,
            updated_at=now_stamp(),
        )
        self._write_project(updated)
        return updated

    def reject_storyboard(
        self,
        project_id: str,
        reason: str,
    ) -> StoryboardProject:
        project = self._require_project(project_id)
        if (
            project.status != "awaiting_storyboard_approval"
            or "storyboard_result" not in project.completed_stages
        ):
            raise ValueError("Storyboard ещё не готов к отклонению.")
        clean_reason = reason.strip()
        if not clean_reason:
            raise ValueError("Причина отклонения не может быть пустой.")

        project_path = self.root / project.project_id
        (project_path / "STORYBOARD-REJECTION.md").write_text(
            f"# Storyboard отклонён\n\n{clean_reason}\n",
            encoding="utf-8",
            newline="\n",
        )
        rejected_at = now_stamp()
        updated = replace(
            project,
            status="rejected",
            storyboard_approved_at="",
            rejected_at=rejected_at,
            updated_at=rejected_at,
        )
        self._write_project(updated)
        return updated

    def complete_project(self, project_id: str) -> StoryboardProject:
        project = self._require_project(project_id)
        expected = tuple(stage.key for stage in STORYBOARD_STAGES)
        if project.status != "review_ready" or project.completed_stages != expected:
            raise ValueError("Этапы Storyboard ещё не завершены.")
        completed = replace(
            project,
            status="completed",
            updated_at=now_stamp(),
        )
        self._write_project(completed)
        return completed

    def read_stage(self, project_id: str, stage_key: str) -> str:
        project = self._require_project(project_id)
        stage = STORYBOARD_STAGE_MAP.get(stage_key)
        if stage is None:
            raise ValueError(f"Неизвестный этап: {stage_key}")
        path = self.root / project.project_id / stage.filename
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""

    @staticmethod
    def _parse_guided_plan(raw_plan: str) -> dict[str, Any]:
        source = raw_plan.strip()
        if source.startswith("```"):
            fenced = re.fullmatch(
                r"```(?:json)?\s*(\{.*\})\s*```",
                source,
                re.DOTALL | re.IGNORECASE,
            )
            if fenced:
                source = fenced.group(1)
        try:
            plan = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ValueError("Автоматический план Storyboard не является JSON.") from exc
        if not isinstance(plan, dict):
            raise ValueError("Автоматический план Storyboard должен быть JSON-объектом.")
        if plan.get("schema_version") != GUIDED_STORYBOARD_PLAN_SCHEMA_VERSION:
            raise ValueError("Неподдерживаемая версия автоматического плана Storyboard.")

        required_text = ("title", "logline", "aspect_ratio", "sheet_prompt")
        for key in required_text:
            if not isinstance(plan.get(key), str) or not str(plan[key]).strip():
                raise ValueError(f"В плане Storyboard отсутствует поле {key}.")
            plan[key] = str(plan[key]).strip()
        duration = plan.get("duration_seconds")
        if not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
            raise ValueError("duration_seconds должен быть положительным целым числом.")

        layout = plan.get("layout")
        if not isinstance(layout, dict):
            raise ValueError("В плане Storyboard отсутствует layout.")
        raw_columns = layout.get("columns")
        raw_rows = layout.get("rows")
        if (
            not isinstance(raw_columns, int)
            or isinstance(raw_columns, bool)
            or raw_columns <= 0
            or not isinstance(raw_rows, int)
            or isinstance(raw_rows, bool)
            or raw_rows <= 0
        ):
            raise ValueError("Размеры layout должны быть положительными целыми числами.")
        columns = raw_columns
        rows = raw_rows

        raw_references = plan.get("references")
        if not isinstance(raw_references, list):
            raise ValueError("references должен быть списком.")
        reference_ids: set[str] = set()
        for item in raw_references:
            if not isinstance(item, dict):
                raise ValueError("Каждый reference должен быть объектом.")
            reference_id = str(item.get("reference_id", "")).strip().upper()
            if not _REFERENCE_ID_RE.fullmatch(reference_id):
                raise ValueError("Некорректный reference_id в плане Storyboard.")
            if reference_id in reference_ids:
                raise ValueError(f"Повторяющийся reference_id: {reference_id}.")
            for key in ("kind", "label", "description", "usage"):
                if not isinstance(item.get(key), str) or not str(item[key]).strip():
                    raise ValueError(f"У {reference_id} отсутствует поле {key}.")
                item[key] = str(item[key]).strip()
            item["reference_id"] = reference_id
            reference_ids.add(reference_id)

        raw_panels = plan.get("panels")
        if not isinstance(raw_panels, list) or not raw_panels:
            raise ValueError("panels должен быть непустым списком.")
        if len(raw_panels) > GUIDED_STORYBOARD_MAX_PANELS:
            raise ValueError("Один storyboard sheet может содержать не более 20 панелей.")
        if columns * rows < len(raw_panels):
            raise ValueError("Сетка Storyboard не вмещает все панели.")
        expected_start = 0
        for expected_order, item in enumerate(raw_panels, start=1):
            if not isinstance(item, dict):
                raise ValueError("Каждая панель должна быть объектом.")
            panel_id = str(item.get("panel_id", "")).strip().upper()
            if panel_id != f"P{expected_order:02d}" or not _PANEL_ID_RE.fullmatch(panel_id):
                raise ValueError("Панели должны идти последовательно: P01, P02, ...")
            if item.get("order") != expected_order:
                raise ValueError("Порядок панелей должен быть последовательным.")
            for key in (
                "timecode",
                "shot_type",
                "visual",
                "action",
                "camera",
                "caption",
            ):
                if not isinstance(item.get(key), str) or not str(item[key]).strip():
                    raise ValueError(f"У {panel_id} отсутствует поле {key}.")
                item[key] = str(item[key]).strip()
            timecode = _TIMECODE_RE.fullmatch(item["timecode"])
            if timecode is None:
                raise ValueError(f"У {panel_id} неверный timecode; нужен формат MM:SS-MM:SS.")
            start = (
                int(timecode.group("start_minutes")) * 60
                + int(timecode.group("start_seconds"))
            )
            end = (
                int(timecode.group("end_minutes")) * 60
                + int(timecode.group("end_seconds"))
            )
            if start != expected_start or end <= start:
                raise ValueError(
                    "Панели должны образовывать непрерывный таймлайн без дыр и перекрытий."
                )
            expected_start = end
            panel_reference_ids = item.get("reference_ids")
            if not isinstance(panel_reference_ids, list) or any(
                not isinstance(reference_id, str)
                or reference_id.strip().upper() not in reference_ids
                for reference_id in panel_reference_ids
            ):
                raise ValueError(f"У {panel_id} есть неизвестный reference_id.")
            item["panel_id"] = panel_id
            item["reference_ids"] = [
                reference_id.strip().upper() for reference_id in panel_reference_ids
            ]
        if expected_start != duration:
            raise ValueError(
                "Таймлайн панелей должен заканчиваться на общей duration_seconds."
            )
        return plan

    @staticmethod
    def _write_json_atomic(target: Path, payload: dict[str, Any]) -> None:
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(target)

    def _next_project_id(self) -> str:
        date_part = datetime.now().strftime("%Y%m%d")
        prefix = f"SB-{date_part}-"
        numbers: list[int] = []
        for path in self.root.glob(f"{prefix}*"):
            suffix = path.name.removeprefix(prefix)
            if suffix.isdigit():
                numbers.append(int(suffix))
        return f"{prefix}{max(numbers, default=0) + 1:03d}"

    def _require_project(self, project_id: str) -> StoryboardProject:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError(f"Storyboard-проект не найден: {project_id}")
        return project

    def _write_project(self, project: StoryboardProject) -> None:
        project_path = self.root / project.project_id
        project_path.mkdir(parents=True, exist_ok=True)
        target = project_path / "project.json"
        temporary = project_path / "project.json.tmp"
        temporary.write_text(
            json.dumps(project.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(target)

    @staticmethod
    def _write_stage(
        project_path: Path,
        stage: StoryboardStage,
        content: str,
    ) -> None:
        (project_path / stage.filename).write_text(
            f"# {stage.title}\n\n{content}\n",
            encoding="utf-8",
            newline="\n",
        )

    @staticmethod
    def _title_from(content: str) -> str:
        title = " ".join(content.splitlines()[0].split())
        return title if len(title) <= 80 else title[:77].rstrip() + "..."

    @staticmethod
    def _stage_index(stage_key: str) -> int:
        for index, stage in enumerate(STORYBOARD_STAGES):
            if stage.key == stage_key:
                return index
        raise ValueError(f"Неизвестный этап: {stage_key}")
