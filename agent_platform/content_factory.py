from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from .vault import now_stamp


@dataclass(frozen=True)
class StageSpec:
    key: str
    title: str
    agent: str
    filename: str
    instructions: str


PIPELINE: tuple[StageSpec, ...] = (
    StageSpec(
        key="brief",
        title="Концепт и бриф",
        agent="producer",
        filename="01-BRIEF.md",
        instructions=(
            "Собери рабочий бриф: цель ролика, аудитория, платформа, предполагаемая "
            "длительность, основной смысл, 3 варианта хука, выбранный хук, формат, "
            "тон, ограничения и вопросы, которые пока остаются допущениями. "
            "Не выдумывай результаты исследований и факты, которых нет во входных данных. "
            "Пиши компактно: до 700 слов, без повторов и незавершённых разделов."
        ),
    ),
    StageSpec(
        key="script",
        title="Сценарий",
        agent="scriptwriter",
        filename="02-SCRIPT.md",
        instructions=(
            "Подготовь производственный сценарий короткого AI-видео: цель публикации, логлайн, "
            "выбранный хук, подходящая задаче структура, voice-over или экранный текст, "
            "визуальная функция каждой сцены, CTA только при необходимости и риски. "
            "Выбери от 2 до 10 сцен по задаче и общей длительности, без фиксированного "
            "шаблона. Сначала определи полный хронометраж из идеи и брифа, затем назначь "
            "каждой сцене реалистичную длительность до 15 секунд: простому действию меньше, "
            "сложному или эмоциональному моменту больше. Не режь ролик механически на равные "
            "отрезки по 3–4 секунды. Сумма duration_seconds всех сцен должна точно совпадать "
            "с полным хронометражем ролика. "
            "Каждая сцена должна быть одним непрерывным визуальным моментом в одной локации и "
            "одном состоянии времени. Если сценарий содержит монтаж воспоминаний, перечисление "
            "разных событий, смену локации или заметное превращение объекта, раздели это на "
            "отдельные сцены и кадры вместо списка действий внутри одного image_prompt. "
            "В конце добавь обязательный SCENE_CONTRACT JSON по контракту профиля scriptwriter: "
            "одна запись на сцену, reference_ids/location_id и один image_prompt на будущий "
            "отдельный файл. "
            "Сначала закончи краткую читаемую часть, затем выведи полный JSON; не повторяй "
            "одинаковое описание вне контракта."
        ),
    ),
    StageSpec(
        key="storyboard",
        title="Раскадровка",
        agent="storyboarder",
        filename="03-STORYBOARD.md",
        instructions=(
            "Сначала прочитай сценарий целиком и составь визуальную библию: визуальная основа, "
            "все повторяющиеся или identity-critical персонажи и предметы, все локации и точное "
            "распределение сущностей по сценам. Только затем построй кадры по порядку. Для каждого "
            "кадра укажи задачу, композицию, что должно считываться сразу, ограничения и переход. "
            "Заранее закладывай появление повторяющихся персонажей, предметов и пространственных "
            "якорей. Отдельный вариант сущности создавай только для устойчивого редизайна, который "
            "нельзя получить движением. Открытие/закрытие, сгибание/выпрямление, складывание и "
            "другие действия одного предмета всегда используют одну каноническую карточку. "
            "В конце добавь обязательный VISUAL_BIBLE_CONTRACT JSON по профилю роли. "
            "Пиши компактно и не переписывай сценарий целиком."
        ),
    ),
    StageSpec(
        key="prompts",
        title="Промпты",
        agent="prompt-engineer",
        filename="04-PROMPTS.md",
        instructions=(
            "Подготовь пакет промптов на основе полной раскадровки: reference prompts для "
            "каждого повторяющегося персонажа, environment/style references, image prompts по "
            "кадрам, требования к сохранению "
            "персонажей и объектов и негативные ограничения. Один image prompt должен создавать "
            "ровно один отдельный кадр: без grid, collage, contact sheet и нескольких панелей. "
            "Сохрани scene_id и число сцен из SCENE_CONTRACT. В конце добавь IMAGE_PROMPT_CONTRACT "
            "JSON schema_version 3 с references, locations и scenes. Каждая сцена должна явно "
            "перечислять reference_ids всех видимых персонажей и location_id; для повторяющейся "
            "локации выбери canonical_scene_id, кадр которого станет визуальным якорем. "
            "Для настоящего устойчивого редизайна сохрани identity_group/state_label/base_reference_id. "
            "Не создавай второй reference для открытого/закрытого, прямого/закрученного, "
            "сложенного/разложенного состояния: это действие будущего video prompt. Передавай сцене только "
            "те references, которые действительно видны или нужны именно в этом кадре. "
            "Все reference prompts и image prompts, отправляемые генератору, пиши только на "
            "английском. Перед контрактом кратко перечисли число персонажей, важных предметов, "
            "локаций, кадров и общий планируемый объём изображений. Не создавай отдельный "
            "референс для каждого одноразового фонового предмета. "
            "Финальные video prompts на этом этапе не создавай. Не дублируй один и тот же "
            "image prompt вне JSON-контракта в нескольких разделах."
        ),
    ),
    StageSpec(
        key="qa",
        title="Проверка пакета",
        agent="qa-delivery",
        filename="05-QA.md",
        instructions=(
            "Выполни pre-generation проверку связки сценарий -> визуальная библия -> раскадровка "
            "-> image prompts. Проверь логическую цепочку, полноту персонажей и важных предметов, "
            "совпадение ID и количества кадров, английский язык production prompts и выполнимость "
            "каждого действия в указанную длительность. Отдельно проверь, что монтажные эпизоды не "
            "спрятаны списком внутри одного кадра, состояния одной сущности связаны через базовый "
            "референс, а в сцену не подставлены все референсы проекта без необходимости. Дай verdict pass/warning/fail, список "
            "найденных проблем, что нужно исправить перед генерацией кадров, и точный следующий "
            "производственный шаг. Не утверждай, "
            "что видео или изображения уже созданы или доставлены."
        ),
    ),
)

STAGES = {stage.key: stage for stage in PIPELINE}


@dataclass(frozen=True)
class ContentRun:
    run_id: str
    idea: str
    status: str
    current_stage: str
    completed_stages: tuple[str, ...]
    created_at: str
    updated_at: str
    last_error: str = ""

    @property
    def current_stage_spec(self) -> StageSpec:
        return STAGES[self.current_stage]

    @classmethod
    def from_dict(cls, data: dict) -> "ContentRun":
        return cls(
            run_id=str(data.get("run_id", "")),
            idea=str(data.get("idea", "")),
            status=str(data.get("status", "queued")),
            current_stage=str(data.get("current_stage", PIPELINE[0].key)),
            completed_stages=tuple(str(item) for item in data.get("completed_stages", [])),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            last_error=str(data.get("last_error", "")),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["completed_stages"] = list(self.completed_stages)
        return data


class ContentFactoryStore:
    """Durable state for content runs and their generated stage artifacts."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.runs_path = project_path / "runs"

    def ensure(self) -> None:
        self.runs_path.mkdir(parents=True, exist_ok=True)

    def create_run(self, idea: str) -> ContentRun:
        clean_idea = idea.strip()
        if not clean_idea:
            raise ValueError("Идея ролика не может быть пустой.")

        self.ensure()
        run_id = self._next_run_id()
        run_path = self.runs_path / run_id
        run_path.mkdir(parents=True, exist_ok=False)

        stamp = now_stamp()
        run = ContentRun(
            run_id=run_id,
            idea=clean_idea,
            status="queued",
            current_stage=PIPELINE[0].key,
            completed_stages=(),
            created_at=stamp,
            updated_at=stamp,
        )
        idea_path = run_path / "00-IDEA.md"
        idea_path.write_text(
            f"# Идея\n\n{clean_idea}\n",
            encoding="utf-8",
            newline="\n",
        )
        (run_path / "delivery.jsonl").write_text("", encoding="utf-8", newline="\n")
        self._write_run(run)
        self._record_artifact(run, "idea", idea_path)
        return run

    def list_runs(self, limit: int = 20) -> list[ContentRun]:
        self.ensure()
        runs: list[ContentRun] = []
        for run_file in self.runs_path.glob("*/run.json"):
            try:
                runs.append(ContentRun.from_dict(json.loads(run_file.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError, TypeError):
                continue
        runs.sort(key=lambda item: item.created_at, reverse=True)
        return runs[:limit]

    def get_run(self, run_id: str) -> ContentRun | None:
        normalized = run_id.strip().upper()
        if not re.fullmatch(r"CF-\d{8}-\d{3}", normalized):
            return None
        run_file = self.runs_path / normalized / "run.json"
        if not run_file.exists():
            return None
        try:
            return ContentRun.from_dict(json.loads(run_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return None

    def mark_running(self, run_id: str) -> ContentRun:
        run = self._require_run(run_id)
        if run.status in {"cancelled", "ready_for_production"}:
            raise ValueError("Завершённый или отменённый запуск нельзя запустить повторно.")
        updated = replace(run, status="running", updated_at=now_stamp(), last_error="")
        self._write_run(updated)
        return updated

    def save_stage(self, run_id: str, stage_key: str, content: str) -> ContentRun:
        run = self._require_run(run_id)
        if stage_key != run.current_stage:
            raise ValueError(f"Ожидался этап {run.current_stage}, получен {stage_key}.")
        stage = STAGES[stage_key]
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Модель вернула пустой результат этапа.")

        artifact_path = self.runs_path / run.run_id / stage.filename
        artifact_path.write_text(
            f"# {stage.title}\n\n{clean_content}\n",
            encoding="utf-8",
            newline="\n",
        )
        completed = list(run.completed_stages)
        if stage_key not in completed:
            completed.append(stage_key)
        is_final = stage_key == PIPELINE[-1].key
        updated = replace(
            run,
            status="ready_for_production" if is_final else "waiting_approval",
            completed_stages=tuple(completed),
            updated_at=now_stamp(),
            last_error="",
        )
        self._write_run(updated)
        self._record_artifact(updated, stage_key, artifact_path)
        return updated

    def advance(self, run_id: str) -> ContentRun:
        run = self._require_run(run_id)
        if run.status != "waiting_approval":
            raise ValueError("Продолжить можно только после проверки текущего этапа.")
        if run.current_stage not in run.completed_stages:
            raise ValueError("Текущий этап ещё не завершён.")
        index = self._stage_index(run.current_stage)
        if index >= len(PIPELINE) - 1:
            return run
        updated = replace(
            run,
            current_stage=PIPELINE[index + 1].key,
            status="queued",
            updated_at=now_stamp(),
            last_error="",
        )
        self._write_run(updated)
        return updated

    def mark_failed(self, run_id: str, error: str) -> ContentRun:
        run = self._require_run(run_id)
        clean_error = error.strip()[:500] or "Неизвестная ошибка"
        updated = replace(
            run,
            status="failed",
            updated_at=now_stamp(),
            last_error=clean_error,
        )
        self._write_run(updated)
        return updated

    def cancel(self, run_id: str) -> ContentRun:
        run = self._require_run(run_id)
        updated = replace(run, status="cancelled", updated_at=now_stamp())
        self._write_run(updated)
        return updated

    def read_artifact(self, run_id: str, stage_key: str) -> str:
        run = self._require_run(run_id)
        if stage_key == "idea":
            path = self.runs_path / run.run_id / "00-IDEA.md"
        else:
            stage = STAGES.get(stage_key)
            if stage is None:
                raise ValueError(f"Неизвестный этап: {stage_key}")
            path = self.runs_path / run.run_id / stage.filename
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def replace_stage_artifact(self, run_id: str, stage_key: str, content: str) -> Path:
        """Persist a validated structural migration for an already completed stage."""

        run = self._require_run(run_id)
        stage = STAGES.get(stage_key)
        if stage is None or stage_key not in run.completed_stages:
            raise ValueError("Заменить можно только уже завершённый этап.")
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Нельзя сохранить пустой артефакт этапа.")
        path = self.runs_path / run.run_id / stage.filename
        path.write_text(
            f"# {stage.title}\n\n{clean_content}\n",
            encoding="utf-8",
            newline="\n",
        )
        self._record_artifact(run, stage_key, path)
        return path

    def previous_artifacts(self, run: ContentRun) -> dict[str, str]:
        current_index = self._stage_index(run.current_stage)
        result: dict[str, str] = {}
        for stage in PIPELINE[: current_index + 1]:
            if stage.key in run.completed_stages:
                result[stage.key] = self.read_artifact(run.run_id, stage.key)
        return result

    def save_visual_artifact(
        self,
        run_id: str,
        artifact_key: str,
        filename: str,
        content: bytes,
    ) -> Path:
        run = self._require_run(run_id)
        if not re.fullmatch(r"[A-Za-z0-9._-]+", filename):
            raise ValueError("Недопустимое имя визуального файла.")
        if not content:
            raise ValueError("Нельзя сохранить пустое изображение.")

        visual_path = self.runs_path / run.run_id / "visuals"
        visual_path.mkdir(parents=True, exist_ok=True)
        target = visual_path / filename
        target.write_bytes(content)
        updated = replace(run, updated_at=now_stamp())
        self._write_run(updated)
        self._record_artifact(updated, artifact_key, target)
        return target

    def run_path(self, run_id: str) -> Path:
        """Return a validated run directory for production modules."""

        run = self._require_run(run_id)
        return self.runs_path / run.run_id

    def next_visual_draft_filename(self, run_id: str, extension: str = ".png") -> str:
        run = self._require_run(run_id)
        visual_path = self.runs_path / run.run_id / "visuals"
        numbers: list[int] = []
        for path in visual_path.glob("visual-draft-v*.*") if visual_path.exists() else []:
            match = re.fullmatch(r"visual-draft-v(\d+)\.[A-Za-z0-9]+", path.name)
            if match:
                numbers.append(int(match.group(1)))
        return f"visual-draft-v{max(numbers, default=0) + 1:03d}{extension}"

    def list_visual_drafts(self, run_id: str) -> list[Path]:
        run = self._require_run(run_id)
        visual_path = self.runs_path / run.run_id / "visuals"
        if not visual_path.exists():
            return []
        return sorted(visual_path.glob("visual-draft-v*.*"))

    def _next_run_id(self) -> str:
        date_part = datetime.now().strftime("%Y%m%d")
        prefix = f"CF-{date_part}-"
        numbers: list[int] = []
        for path in self.runs_path.glob(f"{prefix}*"):
            suffix = path.name.removeprefix(prefix)
            if suffix.isdigit():
                numbers.append(int(suffix))
        return f"{prefix}{(max(numbers, default=0) + 1):03d}"

    def _require_run(self, run_id: str) -> ContentRun:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Запуск не найден: {run_id}")
        return run

    def _write_run(self, run: ContentRun) -> None:
        run_path = self.runs_path / run.run_id
        run_path.mkdir(parents=True, exist_ok=True)
        target = run_path / "run.json"
        temporary = run_path / "run.json.tmp"
        temporary.write_text(
            json.dumps(run.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(target)

    def _record_artifact(self, run: ContentRun, stage_key: str, path: Path) -> None:
        manifest_path = self.runs_path / run.run_id / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {
                "run_id": run.run_id,
                "idea_hash": hashlib.sha256(run.idea.encode("utf-8")).hexdigest(),
                "artifacts": [],
            }

        content = path.read_bytes()
        entry = {
            "stage": stage_key,
            "file": path.name,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "updated_at": now_stamp(),
        }
        artifacts = [item for item in manifest.get("artifacts", []) if item.get("stage") != stage_key]
        artifacts.append(entry)
        manifest["artifacts"] = artifacts
        manifest["updated_at"] = now_stamp()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def _stage_index(self, stage_key: str) -> int:
        for index, stage in enumerate(PIPELINE):
            if stage.key == stage_key:
                return index
        raise ValueError(f"Неизвестный этап: {stage_key}")


def build_stage_messages(
    *,
    run: ContentRun,
    stage: StageSpec,
    project_context: str,
    agent_profile: str,
    previous_artifacts: dict[str, str],
    revision_request: str = "",
) -> list[dict[str, str]]:
    """Build a bounded prompt for one production stage without mixing run state into chat memory."""

    previous_parts: list[str] = []
    for key, content in previous_artifacts.items():
        previous_parts.append(f"## {key}\n{content[:12000]}")
    previous_text = "\n\n".join(previous_parts) or "(предыдущих артефактов пока нет)"

    revision_block = ""
    if revision_request.strip():
        revision_block = (
            "\n\nЗапрос на доработку текущего этапа:\n"
            f"{revision_request.strip()}\n"
            "Перепиши артефакт целиком с учётом замечания."
        )

    return [
        {
            "role": "system",
            "content": (
                "Ты специализированный агент AI-video content factory. "
                f"Текущая роль: {stage.agent}. Текущий этап: {stage.title}.\n\n"
                "Верни только содержимое рабочего артефакта в Markdown. "
                "Оформи его для чтения в Telegram: короткие разделы, ясные заголовки, "
                "списки и выделение ключевых фраз жирным. Не используй Markdown-таблицы. "
                "Допущения и предупреждения оформляй короткими цитатами. "
                "Не добавляй служебные метаданные, Run ID и статус этапа внутрь артефакта. "
                "Не пиши, что видео, изображения или файлы доставлены, если это не было сделано. "
                "Не перескакивай на следующие этапы. Не выдумывай внешние исследования.\n\n"
                f"Профиль роли:\n{agent_profile[:10000]}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Run ID: {run.run_id}\n"
                f"Исходная идея:\n{run.idea}\n\n"
                f"Задача этапа:\n{stage.instructions}\n\n"
                f"Контекст проекта:\n{project_context[:14000]}\n\n"
                f"Предыдущие артефакты:\n{previous_text}"
                f"{revision_block}"
            ),
        },
    ]
