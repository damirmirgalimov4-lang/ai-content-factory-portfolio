from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def now_stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^\w\-]+", "", value, flags=re.UNICODE)
    value = re.sub(r"-{2,}", "-", value).strip("-_")
    return value or f"project-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def append_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(text.rstrip() + "\n")


@dataclass(frozen=True)
class Project:
    slug: str
    path: Path

    @property
    def title(self) -> str:
        project_file = self.path / "PROJECT.md"
        if not project_file.exists():
            return self.slug

        for line in project_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return self.slug


@dataclass(frozen=True)
class Task:
    task_id: str
    text: str
    status: str
    created_at: str
    completed_at: str | None = None


class VaultStore:
    """File-backed memory store for global identity, projects, notes, and agents."""

    def __init__(self, root: Path):
        self.root = root
        self.workspace = root / "workspace"
        self.projects = root / "projects"
        self.agents = root / "agents"
        self.state_file = root / ".state.json"

    def ensure_bootstrap(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.projects.mkdir(parents=True, exist_ok=True)
        self.agents.mkdir(parents=True, exist_ok=True)

        self._ensure_file(
            self.root / "README.md",
            "# Agent Vault\n\n"
            "Это локальная память агента: глобальные сведения, проекты, заметки и профили агентов.\n",
        )
        self._ensure_file(
            self.workspace / "SOUL.md",
            "# Soul\n\n"
            "Характер, тон и базовые принципы агента.\n\n"
            "- Говорить ясно и по делу.\n"
            "- Не притворяться, что знает то, чего нет в памяти.\n"
            "- Помогать превращать хаотичные идеи в проекты, задачи и решения.\n",
        )
        self._ensure_file(
            self.workspace / "USER.md",
            "# User\n\n"
            "Кто владелец системы, как он работает, какие есть предпочтения и ограничения.\n",
        )
        self._ensure_file(
            self.workspace / "MISSION.md",
            "# Mission\n\n"
            "Зачем существует система: быть личной агентной платформой с памятью, проектами и мобильным Telegram-интерфейсом.\n",
        )
        self._ensure_file(
            self.workspace / "MEMORY.md",
            "# Memory Index\n\n"
            "Короткий routing-index: важные факты, ссылки на проекты и темы, куда смотреть за деталями.\n",
        )
        self._ensure_file(
            self.workspace / "GOALS.md",
            "# Goals\n\n"
            "- Собрать Telegram-first агента с памятью и проектами.\n",
        )
        self._ensure_file(
            self.workspace / "PROJECTS.md",
            "# Projects\n\n"
            "Индекс активных и архивных проектов.\n",
        )
        self._ensure_file(
            self.workspace / "PREFERENCES.md",
            "# Preferences\n\n"
            "Предпочтения пользователя по стилю работы, ответам, инструментам и ограничениям.\n",
        )
        self._ensure_file(
            self.workspace / "LEARNED.md",
            "# Learned\n\n"
            "Выводы и устойчивые паттерны, которые агент понял в процессе работы.\n",
        )
        self._ensure_file(
            self.workspace / "INBOX.md",
            "# Inbox\n\n"
            "Сырые входящие мысли и сообщения до разборки по проектам.\n",
        )
        self._ensure_file(
            self.agents / "general.md",
            "# General Agent\n\n"
            "Роль: универсальный помощник по проектам.\n\n"
            "Правила:\n"
            "- Сначала уточнять активный проект.\n"
            "- Важные факты сохранять в память.\n"
            "- Не смешивать проектный и глобальный контекст.\n",
        )
        self._ensure_default_agents()
        for project_path in sorted(self.projects.iterdir()):
            if project_path.is_dir():
                project = Project(slug=project_path.name, path=project_path)
                self._ensure_project_files(project)
                self._register_project(project)
        self._write_state(self._read_state())

    def list_projects(self) -> list[Project]:
        self.ensure_bootstrap()
        result: list[Project] = []
        for path in sorted(self.projects.iterdir()):
            if path.is_dir():
                result.append(Project(slug=path.name, path=path))
        return result

    def create_project(self, name: str, description: str = "") -> Project:
        self.ensure_bootstrap()
        slug = slugify(name)
        project_path = self.projects / slug
        project_path.mkdir(parents=True, exist_ok=True)
        (project_path / "knowledge").mkdir(exist_ok=True)
        (project_path / "conversations").mkdir(exist_ok=True)
        (project_path / "assets").mkdir(exist_ok=True)
        (project_path / "ops").mkdir(exist_ok=True)

        self._ensure_file(
            project_path / "PROJECT.md",
            f"# {name.strip() or slug}\n\n"
            f"Создано: {now_stamp()}\n\n"
            f"Описание: {description.strip() or 'Пока не заполнено.'}\n",
        )
        self._ensure_file(
            project_path / "MEMORY.md",
            "# Project Memory\n\n"
            "Короткий индекс важных фактов по проекту.\n",
        )
        self._ensure_file(
            project_path / "TASKS.md",
            "# Tasks\n\n"
            "- [ ] Описать цель проекта.\n",
        )
        self._ensure_file(
            project_path / "NOTES.md",
            "# Notes\n\n",
        )
        project = Project(slug=slug, path=project_path)
        self._ensure_project_files(project)
        self._register_project(project)
        return project

    def get_project(self, slug_or_name: str) -> Project | None:
        slug = slugify(slug_or_name)
        path = self.projects / slug
        if path.is_dir():
            return Project(slug=slug, path=path)
        return None

    def set_active_project(self, user_id: int, slug_or_name: str) -> Project:
        project = self.get_project(slug_or_name)
        if project is None:
            raise ValueError(f"Проект не найден: {slug_or_name}")
        self._ensure_project_files(project)

        state = self._read_state()
        state.setdefault("active_projects", {})[str(user_id)] = project.slug
        self._write_state(state)
        return project

    def get_active_project(self, user_id: int) -> Project | None:
        self.ensure_bootstrap()
        state = self._read_state()
        slug = state.get("active_projects", {}).get(str(user_id))
        if slug:
            project = self.get_project(slug)
            if project:
                self._ensure_project_files(project)
            return project

        projects = self.list_projects()
        if len(projects) == 1:
            self._ensure_project_files(projects[0])
            return projects[0]
        return None

    def create_task(self, user_id: int, text: str) -> Task:
        project = self.get_active_project(user_id)
        if project is None:
            raise ValueError("Активный проект не выбран. Создай проект через /new_project или выбери через /use.")

        self._ensure_project_files(project)
        tasks = self._read_tasks(project)
        next_number = len(tasks) + 1
        task = Task(
            task_id=f"T{next_number:03d}",
            text=text.strip(),
            status="open",
            created_at=now_stamp(),
        )
        tasks.append(task)
        self._write_tasks(project, tasks)
        return task

    def list_tasks(self, user_id: int, include_done: bool = False) -> list[Task]:
        project = self.get_active_project(user_id)
        if project is None:
            return []

        self._ensure_project_files(project)
        tasks = self._read_tasks(project)
        if include_done:
            return tasks
        return [task for task in tasks if task.status != "done"]

    def complete_task(self, user_id: int, task_id: str) -> Task:
        project = self.get_active_project(user_id)
        if project is None:
            raise ValueError("Активный проект не выбран.")

        self._ensure_project_files(project)
        normalized = self._normalize_task_id(task_id)
        tasks = self._read_tasks(project)

        for index, task in enumerate(tasks):
            if task.task_id == normalized:
                completed = Task(
                    task_id=task.task_id,
                    text=task.text,
                    status="done",
                    created_at=task.created_at,
                    completed_at=now_stamp(),
                )
                tasks[index] = completed
                self._write_tasks(project, tasks)
                return completed

        raise ValueError(f"Задача не найдена: {task_id}")

    def set_pending_action(self, user_id: int, action: dict[str, str]) -> None:
        state = self._read_state()
        state.setdefault("pending_actions", {})[str(user_id)] = {
            "type": action.get("type", "reply_only"),
            "text": action.get("text", ""),
            "reason": action.get("reason", ""),
            "created_at": now_stamp(),
        }
        self._write_state(state)

    def get_pending_action(self, user_id: int) -> dict[str, str] | None:
        state = self._read_state()
        action = state.get("pending_actions", {}).get(str(user_id))
        if not isinstance(action, dict):
            return None
        return {
            "type": str(action.get("type", "reply_only")),
            "text": str(action.get("text", "")),
            "reason": str(action.get("reason", "")),
            "created_at": str(action.get("created_at", "")),
        }

    def clear_pending_action(self, user_id: int) -> None:
        state = self._read_state()
        state.setdefault("pending_actions", {}).pop(str(user_id), None)
        self._write_state(state)

    def set_input_state(self, user_id: int, input_state: dict[str, str]) -> None:
        """Persist a short Telegram form state so prompts survive process restarts."""
        state = self._read_state()
        state.setdefault("input_states", {})[str(user_id)] = {
            **{str(key): str(value) for key, value in input_state.items()},
            "created_at": now_stamp(),
        }
        self._write_state(state)

    def get_input_state(self, user_id: int) -> dict[str, str] | None:
        state = self._read_state()
        value = state.get("input_states", {}).get(str(user_id))
        if not isinstance(value, dict):
            return None
        return {str(key): str(item) for key, item in value.items()}

    def clear_input_state(self, user_id: int) -> None:
        state = self._read_state()
        state.setdefault("input_states", {}).pop(str(user_id), None)
        self._write_state(state)

    def apply_pending_action(self, user_id: int) -> str:
        action = self.get_pending_action(user_id)
        if action is None:
            return "Нет действия для подтверждения."

        action_type = action["type"]
        text = action["text"]
        if action_type == "remember_global":
            path = self.remember_global(text)
            result = f"Записал в глобальную память:\n{path}"
        elif action_type == "add_project_note":
            path = self.add_project_note(user_id, text)
            result = f"Сохранил заметку в активный проект:\n{path}"
        elif action_type == "create_task":
            task = self.create_task(user_id, text)
            result = f"Создал задачу {task.task_id}:\n{task.text}"
        else:
            result = "Действие не требует применения."

        self.clear_pending_action(user_id)
        return result

    def remember_global(self, text: str) -> Path:
        self.ensure_bootstrap()
        path = self.workspace / "MEMORY.md"
        append_markdown(path, f"\n- [{now_stamp()}] {text}")
        return path

    def add_project_note(self, user_id: int, text: str) -> Path:
        project = self.get_active_project(user_id)
        if project is None:
            raise ValueError("Активный проект не выбран. Создай проект через /new_project или выбери через /use.")

        path = project.path / "NOTES.md"
        append_markdown(path, f"\n## {now_stamp()}\n\n{text}\n")
        return path

    def log_conversation(self, user_id: int, text: str) -> Path:
        path = self._conversation_path(user_id)
        append_markdown(path, f"\n## {now_stamp()}\n\n{text}\n")
        return path

    def log_exchange(self, user_id: int, user_text: str, assistant_text: str) -> Path:
        path = self._conversation_path(user_id)
        append_markdown(
            path,
            f"\n## {now_stamp()}\n\n"
            f"### User\n\n{user_text}\n\n"
            f"### Assistant\n\n{assistant_text}\n",
        )
        return path

    def _conversation_path(self, user_id: int) -> Path:
        project = self.get_active_project(user_id)
        if project is None:
            return self.workspace / "INBOX.md"
        return project.path / "conversations" / f"{datetime.now().strftime('%Y-%m-%d')}.md"

    def list_agents(self) -> list[str]:
        self.ensure_bootstrap()
        return sorted(path.stem for path in self.agents.glob("*.md"))

    def create_agent(self, name: str, purpose: str = "") -> Path:
        self.ensure_bootstrap()
        slug = slugify(name)
        path = self.agents / f"{slug}.md"
        self._ensure_file(
            path,
            f"# {name.strip() or slug}\n\n"
            f"Назначение: {purpose.strip() or 'Пока не заполнено.'}\n\n"
            "Правила:\n"
            "- Работать только в рамках активного проекта.\n"
            "- Сохранять важные выводы в память.\n",
        )
        return path

    def read_agent(self, name: str) -> str | None:
        path = self.agents / f"{slugify(name)}.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8").strip()

    def context_summary(self, user_id: int) -> str:
        self.ensure_bootstrap()
        active = self.get_active_project(user_id)
        parts = [
            "Глобальная память:",
            self._tail(self.workspace / "MEMORY.md", max_lines=12),
            "",
            "Миссия:",
            self._tail(self.workspace / "MISSION.md", max_lines=6),
            "",
            "Предпочтения:",
            self._tail(self.workspace / "PREFERENCES.md", max_lines=6),
            "",
            "Цели:",
            self._tail(self.workspace / "GOALS.md", max_lines=8),
        ]

        if active:
            parts.extend(
                [
                    "",
                    f"Активный проект: {active.title} ({active.slug})",
                    "",
                    "Память проекта:",
                    self._tail(active.path / "MEMORY.md", max_lines=10),
                    "",
                    "Задачи проекта:",
                    self._format_tasks_for_context(self.list_tasks(user_id), max_items=8),
                    "",
                    "Последние заметки:",
                    self._tail(active.path / "NOTES.md", max_lines=10),
                ]
            )
        else:
            parts.append("\nАктивный проект не выбран.")

        return "\n".join(parts).strip()

    def _ensure_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content, encoding="utf-8", newline="\n")

    def _read_state(self) -> dict:
        if not self.state_file.exists():
            return {"active_projects": {}}

        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"active_projects": {}}

    def _write_state(self, state: dict) -> None:
        self.state_file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def _ensure_project_files(self, project: Project) -> None:
        (project.path / "knowledge").mkdir(exist_ok=True)
        (project.path / "conversations").mkdir(exist_ok=True)
        (project.path / "assets").mkdir(exist_ok=True)
        (project.path / "ops").mkdir(exist_ok=True)
        self._ensure_file(project.path / "TASKS.md", "# Tasks\n\n")
        self._ensure_file(project.path / "ops" / "tasks.json", "[]\n")

    def _ensure_default_agents(self) -> None:
        defaults = {
            "researcher": (
                "Ищет идеи, тренды, референсы, конкурентов и полезные источники для проекта.\n\n"
                "Фокус: не писать сценарии, а приносить сырьё и выводы."
            ),
            "scriptwriter": (
                "Превращает идею и ресерч в структуру ролика, хук, сцены, текст и варианты подачи.\n\n"
                "Фокус: ясный сценарий под AI-видео."
            ),
            "producer": (
                "Держит пайплайн производства: задачи, статусы, приоритеты, дедлайны и следующий шаг.\n\n"
                "Фокус: довести идею до опубликованного результата."
            ),
            "prompt-engineer": (
                "Готовит промпты для генерации изображений, видео, озвучки и монтажа.\n\n"
                "Фокус: повторяемые промпт-шаблоны и качество визуального результата."
            ),
        }
        for name, purpose in defaults.items():
            path = self.agents / f"{name}.md"
            self._ensure_file(
                path,
                f"# {name}\n\n"
                f"Назначение: {purpose}\n\n"
                "Правила:\n"
                "- Работать только в рамках активного проекта.\n"
                "- Сохранять важные выводы в память через явные команды или подтверждение пользователя.\n",
            )

    def _register_project(self, project: Project) -> None:
        projects_index = self.workspace / "PROJECTS.md"
        content = projects_index.read_text(encoding="utf-8") if projects_index.exists() else ""
        marker = f"`{project.slug}`"
        if marker in content:
            return
        append_markdown(projects_index, f"\n- {marker} - {project.title}")

    def _read_tasks(self, project: Project) -> list[Task]:
        tasks_path = project.path / "ops" / "tasks.json"
        try:
            raw_tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw_tasks = []

        tasks: list[Task] = []
        for item in raw_tasks:
            tasks.append(
                Task(
                    task_id=str(item.get("id", "")),
                    text=str(item.get("text", "")),
                    status=str(item.get("status", "open")),
                    created_at=str(item.get("created_at", "")),
                    completed_at=item.get("completed_at"),
                )
            )
        return tasks

    def _write_tasks(self, project: Project, tasks: list[Task]) -> None:
        raw_tasks = [
            {
                "id": task.task_id,
                "text": task.text,
                "status": task.status,
                "created_at": task.created_at,
                "completed_at": task.completed_at,
            }
            for task in tasks
        ]
        (project.path / "ops" / "tasks.json").write_text(
            json.dumps(raw_tasks, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        lines = ["# Tasks", ""]
        for task in tasks:
            check = "x" if task.status == "done" else " "
            lines.append(f"- [{check}] {task.task_id} {task.text}")
        (project.path / "TASKS.md").write_text(
            "\n".join(lines).rstrip() + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def _normalize_task_id(self, task_id: str) -> str:
        value = task_id.strip().upper()
        if value.isdigit():
            return f"T{int(value):03d}"
        if value.startswith("T") and value[1:].isdigit():
            return f"T{int(value[1:]):03d}"
        return value

    def _format_tasks_for_context(self, tasks: list[Task], max_items: int) -> str:
        if not tasks:
            return "(открытых задач нет)"
        return "\n".join(
            f"- {task.task_id}: {task.text}" for task in tasks[:max_items]
        )

    def _tail(self, path: Path, max_lines: int) -> str:
        if not path.exists():
            return "(файл пока отсутствует)"
        lines = path.read_text(encoding="utf-8").splitlines()
        tail = lines[-max_lines:]
        return "\n".join(tail).strip() or "(пока пусто)"
