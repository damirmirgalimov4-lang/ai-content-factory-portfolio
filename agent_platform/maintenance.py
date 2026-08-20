from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from .content_factory import ContentFactoryStore
from .llm import find_codex_cli
from .vault import now_stamp


class RepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepairRecord:
    repair_id: str
    run_id: str
    status: str
    branch: str
    base_commit: str
    worktree_path: str
    created_at: str
    updated_at: str
    commit: str = ""
    summary: str = ""
    changed_files: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()
    error: str = ""
    applied_commit: str = ""

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["changed_files"] = list(self.changed_files)
        payload["checks"] = list(self.checks)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RepairRecord":
        return cls(
            repair_id=str(payload.get("repair_id", "")),
            run_id=str(payload.get("run_id", "")),
            status=str(payload.get("status", "")),
            branch=str(payload.get("branch", "")),
            base_commit=str(payload.get("base_commit", "")),
            worktree_path=str(payload.get("worktree_path", "")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            commit=str(payload.get("commit", "")),
            summary=str(payload.get("summary", "")),
            changed_files=tuple(str(item) for item in payload.get("changed_files", [])),
            checks=tuple(str(item) for item in payload.get("checks", [])),
            error=str(payload.get("error", "")),
            applied_commit=str(payload.get("applied_commit", "")),
        )


ProgressCallback = Callable[[str], None]


class GitRepairManager:
    """Prepare a bounded Codex repair in a separate Git worktree and gate its application."""

    ACTIVE_STATUSES = {"queued", "preparing", "applying"}
    READY_STATUSES = {"ready", "applied"}
    ALLOWED_PREFIXES = ("agent_platform/", "tests/", "vault/agents/")
    ALLOWED_FILES: set[str] = set()
    ALLOWED_AGENT_FILES = {
        "vault/agents/producer.md",
        "vault/agents/content-strategist.md",
        "vault/agents/scriptwriter.md",
        "vault/agents/storyboarder.md",
        "vault/agents/prompt-engineer.md",
        "vault/agents/qa-delivery.md",
        "vault/agents/video-prompter.md",
    }
    FORBIDDEN_PARTS = {
        ".git",
        ".codex",
        ".agents",
        "runs",
        "imports",
        "conversations",
        "attachments",
        "external",
        "node_modules",
    }

    def __init__(
        self,
        repository_root: Path,
        content_store: ContentFactoryStore,
        *,
        codex_cli_path: str = "",
        model: str = "gpt-5.6-sol",
        timeout_seconds: int = 1200,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.content_store = content_store
        self.executable = find_codex_cli(codex_cli_path)
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.worktrees_root = self.repository_root / ".tmp" / "codex-repairs"
        self._lock = threading.Lock()

    @property
    def is_configured(self) -> bool:
        return self.executable is not None and (self.repository_root / ".git").exists()

    def create(self, run_id: str) -> RepairRecord:
        run = self.content_store.get_run(run_id)
        if run is None:
            raise RepairError(f"Запуск не найден: {run_id}")
        if not self.is_configured:
            raise RepairError("Codex CLI или Git-репозиторий недоступен для ремонта.")

        latest = self.latest_for_run(run.run_id)
        if latest and latest.status in self.ACTIVE_STATUSES | {"ready"}:
            return latest

        repair_id = f"RP-{uuid.uuid4().hex[:8].upper()}"
        base_commit = self._git(self.repository_root, "rev-parse", "HEAD").stdout.strip()
        record = RepairRecord(
            repair_id=repair_id,
            run_id=run.run_id,
            status="queued",
            branch=f"bot-repair/{repair_id.lower()}",
            base_commit=base_commit,
            worktree_path=str(self.worktrees_root / repair_id),
            created_at=now_stamp(),
            updated_at=now_stamp(),
        )
        return self._write(record)

    def get(self, repair_id: str) -> RepairRecord:
        normalized = self._normalize_repair_id(repair_id)
        for path in self.content_store.runs_path.glob(
            f"*/maintenance/{normalized}.json"
        ):
            try:
                return RepairRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise RepairError(f"Состояние ремонта {normalized} повреждено.") from exc
        raise RepairError(f"Ремонт не найден: {normalized}")

    def latest_for_run(self, run_id: str) -> RepairRecord | None:
        maintenance = self.content_store.run_path(run_id) / "maintenance"
        records: list[RepairRecord] = []
        for path in maintenance.glob("RP-*.json") if maintenance.exists() else []:
            try:
                records.append(
                    RepairRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        records.sort(key=lambda item: item.created_at, reverse=True)
        return records[0] if records else None

    def prepare(
        self,
        repair_id: str,
        progress: ProgressCallback | None = None,
    ) -> RepairRecord:
        callback = progress or (lambda _message: None)
        with self._lock:
            record = self.get(repair_id)
            if record.status == "ready":
                return record
            if record.status not in {"queued", "failed"}:
                raise RepairError(f"Ремонт нельзя подготовить из статуса {record.status}.")
            record = self._write(
                replace(record, status="preparing", updated_at=now_stamp(), error="")
            )

            try:
                callback("Создаю отдельную Git-ветку и изолированную рабочую папку.")
                worktree = Path(record.worktree_path)
                self.worktrees_root.mkdir(parents=True, exist_ok=True)
                if not worktree.exists():
                    branch_exists = self._git(
                        self.repository_root,
                        "show-ref",
                        "--verify",
                        f"refs/heads/{record.branch}",
                        check=False,
                    ).returncode == 0
                    if branch_exists:
                        self._git(
                            self.repository_root,
                            "worktree",
                            "add",
                            str(worktree),
                            record.branch,
                        )
                    else:
                        self._git(
                            self.repository_root,
                            "worktree",
                            "add",
                            "-b",
                            record.branch,
                            str(worktree),
                            record.base_commit,
                        )

                callback("Codex изучает ошибку и готовит минимальное исправление.")
                refs_before = self._git(
                    self.repository_root, "show-ref", "--heads", "--tags"
                ).stdout
                git_config = self._git_config_path(worktree)
                config_before = self._file_hash(git_config)
                summary = self._run_codex(worktree, self._repair_prompt(record))
                refs_after = self._git(
                    self.repository_root, "show-ref", "--heads", "--tags"
                ).stdout
                if refs_after != refs_before:
                    raise RepairError(
                        "Codex изменил Git refs вместо обычных файлов. Ремонт отклонён."
                    )
                if self._file_hash(git_config) != config_before:
                    raise RepairError(
                        "Codex изменил общую Git-конфигурацию. Ремонт отклонён."
                    )
                self._validate_forbidden_artifacts(worktree)
                changed_files = self._changed_files(worktree)
                if not changed_files:
                    raise RepairError("Codex завершил работу, но не изменил ни одного файла.")
                self._validate_changed_files(worktree, changed_files)
                if not any(path.startswith("tests/") for path in changed_files):
                    raise RepairError(
                        "Codex не добавил регрессионный тест, подтверждающий исправление."
                    )

                callback("Запускаю независимую проверку кода и полный набор тестов.")
                checks = self._run_gates(worktree)

                self._git(worktree, "add", "--", *changed_files)
                self._git(worktree, "diff", "--cached", "--check")
                self._git(
                    worktree,
                    "-c",
                    "user.name=Content Factory Repair Bot",
                    "-c",
                    "user.email=repair-bot@local.invalid",
                    "commit",
                    "-m",
                    f"prepare {record.run_id} repair via isolated Codex branch",
                )
                commit = self._git(worktree, "rev-parse", "HEAD").stdout.strip()
                record = self._write(
                    replace(
                        record,
                        status="ready",
                        updated_at=now_stamp(),
                        commit=commit,
                        summary=self._clean_text(summary, 5000),
                        changed_files=tuple(changed_files),
                        checks=tuple(checks),
                        error="",
                    )
                )
                callback("Исправление подготовлено в отдельной ветке и прошло тесты.")
                return record
            except Exception as exc:
                failed = replace(
                    record,
                    status="failed",
                    updated_at=now_stamp(),
                    error=self._clean_text(str(exc), 1200),
                )
                self._write(failed)
                raise

    def apply(
        self,
        repair_id: str,
        progress: ProgressCallback | None = None,
    ) -> RepairRecord:
        callback = progress or (lambda _message: None)
        with self._lock:
            record = self.get(repair_id)
            if record.status == "applied":
                return record
            if record.status != "ready" or not record.commit:
                raise RepairError("Применить можно только готовое и проверенное исправление.")

            current_branch = self._git(
                self.repository_root, "branch", "--show-current"
            ).stdout.strip()
            if not current_branch or current_branch.startswith("bot-repair/"):
                raise RepairError(
                    "Основная рабочая папка находится не на обычной ветке. "
                    "Автоматическое применение остановлено."
                )
            ancestry = self._git(
                self.repository_root,
                "merge-base",
                "--is-ancestor",
                record.base_commit,
                record.commit,
                check=False,
            )
            if ancestry.returncode != 0:
                raise RepairError("Ремонтный коммит не связан с зафиксированной базовой версией.")
            commit_files = self._git(
                self.repository_root,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                record.commit,
            ).stdout.splitlines()
            commit_files = sorted(
                {item.strip().replace("\\", "/") for item in commit_files if item.strip()}
            )
            self._validate_changed_files(self.repository_root, commit_files)
            if tuple(commit_files) != tuple(sorted(record.changed_files)):
                raise RepairError(
                    "Состав файлов ремонтного коммита не совпадает с проверенным отчётом."
                )

            dirty = set(self._changed_files(self.repository_root))
            overlap = sorted(dirty.intersection(commit_files))
            if overlap:
                raise RepairError(
                    "Основная папка содержит незакоммиченные изменения в тех же файлах: "
                    + ", ".join(overlap)
                    + ". Автоматическое применение остановлено, чтобы не потерять работу."
                )

            self._write(replace(record, status="applying", updated_at=now_stamp(), error=""))
            head_before = self._git(
                self.repository_root, "rev-parse", "HEAD"
            ).stdout.strip()
            applied_commit = ""
            try:
                callback("Переношу проверенный коммит в основную ветку.")
                self._git(
                    self.repository_root,
                    "-c",
                    "user.name=Content Factory Repair Bot",
                    "-c",
                    "user.email=repair-bot@local.invalid",
                    "cherry-pick",
                    record.commit,
                )
                applied_commit = self._git(
                    self.repository_root, "rev-parse", "HEAD"
                ).stdout.strip()
                callback("Повторно запускаю тесты уже в основной рабочей папке.")
                checks = self._run_gates(self.repository_root)
                applied = replace(
                    record,
                    status="applied",
                    updated_at=now_stamp(),
                    applied_commit=applied_commit,
                    checks=tuple(checks),
                    error="",
                )
                return self._write(applied)
            except Exception as exc:
                rolled_back_cleanly = self._rollback_failed_apply(
                    head_before,
                    applied_commit,
                )
                rollback_note = "" if rolled_back_cleanly else " Автоматический rollback не подтверждён."
                rolled_back = replace(
                    record,
                    status="rolled_back" if rolled_back_cleanly else "rollback_failed",
                    updated_at=now_stamp(),
                    error=self._clean_text(str(exc) + rollback_note, 1200),
                )
                self._write(rolled_back)
                raise RepairError(
                    "Исправление не прошло применение и было откачено: "
                    f"{rolled_back.error}"
                ) from exc

    def discard(self, repair_id: str) -> RepairRecord:
        with self._lock:
            record = self.get(repair_id)
            if record.status in {"applied", "apply_interrupted", "rollback_failed"}:
                raise RepairError(
                    "Это состояние нельзя удалять из Telegram: сначала нужна проверка Git с компьютера."
                )
            worktree = Path(record.worktree_path)
            if worktree.exists():
                self._git(self.repository_root, "worktree", "remove", "--force", str(worktree))
            branch_exists = self._git(
                self.repository_root,
                "show-ref",
                "--verify",
                f"refs/heads/{record.branch}",
                check=False,
            ).returncode == 0
            if branch_exists:
                self._git(self.repository_root, "branch", "-D", record.branch)
            return self._write(
                replace(record, status="discarded", updated_at=now_stamp(), error="")
            )

    def recover_interrupted(self) -> list[RepairRecord]:
        """Mark abandoned workers without guessing whether an interrupted apply is safe."""

        recovered: list[RepairRecord] = []
        for path in self.content_store.runs_path.glob("*/maintenance/RP-*.json"):
            try:
                record = RepairRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if record.status == "preparing":
                recovered.append(
                    self._write(
                        replace(
                            record,
                            status="failed",
                            updated_at=now_stamp(),
                            error=(
                                "Подготовка прервалась при остановке бота. Изолированная ветка "
                                "сохранена; подготовку можно запустить повторно."
                            ),
                        )
                    )
                )
            elif record.status == "applying":
                recovered.append(
                    self._write(
                        replace(
                            record,
                            status="apply_interrupted",
                            updated_at=now_stamp(),
                            error=(
                                "Бот остановился во время переноса исправления. Нужна проверка Git "
                                "с компьютера; автоматическое продолжение запрещено."
                            ),
                        )
                    )
                )
        return recovered

    def describe(self, record: RepairRecord) -> str:
        status = {
            "queued": "в очереди",
            "preparing": "Codex готовит исправление",
            "ready": "готово к применению",
            "applying": "применяется",
            "applied": "применено",
            "failed": "подготовка завершилась ошибкой",
            "rolled_back": "применение отменено и откачено",
            "apply_interrupted": "применение прервалось, нужна проверка с компьютера",
            "rollback_failed": "автоматический rollback не подтверждён, нужна проверка с компьютера",
            "discarded": "отклонено",
        }.get(record.status, record.status)
        lines = [
            f"🧰 Ремонт {record.repair_id}",
            "",
            f"Запуск: {record.run_id}",
            f"Статус: {status}",
            f"Ветка: {record.branch}",
        ]
        if record.changed_files:
            lines.extend(["", "Изменённые файлы:"])
            lines.extend(f"- {path}" for path in record.changed_files)
        if record.checks:
            lines.extend(["", "Проверки:"])
            lines.extend(f"- {check}" for check in record.checks)
        if record.summary:
            lines.extend(["", "Что сделал Codex:", self._clean_text(record.summary, 2500)])
        if record.error:
            lines.extend(["", "Причина:", self._clean_text(record.error, 800)])
        if record.status == "applied":
            lines.extend(
                [
                    "",
                    "Код применён в основную ветку. Запущенный бот продолжает работать на старом коде до перезапуска.",
                ]
            )
        return "\n".join(lines)

    def _repair_prompt(self, record: RepairRecord) -> str:
        run = self.content_store.get_run(record.run_id)
        if run is None:
            raise RepairError(f"Запуск не найден: {record.run_id}")
        artifacts = {
            key: self._clean_text(value, 12000)
            for key, value in self.content_store.previous_artifacts(run).items()
        }
        production_path = self.content_store.run_path(run.run_id) / "production.json"
        production = ""
        if production_path.exists():
            production = self._clean_text(production_path.read_text(encoding="utf-8"), 20000)
        diagnostics_dir = self.content_store.run_path(run.run_id) / "diagnostics"
        diagnostics = ""
        reports = sorted(diagnostics_dir.glob("DIAG-*.md")) if diagnostics_dir.exists() else []
        if reports:
            diagnostics = self._clean_text(reports[-1].read_text(encoding="utf-8"), 12000)
        evidence = {
            "run": self._redact_payload(run.to_dict()),
            "artifacts": artifacts,
            "production_state": production,
            "latest_diagnostic": diagnostics,
        }
        return (
            "Ты maintenance-инженер AI Content Factory. Работаешь в отдельном Git worktree. "
            "Изучи репозиторий и evidence bundle, найди подтверждённую программную причину ошибки, "
            "внеси минимальное исправление и добавь регрессионные тесты.\n\n"
            "ОБЯЗАТЕЛЬНЫЕ ОГРАНИЧЕНИЯ:\n"
            "- меняй только agent_platform/, tests/ или vault/agents/;\n"
            "- не открывай и не создавай .env, auth.json, OAuth/token/cookie файлы;\n"
            "- не выполняй codex login/logout и не управляй авторизацией;\n"
            "- не запускай Telegram-бота, image/video generation, PolzaAI и платные или сетевые запросы;\n"
            "- не изменяй пользовательскую память, runs/imports, исходные материалы и архивы;\n"
            "- не коммить: внешний orchestrator проверит diff, запустит тесты и создаст коммит;\n"
            "- не маскируй проблему ослаблением валидации без доказанной причины;\n"
            "- если evidence недостаточно для безопасной правки, не меняй код и прямо объясни это.\n\n"
            "После правки кратко сообщи: причина, изменённые файлы, добавленный тест, ограничения.\n\n"
            "EVIDENCE BUNDLE (секреты намеренно не передаются):\n"
            + json.dumps(evidence, ensure_ascii=False)
        )

    def _run_codex(self, worktree: Path, prompt: str) -> str:
        if self.executable is None:
            raise RepairError("Codex CLI не найден.")
        command = [
            str(self.executable),
            "exec",
            "--ephemeral",
            "--sandbox",
            "workspace-write",
            "--cd",
            str(worktree),
            "--model",
            self.model,
            "--color",
            "never",
            "--json",
            "-",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=worktree,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
                env=self._safe_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise RepairError(
                f"Codex не завершил подготовку за {self.timeout_seconds} секунд."
            ) from exc
        except OSError as exc:
            raise RepairError(f"Не удалось запустить Codex: {exc}") from exc

        answers, errors = self._parse_codex_events(completed.stdout)
        if completed.returncode != 0:
            reason = errors[-1] if errors else f"код завершения {completed.returncode}"
            raise RepairError(f"Codex не подготовил исправление: {reason}")
        return "\n\n".join(answers).strip() or "Codex подготовил изменения без текстового резюме."

    def _run_gates(self, worktree: Path) -> list[str]:
        commands = [
            (
                "Компиляция Python",
                [sys.executable, "-m", "compileall", "-q", "agent_platform", "tests"],
            ),
            (
                "Полный набор unit-тестов",
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
            ),
            ("Проверка Git diff", ["git", "diff", "--check"]),
        ]
        passed: list[str] = []
        for label, command in commands:
            try:
                result = subprocess.run(
                    command,
                    cwd=worktree,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=300,
                    check=False,
                    env=self._safe_environment(),
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RepairError(f"{label} не запущена: {exc}") from exc
            if result.returncode != 0:
                output = self._clean_text(result.stdout + "\n" + result.stderr, 1600)
                raise RepairError(f"{label} не пройдена. {output}")
            passed.append(f"✅ {label}")
        return passed

    def _validate_changed_files(self, worktree: Path, paths: list[str]) -> None:
        root = worktree.resolve()
        for path in paths:
            normalized = path.replace("\\", "/").strip("/")
            parts = {part.lower() for part in Path(normalized).parts}
            lowered = normalized.lower()
            allowed = normalized in self.ALLOWED_FILES or any(
                normalized.startswith(prefix) for prefix in self.ALLOWED_PREFIXES
            )
            forbidden = bool(parts.intersection(self.FORBIDDEN_PARTS)) or any(
                marker in lowered
                for marker in ("auth.json", "oauth", "cookie", ".env.", "/.env")
            )
            if normalized.startswith("vault/agents/") and normalized not in self.ALLOWED_AGENT_FILES:
                forbidden = True
            if normalized.startswith(("agent_platform/partner_", "tests/test_partner")):
                forbidden = True
            if not allowed or forbidden:
                raise RepairError(f"Codex попытался изменить запрещённый файл: {normalized}")
            target = (worktree / normalized).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RepairError(f"Путь вышел за границы worktree: {normalized}") from exc

    @staticmethod
    def _validate_forbidden_artifacts(worktree: Path) -> None:
        forbidden: list[str] = []
        for path in worktree.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(worktree).as_posix()
            name = path.name.lower()
            parts = {part.lower() for part in path.parts}
            if relative == ".env.example":
                continue
            if (
                name == ".env"
                or name.startswith(".env.")
                or name == "auth.json"
                or name.endswith((".pem", ".p12", ".pfx"))
                or ".codex" in parts
            ):
                forbidden.append(relative)
        if forbidden:
            raise RepairError(
                "В изолированной папке появился запрещённый секретный файл: "
                + ", ".join(sorted(forbidden)[:10])
            )

    def _changed_files(self, worktree: Path) -> list[str]:
        tracked = self._git(
            worktree, "diff", "--name-only", "--relative", "HEAD"
        ).stdout.splitlines()
        untracked = self._git(
            worktree,
            "ls-files",
            "--others",
            "--exclude-standard",
        ).stdout.splitlines()
        return sorted({item.strip().replace("\\", "/") for item in tracked + untracked if item.strip()})

    def _rollback_failed_apply(self, head_before: str, applied_commit: str) -> bool:
        cherry_pick = self._git(
            self.repository_root,
            "rev-parse",
            "--verify",
            "-q",
            "CHERRY_PICK_HEAD",
            check=False,
        )
        if cherry_pick.returncode == 0:
            return (
                self._git(
                    self.repository_root, "cherry-pick", "--abort", check=False
                ).returncode
                == 0
            )
        head = self._git(self.repository_root, "rev-parse", "HEAD", check=False).stdout.strip()
        if applied_commit and head == applied_commit and head != head_before:
            return (
                self._git(
                    self.repository_root,
                    "-c",
                    "user.name=Content Factory Repair Bot",
                    "-c",
                    "user.email=repair-bot@local.invalid",
                    "revert",
                    "--no-edit",
                    head,
                    check=False,
                ).returncode
                == 0
            )
        return head == head_before

    def _write(self, record: RepairRecord) -> RepairRecord:
        target_dir = self.content_store.run_path(record.run_id) / "maintenance"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{record.repair_id}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(target)
        return record

    def _git(
        self,
        cwd: Path,
        *args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RepairError(f"Git-команда не выполнена: {exc}") from exc
        if check and result.returncode != 0:
            reason = self._clean_text(result.stderr or result.stdout, 1000)
            raise RepairError(f"Git {' '.join(args[:2])} завершился ошибкой: {reason}")
        return result

    def _git_config_path(self, worktree: Path) -> Path:
        raw = self._git(worktree, "rev-parse", "--git-common-dir").stdout.strip()
        common = Path(raw)
        if not common.is_absolute():
            common = (worktree / common).resolve()
        return common / "config"

    @staticmethod
    def _file_hash(path: Path) -> str:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return ""

    @staticmethod
    def _safe_environment() -> dict[str, str]:
        denied = re.compile(
            r"(TOKEN|SECRET|PASSWORD|API_KEY|BOT_TOKEN|COOKIE|SESSION|TELEGRAM|POLZA|OPENAI|DEEPGRAM)",
            re.IGNORECASE,
        )
        return {key: value for key, value in os.environ.items() if not denied.search(key)}

    @staticmethod
    def _parse_codex_events(output: str) -> tuple[list[str], list[str]]:
        answers: list[str] = []
        errors: list[str] = []
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message" and item.get("text"):
                    answers.append(str(item["text"]).strip())
                elif item.get("type") == "error" and item.get("message"):
                    errors.append(str(item["message"]).strip())
            elif event.get("type") == "turn.failed":
                errors.append(str(event.get("error", {}).get("message", "")).strip())
        return answers, [item for item in errors if item]

    @staticmethod
    def _normalize_repair_id(value: str) -> str:
        normalized = value.strip().upper()
        if not re.fullmatch(r"RP-[A-F0-9]{8}", normalized):
            raise RepairError(f"Некорректный ID ремонта: {value}")
        return normalized

    @staticmethod
    def _clean_text(value: str, limit: int) -> str:
        clean = re.sub(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b", "[REDACTED]", value)
        clean = re.sub(r"\b(?:sk|sd|ghp|github_pat)_[A-Za-z0-9_-]{12,}\b", "[REDACTED]", clean)
        clean = re.sub(
            r"(?i)(api[_ -]?key|token|password|secret)\s*[:=]\s*\S+",
            r"\1=[REDACTED]",
            clean,
        )
        return clean.strip()[:limit]

    @classmethod
    def _redact_payload(cls, value: object) -> object:
        if isinstance(value, dict):
            return {str(key): cls._redact_payload(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._redact_payload(item) for item in value]
        if isinstance(value, str):
            return cls._clean_text(value, 12000)
        return value
