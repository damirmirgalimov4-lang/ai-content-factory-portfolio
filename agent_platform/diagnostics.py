from __future__ import annotations

import json
import uuid
from pathlib import Path

from .content_factory import ContentFactoryStore
from .llm import LlmClient, LlmError
from .vault import VaultStore, now_stamp


class ReadOnlyRunDiagnostic:
    """Explain one failed production stage from an explicit secret-free evidence bundle."""

    def __init__(
        self,
        content_store: ContentFactoryStore,
        vault: VaultStore,
        llm: LlmClient,
    ) -> None:
        self.content_store = content_store
        self.vault = vault
        self.llm = llm

    def diagnose(self, run_id: str) -> tuple[str, Path]:
        run = self.content_store.get_run(run_id)
        if run is None:
            raise ValueError(f"Запуск не найден: {run_id}")
        if not run.last_error:
            raise ValueError("У запуска нет сохранённой ошибки для диагностики.")
        if not self.llm.is_configured:
            raise LlmError("Codex CLI недоступен для диагностики.")

        artifacts = self.content_store.previous_artifacts(run)
        safe_artifacts = {
            key: value[:16000]
            for key, value in artifacts.items()
        }
        agent_profile = (self.vault.read_agent(run.current_stage_spec.agent) or "")[:12000]
        evidence = {
            "run": run.to_dict(),
            "failed_stage": run.current_stage,
            "agent_profile": agent_profile,
            "artifacts": safe_artifacts,
        }
        report = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Ты диагност контент-завода. Работай только по переданному evidence bundle. "
                        "Не используй инструменты, не открывай файлы, .env, OAuth, токены или настройки "
                        "авторизации. Ничего не изменяй и не утверждай, что ошибка исправлена. Объясни на "
                        "русском: 1) непосредственную причину; 2) относится ли проблема к входным данным, "
                        "контракту агента, LLM или коду; 3) что безопасно повторить из Telegram; 4) когда "
                        "потребуется открыть компьютер; 5) какие данные нельзя терять. Не выводи JSON и "
                        "техническую трассировку."
                    ),
                },
                {
                    "role": "user",
                    "content": "EVIDENCE BUNDLE:\n" + json.dumps(evidence, ensure_ascii=False),
                },
            ]
        )
        clean_report = report.strip()
        if not clean_report:
            raise LlmError("Codex вернул пустой диагностический отчёт.")
        target_dir = self.content_store.run_path(run.run_id) / "diagnostics"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"DIAG-{uuid.uuid4().hex[:8].upper()}.md"
        target.write_text(
            f"# Диагностика {run.run_id}\n\n"
            f"Дата: {now_stamp()}\n\n"
            f"Этап: {run.current_stage_spec.title}\n\n"
            f"Ошибка: {run.last_error}\n\n"
            f"{clean_report}\n",
            encoding="utf-8",
            newline="\n",
        )
        return clean_report, target
