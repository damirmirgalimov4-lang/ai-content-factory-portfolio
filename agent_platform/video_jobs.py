from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .production import ProductionContractError, ProductionStore
from .vault import now_stamp
from .video_provider import (
    VideoGenerationRequest,
    VideoProviderClient,
    VideoProviderError,
)


@dataclass(frozen=True)
class VideoGenerationPreview:
    run_id: str
    approval_id: str
    provider: str
    model: str
    mode: str
    resolution: str
    duration_seconds: int
    aspect_ratio: str
    sound_enabled: bool
    video_count: int
    estimated_cost: str
    seed: int = 0


class VideoJobManager:
    """Persist approval and task IDs before, during, and after paid submissions."""

    def __init__(
        self,
        store: ProductionStore,
        client: VideoProviderClient | dict[str, VideoProviderClient],
    ):
        self.store = store
        if isinstance(client, dict):
            self.clients = dict(client)
            self.default_provider = next(iter(self.clients), "polza")
        else:
            provider_name = str(
                getattr(client, "provider_name", "polza")
            ).strip() or "polza"
            self.clients = {provider_name: client}
            self.default_provider = provider_name

    def prepare(
        self,
        run_id: str,
        *,
        model: str,
        mode: str,
        duration_seconds: int,
        aspect_ratio: str,
        sound_enabled: bool,
        resolution: str = "720p",
        provider: str = "polza",
        seed: int = 0,
    ) -> VideoGenerationPreview:
        provider = provider.strip().lower() or self.default_provider
        client = self._client_for(provider)
        if provider == "ltx" and not client.is_configured:
            raise ProductionContractError(
                "LTX-2.3 выключен feature flag или не настроен и пока недоступен."
            )
        if model == "kling/v3":
            if mode not in {"std", "pro", "4K"}:
                raise ProductionContractError("Kling 3 поддерживает режимы std, pro и 4K.")
            if not 3 <= duration_seconds <= 15:
                raise ProductionContractError("Kling 3 поддерживает длительность 3-15 секунд.")
            if aspect_ratio not in {"16:9", "9:16", "1:1"}:
                raise ProductionContractError("Kling 3 не поддерживает выбранный формат кадра.")
        elif model in {"bytedance/seedance-2", "bytedance/seedance-2-fast"}:
            if duration_seconds not in {5, 10, 15}:
                raise ProductionContractError(
                    "Seedance 2 поддерживает длительность 5, 10 или 15 секунд."
                )
            if aspect_ratio not in {"16:9", "4:3", "1:1", "3:4", "9:16", "21:9"}:
                raise ProductionContractError(
                    "Seedance 2 не поддерживает выбранный формат кадра."
                )
            if resolution not in {"480p", "720p", "1080p"}:
                raise ProductionContractError(
                    "Seedance 2 поддерживает только подтверждённые разрешения 480p, 720p и 1080p."
                )
            if provider == "kie" and resolution not in {"480p", "720p"}:
                raise ProductionContractError(
                    "Kie Seedance 2 поддерживает подтверждённые разрешения 480p и 720p."
                )
        elif model == "ltx-2.3":
            if provider != "ltx":
                raise ProductionContractError("LTX-2.3 должна использовать отдельный LTX worker.")
            if mode != "distilled":
                raise ProductionContractError("LTX-2.3 пока поддерживает только distilled workflow.")
            if (duration_seconds, aspect_ratio, resolution, sound_enabled) != (
                5,
                "16:9",
                "1024x576",
                True,
            ):
                raise ProductionContractError(
                    "LTX-2.3 до benchmark зафиксирована: 5 секунд, 16:9, 1024×576, со звуком."
                )
            if not 0 <= seed <= 2**63 - 1:
                raise ProductionContractError("Seed LTX-2.3 находится вне разрешённого диапазона.")
        elif provider == "ltx":
            raise ProductionContractError("LTX worker настроен только для LTX-2.3.")
        if provider == "kie" and model not in {
            "bytedance/seedance-2",
            "bytedance/seedance-2-fast",
        }:
            raise ProductionContractError(
                "Kie adapter контент-завода настроен только для Seedance 2."
            )
        state = self.store.load(run_id)
        selected = list(state.get("selected_frame_ids", []))
        prompts = state.get("video_prompts", {})
        if not selected or not isinstance(prompts, dict):
            raise ProductionContractError("Сначала выбери кадры и создай video prompts.")
        prompt_ids = list(prompts)
        covered = self._covered_scene_ids(prompts)
        if not prompt_ids or covered != selected:
            raise ProductionContractError(
                "Video prompts не покрывают все выбранные кадры ровно один раз и по порядку."
            )
        if provider == "ltx" and len(prompt_ids) != 1:
            raise ProductionContractError(
                "До первого benchmark LTX разрешён ровно один пятисекундный клип за подтверждение."
            )
        prompt_qa = state.get("video_prompt_qa", {})
        if isinstance(prompt_qa, dict) and prompt_qa.get("verdict") == "fail":
            raise ProductionContractError(
                "Структурная проверка видеопромтов завершилась с ошибкой; платная генерация заблокирована."
            )
        mismatched = [
            prompt_id
            for prompt_id in prompt_ids
            if str(prompts[prompt_id].get("model_id", "")) != model
        ]
        if mismatched:
            raise ProductionContractError(
                "Видеопромпты созданы для другой модели. Сначала пересоздай их для "
                f"{model}: {', '.join(mismatched)}."
            )
        approval_id = uuid.uuid4().hex[:12]
        fingerprint = self._fingerprint(
            prompt_ids,
            prompts,
            model,
            mode,
            resolution,
            duration_seconds,
            aspect_ratio,
            sound_enabled,
            provider,
            seed,
        )
        state["video_approval"] = {
            "status": "pending",
            "approval_id": approval_id,
            "request_fingerprint": fingerprint,
            "provider": provider,
            "model": model,
            "mode": mode,
            "resolution": resolution,
            "duration_seconds": duration_seconds,
            "aspect_ratio": aspect_ratio,
            "sound_enabled": sound_enabled,
            "seed": seed,
            "video_count": len(prompt_ids),
            # Polza media creation does not expose a reliable preflight quote in the used API.
            "estimated_cost": "unavailable",
            "created_at": now_stamp(),
        }
        self.store.write_state(state)
        return self.preview(run_id)

    def preview(self, run_id: str) -> VideoGenerationPreview:
        approval = self.store.load(run_id).get("video_approval", {})
        if not isinstance(approval, dict) or not approval.get("approval_id"):
            raise ProductionContractError("Предпросмотр платной генерации не подготовлен.")
        return VideoGenerationPreview(
            run_id=run_id,
            approval_id=str(approval["approval_id"]),
            provider=str(approval.get("provider") or self.default_provider),
            model=str(approval.get("model", "")),
            mode=str(approval.get("mode", "")),
            resolution=str(approval.get("resolution", "")),
            duration_seconds=int(approval.get("duration_seconds", 0)),
            aspect_ratio=str(approval.get("aspect_ratio", "")),
            sound_enabled=bool(approval.get("sound_enabled", False)),
            video_count=int(approval.get("video_count", 0)),
            estimated_cost=str(approval.get("estimated_cost", "unavailable")),
            seed=int(approval.get("seed", 0)),
        )

    def approve(self, run_id: str, approval_id: str) -> None:
        state = self.store.load(run_id)
        approval = state.get("video_approval", {})
        if approval.get("status") != "pending" or approval.get("approval_id") != approval_id:
            raise ProductionContractError("Подтверждение устарело или уже использовано.")
        approval.update(status="approved", approved_at=now_stamp())
        self.store.write_state(state)

    def cancel(self, run_id: str, approval_id: str = "") -> None:
        state = self.store.load(run_id)
        approval = state.get("video_approval", {})
        if approval_id and approval.get("approval_id") != approval_id:
            raise ProductionContractError("Отмена относится к устаревшему запросу.")
        if approval.get("status") in {"submitting", "submitted"}:
            raise ProductionContractError("Уже отправленные внешние задачи нельзя отменить локально.")
        approval.update(status="cancelled", cancelled_at=now_stamp())
        self.store.write_state(state)

    def submit_approved(
        self,
        run_id: str,
        scene_ids: list[str] | None = None,
        on_progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        state = self.store.load(run_id)
        approval = state.get("video_approval", {})
        provider_name = str(
            approval.get("provider") or self.default_provider
        ).strip().lower()
        client = self._client_for(provider_name)
        if not client.is_configured:
            raise VideoProviderError(
                f"{self._provider_label(provider_name)} API key не задан."
            )
        if approval.get("status") in {"submitted", "reconciliation_required"}:
            # Idempotent restart path: existing task IDs or unknown submissions are never recreated.
            return state
        if approval.get("status") not in {"approved", "partially_submitted"}:
            raise ProductionContractError("Платная генерация не подтверждена пользователем.")
        prompts = state.get("video_prompts", {})
        prompt_ids = list(prompts) if isinstance(prompts, dict) else []
        targets = prompt_ids if scene_ids is None else list(dict.fromkeys(scene_ids))
        if not targets or any(prompt_id not in prompt_ids for prompt_id in targets):
            raise ProductionContractError("Повтор содержит неизвестные видеоклипы.")
        jobs = state.setdefault("video_jobs", {})
        for target_index, prompt_id in enumerate(targets):
            existing = jobs.get(prompt_id, {}) if isinstance(jobs, dict) else {}
            if existing.get("external_task_id"):
                continue
            if existing.get("submission_state") in {"submitting", "unknown"}:
                # After a crash or lost response only manual reconciliation is safe.
                continue
            prompt = prompts.get(prompt_id, {}) if isinstance(prompts, dict) else {}
            source_scene_ids = self._prompt_source_scene_ids(prompt)
            start_scene_id = str(prompt.get("start_scene_id") or source_scene_ids[0]).strip().upper()
            frame_path = self.store.frame_path(run_id, start_scene_id)
            reference_image_paths = self._provider_reference_paths(prompt)
            use_kie_multireference = bool(
                provider_name == "kie"
                and str(approval.get("model", "")).startswith("bytedance/seedance-2")
                and reference_image_paths
            )
            if frame_path is None or not prompt:
                jobs[prompt_id] = {
                    "scene_id": prompt_id,
                    "clip_id": prompt_id,
                    "source_scene_ids": source_scene_ids,
                    "start_scene_id": start_scene_id,
                    "submission_state": "failed",
                    "status": "failed",
                    "error": "Кадр или video prompt отсутствует.",
                    "retry_allowed": True,
                    "updated_at": now_stamp(),
                }
                self.store.write_state(state)
                if on_progress is not None:
                    on_progress(prompt_id, dict(jobs[prompt_id]))
                continue
            request_fingerprint = self._job_fingerprint(
                prompt_id,
                prompt,
                frame_path,
                approval,
                reference_image_paths if use_kie_multireference else (),
            )
            job = {
                "scene_id": prompt_id,
                "clip_id": prompt_id,
                "source_scene_ids": source_scene_ids,
                "start_scene_id": start_scene_id,
                "submission_state": "submitting",
                "status": "submitting",
                # The LTX worker accepts a caller-selected durable job identity.
                # Commit it before the network POST so restart/retry can only poll
                # this exact job and can never create a second paid generation.
                "external_task_id": (
                    f"cf-{request_fingerprint}" if provider_name == "ltx" else ""
                ),
                "request_fingerprint": request_fingerprint,
                "submitted_once": True,
                "created_at": now_stamp(),
                "updated_at": now_stamp(),
                "result_delivered": False,
                "provider": provider_name,
                "reference_image_count": (
                    len(reference_image_paths) if use_kie_multireference else 1
                ),
            }
            jobs[prompt_id] = job
            approval["status"] = "partially_submitted"
            self.store.write_state(state)
            request = VideoGenerationRequest(
                model=str(approval["model"]),
                prompt=str(prompt.get("model_prompt", "")),
                image_path=None if use_kie_multireference else frame_path,
                duration_seconds=int(approval["duration_seconds"]),
                aspect_ratio=str(approval["aspect_ratio"]),
                mode=str(approval["mode"]),
                sound_enabled=bool(approval["sound_enabled"]),
                user=f"{run_id}:{prompt_id}",
                resolution=str(approval.get("resolution", "720p")),
                provider=provider_name,
                reference_image_paths=(
                    reference_image_paths if use_kie_multireference else ()
                ),
                seed=int(approval.get("seed", 0)),
                idempotency_key=str(job["request_fingerprint"]),
            )
            try:
                task = client.create_video_task(request)
            except VideoProviderError as exc:
                job.update(
                    submission_state="unknown" if exc.ambiguous_submission else "failed",
                    status="submission_unknown" if exc.ambiguous_submission else "failed",
                    error=str(exc)[:500],
                    error_status_code=exc.status_code,
                    retry_allowed=(
                        not exc.ambiguous_submission and provider_name != "ltx"
                    ),
                    updated_at=now_stamp(),
                )
                self.store.write_state(state)
                if on_progress is not None:
                    on_progress(prompt_id, dict(job))
                if exc.status_code in {401, 403}:
                    # Authentication and permission failures affect the whole batch.
                    # Stopping here avoids repeated requests that cannot succeed.
                    for blocked_prompt_id in targets[target_index + 1:]:
                        blocked_existing = jobs.get(blocked_prompt_id, {})
                        if blocked_existing.get("external_task_id"):
                            continue
                        blocked_prompt = prompts.get(blocked_prompt_id, {})
                        blocked_sources = self._prompt_source_scene_ids(blocked_prompt)
                        jobs[blocked_prompt_id] = {
                            "scene_id": blocked_prompt_id,
                            "clip_id": blocked_prompt_id,
                            "source_scene_ids": blocked_sources,
                            "start_scene_id": str(
                                blocked_prompt.get("start_scene_id") or blocked_sources[0]
                            ),
                            "submission_state": "failed",
                            "status": "failed",
                            "error": str(exc)[:500],
                            "error_status_code": exc.status_code,
                            "retry_allowed": True,
                            "skipped_after_provider_denial": True,
                            "updated_at": now_stamp(),
                        }
                        if on_progress is not None:
                            on_progress(blocked_prompt_id, dict(jobs[blocked_prompt_id]))
                    self.store.write_state(state)
                    break
                continue
            job.update(
                submission_state="submitted",
                status=task.status,
                external_task_id=task.task_id,
                error=task.error,
                retry_allowed=False,
                updated_at=now_stamp(),
            )
            self.store.write_state(state)
            if on_progress is not None:
                on_progress(prompt_id, dict(job))
        if all(jobs.get(prompt_id, {}).get("external_task_id") for prompt_id in prompt_ids):
            approval["status"] = "submitted"
        elif any(
            jobs.get(prompt_id, {}).get("submission_state") == "unknown"
            for prompt_id in prompt_ids
        ):
            approval["status"] = "reconciliation_required"
        else:
            approval["status"] = "partially_submitted"
        self.store.write_state(state)
        return state

    def poll_existing(self, run_id: str) -> dict[str, Any]:
        state = self.store.load(run_id)
        jobs = state.get("video_jobs", {})
        if not isinstance(jobs, dict):
            return state
        run_path = self.store.runs_path / run_id
        for scene_id, job in jobs.items():
            task_id = str(job.get("external_task_id", "")).strip()
            if not task_id or job.get("status") in {"failed", "cancelled"}:
                continue
            if job.get("status") == "completed" and job.get("video_file"):
                continue
            provider_name = str(
                job.get("provider")
                or state.get("video_approval", {}).get("provider")
                or self.default_provider
            ).strip().lower()
            client = self._client_for(provider_name)
            try:
                task = client.get_task(task_id)
            except VideoProviderError as exc:
                job.update(last_poll_error=str(exc)[:500], updated_at=now_stamp())
                self.store.write_state(state)
                continue
            job.update(status=task.status, error=task.error, updated_at=now_stamp())
            if task.status == "completed":
                if not task.result_url:
                    job.update(
                        status="failed",
                        error=(
                            f"{self._provider_label(provider_name)} завершил задачу "
                            "без result URL."
                        ),
                    )
                else:
                    target = run_path / "videos" / scene_id / "result.mp4"
                    try:
                        client.download_video(task.result_url, target)
                    except VideoProviderError as exc:
                        job.update(status="download_failed", error=str(exc)[:500])
                    else:
                        job.update(
                            video_file=str(target.relative_to(run_path)).replace("\\", "/"),
                            completed_at=now_stamp(),
                        )
            self.store.write_state(state)
        return state

    def mark_delivered(self, run_id: str, scene_id: str) -> None:
        state = self.store.load(run_id)
        job = state.get("video_jobs", {}).get(scene_id)
        if not isinstance(job, dict):
            raise ValueError(f"Video job не найден: {scene_id}")
        job.update(result_delivered=True, delivered_at=now_stamp())
        self.store.write_state(state)

    def retry_failed_submission(self, run_id: str, scene_id: str) -> None:
        self.retry_failed_submissions(run_id, [scene_id])

    def retry_failed_submissions(self, run_id: str, scene_ids: list[str]) -> None:
        """Reset only explicitly approved failed submissions, never adjacent jobs."""

        state = self.store.load(run_id)
        unique_ids = list(dict.fromkeys(scene_ids))
        if not unique_ids:
            raise ProductionContractError("Не выбраны video tasks для повтора.")
        jobs = state.get("video_jobs", {})
        for scene_id in unique_ids:
            job = jobs.get(scene_id) if isinstance(jobs, dict) else None
            if not isinstance(job, dict) or not job.get("retry_allowed"):
                raise ProductionContractError(
                    f"Повтор video task {scene_id} небезопасен или не разрешён."
                )
            if job.get("external_task_id"):
                raise ProductionContractError(
                    f"У задачи {scene_id} уже есть внешний task ID; создание не повторяется."
                )
        for scene_id in unique_ids:
            state["video_jobs"].pop(scene_id, None)
        state["video_approval"]["status"] = "approved"
        self.store.write_state(state)

    @staticmethod
    def _fingerprint(
        selected: list[str],
        prompts: dict[str, Any],
        model: str,
        mode: str,
        resolution: str,
        duration: int,
        aspect: str,
        sound: bool,
        provider: str,
        seed: int,
    ) -> str:
        payload = {
            "selected": selected,
            "prompts": {item: prompts[item] for item in selected},
            "model": model,
            "mode": mode,
            "resolution": resolution,
            "duration": duration,
            "aspect": aspect,
            "sound": sound,
            "provider": provider,
            "seed": seed,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    @staticmethod
    def _prompt_source_scene_ids(prompt: dict[str, Any]) -> list[str]:
        raw = prompt.get("source_scene_ids") if isinstance(prompt, dict) else None
        if isinstance(raw, list) and raw:
            return [str(item).strip().upper() for item in raw]
        legacy = str(prompt.get("scene_id", "")).strip().upper() if isinstance(prompt, dict) else ""
        if not legacy:
            raise ProductionContractError("Видеопромпт не содержит исходных кадров.")
        return [legacy]

    @classmethod
    def _covered_scene_ids(cls, prompts: dict[str, Any]) -> list[str]:
        covered: list[str] = []
        for prompt in prompts.values():
            if not isinstance(prompt, dict):
                raise ProductionContractError("Сохранён повреждённый видеопромпт.")
            covered.extend(cls._prompt_source_scene_ids(prompt))
        return covered

    @staticmethod
    def _job_fingerprint(
        scene_id: str,
        prompt: dict[str, Any],
        frame_path: Path,
        approval: dict[str, Any],
        reference_image_paths: tuple[Path, ...] = (),
    ) -> str:
        reference_hashes = [
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in reference_image_paths
        ]
        payload = {
            "scene_id": scene_id,
            "prompt": prompt.get("model_prompt", ""),
            "frame_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
            "reference_image_sha256": reference_hashes,
            "request_fingerprint": approval.get("request_fingerprint", ""),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _provider_reference_paths(prompt: dict[str, Any]) -> tuple[Path, ...]:
        """Read the exact ordered files whose @ImageN tags were validated."""

        raw_paths = prompt.get("provider_reference_files", [])
        if not isinstance(raw_paths, list):
            raise ProductionContractError(
                "Видеопромпт содержит повреждённый список референсов."
            )
        paths = tuple(
            Path(str(value))
            for value in raw_paths
            if str(value).strip()
        )
        if paths and any(not path.is_file() for path in paths):
            missing = [path.name for path in paths if not path.is_file()]
            raise ProductionContractError(
                "Не найдены файлы Seedance-референсов: " + ", ".join(missing)
            )
        return paths

    def _client_for(self, provider: str) -> VideoProviderClient:
        try:
            return self.clients[provider]
        except KeyError as exc:
            raise VideoProviderError(
                f"Видеопровайдер не подключён: {provider or 'не указан'}."
            ) from exc

    @staticmethod
    def _provider_label(provider: str) -> str:
        return {
            "kie": "Kie",
            "polza": "PolzaAI",
            "viktor": "Viktor",
            "ltx": "LTX worker",
        }.get(provider, provider)
