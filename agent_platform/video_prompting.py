from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import Settings
from .llm import LlmClient, find_codex_cli, safe_subprocess_environment
from .production import ProductionContractError, ProductionStore, SceneSpec
from .seedance_prompting import (
    MAX_SEEDANCE_REFERENCE_IMAGES,
    SEEDANCE_FINAL_LOCK,
    build_seedance_prompt,
)
from .vault import now_stamp


class ImageInspectionError(RuntimeError):
    pass


class ImageInspector(Protocol):
    @property
    def is_configured(self) -> bool:
        raise NotImplementedError

    def inspect(self, path: Path) -> str:
        raise NotImplementedError


@dataclass
class NoImageInspector:
    reason: str = "Инспектор изображений не настроен."

    @property
    def is_configured(self) -> bool:
        return False

    def inspect(self, path: Path) -> str:
        raise ImageInspectionError(self.reason)


class CodexImageInspector:
    """Inspect a local frame through installed Codex CLI without managing its OAuth."""

    def __init__(self, settings: Settings):
        self.executable = find_codex_cli(settings.codex_cli_path)
        self.model = settings.codex_chat_model
        self.timeout_seconds = settings.codex_timeout_seconds

    @property
    def is_configured(self) -> bool:
        return self.executable is not None

    def inspect(self, path: Path) -> str:
        target = path.resolve()
        if not target.is_file():
            raise ImageInspectionError(f"Кадр не найден: {path.name}")
        if self.executable is None:
            raise ImageInspectionError("Codex CLI не найден для анализа кадра.")
        prompt = (
            f"Inspect the local raster image at this exact path: {target}. Use the image viewing "
            "capability. Do not modify files and do not generate a new image. Return one compact "
            "factual paragraph in Russian describing only what is visibly present: subject identity "
            "and appearance, wardrobe, objects, composition, environment, lighting, colors, and "
            "anything that must remain stable during image-to-video animation. Do not infer brand "
            "facts or hidden actions."
        )
        command = [
            str(self.executable),
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--cd",
            str(target.parent),
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
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
                env=safe_subprocess_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise ImageInspectionError("Анализ кадра не завершился в установленный срок.") from exc
        except OSError as exc:
            raise ImageInspectionError(f"Не удалось запустить инспектор кадра: {exc}") from exc
        answer, error = self._parse_events(completed.stdout)
        if answer:
            if self.is_error_response(answer):
                raise ImageInspectionError(
                    "Codex не смог прочитать изображение: " + answer[:350]
                )
            return answer
        raise ImageInspectionError(error or "Codex не вернул описание изображения.")

    @staticmethod
    def is_error_response(text: str) -> bool:
        """Reject tool/runtime failures that arrived inside an agent-message event."""

        normalized = " ".join(str(text).lower().split())
        markers = (
            "createprocesswithlogonw",
            "winerror",
            "failed to spawn",
            "cannot inspect",
            "can't inspect",
            "не могу просмотреть",
            "не удалось открыть изображение",
            "не удалось запустить",
            "image viewing capability is unavailable",
        )
        return not normalized or any(marker in normalized for marker in markers)

    @staticmethod
    def _parse_events(output: str) -> tuple[str, str]:
        answers: list[str] = []
        errors: list[str] = []
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message" and str(item.get("text", "")).strip():
                    answers.append(str(item["text"]).strip())
                elif item.get("type") == "error":
                    errors.append(str(item.get("message", "")).strip())
            elif event.get("type") in {"turn.failed", "error"}:
                errors.append(str(event.get("message", "")).strip())
        return (answers[-1] if answers else "", errors[-1][:500] if errors else "")


MODEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    # Confirmed in PolzaAI's official Kling 3 guide on 2026-07-13.
    "kling/v3": {
        "duration_min": 3,
        "duration_max": 15,
        "aspect_ratios": {"16:9", "9:16", "1:1"},
        "sound": True,
        "image_to_video": True,
        "modes": {"std", "pro", "4K"},
        "max_prompt_chars": 2500,
    },
    # Confirmed in PolzaAI's official Seedance 2 guide and model card on 2026-07-15.
    "bytedance/seedance-2": {
        "duration_values": {5, 10, 15},
        "aspect_ratios": {"16:9", "4:3", "1:1", "3:4", "9:16", "21:9"},
        "sound": True,
        "image_to_video": True,
        "multi_shots": False,
        "max_prompt_chars": 5000,
    },
    # Fixed pre-benchmark contract from the official LTX-2.3 distilled I2V pipeline.
    "ltx-2.3": {
        "duration_values": {5},
        "aspect_ratios": {"16:9"},
        "sound": True,
        "image_to_video": True,
        "max_prompt_chars": 5000,
    },
}


class VideoPromptBuilder:
    """Build provider-ready prompts for continuous logical clips of selected frames."""

    def __init__(
        self,
        store: ProductionStore,
        inspector: ImageInspector,
        prompt_llm: LlmClient | None = None,
    ):
        self.store = store
        self.inspector = inspector
        self.prompt_llm = prompt_llm

    def build(
        self,
        run_id: str,
        *,
        model_id: str,
        duration_seconds: int,
        aspect_ratio: str,
        sound_enabled: bool,
        provider_name: str = "polza",
        on_progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        capabilities = MODEL_CAPABILITIES.get(model_id)
        warnings: list[str] = []
        if capabilities:
            duration_values = capabilities.get("duration_values")
            if duration_values and duration_seconds not in duration_values:
                allowed = ", ".join(str(value) for value in sorted(duration_values))
                raise ProductionContractError(
                    f"{model_id}: допустимая длительность: {allowed} секунд."
                )
            if not duration_values and not (
                capabilities["duration_min"]
                <= duration_seconds
                <= capabilities["duration_max"]
            ):
                raise ProductionContractError(
                    f"{model_id}: длительность должна быть от {capabilities['duration_min']} "
                    f"до {capabilities['duration_max']} секунд."
                )
            if aspect_ratio not in capabilities["aspect_ratios"]:
                raise ProductionContractError(f"{model_id}: неподдерживаемое соотношение {aspect_ratio}.")
            if sound_enabled and not capabilities["sound"]:
                raise ProductionContractError(f"{model_id}: звук не подтверждён документацией.")
        else:
            warnings.append("Возможности модели не зафиксированы: параметры требуют ручной проверки.")

        state = self.store.load(run_id)
        selected = list(state.get("selected_frame_ids", []))
        if not selected:
            raise ProductionContractError("Не выбрано ни одного готового кадра.")
        scene_map = {scene.scene_id: scene for scene in self.store.scenes(run_id)}
        use_multireference = bool(
            provider_name == "kie" and model_id.startswith("bytedance/seedance-2")
        )
        clips = self._plan_clips(
            selected,
            scene_map,
            duration_seconds,
            allow_multireference=use_multireference,
        )
        prompts: list[dict[str, Any]] = []
        inspection_cache: dict[Path, dict[str, str]] = {}
        selected_positions = {scene_id: index for index, scene_id in enumerate(selected)}
        for index, clip_scenes in enumerate(clips):
            clip_id = f"V{index + 1:02d}"
            source_scene_ids = [scene.scene_id for scene in clip_scenes]
            start_scene = clip_scenes[0]
            start_frame_path = self.store.frame_path(run_id, start_scene.scene_id)
            if start_frame_path is None:
                raise ProductionContractError(
                    f"Стартовый кадр {start_scene.scene_id} для {clip_id} не найден на диске."
                )
            frame_evidence: list[dict[str, Any]] = []
            for scene in clip_scenes:
                frame_path = self.store.frame_path(run_id, scene.scene_id)
                if frame_path is None:
                    raise ProductionContractError(
                        f"Выбранный кадр {scene.scene_id} не найден на диске."
                    )
                observed = self._inspect_or_fallback(
                    frame_path,
                    inspection_cache,
                    fallback=self._frame_contract_description(scene),
                )
                frame_evidence.append(
                    {
                        "scene_id": scene.scene_id,
                        "file": str(frame_path),
                        **observed,
                    }
                )
            reference_inputs = self._reference_evidence(
                run_id,
                state,
                clip_scenes,
                inspection_cache,
            )
            reference_manifest, omitted_references = self._build_reference_manifest(
                frame_evidence,
                reference_inputs,
                use_multireference=use_multireference,
            )
            first_position = selected_positions[start_scene.scene_id]
            last_position = selected_positions[clip_scenes[-1].scene_id]
            previous_scene = scene_map.get(selected[first_position - 1]) if first_position > 0 else None
            next_scene = (
                scene_map.get(selected[last_position + 1])
                if last_position + 1 < len(selected)
                else None
            )
            timeline = self._build_timeline(clip_scenes, duration_seconds)
            self._attach_timeline_reference_tags(timeline, reference_manifest)
            universal = self._universal_prompt(
                clip_id,
                clip_scenes,
                frame_evidence,
                previous_scene,
                next_scene,
                duration_seconds,
                aspect_ratio,
                sound_enabled,
                reference_inputs,
                reference_manifest,
                timeline,
            )
            summary_ru = self._clip_summary_ru(clip_scenes)
            if model_id == "bytedance/seedance-2":
                if self.prompt_llm is None:
                    raise ProductionContractError(
                        "Seedance 2 требует GPT-промптер, работающий по seedance_guide (2)."
                    )
                seedance = build_seedance_prompt(
                    self.prompt_llm,
                    scene_id=clip_id,
                    universal_prompt=universal,
                    duration_seconds=duration_seconds,
                    aspect_ratio=aspect_ratio,
                    sound_enabled=sound_enabled,
                    reference_manifest=reference_manifest,
                )
                summary_ru = seedance.summary_ru
                model_prompt = seedance.prompt_zh
            else:
                model_prompt = self._adapt_to_model(model_id, universal)
            item_warnings = list(warnings)
            fallback_count = sum(
                item.get("inspection_status") == "contract_fallback"
                for item in [*frame_evidence, *reference_inputs]
            )
            if fallback_count:
                item_warnings.append(
                    f"Визуальный инспектор недоступен для {fallback_count} входов; "
                    "использованы проверенные описания из производственного контракта."
                )
            if omitted_references:
                item_warnings.append(
                    "Отдельные карточки не прикреплены, потому что их внешний вид уже "
                    "зафиксирован сценическими кадрами: "
                    + ", ".join(item["reference_id"] for item in omitted_references)
                    + "."
                )
            if len(clip_scenes) > 1 and not use_multireference:
                item_warnings.append(
                    "Провайдер получает только стартовый кадр; остальные кадры клипа "
                    "использованы для планирования таймлайна и проверки continuity."
                )
            max_prompt_chars = int(capabilities.get("max_prompt_chars", 0)) if capabilities else 0
            if max_prompt_chars and len(model_prompt) > max_prompt_chars:
                raise ProductionContractError(
                    f"{clip_id}: видеопромпт длиннее лимита модели "
                    f"({len(model_prompt)} > {max_prompt_chars})."
                )
            provider_frame_ids = [
                str(item["scene_ids"][0])
                for item in reference_manifest
                if item.get("kind") in {"start_frame", "continuation_frame"}
                and item.get("scene_ids")
            ]
            item = {
                "clip_id": clip_id,
                # scene_id is retained as a compatibility alias for existing job storage.
                "scene_id": clip_id,
                "source_scene_ids": source_scene_ids,
                "source_frame_ids": source_scene_ids,
                "start_scene_id": start_scene.scene_id,
                "location_id": start_scene.location_id,
                "frame_file": str(start_frame_path),
                "source_frame_files": [item["file"] for item in frame_evidence],
                "frame_evidence": frame_evidence,
                "image_description": frame_evidence[0]["description"],
                "reference_inputs": reference_inputs,
                "reference_manifest": reference_manifest,
                "provider_reference_files": [
                    str(item["file"]) for item in reference_manifest
                ],
                "provider_reference_count": len(reference_manifest),
                "provider_input_frame_ids": provider_frame_ids,
                "planning_only_frame_ids": [
                    scene_id
                    for scene_id in source_scene_ids
                    if scene_id not in provider_frame_ids
                ],
                "omitted_reference_inputs": omitted_references,
                "timeline": timeline,
                "universal_prompt": universal,
                "model_id": model_id,
                "provider": provider_name,
                "model_prompt": model_prompt,
                "prompt_summary_ru": summary_ru,
                "prompt_guide": (
                    "seedance_guide (2)" if model_id == "bytedance/seedance-2" else "model adapter"
                ),
                "duration_seconds": duration_seconds,
                "aspect_ratio": aspect_ratio,
                "sound_enabled": sound_enabled,
                "warnings": item_warnings,
            }
            prompts.append(item)
            if on_progress is not None:
                on_progress(index + 1, len(clips), item)
        self.store.save_video_prompts(run_id, prompts)
        self.store.save_video_prompt_qa(
            run_id,
            self._build_structural_qa(prompts, selected, scene_map, duration_seconds),
        )
        return prompts

    @staticmethod
    def _build_structural_qa(
        prompts: list[dict[str, Any]],
        selected: list[str],
        scene_map: dict[str, SceneSpec],
        duration_seconds: int,
    ) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        covered = [
            str(scene_id)
            for prompt in prompts
            for scene_id in prompt.get("source_scene_ids", [])
        ]
        if covered != selected:
            errors.append("Выбранные кадры покрыты не полностью или не по порядку.")
        for prompt in prompts:
            clip_id = str(prompt.get("clip_id", "видеоклип"))
            source_ids = [str(item) for item in prompt.get("source_scene_ids", [])]
            source_scenes = [scene_map[item] for item in source_ids if item in scene_map]
            locations = {scene.location_id for scene in source_scenes if scene.location_id}
            if len(source_scenes) != len(source_ids):
                errors.append(f"{clip_id}: не все исходные сцены найдены в контракте.")
            if len(source_scenes) > 1 and (not locations or len(locations) != 1):
                errors.append(f"{clip_id}: объединены кадры из разных или неизвестных локаций.")
            for previous, current in zip(source_scenes, source_scenes[1:]):
                if not VideoPromptBuilder._is_continuous_transition(previous.transition):
                    errors.append(
                        f"{clip_id}: {previous.scene_id} и {current.scene_id} не описаны "
                        "как один непрерывный кадр без склейки."
                    )
                if previous.reference_ids or current.reference_ids:
                    if not set(previous.reference_ids) & set(current.reference_ids):
                        errors.append(
                            f"{clip_id}: {previous.scene_id} и {current.scene_id} не имеют "
                            "общего визуального якоря для непрерывного действия."
                        )
            manifest = prompt.get("reference_manifest", [])
            if not isinstance(manifest, list) or not manifest:
                errors.append(f"{clip_id}: отсутствует карта референсов @ImageN.")
                manifest = []
            expected_tags = [
                f"@Image{index}" for index in range(1, len(manifest) + 1)
            ]
            actual_tags = [
                str(item.get("tag", ""))
                for item in manifest
                if isinstance(item, dict)
            ]
            if actual_tags != expected_tags:
                errors.append(f"{clip_id}: нарушен порядок тегов @ImageN.")
            try:
                provider_reference_count = int(
                    prompt.get("provider_reference_count", len(manifest))
                )
            except (TypeError, ValueError):
                provider_reference_count = -1
            if provider_reference_count != len(manifest):
                errors.append(
                    f"{clip_id}: число provider reference-файлов не совпадает с картой @ImageN."
                )
            allowed_roles = {
                "start_frame",
                "continuation_frame",
                "character",
                "wardrobe",
                "prop",
                "environment",
                "camera",
                "style",
                "additional_visual_reference",
            }
            source_id_set = set(source_ids)
            for item in manifest:
                if not isinstance(item, dict):
                    continue
                tag = str(item.get("tag", "reference"))
                role = str(item.get("role", ""))
                label = str(item.get("label", "")).strip()
                usage = str(item.get("usage", "")).strip()
                description = str(item.get("description", "")).strip()
                item_scene_ids = {
                    str(value) for value in item.get("scene_ids", []) if str(value).strip()
                }
                if role not in allowed_roles:
                    errors.append(f"{clip_id}: {tag} имеет неизвестную смысловую роль.")
                if not label or not usage or not description:
                    errors.append(
                        f"{clip_id}: {tag} не содержит полную подпись, роль и назначение."
                    )
                if role not in {"start_frame", "continuation_frame"} and not (
                    item_scene_ids & source_id_set
                ):
                    errors.append(
                        f"{clip_id}: {tag} прикреплён, но не используется ни одной сценой клипа."
                    )
            files = [
                Path(str(item.get("file", "")))
                for item in manifest
                if isinstance(item, dict)
            ]
            if len(files) != len(manifest) or any(not path.is_file() for path in files):
                errors.append(f"{clip_id}: один или несколько файлов референсов отсутствуют.")
            resolved_files = [str(path.resolve()) for path in files if path.is_file()]
            if len(resolved_files) != len(set(resolved_files)):
                errors.append(f"{clip_id}: один файл прикреплён под несколькими тегами.")
            stored_files = [
                str(value) for value in prompt.get("provider_reference_files", [])
            ]
            if stored_files != [str(path) for path in files]:
                errors.append(
                    f"{clip_id}: порядок файлов провайдера расходится с картой @ImageN."
                )
            model_prompt = str(prompt.get("model_prompt", ""))
            prompt_tags = {
                f"@Image{value}" for value in re.findall(r"@Image([1-9]\d*)", model_prompt)
            }
            if prompt_tags != set(expected_tags):
                errors.append(
                    f"{clip_id}: теги в тексте промпта не совпадают с прикреплёнными файлами."
                )
            universal_prompt = str(prompt.get("universal_prompt", ""))
            if "Temporal, visual and physical continuity contract:" not in universal_prompt:
                errors.append(f"{clip_id}: отсутствует полный continuity contract.")
            if "the final frame holds the reached final state" not in universal_prompt:
                errors.append(f"{clip_id}: не зафиксировано конечное состояние сцены.")
            if str(prompt.get("model_id", "")) == "bytedance/seedance-2" and not model_prompt.endswith(
                SEEDANCE_FINAL_LOCK
            ):
                errors.append(f"{clip_id}: Seedance prompt не содержит финальный continuity lock.")
            accounted_references: set[str] = set()
            for collection_name in ("reference_manifest", "omitted_reference_inputs"):
                collection = prompt.get(collection_name, [])
                if not isinstance(collection, list):
                    continue
                for item in collection:
                    if not isinstance(item, dict):
                        continue
                    reference_id = str(item.get("reference_id", "")).strip()
                    if reference_id and not reference_id.startswith("FRAME-"):
                        accounted_references.add(reference_id)
                    accounted_references.update(
                        str(alias).strip()
                        for alias in item.get("aliases", [])
                        if str(alias).strip()
                    )
            required_references = {
                reference_id
                for scene in source_scenes
                for reference_id in scene.reference_ids
            }
            missing = sorted(required_references - accounted_references)
            if missing:
                errors.append(
                    f"{clip_id}: не учтены референсы {', '.join(missing)}."
                )
            timeline = prompt.get("timeline", [])
            if not isinstance(timeline, list) or len(timeline) != len(source_ids):
                errors.append(f"{clip_id}: таймлайн не соответствует исходным кадрам.")
            elif timeline:
                if float(timeline[0].get("start_seconds", -1)) != 0:
                    errors.append(f"{clip_id}: таймлайн начинается не с нулевой секунды.")
                if float(timeline[-1].get("end_seconds", -1)) != float(duration_seconds):
                    errors.append(f"{clip_id}: таймлайн не закрывает всю длительность клипа.")
                for previous, current in zip(timeline, timeline[1:]):
                    if float(previous.get("end_seconds", -1)) != float(
                        current.get("start_seconds", -2)
                    ):
                        errors.append(f"{clip_id}: в таймлайне найден разрыв или наложение.")
                        break
                timeline_tags = {
                    str(tag)
                    for beat in timeline
                    if isinstance(beat, dict)
                    for tag in beat.get("reference_tags", [])
                }
                if not set(expected_tags).issubset(timeline_tags):
                    errors.append(
                        f"{clip_id}: не все @ImageN привязаны к действиям таймлайна."
                    )
                semantic_timeline_tags: set[str] = set()
                for beat in timeline:
                    uses = beat.get("reference_uses", []) if isinstance(beat, dict) else []
                    if not isinstance(uses, list):
                        errors.append(f"{clip_id}: повреждена смысловая карта таймлайна.")
                        continue
                    context = str(beat.get("reference_context", ""))
                    if re.search(r"@Image[1-9]\d*\s+@Image[1-9]\d*", context):
                        errors.append(
                            f"{clip_id}: в таймлайне найден бессмысленный список @ImageN."
                        )
                    for use in uses:
                        if not isinstance(use, dict):
                            continue
                        tag = str(use.get("tag", ""))
                        if not all(
                            str(use.get(key, "")).strip()
                            for key in ("tag", "role", "label", "usage")
                        ):
                            errors.append(
                                f"{clip_id}: {tag or 'reference'} не связан с объектом действия."
                            )
                        semantic_timeline_tags.add(tag)
                if not set(expected_tags).issubset(semantic_timeline_tags):
                    errors.append(
                        f"{clip_id}: не все @ImageN имеют смысловую роль внутри таймлайна."
                    )
            fallback_items = [
                item
                for collection_name in ("frame_evidence", "reference_inputs")
                for item in prompt.get(collection_name, [])
                if isinstance(item, dict)
                and item.get("inspection_status") == "contract_fallback"
            ]
            if fallback_items:
                warnings.append(
                    f"{clip_id}: визуальная проверка недоступна для "
                    f"{len(fallback_items)} входов; использован контракт."
                )
            omitted = prompt.get("omitted_reference_inputs", [])
            if isinstance(omitted, list) and omitted:
                warnings.append(
                    f"{clip_id}: {len(omitted)} карточек не прикреплены отдельно; "
                    "они остаются зафиксированы в сценическом кадре."
                )
            if prompt.get("planning_only_frame_ids"):
                warnings.append(
                    f"{clip_id}: дополнительные кадры используются для планирования, "
                    "но API получает только стартовый кадр."
                )
        verdict = "fail" if errors else ("warning" if warnings else "pass")
        return {
            "status": "completed",
            "verdict": verdict,
            "checked_at": now_stamp(),
            "selected_frame_count": len(selected),
            "video_clip_count": len(prompts),
            "covered_scene_ids": covered,
            "errors": errors,
            "warnings": warnings,
            "ready_for_paid_generation": not errors,
            "scope": (
                "Структурная проверка ID, порядка, локаций, референсов и таймлайна. "
                "Она не доказывает художественное качество будущего видео."
            ),
        }

    @staticmethod
    def _plan_clips(
        selected: list[str],
        scene_map: dict[str, SceneSpec],
        duration_seconds: int,
        *,
        allow_multireference: bool = False,
    ) -> list[list[SceneSpec]]:
        """Group only scenes explicitly written as one continuous causal shot."""

        ordered: list[SceneSpec] = []
        for scene_id in selected:
            scene = scene_map.get(scene_id)
            if scene is None:
                raise ProductionContractError(f"Сцена {scene_id} отсутствует в контракте.")
            ordered.append(scene)
        clips: list[list[SceneSpec]] = []
        for scene in ordered:
            if not clips:
                clips.append([scene])
                continue
            current = clips[-1]
            previous = current[-1]
            same_known_location = bool(
                scene.location_id
                and previous.location_id
                and scene.location_id == previous.location_id
            )
            adjacent = scene.order == previous.order + 1
            planned_duration = sum(item.duration_seconds for item in current) + scene.duration_seconds
            # A later identity cannot appear reliably if it is absent from the only frame
            # actually uploaded to the image-to-video provider.
            references_visible_from_start = allow_multireference or set(
                scene.reference_ids
            ).issubset(set(current[0].reference_ids))
            shared_visual_anchor = bool(
                set(previous.reference_ids) & set(scene.reference_ids)
            ) or (not previous.reference_ids and not scene.reference_ids)
            causal_transition = VideoPromptBuilder._is_continuous_transition(
                previous.transition
            )
            if (
                same_known_location
                and adjacent
                and references_visible_from_start
                and shared_visual_anchor
                and causal_transition
                and planned_duration <= duration_seconds
            ):
                current.append(scene)
            else:
                clips.append([scene])
        return clips

    @staticmethod
    def _is_continuous_transition(transition: str) -> bool:
        """Accept merging only when the storyboard explicitly promises one uncut shot."""

        text = " ".join(str(transition).lower().split())
        if not text:
            return False
        explicit_no_cut = (
            "without a cut",
            "uncut",
            "без склей",
            "единым кадром",
            "одним кадром",
        )
        if any(marker in text for marker in explicit_no_cut):
            return True
        blocked = (
            "cut",
            "склей",
            "fade",
            "затемнен",
            "затемнён",
            "wipe",
            "монтаж",
            "смена кадра",
            "jump",
        )
        if any(marker in text for marker in blocked):
            return False
        continuous = (
            "continuous",
            "one shot",
            "single shot",
            "непрерыв",
            "камера продолж",
            "движение продолжа",
        )
        return any(marker in text for marker in continuous)

    @staticmethod
    def _build_timeline(
        scenes: list[SceneSpec], duration_seconds: int
    ) -> list[dict[str, Any]]:
        """Map scenario durations proportionally onto the provider clip duration."""

        total_weight = sum(max(scene.duration_seconds, 0.1) for scene in scenes)
        cursor = 0.0
        timeline: list[dict[str, Any]] = []
        for index, scene in enumerate(scenes):
            end = (
                float(duration_seconds)
                if index == len(scenes) - 1
                else round(
                    cursor
                    + duration_seconds * max(scene.duration_seconds, 0.1) / total_weight,
                    1,
                )
            )
            end = max(end, cursor + 0.1)
            timeline.append(
                {
                    "start_seconds": round(cursor, 1),
                    "end_seconds": round(min(end, float(duration_seconds)), 1),
                    "scene_id": scene.scene_id,
                    "purpose": scene.purpose,
                    "action": scene.physical_action or "subtle natural motion only",
                    "camera": scene.camera_movement or "static",
                    "entry_state": (
                        "Begin from the exact attached first frame."
                        if index == 0
                        else (
                            "Continue from the exact body, hand, prop and camera positions "
                            "reached at the end of the previous beat."
                        )
                    ),
                }
            )
            cursor = end
        timeline[-1]["end_seconds"] = float(duration_seconds)
        final_action = str(timeline[-1]["action"]).strip()
        timeline[-1]["final_state"] = (
            f"Hold the natural completed result of '{final_action}' in the final frame; "
            "every visible character and object remains in the last position reached by "
            "that action."
        )
        return timeline

    @staticmethod
    def _clip_summary_ru(scenes: list[SceneSpec]) -> str:
        purposes = [scene.purpose.strip() for scene in scenes if scene.purpose.strip()]
        if not purposes:
            return "Один непрерывный фрагмент по выбранным кадрам."
        return "Непрерывный фрагмент: " + "; затем ".join(purposes) + "."

    def _inspect_or_fallback(
        self,
        path: Path,
        cache: dict[Path, dict[str, str]],
        *,
        fallback: str,
    ) -> dict[str, str]:
        """Use real visual evidence when available, otherwise mark contract evidence."""

        target = path.resolve()
        if target not in cache:
            result = {
                "description": fallback.strip()[:1800],
                "inspection_status": "contract_fallback",
                "inspection_error": "",
            }
            if self.inspector.is_configured:
                try:
                    description = self.inspector.inspect(target).strip()
                    if CodexImageInspector.is_error_response(description):
                        raise ImageInspectionError(description)
                    result = {
                        "description": description[:1800],
                        "inspection_status": "visual",
                        "inspection_error": "",
                    }
                except (ImageInspectionError, OSError, RuntimeError) as exc:
                    result["inspection_error"] = str(exc).strip()[:350]
            else:
                result["inspection_error"] = "Визуальный инспектор не настроен."
            cache[target] = result
        return dict(cache[target])

    @staticmethod
    def _frame_contract_description(scene: SceneSpec) -> str:
        continuity = "; ".join(
            f"{key}: {value}" for key, value in scene.continuity.items() if value
        )
        return (
            f"Контракт кадра {scene.scene_id}. Визуал: {scene.visual}. "
            f"Действие: {scene.physical_action or 'только естественное микродвижение'}. "
            f"Камера: {scene.camera_movement or 'static'}. "
            f"Continuity: {continuity or 'сохранить все детали исходного кадра'}."
        )

    def _reference_evidence(
        self,
        run_id: str,
        state: dict[str, Any],
        scenes: list[SceneSpec],
        cache: dict[Path, dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Collect every planned character/object input, not only the frame's capped inputs."""

        reference_map = state.get("references", {})
        if not isinstance(reference_map, dict):
            return []
        ordered_ids = [
            reference_id
            for scene in scenes
            for reference_id in scene.reference_ids
        ]
        aliases_by_canonical: dict[str, list[str]] = {}
        scenes_by_canonical: dict[str, list[str]] = {}
        for scene in scenes:
            for reference_id in scene.reference_ids:
                canonical = self._canonical_reference_id(reference_map, reference_id)
                aliases_by_canonical.setdefault(canonical, [])
                if reference_id != canonical:
                    aliases_by_canonical[canonical].append(reference_id)
                scenes_by_canonical.setdefault(canonical, []).append(scene.scene_id)

        evidence: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_reference_id in ordered_ids:
            reference_id = self._canonical_reference_id(
                reference_map, raw_reference_id
            )
            if reference_id in seen:
                continue
            seen.add(reference_id)
            stored = reference_map.get(reference_id)
            if not isinstance(stored, dict):
                continue
            path = self.store.reference_path(run_id, reference_id)
            if path is None:
                continue
            fallback = (
                f"{reference_id} · {stored.get('name', reference_id)}. "
                f"Каноническое описание: {stored.get('prompt', '')}"
            )
            stored_description = str(stored.get("description", "")).strip()
            if stored_description and not CodexImageInspector.is_error_response(
                stored_description
            ):
                observed = {
                    "description": stored_description[:1800],
                    "inspection_status": "stored_visual",
                    "inspection_error": "",
                }
            else:
                observed = self._inspect_or_fallback(
                    path,
                    cache,
                    fallback=fallback,
                )
                if observed["inspection_status"] == "visual":
                    self.store.set_reference_description(
                        run_id, reference_id, observed["description"]
                    )
            source_kind = str(stored.get("kind", "reference")).strip().lower()
            label = str(stored.get("name", "")).strip() or reference_id
            semantic_role = self._semantic_reference_role(
                source_kind,
                label=label,
                description=(
                    str(observed.get("description", ""))
                    or str(stored.get("prompt", ""))
                ),
                reference_id=reference_id,
            )
            scene_ids = list(
                dict.fromkeys(scenes_by_canonical.get(reference_id, []))
            )
            evidence.append(
                {
                    "reference_id": reference_id,
                    "aliases": list(
                        dict.fromkeys(aliases_by_canonical.get(reference_id, []))
                    ),
                    "role": semantic_role,
                    "kind": source_kind,
                    "source_kind": source_kind,
                    "label": label,
                    "file": str(path),
                    "scene_ids": scene_ids,
                    "used_by": scene_ids,
                    "usage": self._reference_usage(semantic_role, label),
                    **observed,
                }
            )
        return evidence

    @staticmethod
    def _semantic_reference_role(
        source_kind: str,
        *,
        label: str,
        description: str,
        reference_id: str,
    ) -> str:
        """Map storage kinds to concrete visual roles without inventing missing metadata."""

        kind = str(source_kind).strip().lower()
        evidence = " ".join((label, description, reference_id)).lower()
        wardrobe_markers = (
            "wardrobe",
            "outfit",
            "clothing",
            "jacket",
            "dress",
            "shirt",
            "одежд",
            "костюм",
            "куртк",
            "плать",
            "рубаш",
        )
        camera_markers = (
            "camera angle",
            "camera view",
            "framing",
            "composition reference",
            "lens perspective",
            "ракурс",
            "кадрирован",
            "перспектив",
            "композиция камеры",
        )
        if kind == "character":
            return "character"
        if kind == "object":
            if any(marker in evidence for marker in wardrobe_markers):
                return "wardrobe"
            return "prop"
        if kind in {"style", "environment"} and any(
            marker in evidence for marker in camera_markers
        ):
            return "camera"
        if kind == "environment":
            return "environment"
        if kind == "style":
            return "style"
        return "additional_visual_reference"

    @staticmethod
    def _reference_usage(role: str, label: str) -> str:
        """Describe how one attachment constrains the generated action."""

        usage = {
            "character": (
                "preserve this character's identity, face, anatomy, proportions and wardrobe"
            ),
            "wardrobe": "preserve this exact wardrobe design, material, fit and color",
            "prop": (
                "use this exact prop when the action calls for it and preserve its design, "
                "material, scale and color"
            ),
            "environment": (
                "preserve this environment's layout, spatial anchors, background and lighting"
            ),
            "camera": "preserve this framing, perspective and camera angle",
            "style": "preserve this visual style, palette and rendering treatment",
            "additional_visual_reference": (
                "use only as an additional visual reference for its linked scene; do not infer "
                "a new character or object"
            ),
        }.get(
            role,
            "use only as an additional visual reference; do not infer a new character or object",
        )
        return f"{usage}: {label}"

    @staticmethod
    def _canonical_reference_id(
        reference_map: dict[str, Any], reference_id: str
    ) -> str:
        """Collapse transient object states while retaining real redesign variants."""

        current = reference_id
        visited: set[str] = set()
        while current not in visited:
            visited.add(current)
            item = reference_map.get(current)
            if not isinstance(item, dict):
                break
            base = str(item.get("base_reference_id", "")).strip()
            if (
                not base
                or str(item.get("kind", "")).strip() != "object"
                or not VideoPromptBuilder._is_transient_object_state(item)
            ):
                break
            current = base
        return current

    @staticmethod
    def _is_transient_object_state(item: dict[str, Any]) -> bool:
        text = " ".join(
            str(item.get(key, "")).lower()
            for key in ("name", "state_label", "prompt")
        )
        transient_markers = (
            "open",
            "closed",
            "opening",
            "closing",
            "straight",
            "spiral",
            "curly",
            "folded",
            "unfolded",
            "inflated",
            "deflated",
            "extended",
            "retracted",
            "открыт",
            "закрыт",
            "раскрыт",
            "прям",
            "спирал",
            "кудр",
            "сложен",
            "разложен",
            "надут",
            "сдут",
            "выдвинут",
            "втянут",
        )
        return any(marker in text for marker in transient_markers)

    @staticmethod
    def _build_reference_manifest(
        frame_evidence: list[dict[str, Any]],
        reference_inputs: list[dict[str, Any]],
        *,
        use_multireference: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build one ordered manifest shared by prompt text, QA and provider payload."""

        if not frame_evidence:
            raise ProductionContractError("Для видеопромпта отсутствует стартовый кадр.")
        frame_candidates: list[dict[str, Any]] = []
        for index, item in enumerate(frame_evidence):
            scene_id = str(item["scene_id"])
            is_start = index == 0
            role = "start_frame" if is_start else "continuation_frame"
            label = (
                f"Стартовый кадр {scene_id}"
                if is_start
                else f"Кадр продолжения {scene_id}"
            )
            frame_candidates.append(
                {
                    "reference_id": f"FRAME-{scene_id}",
                    "aliases": [],
                    "kind": role,
                    "source_kind": role,
                    "role": role,
                    "label": label,
                    "file": str(item["file"]),
                    "description": str(item.get("description", "")).strip()
                    or label,
                    "scene_ids": [scene_id],
                    "used_by": [scene_id],
                    "usage": (
                        "set the exact opening composition and first video frame"
                        if is_start
                        else (
                            "guide the visual target for this later beat without a cut, "
                            "teleportation or identity change"
                        )
                    ),
                    "inspection_status": str(
                        item.get("inspection_status", "contract_fallback")
                    ),
                }
            )
        semantic_roles = {
            "character",
            "wardrobe",
            "prop",
            "environment",
            "camera",
            "style",
            "additional_visual_reference",
        }
        normalized_inputs: list[dict[str, Any]] = []
        for raw in reference_inputs:
            item = dict(raw)
            reference_id = str(item.get("reference_id", "")).strip() or "REFERENCE"
            source_kind = str(
                item.get("source_kind") or item.get("kind") or item.get("role") or "reference"
            ).strip().lower()
            label = str(item.get("label", "")).strip() or reference_id
            description = str(item.get("description", "")).strip()
            role = str(item.get("role", "")).strip().lower()
            if role not in semantic_roles:
                role = VideoPromptBuilder._semantic_reference_role(
                    source_kind,
                    label=label,
                    description=description,
                    reference_id=reference_id,
                )
            raw_scene_ids = item.get("scene_ids", [])
            if isinstance(raw_scene_ids, str):
                raw_scene_ids = [raw_scene_ids]
            elif not isinstance(raw_scene_ids, (list, tuple)):
                raw_scene_ids = []
            item.update(
                {
                    "reference_id": reference_id,
                    "source_kind": source_kind,
                    "kind": source_kind,
                    "role": role,
                    "label": label,
                    "description": description or f"Additional visual reference: {label}",
                    "usage": str(item.get("usage", "")).strip()
                    or VideoPromptBuilder._reference_usage(role, label),
                    "scene_ids": [
                        str(scene_id)
                        for scene_id in raw_scene_ids
                        if str(scene_id).strip()
                    ],
                }
            )
            item["used_by"] = list(item["scene_ids"])
            normalized_inputs.append(item)
        clip_scene_ids = {
            str(item.get("scene_id", ""))
            for item in frame_evidence
            if str(item.get("scene_id", "")).strip()
        }
        eligible_inputs = [
            item
            for item in normalized_inputs
            if set(item.get("scene_ids", [])) & clip_scene_ids
        ]
        unused_inputs = [
            {
                **item,
                "omission_reason": "not_used_by_clip",
            }
            for item in normalized_inputs
            if not set(item.get("scene_ids", [])) & clip_scene_ids
        ]
        if not use_multireference:
            selected = frame_candidates[:1]
            omitted = [dict(item) for item in normalized_inputs]
        else:
            role_priority = (
                "character",
                "wardrobe",
                "prop",
                "environment",
                "camera",
                "style",
                "additional_visual_reference",
            )
            asset_candidates = [
                dict(item)
                for role in role_priority
                for item in eligible_inputs
                if item.get("role") == role
            ]
            known_assets = {
                str(item.get("reference_id", "")) for item in asset_candidates
            }
            asset_candidates.extend(
                dict(item)
                for item in eligible_inputs
                if str(item.get("reference_id", "")) not in known_assets
            )
            # Character and prop identity is harder to recover from text than a later
            # storyboard composition, so assets receive the remaining slots first.
            candidates = [frame_candidates[0], *asset_candidates, *frame_candidates[1:]]
            unique: list[dict[str, Any]] = []
            seen_paths: set[Path] = set()
            for item in candidates:
                path = Path(str(item.get("file", "")))
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                unique.append(item)
            if len(frame_candidates) > MAX_SEEDANCE_REFERENCE_IMAGES:
                raise ProductionContractError(
                    "Один клип содержит больше кадров, чем Seedance принимает референсов."
                )
            selected = unique[:MAX_SEEDANCE_REFERENCE_IMAGES]
            selected_paths = {
                Path(str(item["file"])).resolve() for item in selected
            }
            omitted_by_limit = [
                {
                    **item,
                    "omission_reason": "provider_reference_limit",
                }
                for item in eligible_inputs
                if Path(str(item.get("file", ""))).is_file()
                and Path(str(item["file"])).resolve() not in selected_paths
            ]
            omitted = [*unused_inputs, *omitted_by_limit]

        manifest: list[dict[str, Any]] = []
        for index, item in enumerate(selected, 1):
            manifest.append(
                {
                    **item,
                    "tag": f"@Image{index}",
                }
            )
        return manifest, omitted

    @staticmethod
    def _attach_timeline_reference_tags(
        timeline: list[dict[str, Any]],
        manifest: list[dict[str, Any]],
    ) -> None:
        for beat_index, beat in enumerate(timeline):
            scene_id = str(beat.get("scene_id", ""))
            relevant: list[dict[str, Any]] = []
            for item in manifest:
                role = str(item.get("role", "additional_visual_reference"))
                if role == "start_frame":
                    include = beat_index == 0
                else:
                    include = scene_id in item.get("scene_ids", [])
                if not include:
                    continue
                relevant.append(
                    {
                        "tag": str(item.get("tag", "")),
                        "reference_id": str(item.get("reference_id", "")),
                        "role": role,
                        "label": str(item.get("label", "")).strip()
                        or str(item.get("reference_id", "reference")),
                        "usage": str(item.get("usage", "")).strip()
                        or VideoPromptBuilder._reference_usage(
                            role,
                            str(item.get("label", "reference")),
                        ),
                    }
                )
            beat["reference_uses"] = relevant
            beat["reference_tags"] = [item["tag"] for item in relevant]
            beat["reference_context"] = VideoPromptBuilder._reference_context(
                relevant
            )

    @staticmethod
    def _reference_context(reference_uses: list[dict[str, Any]]) -> str:
        """Turn reference tags into grammatical constraints tied to their visual role."""

        clauses: list[str] = []
        for item in reference_uses:
            tag = str(item.get("tag", ""))
            label = str(item.get("label", "reference"))
            role = str(item.get("role", "additional_visual_reference"))
            if role == "start_frame":
                clause = f"begin exactly from {tag} ({label})"
            elif role == "continuation_frame":
                clause = (
                    f"move continuously toward the composition in {tag} ({label}) "
                    "without a cut or teleportation"
                )
            elif role == "character":
                clause = f"use {tag} ({label}) for the acting character's exact identity"
            elif role == "wardrobe":
                clause = f"keep the acting character's wardrobe identical to {tag} ({label})"
            elif role == "prop":
                clause = (
                    f"use {tag} ({label}) as the exact prop involved in this action and "
                    "keep it persistent"
                )
            elif role == "environment":
                clause = f"keep the environment and spatial layout from {tag} ({label})"
            elif role == "camera":
                clause = f"keep the framing and perspective from {tag} ({label})"
            elif role == "style":
                clause = f"keep the visual treatment from {tag} ({label})"
            else:
                clause = (
                    f"use {tag} ({label}) only as the linked additional visual reference"
                )
            clauses.append(clause)
        return "; ".join(clauses) or "use only the visible evidence in the attached start frame"

    @staticmethod
    def _universal_prompt(
        clip_id: str,
        scenes: list[SceneSpec],
        frame_evidence: list[dict[str, Any]],
        previous_scene: SceneSpec | None,
        next_scene: SceneSpec | None,
        duration: int,
        aspect_ratio: str,
        sound_enabled: bool,
        reference_inputs: list[dict[str, Any]],
        reference_manifest: list[dict[str, Any]],
        timeline: list[dict[str, Any]],
    ) -> str:
        continuity_entries: list[str] = []
        for scene in scenes:
            continuity_entries.extend(
                f"{key}: {value}" for key, value in scene.continuity.items() if value
            )
        continuity = "; ".join(dict.fromkeys(continuity_entries)) or (
            "preserve every visible identity and design detail from the source frame"
        )
        context = []
        if previous_scene:
            context.append(f"Previous scene function: {previous_scene.purpose or previous_scene.visual}")
        if next_scene:
            context.append(f"Next scene function: {next_scene.purpose or next_scene.visual}")
        requested_sounds = "; ".join(
            dict.fromkeys(scene.sound for scene in scenes if scene.sound)
        )
        sound = requested_sounds if sound_enabled and requested_sounds else (
            "preserve natural ambient sound only" if sound_enabled else "no generated audio"
        )
        reference_locks = "; ".join(
            f"{item['reference_id']} ({item['role']}): {item['description'][:350]}"
            for item in reference_inputs
            if item.get("description")
        ) or "no separate character or location card was used"
        observed_frames = "; ".join(
            f"{item['scene_id']}: {item['description']}" for item in frame_evidence
        )
        manifest_text = "; ".join(
            f"{item['tag']} [{item.get('role', 'additional_visual_reference')}] "
            f"{item.get('label') or item['reference_id']}: "
            f"{str(item.get('usage', 'additional visual reference'))[:240]}; "
            f"visual evidence: {str(item.get('description', ''))[:220]}"
            for item in reference_manifest
        )
        timeline_text = " ".join(
            (
                f"[{VideoPromptBuilder._time_label(float(item['start_seconds']))}-"
                f"{VideoPromptBuilder._time_label(float(item['end_seconds']))}] "
                f"{item['scene_id']}: {item.get('entry_state', '')} "
                f"Reference use: {item.get('reference_context', '')}. "
                f"Action: {item['action']}; camera: {item['camera']}. "
                f"{item.get('final_state', '')}"
            )
            for item in timeline
        )
        transitions = "; ".join(
            f"{scene.scene_id}: {scene.transition or 'continuous transition'}"
            for scene in scenes
        )
        return (
            f"One continuous clip {clip_id}, {duration} seconds, {aspect_ratio}; source frames: "
            f"{', '.join(scene.scene_id for scene in scenes)}. Visual evidence: {observed_frames}. "
            f"Semantic reference map: {manifest_text}. @Image1 is the exact first frame. Each other "
            "@ImageN has one stable role and appears only with its actual person, prop, environment, "
            "framing or style. Never invent, renumber or list tags without an action relationship. "
            f"Timeline: {timeline_text} Keep each beat physically coherent and use only the stated "
            "main action and camera movement. Environment motion must remain subtle and supported "
            "by visible elements. "
            f"Continuity locks: {continuity}. Do not change face, body, wardrobe, colors, materials, "
            f"objects, spatial layout, environment, or lighting identity. Audio: {sound}. "
            f"Canonical visual references used to build this frame: {reference_locks}. "
            f"Transition intent by source scene: {transitions}. "
            + (" ".join(context) + ". " if context else "")
            + "Animate open/close, bend/straighten, fold/unfold and similar state changes as actions "
            "on the same referenced object, never as a duplicate identity. "
            "Temporal, visual and physical continuity contract: objects persist until visibly placed, "
            "transferred, removed or naturally out of frame; motion continues from the prior body, "
            "hand, prop and camera positions; identity, face, anatomy, fingers, wardrobe, prop shape, "
            "material, scale and color stay stable; hand contact is correct with no clipping; weight, "
            "inertia and surface contact remain realistic; camera follows one smooth motivated path; "
            "this is one causal episode and the final frame holds the reached final state. Exclude "
            "teleportation, morphing, disappearance, duplication, identity change, broken anatomy, "
            "sudden transitions, inconsistent lighting/background, unexplained movement, discontinuous "
            "motion, extra people or objects, invented text, cuts, angle jumps and invented events."
        )

    @staticmethod
    def _time_label(seconds: float) -> str:
        whole = int(seconds)
        fraction = seconds - whole
        return f"0:{whole:02d}" if fraction < 0.05 else f"0:{seconds:04.1f}"

    @staticmethod
    def _adapt_to_model(model_id: str, universal: str) -> str:
        if model_id == "kling/v3":
            return (
                "Kling 3 image-to-video; the supplied start frame is the exact first frame. "
                + universal
            )
        if model_id == "ltx-2.3":
            return (
                "LTX-2.3 distilled image-to-video with synchronized audio; the supplied image "
                "is the exact first frame. "
                + universal
            )
        return universal


def create_image_inspector(settings: Settings) -> ImageInspector:
    inspector = CodexImageInspector(settings)
    return inspector if inspector.is_configured else NoImageInspector()
