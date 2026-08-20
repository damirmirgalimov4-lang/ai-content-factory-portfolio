from __future__ import annotations

import json
import re
from typing import Any

from .production import (
    ProductionContractError,
    extract_json_contract,
    parse_scene_contract,
)


def present_stage(stage_key: str, artifact: str) -> str:
    """Turn durable machine-rich artifacts into readable Telegram stage reports."""

    source = artifact.strip()
    if stage_key == "script":
        return _present_script(source)
    if stage_key == "storyboard":
        return _present_storyboard(source)
    if stage_key == "prompts":
        return _present_image_prompts(source)
    if stage_key in {"brief", "qa"}:
        return _compact_text(source, 1800)
    return source


def present_video_prompts(
    run_id: str,
    prompts: dict[str, Any],
    qa_report: dict[str, Any] | None = None,
) -> str:
    """Show logical clips, their evidence and exact provider prompts without state JSON."""

    blocks = [f"# Видеопромпты · {run_id}"]
    ordered = sorted(
        prompts.items(),
        key=lambda item: _scene_sort_key(item[0]),
    )
    for clip_id, raw in ordered:
        item = raw if isinstance(raw, dict) else {}
        model = str(item.get("model_id", "не указана"))
        duration = str(item.get("duration_seconds", "?"))
        aspect_ratio = str(item.get("aspect_ratio", "?"))
        sound = "да" if item.get("sound_enabled") else "нет"
        prompt = str(item.get("model_prompt", "")).strip() or "Промпт отсутствует."
        summary_ru = str(item.get("prompt_summary_ru", "")).strip()
        source_ids = [str(value) for value in item.get("source_scene_ids", [])]
        manifest = item.get("reference_manifest", [])
        reference_labels = []
        for reference in manifest:
            if not isinstance(reference, dict) or not reference.get("tag"):
                continue
            role = str(reference.get("role", "additional_visual_reference"))
            label = str(
                reference.get("label") or reference.get("reference_id") or "референс"
            )
            scenes = ", ".join(str(value) for value in reference.get("scene_ids", []))
            reference_labels.append(
                f"- **{reference.get('tag')} — {label}** · {_reference_role_ru(role)}"
                + (f" · {scenes}" if scenes else "")
            )
        timeline_lines = []
        for beat in item.get("timeline", []):
            if not isinstance(beat, dict):
                continue
            uses = []
            for reference in beat.get("reference_uses", []):
                if not isinstance(reference, dict):
                    continue
                uses.append(
                    f"{reference.get('tag')} — {reference.get('label')} "
                    f"({_reference_role_ru(str(reference.get('role', '')))})"
                )
            reference_line = (
                "\n  Референсы: " + "; ".join(uses)
                if uses
                else ""
            )
            timeline_lines.append(
                f"- [{_seconds_label(beat.get('start_seconds'))}–"
                f"{_seconds_label(beat.get('end_seconds'))}] "
                f"{beat.get('action')}\n  Камера: {beat.get('camera')}"
                f"{reference_line}"
                + (
                    f"\n  Финал: {beat.get('final_state')}"
                    if beat.get("final_state")
                    else ""
                )
            )
        blocks.append(
            "\n".join(
                [
                    f"## {clip_id} · {duration} сек. · {', '.join(source_ids) or 'без сцен'}",
                    f"**Модель:** {model} · {aspect_ratio} · звук: {sound}",
                    "**Референсы:**",
                    *(reference_labels or ["- @Image1: стартовый кадр"]),
                    "**Таймлайн:**",
                    *(timeline_lines or ["- Таймлайн отсутствует."]),
                    *([f"**Суть:** {summary_ru}"] if summary_ru else []),
                    "**Точный промпт модели:**",
                    prompt,
                ]
            )
        )
    if isinstance(qa_report, dict) and qa_report.get("status") == "completed":
        verdict = str(qa_report.get("verdict", "unknown")).upper()
        errors = qa_report.get("errors", [])
        warnings = qa_report.get("warnings", [])
        blocks.append(
            "\n".join(
                [
                    f"## Проверка · {verdict}",
                    f"Кадров: {qa_report.get('selected_frame_count', '?')} · "
                    f"Клипов: {qa_report.get('video_clip_count', '?')}",
                    *(["**Ошибки:** " + "; ".join(str(item) for item in errors)] if errors else []),
                    *(
                        ["**Предупреждения:** " + "; ".join(str(item) for item in warnings)]
                        if warnings
                        else []
                    ),
                ]
            )
        )
    return "\n\n".join(blocks)


def _reference_role_ru(role: str) -> str:
    return {
        "start_frame": "первый кадр",
        "continuation_frame": "кадр продолжения",
        "character": "персонаж",
        "wardrobe": "одежда",
        "prop": "предмет",
        "environment": "окружение",
        "camera": "ракурс камеры",
        "style": "визуальный стиль",
        "additional_visual_reference": "дополнительный визуальный референс",
    }.get(role, "дополнительный визуальный референс")


def _seconds_label(raw: Any) -> str:
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return "?:??"
    whole = int(seconds)
    if abs(seconds - whole) < 0.05:
        return f"0:{whole:02d}"
    return f"0:{seconds:04.1f}"


def _present_script(source: str) -> str:
    human = _before_contract(source, "SCENE_CONTRACT")
    try:
        scenes = parse_scene_contract(source)
    except ProductionContractError:
        return _compact_text(human or source, 1800)
    lines = [
        (
            f"- **{scene.scene_id} · {scene.duration_seconds:g} сек.** "
            f"{scene.purpose}: {scene.physical_action or scene.visual}"
        )
        for scene in scenes
    ]
    intro = _compact_text(human, 500) if human else ""
    body = "\n".join(
        [
            f"**Сцен в производстве:** {len(scenes)}",
            *lines,
            "Полная служебная структура сохранена в файле запуска.",
        ]
    )
    return f"{intro}\n\n{body}" if intro else body


def _present_storyboard(source: str) -> str:
    human = _before_contract(source, "VISUAL_BIBLE_CONTRACT")
    try:
        payload = extract_json_contract(source, "VISUAL_BIBLE_CONTRACT")
    except ProductionContractError:
        return human or source
    assets = payload.get("assets", [])
    locations = payload.get("locations", [])
    frames = payload.get("frames", [])
    if not all(isinstance(item, list) for item in (assets, locations, frames)):
        return human or source
    characters = sum(
        isinstance(item, dict) and item.get("kind") == "character" for item in assets
    )
    objects = sum(
        isinstance(item, dict) and item.get("kind") == "object" for item in assets
    )
    frame_lines = []
    for raw in frames:
        if not isinstance(raw, dict):
            continue
        refs = ", ".join(str(value) for value in raw.get("reference_ids", []))
        frame_lines.append(
            f"- **{raw.get('scene_id')} · {raw.get('location_id')}** "
            f"{raw.get('task')} · refs: {refs or 'нет'}"
        )
    summary = "\n".join(
        [
            (
                "**Визуальная библия:** "
                f"персонажей {characters}, предметов {objects}, "
                f"локаций {len(locations)}, кадров {len(frames)}"
            ),
            *frame_lines,
        ]
    )
    intro = _compact_text(human, 400) if human else ""
    return f"{intro}\n\n{summary}" if intro else summary


def _present_image_prompts(source: str) -> str:
    human = _before_contract(source, "IMAGE_PROMPT_CONTRACT")
    try:
        payload = extract_json_contract(source, "IMAGE_PROMPT_CONTRACT")
    except ProductionContractError:
        return human or source
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list):
        return human or source

    prompts: list[tuple[str, str]] = []
    for item in raw_scenes:
        if not isinstance(item, dict):
            continue
        scene_id = str(item.get("scene_id", "")).strip().upper()
        prompt = str(item.get("image_prompt", "")).strip()
        if scene_id and prompt:
            prompts.append((scene_id, prompt))
    prompts.sort(key=lambda item: _scene_sort_key(item[0]))

    raw_references = payload.get("references", [])
    raw_locations = payload.get("locations", [])
    blocks: list[str] = []
    if isinstance(raw_references, list):
        character_count = sum(
            isinstance(item, dict) and item.get("kind") == "character"
            for item in raw_references
        )
        object_count = sum(
            isinstance(item, dict) and item.get("kind") == "object"
            for item in raw_references
        )
        additional_count = sum(
            isinstance(item, dict) and item.get("kind") in {"environment", "style"}
            for item in raw_references
        )
        blocks.append(
            "# План генерации\n"
            f"**Персонажи:** {character_count}\n"
            f"**Важные предметы:** {object_count}\n"
            f"**Локации в маршруте:** {len(raw_locations) if isinstance(raw_locations, list) else 0}\n"
            f"**Дополнительные environment/style references:** {additional_count}\n"
            f"**Кадры:** {len(prompts)}\n"
            f"**Всего изображений:** {len(raw_references) + len(prompts)}"
        )
    if isinstance(raw_references, list) and raw_references:
        blocks.append(f"# Канонические референсы · {len(raw_references)}")
        for raw in raw_references:
            if not isinstance(raw, dict):
                continue
            reference_id = str(raw.get("reference_id", "")).strip()
            name = str(raw.get("name", reference_id)).strip()
            kind = str(raw.get("kind", "reference")).strip()
            prompt = str(raw.get("prompt", "")).strip()
            if reference_id and prompt:
                state_label = str(raw.get("state_label", "")).strip()
                base_reference_id = str(raw.get("base_reference_id", "")).strip()
                state_note = f"\n**Состояние:** {state_label}" if state_label else ""
                base_note = (
                    f"\n**Базовая карточка:** {base_reference_id}"
                    if base_reference_id
                    else ""
                )
                blocks.append(
                    f"## {reference_id} · {name}\n**Тип:** {kind}\n"
                    f"{state_note}{base_note}\n"
                    f"**Промпт карточки/референса:**\n{_compact_text(prompt, 420)}"
                )
    blocks.append(f"# Промпты отдельных кадров · {len(prompts)}")
    for scene_id, prompt in prompts:
        blocks.append(
            "\n".join(
                [
                    f"### Кадр {scene_id}",
                    "**Промпт, который будет отправлен генератору:**",
                    _compact_text(prompt, 420),
                ]
            )
        )
    blocks.append(
        "Каждый блок создаёт один отдельный файл. Служебный JSON сохранён только внутри запуска."
    )
    return "\n\n".join(blocks)


def _before_contract(source: str, marker: str) -> str:
    match = re.search(rf"(?im)^\s*(?:#+\s*)?{re.escape(marker)}\s*$", source)
    if match:
        return source[: match.start()].rstrip()
    # Some providers return a valid fenced contract but omit its textual marker.
    # Validation accepts that form, so presentation must hide it as well.
    for candidate in re.finditer(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        source,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        try:
            payload = json.loads(candidate.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("scenes"), list):
            return source[: candidate.start()].rstrip()
    return source


def _compact_text(source: str, limit: int) -> str:
    """Trim presentation only; the complete durable artifact remains on disk."""

    clean = re.sub(r"\n{3,}", "\n\n", source.strip())
    if len(clean) <= limit:
        return clean
    candidate = clean[:limit]
    boundaries = [
        candidate.rfind("\n\n"),
        candidate.rfind(". "),
        candidate.rfind("! "),
        candidate.rfind("? "),
    ]
    boundary = max(boundaries)
    if boundary >= max(200, limit // 2):
        candidate = candidate[: boundary + 1]
    return candidate.rstrip() + "\n\nПолная версия сохранена в файле запуска."


def _scene_sort_key(scene_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", scene_id)
    return (int(match.group(1)) if match else 10**9, scene_id)
