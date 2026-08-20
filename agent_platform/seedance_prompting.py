from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .llm import LlmClient, LlmError


SEEDANCE_FINAL_LOCK = (
    "Continuity lock: one coherent episode; persistent objects until visibly placed, "
    "transferred, removed, or naturally out of frame; continuous motion from the prior pose; "
    "stable identity, face, anatomy, fingers, wardrobe, props, materials, scale, environment, "
    "lighting, and camera path; realistic contact, weight, and inertia; no clipping, "
    "teleportation, morphing, disappearance, duplication, identity change, broken anatomy, "
    "sudden cuts, inconsistent background, camera teleportation, unexplained movement, or "
    "discontinuous motion. Final state: keep every character and object in the last position "
    "naturally reached by the stated action."
)
MAX_SEEDANCE_REFERENCE_IMAGES = 9
REFERENCE_TAG_PATTERN = re.compile(r"@Image([1-9]\d*)")

SEEDANCE_SYSTEM_PROMPT = """Ты специализированный видеопромтер Seedance 2.

Пиши промпт только по правилам рабочего гайда seedance_guide (2):
- результат состоит из краткого описания на русском в 1-2 предложениях и нативного китайского ZH-промпта;
- китайский текст является режиссёрским рерайтом, а не дословным переводом;
- структура ZH: REFERENCE MAP -> FORMAT -> STYLE -> COLOR -> ENVIRONMENT -> TIMELINE;
- используй только переданные теги @ImageN и не меняй их нумерацию;
- @Image1 является точным исходным кадром и должен быть назначен первым кадром видео;
- остальные @ImageN являются каноническими персонажами, предметами или кадрами продолжения;
- карта каждого изображения содержит роль, подпись и назначение; не заменяй их голым списком тегов;
- в каждом временном блоке указывай релевантный @ImageN грамматически рядом с тем персонажем,
  предметом, окружением или ракурсом, который он фиксирует;
- нельзя писать подряд бессмысленную последовательность вида «@Image1 @Image2 @Image3»;
- один и тот же референс разрешено использовать в нескольких временных блоках;
- для каждого временного блока используй одно основное физическое действие и одно движение камеры;
- если универсальное описание содержит несколько последовательных блоков одной локации, сохрани их порядок и временные границы в TIMELINE;
- описывай наблюдаемую физику, а не абстрактные эмоции;
- изменение состояния предмета (открывается, выпрямляется, складывается) описывай действием, не придумывай второй референс;
- каждый появившийся предмет сохраняется до явной передачи, размещения, удаления или естественного выхода из кадра;
- каждое действие начинается из предыдущей позы и заканчивается понятным следующим положением;
- лицо, анатомия, пальцы, одежда, форма, материал, масштаб и цвет объектов остаются стабильными;
- руки физически корректно контактируют с предметами; исключи clipping, телепортацию и нарушение веса/инерции;
- камера движется по одной плавной траектории без скачка ракурса, масштаба или перспективы;
- весь prompt описывает один причинно связанный эпизод и явно фиксирует конечное состояние;
- не добавляй новые сцены, персонажей, предметы, свойства продукта или возможности API;
- таймлайн обязан укладываться в указанную длительность;
- если звук выключен, не добавляй речь, музыку и звуковые эффекты;
- последняя строка ZH-промпта строго равна переданному CONTINUITY LOCK;
- полный ZH-промпт не длиннее 1800 символов.

Не используй рекомендации об обходе модерации и внешние workaround-инструменты.
Верни только JSON без markdown:
{"summary_ru":"1-2 предложения", "prompt_zh":"полный промпт"}
"""


@dataclass(frozen=True)
class SeedancePrompt:
    summary_ru: str
    prompt_zh: str


def build_seedance_prompt(
    llm: LlmClient,
    *,
    scene_id: str,
    universal_prompt: str,
    duration_seconds: int,
    aspect_ratio: str,
    sound_enabled: bool,
    reference_manifest: list[dict[str, Any]] | None = None,
) -> SeedancePrompt:
    if not llm.is_configured:
        raise LlmError("Seedance-промптеру недоступна основная GPT-модель.")
    manifest = reference_manifest or [
        {
            "tag": "@Image1",
            "reference_id": "START-FRAME",
            "kind": "start_frame",
            "description": "Точный исходный кадр и первый кадр видео.",
            "scene_ids": [scene_id],
        }
    ]
    expected_tags = tuple(str(item.get("tag", "")).strip() for item in manifest)
    if (
        not expected_tags
        or expected_tags[0] != "@Image1"
        or expected_tags
        != tuple(f"@Image{index}" for index in range(1, len(expected_tags) + 1))
    ):
        raise ValueError("Карта Seedance-референсов должна содержать последовательные @Image1..N.")
    if len(expected_tags) > MAX_SEEDANCE_REFERENCE_IMAGES:
        raise ValueError(
            f"Seedance-пакет содержит больше {MAX_SEEDANCE_REFERENCE_IMAGES} изображений."
        )
    manifest_lines = []
    for item in manifest:
        scene_ids = ", ".join(str(value) for value in item.get("scene_ids", []))
        manifest_lines.append(
            f"{item['tag']} = {item.get('label') or item.get('reference_id', 'reference')}; "
            f"роль: {item.get('role', item.get('kind', 'additional_visual_reference'))}; "
            f"назначение: {str(item.get('usage', '')).strip()[:220] or 'additional visual reference'}; "
            f"сцены: {scene_ids or 'весь клип'}; описание: "
            f"{str(item.get('description', '')).strip()[:220] or 'описание отсутствует'}"
        )
    request = (
        f"Сцена: {scene_id}\n"
        f"Длительность: {duration_seconds} секунд\n"
        f"Формат: {aspect_ratio}\n"
        f"Звук: {'включён' if sound_enabled else 'выключен'}\n\n"
        "КАРТА РЕАЛЬНО ПРИКРЕПЛЁННЫХ ИЗОБРАЖЕНИЙ:\n"
        + "\n".join(manifest_lines)
        + "\n\n"
        "Ниже находится проверенное универсальное описание сцены. Не расширяй его "
        "новыми событиями. Преврати его в один Seedance 2 multimodal reference-to-video prompt "
        "с тем же таймлайном. В начале ZH-промпта кратко зафиксируй роль каждого тега. "
        "В TIMELINE обязательно связывай каждый релевантный тег с конкретным персонажем, "
        "предметом, окружением или ракурсом прямо внутри описания действия. "
        "@Image1 явно назначь первым кадром видео.\n\n"
        f"{universal_prompt}\n\nCONTINUITY LOCK (последняя строка без изменений):\n"
        f"{SEEDANCE_FINAL_LOCK}"
    )
    messages = [
        {"role": "system", "content": SEEDANCE_SYSTEM_PROMPT},
        {"role": "user", "content": request},
    ]
    last_error: ValueError | None = None
    for attempt in range(2):
        raw = llm.chat(messages)
        try:
            return parse_seedance_prompt(
                raw,
                expected_tags=expected_tags,
                timeline_tags=expected_tags,
            )
        except ValueError as exc:
            last_error = exc
            if attempt:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": raw[:6000]},
                    {
                        "role": "user",
                        "content": (
                            "Исправь только структуру ответа, не меняя события. "
                            f"Ошибка проверки: {exc}. Используй ровно теги "
                            f"{', '.join(expected_tags)}; каждый тег должен быть и в карте "
                            "референсов, и хотя бы в одном блоке TIMELINE, но теги нельзя "
                            "перечислять подряд без смысловой связи с объектом. Верни только JSON."
                        ),
                    },
                ]
            )
    raise last_error or ValueError("Seedance-промпт не прошёл проверку.")


def parse_seedance_prompt(
    raw: str,
    *,
    expected_tags: tuple[str, ...] = ("@Image1",),
    timeline_tags: tuple[str, ...] = (),
) -> SeedancePrompt:
    cleaned = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Seedance-промптер вернул не JSON.")
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Seedance-промптер вернул JSON неправильного типа.")

    summary = str(payload.get("summary_ru", "")).strip()
    prompt = str(payload.get("prompt_zh", "")).strip()
    if not summary or not prompt:
        raise ValueError("Seedance-промптер не заполнил русское описание или ZH-промпт.")
    if not re.search(r"[\u4e00-\u9fff]", prompt):
        raise ValueError("Seedance ZH-промпт не содержит китайского текста.")
    found_tags = {
        f"@Image{match}" for match in REFERENCE_TAG_PATTERN.findall(prompt)
    }
    expected = set(expected_tags)
    missing = sorted(expected - found_tags)
    unknown = sorted(found_tags - expected)
    if missing:
        raise ValueError(
            "Seedance ZH-промпт не использует обязательные референсы: "
            + ", ".join(missing)
            + "."
        )
    if unknown:
        raise ValueError(
            "Seedance ZH-промпт придумал отсутствующие референсы: "
            + ", ".join(unknown)
            + "."
        )
    timeline_match = re.search(r"\[\s*0[:：]00", prompt)
    if timeline_tags and timeline_match is None:
        raise ValueError("Seedance ZH-промпт не содержит проверяемый TIMELINE.")
    timeline_text = prompt[timeline_match.start() :] if timeline_match else ""
    if re.search(
        r"@Image[1-9]\d*(?:\s*[,，/+&]\s*|\s+)@Image[1-9]\d*",
        timeline_text,
    ):
        raise ValueError(
            "В TIMELINE найден бессмысленный список @ImageN без семантической связи."
        )
    missing_in_timeline = [
        tag for tag in timeline_tags if tag not in timeline_text
    ]
    if missing_in_timeline:
        raise ValueError(
            "В TIMELINE не указаны релевантные референсы: "
            + ", ".join(missing_in_timeline)
            + "."
        )
    if not prompt.endswith(SEEDANCE_FINAL_LOCK):
        prompt = f"{prompt.rstrip()}\n{SEEDANCE_FINAL_LOCK}"
    if len(prompt) > 1800:
        raise ValueError(
            f"Seedance ZH-промпт длиннее лимита гайда: {len(prompt)} из 1800 символов."
        )
    return SeedancePrompt(summary_ru=summary, prompt_zh=prompt)
