from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .vault import now_stamp


SCENE_ID_PATTERN = re.compile(r"S\d{2,4}")
REFERENCE_ID_PATTERN = re.compile(r"REF-[A-Z0-9][A-Z0-9-]{0,31}")
LOCATION_ID_PATTERN = re.compile(r"LOC-[A-Z0-9][A-Z0-9-]{0,31}")
IDENTITY_GROUP_PATTERN = re.compile(r"ENTITY-[A-Z0-9][A-Z0-9-]{0,31}")
REFERENCE_KINDS = {"character", "environment", "style", "object"}
STORYBOARD_ASSET_KINDS = {"character", "object"}
EMPTY_CONTRACT_MARKERS = {
    "",
    "-",
    "N/A",
    "NA",
    "NONE",
    "NULL",
    "NOT APPLICABLE",
    "NOT_APPLICABLE",
}


class ProductionContractError(ValueError):
    pass


def normalize_optional_contract_value(value: Any) -> str:
    """Treat common LLM placeholders as an absent optional contract value."""

    if value is None:
        return ""
    normalized = str(value).strip().upper()
    return "" if normalized in EMPTY_CONTRACT_MARKERS else normalized


def normalize_reference_kind(raw_kind: Any, reference_id: str) -> str:
    """Recover common LLM enum drift only when the reference ID makes the intent explicit."""

    kind = str(raw_kind or "character").strip().lower()
    if kind in REFERENCE_KINDS:
        return kind

    normalized_id = reference_id.upper()
    id_hints = (
        ("character", ("REF-CHAR", "REF-PERSON", "REF-HERO", "REF-CAST")),
        ("environment", ("REF-ENV", "REF-LOC", "REF-BG", "REF-ROOM")),
        ("style", ("REF-STYLE", "REF-LOOK", "REF-LIGHT", "REF-PALETTE")),
        ("object", ("REF-OBJ", "REF-PROP", "REF-ITEM", "REF-PRODUCT")),
    )
    for candidate, prefixes in id_hints:
        if normalized_id.startswith(prefixes):
            return candidate

    tokens = set(re.findall(r"[a-z]+", kind))
    matches = tokens & REFERENCE_KINDS
    if len(matches) == 1:
        return matches.pop()
    raise ProductionContractError(
        f"{reference_id}: kind должен быть character, environment, style или object."
    )


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    order: int
    duration_seconds: float
    purpose: str
    visual: str
    physical_action: str
    camera_movement: str
    voiceover: str
    on_screen_text: str
    sound: str
    transition: str
    continuity: dict[str, str]
    image_prompt: str
    reference_ids: tuple[str, ...] = ()
    location_id: str = ""
    location_reference_scene_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], fallback_order: int) -> "SceneSpec":
        scene_id = str(data.get("scene_id", f"S{fallback_order:02d}")).strip().upper()
        if not SCENE_ID_PATTERN.fullmatch(scene_id):
            raise ProductionContractError(f"Недопустимый scene_id: {scene_id}")
        try:
            order = int(data.get("order", fallback_order))
            duration = float(data.get("duration_seconds", 0))
        except (TypeError, ValueError) as exc:
            raise ProductionContractError(f"Некорректные числа в сцене {scene_id}.") from exc
        if order < 1 or duration <= 0:
            raise ProductionContractError(
                f"Сцена {scene_id}: order и duration_seconds должны быть положительными."
            )
        continuity_raw = data.get("continuity", {})
        if not isinstance(continuity_raw, dict):
            raise ProductionContractError(f"Сцена {scene_id}: continuity должен быть объектом.")
        continuity = {
            str(key): str(value).strip()
            for key, value in continuity_raw.items()
            if str(value).strip()
        }
        visual = str(data.get("visual", "")).strip()
        image_prompt = str(data.get("image_prompt", "")).strip()
        if not visual or not image_prompt:
            raise ProductionContractError(
                f"Сцена {scene_id}: обязательны visual и image_prompt."
            )
        raw_reference_ids = data.get("reference_ids", [])
        if not isinstance(raw_reference_ids, (list, tuple)):
            raise ProductionContractError(
                f"Сцена {scene_id}: reference_ids должен быть списком."
            )
        normalized_reference_ids = (
            normalize_optional_contract_value(item) for item in raw_reference_ids
        )
        reference_ids = tuple(
            dict.fromkeys(item for item in normalized_reference_ids if item)
        )
        if any(not REFERENCE_ID_PATTERN.fullmatch(item) for item in reference_ids):
            raise ProductionContractError(
                f"Сцена {scene_id}: найден некорректный reference_id."
            )
        location_id = normalize_optional_contract_value(data.get("location_id", ""))
        if location_id and not LOCATION_ID_PATTERN.fullmatch(location_id):
            raise ProductionContractError(
                f"Сцена {scene_id}: некорректный location_id {location_id}."
            )
        location_reference_scene_id = normalize_optional_contract_value(
            data.get("location_reference_scene_id", "")
        )
        if location_reference_scene_id == scene_id:
            location_reference_scene_id = ""
        if (
            location_reference_scene_id
            and not SCENE_ID_PATTERN.fullmatch(location_reference_scene_id)
        ):
            raise ProductionContractError(
                f"Сцена {scene_id}: некорректный location_reference_scene_id."
            )
        return cls(
            scene_id=scene_id,
            order=order,
            duration_seconds=duration,
            purpose=str(data.get("purpose", "")).strip(),
            visual=visual,
            physical_action=str(data.get("physical_action", "")).strip(),
            camera_movement=str(data.get("camera_movement", "static")).strip() or "static",
            voiceover=str(data.get("voiceover", "")).strip(),
            on_screen_text=str(data.get("on_screen_text", "")).strip(),
            sound=str(data.get("sound", "")).strip(),
            transition=str(data.get("transition", "")).strip(),
            continuity=continuity,
            image_prompt=image_prompt,
            reference_ids=reference_ids,
            location_id=location_id,
            location_reference_scene_id=location_reference_scene_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceSpec:
    reference_id: str
    kind: str
    name: str
    prompt: str
    scene_ids: tuple[str, ...]
    identity_group: str = ""
    state_label: str = ""
    base_reference_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceSpec":
        reference_id = str(data.get("reference_id", "")).strip().upper()
        if not REFERENCE_ID_PATTERN.fullmatch(reference_id):
            raise ProductionContractError(f"Некорректный reference_id: {reference_id}")
        kind = normalize_reference_kind(data.get("kind", "character"), reference_id)
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            raise ProductionContractError(f"{reference_id}: reference prompt пустой.")
        raw_scene_ids = data.get("scene_ids", [])
        if not isinstance(raw_scene_ids, (list, tuple)):
            raise ProductionContractError(f"{reference_id}: scene_ids должен быть списком.")
        scene_ids = tuple(
            dict.fromkeys(
                str(item).strip().upper()
                for item in raw_scene_ids
                if str(item).strip()
            )
        )
        if any(not SCENE_ID_PATTERN.fullmatch(item) for item in scene_ids):
            raise ProductionContractError(f"{reference_id}: некорректный scene_id.")
        identity_group = normalize_optional_contract_value(data.get("identity_group", ""))
        state_label = str(data.get("state_label", "")).strip()
        base_reference_id = normalize_optional_contract_value(
            data.get("base_reference_id", "")
        )
        if identity_group and not IDENTITY_GROUP_PATTERN.fullmatch(identity_group):
            raise ProductionContractError(
                f"{reference_id}: некорректный identity_group {identity_group}."
            )
        if bool(identity_group) != bool(state_label):
            raise ProductionContractError(
                f"{reference_id}: identity_group и state_label должны задаваться вместе."
            )
        if base_reference_id and not REFERENCE_ID_PATTERN.fullmatch(base_reference_id):
            raise ProductionContractError(
                f"{reference_id}: некорректный base_reference_id."
            )
        if base_reference_id == reference_id:
            raise ProductionContractError(
                f"{reference_id}: референс не может зависеть сам от себя."
            )
        return cls(
            reference_id=reference_id,
            kind=kind,
            name=str(data.get("name", reference_id)).strip() or reference_id,
            prompt=prompt,
            scene_ids=scene_ids,
            identity_group=identity_group,
            state_label=state_label,
            base_reference_id=base_reference_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocationSpec:
    location_id: str
    name: str
    description: str
    scene_ids: tuple[str, ...]
    canonical_scene_id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocationSpec":
        location_id = str(data.get("location_id", "")).strip().upper()
        if not LOCATION_ID_PATTERN.fullmatch(location_id):
            raise ProductionContractError(f"Некорректный location_id: {location_id}")
        raw_scene_ids = data.get("scene_ids", [])
        if not isinstance(raw_scene_ids, (list, tuple)):
            raise ProductionContractError(f"{location_id}: scene_ids должен быть списком.")
        scene_ids = tuple(
            dict.fromkeys(
                str(item).strip().upper()
                for item in raw_scene_ids
                if str(item).strip()
            )
        )
        if not scene_ids or any(not SCENE_ID_PATTERN.fullmatch(item) for item in scene_ids):
            raise ProductionContractError(
                f"{location_id}: нужен непустой список корректных scene_ids."
            )
        canonical = str(data.get("canonical_scene_id", scene_ids[0])).strip().upper()
        if canonical not in scene_ids:
            raise ProductionContractError(
                f"{location_id}: canonical_scene_id должен входить в scene_ids."
            )
        return cls(
            location_id=location_id,
            name=str(data.get("name", location_id)).strip() or location_id,
            description=str(data.get("description", "")).strip(),
            scene_ids=scene_ids,
            canonical_scene_id=canonical,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StoryboardAssetSpec:
    reference_id: str
    kind: str
    name: str
    description: str
    scene_ids: tuple[str, ...]
    identity_group: str = ""
    state_label: str = ""
    base_reference_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoryboardAssetSpec":
        reference_id = str(data.get("reference_id", "")).strip().upper()
        if not REFERENCE_ID_PATTERN.fullmatch(reference_id):
            raise ProductionContractError(
                f"Некорректный reference_id визуальной библии: {reference_id}"
            )
        kind = str(data.get("kind", "")).strip().lower()
        if kind not in STORYBOARD_ASSET_KINDS:
            raise ProductionContractError(
                f"{reference_id}: kind должен быть character или object."
            )
        raw_scene_ids = data.get("scene_ids", [])
        if not isinstance(raw_scene_ids, list):
            raise ProductionContractError(f"{reference_id}: scene_ids должен быть списком.")
        scene_ids = tuple(
            dict.fromkeys(str(item).strip().upper() for item in raw_scene_ids if str(item).strip())
        )
        if not scene_ids or any(not SCENE_ID_PATTERN.fullmatch(item) for item in scene_ids):
            raise ProductionContractError(
                f"{reference_id}: нужен непустой список корректных scene_ids."
            )
        description = str(data.get("description", "")).strip()
        if not description:
            raise ProductionContractError(f"{reference_id}: description не заполнен.")
        identity_group = normalize_optional_contract_value(data.get("identity_group", ""))
        state_label = str(data.get("state_label", "")).strip()
        base_reference_id = normalize_optional_contract_value(
            data.get("base_reference_id", "")
        )
        if identity_group and not IDENTITY_GROUP_PATTERN.fullmatch(identity_group):
            raise ProductionContractError(
                f"{reference_id}: некорректный identity_group {identity_group}."
            )
        if bool(identity_group) != bool(state_label):
            raise ProductionContractError(
                f"{reference_id}: identity_group и state_label должны задаваться вместе."
            )
        if base_reference_id and not REFERENCE_ID_PATTERN.fullmatch(base_reference_id):
            raise ProductionContractError(
                f"{reference_id}: некорректный base_reference_id."
            )
        if base_reference_id == reference_id:
            raise ProductionContractError(
                f"{reference_id}: asset не может зависеть сам от себя."
            )
        return cls(
            reference_id=reference_id,
            kind=kind,
            name=str(data.get("name", reference_id)).strip() or reference_id,
            description=description,
            scene_ids=scene_ids,
            identity_group=identity_group,
            state_label=state_label,
            base_reference_id=base_reference_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StoryboardFrameSpec:
    scene_id: str
    location_id: str
    reference_ids: tuple[str, ...]
    task: str
    composition: str
    must_show: str
    constraints: str
    transition: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoryboardFrameSpec":
        scene_id = str(data.get("scene_id", "")).strip().upper()
        if not SCENE_ID_PATTERN.fullmatch(scene_id):
            raise ProductionContractError(f"Некорректный scene_id кадра: {scene_id}")
        location_id = str(data.get("location_id", "")).strip().upper()
        if not LOCATION_ID_PATTERN.fullmatch(location_id):
            raise ProductionContractError(
                f"{scene_id}: нужен корректный обязательный location_id."
            )
        raw_reference_ids = data.get("reference_ids", [])
        if not isinstance(raw_reference_ids, list):
            raise ProductionContractError(f"{scene_id}: reference_ids должен быть списком.")
        reference_ids = tuple(
            dict.fromkeys(
                str(item).strip().upper()
                for item in raw_reference_ids
                if str(item).strip()
            )
        )
        if any(not REFERENCE_ID_PATTERN.fullmatch(item) for item in reference_ids):
            raise ProductionContractError(f"{scene_id}: некорректный reference_id.")
        required_text = {
            "task": str(data.get("task", "")).strip(),
            "composition": str(data.get("composition", "")).strip(),
            "must_show": str(data.get("must_show", "")).strip(),
            "constraints": str(data.get("constraints", "")).strip(),
            "transition": str(data.get("transition", "")).strip(),
        }
        missing = [key for key, value in required_text.items() if not value]
        if missing:
            raise ProductionContractError(
                f"{scene_id}: не заполнены поля {', '.join(missing)}."
            )
        return cls(
            scene_id=scene_id,
            location_id=location_id,
            reference_ids=reference_ids,
            **required_text,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualBible:
    visual_basis: str
    assets: tuple[StoryboardAssetSpec, ...]
    locations: tuple[LocationSpec, ...]
    frames: tuple[StoryboardFrameSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "visual_basis": self.visual_basis,
            "assets": [item.to_dict() for item in self.assets],
            "locations": [item.to_dict() for item in self.locations],
            "frames": [item.to_dict() for item in self.frames],
        }


def extract_json_contract(text: str, marker: str = "") -> dict[str, Any]:
    """Extract one fenced JSON contract, preferring a block following the marker."""

    source = text or ""
    if marker:
        marker_index = source.find(marker)
        if marker_index >= 0:
            source = source[marker_index + len(marker) :]
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", source, flags=re.DOTALL | re.IGNORECASE)
    candidates = blocks or re.findall(r"(\{\s*\"schema_version\".*\})", source, flags=re.DOTALL)
    for candidate in candidates:
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return decoded
    raise ProductionContractError(f"Не найден валидный JSON-контракт {marker or 'сцен' }.")


def strip_json_contract(text: str, marker: str) -> str:
    """Remove a malformed fenced machine block while preserving readable stage prose."""

    pattern = re.compile(
        rf"{re.escape(marker)}\s*```(?:json)?\s*.*?```",
        flags=re.DOTALL | re.IGNORECASE,
    )
    return pattern.sub("", text or "").strip()


def parse_scene_contract(text: str) -> list[SceneSpec]:
    payload = extract_json_contract(text, "SCENE_CONTRACT")
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ProductionContractError("SCENE_CONTRACT не содержит непустой список scenes.")
    scenes = [SceneSpec.from_dict(item, index) for index, item in enumerate(raw_scenes, 1) if isinstance(item, dict)]
    if len(scenes) != len(raw_scenes):
        raise ProductionContractError("Каждая сцена должна быть JSON-объектом.")
    scenes.sort(key=lambda item: item.order)
    ids = [scene.scene_id for scene in scenes]
    orders = [scene.order for scene in scenes]
    if len(ids) != len(set(ids)) or len(orders) != len(set(orders)):
        raise ProductionContractError("scene_id и order должны быть уникальными.")
    return scenes


_TARGET_DURATION_PATTERNS = (
    re.compile(
        r"(?:длительност(?:ь|ью)?|хронометраж)\s*[:=\-–—]?\s*"
        r"(?P<seconds>\d+(?:[.,]\d+)?)\s*(?:секунд(?:а|ы)?|сек\.?|seconds?|secs?)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:ролик|видео|reel|video)\s+(?:длительностью\s+|на\s+)?"
        r"(?P<seconds>\d+(?:[.,]\d+)?)\s*(?:секунд(?:а|ы)?|сек\.?|seconds?|secs?)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?P<seconds>\d+(?:[.,]\d+)?)\s*[- ]?"
        r"(?:секундн\w*|second(?:-long)?)\s+(?:ролик|видео|reel|video)",
        flags=re.IGNORECASE,
    ),
)


def extract_target_duration_seconds(source_text: str) -> float | None:
    """Read an explicitly requested total runtime from human-readable source text."""

    for pattern in _TARGET_DURATION_PATTERNS:
        match = pattern.search(source_text or "")
        if not match:
            continue
        value = float(match.group("seconds").replace(",", "."))
        if value > 0:
            return value
    return None


def validate_script_scene_plan(
    scenes: Sequence[SceneSpec],
    source_text: str = "",
) -> None:
    """Validate planning rules for new content-factory scripts."""

    if not 2 <= len(scenes) <= 10:
        raise ProductionContractError(
            "Сценарий должен содержать от 2 до 10 сцен, выбранных по смыслу и хронометражу."
        )
    too_long = [
        scene.scene_id for scene in scenes if scene.duration_seconds > 15
    ]
    if too_long:
        raise ProductionContractError(
            "Одна сцена не должна быть длиннее 15 секунд; раздели непрерывное "
            f"действие на производственные отрезки: {', '.join(too_long)}."
        )
    target = extract_target_duration_seconds(source_text)
    if target is None:
        return
    actual = sum(scene.duration_seconds for scene in scenes)
    if abs(actual - target) > 0.25:
        raise ProductionContractError(
            "Сумма duration_seconds должна совпадать с хронометражем ролика: "
            f"ожидалось {target:g} сек., получено {actual:g} сек."
        )


def parse_visual_bible_contract(text: str, scenes: list[SceneSpec]) -> VisualBible:
    """Validate the storyboard inventory and its one-to-one mapping to script scenes."""

    payload = extract_json_contract(text, "VISUAL_BIBLE_CONTRACT")
    visual_basis = str(payload.get("visual_basis", "")).strip()
    if not visual_basis:
        raise ProductionContractError("VISUAL_BIBLE_CONTRACT: visual_basis не заполнен.")
    raw_assets = payload.get("assets", [])
    raw_locations = payload.get("locations", [])
    raw_frames = payload.get("frames", [])
    if not isinstance(raw_assets, list) or not isinstance(raw_locations, list):
        raise ProductionContractError("VISUAL_BIBLE_CONTRACT: assets и locations должны быть списками.")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ProductionContractError("VISUAL_BIBLE_CONTRACT: frames должен быть непустым списком.")
    assets = tuple(StoryboardAssetSpec.from_dict(item) for item in raw_assets if isinstance(item, dict))
    locations = tuple(LocationSpec.from_dict(item) for item in raw_locations if isinstance(item, dict))
    frames = tuple(StoryboardFrameSpec.from_dict(item) for item in raw_frames if isinstance(item, dict))
    if len(assets) != len(raw_assets) or len(locations) != len(raw_locations) or len(frames) != len(raw_frames):
        raise ProductionContractError("Каждый элемент визуальной библии должен быть JSON-объектом.")

    scene_ids = [scene.scene_id for scene in sorted(scenes, key=lambda item: item.order)]
    frame_ids = [frame.scene_id for frame in frames]
    if frame_ids != scene_ids:
        raise ProductionContractError(
            "Кадры визуальной библии должны один в один совпадать с порядком сцен сценария."
        )
    asset_map = {item.reference_id: item for item in assets}
    location_map = {item.location_id: item for item in locations}
    if len(asset_map) != len(assets) or len(location_map) != len(locations):
        raise ProductionContractError("ID assets и locations должны быть уникальными.")
    _validate_reference_dependencies(assets)
    if not locations:
        raise ProductionContractError("Визуальная библия должна описывать хотя бы одну локацию.")

    used_assets: dict[str, list[str]] = {item.reference_id: [] for item in assets}
    used_locations: dict[str, list[str]] = {item.location_id: [] for item in locations}
    for frame in frames:
        unknown_assets = set(frame.reference_ids) - set(asset_map)
        if unknown_assets:
            raise ProductionContractError(
                f"{frame.scene_id}: неизвестные assets {', '.join(sorted(unknown_assets))}."
            )
        if frame.location_id not in location_map:
            raise ProductionContractError(
                f"{frame.scene_id}: неизвестная локация {frame.location_id}."
            )
        for reference_id in frame.reference_ids:
            used_assets[reference_id].append(frame.scene_id)
        used_locations[frame.location_id].append(frame.scene_id)

    for asset in assets:
        if tuple(used_assets[asset.reference_id]) != asset.scene_ids:
            raise ProductionContractError(
                f"{asset.reference_id}: scene_ids не совпадают с фактическими кадрами."
            )
    for location in locations:
        if tuple(used_locations[location.location_id]) != location.scene_ids:
            raise ProductionContractError(
                f"{location.location_id}: scene_ids не совпадают с фактическими кадрами."
            )
    return VisualBible(visual_basis, assets, locations, frames)


def validate_image_plan_against_visual_bible(
    bible: VisualBible,
    scenes: list[SceneSpec],
    references: list[ReferenceSpec],
    locations: list[LocationSpec],
) -> None:
    """Ensure prompt generation cannot silently drop storyboard entities or links."""

    reference_map = {item.reference_id: item for item in references}
    location_map = {item.location_id: item for item in locations}
    scene_map = {item.scene_id: item for item in scenes}
    for asset in bible.assets:
        generated = reference_map.get(asset.reference_id)
        if generated is None:
            raise ProductionContractError(
                f"Промптер потерял обязательный референс {asset.reference_id}."
            )
        if generated.kind != asset.kind or generated.scene_ids != asset.scene_ids:
            raise ProductionContractError(
                f"{asset.reference_id}: тип или связи со сценами отличаются от раскадровки."
            )
        if (
            generated.identity_group != asset.identity_group
            or generated.state_label != asset.state_label
            or generated.base_reference_id != asset.base_reference_id
        ):
            raise ProductionContractError(
                f"{asset.reference_id}: связь состояний отличается от раскадровки."
            )
    for planned in bible.locations:
        generated = location_map.get(planned.location_id)
        if generated is None or generated.scene_ids != planned.scene_ids:
            raise ProductionContractError(
                f"{planned.location_id}: локация или её сцены отличаются от раскадровки."
            )
    for frame in bible.frames:
        scene = scene_map.get(frame.scene_id)
        if scene is None:
            raise ProductionContractError(f"Промптер потерял кадр {frame.scene_id}.")
        if scene.location_id != frame.location_id:
            raise ProductionContractError(
                f"{frame.scene_id}: location_id отличается от раскадровки."
            )
        if not set(frame.reference_ids).issubset(scene.reference_ids):
            raise ProductionContractError(
                f"{frame.scene_id}: промптер потерял персонажа или предмет из раскадровки."
            )


def validate_english_image_prompts(
    scenes: list[SceneSpec], references: list[ReferenceSpec]
) -> None:
    """Reject production prompts containing Cyrillic while allowing Russian UI summaries."""

    cyrillic = re.compile(r"[А-Яа-яЁё]")
    invalid = [scene.scene_id for scene in scenes if cyrillic.search(scene.image_prompt)]
    invalid.extend(
        item.reference_id for item in references if cyrillic.search(item.prompt)
    )
    if invalid:
        raise ProductionContractError(
            "Промпты генерации должны быть на английском. Исправь: "
            + ", ".join(invalid)
            + "."
        )


def generation_plan_summary(
    scenes: list[SceneSpec], references: list[ReferenceSpec], locations: list[LocationSpec]
) -> dict[str, int]:
    """Calculate the actual image workload from validated entities, never from LLM prose."""

    counts = {
        "characters": sum(item.kind == "character" for item in references),
        "objects": sum(item.kind == "object" for item in references),
        "environment_references": sum(item.kind == "environment" for item in references),
        "style_references": sum(item.kind == "style" for item in references),
        "locations": len(locations),
        "frames": len(scenes),
    }
    counts["total_images"] = (
        counts["characters"]
        + counts["objects"]
        + counts["environment_references"]
        + counts["style_references"]
        + counts["frames"]
    )
    return counts


def merge_image_prompt_contract(scenes: list[SceneSpec], prompt_text: str) -> list[SceneSpec]:
    """Replace image prompts only when a complete one-to-one prompt contract is valid."""

    payload = extract_json_contract(prompt_text, "IMAGE_PROMPT_CONTRACT")
    raw = payload.get("scenes")
    if not isinstance(raw, list):
        raise ProductionContractError("IMAGE_PROMPT_CONTRACT должен содержать scenes.")
    prompt_items = {
        str(item.get("scene_id", "")).strip().upper(): item
        for item in raw
        if isinstance(item, dict)
    }
    prompts = {
        scene_id: str(item.get("image_prompt", "")).strip()
        for scene_id, item in prompt_items.items()
    }
    expected = {scene.scene_id for scene in scenes}
    if set(prompts) != expected or any(not value for value in prompts.values()):
        raise ProductionContractError(
            "Число и scene_id image prompts должны точно совпадать со сценами сценария."
        )
    merged: list[SceneSpec] = []
    for scene in scenes:
        item = prompt_items[scene.scene_id]
        data = {**scene.to_dict(), "image_prompt": prompts[scene.scene_id]}
        for key in ("reference_ids", "location_id", "location_reference_scene_id"):
            if key in item:
                data[key] = item[key]
        if "reference_ids" not in item and not data.get("reference_ids"):
            data["reference_ids"] = list(
                dict.fromkeys(REFERENCE_ID_PATTERN.findall(prompts[scene.scene_id].upper()))
            )
        merged.append(SceneSpec.from_dict(data, scene.order))
    return merged


def parse_reference_plan(
    scenes: list[SceneSpec], prompt_text: str
) -> tuple[list[ReferenceSpec], list[LocationSpec]]:
    """Read explicit v2 references, with a bounded migration for old REF-A prose."""

    payload = extract_json_contract(prompt_text, "IMAGE_PROMPT_CONTRACT")
    raw_references = payload.get("references", [])
    raw_locations = payload.get("locations", [])
    if raw_references and not isinstance(raw_references, list):
        raise ProductionContractError("IMAGE_PROMPT_CONTRACT references должен быть списком.")
    if raw_locations and not isinstance(raw_locations, list):
        raise ProductionContractError("IMAGE_PROMPT_CONTRACT locations должен быть списком.")

    references = [
        ReferenceSpec.from_dict(item)
        for item in raw_references
        if isinstance(item, dict)
    ]
    locations = [
        LocationSpec.from_dict(item)
        for item in raw_locations
        if isinstance(item, dict)
    ]
    if references:
        _validate_reference_links(scenes, references, locations)
        return references, locations

    references = _parse_legacy_reference_prompts(scenes, prompt_text)
    _validate_reference_links(scenes, references, locations)
    return references, locations


def _validate_reference_links(
    scenes: list[SceneSpec],
    references: list[ReferenceSpec],
    locations: list[LocationSpec],
) -> None:
    scene_ids = {scene.scene_id for scene in scenes}
    scenes_by_id = {scene.scene_id: scene for scene in scenes}
    reference_ids = {item.reference_id for item in references}
    location_ids = {item.location_id for item in locations}
    if len(reference_ids) != len(references):
        raise ProductionContractError("reference_id должны быть уникальными.")
    if len(location_ids) != len(locations):
        raise ProductionContractError("location_id должны быть уникальными.")
    _validate_reference_dependencies(references)
    for reference in references:
        if not set(reference.scene_ids).issubset(scene_ids):
            raise ProductionContractError(
                f"{reference.reference_id}: найдены неизвестные scene_ids."
            )
    for location in locations:
        if not set(location.scene_ids).issubset(scene_ids):
            raise ProductionContractError(
                f"{location.location_id}: найдены неизвестные scene_ids."
            )
    for scene in scenes:
        if not set(scene.reference_ids).issubset(reference_ids):
            missing = sorted(set(scene.reference_ids) - reference_ids)
            raise ProductionContractError(
                f"{scene.scene_id}: не описаны references {', '.join(missing)}."
            )
        if scene.location_id and scene.location_id not in location_ids:
            raise ProductionContractError(
                f"{scene.scene_id}: не описана локация {scene.location_id}."
            )
        if scene.location_reference_scene_id:
            source = scenes_by_id.get(scene.location_reference_scene_id)
            if source is None:
                raise ProductionContractError(
                    f"{scene.scene_id}: неизвестная reference-сцена локации."
                )
            if source.order >= scene.order:
                raise ProductionContractError(
                    f"{scene.scene_id}: reference-сцена локации должна идти раньше."
                )
            if scene.location_id and source.location_id != scene.location_id:
                raise ProductionContractError(
                    f"{scene.scene_id}: reference-сцена относится к другой локации."
                )


def _validate_reference_dependencies(
    references: list[ReferenceSpec] | list[StoryboardAssetSpec],
) -> None:
    """Validate identity-state inheritance without relying on generation order."""

    by_id = {item.reference_id: item for item in references}
    for item in references:
        base_id = item.base_reference_id
        if not base_id:
            continue
        if item.kind == "object" and _is_transient_object_variant(item):
            raise ProductionContractError(
                f"{item.reference_id}: временное состояние предмета нужно описать "
                "действием в сцене, а не отдельным референсом."
            )
        base = by_id.get(base_id)
        if base is None:
            raise ProductionContractError(
                f"{item.reference_id}: базовый референс {base_id} не описан."
            )
        if base.kind != item.kind:
            raise ProductionContractError(
                f"{item.reference_id}: состояние и базовый референс должны иметь один тип."
            )
        if not item.identity_group or item.identity_group != base.identity_group:
            raise ProductionContractError(
                f"{item.reference_id}: состояния должны иметь общий identity_group."
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(reference_id: str) -> None:
        if reference_id in visited:
            return
        if reference_id in visiting:
            raise ProductionContractError("Найдена циклическая зависимость референсов.")
        visiting.add(reference_id)
        base_id = by_id[reference_id].base_reference_id
        if base_id:
            visit(base_id)
        visiting.remove(reference_id)
        visited.add(reference_id)

    for reference_id in by_id:
        visit(reference_id)


def _is_transient_object_variant(
    item: ReferenceSpec | StoryboardAssetSpec,
) -> bool:
    source = " ".join(
        (
            item.name,
            item.state_label,
            getattr(item, "prompt", ""),
            getattr(item, "description", ""),
        )
    ).lower()
    markers = (
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
    return any(marker in source for marker in markers)


def _parse_legacy_reference_prompts(
    scenes: list[SceneSpec], prompt_text: str
) -> list[ReferenceSpec]:
    """Migrate old readable `### REF-A` sections without treating arbitrary prose as state."""

    source = prompt_text or ""
    heading_pattern = re.compile(
        r"(?im)^###\s+(REF-[A-Z0-9][A-Z0-9-]{0,31})\s*(?:[—-]\s*)?([^\r\n]*)$"
    )
    matches = list(heading_pattern.finditer(source))
    references: list[ReferenceSpec] = []
    for index, match in enumerate(matches):
        reference_id = match.group(1).upper()
        name = re.sub(r"\s*\([^)]*\)\s*$", "", match.group(2)).strip() or reference_id
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        block = source[match.end() : end]
        prompt_match = re.search(
            r"(?im)^\*\*Prompt:\*\*\s*(.+?)(?=\n\s*\n|\n---|\Z)",
            block,
            flags=re.DOTALL,
        )
        if not prompt_match:
            continue
        prompt = " ".join(prompt_match.group(1).split())
        scene_ids = tuple(
            scene.scene_id
            for scene in scenes
            if reference_id in scene.image_prompt.upper()
        )
        haystack = f"{name} {prompt}".lower()
        if any(token in haystack for token in ("environment", "location", "palette", "палитр", "локац")):
            kind = "environment"
        elif any(token in haystack for token in ("style guide", "lighting guide", "свет")):
            kind = "style"
        else:
            kind = "character"
        references.append(
            ReferenceSpec(reference_id, kind, name, prompt, scene_ids)
        )
    return references


def build_scene_contract_messages(
    script: str,
    storyboard: str,
    prompts: str,
    *,
    source_context: str = "",
) -> list[dict[str, str]]:
    """Create a strict migration request for older runs that lack machine-readable scenes."""

    return [
        {
            "role": "system",
            "content": (
                "Convert the approved production package into strict JSON. Return JSON only, no markdown. "
                "Use 2 to 10 scenes chosen from the source and requested total runtime. Never force "
                "six scenes and never split time into equal 3-4 second blocks by habit. Give a simple "
                "beat less time and a complex or emotional beat more time. Every scene must be one "
                "continuous visual moment, last no more than 15 seconds, and the sum of all "
                "duration_seconds must exactly equal the requested video duration. Each scene needs: "
                "scene_id S01..., order, duration_seconds > 0, purpose, visual, physical_action, "
                "camera_movement, voiceover, on_screen_text, sound, transition, continuity object, "
                "reference_ids array, location_id, location_reference_scene_id, and one image_prompt "
                "for one separate image. Never request grids or collages. "
                "Do not invent product facts. Output {schema_version:1, scenes:[...]}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"SOURCE REQUIREMENTS:\n{source_context[:8000]}\n\n"
                f"SCRIPT:\n{script[:14000]}\n\nSTORYBOARD:\n{storyboard[:14000]}\n\n"
                f"IMAGE PROMPTS:\n{prompts[:14000]}"
            ),
        },
    ]


def parse_plain_scene_json(text: str) -> list[SceneSpec]:
    """Parse a JSON-only migration response."""

    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ProductionContractError("Модель не вернула валидный JSON сцен.") from exc
    if not isinstance(payload, dict):
        raise ProductionContractError("JSON сцен должен быть объектом.")
    wrapped = f"SCENE_CONTRACT\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
    return parse_scene_contract(wrapped)


def format_scene_contract(scenes: list[SceneSpec]) -> str:
    """Serialize validated scenes into the canonical Markdown contract."""

    payload = {
        "schema_version": 1,
        "scenes": [scene.to_dict() for scene in scenes],
    }
    return (
        "SCENE_CONTRACT\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```"
    )


def build_visual_bible_contract_messages(
    script: str, storyboard: str, scenes: list[SceneSpec]
) -> list[dict[str, str]]:
    """Repair a missing storyboard contract without changing the approved scenario."""

    source = [
        {
            "scene_id": item.scene_id,
            "order": item.order,
            "purpose": item.purpose,
            "visual": item.visual,
            "continuity": item.continuity,
            "transition": item.transition,
        }
        for item in scenes
    ]
    return [
        {
            "role": "system",
            "content": (
                "Return strict JSON only. Build a visual bible before framing. Output exactly "
                "{schema_version:2,visual_basis,assets:[{reference_id,kind,name,description,scene_ids,"
                "identity_group,state_label,base_reference_id}],"
                "locations:[{location_id,name,description,scene_ids,canonical_scene_id}],"
                "frames:[{scene_id,location_id,reference_ids,task,composition,must_show,constraints,transition}]}. "
                "Asset kind is character or object. Include every recurring or identity-critical character "
                "and object, but do not create assets for disposable background props. Keep every scene_id "
                "exactly once and in source order. Every frame must use one declared location. Asset and "
                "location scene_ids must exactly match their use in frames. Anticipate future scenes: establish "
                "recurring characters, objects and spatial anchors before they become important. Create a dependent "
                "asset variant only for a persistent redesign that cannot be animated from one canonical image. "
                "Opening/closing, bending/straightening, folding/unfolding and similar temporary object states "
                "must use one asset and be described later as an action. Persistent variants share one "
                "identity_group (ENTITY-...), a precise state_label and the earliest base_reference_id. "
                "Use empty strings when these fields do not apply. Do not invent "
                "plot events, characters, product facts or extra frames."
            ),
        },
        {
            "role": "user",
            "content": (
                "APPROVED SCENES:\n"
                + json.dumps(source, ensure_ascii=False)
                + f"\n\nSCRIPT:\n{script[:14000]}\n\nSTORYBOARD DRAFT:\n{storyboard[:14000]}"
            ),
        },
    ]


def parse_plain_visual_bible_json(text: str, scenes: list[SceneSpec]) -> VisualBible:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ProductionContractError(
            "Модель не вернула валидный JSON визуальной библии."
        ) from exc
    if not isinstance(payload, dict):
        raise ProductionContractError("JSON визуальной библии должен быть объектом.")
    wrapped = (
        "VISUAL_BIBLE_CONTRACT\n```json\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n```"
    )
    return parse_visual_bible_contract(wrapped, scenes)


def format_visual_bible_contract(bible: VisualBible) -> str:
    return (
        "VISUAL_BIBLE_CONTRACT\n```json\n"
        + json.dumps(bible.to_dict(), ensure_ascii=False, indent=2)
        + "\n```"
    )


def build_image_prompt_contract_messages(
    scenes: list[SceneSpec], draft: str, visual_bible: VisualBible | None = None
) -> list[dict[str, str]]:
    """Request a strict one-to-one prompt contract when prose omitted it."""

    scene_source = [
        {
            "scene_id": scene.scene_id,
            "purpose": scene.purpose,
            "visual": scene.visual,
            "continuity": scene.continuity,
            "existing_image_prompt": scene.image_prompt,
        }
        for scene in scenes
    ]
    return [
        {
            "role": "system",
            "content": (
                "Return strict JSON only, without markdown. Output exactly "
                "{schema_version:3, references:[{reference_id,kind,name,prompt,scene_ids,identity_group,"
                "state_label,base_reference_id}], "
                "locations:[{location_id,name,description,scene_ids,canonical_scene_id}], "
                "scenes:[{scene_id,image_prompt,reference_ids,location_id,"
                "location_reference_scene_id}]}. Preserve every supplied "
                "scene_id exactly once and keep the same order. Each image_prompt must create one "
                "separate image, never a grid, collage, contact sheet, split screen, or multiple "
                "panels. Create one reusable character reference for every recurring character. "
                "references[].kind must be exactly one of character, environment, style, object. "
                "Use an empty string for absent optional IDs; never use null, none, N/A, or '-'. "
                "The canonical/first scene of a location must have an empty location_reference_scene_id. "
                "Later scenes may reference only an earlier scene with the same location_id. "
                "Put every character visible in a scene into reference_ids. Group scenes that truly "
                "share the same physical location under one location_id and choose the earliest useful "
                "canonical_scene_id. Do not group locations merely because their color palette matches. "
                "Propagate identity_group, state_label and base_reference_id exactly from the visual bible. "
                "Never create a dependent object variant for an open/closed, straight/curly, folded/unfolded "
                "or similar state that can be animated as an action. Use a persistent dependent variant only "
                "where a real redesign is visible; never attach all references "
                "to every scene. All references[].prompt and scenes[].image_prompt values must be written in English. "
                "Preserve continuity and do not invent product facts."
            ),
        },
        {
            "role": "user",
            "content": (
                "SCENES:\n"
                + json.dumps(scene_source, ensure_ascii=False)
                + (
                    "\n\nAPPROVED VISUAL BIBLE:\n"
                    + json.dumps(visual_bible.to_dict(), ensure_ascii=False)
                    if visual_bible is not None
                    else ""
                )
                + f"\n\nDRAFT PROMPTS:\n{draft[:16000]}"
            ),
        },
    ]


def parse_plain_image_prompt_json(
    scenes: list[SceneSpec], text: str
) -> list[SceneSpec]:
    """Validate a JSON-only image-prompt repair response."""

    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ProductionContractError(
            "Модель не вернула валидный JSON image prompts."
        ) from exc
    if not isinstance(payload, dict):
        raise ProductionContractError("JSON image prompts должен быть объектом.")
    wrapped = (
        "IMAGE_PROMPT_CONTRACT\n```json\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n```"
    )
    return merge_image_prompt_contract(scenes, wrapped)


def format_image_prompt_contract(
    scenes: list[SceneSpec],
    references: list[ReferenceSpec] | None = None,
    locations: list[LocationSpec] | None = None,
) -> str:
    """Serialize only the one-to-one image prompt mapping."""

    payload = {
        "schema_version": 3,
        "references": [item.to_dict() for item in (references or [])],
        "locations": [item.to_dict() for item in (locations or [])],
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "image_prompt": scene.image_prompt,
                "reference_ids": list(scene.reference_ids),
                "location_id": scene.location_id,
                "location_reference_scene_id": scene.location_reference_scene_id,
            }
            for scene in scenes
        ],
    }
    return (
        "IMAGE_PROMPT_CONTRACT\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```"
    )


class ProductionStore:
    """Atomic durable state for frames, prompts, approvals, and provider tasks."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.runs_path = project_path / "runs"

    def load(self, run_id: str) -> dict[str, Any]:
        path = self._state_path(run_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        return self._empty_state(run_id)

    def recover_interrupted_images(self, run_id: str) -> int:
        """Turn stale in-progress image attempts into retryable failures after a restart."""

        state = self.load(run_id)
        recovered = 0
        message = "Процесс генерации был прерван перезапуском бота. Действие можно повторить."
        for collection_name in ("references", "frames"):
            collection = state.get(collection_name, {})
            if not isinstance(collection, dict):
                continue
            for item in collection.values():
                if not isinstance(item, dict) or item.get("status") != "generating":
                    continue
                attempts = item.get("attempts", [])
                if isinstance(attempts, list) and attempts:
                    latest = attempts[-1]
                    if isinstance(latest, dict) and latest.get("status") == "generating":
                        latest.update(
                            status="failed",
                            error=message,
                            completed_at=now_stamp(),
                        )
                item.update(status="failed", last_error=message)
                recovered += 1
        if recovered:
            state["updated_at"] = now_stamp()
            self._write(state)
        return recovered

    def save_scene_contract(self, run_id: str, scenes: list[SceneSpec]) -> dict[str, Any]:
        if not scenes:
            raise ProductionContractError("Нельзя сохранить пустой список сцен.")
        state = self.load(run_id)
        old_frames = state.get("frames", {}) if isinstance(state.get("frames"), dict) else {}
        frames: dict[str, Any] = {}
        for scene in scenes:
            previous = old_frames.get(scene.scene_id, {})
            if previous.get("prompt") == scene.image_prompt:
                frames[scene.scene_id] = previous
            else:
                frames[scene.scene_id] = {
                    "scene_id": scene.scene_id,
                    "prompt": scene.image_prompt,
                    "status": "pending",
                    "attempts": [],
                    "latest_file": "",
                    "selected": False,
                    "last_error": "",
                }
        state["scenes"] = [scene.to_dict() for scene in sorted(scenes, key=lambda item: item.order)]
        state["frames"] = frames
        valid_ids = set(frames)
        state["selected_frame_ids"] = [
            item for item in state.get("selected_frame_ids", []) if item in valid_ids
        ]
        state["updated_at"] = now_stamp()
        self._write(state)
        return state

    def save_reference_plan(
        self,
        run_id: str,
        references: list[ReferenceSpec],
        locations: list[LocationSpec],
    ) -> dict[str, Any]:
        """Persist plans while retaining generated files whose defining prompt did not change."""

        state = self.load(run_id)
        old = state.get("references", {}) if isinstance(state.get("references"), dict) else {}
        stored: dict[str, Any] = {}
        for spec in references:
            previous = old.get(spec.reference_id, {})
            if previous.get("prompt") == spec.prompt and previous.get("kind") == spec.kind:
                item = dict(previous)
                item.update(spec.to_dict())
            else:
                item = {
                    **spec.to_dict(),
                    "status": "pending",
                    "attempts": [],
                    "latest_file": "",
                    "last_error": "",
                    "description": "",
                }
            stored[spec.reference_id] = item
        state["references"] = stored
        state["locations"] = {
            spec.location_id: spec.to_dict() for spec in locations
        }
        state["updated_at"] = now_stamp()
        self._write(state)
        return state

    def start_reference(
        self,
        run_id: str,
        reference_id: str,
        *,
        reference_inputs: list[dict[str, str]] | None = None,
    ) -> int:
        state = self.load(run_id)
        reference = self._require_reference(state, reference_id)
        attempt = len(reference.get("attempts", [])) + 1
        reference.setdefault("attempts", []).append({
            "attempt": attempt,
            "status": "generating",
            "created_at": now_stamp(),
            "reference_inputs": reference_inputs or [],
        })
        reference.update(status="generating", last_error="")
        state["image_qa"] = {"status": "not_run"}
        state["updated_at"] = now_stamp()
        self._write(state)
        return attempt

    def complete_reference(
        self,
        run_id: str,
        reference_id: str,
        attempt: int,
        content: bytes,
        extension: str,
    ) -> Path:
        if not content:
            raise ValueError("Нельзя сохранить пустой reference image.")
        safe_extension = extension.lower() if extension.lower() in {".png", ".jpg", ".jpeg", ".webp"} else ".png"
        run_path = self._run_path(run_id)
        reference_dir = run_path / "references" / reference_id
        reference_dir.mkdir(parents=True, exist_ok=True)
        target = reference_dir / f"reference-v{attempt:03d}{safe_extension}"
        target.write_bytes(content)

        state = self.load(run_id)
        reference = self._require_reference(state, reference_id)
        prompt_path = reference_dir / f"prompt-v{attempt:03d}.txt"
        prompt_path.write_text(
            str(reference.get("prompt", "")).strip() + "\n",
            encoding="utf-8",
            newline="\n",
        )
        entry = self._attempt(reference, attempt)
        entry.update(
            status="ready",
            file=str(target.relative_to(run_path)).replace("\\", "/"),
            prompt_file=str(prompt_path.relative_to(run_path)).replace("\\", "/"),
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            completed_at=now_stamp(),
        )
        reference.update(status="ready", latest_file=entry["file"], last_error="")
        state["updated_at"] = now_stamp()
        self._write(state)
        return target

    def fail_reference(
        self, run_id: str, reference_id: str, attempt: int, error: str
    ) -> None:
        state = self.load(run_id)
        reference = self._require_reference(state, reference_id)
        clean_error = error.strip()[:500] or "Неизвестная ошибка"
        self._attempt(reference, attempt).update(
            status="failed", error=clean_error, completed_at=now_stamp()
        )
        reference.update(status="failed", last_error=clean_error)
        state["updated_at"] = now_stamp()
        self._write(state)

    def cancel_reference(self, run_id: str, reference_id: str, attempt: int) -> None:
        state = self.load(run_id)
        reference = self._require_reference(state, reference_id)
        message = "Генерация остановлена пользователем."
        self._attempt(reference, attempt).update(
            status="cancelled", error=message, completed_at=now_stamp()
        )
        reference.update(status="cancelled", last_error=message)
        state["updated_at"] = now_stamp()
        self._write(state)

    def reference_path(self, run_id: str, reference_id: str) -> Path | None:
        state = self.load(run_id)
        reference = self._require_reference(state, reference_id)
        relative = str(reference.get("latest_file", "")).strip()
        path = self._run_path(run_id) / relative if relative else None
        return path if path and path.is_file() else None

    def set_reference_description(
        self, run_id: str, reference_id: str, description: str
    ) -> None:
        state = self.load(run_id)
        reference = self._require_reference(state, reference_id)
        reference["description"] = description.strip()[:4000]
        state["updated_at"] = now_stamp()
        self._write(state)

    def scenes(self, run_id: str) -> list[SceneSpec]:
        state = self.load(run_id)
        return [
            SceneSpec.from_dict(item, index)
            for index, item in enumerate(state.get("scenes", []), 1)
            if isinstance(item, dict)
        ]

    def start_frame(
        self,
        run_id: str,
        scene_id: str,
        *,
        reference_inputs: list[dict[str, str]] | None = None,
        omitted_reference_inputs: list[dict[str, str]] | None = None,
    ) -> int:
        state = self.load(run_id)
        frame = self._require_frame(state, scene_id)
        attempt = len(frame.get("attempts", [])) + 1
        frame.setdefault("attempts", []).append({
            "attempt": attempt,
            "status": "generating",
            "created_at": now_stamp(),
            "reference_inputs": reference_inputs or [],
            "omitted_reference_inputs": omitted_reference_inputs or [],
        })
        frame.update(status="generating", last_error="")
        state["image_qa"] = {"status": "not_run"}
        state["updated_at"] = now_stamp()
        self._write(state)
        return attempt

    def complete_frame(
        self,
        run_id: str,
        scene_id: str,
        attempt: int,
        content: bytes,
        extension: str,
    ) -> Path:
        if not content:
            raise ValueError("Нельзя сохранить пустой кадр.")
        safe_extension = extension.lower() if extension.lower() in {".png", ".jpg", ".jpeg", ".webp"} else ".png"
        run_path = self._run_path(run_id)
        frame_dir = run_path / "frames" / scene_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        target = frame_dir / f"frame-v{attempt:03d}{safe_extension}"
        target.write_bytes(content)
        prompt_path = frame_dir / f"prompt-v{attempt:03d}.txt"

        state = self.load(run_id)
        frame = self._require_frame(state, scene_id)
        prompt_path.write_text(str(frame.get("prompt", "")).strip() + "\n", encoding="utf-8", newline="\n")
        entry = self._attempt(frame, attempt)
        entry.update(
            status="ready",
            file=str(target.relative_to(run_path)).replace("\\", "/"),
            prompt_file=str(prompt_path.relative_to(run_path)).replace("\\", "/"),
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            completed_at=now_stamp(),
        )
        frame.update(
            status="ready",
            latest_file=entry["file"],
            selected=True,
            last_error="",
        )
        selected = [item for item in state.get("selected_frame_ids", []) if item != scene_id]
        selected.append(scene_id)
        state["selected_frame_ids"] = self._ordered_ids(state, selected)
        state["updated_at"] = now_stamp()
        self._write(state)
        return target

    def fail_frame(self, run_id: str, scene_id: str, attempt: int, error: str) -> None:
        state = self.load(run_id)
        frame = self._require_frame(state, scene_id)
        clean_error = error.strip()[:500] or "Неизвестная ошибка"
        self._attempt(frame, attempt).update(
            status="failed", error=clean_error, completed_at=now_stamp()
        )
        frame.update(status="failed", last_error=clean_error)
        state["updated_at"] = now_stamp()
        self._write(state)

    def cancel_frame(self, run_id: str, scene_id: str, attempt: int) -> None:
        state = self.load(run_id)
        frame = self._require_frame(state, scene_id)
        message = "Генерация остановлена пользователем."
        self._attempt(frame, attempt).update(
            status="cancelled", error=message, completed_at=now_stamp()
        )
        frame.update(status="cancelled", last_error=message)
        state["updated_at"] = now_stamp()
        self._write(state)

    def set_selected(self, run_id: str, scene_id: str, selected: bool) -> dict[str, Any]:
        state = self.load(run_id)
        frame = self._require_frame(state, scene_id)
        if selected and frame.get("status") != "ready":
            raise ValueError("Выбрать можно только готовый кадр.")
        frame["selected"] = selected
        ids = [item for item in state.get("selected_frame_ids", []) if item != scene_id]
        if selected:
            ids.append(scene_id)
        state["selected_frame_ids"] = self._ordered_ids(state, ids)
        state["updated_at"] = now_stamp()
        self._write(state)
        return state

    def select_all_ready(self, run_id: str) -> dict[str, Any]:
        state = self.load(run_id)
        selected: list[str] = []
        for scene_id, frame in state.get("frames", {}).items():
            ready = frame.get("status") == "ready"
            frame["selected"] = ready
            if ready:
                selected.append(scene_id)
        state["selected_frame_ids"] = self._ordered_ids(state, selected)
        state["updated_at"] = now_stamp()
        self._write(state)
        return state

    def frame_path(self, run_id: str, scene_id: str) -> Path | None:
        state = self.load(run_id)
        frame = self._require_frame(state, scene_id)
        relative = str(frame.get("latest_file", "")).strip()
        path = self._run_path(run_id) / relative if relative else None
        return path if path and path.is_file() else None

    def set_video_settings(
        self,
        run_id: str,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist one run's provider profile without risking duplicate paid tasks."""

        state = self.load(run_id)
        jobs = state.get("video_jobs", {})
        if isinstance(jobs, dict) and any(
            isinstance(job, dict)
            and (
                job.get("external_task_id")
                or job.get("submission_state") in {"submitting", "unknown"}
            )
            for job in jobs.values()
        ):
            raise ProductionContractError(
                "Нельзя сменить модель или качество, пока существуют отправленные "
                "или неоднозначные внешние video tasks."
            )
        required = {
            "code",
            "model",
            "model_label",
            "quality_label",
            "mode",
            "resolution",
            "duration_seconds",
            "aspect_ratio",
            "sound_enabled",
        }
        if not required.issubset(settings):
            raise ProductionContractError("Профиль видеогенерации заполнен не полностью.")
        normalized = {key: settings[key] for key in required}
        # Legacy runs predate explicit provider routing and deterministic local seeds.
        normalized["provider"] = str(settings.get("provider") or "polza")
        try:
            normalized["seed"] = int(settings.get("seed", 0))
        except (TypeError, ValueError) as exc:
            raise ProductionContractError("Seed видеопрофиля должен быть целым числом.") from exc
        if not 0 <= normalized["seed"] <= 2**63 - 1:
            raise ProductionContractError("Seed видеопрофиля находится вне разрешённого диапазона.")
        changed = state.get("video_settings") != normalized
        state["video_settings"] = normalized
        if changed:
            # Files remain as an audit trail; active state is cleared because prompts
            # and approvals are model-specific and cannot be reused safely.
            state["video_prompts"] = {}
            state["video_clip_ids"] = []
            state["video_prompt_qa"] = {"status": "not_run"}
            state["video_jobs"] = {}
            state["video_approval"] = {"status": "not_requested"}
        state["updated_at"] = now_stamp()
        self._write(state)
        return state

    def save_video_prompts(self, run_id: str, prompts: list[dict[str, Any]]) -> Path:
        state = self.load(run_id)
        settings = state.get("video_settings", {})
        if not isinstance(settings, dict) or not settings.get("model"):
            raise ProductionContractError(
                "Сначала выбери видеомодель и качество для этого запуска."
            )
        jobs = state.get("video_jobs", {})
        if isinstance(jobs, dict) and any(
            isinstance(job, dict)
            and (
                job.get("external_task_id")
                or job.get("submission_state") in {"submitting", "unknown"}
            )
            for job in jobs.values()
        ):
            raise ProductionContractError(
                "Нельзя заменить видеопромпты, пока у запуска есть отправленные "
                "или неоднозначные внешние video tasks."
            )
        selected = list(state.get("selected_frame_ids", []))
        prompt_ids = [
            str(item.get("clip_id") or item.get("scene_id") or "").strip().upper()
            for item in prompts
        ]
        if any(not item for item in prompt_ids) or len(set(prompt_ids)) != len(prompt_ids):
            raise ProductionContractError("Каждый видеоклип должен иметь уникальный непустой ID.")
        covered_scene_ids: list[str] = []
        for prompt_index, item in enumerate(prompts):
            raw_source_ids = item.get("source_scene_ids")
            source_ids = (
                [str(value).strip().upper() for value in raw_source_ids]
                if isinstance(raw_source_ids, list) and raw_source_ids
                else [str(item.get("scene_id", "")).strip().upper()]
            )
            start_scene_id = str(item.get("start_scene_id") or source_ids[0]).strip().upper()
            if start_scene_id != source_ids[0]:
                raise ProductionContractError(
                    f"{prompt_ids[prompt_index]}: "
                    "start_scene_id должен быть первым исходным кадром."
                )
            if self.frame_path(run_id, start_scene_id) is None:
                raise ProductionContractError(
                    f"Стартовый кадр {start_scene_id} для видеоклипа не найден."
                )
            covered_scene_ids.extend(source_ids)
        if covered_scene_ids != selected:
            raise ProductionContractError(
                "Видеоклипы должны покрывать все выбранные кадры ровно один раз и по порядку."
            )
        if any(str(item.get("model_id", "")) != settings["model"] for item in prompts):
            raise ProductionContractError(
                "Модель видеопромптов не совпадает с профилем запуска."
            )
        run_path = self._run_path(run_id)
        prompt_dir = run_path / "video-prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        index: list[dict[str, Any]] = []
        for prompt_id, item in zip(prompt_ids, prompts):
            target = prompt_dir / f"{prompt_id}.json"
            target.write_text(
                json.dumps(item, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            index.append(item)
        combined = prompt_dir / "ALL-VIDEO-PROMPTS.json"
        combined.write_text(
            json.dumps({"schema_version": 2, "clips": index}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        state["video_prompts"] = {
            prompt_id: item for prompt_id, item in zip(prompt_ids, prompts)
        }
        state["video_clip_ids"] = prompt_ids
        state["video_prompt_qa"] = {"status": "not_run"}
        # Failed submissions without an external task ID are safe to replace when
        # the user switches the video model and regenerates model-specific prompts.
        state["video_jobs"] = {}
        state["video_approval"] = {"status": "not_requested"}
        state["updated_at"] = now_stamp()
        self._write(state)
        return combined

    def write_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = now_stamp()
        self._write(state)

    def save_image_qa(self, run_id: str, report: dict[str, Any]) -> Path:
        """Persist the post-image gate beside the generated assets and in durable state."""

        state = self.load(run_id)
        run_path = self._run_path(run_id)
        target = run_path / "06-IMAGE-QA.json"
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        state["image_qa"] = report
        state["updated_at"] = now_stamp()
        self._write(state)
        return target

    def save_video_prompt_qa(self, run_id: str, report: dict[str, Any]) -> Path:
        """Persist the structural continuity gate for logical video clips."""

        state = self.load(run_id)
        run_path = self._run_path(run_id)
        target = run_path / "07-VIDEO-PROMPT-QA.json"
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        state["video_prompt_qa"] = report
        state["updated_at"] = now_stamp()
        self._write(state)
        return target

    def _empty_state(self, run_id: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": run_id,
            "scenes": [],
            "frames": {},
            "references": {},
            "locations": {},
            "selected_frame_ids": [],
            "video_settings": {},
            "video_prompts": {},
            "video_clip_ids": [],
            "video_jobs": {},
            "video_approval": {"status": "not_requested"},
            "image_qa": {"status": "not_run"},
            "video_prompt_qa": {"status": "not_run"},
            "created_at": now_stamp(),
            "updated_at": now_stamp(),
        }

    def _state_path(self, run_id: str) -> Path:
        return self._run_path(run_id) / "production.json"

    def _run_path(self, run_id: str) -> Path:
        normalized = run_id.strip().upper()
        if not re.fullmatch(r"CF-\d{8}-\d{3}", normalized):
            raise ValueError(f"Недопустимый run ID: {run_id}")
        path = self.runs_path / normalized
        if not path.is_dir() or not (path / "run.json").is_file():
            raise ValueError(f"Запуск не найден: {normalized}")
        return path

    def _write(self, state: dict[str, Any]) -> None:
        target = self._state_path(str(state["run_id"]))
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(target)

    @staticmethod
    def _require_frame(state: dict[str, Any], scene_id: str) -> dict[str, Any]:
        frames = state.get("frames", {})
        frame = frames.get(scene_id) if isinstance(frames, dict) else None
        if not isinstance(frame, dict):
            raise ValueError(f"Сцена не найдена: {scene_id}")
        return frame

    @staticmethod
    def _require_reference(state: dict[str, Any], reference_id: str) -> dict[str, Any]:
        references = state.get("references", {})
        reference = references.get(reference_id) if isinstance(references, dict) else None
        if not isinstance(reference, dict):
            raise ValueError(f"Референс не найден: {reference_id}")
        return reference

    @staticmethod
    def _attempt(frame: dict[str, Any], attempt: int) -> dict[str, Any]:
        for item in frame.get("attempts", []):
            if item.get("attempt") == attempt:
                return item
        raise ValueError(f"Попытка кадра не найдена: {attempt}")

    @staticmethod
    def _ordered_ids(state: dict[str, Any], ids: list[str]) -> list[str]:
        order = {
            str(scene.get("scene_id")): int(scene.get("order", index))
            for index, scene in enumerate(state.get("scenes", []), 1)
            if isinstance(scene, dict)
        }
        return sorted(set(ids), key=lambda item: order.get(item, 999999))
