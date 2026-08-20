from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .llm import LlmClient
from .production import ProductionContractError, ProductionStore
from .video_prompting import ImageInspector


@dataclass(frozen=True)
class ImageQaResult:
    verdict: str
    summary: str
    ready_for_video_prompts: bool
    issues: tuple[dict[str, str], ...]
    checked_items: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "verdict": self.verdict,
            "summary": self.summary,
            "ready_for_video_prompts": self.ready_for_video_prompts,
            "issues": list(self.issues),
            "checked_items": self.checked_items,
        }


class PostImageQa:
    """Compare real generated files with durable scene and reference contracts."""

    def __init__(
        self,
        store: ProductionStore,
        inspector: ImageInspector,
        llm: LlmClient,
    ) -> None:
        self.store = store
        self.inspector = inspector
        self.llm = llm

    def run(self, run_id: str) -> ImageQaResult:
        if not self.inspector.is_configured or not self.llm.is_configured:
            raise ProductionContractError(
                "Для визуальной проверки нужны Codex image inspector и production LLM."
            )
        state = self.store.load(run_id)
        references = state.get("references", {})
        frames = state.get("frames", {})
        if not isinstance(references, dict) or not isinstance(frames, dict) or not frames:
            raise ProductionContractError("План изображений ещё не подготовлен.")
        missing = [
            item_id
            for collection in (references, frames)
            for item_id, item in collection.items()
            if not isinstance(item, dict) or item.get("status") != "ready"
        ]
        if missing:
            result = ImageQaResult(
                verdict="fail",
                summary="Не все запланированные изображения готовы.",
                ready_for_video_prompts=False,
                issues=tuple(
                    {"item_id": item_id, "severity": "error", "message": "Файл не готов."}
                    for item_id in missing
                ),
                checked_items=0,
            )
            self.store.save_image_qa(run_id, result.to_dict())
            return result

        inspected_references: dict[str, str] = {}
        for reference_id in references:
            path = self.store.reference_path(run_id, reference_id)
            if path is None:
                raise ProductionContractError(f"Файл {reference_id} отсутствует на диске.")
            inspected_references[reference_id] = self.inspector.inspect(path)
            self.store.set_reference_description(
                run_id, reference_id, inspected_references[reference_id]
            )

        scene_map = {item.scene_id: item for item in self.store.scenes(run_id)}
        evidence: list[dict[str, Any]] = []
        for scene_id, frame in frames.items():
            path = self.store.frame_path(run_id, scene_id)
            if path is None:
                raise ProductionContractError(f"Файл кадра {scene_id} отсутствует на диске.")
            scene = scene_map.get(scene_id)
            if scene is None:
                raise ProductionContractError(f"Контракт сцены {scene_id} отсутствует.")
            evidence.append(
                {
                    "scene_id": scene_id,
                    "expected_visual": scene.visual,
                    "expected_prompt": scene.image_prompt,
                    "location_id": scene.location_id,
                    "reference_ids": list(scene.reference_ids),
                    "reference_descriptions": {
                        item: inspected_references.get(item, "") for item in scene.reference_ids
                    },
                    "observed_frame": self.inspector.inspect(path),
                }
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict post-generation visual QA gate. Compare only the supplied factual "
                    "image observations with expected scenes and canonical references. Return JSON only: "
                    "{verdict:'pass'|'warning'|'fail',summary,ready_for_video_prompts:boolean,"
                    "issues:[{item_id,severity:'warning'|'error',message}]}. Check character identity, "
                    "important objects, location continuity, scene meaning, and obvious contradictions. "
                    "Do not claim pixel-perfect identity from textual evidence. Use warning when evidence "
                    "is insufficient and fail only for a concrete contradiction or missing required entity."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "canonical_references": inspected_references,
                        "frames": evidence,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        payload = self._parse_json(self.llm.chat(messages))
        verdict = str(payload.get("verdict", "")).strip().lower()
        if verdict not in {"pass", "warning", "fail"}:
            raise ProductionContractError("Visual QA вернул недопустимый verdict.")
        raw_issues = payload.get("issues", [])
        if not isinstance(raw_issues, list):
            raise ProductionContractError("Visual QA issues должен быть списком.")
        known_ids = set(references) | set(frames)
        issues: list[dict[str, str]] = []
        for raw in raw_issues:
            if not isinstance(raw, dict):
                continue
            item_id = str(raw.get("item_id", "")).strip().upper()
            severity = str(raw.get("severity", "warning")).strip().lower()
            message = str(raw.get("message", "")).strip()
            if item_id not in known_ids or severity not in {"warning", "error"} or not message:
                raise ProductionContractError("Visual QA вернул некорректную проблему.")
            issues.append({"item_id": item_id, "severity": severity, "message": message})
        ready = bool(payload.get("ready_for_video_prompts")) and verdict != "fail"
        result = ImageQaResult(
            verdict=verdict,
            summary=str(payload.get("summary", "")).strip() or "Проверка завершена.",
            ready_for_video_prompts=ready,
            issues=tuple(issues),
            checked_items=len(inspected_references) + len(evidence),
        )
        self.store.save_image_qa(run_id, result.to_dict())
        return result

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        source = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", source, re.DOTALL | re.I)
        if fenced:
            source = fenced.group(1)
        try:
            payload = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ProductionContractError("Visual QA не вернул валидный JSON.") from exc
        if not isinstance(payload, dict):
            raise ProductionContractError("Visual QA JSON должен быть объектом.")
        return payload
