from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Callable

from .image_generation import (
    ImageClient,
    ImageGenerationError,
    ImageReference,
    MAX_IMAGE_REFERENCE_FILES,
    build_single_frame_prompt,
    generate_image_with_retry,
)
from .production import ProductionStore, SceneSpec


@dataclass(frozen=True)
class FrameResult:
    scene_id: str
    status: str
    path: str = ""
    error: str = ""


@dataclass(frozen=True)
class FrameBatchResult:
    results: tuple[FrameResult, ...]
    was_cancelled: bool = False

    @property
    def ready_count(self) -> int:
        return sum(item.status == "ready" for item in self.results)

    @property
    def failed_count(self) -> int:
        return sum(item.status == "failed" for item in self.results)


@dataclass(frozen=True)
class FrameReferencePlan:
    attached: tuple[ImageReference, ...]
    omitted: tuple[tuple[str, str, str], ...]


ProgressCallback = Callable[[int, int, FrameResult], None]
CancelCheck = Callable[[], bool]


class FrameBatchGenerator:
    """Generate each scene independently so one provider error cannot abort the batch."""

    def __init__(self, store: ProductionStore, image_client: ImageClient):
        self.store = store
        self.image_client = image_client

    def generate(
        self,
        run_id: str,
        *,
        scene_ids: list[str] | None = None,
        aspect_ratio: str = "9:16",
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
    ) -> FrameBatchResult:
        scenes = self.store.scenes(run_id)
        if scene_ids is not None:
            requested = set(scene_ids)
            scenes = [scene for scene in scenes if scene.scene_id in requested]
            missing = requested - {scene.scene_id for scene in scenes}
            if missing:
                raise ValueError(f"Неизвестные сцены: {', '.join(sorted(missing))}")
        results: list[FrameResult] = []
        total = len(scenes)
        for index, scene in enumerate(scenes, 1):
            if should_cancel and should_cancel():
                return FrameBatchResult(tuple(results), was_cancelled=True)
            result = self._generate_one(run_id, scene, aspect_ratio)
            results.append(result)
            if on_progress:
                on_progress(index, total, result)
            if result.status == "cancelled":
                return FrameBatchResult(tuple(results), was_cancelled=True)
        return FrameBatchResult(tuple(results))

    def _generate_one(self, run_id: str, scene: SceneSpec, aspect_ratio: str) -> FrameResult:
        try:
            reference_plan = self._reference_plan(run_id, scene)
            references = reference_plan.attached
            attempt = self.store.start_frame(
                run_id,
                scene.scene_id,
                reference_inputs=[
                    {
                        "reference_id": item.reference_id,
                        "role": item.role,
                        "file": str(item.path),
                    }
                    for item in references
                ],
                omitted_reference_inputs=[
                    {
                        "reference_id": reference_id,
                        "role": role,
                        "description": description,
                    }
                    for reference_id, role, description in reference_plan.omitted
                ],
            )
            prompt = build_single_frame_prompt(
                scene_id=scene.scene_id,
                visual=scene.visual,
                image_prompt=scene.image_prompt,
                continuity=scene.continuity,
                aspect_ratio=aspect_ratio,
                reference_inputs=[
                    (item.reference_id, item.role) for item in references
                ],
                text_only_references=reference_plan.omitted,
            )
            image = generate_image_with_retry(
                self.image_client,
                prompt,
                references,
            )
            path = self.store.complete_frame(
                run_id,
                scene.scene_id,
                attempt,
                image.content,
                image.extension,
            )
            return FrameResult(scene.scene_id, "ready", path=str(path))
        except ImageGenerationError as exc:
            attempt = locals().get("attempt")
            if attempt is None:
                attempt = self.store.start_frame(run_id, scene.scene_id)
            if exc.code == "generation_cancelled":
                self.store.cancel_frame(run_id, scene.scene_id, attempt)
                return FrameResult(scene.scene_id, "cancelled", error=str(exc))
            self.store.fail_frame(run_id, scene.scene_id, attempt, str(exc))
            return FrameResult(scene.scene_id, "failed", error=str(exc))
        except Exception as exc:
            attempt = locals().get("attempt")
            if attempt is None:
                attempt = self.store.start_frame(run_id, scene.scene_id)
            self.store.fail_frame(run_id, scene.scene_id, attempt, str(exc))
            return FrameResult(scene.scene_id, "failed", error=str(exc))

    def _reference_plan(
        self, run_id: str, scene: SceneSpec
    ) -> FrameReferencePlan:
        """Prioritize visual anchors within the image tool's five-file limit."""

        state = self.store.load(run_id)
        candidates: list[tuple[int, int, ImageReference, str]] = []
        stored = state.get("references", {})
        kind_priority = {
            "character": 0,
            "location-continuity": 1,
            "environment": 2,
            "object": 3,
            "style": 4,
        }
        for position, reference_id in enumerate(scene.reference_ids):
            item = stored.get(reference_id, {}) if isinstance(stored, dict) else {}
            path = self.store.reference_path(run_id, reference_id)
            if path is None:
                raise ImageGenerationError(
                    f"Сначала создай обязательный референс {reference_id} для {scene.scene_id}.",
                    code="required_reference_not_ready",
                )
            kind = str(item.get("kind", "reference"))
            description = str(
                item.get("prompt") or item.get("name") or "Declared visual continuity anchor."
            )
            candidates.append(
                (
                    kind_priority.get(kind, 5),
                    position,
                    ImageReference(reference_id, kind, path),
                    description,
                )
            )

        location_source = self._location_reference_scene(run_id, state, scene)
        if location_source:
            path = self.store.frame_path(run_id, location_source)
            if path is not None:
                candidates.append(
                    (
                        kind_priority["location-continuity"],
                        len(candidates),
                        ImageReference(
                            f"FRAME-{location_source}",
                            "location-continuity",
                            path,
                        ),
                        (
                            f"Preserve the spatial layout and environment from scene "
                            f"{location_source}."
                        ),
                    )
                )

        candidates.sort(key=lambda item: (item[0], item[1]))
        unique: list[tuple[ImageReference, str]] = []
        seen_paths: set[Path] = set()
        for _, _, reference, description in candidates:
            resolved = reference.path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            unique.append((reference, description))

        attached = tuple(
            reference
            for reference, _ in unique[:MAX_IMAGE_REFERENCE_FILES]
        )
        omitted = tuple(
            (reference.reference_id, reference.role, description)
            for reference, description in unique[MAX_IMAGE_REFERENCE_FILES:]
        )
        return FrameReferencePlan(attached, omitted)

    def _reference_inputs(
        self, run_id: str, scene: SceneSpec
    ) -> tuple[ImageReference, ...]:
        """Compatibility helper returning only files that will be attached."""

        return self._reference_plan(run_id, scene).attached

    def _location_reference_scene(
        self, run_id: str, state: dict, scene: SceneSpec
    ) -> str:
        scenes = self.store.scenes(run_id)
        earlier = [item for item in scenes if item.order < scene.order]
        if scene.location_reference_scene_id:
            source = next(
                (
                    item
                    for item in earlier
                    if item.scene_id == scene.location_reference_scene_id
                    and (not scene.location_id or item.location_id == scene.location_id)
                ),
                None,
            )
            return source.scene_id if source is not None else ""
        if scene.location_id:
            locations = state.get("locations", {})
            location = locations.get(scene.location_id, {}) if isinstance(locations, dict) else {}
            candidates = [
                str(location.get("canonical_scene_id", "")),
                *reversed([item.scene_id for item in earlier if item.location_id == scene.location_id]),
            ]
            for candidate in candidates:
                if candidate and candidate != scene.scene_id and self.store.frame_path(run_id, candidate):
                    return candidate
            return ""

        current_environment = scene.continuity.get("environment", "")
        best: tuple[float, str] = (0.0, "")
        for candidate in earlier:
            if self.store.frame_path(run_id, candidate.scene_id) is None:
                continue
            score = _location_similarity(
                current_environment,
                candidate.continuity.get("environment", ""),
            )
            if score > best[0]:
                best = (score, candidate.scene_id)
        return best[1] if best[0] >= 0.6 else ""


def _location_similarity(left: str, right: str) -> float:
    """Conservative fallback for legacy runs that lack explicit location IDs."""

    stop = {
        "тот", "та", "то", "же", "и", "в", "на", "с", "со", "из", "для",
        "the", "a", "an", "same", "with", "in", "on", "of",
    }
    left_tokens = {
        token for token in re.findall(r"[a-zа-яё]{3,}", left.lower()) if token not in stop
    }
    right_tokens = {
        token for token in re.findall(r"[a-zа-яё]{3,}", right.lower()) if token not in stop
    }
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if len(overlap) < 2:
        return 0.0
    return len(overlap) / min(len(left_tokens), len(right_tokens))
