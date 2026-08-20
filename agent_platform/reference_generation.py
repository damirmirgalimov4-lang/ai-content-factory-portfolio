from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .image_generation import (
    ImageClient,
    ImageGenerationError,
    ImageReference,
    generate_image_with_retry,
)
from .production import ProductionStore


@dataclass(frozen=True)
class ReferenceResult:
    reference_id: str
    kind: str
    status: str
    path: str = ""
    error: str = ""


@dataclass(frozen=True)
class ReferenceBatchResult:
    results: tuple[ReferenceResult, ...]
    was_cancelled: bool = False

    @property
    def ready_count(self) -> int:
        return sum(item.status == "ready" for item in self.results)

    @property
    def failed_count(self) -> int:
        return sum(item.status == "failed" for item in self.results)


ProgressCallback = Callable[[int, int, ReferenceResult], None]
CancelCheck = Callable[[], bool]


class ReferenceBatchGenerator:
    """Generate reusable identity/style assets before scene frames."""

    def __init__(self, store: ProductionStore, image_client: ImageClient):
        self.store = store
        self.image_client = image_client

    def generate(
        self,
        run_id: str,
        *,
        reference_ids: list[str] | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
    ) -> ReferenceBatchResult:
        state = self.store.load(run_id)
        references = state.get("references", {})
        if not isinstance(references, dict):
            references = {}
        ordered = list(references.values())
        if reference_ids is not None:
            requested = {item.upper() for item in reference_ids}
            ordered = [
                item for item in ordered if str(item.get("reference_id", "")) in requested
            ]
            missing = requested - {
                str(item.get("reference_id", "")) for item in ordered
            }
            if missing:
                raise ValueError(f"Неизвестные референсы: {', '.join(sorted(missing))}")
        ordered = _order_reference_dependencies(ordered)

        results: list[ReferenceResult] = []
        for index, item in enumerate(ordered, 1):
            if should_cancel and should_cancel():
                return ReferenceBatchResult(tuple(results), was_cancelled=True)
            result = self._generate_one(run_id, item)
            results.append(result)
            if on_progress:
                on_progress(index, len(ordered), result)
            if result.status == "cancelled":
                return ReferenceBatchResult(tuple(results), was_cancelled=True)
        return ReferenceBatchResult(tuple(results))

    def _generate_one(self, run_id: str, item: dict) -> ReferenceResult:
        reference_id = str(item.get("reference_id", ""))
        kind = str(item.get("kind", "character"))
        try:
            references = self._base_reference_input(run_id, item)
            attempt = self.store.start_reference(
                run_id,
                reference_id,
                reference_inputs=[
                    {
                        "reference_id": reference.reference_id,
                        "role": reference.role,
                        "file": str(reference.path),
                    }
                    for reference in references
                ],
            )
            prompt = build_reference_prompt(
                reference_id=reference_id,
                kind=kind,
                name=str(item.get("name", reference_id)),
                source_prompt=str(item.get("prompt", "")),
                identity_group=str(item.get("identity_group", "")),
                state_label=str(item.get("state_label", "")),
                base_reference_id=str(item.get("base_reference_id", "")),
            )
            image = generate_image_with_retry(
                self.image_client,
                prompt,
                references,
            )
            path = self.store.complete_reference(
                run_id,
                reference_id,
                attempt,
                image.content,
                image.extension,
            )
            return ReferenceResult(reference_id, kind, "ready", path=str(path))
        except ImageGenerationError as exc:
            attempt = locals().get("attempt")
            if attempt is None:
                attempt = self.store.start_reference(run_id, reference_id)
            if exc.code == "generation_cancelled":
                self.store.cancel_reference(run_id, reference_id, attempt)
                return ReferenceResult(reference_id, kind, "cancelled", error=str(exc))
            self.store.fail_reference(run_id, reference_id, attempt, str(exc))
            return ReferenceResult(reference_id, kind, "failed", error=str(exc))
        except Exception as exc:
            attempt = locals().get("attempt")
            if attempt is None:
                attempt = self.store.start_reference(run_id, reference_id)
            self.store.fail_reference(run_id, reference_id, attempt, str(exc))
            return ReferenceResult(reference_id, kind, "failed", error=str(exc))

    def _base_reference_input(
        self, run_id: str, item: dict
    ) -> tuple[ImageReference, ...]:
        base_reference_id = str(item.get("base_reference_id", "")).strip().upper()
        if not base_reference_id:
            return ()
        path = self.store.reference_path(run_id, base_reference_id)
        if path is None:
            raise ImageGenerationError(
                f"Сначала создай базовое состояние {base_reference_id}.",
                code="required_reference_not_ready",
            )
        return (ImageReference(base_reference_id, "identity-base", path),)


def build_reference_prompt(
    *,
    reference_id: str,
    kind: str,
    name: str,
    source_prompt: str,
    identity_group: str = "",
    state_label: str = "",
    base_reference_id: str = "",
) -> str:
    """Wrap role-specific source text in one stable production template."""

    source = source_prompt.strip()
    inherited_identity = ""
    if base_reference_id:
        inherited_identity = (
            f"\nThis is the '{state_label}' state of {identity_group}. Use the attached "
            f"{base_reference_id} image as the strict base identity. Preserve the exact face or body "
            "geometry, silhouette, proportions, permanent markings and recognisable design. Change "
            "only the story-state details explicitly requested below. Do not create a different subject.\n"
        )
    if kind == "character":
        return (
            f"Create the canonical production character identity sheet {reference_id}: {name}.\n"
            "This one image is deliberately a technical multi-view reference sheet, not a story scene.\n"
            "Use a pure white studio background (#FFFFFF) and a clean two-row contact-sheet layout. "
            "Top row: four full-body views of the exact same character in this order: front, left "
            "profile, right profile, back. Bottom row: three close-up head views: front, left profile, "
            "right profile. Keep a relaxed neutral A-pose, identical body proportions, face structure, "
            "fur or skin pattern, hairstyle, outfit, colors, materials, accessories, age and rendering "
            "style in every panel. Use soft even neutral studio lighting, clear silhouettes, consistent "
            "scale and minimal shadows. Do not redesign, simplify, beautify, age, recolor, replace the "
            "outfit, add props, add labels, or create alternate versions.\n\n"
            f"{inherited_identity}\nCANONICAL CHARACTER DESCRIPTION:\n{source}"
        )
    if kind == "environment":
        return (
            f"Create the canonical environment reference {reference_id}: {name}.\n"
            "Show one coherent wide production view of the location with readable spatial layout, "
            "architecture or terrain, palette, materials, weather and lighting. No characters, no "
            "contact sheet, no split screen, no text, no labels, no logo and no watermark. This image "
            "will be reused as a visual location anchor for later scene frames.\n\n"
            f"CANONICAL ENVIRONMENT DESCRIPTION:\n{source}"
        )
    if kind == "style":
        return (
            f"Create one canonical visual style guide image {reference_id}: {name}.\n"
            "Use one full-frame composition that clearly establishes palette, materials, lighting, "
            "contrast and rendering language. No characters, grid, labels, captions, logo or watermark.\n\n"
            f"STYLE DESCRIPTION:\n{source}"
        )
    return (
        f"Create a clean canonical object reference image {reference_id}: {name}.\n"
        "Show one object on a pure white background, fully visible, with neutral lighting and no text, "
        "labels, logo, watermark or alternate designs. Preserve exact shape, colors and materials.\n\n"
        f"{inherited_identity}\nOBJECT DESCRIPTION:\n{source}"
    )


def _order_reference_dependencies(items: list[dict]) -> list[dict]:
    """Generate a base identity before any visual state derived from it."""

    by_id = {
        str(item.get("reference_id", "")).strip().upper(): item for item in items
    }
    ordered: list[dict] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(reference_id: str) -> None:
        if reference_id in visited:
            return
        if reference_id in visiting:
            raise ValueError("Найдена циклическая зависимость референсов.")
        visiting.add(reference_id)
        item = by_id[reference_id]
        base_id = str(item.get("base_reference_id", "")).strip().upper()
        if base_id in by_id:
            visit(base_id)
        visiting.remove(reference_id)
        visited.add(reference_id)
        ordered.append(item)

    for reference_id in by_id:
        visit(reference_id)
    return ordered
