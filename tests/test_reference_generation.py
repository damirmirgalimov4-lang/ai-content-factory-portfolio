from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_platform.content_factory import ContentFactoryStore
from agent_platform.frame_generation import FrameBatchGenerator
from agent_platform.image_generation import GeneratedImage, ImageReference
from agent_platform.production import (
    LocationSpec,
    ProductionStore,
    ReferenceSpec,
    SceneSpec,
    merge_image_prompt_contract,
    parse_reference_plan,
    parse_scene_contract,
)
from agent_platform.reference_generation import (
    ReferenceBatchGenerator,
    build_reference_prompt,
)


class ReferenceAwareImageClient:
    is_configured = True

    def __init__(self):
        self.calls: list[tuple[str, tuple[ImageReference, ...]]] = []

    def generate(self, prompt: str, references=()) -> GeneratedImage:
        self.calls.append((prompt, tuple(references)))
        return GeneratedImage(content=f"image-{len(self.calls)}".encode())


def scene(
    index: int,
    *,
    reference_ids: tuple[str, ...] = (),
    location_id: str = "",
    environment: str = "studio",
) -> SceneSpec:
    return SceneSpec(
        scene_id=f"S{index:02d}",
        order=index,
        duration_seconds=3,
        purpose=f"Scene {index}",
        visual=f"Visible scene {index}",
        physical_action="one action",
        camera_movement="static",
        voiceover="",
        on_screen_text="",
        sound="",
        transition="cut",
        continuity={"environment": environment},
        image_prompt=f"one frame {index}",
        reference_ids=reference_ids,
        location_id=location_id,
    )


class ReferenceContractTest(unittest.TestCase):
    def test_optional_scene_ids_accept_common_empty_llm_markers(self) -> None:
        item = scene(1).to_dict()
        item.update(
            location_id=None,
            location_reference_scene_id="N/A",
            reference_ids=[None, "null", ""],
        )

        parsed = SceneSpec.from_dict(item, 1)

        self.assertEqual(parsed.location_id, "")
        self.assertEqual(parsed.location_reference_scene_id, "")
        self.assertEqual(parsed.reference_ids, ())

    def test_canonical_scene_self_reference_is_normalized_to_empty(self) -> None:
        item = scene(1, location_id="LOC-HOME").to_dict()
        item["location_reference_scene_id"] = "S01"

        parsed = SceneSpec.from_dict(item, 1)

        self.assertEqual(parsed.location_reference_scene_id, "")

    def test_location_reference_must_point_to_earlier_scene_in_same_location(self) -> None:
        base = [scene(1), scene(2)]
        payload = {
            "schema_version": 2,
            "references": [],
            "locations": [
                {
                    "location_id": "LOC-A",
                    "name": "A",
                    "description": "first",
                    "scene_ids": ["S01"],
                    "canonical_scene_id": "S01",
                },
                {
                    "location_id": "LOC-B",
                    "name": "B",
                    "description": "second",
                    "scene_ids": ["S02"],
                    "canonical_scene_id": "S02",
                },
            ],
            "scenes": [
                {
                    "scene_id": "S01",
                    "image_prompt": "first",
                    "reference_ids": [],
                    "location_id": "LOC-A",
                    "location_reference_scene_id": "",
                },
                {
                    "scene_id": "S02",
                    "image_prompt": "second",
                    "reference_ids": [],
                    "location_id": "LOC-B",
                    "location_reference_scene_id": "S01",
                },
            ],
        }
        artifact = "IMAGE_PROMPT_CONTRACT\n```json\n" + json.dumps(payload) + "\n```"
        merged = merge_image_prompt_contract(base, artifact)

        with self.assertRaisesRegex(ValueError, "другой локации"):
            parse_reference_plan(merged, artifact)

    def test_compound_reference_kind_uses_explicit_reference_id_hint(self) -> None:
        reference = ReferenceSpec.from_dict(
            {
                "reference_id": "REF-CHAR-EXTRAS-01",
                "kind": "character_environment_style",
                "name": "Background extras",
                "prompt": "consistent supporting cast",
                "scene_ids": ["S01"],
            }
        )

        self.assertEqual(reference.kind, "character")

    def test_ambiguous_reference_kind_without_id_hint_stays_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "kind должен быть"):
            ReferenceSpec.from_dict(
                {
                    "reference_id": "REF-MIXED-01",
                    "kind": "character_environment_style",
                    "name": "Mixed guide",
                    "prompt": "ambiguous guide",
                    "scene_ids": ["S01"],
                }
            )

    def test_v2_contract_links_multiple_characters_and_location(self) -> None:
        base = [scene(1), scene(2)]
        payload = {
            "schema_version": 2,
            "references": [
                {
                    "reference_id": "REF-HERO",
                    "kind": "character",
                    "name": "Hero",
                    "prompt": "same hero",
                    "scene_ids": ["S01", "S02"],
                },
                {
                    "reference_id": "REF-FRIEND",
                    "kind": "character",
                    "name": "Friend",
                    "prompt": "same friend",
                    "scene_ids": ["S02"],
                },
            ],
            "locations": [
                {
                    "location_id": "LOC-HOME",
                    "name": "Home",
                    "description": "same room",
                    "scene_ids": ["S01", "S02"],
                    "canonical_scene_id": "S01",
                }
            ],
            "scenes": [
                {
                    "scene_id": "S01",
                    "image_prompt": "hero in room",
                    "reference_ids": ["REF-HERO"],
                    "location_id": "LOC-HOME",
                    "location_reference_scene_id": "",
                },
                {
                    "scene_id": "S02",
                    "image_prompt": "hero and friend in same room",
                    "reference_ids": ["REF-HERO", "REF-FRIEND"],
                    "location_id": "LOC-HOME",
                    "location_reference_scene_id": "S01",
                },
            ],
        }
        artifact = "IMAGE_PROMPT_CONTRACT\n```json\n" + json.dumps(payload) + "\n```"
        merged = merge_image_prompt_contract(base, artifact)
        references, locations = parse_reference_plan(merged, artifact)

        self.assertEqual(merged[1].reference_ids, ("REF-HERO", "REF-FRIEND"))
        self.assertEqual(merged[1].location_reference_scene_id, "S01")
        self.assertEqual(len(references), 2)
        self.assertEqual(locations[0].canonical_scene_id, "S01")

    def test_character_template_keeps_identity_and_white_background(self) -> None:
        prompt = build_reference_prompt(
            reference_id="REF-HERO",
            kind="character",
            name="Hero",
            source_prompt="blue-haired woman in a red jacket",
        )
        self.assertIn("pure white", prompt)
        self.assertIn("front, left profile, right profile, back", prompt)
        self.assertIn("Do not redesign", prompt)
        self.assertIn("blue-haired woman", prompt)

    def test_v3_contract_preserves_linked_visual_states(self) -> None:
        base = [scene(1), scene(2)]
        payload = {
            "schema_version": 3,
            "references": [
                {
                    "reference_id": "REF-CAR-OLD",
                    "kind": "object",
                    "name": "Old car",
                    "prompt": "The same blue coupe before restoration",
                    "scene_ids": ["S01"],
                    "identity_group": "ENTITY-CAR-01",
                    "state_label": "old and damaged",
                    "base_reference_id": "",
                },
                {
                    "reference_id": "REF-CAR-RESTORED",
                    "kind": "object",
                    "name": "Restored car",
                    "prompt": "The same blue coupe after restoration",
                    "scene_ids": ["S02"],
                    "identity_group": "ENTITY-CAR-01",
                    "state_label": "fully restored",
                    "base_reference_id": "REF-CAR-OLD",
                },
            ],
            "locations": [],
            "scenes": [
                {
                    "scene_id": "S01",
                    "image_prompt": "Old blue coupe in a workshop",
                    "reference_ids": ["REF-CAR-OLD"],
                    "location_id": "",
                    "location_reference_scene_id": "",
                },
                {
                    "scene_id": "S02",
                    "image_prompt": "Restored blue coupe in the same workshop",
                    "reference_ids": ["REF-CAR-RESTORED"],
                    "location_id": "",
                    "location_reference_scene_id": "",
                },
            ],
        }
        artifact = "IMAGE_PROMPT_CONTRACT\n```json\n" + json.dumps(payload) + "\n```"
        merged = merge_image_prompt_contract(base, artifact)
        references, _ = parse_reference_plan(merged, artifact)

        self.assertEqual(references[0].identity_group, "ENTITY-CAR-01")
        self.assertEqual(references[1].state_label, "fully restored")
        self.assertEqual(references[1].base_reference_id, "REF-CAR-OLD")

    def test_linked_state_rejects_missing_base_reference(self) -> None:
        base = [scene(1)]
        payload = {
            "schema_version": 3,
            "references": [{
                "reference_id": "REF-CAR-RESTORED",
                "kind": "object",
                "name": "Restored car",
                "prompt": "Restored blue coupe",
                "scene_ids": ["S01"],
                "identity_group": "ENTITY-CAR-01",
                "state_label": "restored",
                "base_reference_id": "REF-CAR-OLD",
            }],
            "locations": [],
            "scenes": [{
                "scene_id": "S01",
                "image_prompt": "Restored blue coupe",
                "reference_ids": ["REF-CAR-RESTORED"],
                "location_id": "",
                "location_reference_scene_id": "",
            }],
        }
        artifact = "IMAGE_PROMPT_CONTRACT\n```json\n" + json.dumps(payload) + "\n```"
        merged = merge_image_prompt_contract(base, artifact)

        with self.assertRaisesRegex(ValueError, "REF-CAR-OLD"):
            parse_reference_plan(merged, artifact)

    def test_transient_object_state_is_an_action_not_a_second_reference(self) -> None:
        base = [scene(1), scene(2)]
        payload = {
            "schema_version": 3,
            "references": [
                {
                    "reference_id": "REF-DOME",
                    "kind": "object",
                    "name": "Dome",
                    "prompt": "Canonical transparent dome",
                    "scene_ids": ["S01"],
                    "identity_group": "ENTITY-DOME-01",
                    "state_label": "canonical",
                    "base_reference_id": "",
                },
                {
                    "reference_id": "REF-DOME-OPEN",
                    "kind": "object",
                    "name": "Open dome",
                    "prompt": "The same dome after opening",
                    "scene_ids": ["S02"],
                    "identity_group": "ENTITY-DOME-01",
                    "state_label": "open",
                    "base_reference_id": "REF-DOME",
                },
            ],
            "locations": [],
            "scenes": [
                {
                    "scene_id": "S01",
                    "image_prompt": "A canonical dome before it opens",
                    "reference_ids": ["REF-DOME"],
                    "location_id": "",
                    "location_reference_scene_id": "",
                },
                {
                    "scene_id": "S02",
                    "image_prompt": "The same dome opening as an action",
                    "reference_ids": ["REF-DOME-OPEN"],
                    "location_id": "",
                    "location_reference_scene_id": "",
                },
            ],
        }
        artifact = (
            "IMAGE_PROMPT_CONTRACT\n```json\n"
            + json.dumps(payload)
            + "\n```"
        )
        merged = merge_image_prompt_contract(base, artifact)

        with self.assertRaisesRegex(ValueError, "действием в сцене"):
            parse_reference_plan(merged, artifact)


class ReferenceGenerationFlowTest(unittest.TestCase):
    def _create(self, root: Path):
        content = ContentFactoryStore(root)
        run = content.create_run("Reference flow")
        store = ProductionStore(root)
        scenes = [
            scene(1, reference_ids=("REF-HERO",), location_id="LOC-HOME"),
            scene(
                2,
                reference_ids=("REF-HERO", "REF-FRIEND"),
                location_id="LOC-HOME",
            ),
        ]
        store.save_scene_contract(run.run_id, scenes)
        store.save_reference_plan(
            run.run_id,
            [
                ReferenceSpec("REF-HERO", "character", "Hero", "hero design", ("S01", "S02")),
                ReferenceSpec("REF-FRIEND", "character", "Friend", "friend design", ("S02",)),
            ],
            [LocationSpec("LOC-HOME", "Home", "same room", ("S01", "S02"), "S01")],
        )
        return run, store

    def test_cards_are_generated_first_and_every_scene_gets_real_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run, store = self._create(Path(temp_dir))
            client = ReferenceAwareImageClient()
            refs = ReferenceBatchGenerator(store, client).generate(run.run_id)
            self.assertEqual(refs.ready_count, 2)

            frames = FrameBatchGenerator(store, client).generate(run.run_id)
            self.assertEqual(frames.ready_count, 2)
            first_references = client.calls[2][1]
            second_references = client.calls[3][1]
            self.assertEqual(
                {item.reference_id for item in first_references},
                {"REF-HERO"},
            )
            self.assertEqual(
                {item.reference_id for item in second_references},
                {"REF-HERO", "REF-FRIEND", "FRAME-S01"},
            )
            self.assertTrue(all(item.path.is_file() for item in second_references))

            state = store.load(run.run_id)
            used = state["frames"]["S02"]["attempts"][-1]["reference_inputs"]
            self.assertEqual(len(used), 3)

    def test_restart_turns_stale_image_attempts_into_retryable_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run, store = self._create(Path(temp_dir))
            reference_attempt = store.start_reference(run.run_id, "REF-HERO")
            frame_attempt = store.start_frame(run.run_id, "S01")

            recovered = store.recover_interrupted_images(run.run_id)
            state = store.load(run.run_id)

            self.assertEqual(recovered, 2)
            self.assertEqual(state["references"]["REF-HERO"]["status"], "failed")
            self.assertEqual(state["frames"]["S01"]["status"], "failed")
            self.assertEqual(
                state["references"]["REF-HERO"]["attempts"][reference_attempt - 1]["status"],
                "failed",
            )
            self.assertEqual(
                state["frames"]["S01"]["attempts"][frame_attempt - 1]["status"],
                "failed",
            )

    def test_dependent_state_uses_generated_base_image_even_if_saved_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            content = ContentFactoryStore(root)
            run = content.create_run("Linked car states")
            store = ProductionStore(root)
            scenes = [
                scene(1, reference_ids=("REF-CAR-OLD",)),
                scene(2, reference_ids=("REF-CAR-RESTORED",)),
            ]
            store.save_scene_contract(run.run_id, scenes)
            store.save_reference_plan(
                run.run_id,
                [
                    ReferenceSpec(
                        "REF-CAR-RESTORED", "object", "Restored car",
                        "same blue coupe after restoration", ("S02",),
                        "ENTITY-CAR-01", "fully restored", "REF-CAR-OLD",
                    ),
                    ReferenceSpec(
                        "REF-CAR-OLD", "object", "Old car",
                        "same blue coupe before restoration", ("S01",),
                        "ENTITY-CAR-01", "old and damaged", "",
                    ),
                ],
                [],
            )
            client = ReferenceAwareImageClient()

            result = ReferenceBatchGenerator(store, client).generate(run.run_id)

            self.assertEqual(result.ready_count, 2)
            self.assertIn("REF-CAR-OLD", client.calls[0][0])
            self.assertEqual(client.calls[0][1], ())
            self.assertIn("REF-CAR-RESTORED", client.calls[1][0])
            self.assertEqual(
                [(item.reference_id, item.role) for item in client.calls[1][1]],
                [("REF-CAR-OLD", "identity-base")],
            )
            state = store.load(run.run_id)
            attempt = state["references"]["REF-CAR-RESTORED"]["attempts"][-1]
            self.assertEqual(
                attempt["reference_inputs"][0]["reference_id"],
                "REF-CAR-OLD",
            )


if __name__ == "__main__":
    unittest.main()
