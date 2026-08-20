import re
import tempfile
import unittest
from pathlib import Path

from agent_platform.shared_content import (
    SharedContentStore,
    SharedPermissionError,
    SharedTransitionError,
    split_manual_sources,
)


class SharedContentStoreTest(unittest.TestCase):
    def test_manual_source_split_preserves_notes_and_splits_url_batches(self) -> None:
        note = "Идея ролика\nСделать упор на практическую пользу"
        urls = "https://example.com/one\nhttps://example.com/two"

        self.assertEqual(split_manual_sources(note), [note])
        self.assertEqual(
            split_manual_sources(urls),
            ["https://example.com/one", "https://example.com/two"],
        )

    def test_ids_and_items_survive_new_store_instance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "shared"
            first_store = SharedContentStore(root)
            first = first_store.create_item("partner", "Первый референс")
            second = first_store.create_item("partner", "Второй референс")

            restarted = SharedContentStore(root)

            self.assertRegex(first.item_id, r"^CR-\d{8}-001$")
            self.assertRegex(second.item_id, r"^CR-\d{8}-002$")
            self.assertEqual(restarted.require(first.item_id).source_text, "Первый референс")
            self.assertEqual(len(restarted.list_items(limit=10)), 2)

    def test_structured_idea_metadata_survives_and_can_be_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SharedContentStore(Path(temp_dir))
            material = store.create_item("partner", "Обычный файл")
            idea = store.create_item(
                "partner",
                "Производственный пакет",
                item_kind="production_idea",
                metadata={
                    "kind": "production_idea",
                    "idea": "Показать автоматизацию через один понятный кейс",
                    "analytics": {"total_views": 120000},
                },
            )

            restarted = SharedContentStore(Path(temp_dir))
            saved = restarted.require(idea.item_id)

            self.assertEqual(saved.item_kind, "production_idea")
            self.assertEqual(saved.metadata["analytics"]["total_views"], 120000)
            self.assertEqual(
                [item.item_id for item in restarted.list_items(item_kinds={"material"})],
                [material.item_id],
            )
            self.assertEqual(
                [
                    item.item_id
                    for item in restarted.list_items(item_kinds={"production_idea"})
                ],
                [idea.item_id],
            )

    def test_role_gated_lifecycle_keeps_immutable_event_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SharedContentStore(Path(temp_dir))
            item = store.create_item("partner", "https://example.com/reel")
            handed_off = store.handoff("partner", item.item_id)
            accepted = store.accept("owner", handed_off.item_id)
            linked = store.link_run("owner", accepted.item_id, "CF-20260721-001")
            ready = store.mark_ready(linked.item_id, linked.linked_run_id)

            self.assertEqual(ready.status, "ready")
            self.assertEqual(ready.linked_run_id, "CF-20260721-001")
            actions = [event["action"] for event in store.events(item.item_id)]
            self.assertEqual(
                actions,
                [
                    "created",
                    "handoff_requested",
                    "accepted",
                    "linked_to_content_factory",
                    "production_ready",
                ],
            )

    def test_roles_and_invalid_transitions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SharedContentStore(Path(temp_dir))
            item = store.create_item("partner", "Материал")

            with self.assertRaises(SharedPermissionError):
                store.handoff("owner", item.item_id)
            with self.assertRaises(SharedPermissionError):
                store.accept("partner", item.item_id)
            with self.assertRaises(SharedTransitionError):
                store.link_run("owner", item.item_id, "CF-20260721-001")

    def test_accepted_item_can_be_returned_and_handed_off_again(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SharedContentStore(Path(temp_dir))
            item = store.create_item("partner", "Черновик")
            store.handoff("partner", item.item_id)
            store.accept("owner", item.item_id)

            returned = store.return_to_partner(
                "owner", item.item_id, "Нужна ссылка на оригинал."
            )
            handed_off = store.handoff("partner", returned.item_id, "Ссылка добавлена.")

            self.assertEqual(returned.status, "returned")
            self.assertIn("Нужна ссылка", returned.notes)
            self.assertEqual(handed_off.status, "handoff_requested")
            self.assertIn("Ссылка добавлена", handed_off.notes)

    def test_media_is_copied_and_cannot_be_attached_by_other_role(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SharedContentStore(root / "shared")
            source = root / "unsafe source name.jpg"
            source.write_bytes(b"image")
            item = store.create_item("partner", "Фото", source_type="photo")

            stored = store.store_media(
                "partner", item.item_id, source, original_name="unsafe source name.jpg"
            )

            self.assertTrue((store.root / stored.media_path).is_file())
            self.assertFalse(re.search(r"\s", Path(stored.media_path).name))
            with self.assertRaises(SharedPermissionError):
                store.store_media("owner", item.item_id, source)


if __name__ == "__main__":
    unittest.main()
