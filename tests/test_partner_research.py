from __future__ import annotations

import tempfile
import unittest
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from agent_platform.partner_research import (
    BrightDataInstagramClient,
    JsonHttpTransport,
    ResearchAccount,
    ResearchError,
    ResearchStore,
    YouTubeResearchClient,
    build_production_idea_package,
    canonical_source_key,
    idea_similarity,
    parse_account_import,
    rank_trending_items,
)
from agent_platform.partner_assistant import (
    build_partner_factory_script_messages,
    parse_partner_trend_selection,
    validate_partner_factory_script,
    validate_partner_trend_fidelity,
)


class FakeYouTubeTransport:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def request(self, method, url, **kwargs):
        self.urls.append(url)
        resource = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
        if resource == "channels":
            return {
                "items": [
                    {
                        "contentDetails": {"relatedPlaylists": {"uploads": "UU-test"}},
                        "snippet": {"title": "Creator"},
                    }
                ]
            }
        if resource == "playlistItems":
            return {
                "items": [
                    {"contentDetails": {"videoId": "video-1"}},
                    {"contentDetails": {"videoId": "video-2"}},
                ]
            }
        if resource == "search":
            return {"items": [{"id": {"videoId": "video-1"}}]}
        if resource == "videos":
            return {
                "items": [
                    {
                        "id": "video-1",
                        "snippet": {
                            "title": "AI automation case",
                            "description": "A useful automation breakdown",
                            "channelTitle": "Creator",
                            "publishedAt": "2026-07-20T10:00:00Z",
                            "thumbnails": {"high": {"url": "https://img/one.jpg"}},
                        },
                        "statistics": {"viewCount": "12000", "likeCount": "500", "commentCount": "20"},
                        "contentDetails": {"duration": "PT1M5S"},
                    },
                    {
                        "id": "video-2",
                        "snippet": {
                            "title": "Unrelated vlog",
                            "description": "Nothing relevant",
                            "channelTitle": "Creator",
                            "publishedAt": "2026-07-19T10:00:00Z",
                            "thumbnails": {},
                        },
                        "statistics": {"viewCount": "900", "likeCount": "10"},
                        "contentDetails": {"duration": "PT45S"},
                    },
                ]
            }
        raise AssertionError(resource)


class FakeBrightDataTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if "/datasets/v3/snapshots?" in url:
            return []
        if "/datasets/v3/trigger?" in url:
            return {"snapshot_id": "snapshot-123"}
        if url.endswith("/datasets/v3/progress/snapshot-123"):
            return {"status": "ready"}
        if url.endswith("/datasets/v3/snapshot/snapshot-123/cancel"):
            return "OK"
        if url.endswith("/datasets/v3/snapshot/snapshot-123?format=json"):
            return [
                {
                    "post_id": "ig-1",
                    "url": "https://www.instagram.com/reel/ABC/",
                    "description": "Automation hook",
                    "user_posted": "creator",
                    "video_play_count": 25000,
                    "likes": 1200,
                    "num_comments": 40,
                    "length": "28.4",
                    "date_posted": "2026-07-30T10:00:00.000Z",
                    "thumbnail": "https://img.example/reel.jpg",
                }
            ]
        raise AssertionError(url)


class PartnerResearchTest(unittest.TestCase):
    def test_account_import_supports_text_and_csv_and_rejects_reel_urls(self) -> None:
        parsed = parse_account_import(
            "youtube:@alpha\n"
            "https://www.youtube.com/channel/UC1234567890\n"
            "instagram,beta\n"
            "https://www.instagram.com/gamma/\n"
            "https://www.instagram.com/reel/ABC/\n"
        )

        self.assertEqual(
            {(item.platform, item.handle) for item in parsed},
            {
                ("youtube", "@alpha"),
                ("youtube", "UC1234567890"),
                ("instagram", "beta"),
                ("instagram", "gamma"),
            },
        )

    def test_store_persists_accounts_runs_results_links_and_remote_task_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "research"
            store = ResearchStore(root)
            first = store.import_accounts("youtube:@alpha\ninstagram:beta")
            second = store.import_accounts("youtube:@alpha")
            run = store.create_run(
                "instagram",
                "automation",
                777,
                workflow="auto_content",
            )
            store.update_run(run.run_id, "running", provider_task_id="task-123")

            restarted = ResearchStore(root)
            resumable = restarted.recover_interrupted()

            self.assertEqual(first.imported, 2)
            self.assertEqual(second.existing, 1)
            self.assertEqual(len(restarted.list_accounts()), 2)
            self.assertEqual(resumable[0].provider_task_id, "task-123")
            self.assertEqual(resumable[0].workflow, "auto_content")

            results = restarted.save_results(
                run.run_id,
                [
                    {
                        "platform": "instagram",
                        "external_id": "ig-1",
                        "source_url": "https://www.instagram.com/reel/ABC/",
                        "title": "Hook",
                        "creator": "alpha",
                        "views": 1000,
                    }
                ],
            )
            script = root / "script.md"
            script.write_text("script", encoding="utf-8")
            restarted.link_script(results[0].result_id, script)
            restarted.link_shared_item(results[0].result_id, "CR-20260721-001")
            restarted.save_idea_package(
                results[0].result_id,
                {"kind": "production_idea", "idea": "Automation case"},
            )

            saved = restarted.require_result(results[0].result_id)
            self.assertEqual(saved.script_path, str(script))
            self.assertEqual(saved.shared_item_id, "CR-20260721-001")
            self.assertEqual(saved.idea_package["idea"], "Automation case")
            self.assertEqual(restarted.require_run(run.run_id).status, "completed")

    def test_failed_auto_content_with_saved_results_survives_reopen_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "research"
            store = ResearchStore(root)
            run = store.create_run(
                "instagram",
                "",
                777,
                workflow="auto_content",
            )
            store.update_run(
                run.run_id,
                "running",
                provider_task_id="snapshot-123",
            )
            saved = store.save_results(
                run.run_id,
                [
                    {
                        "platform": "instagram",
                        "external_id": "saved-reel",
                        "source_url": "https://instagram.example/saved-reel",
                        "title": "Saved material",
                        "creator": "saved-account",
                        "views": 12_000,
                    }
                ],
                mark_completed=False,
            )
            store.update_run(run.run_id, "failed", error="script generation failed")

            reopened = ResearchStore(root)
            reopened_run = reopened.require_run(run.run_id)
            reopened_results = reopened.list_results(run.run_id)

            self.assertEqual(reopened_run.workflow, "auto_content")
            self.assertEqual(reopened_run.status, "failed")
            self.assertEqual(reopened_run.provider_task_id, "snapshot-123")
            self.assertEqual(
                [result.result_id for result in reopened_results],
                [saved[0].result_id],
            )

    def test_trend_ranking_prefers_fresh_velocity_over_old_raw_total(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(Path(temp_dir))
            run = store.create_run("youtube", "", 777)
            store.update_run(run.run_id, "running")
            items = store.save_results(
                run.run_id,
                [
                    {
                        "platform": "youtube",
                        "external_id": "fresh",
                        "source_url": "https://youtube.example/fresh",
                        "title": "Fresh",
                        "published_at": "2026-07-26T12:00:00Z",
                        "views": 80_000,
                        "likes": 5_000,
                        "comments": 300,
                    },
                    {
                        "platform": "youtube",
                        "external_id": "old",
                        "source_url": "https://youtube.example/old",
                        "title": "Old",
                        "published_at": "2025-01-01T12:00:00Z",
                        "views": 120_000,
                        "likes": 2_000,
                        "comments": 50,
                    },
                ],
            )

            ranked = rank_trending_items(
                items,
                now=datetime(2026, 7, 27, 12, tzinfo=timezone.utc),
            )

            self.assertEqual(ranked[0].item.external_id, "fresh")

    def test_production_idea_package_uses_only_real_ranked_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(Path(temp_dir))
            run = store.create_run("youtube", "", 777, workflow="auto_content")
            store.update_run(run.run_id, "running")
            items = store.save_results(
                run.run_id,
                [
                    {
                        "platform": "youtube",
                        "external_id": "one",
                        "source_url": "https://youtube.example/one",
                        "title": "One",
                        "creator": "alpha",
                        "published_at": "2026-07-26T12:00:00Z",
                        "views": 80_000,
                        "likes": 5_000,
                        "comments": 300,
                    },
                    {
                        "platform": "youtube",
                        "external_id": "two",
                        "source_url": "https://youtube.example/two",
                        "title": "Two",
                        "creator": "beta",
                        "published_at": "2026-07-24T12:00:00Z",
                        "views": 40_000,
                        "likes": 2_000,
                        "comments": 100,
                    },
                ],
            )
            ranked = rank_trending_items(
                items,
                now=datetime(2026, 7, 27, 12, tzinfo=timezone.utc),
            )
            package = build_production_idea_package(
                run_id=run.run_id,
                primary_result_id=ranked[0].item.result_id,
                evidence_result_ids=[candidate.item.result_id for candidate in ranked],
                idea="Новая самостоятельная идея",
                reason="Два свежих ролика подтверждают интерес",
                content_format="ai",
                script="Сценарий",
                candidates=ranked,
                source_premise="Человек показывает практический процесс",
                adaptation_changes=("убрать конкретный бренд",),
            )

            self.assertEqual(package["analytics"]["evidence_count"], 2)
            self.assertEqual(package["analytics"]["total_views"], 120_000)
            self.assertEqual(package["analytics"]["total_likes"], 7_000)
            self.assertEqual(package["analytics"]["recent_evidence_count_14d"], 2)
            self.assertEqual(package["schema_version"], 2)
            self.assertEqual(
                package["source_premise"],
                "Человек показывает практический процесс",
            )
            self.assertEqual(
                package["adaptation_changes"],
                ["убрать конкретный бренд"],
            )
            self.assertEqual(
                package["production_target"],
                "ai_video_content_factory",
            )
            self.assertIs(package["requires_live_shoot"], False)
            self.assertEqual(package["format"], "ai")
            self.assertEqual(
                {item["url"] for item in package["evidence"]},
                {"https://youtube.example/one", "https://youtube.example/two"},
            )

    def test_production_idea_package_rejects_live_or_hybrid_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(Path(temp_dir))
            run = store.create_run("youtube", "", 777, workflow="auto_content")
            item = store.save_results(
                run.run_id,
                [
                    {
                        "platform": "youtube",
                        "external_id": "one",
                        "source_url": "https://youtube.example/one",
                        "title": "One",
                    }
                ],
            )[0]
            ranked = rank_trending_items([item])

            with self.assertRaises(ValueError):
                build_production_idea_package(
                    run_id=run.run_id,
                    primary_result_id=item.result_id,
                    evidence_result_ids=[item.result_id],
                    idea="Личный ролик партнёра",
                    reason="Не относится к AI-only производству",
                    content_format="hybrid",
                    script="Сценарий",
                    candidates=ranked,
                )

    def test_instagram_reel_is_consumed_once_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(Path(temp_dir))
            first_run = store.create_run(
                "instagram", "", 777, workflow="auto_content"
            )
            first_item = store.save_results(
                first_run.run_id,
                [
                    {
                        "platform": "instagram",
                        "external_id": "ABC",
                        "source_url": "https://www.instagram.com/reel/ABC/",
                        "title": "First",
                        "duration_seconds": 30,
                        "views": 100_000,
                    }
                ],
            )[0]
            first_ranked = rank_trending_items([first_item])
            first_package = build_production_idea_package(
                run_id=first_run.run_id,
                primary_result_id=first_item.result_id,
                evidence_result_ids=[first_item.result_id],
                idea="Автоматизация обработки заявок на реальном примере",
                reason="Свежий популярный ролик",
                content_format="ai",
                script="Сценарий",
                candidates=first_ranked,
            )
            store.save_production_idea(first_item.result_id, first_package)

            second_run = store.create_run(
                "instagram", "", 777, workflow="auto_content"
            )
            second_item = store.save_results(
                second_run.run_id,
                [
                    {
                        "platform": "instagram",
                        "external_id": "provider-changed-id",
                        "source_url": "https://instagram.com/reel/ABC/?utm_source=test",
                        "title": "Same reel",
                        "duration_seconds": 30,
                        "views": 120_000,
                    }
                ],
            )[0]

            self.assertEqual(
                canonical_source_key(
                    first_item.platform,
                    first_item.external_id,
                    first_item.source_url,
                ),
                canonical_source_key(
                    second_item.platform,
                    second_item.external_id,
                    second_item.source_url,
                ),
            )
            self.assertEqual(
                store.filter_available_candidates(
                    rank_trending_items([second_item])
                ),
                [],
            )

    def test_long_youtube_allows_new_idea_but_rejects_repeated_idea(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(Path(temp_dir))
            first_run = store.create_run(
                "youtube", "", 777, workflow="auto_content"
            )
            first_item = store.save_results(
                first_run.run_id,
                [
                    {
                        "platform": "youtube",
                        "external_id": "long-video",
                        "source_url": "https://www.youtube.com/watch?v=long-video",
                        "title": "Long analysis",
                        "duration_seconds": 1800,
                        "views": 200_000,
                    }
                ],
            )[0]
            first_ranked = rank_trending_items([first_item])
            store.save_production_idea(
                first_item.result_id,
                build_production_idea_package(
                    run_id=first_run.run_id,
                    primary_result_id=first_item.result_id,
                    evidence_result_ids=[first_item.result_id],
                    idea="Показать автоматизацию обработки заявок в бизнесе",
                    reason="Полезный разбор",
                    content_format="ai",
                    script="Сценарий 1",
                    candidates=first_ranked,
                ),
            )

            second_run = store.create_run(
                "youtube", "", 777, workflow="auto_content"
            )
            second_item = store.save_results(
                second_run.run_id,
                [
                    {
                        "platform": "youtube",
                        "external_id": "long-video",
                        "source_url": "https://youtu.be/long-video",
                        "title": "Long analysis",
                        "duration_seconds": 1800,
                        "views": 210_000,
                    }
                ],
            )[0]
            second_ranked = rank_trending_items([second_item])
            self.assertEqual(len(store.filter_available_candidates(second_ranked)), 1)

            duplicate_package = build_production_idea_package(
                run_id=second_run.run_id,
                primary_result_id=second_item.result_id,
                evidence_result_ids=[second_item.result_id],
                idea="Показать автоматизацию обработки заявок в бизнесе",
                reason="Другой фрагмент длинного видео",
                content_format="ai",
                script="Сценарий 2",
                candidates=second_ranked,
            )
            with self.assertRaises(ResearchError) as duplicate:
                store.save_production_idea(
                    second_item.result_id,
                    duplicate_package,
                )
            self.assertEqual(duplicate.exception.code, "duplicate_idea")

            new_package = build_production_idea_package(
                run_id=second_run.run_id,
                primary_result_id=second_item.result_id,
                evidence_result_ids=[second_item.result_id],
                idea="Разобрать контроль качества AI-ответов оператора",
                reason="Отдельный тезис длинного видео",
                content_format="ai",
                script="Сценарий 3",
                candidates=second_ranked,
            )
            saved = store.save_production_idea(
                second_item.result_id,
                new_package,
            )
            self.assertEqual(
                saved.idea_package["idea"],
                "Разобрать контроль качества AI-ответов оператора",
            )

    def test_short_youtube_video_is_consumed_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(Path(temp_dir))
            first_run = store.create_run(
                "youtube", "", 777, workflow="auto_content"
            )
            first_item = store.save_results(
                first_run.run_id,
                [
                    {
                        "platform": "youtube",
                        "external_id": "short-video",
                        "source_url": "https://youtube.com/shorts/short-video",
                        "duration_seconds": 45,
                        "views": 90_000,
                    }
                ],
            )[0]
            ranked = rank_trending_items([first_item])
            store.save_production_idea(
                first_item.result_id,
                build_production_idea_package(
                    run_id=first_run.run_id,
                    primary_result_id=first_item.result_id,
                    evidence_result_ids=[first_item.result_id],
                    idea="Короткий кейс автоматизации записи клиента",
                    reason="Высокая скорость просмотров",
                    content_format="ai",
                    script="Сценарий",
                    candidates=ranked,
                ),
            )
            second_run = store.create_run(
                "youtube", "", 777, workflow="auto_content"
            )
            second_item = store.save_results(
                second_run.run_id,
                [
                    {
                        "platform": "youtube",
                        "external_id": "short-video",
                        "source_url": "https://www.youtube.com/watch?v=short-video",
                        "duration_seconds": 45,
                        "views": 95_000,
                    }
                ],
            )[0]

            self.assertFalse(
                store.filter_available_candidates(
                    rank_trending_items([second_item])
                )
            )

    def test_idea_similarity_detects_reworded_duplicate(self) -> None:
        score = idea_similarity(
            "Show real business automation using lead processing example",
            "Explain lead-processing automation with a real business example",
        )

        self.assertGreaterEqual(score, 0.72)

    def test_trend_selection_rejects_evidence_not_supplied_by_collector(self) -> None:
        with self.assertRaises(ValueError):
            parse_partner_trend_selection(
                '{"result_id": 1, "evidence_result_ids": [1, 999], '
                '"source_premise": "Source idea", "idea": "Idea", '
                '"adaptation_changes": [], "theme_changed": false, '
                '"reason": "Reason", "format": "ai"}',
                allowed_result_ids={1, 2},
            )

    def test_trend_selection_keeps_only_primary_source(self) -> None:
        selection = parse_partner_trend_selection(
            '{"result_id": 1, "evidence_result_ids": [1, 2], '
            '"source_premise": "One reel premise", '
            '"idea": "Idea from one reel", "adaptation_changes": [], '
            '"theme_changed": false, "reason": "Reason", "format": "ai"}',
            allowed_result_ids={1, 2},
        )

        self.assertEqual(selection.result_id, 1)
        self.assertEqual(selection.evidence_result_ids, (1,))

    def test_trend_selection_rejects_personal_or_hybrid_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "только идеи"):
            parse_partner_trend_selection(
                '{"result_id": 1, "evidence_result_ids": [1], '
                '"source_premise": "партнёр рассказывает в кадре", '
                '"idea": "партнёр рассказывает в кадре", '
                '"adaptation_changes": [], "theme_changed": false, "reason": "Reason", '
                '"format": "hybrid"}',
                allowed_result_ids={1},
            )

    def test_trend_selection_rejects_declared_theme_change(self) -> None:
        with self.assertRaisesRegex(ValueError, "изменить тему"):
            parse_partner_trend_selection(
                '{"result_id": 1, "evidence_result_ids": [1], '
                '"source_premise": "Человек открывает коробку", '
                '"idea": "Коробка из будущего", "adaptation_changes": [], '
                '"theme_changed": true, "reason": "Reason", "format": "ai"}',
                allowed_result_ids={1},
            )

    def test_trend_fidelity_rejects_future_or_automation_not_in_source(self) -> None:
        selection = parse_partner_trend_selection(
            '{"result_id": 1, "evidence_result_ids": [1], '
            '"source_premise": "Обычный человек открывает коробку", '
            '"idea": "Робот из будущего автоматически открывает коробку", '
            '"adaptation_changes": ["заменить человека"], '
            '"theme_changed": false, "reason": "Reason", "format": "ai"}',
            allowed_result_ids={1},
        )

        with self.assertRaisesRegex(ValueError, "новая тема"):
            validate_partner_trend_fidelity(
                selection,
                source_title="Человек открывает посылку",
                source_description="Он распаковывает коробку и удивляется содержимому.",
            )

    def test_trend_fidelity_allows_theme_already_present_in_source(self) -> None:
        selection = parse_partner_trend_selection(
            '{"result_id": 1, "evidence_result_ids": [1], '
            '"source_premise": "Робот будущего открывает коробку", '
            '"idea": "Тот же робот будущего открывает нейтральную коробку", '
            '"adaptation_changes": ["убрать бренд коробки"], '
            '"theme_changed": false, "reason": "Reason", "format": "ai"}',
            allowed_result_ids={1},
        )

        validate_partner_trend_fidelity(
            selection,
            source_title="Future robot unboxing",
            source_description="A futuristic robot opens a branded package.",
        )

    def test_trend_fidelity_treats_ai_video_as_production_method(self) -> None:
        selection = parse_partner_trend_selection(
            '{"result_id": 1, "evidence_result_ids": [1], '
            '"source_premise": "Человек берёт чашку со стола", '
            '"idea": "AI-видео: вымышленный AI-персонаж берёт ту же чашку", '
            '"adaptation_changes": ["заменить человека AI-персонажем"], '
            '"theme_changed": false, "reason": "Reason", "format": "ai"}',
            allowed_result_ids={1},
        )

        validate_partner_trend_fidelity(
            selection,
            source_title="Человек берёт чашку",
            source_description="Обычное бытовое действие за столом.",
        )

    def test_trend_fidelity_still_rejects_ai_as_new_story_theme(self) -> None:
        selection = parse_partner_trend_selection(
            '{"result_id": 1, "evidence_result_ids": [1], '
            '"source_premise": "Человек берёт чашку со стола", '
            '"idea": "AI управляет офисом и автоматически выдаёт чашку", '
            '"adaptation_changes": [], "theme_changed": false, '
            '"reason": "Reason", "format": "ai"}',
            allowed_result_ids={1},
        )

        with self.assertRaisesRegex(ValueError, "новая тема"):
            validate_partner_trend_fidelity(
                selection,
                source_title="Человек берёт чашку",
                source_description="Обычное бытовое действие за столом.",
            )

    def test_factory_script_prompt_excludes_personal_memory_and_validator_blocks_partner(self) -> None:
        messages = build_partner_factory_script_messages("Источник: https://example.test")
        combined = "\n".join(message["content"] for message in messages)

        self.assertIn("AI-video content factory", combined)
        self.assertIn("не личный ассистент", combined)
        self.assertIn("AI — способ снять этот же сюжет", combined)
        self.assertIn("не добавляй тему", combined.lower())
        self.assertNotIn("Контекст из изолированной памяти партнёра", combined)
        invalid = (
            "## Название и цель\n\nТестовая цель для производства.\n\n"
            "## Хук\n\nВидимый результат в первые секунды.\n\n"
            "## Сценарная структура\n\n### Сцена 1 · 0-5 сек\n\n"
            + "партнёр в кадре рассказывает о процессе. " * 20
            + "\n\n## Финал\n\nОбещание хука закрыто.\n\n"
            "## Производственные ограничения\n\n"
            "**Режим производства:** только AI-видео; живая съёмка не требуется."
        )
        with self.assertRaisesRegex(ValueError, "партнёра"):
            validate_partner_factory_script(invalid)

    def test_factory_script_validator_rejects_theme_added_after_selection(self) -> None:
        script = (
            "## Название и цель\n\nОткрытие обычной посылки.\n\n"
            "## Хук\n\nЧеловек замечает странный звук внутри коробки.\n\n"
            "## Сценарная структура\n\n### Сцена 1 · 0-5 сек\n\n"
            "**Функция:** открыть коробку. **Визуал:** робот из будущего в "
            "киберпанк-лаборатории. **Действие:** робот открывает коробку. "
            "**Камера:** плавный push-in. **Озвучка:** внутри что-то шумит. "
            "**Переход:** крышка поднимается. "
            + "Неоновая лаборатория остаётся неизменной. " * 8
            + "\n\n## Финал\n\nВ коробке оказывается подарок.\n\n"
            "## Производственные ограничения\n\n"
            "**Режим производства:** только AI-видео; живая съёмка не требуется."
        )

        with self.assertRaisesRegex(ValueError, "новую тему"):
            validate_partner_factory_script(
                script,
                source_text=(
                    "Человек открывает обычную посылку. "
                    "Он слышит звук и находит подарок внутри."
                ),
            )

    def test_factory_script_allows_required_ai_production_section(self) -> None:
        script = (
            "## Название и цель\n\nОбычное открытие посылки.\n\n"
            "## Хук\n\nЧеловек слышит звук внутри коробки.\n\n"
            "## Сценарная структура\n\n### Сцена 1 · 0-5 сек\n\n"
            "**Функция:** открыть коробку. **Визуал:** нейтральный AI-персонаж "
            "у обычного стола. **Действие:** персонаж открывает коробку. "
            "**Камера:** плавный push-in. **Переход:** крышка поднимается. "
            + "Комната и коробка остаются неизменными. " * 9
            + "\n\n## Финал\n\nВнутри находится подарок.\n\n"
            "## Производственные ограничения\n\n"
            "**Режим производства:** только AI-видео; живая съёмка не требуется."
        )

        validated = validate_partner_factory_script(
            script,
            source_text=(
                "Человек открывает обычную посылку. "
                "Он слышит звук и находит подарок внутри."
            ),
        )

        self.assertEqual(validated, script)

    def test_auto_content_collection_can_remain_running_until_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(Path(temp_dir))
            run = store.create_run("youtube", "", 777, workflow="auto_content")
            store.update_run(run.run_id, "running")

            store.save_results(
                run.run_id,
                [{"platform": "youtube", "source_url": "https://youtu.be/a"}],
                mark_completed=False,
            )

            self.assertEqual(store.require_run(run.run_id).status, "running")

    def test_legacy_completed_run_without_script_is_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(Path(temp_dir))
            run = store.create_run("instagram", "", 777, workflow="auto_content")
            store.update_run(run.run_id, "running")
            store.save_results(
                run.run_id,
                [{"platform": "instagram", "source_url": "https://instagram.com/reel/a"}],
            )

            recovered = store.recover_incomplete_auto_content()

            self.assertEqual([item.run_id for item in recovered], [run.run_id])
            failed = store.require_run(run.run_id)
            self.assertEqual(failed.status, "failed")
            self.assertIn("повторить только этап", failed.error)

    def test_interrupted_local_run_is_failed_instead_of_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(Path(temp_dir))
            run = store.create_run("youtube", "ai", 777)
            store.update_run(run.run_id, "running")

            resumable = ResearchStore(Path(temp_dir)).recover_interrupted()

            self.assertEqual(resumable, [])
            recovered = store.require_run(run.run_id)
            self.assertEqual(recovered.status, "failed")
            self.assertIn("перезапуском", recovered.error)

    def test_youtube_collect_uses_channel_uploads_and_normalizes_metrics(self) -> None:
        transport = FakeYouTubeTransport()
        client = YouTubeResearchClient("secret-youtube-key", transport=transport)
        account = ResearchAccount(
            1,
            "youtube",
            "@alpha",
            "https://www.youtube.com/@alpha",
            True,
            "now",
            "now",
        )

        results = client.collect([account], "automation", limit=10)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["external_id"], "video-1")
        self.assertEqual(results[0]["duration_seconds"], 65)
        self.assertEqual(results[0]["views"], 12000)
        self.assertTrue(any("channels" in url for url in transport.urls))
        self.assertTrue(any("playlistItems" in url for url in transport.urls))
        self.assertTrue(any("videos" in url for url in transport.urls))

    def test_youtube_global_search_limits_results_to_configured_period(self) -> None:
        transport = FakeYouTubeTransport()
        client = YouTubeResearchClient(
            "secret-youtube-key",
            transport=transport,
            days=14,
        )

        results = client.collect([], "automation", limit=10)

        self.assertTrue(results)
        search_url = next(url for url in transport.urls if "/search?" in url)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(search_url).query)
        self.assertIn("publishedAfter", query)
        self.assertRegex(query["publishedAfter"][0], r"^\d{4}-\d{2}-\d{2}T")

    def test_brightdata_start_and_resume_use_one_snapshot_and_normalize_reel(self) -> None:
        transport = FakeBrightDataTransport()
        client = BrightDataInstagramClient(
            "secret-brightdata-token",
            transport=transport,
            poll_interval_seconds=1,
        )
        account = ResearchAccount(
            1,
            "instagram",
            "creator",
            "https://www.instagram.com/creator/",
            True,
            "now",
            "now",
        )

        task_id = client.start([account], "", 20)
        results = client.wait_for_results(task_id, timeout_seconds=30)

        self.assertEqual(task_id, "snapshot-123")
        self.assertEqual(results[0]["platform"], "instagram")
        self.assertEqual(results[0]["views"], 25000)
        self.assertEqual(results[0]["duration_seconds"], 28)
        trigger_calls = [call for call in transport.calls if "/trigger?" in call[1]]
        self.assertEqual(len(trigger_calls), 1)
        parsed_query = urllib.parse.parse_qs(
            urllib.parse.urlparse(trigger_calls[0][1]).query
        )
        self.assertEqual(parsed_query["dataset_id"], ["gd_lyclm20il4r5helnj"])
        self.assertEqual(parsed_query["type"], ["discover_new"])
        self.assertEqual(parsed_query["discover_by"], ["url"])
        self.assertEqual(
            trigger_calls[0][2]["headers"]["Authorization"],
            "Bearer secret-brightdata-token",
        )
        self.assertEqual(
            trigger_calls[0][2]["payload"][0]["url"],
            "https://www.instagram.com/creator/",
        )
        self.assertEqual(trigger_calls[0][2]["payload"][0]["num_of_posts"], 5)
        self.assertNotIn("secret-brightdata-token", trigger_calls[0][1])

    def test_brightdata_connection_check_and_cancel_do_not_start_collection(self) -> None:
        transport = FakeBrightDataTransport()
        client = BrightDataInstagramClient(
            "secret-brightdata-token",
            transport=transport,
        )

        client.check_connection()
        client.abort("snapshot-123")

        self.assertFalse(any("/trigger?" in call[1] for call in transport.calls))
        self.assertTrue(any("/snapshots?" in call[1] for call in transport.calls))
        self.assertTrue(any(call[1].endswith("/cancel") for call in transport.calls))

    def test_brightdata_requires_saved_instagram_accounts(self) -> None:
        client = BrightDataInstagramClient(
            "secret-brightdata-token",
            transport=FakeBrightDataTransport(),
        )

        with self.assertRaises(ResearchError) as caught:
            client.start([], "automation", 20)

        self.assertEqual(caught.exception.code, "instagram_accounts_required")

    def test_brightdata_distributes_total_limit_without_hardcoded_count(self) -> None:
        transport = FakeBrightDataTransport()
        client = BrightDataInstagramClient(
            "secret-brightdata-token",
            transport=transport,
            results_per_account=5,
        )
        accounts = [
            ResearchAccount(
                index,
                "instagram",
                f"creator{index}",
                f"https://www.instagram.com/creator{index}/",
                True,
                "now",
                "now",
            )
            for index in range(1, 5)
        ]

        client.start(accounts, "", 7)

        trigger = next(call for call in transport.calls if "/trigger?" in call[1])
        quotas = [item["num_of_posts"] for item in trigger[2]["payload"]]
        self.assertEqual(sum(quotas), 7)
        self.assertEqual(quotas, [2, 2, 2, 1])

    def test_brightdata_cancelled_wait_aborts_snapshot(self) -> None:
        transport = FakeBrightDataTransport()
        client = BrightDataInstagramClient(
            "secret-brightdata-token",
            transport=transport,
        )

        with self.assertRaises(ResearchError) as caught:
            client.wait_for_results(
                "snapshot-123",
                cancelled=lambda: True,
                timeout_seconds=30,
            )

        self.assertEqual(caught.exception.code, "cancelled")
        self.assertTrue(any(call[1].endswith("/cancel") for call in transport.calls))

    def test_brightdata_missing_key_is_reported_without_secret_data(self) -> None:
        client = BrightDataInstagramClient("", transport=FakeBrightDataTransport())

        with self.assertRaises(ResearchError) as caught:
            client.check_connection()

        self.assertEqual(caught.exception.code, "instagram_not_configured")
        self.assertIn("BRIGHTDATA_API_TOKEN", str(caught.exception))

    def test_external_http_errors_never_expose_brightdata_token(self) -> None:
        transport = JsonHttpTransport()
        for status in (401, 403, 429, 500):
            with self.subTest(status=status):
                error = urllib.error.HTTPError(
                    "https://api.brightdata.com/datasets/v3/snapshots",
                    status,
                    "provider error",
                    {},
                    None,
                )
                with patch(
                    "agent_platform.partner_research.urllib.request.urlopen",
                    side_effect=error,
                ):
                    with self.assertRaises(ResearchError) as caught:
                        transport.request(
                            "GET",
                            "https://api.brightdata.com/datasets/v3/snapshots",
                            headers={"Authorization": "Bearer secret-brightdata-token"},
                        )

                self.assertEqual(caught.exception.code, f"http_{status}")
                self.assertNotIn("secret-brightdata-token", str(caught.exception))

    def test_image_state_survives_restart_without_restarting_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = ResearchStore(root)
            asset = store.create_image("A clean portrait", 777)
            store.update_image(asset.asset_id, "generating")

            recovered = ResearchStore(root)
            count = recovered.recover_images()

            self.assertEqual(count, 1)
            self.assertEqual(recovered.require_image(asset.asset_id).status, "failed")


if __name__ == "__main__":
    unittest.main()
