from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


RESEARCH_PROVIDERS = {"youtube", "instagram"}
RUN_STATUSES = {
    "pending_confirmation",
    "running",
    "completed",
    "cancelled",
    "failed",
}
RUN_WORKFLOWS = {"reference_search", "auto_content"}
IMAGE_STATUSES = {"pending_confirmation", "generating", "completed", "cancelled", "failed"}
LONG_FORM_YOUTUBE_SECONDS = 600
IDEA_SIMILARITY_THRESHOLD = 0.72

_IDEA_STOP_WORDS = {
    "ai",
    "the",
    "and",
    "for",
    "how",
    "idea",
    "reel",
    "video",
    "без",
    "видео",
    "для",
    "идея",
    "как",
    "на",
    "показать",
    "про",
    "ролик",
    "с",
    "что",
    "это",
}
_IDEA_SUFFIXES = (
    "изация",
    "ировать",
    "ирование",
    "иями",
    "ями",
    "ами",
    "ого",
    "ему",
    "ыми",
    "ими",
    "ий",
    "ый",
    "ая",
    "ое",
    "ые",
    "ов",
    "ев",
    "ам",
    "ям",
    "ах",
    "ях",
    "ing",
    "ed",
    "es",
    "s",
)


class ResearchError(RuntimeError):
    """Safe user-facing research error that never embeds credentials."""

    def __init__(self, message: str, *, code: str = "research_error"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AccountCandidate:
    platform: str
    handle: str
    source_url: str


@dataclass(frozen=True)
class AccountImportResult:
    imported: int
    existing: int
    invalid: int
    candidates: tuple[AccountCandidate, ...]
    duplicates: int = 0


@dataclass(frozen=True)
class ResearchAccount:
    account_id: int
    platform: str
    handle: str
    source_url: str
    active: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ResearchRun:
    run_id: str
    provider: str
    workflow: str
    query: str
    status: str
    result_count: int
    provider_task_id: str
    error: str
    requested_by: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ResearchItem:
    result_id: int
    run_id: str
    platform: str
    external_id: str
    title: str
    source_url: str
    creator: str
    published_at: str
    duration_seconds: int
    views: int
    likes: int
    comments: int
    thumbnail_url: str
    description: str
    script_path: str
    shared_item_id: str
    idea_package_json: str
    created_at: str

    @property
    def idea_package(self) -> dict[str, Any]:
        try:
            value = json.loads(self.idea_package_json or "{}")
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class RankedResearchItem:
    item: ResearchItem
    score: float
    age_days: float
    views_per_day: float
    engagement_rate: float


@dataclass(frozen=True)
class IdeaHistoryItem:
    idea_id: int
    run_id: str
    result_id: int
    platform: str
    primary_source_key: str
    idea_key: str
    idea_text: str
    created_at: str


@dataclass(frozen=True)
class ImageAsset:
    asset_id: str
    prompt: str
    reference_path: str
    status: str
    file_path: str
    error: str
    requested_by: int
    created_at: str
    updated_at: str


def canonical_source_key(
    platform: str,
    external_id: str,
    source_url: str,
) -> str:
    """Return one stable identity for URL variants of the same source video."""

    clean_platform = platform.strip().lower()
    clean_external_id = external_id.strip()
    clean_url = source_url.strip()
    parsed = urllib.parse.urlparse(clean_url)
    path_parts = [part for part in parsed.path.split("/") if part]
    identity = ""
    if clean_platform == "youtube":
        query = urllib.parse.parse_qs(parsed.query)
        if query.get("v"):
            identity = str(query["v"][0]).strip()
        elif parsed.netloc.casefold() in {"youtu.be", "www.youtu.be"} and path_parts:
            identity = path_parts[0]
        elif path_parts and path_parts[0].casefold() in {"shorts", "embed"}:
            identity = path_parts[1] if len(path_parts) > 1 else ""
    elif clean_platform == "instagram":
        for marker in ("reel", "reels", "p", "tv"):
            try:
                marker_index = [part.casefold() for part in path_parts].index(marker)
            except ValueError:
                continue
            if marker_index + 1 < len(path_parts):
                identity = path_parts[marker_index + 1]
                break
    identity = identity or clean_external_id
    if not identity:
        normalized_url = (
            f"{parsed.netloc.casefold()}{parsed.path.rstrip('/').casefold()}"
            if parsed.netloc
            else clean_url.casefold().rstrip("/")
        )
        identity = normalized_url
    return hashlib.sha256(
        f"{clean_platform}|{identity.casefold()}".encode("utf-8")
    ).hexdigest()


def idea_fingerprint(value: str) -> str:
    normalized = " ".join(_idea_tokens(value))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def idea_similarity(first: str, second: str) -> float:
    """Estimate duplicate meaning without an external embedding dependency."""

    first_tokens = set(_idea_tokens(first))
    second_tokens = set(_idea_tokens(second))
    if not first_tokens or not second_tokens:
        return 0.0
    intersection = len(first_tokens & second_tokens)
    dice = (2.0 * intersection) / (len(first_tokens) + len(second_tokens))
    containment = intersection / min(len(first_tokens), len(second_tokens))
    first_text = " ".join(sorted(first_tokens))
    second_text = " ".join(sorted(second_tokens))
    sequence = SequenceMatcher(None, first_text, second_text).ratio()
    containment_score = containment * 0.9 if intersection >= 2 else 0.0
    return max(dice, containment_score, sequence * 0.9)


def rank_trending_items(
    items: Sequence[ResearchItem],
    *,
    limit: int = 10,
    now: datetime | None = None,
) -> list[RankedResearchItem]:
    """Rank recent videos by proven reach, current velocity, and engagement."""

    reference_time = now or datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    ranked: list[RankedResearchItem] = []
    for item in items:
        published = _parse_published_at(item.published_at)
        age_days = (
            max(0.25, (reference_time - published).total_seconds() / 86400)
            if published is not None
            else 30.0
        )
        views_per_day = item.views / age_days
        engagement_rate = (
            (item.likes + 2 * item.comments) / item.views
            if item.views > 0
            else 0.0
        )
        freshness = max(0.0, 1.0 - min(age_days, 60.0) / 60.0)
        score = (
            math.log10(item.views + 1) * 0.30
            + math.log10(views_per_day + 1) * 0.50
            + min(engagement_rate, 0.25) * 4.0
            + freshness * 0.75
        )
        ranked.append(
            RankedResearchItem(
                item=item,
                score=score,
                age_days=age_days,
                views_per_day=views_per_day,
                engagement_rate=engagement_rate,
            )
        )
    ranked.sort(
        key=lambda candidate: (
            candidate.score,
            candidate.item.views,
            candidate.item.likes,
            -candidate.item.result_id,
        ),
        reverse=True,
    )
    return ranked[: max(1, min(int(limit), 50))]


def build_production_idea_package(
    *,
    run_id: str,
    primary_result_id: int,
    evidence_result_ids: Sequence[int],
    idea: str,
    reason: str,
    content_format: str,
    script: str,
    candidates: Sequence[RankedResearchItem],
    source_premise: str = "",
    adaptation_changes: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a durable, evidence-backed handoff without model-invented metrics."""

    if content_format.strip().lower() != "ai":
        raise ValueError(
            "Производственная идея радара должна предназначаться только для AI-видео."
        )
    by_id = {candidate.item.result_id: candidate for candidate in candidates}
    ordered_ids: list[int] = []
    for result_id in [primary_result_id, *evidence_result_ids]:
        normalized = int(result_id)
        if normalized not in by_id:
            raise ValueError(f"Не найден результат-доказательство: {normalized}")
        if normalized not in ordered_ids:
            ordered_ids.append(normalized)
    ordered_ids = ordered_ids[:3]
    evidence: list[dict[str, Any]] = []
    for result_id in ordered_ids:
        candidate = by_id[result_id]
        item = candidate.item
        evidence.append(
            {
                "result_id": item.result_id,
                "platform": item.platform,
                "external_id": item.external_id,
                "creator": item.creator,
                "title": item.title,
                "url": item.source_url,
                "published_at": item.published_at,
                "duration_seconds": item.duration_seconds,
                "source_key": canonical_source_key(
                    item.platform,
                    item.external_id,
                    item.source_url,
                ),
                "age_days": round(candidate.age_days, 2),
                "views": item.views,
                "likes": item.likes,
                "comments": item.comments,
                "views_per_day": round(candidate.views_per_day, 2),
                "engagement_rate": round(candidate.engagement_rate, 6),
                "trend_score": round(candidate.score, 4),
            }
        )

    evidence_count = len(evidence)
    analytics = {
        "evidence_count": evidence_count,
        "recent_evidence_count_14d": sum(
            float(item["age_days"]) <= 14.0 for item in evidence
        ),
        "total_views": sum(int(item["views"]) for item in evidence),
        "total_likes": sum(int(item["likes"]) for item in evidence),
        "total_comments": sum(int(item["comments"]) for item in evidence),
        "combined_views_per_day": round(
            sum(float(item["views_per_day"]) for item in evidence),
            2,
        ),
        "average_engagement_rate": round(
            sum(float(item["engagement_rate"]) for item in evidence)
            / max(evidence_count, 1),
            6,
        ),
        "max_trend_score": max(
            (float(item["trend_score"]) for item in evidence),
            default=0.0,
        ),
    }
    return {
        "schema_version": 2,
        "kind": "production_idea",
        "production_target": "ai_video_content_factory",
        "requires_live_shoot": False,
        "radar_run_id": run_id.strip().upper(),
        "primary_result_id": int(primary_result_id),
        "idea_key": idea_fingerprint(idea),
        "source_premise": source_premise.strip() or idea.strip(),
        "idea": idea.strip(),
        "adaptation_changes": [
            str(change).strip()
            for change in adaptation_changes
            if str(change).strip()
        ][:3],
        "format": "ai",
        "reason": reason.strip(),
        "analytics": analytics,
        "evidence": evidence,
        "script": script.strip(),
        "created_at": _now(),
    }


def parse_account_import(text: str) -> tuple[AccountCandidate, ...]:
    """Normalize text/CSV account lists without guessing video links are creator accounts."""

    candidates, _, _ = _parse_account_import_with_stats(text)
    return candidates


def _parse_account_import_with_stats(
    text: str,
) -> tuple[tuple[AccountCandidate, ...], int, int]:
    """Return unique accounts while keeping invalid rows separate from duplicates."""

    clean = text.strip().lstrip("\ufeff")
    if not clean:
        return (), 0, 0

    rows: list[tuple[str, str]] = []
    parsed_csv = list(csv.reader(io.StringIO(clean)))
    if parsed_csv and _looks_like_header(parsed_csv[0]):
        header = [cell.strip().lower() for cell in parsed_csv[0]]
        for row in parsed_csv[1:]:
            values = {header[index]: value.strip() for index, value in enumerate(row) if index < len(header)}
            platform = values.get("platform", values.get("платформа", ""))
            account = (
                values.get("url")
                or values.get("link")
                or values.get("ссылка")
                or values.get("handle")
                or values.get("username")
                or values.get("account")
                or values.get("аккаунт")
                or ""
            )
            if account:
                rows.append((platform, account))
    else:
        for line in clean.splitlines():
            value = line.strip()
            if not value:
                continue
            if "," in value and not value.lower().startswith(("http://", "https://")):
                cells = [cell.strip() for cell in next(csv.reader([value])) if cell.strip()]
                if len(cells) >= 2 and cells[0].lower() in {"youtube", "yt", "instagram", "ig"}:
                    rows.append((cells[0], cells[1]))
                    continue
            rows.append(("", value))

    candidates: list[AccountCandidate] = []
    seen: set[tuple[str, str]] = set()
    invalid = 0
    duplicates = 0
    for platform_hint, raw_value in rows:
        candidate = _parse_account_candidate(raw_value, platform_hint)
        if candidate is None:
            invalid += 1
            continue
        key = (candidate.platform, candidate.handle.casefold())
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        candidates.append(candidate)
    return tuple(candidates), invalid, duplicates


def _looks_like_header(row: Sequence[str]) -> bool:
    fields = {cell.strip().lower() for cell in row}
    return bool(fields & {"platform", "платформа"}) and bool(
        fields & {"url", "link", "ссылка", "handle", "username", "account", "аккаунт"}
    )


def _parse_account_candidate(raw_value: str, platform_hint: str = "") -> AccountCandidate | None:
    value = raw_value.strip().strip('"\'')
    hint = platform_hint.strip().lower()
    prefix_match = re.match(r"^(youtube|yt|instagram|ig)\s*:\s*(.+)$", value, re.IGNORECASE)
    if prefix_match:
        hint = prefix_match.group(1).lower()
        value = prefix_match.group(2).strip()
    if hint == "yt":
        hint = "youtube"
    if hint == "ig":
        hint = "instagram"

    youtube = re.search(
        r"(?:https?://)?(?:www\.)?(?:youtube\.com|youtu\.be)/(?:@(?P<handle>[A-Za-z0-9._-]+)|channel/(?P<channel>UC[A-Za-z0-9_-]+))/?",
        value,
        re.IGNORECASE,
    )
    if youtube:
        handle = youtube.group("channel") or f"@{youtube.group('handle')}"
        return AccountCandidate("youtube", handle, _youtube_profile_url(handle))

    instagram = re.search(
        r"(?:https?://)?(?:www\.)?instagram\.com/(?P<handle>[A-Za-z0-9._]+)/?",
        value,
        re.IGNORECASE,
    )
    if instagram:
        handle = instagram.group("handle")
        if handle.lower() in {"p", "reel", "reels", "stories", "explore"}:
            return None
        return AccountCandidate("instagram", handle, f"https://www.instagram.com/{handle}/")

    bare = value.lstrip("@").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{2,100}", bare):
        return None
    if hint == "youtube":
        handle = bare if bare.upper().startswith("UC") else f"@{bare}"
        return AccountCandidate("youtube", handle, _youtube_profile_url(handle))
    if hint in {"", "instagram"}:
        return AccountCandidate("instagram", bare, f"https://www.instagram.com/{bare}/")
    return None


def _youtube_profile_url(handle: str) -> str:
    if handle.upper().startswith("UC"):
        return f"https://www.youtube.com/channel/{handle}"
    return f"https://www.youtube.com/{handle}"


class ResearchStore:
    """Durable, project-scoped source/result/image state for Partner's research workflow."""

    def __init__(self, root: Path):
        self.root = root
        self.database_path = root / "research.sqlite3"
        self.images_root = root / "images"
        self.scripts_root = root / "scripts"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.images_root.mkdir(parents=True, exist_ok=True)
        self.scripts_root.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS counters (
                    kind TEXT NOT NULL,
                    day TEXT NOT NULL,
                    value INTEGER NOT NULL,
                    PRIMARY KEY(kind, day)
                );

                CREATE TABLE IF NOT EXISTS accounts (
                    account_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    handle TEXT NOT NULL,
                    handle_key TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(platform, handle_key)
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    workflow TEXT NOT NULL DEFAULT 'reference_search',
                    query TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    result_count INTEGER NOT NULL DEFAULT 0,
                    provider_task_id TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    requested_by INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    external_id TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    source_url TEXT NOT NULL,
                    creator TEXT NOT NULL DEFAULT '',
                    published_at TEXT NOT NULL DEFAULT '',
                    duration_seconds INTEGER NOT NULL DEFAULT 0,
                    views INTEGER NOT NULL DEFAULT 0,
                    likes INTEGER NOT NULL DEFAULT 0,
                    comments INTEGER NOT NULL DEFAULT 0,
                    thumbnail_url TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    raw_json TEXT NOT NULL DEFAULT '{}',
                    script_path TEXT NOT NULL DEFAULT '',
                    shared_item_id TEXT NOT NULL DEFAULT '',
                    idea_package_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, fingerprint),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS images (
                    asset_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    reference_path TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    file_path TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    requested_by INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS idea_history (
                    idea_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    result_id INTEGER NOT NULL,
                    platform TEXT NOT NULL,
                    primary_source_key TEXT NOT NULL,
                    idea_key TEXT NOT NULL,
                    idea_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id),
                    FOREIGN KEY(result_id) REFERENCES results(result_id)
                );

                CREATE TABLE IF NOT EXISTS source_usage (
                    usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    result_id INTEGER NOT NULL,
                    platform TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    duration_seconds INTEGER NOT NULL DEFAULT 0,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, source_key),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id),
                    FOREIGN KEY(result_id) REFERENCES results(result_id)
                );

                CREATE INDEX IF NOT EXISTS idx_results_rank
                ON results(run_id, views DESC, likes DESC, result_id);
                CREATE INDEX IF NOT EXISTS idx_runs_updated
                ON runs(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_images_updated
                ON images(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_idea_history_created
                ON idea_history(created_at DESC, idea_id DESC);
                CREATE INDEX IF NOT EXISTS idx_idea_history_key
                ON idea_history(idea_key);
                CREATE INDEX IF NOT EXISTS idx_source_usage_lookup
                ON source_usage(platform, source_key);
                """
            )
            run_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(runs)").fetchall()
            }
            if "workflow" not in run_columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN workflow TEXT NOT NULL DEFAULT 'reference_search'"
                )
            result_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(results)").fetchall()
            }
            if "idea_package_json" not in result_columns:
                connection.execute(
                    "ALTER TABLE results ADD COLUMN idea_package_json TEXT NOT NULL DEFAULT '{}'"
                )
            self._backfill_dedupe_history(connection)

    def _backfill_dedupe_history(self, connection: sqlite3.Connection) -> None:
        """Index old production packages so an upgrade cannot reuse their sources."""

        rows = connection.execute(
            """
            SELECT result_id, run_id, platform, external_id, source_url,
                   duration_seconds, idea_package_json, created_at
            FROM results
            WHERE idea_package_json IS NOT NULL
              AND idea_package_json != ''
              AND idea_package_json != '{}'
            ORDER BY result_id
            """
        ).fetchall()
        for row in rows:
            try:
                package = json.loads(str(row["idea_package_json"]))
            except (TypeError, ValueError):
                continue
            if not isinstance(package, dict) or package.get("kind") != "production_idea":
                continue
            idea_text = str(package.get("idea", "")).strip()
            if not idea_text:
                continue
            platform = str(row["platform"]).strip().lower()
            primary_source_key = canonical_source_key(
                platform,
                str(row["external_id"]),
                str(row["source_url"]),
            )
            created_at = str(package.get("created_at", "")).strip() or str(row["created_at"])
            connection.execute(
                """
                INSERT OR IGNORE INTO idea_history(
                    run_id, result_id, platform, primary_source_key,
                    idea_key, idea_text, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row["run_id"]),
                    int(row["result_id"]),
                    platform,
                    primary_source_key,
                    str(package.get("idea_key", "")).strip()
                    or idea_fingerprint(idea_text),
                    idea_text,
                    created_at,
                ),
            )

            raw_evidence = package.get("evidence", [])
            evidence = raw_evidence if isinstance(raw_evidence, list) else []
            if not evidence:
                evidence = [
                    {
                        "result_id": row["result_id"],
                        "platform": platform,
                        "external_id": row["external_id"],
                        "url": row["source_url"],
                        "duration_seconds": row["duration_seconds"],
                    }
                ]
            primary_result_id = int(package.get("primary_result_id", row["result_id"]) or row["result_id"])
            for raw in evidence:
                if not isinstance(raw, dict):
                    continue
                source_platform = str(raw.get("platform", platform)).strip().lower()
                source_external_id = str(raw.get("external_id", "")).strip()
                source_url = str(raw.get("url", raw.get("source_url", ""))).strip()
                if not source_external_id and not source_url:
                    continue
                source_key = str(raw.get("source_key", "")).strip() or canonical_source_key(
                    source_platform,
                    source_external_id,
                    source_url,
                )
                source_result_id = int(raw.get("result_id", row["result_id"]) or row["result_id"])
                connection.execute(
                    """
                    INSERT OR IGNORE INTO source_usage(
                        run_id, result_id, platform, source_key,
                        duration_seconds, role, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(row["run_id"]),
                        source_result_id,
                        source_platform,
                        source_key,
                        max(0, int(raw.get("duration_seconds", 0) or 0)),
                        "primary" if source_result_id == primary_result_id else "evidence",
                        created_at,
                    ),
                )

    def import_accounts(self, text: str) -> AccountImportResult:
        candidates, invalid, duplicates = _parse_account_import_with_stats(text)
        imported = 0
        existing = 0
        self.ensure()
        stamp = _now()
        with self._connect() as connection:
            for candidate in candidates:
                current = connection.execute(
                    "SELECT account_id FROM accounts WHERE platform = ? AND handle_key = ?",
                    (candidate.platform, candidate.handle.casefold()),
                ).fetchone()
                if current:
                    existing += 1
                    connection.execute(
                        "UPDATE accounts SET source_url = ?, active = 1, updated_at = ? WHERE account_id = ?",
                        (candidate.source_url, stamp, current["account_id"]),
                    )
                    continue
                connection.execute(
                    """
                    INSERT INTO accounts(platform, handle, handle_key, source_url, active, created_at, updated_at)
                    VALUES(?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        candidate.platform,
                        candidate.handle,
                        candidate.handle.casefold(),
                        candidate.source_url,
                        stamp,
                        stamp,
                    ),
                )
                imported += 1
        return AccountImportResult(
            imported=imported,
            existing=existing,
            invalid=invalid,
            candidates=candidates,
            duplicates=duplicates,
        )

    def list_accounts(self, platform: str = "", *, active_only: bool = True) -> list[ResearchAccount]:
        self.ensure()
        clauses: list[str] = []
        parameters: list[object] = []
        if platform:
            clauses.append("platform = ?")
            parameters.append(_provider(platform))
        if active_only:
            clauses.append("active = 1")
        query = "SELECT * FROM accounts"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY platform, handle_key"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_account_from_row(row) for row in rows]

    def create_run(
        self,
        provider: str,
        query: str,
        requested_by: int,
        *,
        workflow: str = "reference_search",
    ) -> ResearchRun:
        clean_provider = _provider(provider)
        clean_workflow = workflow.strip().lower()
        if clean_workflow not in RUN_WORKFLOWS:
            raise ValueError(f"Неизвестный workflow поиска: {workflow}")
        self.ensure()
        stamp = _now()
        run_id = self._next_id("research", "RS")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, provider, workflow, query, status, requested_by, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, 'pending_confirmation', ?, ?, ?)
                """,
                (
                    run_id,
                    clean_provider,
                    clean_workflow,
                    query.strip(),
                    int(requested_by),
                    stamp,
                    stamp,
                ),
            )
        return self.require_run(run_id)

    def get_run(self, run_id: str) -> ResearchRun | None:
        self.ensure()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id.strip().upper(),)
            ).fetchone()
        return _run_from_row(row) if row else None

    def require_run(self, run_id: str) -> ResearchRun:
        run = self.get_run(run_id)
        if run is None:
            raise ResearchError(f"Поиск не найден: {run_id}", code="run_not_found")
        return run

    def list_runs(self, limit: int = 20) -> list[ResearchRun]:
        self.ensure()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY updated_at DESC, run_id DESC LIMIT ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [_run_from_row(row) for row in rows]

    def update_run(
        self,
        run_id: str,
        status: str,
        *,
        provider_task_id: str | None = None,
        result_count: int | None = None,
        error: str = "",
    ) -> ResearchRun:
        if status not in RUN_STATUSES:
            raise ValueError(f"Неизвестный статус поиска: {status}")
        current = self.require_run(run_id)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs SET status = ?, provider_task_id = ?, result_count = ?, error = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    current.provider_task_id if provider_task_id is None else provider_task_id.strip(),
                    current.result_count if result_count is None else max(0, int(result_count)),
                    error.strip()[:1000],
                    _now(),
                    current.run_id,
                ),
            )
        return self.require_run(current.run_id)

    def recover_incomplete_auto_content(self) -> list[ResearchRun]:
        """Expose legacy runs where collection completed but production did not."""

        self.ensure()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT runs.run_id
                FROM runs
                WHERE runs.workflow = 'auto_content'
                  AND runs.status = 'completed'
                  AND runs.result_count > 0
                  AND NOT EXISTS (
                      SELECT 1 FROM results
                      WHERE results.run_id = runs.run_id
                        AND TRIM(COALESCE(results.script_path, '')) <> ''
                  )
                ORDER BY runs.created_at
                """
            ).fetchall()
        recovered: list[ResearchRun] = []
        for row in rows:
            recovered.append(
                self.update_run(
                    str(row["run_id"]),
                    "failed",
                    error=(
                        "Сбор роликов завершён, но сценарий не создан. Старая версия "
                        "не сохранила точную причину; можно повторить только этап идеи "
                        "и сценария без нового сбора."
                    ),
                )
            )
        return recovered

    def recover_interrupted(self) -> list[ResearchRun]:
        """Fail local runs after restart; remote snapshots keep their ID for safe resume."""

        recovered: list[ResearchRun] = []
        for run in self.list_runs(limit=100):
            if run.status != "running":
                continue
            if run.provider == "instagram" and run.provider_task_id:
                recovered.append(run)
            else:
                self.update_run(
                    run.run_id,
                    "failed",
                    error="Процесс был прерван перезапуском. Поиск можно запустить повторно.",
                )
        return recovered

    def save_results(
        self,
        run_id: str,
        items: Sequence[dict[str, Any]],
        *,
        mark_completed: bool = True,
    ) -> list[ResearchItem]:
        """Persist collected media; auto-content callers finish after script handoff."""

        run = self.require_run(run_id)
        stamp = _now()
        self.ensure()
        with self._connect() as connection:
            for item in items:
                source_url = str(item.get("source_url", "")).strip()
                if not source_url:
                    continue
                platform = str(item.get("platform", run.provider)).strip().lower()
                external_id = str(item.get("external_id", "")).strip()
                fingerprint = hashlib.sha256(
                    f"{platform}|{external_id or source_url}".encode("utf-8")
                ).hexdigest()
                connection.execute(
                    """
                    INSERT INTO results(
                        run_id, fingerprint, platform, external_id, title, source_url, creator,
                        published_at, duration_seconds, views, likes, comments, thumbnail_url,
                        description, raw_json, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, fingerprint) DO UPDATE SET
                        title = excluded.title, creator = excluded.creator,
                        published_at = excluded.published_at, duration_seconds = excluded.duration_seconds,
                        views = excluded.views, likes = excluded.likes, comments = excluded.comments,
                        thumbnail_url = excluded.thumbnail_url, description = excluded.description,
                        raw_json = excluded.raw_json
                    """,
                    (
                        run.run_id,
                        fingerprint,
                        platform,
                        external_id,
                        str(item.get("title", "")).strip(),
                        source_url,
                        str(item.get("creator", "")).strip(),
                        str(item.get("published_at", "")).strip(),
                        max(0, int(item.get("duration_seconds", 0) or 0)),
                        max(0, int(item.get("views", 0) or 0)),
                        max(0, int(item.get("likes", 0) or 0)),
                        max(0, int(item.get("comments", 0) or 0)),
                        str(item.get("thumbnail_url", "")).strip(),
                        str(item.get("description", "")).strip()[:12000],
                        json.dumps(item.get("raw", {}), ensure_ascii=False)[:50000],
                        stamp,
                    ),
                )
        results = self.list_results(run.run_id, limit=500)
        self.update_run(
            run.run_id,
            "completed" if mark_completed else "running",
            result_count=len(results),
            error="",
        )
        return results

    def list_results(self, run_id: str = "", limit: int = 20) -> list[ResearchItem]:
        self.ensure()
        query = "SELECT * FROM results"
        parameters: list[object] = []
        if run_id:
            query += " WHERE run_id = ?"
            parameters.append(run_id.strip().upper())
        query += " ORDER BY views DESC, likes DESC, result_id ASC LIMIT ?"
        parameters.append(max(1, min(int(limit), 500)))
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_item_from_row(row) for row in rows]

    def get_result(self, result_id: int) -> ResearchItem | None:
        self.ensure()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM results WHERE result_id = ?", (int(result_id),)
            ).fetchone()
        return _item_from_row(row) if row else None

    def require_result(self, result_id: int) -> ResearchItem:
        item = self.get_result(result_id)
        if item is None:
            raise ResearchError(f"Результат не найден: {result_id}", code="result_not_found")
        return item

    def link_script(self, result_id: int, script_path: Path) -> ResearchItem:
        item = self.require_result(result_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE results SET script_path = ? WHERE result_id = ?",
                (str(script_path), item.result_id),
            )
        return self.require_result(item.result_id)

    def save_production_script(
        self,
        run_id: str,
        result_id: int,
        *,
        source_text: str,
        script_text: str,
    ) -> Path:
        """Persist radar production output outside Partner's personal/project memory."""

        run = self.require_run(run_id)
        item = self.require_result(result_id)
        if item.run_id != run.run_id:
            raise ResearchError(
                "Результат не принадлежит указанному запуску радара.",
                code="result_run_mismatch",
            )
        clean_script = script_text.strip()
        if not clean_script:
            raise ResearchError(
                "Производственный сценарий пуст.",
                code="invalid_factory_script",
            )
        self.ensure()
        path = self.scripts_root / f"{run.run_id}-{item.result_id}.md"
        path.write_text(
            "# Производственный сценарий радара\n\n"
            f"Запуск: {run.run_id}\n"
            f"Результат: {item.result_id}\n"
            "Получатель: AI-video content factory\n\n"
            "## Входные данные\n\n"
            f"{source_text.strip()}\n\n"
            "## Результат\n\n"
            f"{clean_script}\n",
            encoding="utf-8",
            newline="\n",
        )
        self.link_script(item.result_id, path)
        return path

    def link_shared_item(self, result_id: int, shared_item_id: str) -> ResearchItem:
        item = self.require_result(result_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE results SET shared_item_id = ? WHERE result_id = ?",
                (shared_item_id.strip().upper(), item.result_id),
            )
        return self.require_result(item.result_id)

    def save_idea_package(
        self,
        result_id: int,
        package: dict[str, Any],
    ) -> ResearchItem:
        item = self.require_result(result_id)
        serialized = json.dumps(
            package,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            connection.execute(
                "UPDATE results SET idea_package_json = ? WHERE result_id = ?",
                (serialized, item.result_id),
            )
        return self.require_result(item.result_id)

    def list_idea_history(self, limit: int = 200) -> list[IdeaHistoryItem]:
        self.ensure()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM idea_history
                ORDER BY created_at DESC, idea_id DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 2000)),),
            ).fetchall()
        return [_idea_history_from_row(row) for row in rows]

    def used_idea_texts(self, limit: int = 200) -> list[str]:
        return [item.idea_text for item in self.list_idea_history(limit=limit)]

    def find_similar_idea(
        self,
        idea_text: str,
        *,
        threshold: float = IDEA_SIMILARITY_THRESHOLD,
    ) -> tuple[IdeaHistoryItem, float] | None:
        clean_idea = idea_text.strip()
        if not clean_idea:
            return None
        idea_key = idea_fingerprint(clean_idea)
        best: tuple[IdeaHistoryItem, float] | None = None
        for existing in self.list_idea_history(limit=2000):
            score = (
                1.0
                if existing.idea_key == idea_key
                else idea_similarity(clean_idea, existing.idea_text)
            )
            if score < threshold:
                continue
            if best is None or score > best[1]:
                best = (existing, score)
        return best

    def source_usage_count(self, item: ResearchItem) -> int:
        self.ensure()
        source_key = canonical_source_key(
            item.platform,
            item.external_id,
            item.source_url,
        )
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS value
                FROM source_usage
                WHERE platform = ? AND source_key = ?
                """,
                (item.platform, source_key),
            ).fetchone()
        return int(row["value"] or 0)

    def filter_available_candidates(
        self,
        candidates: Sequence[RankedResearchItem],
    ) -> list[RankedResearchItem]:
        """Exclude consumed short sources while allowing new ideas from long YouTube."""

        self.ensure()
        with self._connect() as connection:
            used_source_keys = {
                (str(row["platform"]), str(row["source_key"]))
                for row in connection.execute(
                    "SELECT DISTINCT platform, source_key FROM source_usage"
                ).fetchall()
            }
        available: list[RankedResearchItem] = []
        seen_in_batch: set[str] = set()
        for candidate in candidates:
            item = candidate.item
            source_key = canonical_source_key(
                item.platform,
                item.external_id,
                item.source_url,
            )
            if source_key in seen_in_batch:
                continue
            seen_in_batch.add(source_key)
            already_used = (item.platform, source_key) in used_source_keys
            reusable_long_video = (
                item.platform == "youtube"
                and item.duration_seconds >= LONG_FORM_YOUTUBE_SECONDS
            )
            if not already_used or reusable_long_video:
                available.append(candidate)
        return available

    def save_production_idea(
        self,
        result_id: int,
        package: dict[str, Any],
    ) -> ResearchItem:
        """Atomically reserve the idea and every consumed source before handoff."""

        item = self.require_result(result_id)
        if package.get("kind") != "production_idea":
            raise ResearchError(
                "Пакет не является производственной идеей.",
                code="invalid_idea_package",
            )
        idea_text = str(package.get("idea", "")).strip()
        if not idea_text:
            raise ResearchError("Идея не заполнена.", code="invalid_idea_package")
        if int(package.get("primary_result_id", result_id) or result_id) != item.result_id:
            raise ResearchError(
                "Главный источник пакета не совпадает с сохраняемым результатом.",
                code="invalid_idea_package",
            )

        normalized_package = json.loads(
            json.dumps(package, ensure_ascii=False)
        )
        normalized_package["idea_key"] = idea_fingerprint(idea_text)
        raw_evidence = normalized_package.get("evidence", [])
        evidence = raw_evidence if isinstance(raw_evidence, list) else []
        if not evidence:
            evidence = [
                {
                    "result_id": item.result_id,
                    "platform": item.platform,
                    "external_id": item.external_id,
                    "url": item.source_url,
                    "duration_seconds": item.duration_seconds,
                }
            ]
            normalized_package["evidence"] = evidence

        sources: list[dict[str, Any]] = []
        for raw in evidence:
            if not isinstance(raw, dict):
                continue
            source_result_id = int(raw.get("result_id", item.result_id) or item.result_id)
            source_item = self.require_result(source_result_id)
            platform = str(raw.get("platform", source_item.platform)).strip().lower()
            external_id = str(raw.get("external_id", source_item.external_id)).strip()
            source_url = str(
                raw.get("url", raw.get("source_url", source_item.source_url))
            ).strip()
            duration_seconds = max(
                0,
                int(raw.get("duration_seconds", source_item.duration_seconds) or 0),
            )
            source_key = str(raw.get("source_key", "")).strip() or canonical_source_key(
                platform,
                external_id,
                source_url,
            )
            raw.update(
                {
                    "platform": platform,
                    "external_id": external_id,
                    "url": source_url,
                    "duration_seconds": duration_seconds,
                    "source_key": source_key,
                }
            )
            sources.append(
                {
                    "result_id": source_result_id,
                    "platform": platform,
                    "source_key": source_key,
                    "duration_seconds": duration_seconds,
                }
            )
        if not sources:
            raise ResearchError(
                "У идеи нет проверяемого исходного ролика.",
                code="invalid_idea_package",
            )

        serialized = json.dumps(
            normalized_package,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_run = connection.execute(
                "SELECT result_id FROM idea_history WHERE run_id = ?",
                (item.run_id,),
            ).fetchone()
            if existing_run:
                if int(existing_run["result_id"]) != item.result_id:
                    raise ResearchError(
                        "Для этого поиска уже сохранена другая идея.",
                        code="duplicate_idea",
                    )
                connection.execute(
                    "UPDATE results SET idea_package_json = ? WHERE result_id = ?",
                    (serialized, item.result_id),
                )
                saved_row = connection.execute(
                    "SELECT * FROM results WHERE result_id = ?",
                    (item.result_id,),
                ).fetchone()
                return _item_from_row(saved_row)

            for row in connection.execute(
                "SELECT * FROM idea_history ORDER BY idea_id"
            ).fetchall():
                existing = _idea_history_from_row(row)
                score = (
                    1.0
                    if existing.idea_key == normalized_package["idea_key"]
                    else idea_similarity(idea_text, existing.idea_text)
                )
                if score >= IDEA_SIMILARITY_THRESHOLD:
                    raise ResearchError(
                        "Такая идея уже использовалась. Радар должен выбрать другую.",
                        code="duplicate_idea",
                    )

            for source in sources:
                used = connection.execute(
                    """
                    SELECT COUNT(*) AS value
                    FROM source_usage
                    WHERE platform = ? AND source_key = ?
                    """,
                    (source["platform"], source["source_key"]),
                ).fetchone()
                reusable_long_video = (
                    source["platform"] == "youtube"
                    and int(source["duration_seconds"]) >= LONG_FORM_YOUTUBE_SECONDS
                )
                if int(used["value"] or 0) > 0 and not reusable_long_video:
                    raise ResearchError(
                        "Этот исходный ролик уже использовался для другой идеи.",
                        code="duplicate_source",
                    )

            stamp = str(normalized_package.get("created_at", "")).strip() or _now()
            primary_source_key = canonical_source_key(
                item.platform,
                item.external_id,
                item.source_url,
            )
            connection.execute(
                "UPDATE results SET idea_package_json = ? WHERE result_id = ?",
                (serialized, item.result_id),
            )
            connection.execute(
                """
                INSERT INTO idea_history(
                    run_id, result_id, platform, primary_source_key,
                    idea_key, idea_text, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.run_id,
                    item.result_id,
                    item.platform,
                    primary_source_key,
                    normalized_package["idea_key"],
                    idea_text,
                    stamp,
                ),
            )
            for source in sources:
                connection.execute(
                    """
                    INSERT INTO source_usage(
                        run_id, result_id, platform, source_key,
                        duration_seconds, role, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.run_id,
                        source["result_id"],
                        source["platform"],
                        source["source_key"],
                        source["duration_seconds"],
                        "primary"
                        if int(source["result_id"]) == item.result_id
                        else "evidence",
                        stamp,
                    ),
                )
        return self.require_result(item.result_id)

    def create_image(self, prompt: str, requested_by: int, reference_path: Path | None = None) -> ImageAsset:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Промпт изображения не может быть пустым.")
        self.ensure()
        asset_id = self._next_id("image", "DI")
        stamp = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO images(asset_id, prompt, reference_path, status, requested_by, created_at, updated_at)
                VALUES(?, ?, ?, 'pending_confirmation', ?, ?, ?)
                """,
                (
                    asset_id,
                    clean_prompt[:24000],
                    str(reference_path.resolve()) if reference_path else "",
                    int(requested_by),
                    stamp,
                    stamp,
                ),
            )
        return self.require_image(asset_id)

    def get_image(self, asset_id: str) -> ImageAsset | None:
        self.ensure()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM images WHERE asset_id = ?", (asset_id.strip().upper(),)
            ).fetchone()
        return _image_from_row(row) if row else None

    def require_image(self, asset_id: str) -> ImageAsset:
        asset = self.get_image(asset_id)
        if asset is None:
            raise ResearchError(f"Изображение не найдено: {asset_id}", code="image_not_found")
        return asset

    def update_image(
        self,
        asset_id: str,
        status: str,
        *,
        file_path: Path | None = None,
        error: str = "",
    ) -> ImageAsset:
        if status not in IMAGE_STATUSES:
            raise ValueError(f"Неизвестный статус изображения: {status}")
        current = self.require_image(asset_id)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE images SET status = ?, file_path = ?, error = ?, updated_at = ? WHERE asset_id = ?
                """,
                (
                    status,
                    current.file_path if file_path is None else str(file_path.resolve()),
                    error.strip()[:1000],
                    _now(),
                    current.asset_id,
                ),
            )
        return self.require_image(current.asset_id)

    def recover_images(self) -> int:
        self.ensure()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE images SET status = 'failed',
                    error = 'Генерация была прервана перезапуском. Её можно повторить.',
                    updated_at = ?
                WHERE status = 'generating'
                """,
                (_now(),),
            )
        return int(cursor.rowcount)

    def analytics(self) -> dict[str, int]:
        self.ensure()
        with self._connect() as connection:
            account_rows = connection.execute(
                "SELECT platform, COUNT(*) AS value FROM accounts WHERE active = 1 GROUP BY platform"
            ).fetchall()
            totals = connection.execute(
                """
                SELECT COUNT(*) AS results,
                       COALESCE(SUM(views), 0) AS views,
                       COALESCE(SUM(likes), 0) AS likes,
                       SUM(CASE WHEN script_path <> '' THEN 1 ELSE 0 END) AS scripts,
                       SUM(CASE WHEN shared_item_id <> '' THEN 1 ELSE 0 END) AS handoffs
                FROM results
                """
            ).fetchone()
            images = connection.execute(
                "SELECT COUNT(*) AS value FROM images WHERE status = 'completed'"
            ).fetchone()
            completed_runs = connection.execute(
                "SELECT COUNT(*) AS value FROM runs WHERE status = 'completed'"
            ).fetchone()
        result = {"youtube_accounts": 0, "instagram_accounts": 0}
        for row in account_rows:
            result[f"{row['platform']}_accounts"] = int(row["value"])
        result.update(
            {
                "results": int(totals["results"] or 0),
                "views": int(totals["views"] or 0),
                "likes": int(totals["likes"] or 0),
                "scripts": int(totals["scripts"] or 0),
                "handoffs": int(totals["handoffs"] or 0),
                "images": int(images["value"] or 0),
                "completed_runs": int(completed_runs["value"] or 0),
            }
        )
        return result

    def _next_id(self, kind: str, prefix: str) -> str:
        self.ensure()
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value FROM counters WHERE kind = ? AND day = ?", (kind, day)
            ).fetchone()
            value = int(row["value"]) + 1 if row else 1
            connection.execute(
                """
                INSERT INTO counters(kind, day, value) VALUES(?, ?, ?)
                ON CONFLICT(kind, day) DO UPDATE SET value = excluded.value
                """,
                (kind, day, value),
            )
        return f"{prefix}-{day}-{value:03d}"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()


class JsonHttpTransport:
    """Credential-safe JSON transport shared by external research adapters."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        timeout: int = 60,
    ) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request_headers = {"Accept": "application/json", **(headers or {})}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise ResearchError(
                    "Сервис отклонил авторизацию или права ключа.", code=f"http_{exc.code}"
                ) from exc
            if exc.code == 429:
                raise ResearchError(
                    "Сервис временно ограничил частоту запросов.", code="http_429"
                ) from exc
            raise ResearchError(
                f"Внешний сервис вернул HTTP {exc.code}.", code=f"http_{exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ResearchError("Не удалось связаться с внешним сервисом.", code="network_error") from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ResearchError("Внешний сервис вернул невалидный JSON.", code="invalid_json") from exc


class YouTubeResearchClient:
    """Official YouTube Data API adapter for known creators and keyword discovery."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://www.googleapis.com/youtube/v3",
        transport: JsonHttpTransport | None = None,
        days: int = 30,
        results_per_account: int = 5,
    ):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.transport = transport or JsonHttpTransport()
        self.days = max(1, min(int(days), 3650))
        self.results_per_account = max(1, min(int(results_per_account), 50))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def collect(
        self,
        accounts: Sequence[ResearchAccount],
        query: str,
        *,
        limit: int = 50,
        cancelled: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.is_configured:
            raise ResearchError(
                "YouTube API не подключён. Добавь PARTNER_YOUTUBE_API_KEY.",
                code="youtube_not_configured",
            )
        is_cancelled = cancelled or (lambda: False)
        video_ids: list[str] = []
        if accounts:
            for account in accounts:
                if is_cancelled():
                    raise ResearchError("Поиск остановлен.", code="cancelled")
                try:
                    channel = self._channel(account.handle)
                except ResearchError as exc:
                    if exc.code == "channel_not_found":
                        continue
                    raise
                uploads = (
                    channel.get("contentDetails", {})
                    .get("relatedPlaylists", {})
                    .get("uploads", "")
                )
                if not uploads:
                    continue
                response = self._get(
                    "playlistItems",
                    part="contentDetails",
                    playlistId=uploads,
                    maxResults=self.results_per_account,
                )
                for item in response.get("items", []):
                    video_id = str(item.get("contentDetails", {}).get("videoId", "")).strip()
                    if video_id and video_id not in video_ids:
                        video_ids.append(video_id)
                if len(video_ids) >= limit:
                    break
        else:
            clean_query = query.strip()
            if not clean_query:
                raise ResearchError("Для поиска без списка каналов нужна тема.", code="query_required")
            response = self._get(
                "search",
                part="snippet",
                q=clean_query,
                type="video",
                order="viewCount",
                publishedAfter=(
                    datetime.now(timezone.utc) - timedelta(days=self.days)
                ).isoformat(timespec="seconds").replace("+00:00", "Z"),
                maxResults=min(max(1, int(limit)), 50),
            )
            video_ids = [
                str(item.get("id", {}).get("videoId", "")).strip()
                for item in response.get("items", [])
                if str(item.get("id", {}).get("videoId", "")).strip()
            ]
        if not video_ids:
            return []
        details: list[dict[str, Any]] = []
        for index in range(0, len(video_ids), 50):
            if is_cancelled():
                raise ResearchError("Поиск остановлен.", code="cancelled")
            response = self._get(
                "videos",
                part="snippet,statistics,contentDetails",
                id=",".join(video_ids[index : index + 50]),
                maxResults=50,
            )
            details.extend(response.get("items", []))
        normalized = [_normalize_youtube(item) for item in details]
        keywords = [word.casefold() for word in re.findall(r"[\w-]{3,}", query, re.UNICODE)]
        if keywords and accounts:
            filtered = [
                item
                for item in normalized
                if any(word in f"{item['title']} {item['description']}".casefold() for word in keywords)
            ]
            if filtered:
                normalized = filtered
        normalized.sort(key=lambda item: (int(item["views"]), int(item["likes"])), reverse=True)
        return normalized[: max(1, int(limit))]

    def _channel(self, handle: str) -> dict[str, Any]:
        parameters: dict[str, Any] = {"part": "snippet,contentDetails"}
        if handle.upper().startswith("UC"):
            parameters["id"] = handle
        else:
            parameters["forHandle"] = handle.lstrip("@")
        response = self._get("channels", **parameters)
        items = response.get("items", [])
        if not items:
            raise ResearchError(f"YouTube-канал не найден: {handle}", code="channel_not_found")
        return dict(items[0])

    def _get(self, resource: str, **parameters: Any) -> dict[str, Any]:
        parameters["key"] = self.api_key
        url = f"{self.base_url}/{resource}?{urllib.parse.urlencode(parameters)}"
        decoded = self.transport.request("GET", url, timeout=60)
        if not isinstance(decoded, dict):
            raise ResearchError("YouTube вернул неожиданный формат.", code="invalid_response")
        return decoded


class BrightDataInstagramClient:
    """Bright Data Instagram Reels adapter with durable snapshot IDs."""

    def __init__(
        self,
        api_token: str,
        *,
        base_url: str = "https://api.brightdata.com",
        dataset_id: str = "gd_lyclm20il4r5helnj",
        transport: JsonHttpTransport | None = None,
        days: int = 30,
        results_per_account: int = 5,
        poll_interval_seconds: int = 8,
    ):
        self.api_token = api_token.strip()
        self.base_url = base_url.rstrip("/")
        self.dataset_id = dataset_id.strip()
        self.transport = transport or JsonHttpTransport()
        self.days = max(1, min(int(days), 3650))
        self.results_per_account = max(1, min(int(results_per_account), 100))
        self.poll_interval_seconds = max(1, int(poll_interval_seconds))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_token and self.dataset_id)

    def check_connection(self) -> None:
        """Validate authentication without starting a billable collection."""

        if not self.is_configured:
            raise ResearchError(
                "Instagram-парсер не подключён. Добавь BRIGHTDATA_API_TOKEN.",
                code="instagram_not_configured",
            )
        query_string = urllib.parse.urlencode(
            {"dataset_id": self.dataset_id, "limit": 1}
        )
        decoded = self.transport.request(
            "GET",
            f"{self.base_url}/datasets/v3/snapshots?{query_string}",
            headers=self._headers(),
            timeout=60,
        )
        if not isinstance(decoded, list):
            raise ResearchError(
                "Bright Data вернул неожиданный ответ при проверке подключения.",
                code="invalid_response",
            )

    def start(self, accounts: Sequence[ResearchAccount], query: str, limit: int) -> str:
        del query  # Theme filtering happens after collection against verified records.
        if not self.is_configured:
            raise ResearchError(
                "Instagram-парсер не подключён. Добавь BRIGHTDATA_API_TOKEN.",
                code="instagram_not_configured",
            )
        if not accounts:
            raise ResearchError(
                "Для поиска Bright Data нужны сохранённые Instagram-аккаунты.",
                code="instagram_accounts_required",
            )

        total_limit = min(
            max(1, int(limit)),
            len(accounts) * self.results_per_account,
        )
        selected_accounts = list(accounts[: min(len(accounts), total_limit)])
        base_quota, extra = divmod(total_limit, len(selected_accounts))
        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=self.days)).strftime("%m-%d-%Y")
        end_date = now.strftime("%m-%d-%Y")
        payload: list[dict[str, Any]] = []
        for index, account in enumerate(selected_accounts):
            profile_url = (
                account.source_url.strip()
                or f"https://www.instagram.com/{account.handle.strip().lstrip('@')}/"
            )
            payload.append(
                {
                    "url": profile_url,
                    "num_of_posts": min(
                        self.results_per_account,
                        base_quota + (1 if index < extra else 0),
                    ),
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )

        query_string = urllib.parse.urlencode(
            {
                "dataset_id": self.dataset_id,
                "type": "discover_new",
                "discover_by": "url",
                "format": "json",
                "include_errors": "true",
            }
        )
        decoded = self.transport.request(
            "POST",
            f"{self.base_url}/datasets/v3/trigger?{query_string}",
            headers=self._headers(),
            payload=payload,
            timeout=60,
        )
        snapshot_id = (
            str(decoded.get("snapshot_id", "")).strip()
            if isinstance(decoded, dict)
            else ""
        )
        if not snapshot_id:
            raise ResearchError(
                "Bright Data не вернул snapshot ID.",
                code="missing_task_id",
            )
        return snapshot_id

    def wait_for_results(
        self,
        task_id: str,
        *,
        cancelled: Callable[[], bool] | None = None,
        timeout_seconds: int = 900,
    ) -> list[dict[str, Any]]:
        snapshot_id = task_id.strip()
        if not snapshot_id:
            raise ResearchError("Не указан snapshot ID.", code="missing_task_id")
        is_cancelled = cancelled or (lambda: False)
        deadline = time.monotonic() + max(30, int(timeout_seconds))
        while time.monotonic() < deadline:
            if is_cancelled():
                self.abort(snapshot_id)
                raise ResearchError("Поиск остановлен.", code="cancelled")
            decoded = self.transport.request(
                "GET",
                f"{self.base_url}/datasets/v3/progress/"
                f"{urllib.parse.quote(snapshot_id, safe='')}",
                headers=self._headers(),
                timeout=60,
            )
            status = (
                str(decoded.get("status", "")).strip().lower()
                if isinstance(decoded, dict)
                else ""
            )
            if status == "ready":
                break
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise ResearchError(
                    f"Bright Data завершил сбор со статусом {status}.",
                    code="snapshot_failed",
                )
            time.sleep(self.poll_interval_seconds)
        else:
            raise ResearchError(
                "Instagram-парсер не завершился вовремя.",
                code="timeout",
            )

        decoded = self.transport.request(
            "GET",
            f"{self.base_url}/datasets/v3/snapshot/"
            f"{urllib.parse.quote(snapshot_id, safe='')}?format=json",
            headers=self._headers(),
            timeout=120,
        )
        records: Any = decoded
        if isinstance(decoded, dict):
            records = decoded.get("data") or decoded.get("results")
        if not isinstance(records, list):
            raise ResearchError(
                "Bright Data snapshot имеет неожиданный формат.",
                code="invalid_response",
            )
        normalized: list[dict[str, Any]] = []
        for item in records:
            if not isinstance(item, dict):
                continue
            if item.get("error") and not item.get("url"):
                continue
            result = _normalize_instagram(item)
            if result["source_url"]:
                normalized.append(result)
        return normalized

    def abort(self, task_id: str) -> None:
        if not self.is_configured or not task_id.strip():
            return
        try:
            self.transport.request(
                "POST",
                f"{self.base_url}/datasets/v3/snapshot/"
                f"{urllib.parse.quote(task_id.strip(), safe='')}/cancel",
                headers=self._headers(),
                timeout=60,
            )
        except ResearchError:
            # Cancellation is best-effort; a finished snapshot can reject the request.
            return

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}


class ApifyInstagramClient:
    """Optional paid Instagram adapter with resumable task IDs and a hard charge cap."""

    def __init__(
        self,
        api_token: str,
        *,
        base_url: str = "https://api.apify.com/v2",
        actor_id: str = "apify~instagram-scraper",
        max_total_charge_usd: float = 1.0,
        transport: JsonHttpTransport | None = None,
        days: int = 30,
        results_per_account: int = 5,
        poll_interval_seconds: int = 8,
    ):
        self.api_token = api_token.strip()
        self.base_url = base_url.rstrip("/")
        self.actor_id = actor_id.strip().replace("/", "~")
        self.max_total_charge_usd = max(0.01, float(max_total_charge_usd))
        self.transport = transport or JsonHttpTransport()
        self.days = max(1, min(int(days), 3650))
        self.results_per_account = max(1, min(int(results_per_account), 100))
        self.poll_interval_seconds = max(1, int(poll_interval_seconds))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_token and self.actor_id)

    def start(self, accounts: Sequence[ResearchAccount], query: str, limit: int) -> str:
        if not self.is_configured:
            raise ResearchError(
                "Instagram-парсер не подключён. Добавь PARTNER_APIFY_API_TOKEN.",
                code="instagram_not_configured",
            )
        if accounts:
            payload: dict[str, Any] = {
                "directUrls": [account.source_url or f"https://www.instagram.com/{account.handle}/" for account in accounts],
                "resultsType": "reels",
                "resultsLimit": min(self.results_per_account, max(1, int(limit))),
                "onlyPostsNewerThan": f"{self.days} days",
            }
        else:
            if not query.strip():
                raise ResearchError("Для поиска без аккаунтов нужна тема.", code="query_required")
            payload = {
                "search": query.strip(),
                "searchType": "user",
                "searchLimit": min(max(1, int(limit)), 250),
                "resultsType": "reels",
                "resultsLimit": self.results_per_account,
                "onlyPostsNewerThan": f"{self.days} days",
            }
        query_string = urllib.parse.urlencode(
            {"waitForFinish": 0, "maxTotalChargeUsd": f"{self.max_total_charge_usd:.2f}"}
        )
        decoded = self.transport.request(
            "POST",
            f"{self.base_url}/acts/{urllib.parse.quote(self.actor_id, safe='~')}/runs?{query_string}",
            headers=self._headers(),
            payload=payload,
            timeout=60,
        )
        data = decoded.get("data", {}) if isinstance(decoded, dict) else {}
        task_id = str(data.get("id", "")).strip()
        if not task_id:
            raise ResearchError("Apify не вернул task ID.", code="missing_task_id")
        return task_id

    def wait_for_results(
        self,
        task_id: str,
        *,
        cancelled: Callable[[], bool] | None = None,
        timeout_seconds: int = 900,
    ) -> list[dict[str, Any]]:
        is_cancelled = cancelled or (lambda: False)
        deadline = time.monotonic() + max(30, int(timeout_seconds))
        dataset_id = ""
        while time.monotonic() < deadline:
            if is_cancelled():
                self.abort(task_id)
                raise ResearchError("Поиск остановлен.", code="cancelled")
            decoded = self.transport.request(
                "GET",
                f"{self.base_url}/actor-runs/{urllib.parse.quote(task_id, safe='')}",
                headers=self._headers(),
                timeout=60,
            )
            data = decoded.get("data", {}) if isinstance(decoded, dict) else {}
            status = str(data.get("status", "")).upper()
            dataset_id = str(data.get("defaultDatasetId", "")).strip() or dataset_id
            if status == "SUCCEEDED":
                break
            if status in {"FAILED", "ABORTED", "TIMED-OUT"}:
                raise ResearchError(f"Instagram-парсер завершился со статусом {status}.", code="actor_failed")
            time.sleep(self.poll_interval_seconds)
        else:
            raise ResearchError("Instagram-парсер не завершился вовремя.", code="timeout")
        if not dataset_id:
            raise ResearchError("Apify не вернул dataset ID.", code="missing_dataset_id")
        decoded = self.transport.request(
            "GET",
            f"{self.base_url}/datasets/{urllib.parse.quote(dataset_id, safe='')}/items?clean=true&format=json",
            headers=self._headers(),
            timeout=120,
        )
        if not isinstance(decoded, list):
            raise ResearchError("Apify dataset имеет неожиданный формат.", code="invalid_response")
        return [_normalize_instagram(item) for item in decoded if isinstance(item, dict)]

    def abort(self, task_id: str) -> None:
        if not self.is_configured:
            return
        try:
            self.transport.request(
                "POST",
                f"{self.base_url}/actor-runs/{urllib.parse.quote(task_id, safe='')}/abort",
                headers=self._headers(),
                payload={},
                timeout=30,
            )
        except ResearchError:
            return

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}


def _normalize_youtube(item: dict[str, Any]) -> dict[str, Any]:
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    content = item.get("contentDetails", {})
    video_id = str(item.get("id", "")).strip()
    thumbnails = snippet.get("thumbnails", {})
    thumbnail = ""
    for size in ("maxres", "standard", "high", "medium", "default"):
        value = thumbnails.get(size, {})
        if value.get("url"):
            thumbnail = str(value["url"])
            break
    return {
        "platform": "youtube",
        "external_id": video_id,
        "title": str(snippet.get("title", "")),
        "source_url": f"https://www.youtube.com/watch?v={video_id}",
        "creator": str(snippet.get("channelTitle", "")),
        "published_at": str(snippet.get("publishedAt", "")),
        "duration_seconds": _iso8601_seconds(str(content.get("duration", ""))),
        "views": _safe_int(statistics.get("viewCount")),
        "likes": _safe_int(statistics.get("likeCount")),
        "comments": _safe_int(statistics.get("commentCount")),
        "thumbnail_url": thumbnail,
        "description": str(snippet.get("description", "")),
        "raw": item,
    }


def _normalize_instagram(item: dict[str, Any]) -> dict[str, Any]:
    external_id = str(
        item.get("post_id")
        or item.get("content_id")
        or item.get("id")
        or item.get("shortCode")
        or item.get("shortcode")
        or ""
    ).strip()
    source_url = str(item.get("url") or item.get("inputUrl") or "").strip()
    if not external_id:
        external_id = source_url
    caption = str(
        item.get("caption")
        or item.get("description")
        or item.get("text")
        or ""
    ).strip()
    title = caption.splitlines()[0][:180] if caption else "Instagram Reel"
    return {
        "platform": "instagram",
        "external_id": external_id,
        "title": title,
        "source_url": source_url,
        "creator": str(
            item.get("ownerUsername")
            or item.get("username")
            or item.get("user_posted")
            or ""
        ),
        "published_at": str(
            item.get("timestamp")
            or item.get("takenAt")
            or item.get("date_posted")
            or ""
        ),
        "duration_seconds": _safe_int(
            item.get("videoDuration")
            or item.get("duration")
            or item.get("length")
        ),
        "views": _safe_int(
            item.get("videoPlayCount")
            or item.get("video_play_count")
            or item.get("videoViewCount")
            or item.get("viewsCount")
            or item.get("views")
        ),
        "likes": _safe_int(item.get("likesCount") or item.get("likes")),
        "comments": _safe_int(
            item.get("commentsCount")
            or item.get("comments")
            or item.get("num_comments")
        ),
        "thumbnail_url": str(
            item.get("displayUrl")
            or item.get("thumbnailUrl")
            or item.get("imageUrl")
            or item.get("thumbnail")
            or ""
        ),
        "description": caption,
        "raw": item,
    }


def _iso8601_seconds(value: str) -> int:
    match = re.fullmatch(r"P(?:\d+D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
    if not match:
        return 0
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(float(value or 0)))
    except (TypeError, ValueError):
        return 0


def _parse_published_at(value: str) -> datetime | None:
    clean = value.strip()
    if not clean:
        return None
    if re.fullmatch(r"\d{10}(?:\.\d+)?", clean):
        try:
            return datetime.fromtimestamp(float(clean), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(clean.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _provider(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in RESEARCH_PROVIDERS:
        raise ValueError(f"Неизвестная платформа: {value}")
    return normalized


def _account_from_row(row: sqlite3.Row) -> ResearchAccount:
    data = dict(row)
    data.pop("handle_key", None)
    data["active"] = bool(data["active"])
    return ResearchAccount(**data)


def _run_from_row(row: sqlite3.Row) -> ResearchRun:
    return ResearchRun(**dict(row))


def _item_from_row(row: sqlite3.Row) -> ResearchItem:
    data = dict(row)
    data.pop("fingerprint", None)
    data.pop("raw_json", None)
    return ResearchItem(**data)


def _image_from_row(row: sqlite3.Row) -> ImageAsset:
    return ImageAsset(**dict(row))


def _idea_history_from_row(row: sqlite3.Row) -> IdeaHistoryItem:
    return IdeaHistoryItem(**dict(row))


def _idea_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.findall(r"[a-zа-яё0-9]{2,}", value.casefold(), re.IGNORECASE):
        if raw in _IDEA_STOP_WORDS:
            continue
        token = raw
        if not token.isdigit():
            for suffix in _IDEA_SUFFIXES:
                if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                    token = token[: -len(suffix)]
                    break
        if token and token not in _IDEA_STOP_WORDS:
            tokens.append(token)
    return tokens


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
