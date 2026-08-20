from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs without adding a runtime dependency."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def parse_int_set(value: str) -> set[int]:
    result: set[int] = set()
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        result.add(int(item))
    return result


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    telegram_allowed_user_ids: set[int]
    vault_path: Path
    openai_api_key: str
    openai_base_url: str
    openai_image_model: str
    openai_image_size: str
    openai_image_quality: str
    deepgram_api_key: str
    deepgram_model: str
    deepgram_language: str
    poll_timeout_seconds: int = 30
    codex_cli_path: str = ""
    codex_chat_model: str = "gpt-5.6-sol"
    codex_timeout_seconds: int = 180
    codex_production_timeout_seconds: int = 600
    codex_workdir: Path = Path(".")
    image_provider: str = "codex"
    codex_image_timeout_seconds: int = 600
    polza_api_key: str = ""
    polza_base_url: str = "https://polza.ai/api"
    polza_poll_interval_seconds: int = 8
    polza_timeout_seconds: int = 900
    polza_max_status_retries: int = 3
    kie_api_key: str = ""
    kie_base_url: str = "https://api.kie.ai"
    kie_upload_base_url: str = "https://kieai.redpandaai.co"
    kie_poll_interval_seconds: int = 8
    kie_timeout_seconds: int = 900
    kie_max_status_retries: int = 3
    viktor_api_key: str = ""
    viktor_base_url: str = "https://api.viktor.com"
    viktor_poll_interval_seconds: int = 8
    viktor_timeout_seconds: int = 1200
    viktor_max_status_retries: int = 3
    ltx_video_enabled: bool = False
    ltx_api_token: str = ""
    ltx_base_url: str = ""
    ltx_poll_interval_seconds: int = 8
    ltx_timeout_seconds: int = 1800
    ltx_max_status_retries: int = 3
    shared_content_path: Path | None = None
    youtube_api_key: str = ""
    youtube_base_url: str = "https://www.googleapis.com/youtube/v3"
    brightdata_api_token: str = ""
    brightdata_base_url: str = "https://api.brightdata.com"
    brightdata_instagram_dataset_id: str = "gd_lyclm20il4r5helnj"
    brightdata_poll_interval_seconds: int = 8
    apify_api_token: str = ""
    apify_base_url: str = "https://api.apify.com/v2"
    apify_instagram_actor: str = "apify~instagram-scraper"
    apify_max_total_charge_usd: float = 1.0
    research_days: int = 30
    research_results_limit: int = 50
    research_results_per_account: int = 5
    telegram_tester_user_ids: set[int] = field(default_factory=set)
    test_vault_path: Path | None = None
    test_shared_content_path: Path | None = None
    content_factory_bot_username: str = "ContentFactoryExampleBot"
    radar_vault_path: Path | None = None
    radar_redirect_to_content_factory: bool = False

    @classmethod
    def load(
        cls,
        env_path: Path = Path(".env"),
        *,
        env_prefix: str = "",
    ) -> "Settings":
        load_dotenv(env_path)

        def getenv(name: str, default: str = "") -> str:
            return os.getenv(f"{env_prefix}{name}", default)

        token = getenv("TELEGRAM_BOT_TOKEN").strip()
        allowed_ids_raw = (
            getenv("TELEGRAM_ALLOWED_USER_IDS").strip()
            or getenv("TELEGRAM_CHAT_ID").strip()
        )

        allowed_ids = parse_int_set(allowed_ids_raw) if allowed_ids_raw else set()
        tester_ids_raw = getenv("TELEGRAM_TESTER_USER_IDS").strip()
        tester_ids = parse_int_set(tester_ids_raw) if tester_ids_raw else set()
        vault_path = Path(getenv("VAULT_PATH", "./vault")).expanduser()
        poll_timeout = int(getenv("TELEGRAM_POLL_TIMEOUT_SECONDS", "30"))
        openai_api_key = getenv("OPENAI_API_KEY").strip()
        openai_base_url = getenv(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1",
        ).strip().rstrip("/")
        openai_image_model = getenv("OPENAI_IMAGE_MODEL", "gpt-image-2").strip()
        openai_image_size = getenv("OPENAI_IMAGE_SIZE", "1024x1536").strip()
        openai_image_quality = getenv("OPENAI_IMAGE_QUALITY", "low").strip()
        deepgram_api_key = getenv("DEEPGRAM_API_KEY").strip()
        deepgram_model = getenv("DEEPGRAM_MODEL", "nova-3").strip()
        deepgram_language = getenv("DEEPGRAM_LANGUAGE", "ru").strip()
        codex_cli_path = getenv("CODEX_CLI_PATH").strip()
        codex_chat_model = getenv("CODEX_CHAT_MODEL", "gpt-5.6-sol").strip()
        codex_timeout_seconds = int(getenv("CODEX_TIMEOUT_SECONDS", "180"))
        codex_production_timeout_seconds = int(
            getenv("CODEX_PRODUCTION_TIMEOUT_SECONDS", "600")
        )
        codex_workdir = Path(getenv("CODEX_WORKDIR", ".")).expanduser()
        image_provider = getenv("IMAGE_PROVIDER", "codex").strip().lower()
        codex_image_timeout_seconds = int(
            getenv("CODEX_IMAGE_TIMEOUT_SECONDS", "600")
        )
        # Keep compatibility with the exact mixed-case name already present locally.
        polza_api_key = (
            getenv("POLZA_AI_API_KEY").strip()
            or getenv("PolzaAi_API_KEY").strip()
        )
        polza_base_url = getenv("POLZA_BASE_URL", "https://polza.ai/api").strip().rstrip("/")
        polza_poll_interval_seconds = int(getenv("POLZA_POLL_INTERVAL_SECONDS", "8"))
        polza_timeout_seconds = int(getenv("POLZA_TIMEOUT_SECONDS", "900"))
        polza_max_status_retries = int(getenv("POLZA_MAX_STATUS_RETRIES", "3"))
        # Accept the lowercase local name once, but document and prefer KIE_API_KEY.
        kie_api_key = getenv("KIE_API_KEY").strip() or getenv("kie").strip()
        kie_base_url = getenv("KIE_BASE_URL", "https://api.kie.ai").strip().rstrip("/")
        kie_upload_base_url = getenv(
            "KIE_UPLOAD_BASE_URL",
            "https://kieai.redpandaai.co",
        ).strip().rstrip("/")
        kie_poll_interval_seconds = int(getenv("KIE_POLL_INTERVAL_SECONDS", "8"))
        kie_timeout_seconds = int(getenv("KIE_TIMEOUT_SECONDS", "900"))
        kie_max_status_retries = int(getenv("KIE_MAX_STATUS_RETRIES", "3"))
        viktor_api_key = getenv("VIKTOR_API_KEY").strip()
        viktor_base_url = getenv(
            "VIKTOR_BASE_URL",
            "https://api.viktor.com",
        ).strip().rstrip("/")
        viktor_poll_interval_seconds = int(
            getenv("VIKTOR_POLL_INTERVAL_SECONDS", "8")
        )
        viktor_timeout_seconds = int(getenv("VIKTOR_TIMEOUT_SECONDS", "1200"))
        viktor_max_status_retries = int(
            getenv("VIKTOR_MAX_STATUS_RETRIES", "3")
        )
        ltx_video_enabled = getenv("LTX_VIDEO_ENABLED", "").strip().casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }
        ltx_api_token = getenv("LTX_API_TOKEN").strip()
        ltx_base_url = getenv("LTX_BASE_URL").strip().rstrip("/")
        ltx_poll_interval_seconds = int(getenv("LTX_POLL_INTERVAL_SECONDS", "8"))
        ltx_timeout_seconds = int(getenv("LTX_TIMEOUT_SECONDS", "1800"))
        ltx_max_status_retries = int(getenv("LTX_MAX_STATUS_RETRIES", "3"))
        shared_content_path = Path(
            getenv("SHARED_CONTENT_PATH", "./shared-content")
        ).expanduser()
        youtube_api_key = getenv("YOUTUBE_API_KEY").strip()
        youtube_base_url = getenv(
            "YOUTUBE_BASE_URL", "https://www.googleapis.com/youtube/v3"
        ).strip().rstrip("/")
        brightdata_api_token = getenv("BRIGHTDATA_API_TOKEN").strip()
        brightdata_base_url = getenv(
            "BRIGHTDATA_BASE_URL", "https://api.brightdata.com"
        ).strip().rstrip("/")
        brightdata_instagram_dataset_id = getenv(
            "BRIGHTDATA_INSTAGRAM_DATASET_ID", "gd_lyclm20il4r5helnj"
        ).strip()
        brightdata_poll_interval_seconds = int(
            getenv("BRIGHTDATA_POLL_INTERVAL_SECONDS", "8")
        )
        apify_api_token = getenv("APIFY_API_TOKEN").strip()
        apify_base_url = getenv(
            "APIFY_BASE_URL", "https://api.apify.com/v2"
        ).strip().rstrip("/")
        apify_instagram_actor = getenv(
            "APIFY_INSTAGRAM_ACTOR", "apify~instagram-scraper"
        ).strip()
        apify_max_total_charge_usd = float(
            getenv("APIFY_MAX_TOTAL_CHARGE_USD", "1.0")
        )
        research_days = int(getenv("RESEARCH_DAYS", "30"))
        research_results_limit = int(getenv("RESEARCH_RESULTS_LIMIT", "50"))
        research_results_per_account = int(
            getenv("RESEARCH_RESULTS_PER_ACCOUNT", "5")
        )
        test_vault_path = Path(
            getenv("TEST_VAULT_PATH", "./vault-partner-test")
        ).expanduser()
        test_shared_content_path = Path(
            getenv("TEST_SHARED_CONTENT_PATH", "./shared-content-test")
        ).expanduser()
        content_factory_bot_username = getenv(
            "CONTENT_FACTORY_BOT_USERNAME",
            "ContentFactoryExampleBot",
        ).strip().lstrip("@")
        radar_vault_raw = getenv("RADAR_VAULT_PATH").strip()
        radar_vault_path = (
            Path(radar_vault_raw).expanduser() if radar_vault_raw else None
        )
        radar_redirect_to_content_factory = getenv(
            "RADAR_REDIRECT_TO_CONTENT_FACTORY",
            "",
        ).strip().casefold() in {"1", "true", "yes", "on"}

        return cls(
            telegram_bot_token=token,
            telegram_allowed_user_ids=allowed_ids,
            vault_path=vault_path,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_image_model=openai_image_model,
            openai_image_size=openai_image_size,
            openai_image_quality=openai_image_quality,
            deepgram_api_key=deepgram_api_key,
            deepgram_model=deepgram_model,
            deepgram_language=deepgram_language,
            poll_timeout_seconds=poll_timeout,
            codex_cli_path=codex_cli_path,
            codex_chat_model=codex_chat_model,
            codex_timeout_seconds=codex_timeout_seconds,
            codex_production_timeout_seconds=codex_production_timeout_seconds,
            codex_workdir=codex_workdir,
            image_provider=image_provider,
            codex_image_timeout_seconds=codex_image_timeout_seconds,
            polza_api_key=polza_api_key,
            polza_base_url=polza_base_url,
            polza_poll_interval_seconds=polza_poll_interval_seconds,
            polza_timeout_seconds=polza_timeout_seconds,
            polza_max_status_retries=polza_max_status_retries,
            kie_api_key=kie_api_key,
            kie_base_url=kie_base_url,
            kie_upload_base_url=kie_upload_base_url,
            kie_poll_interval_seconds=kie_poll_interval_seconds,
            kie_timeout_seconds=kie_timeout_seconds,
            kie_max_status_retries=kie_max_status_retries,
            viktor_api_key=viktor_api_key,
            viktor_base_url=viktor_base_url,
            viktor_poll_interval_seconds=viktor_poll_interval_seconds,
            viktor_timeout_seconds=viktor_timeout_seconds,
            viktor_max_status_retries=viktor_max_status_retries,
            ltx_video_enabled=ltx_video_enabled,
            ltx_api_token=ltx_api_token,
            ltx_base_url=ltx_base_url,
            ltx_poll_interval_seconds=ltx_poll_interval_seconds,
            ltx_timeout_seconds=ltx_timeout_seconds,
            ltx_max_status_retries=ltx_max_status_retries,
            shared_content_path=shared_content_path,
            youtube_api_key=youtube_api_key,
            youtube_base_url=youtube_base_url,
            brightdata_api_token=brightdata_api_token,
            brightdata_base_url=brightdata_base_url,
            brightdata_instagram_dataset_id=brightdata_instagram_dataset_id,
            brightdata_poll_interval_seconds=brightdata_poll_interval_seconds,
            apify_api_token=apify_api_token,
            apify_base_url=apify_base_url,
            apify_instagram_actor=apify_instagram_actor,
            apify_max_total_charge_usd=apify_max_total_charge_usd,
            research_days=research_days,
            research_results_limit=research_results_limit,
            research_results_per_account=research_results_per_account,
            telegram_tester_user_ids=tester_ids,
            test_vault_path=test_vault_path,
            test_shared_content_path=test_shared_content_path,
            content_factory_bot_username=content_factory_bot_username,
            radar_vault_path=radar_vault_path,
            radar_redirect_to_content_factory=radar_redirect_to_content_factory,
        )
