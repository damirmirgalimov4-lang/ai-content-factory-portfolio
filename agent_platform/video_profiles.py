from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VideoProfile:
    code: str
    model_key: str
    model: str
    model_label: str
    quality_label: str
    mode: str
    resolution: str
    duration_seconds: int = 5
    aspect_ratio: str = "9:16"
    sound_enabled: bool = False
    provider: str = "polza"
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "model_key": self.model_key,
            "model": self.model,
            "model_label": self.model_label,
            "quality_label": self.quality_label,
            "mode": self.mode,
            "resolution": self.resolution,
            "duration_seconds": self.duration_seconds,
            "aspect_ratio": self.aspect_ratio,
            "sound_enabled": self.sound_enabled,
            "provider": self.provider,
            "seed": self.seed,
        }


VIDEO_PROFILES: dict[str, VideoProfile] = {
    "ks": VideoProfile(
        "ks", "kling", "kling/v3", "Kling 3", "Standard", "std", ""
    ),
    "kp": VideoProfile(
        "kp", "kling", "kling/v3", "Kling 3", "Professional", "pro", ""
    ),
    "k4": VideoProfile(
        "k4", "kling", "kling/v3", "Kling 3", "4K", "4K", ""
    ),
    "s480": VideoProfile(
        "s480",
        "seedance",
        "bytedance/seedance-2",
        "Seedance 2",
        "480p · без звука",
        "",
        "480p",
        provider="kie",
    ),
    "s480a": VideoProfile(
        "s480a",
        "seedance",
        "bytedance/seedance-2",
        "Seedance 2",
        "480p · со звуком",
        "",
        "480p",
        sound_enabled=True,
        provider="kie",
    ),
    "s720": VideoProfile(
        "s720", "seedance", "bytedance/seedance-2", "Seedance 2", "720p", "", "720p"
    ),
    "s1080": VideoProfile(
        "s1080", "seedance", "bytedance/seedance-2", "Seedance 2", "1080p", "", "1080p"
    ),
    "l23": VideoProfile(
        "l23",
        "ltx",
        "ltx-2.3",
        "LTX-2.3",
        "1024×576 · distilled · со звуком",
        "distilled",
        "1024x576",
        duration_seconds=5,
        aspect_ratio="16:9",
        sound_enabled=True,
        provider="ltx",
        seed=42,
    ),
}


def video_profile(code: str) -> VideoProfile:
    try:
        return VIDEO_PROFILES[code]
    except KeyError as exc:
        raise ValueError(f"Неизвестный профиль видеогенерации: {code}") from exc


def profiles_for_model(model_key: str) -> list[VideoProfile]:
    return [
        profile
        for profile in VIDEO_PROFILES.values()
        if profile.model_key == model_key
    ]


def profile_label(settings: dict[str, Any] | None) -> str:
    if not isinstance(settings, dict) or not settings:
        return "не выбраны"
    model = str(settings.get("model_label") or settings.get("model") or "модель")
    quality = str(settings.get("quality_label") or settings.get("resolution") or settings.get("mode") or "")
    return f"{model} · {quality}" if quality else model
