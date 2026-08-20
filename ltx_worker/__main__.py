from __future__ import annotations

import os

from .api import create_server
from .config import WorkerSettings
from .runner import ComfyUiRunner
from .service import VideoWorkerService
from .store import JobStore


def build_remote():
    """Return an ImmersExec bridge when LTX_EXECUTOR=immers, else None (local)."""
    if os.getenv("LTX_EXECUTOR", "local").strip().casefold() != "immers":
        return None
    from .immers_exec import ImmersExec
    from .immers_power import ImmersPower

    return ImmersExec.from_env(ImmersPower.from_env())


def main() -> None:
    settings = WorkerSettings.from_env()
    settings.validate_server()
    settings.root.mkdir(parents=True, exist_ok=True)
    service = VideoWorkerService(
        settings,
        JobStore(settings.database_path),
        ComfyUiRunner(settings, remote=build_remote()),
    )
    server = create_server((settings.host, settings.port), settings, service)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
