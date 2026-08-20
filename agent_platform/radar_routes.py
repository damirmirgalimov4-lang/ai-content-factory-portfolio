from __future__ import annotations


RADAR_COMMANDS = {"/discover", "/research", "/results", "/radar"}

RADAR_CALLBACKS = {
    "research:home",
    "research:auto",
    "research:accounts",
    "research:import",
    "research:youtube",
    "research:instagram",
    "research:results",
    "scripts:list",
    "auto:youtube",
    "auto:instagram",
}

RADAR_CALLBACK_PREFIXES = (
    "research_confirm:",
    "research_content_retry:",
    "research_cancel:",
    "research_run_results:",
    "research_run:",
    "result_script:",
    "result_handoff:",
    "result:",
    "radar_shared_item:",
)

RADAR_INPUT_KINDS = {
    "research_account_import",
    "research_query",
}


def is_radar_callback(data: str) -> bool:
    return data in RADAR_CALLBACKS or data.startswith(RADAR_CALLBACK_PREFIXES)
