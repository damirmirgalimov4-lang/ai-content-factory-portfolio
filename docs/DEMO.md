# Provider-free demo walkthrough

This walkthrough demonstrates the engineering behavior without Telegram credentials, model access, a GPU, or paid provider calls.

## 1. Run the safety gate

```bash
python3 scripts/security_scan.py
```

Expected result:

```text
SECURITY_SCAN_OK files=<count>
```

The scan covers tracked text files (including extensionless release files), secret signatures, private paths, IP addresses, email addresses, UUIDs, and forbidden private directories. Narrow exceptions exist only for reviewed RFC-reserved fixtures and ComfyUI workflow node identifiers.

## 2. Run the deterministic suite

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

The suite uses temporary directories, fake provider transports, and loopback HTTP servers. It does not contact paid services.

Representative coverage:

- content stage transitions and approval;
- structured scene and image-prompt contracts;
- storyboard planning and revision;
- reference continuity and frame generation;
- image QA and post-image QA;
- provider request/response normalization;
- paid-job approval, fingerprinting, deduplication, polling and delivery;
- LTX bearer authentication, SQLite jobs, restart recovery and MP4 checks;
- OpenStack VM state transitions and money-safe shelving retries;
- SSH/scp argument construction and detached ComfyUI startup;
- isolated partner workspace and shared queue;
- security and privacy regressions.

## 3. Inspect the main orchestration path

Recommended review order:

1. `agent_platform/content_factory.py` — stage state machine.
2. `agent_platform/production.py` — production contracts and artifacts.
3. `agent_platform/llm.py` — LLM protocol and Codex boundary.
4. `agent_platform/video_provider.py` — provider interface.
5. `agent_platform/video_jobs.py` — approval, duplicate-spend guards, and reconciliation.
6. `agent_platform/ltx.py` — authenticated LTX client.
7. `ltx_worker/service.py` and `ltx_worker/store.py` — durable worker jobs.
8. `ltx_worker/immers_exec.py` — remote execution and cleanup.

## 4. Optional local Telegram run

Create local configuration:

```bash
cp .env.example .env
```

Fill only your own BotFather token and Telegram user ID. The main bot refuses to start with an empty allowlist.

```bash
python3 -m agent_platform
```

Model/image/video operations remain unavailable until their corresponding clients are configured. Keep all paid feature flags off when exploring the UI.

## Private infrastructure validation

An owner-reported private smoke test previously exercised the full self-hosted path:

```text
Telegram orchestration
  → authenticated worker job
  → on-demand remote GPU
  → ComfyUI + LTX-2.3
  → validated MP4
  → result delivery
  → VM shelving
```

The public repository intentionally omits the source image, prompt payload, credentials, IP address, VM ID, job ID, Telegram identifiers and logs, so the private smoke is not independently reproducible from this repository.

This smoke is evidence of integration work, not a public hosted demo or a claim of zero remaining rollout risk.
