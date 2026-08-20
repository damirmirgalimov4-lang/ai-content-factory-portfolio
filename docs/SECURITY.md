# Security model

## Public snapshot policy

This repository is rebuilt from an allowlisted source snapshot with a new Git history. It intentionally excludes:

- `.env` and production service configuration;
- API keys, passwords, OAuth data and Telegram tokens;
- numeric Telegram user/chat identifiers;
- personal conversations and vault contents;
- imported message archives;
- generated media, payloads and logs;
- production hostnames, IP addresses, VM/server identifiers and SSH keys;
- internal progress reports and agent-training notes.

Synthetic roles (`owner`, `partner`) and RFC-reserved example addresses replace private labels and infrastructure.

## Runtime controls

- The main Telegram bot requires an explicit allowlist, accepts only a one-to-one private chat whose chat ID matches the sender ID, and fails closed.
- The optional partner assistant keeps a separate runtime and uses one-time pairing.
- Paid video submission requires persisted human approval.
- Job fingerprints and provider task IDs prevent blind duplicate submissions.
- Ambiguous provider responses require reconciliation.
- Codex subprocesses receive a reduced environment. Chat and inspection use a
  read-only sandbox; image generation and repair use narrowly scoped writable directories.
- LTX endpoints require bearer authentication.
- LTX inference is default-off and accepts only validated loopback ComfyUI origins.
- Remote commands use argument arrays and quoted paths rather than interpolated shell input.
- GPU shelving is attempted after both success and failure.

## Repository gate

```bash
python3 scripts/security_scan.py
```

The scanner is deliberately conservative and fails the release gate on high-confidence credentials, private paths, personal labels, non-example IP addresses, unapproved email addresses, UUIDs outside the reviewed ComfyUI workflow, and forbidden private directories.

This scanner is defense in depth, not a substitute for provider-side secret rotation or GitHub secret scanning.

## Secret handling

1. Copy `.env.example` to `.env` locally.
2. Store real values only outside Git.
3. Use separate credentials per service and environment.
4. Rotate any credential immediately if it appears in terminal output, a commit, an issue, or a shared artifact.
5. Never put production values into tests; construct synthetic fixtures from fragments when testing redaction.
6. Keep paid providers and `LTX_INFERENCE_ENABLED` off by default.

## Threat boundaries and known limits

- File-backed main application state assumes one active process.
- The public snapshot does not provide a production network perimeter or secret manager.
- Automatic cross-provider LLM failover is not present.
- LTX rollout still needs stronger approval binding and broader ambiguous-result reconciliation before general enablement.

See [LIMITATIONS.md](LIMITATIONS.md) for the complete list.
