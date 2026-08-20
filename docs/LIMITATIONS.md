# Current limitations and roadmap

This document separates implemented behavior from future work.

## Current limitations

### Single-process application state

The content workflow uses inspectable atomic files. This is appropriate for the current single-user deployment, but multiple active controllers would need shared transactional state and durable Telegram update acknowledgement.

### Telegram controller size

The main controller contains transport, routing and orchestration glue in one large module. Domain stores and provider clients are already separate; command handlers should be extracted incrementally rather than through a high-risk rewrite.

### Codex-only runtime construction

An `LlmClient` protocol exists, but current runtime factories instantiate Codex CLI clients. The partner and research paths also need to use a unified factory before automatic model failover can cover the whole system.

### LTX rollout hardening

The LTX happy path and remote GPU lifecycle have been validated, but the feature remains default-off. Before broad production enablement:

- bind approval to immutable source/profile hashes;
- improve reconciliation after restart or ambiguous remote completion;
- surface cost estimates and actual billing where the provider exposes them;
- keep a separately monitored money-stop alert when shelving itself fails;
- replace first-connection SSH trust-on-first-use with operator-provisioned `known_hosts` pinning;
- maintain an audited systemd/runtime deployment contract.

### Quality evaluation

Current quality controls combine structural contracts, deterministic validation and human approval. A versioned offline evaluation dataset and model/prompt comparison dashboard are not yet included.

### No model training

The repository integrates and orchestrates existing multimodal models. It does not claim training, fine-tuning, feature engineering or ML research.

## Roadmap

1. Add a unified LLM factory with typed errors, cooldowns and provider fallback.
2. Add an offline synthetic eval set with prompt/model/version provenance.
3. Move Telegram update acknowledgement and main state to a transactional queue/database for multi-instance operation.
4. Extract Telegram use-case handlers behind application services.
5. Add provider latency, cost and reconciliation metrics.
6. Harden LTX approval binding and publish an infrastructure-neutral deployment contract.

Roadmap items are not described as completed features elsewhere in the repository.
