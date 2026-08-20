# AI-assisted development methodology

AI Content Factory was developed with coding agents as implementation collaborators. That fact is disclosed rather than hidden because operating coding agents safely is part of the engineering work demonstrated here.

## Human-owned responsibilities

- product scope and workflow requirements;
- architecture and trust-boundary decisions;
- approval and cost-control policy;
- acceptance criteria for every stage;
- review of generated patches;
- test and regression requirements;
- live infrastructure verification;
- incident diagnosis and the decision to stop paid resources;
- final responsibility for published code.

## Agent-assisted responsibilities

- code drafting and mechanical refactoring;
- test scaffolding;
- documentation drafts;
- static inspection and independent review;
- repetitive provider adapter work.

## Verification policy

A coding-agent result is not treated as evidence by itself. A change must be checked through deterministic tests, static/security gates, source review and—only after explicit approval—live provider validation. Production credentials are never placed in prompts, commits or public artifacts.

Failures become regression tests or explicit system rules. Paid operations remain human-gated, and ambiguous results are reconciled rather than retried automatically.

## What a reviewer can assess

The public code is intended to support a technical discussion of:

- structured LLM outputs and deterministic validation;
- state-machine design for human-in-the-loop workflows;
- provider abstraction and error normalization;
- duplicate-spend guards and manual reconciliation for paid external calls;
- multimodal continuity across scenes;
- restart-safe inference jobs;
- remote GPU lifecycle and cost safety;
- trade-offs between stdlib simplicity and horizontal scalability.
