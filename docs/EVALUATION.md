# Evaluation strategy

## What is evaluated today

The current system evaluates reliability at three levels.

### 1. Deterministic contract checks

LLM results are parsed into explicit scene, visual-bible, reference and image-prompt contracts. Validators check identifiers, required fields, scene coverage, language constraints, continuity references and downstream compatibility.

These checks answer: **Can this semantic output safely enter the next production stage?**

### 2. Workflow and provider regression tests

Provider-free tests exercise state transitions, approval gates, fingerprints, ambiguous submissions, polling, restart recovery, API authentication, remote command construction and cleanup behavior.

These checks answer: **Will the system preserve its safety and duplicate-spend invariants when dependencies fail?**

### 3. Human review

A human approves semantic artifacts and paid generation. This is intentional: style, narrative quality, brand fit and visual continuity are not reduced to a single unreliable scalar.

This check answers: **Is the result good enough and appropriate to spend money on?**

## Current evidence

- The public snapshot includes privacy, packaging, private-chat, and fail-closed authorization regressions.
- The documented local release gate runs compilation, the privacy scan, and the full suite before publication.
- An owner-reported private live smoke exercised one LTX end-to-end infrastructure path; it is not independently reproducible from this repository.

## What is not claimed

This repository does not currently include:

- model training or fine-tuning;
- a statistically representative content-quality benchmark;
- a labelled golden dataset for narrative or image quality;
- calibrated LLM-as-judge scoring;
- cross-model cost/quality/latency leaderboards;
- production SLO dashboards.

## Next evaluation milestone

A useful next step is an offline, versioned evaluation set containing synthetic briefs and expected invariants:

| Dimension | Example metric |
|---|---|
| Contract validity | valid outputs / total attempts |
| Scene coverage | referenced scenes / required scenes |
| Continuity | consistent character/reference IDs |
| Repair efficiency | successful repairs and extra model calls |
| Human acceptance | approve / revise / reject rate |
| Provider reliability | submit, poll and download success rate |
| Cost safety | duplicate paid submissions prevented |
| Latency | p50/p95 by stage and provider |

The dataset should contain no private conversations or production media. Human ratings should be separated from deterministic checks, and model/version/prompt hashes should be stored with every run.
