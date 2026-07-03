# Agentic Classification Guide

Cisco AI BOM uses a three-tier detection architecture. Tiers 1 and 2 generate candidates deterministically. Tier 3 (agentic classification) uses an LLM agent as the mandatory final classifier — every candidate must be confirmed or rejected by the agent.

The `--llm-model` option (or `AIBOM_LLM_MODEL` env var) is **required**.

If the required runtime extras are missing, `cisco-aibom analyze` fails fast with the exact `uv tool install ...` command to add the missing agentic or provider integration packages.

## Three-Tier Detection

| Tier | Method | Purpose | Examples |
|------|--------|---------|---------|
| **1 — Candidate Generation** | Import analysis, manifest parsing, file-type detection, known SDK patterns | Discover potential AI components | `from openai import OpenAI`, `langchain==0.3.1` in requirements.txt, `.safetensors` model files |
| **2 — Cross-Reference** | Multi-file env-var resolution, package index correlation | Enrich candidates with cross-file context | `os.getenv("MODEL_NAME")` resolved via `.env` or `docker-compose.yaml` |
| **3 — Agentic Classification** | LLM reasoning over code context, package metadata, import chains | Confirm or reject every candidate | Is this `Server()` an MCP server or a web server? Is this endpoint an AI inference URL or unrelated? |

## Prerequisites

Install the agentic extra plus the integration package for your LLM provider:

```bash
# OpenAI / Azure OpenAI
uv tool install --python 3.13 "cisco-aibom[agentic,llm-openai]"

# AWS Bedrock
uv tool install --python 3.13 "cisco-aibom[agentic,llm-aws]"

# Anthropic Claude (direct API)
uv tool install --python 3.13 "cisco-aibom[agentic,llm-anthropic]"

# Google Gemini
uv tool install --python 3.13 "cisco-aibom[agentic,llm-google]"

# Ollama (no extra provider package needed)
uv tool install --python 3.13 "cisco-aibom[agentic]"

# All providers at once
uv tool install --python 3.13 "cisco-aibom[all]"
```

## Supported LLM Providers

### OpenAI

```bash
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-5.4 \
  --llm-provider openai \
  --llm-api-key $OPENAI_API_KEY
```

### Azure OpenAI

```bash
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-5.4 \
  --llm-provider azure_openai \
  --llm-api-base https://my-endpoint.openai.azure.com \
  --llm-api-key $AZURE_OPENAI_API_KEY \
  --llm-api-version 2024-12-01-preview
```

`--llm-api-version` is required for Azure OpenAI.

### AWS Bedrock

```bash
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model us.anthropic.claude-sonnet-4-20250514-v1:0 \
  --llm-provider bedrock
```

Bedrock uses your configured AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`). No `--llm-api-key` is needed.

### Anthropic (direct)

```bash
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model claude-sonnet-4-20250514 \
  --llm-provider anthropic \
  --llm-api-key $ANTHROPIC_API_KEY
```

### Google (Gemini)

```bash
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gemini-2.0-flash \
  --llm-provider google_genai \
  --llm-api-key $GOOGLE_API_KEY
```

### Ollama (local)

```bash
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gemma3:12b \
  --llm-provider ollama \
  --llm-api-base http://localhost:11434
```

No API key is needed for Ollama.

## Configuration via `.env` File

Instead of passing LLM options on the command line, you can create a `.env` file in your project directory:

```bash
# .env
AIBOM_LLM_MODEL=gpt-5.4
AIBOM_LLM_PROVIDER=azure_openai
AIBOM_LLM_API_BASE=https://my-endpoint.openai.azure.com
AIBOM_LLM_API_KEY=sk-...
AIBOM_LLM_API_VERSION=2024-12-01-preview
```

The CLI auto-loads `.env` from the current working directory. To use a different file, set `AIBOM_ENV_FILE`:

```bash
AIBOM_ENV_FILE=/path/to/my-config.env cisco-aibom analyze ./my-app -o json -O report.json
```

Environment variables set in the shell take precedence over `.env` values.

## Tuning Options

| Option | Default | Description |
|--------|---------|-------------|
| `--agentic-batch-size` | `5` | Max components grouped into a single LLM invocation. Larger batches reduce API calls but may hit token limits. |
| `--agentic-concurrency` | `1` | Max parallel LLM batches. Increase for faster scans if your provider allows concurrent requests. |
| `--agentic-timeout` | `120` | Wall-clock timeout in seconds per batch. Batches exceeding this are marked as `batch_timeout` and skipped. |
| `--agentic-fast-model` | — | A cheaper/faster model for simple confirmations (e.g. registry lookups, dependency checks). The primary `--llm-model` is used for complex reasoning. |
| `--agentic-max-consecutive-failures` | `3` | Circuit-breaker threshold: skip the rest of a tier after this many consecutive batch failures. Raise it to push through a flaky endpoint (env `AIBOM_AGENTIC_MAX_FAILURES`). |
| `--agentic-max-retry-seconds` | `1200` | Aggregate wall-clock budget for all retry activity in a run. Bounds a persistently-failing model so the scan finishes with degraded components instead of retrying for hours; `0` disables the retry pass (env `AIBOM_AGENTIC_MAX_RETRY_SECONDS`). |
| `--llm-reasoning` | `auto` | `auto` / `off` / `on`. `off` disables model "thinking" using the correct per-provider parameter (env `AIBOM_LLM_REASONING`). See [Self-hosted & reasoning-model tuning](#self-hosted--reasoning-model-tuning). |
| `--llm-init-kwargs` | — | JSON object of provider-specific init kwargs merged verbatim into the model constructor — an escape hatch for advanced tuning (env `AIBOM_LLM_INIT_KWARGS`). |
| `--progress` | `auto` | Show live per-stage and per-scanner progress in interactive terminals. |
| `--include-code-snippets` | `off` | Include raw code snippets inside per-finding decision annotations. |

### Batch sizing guidance

- **Small repos (< 50 components):** Default settings work well.
- **Medium repos (50–200 components):** Consider `--agentic-concurrency 2` and `--agentic-batch-size 20`.
- **Large repos (200+ components):** Use `--agentic-concurrency 4` and `--agentic-fast-model` for a two-tier approach.

### Self-hosted & reasoning-model tuning

The defaults (`--agentic-concurrency 1`, `--agentic-timeout 120`) are tuned for
fast hosted APIs. A slow self-hosted endpoint or a verbose reasoning ("thinking")
model needs different settings — otherwise batches silently time out, the circuit
breaker trips, and the run produces **0 agentic enrichment**, which is
indistinguishable from "the model found nothing to add."

- **Disable/limit thinking first.** Reasoning verbosity is the number-one cause
  of batch timeouts: a model that emits pages of reasoning per batch blows past
  `--agentic-timeout`. Use `--llm-reasoning off`; it emits the correct parameter
  for each provider: the `chat_template_kwargs` flag for self-hosted
  OpenAI-compatible endpoints (vLLM — detected by a custom `--llm-api-base`),
  `reasoning_effort` for native OpenAI/Azure reasoners, Anthropic/Bedrock
  `thinking` disable, and Gemini `thinking_budget=0`. Non-reasoning native
  OpenAI/Azure models have no thinking to toggle, so the flag is a no-op there.
  For anything the flag doesn't cover, drop to `--llm-init-kwargs '<json>'`.
- **Raise `--agentic-concurrency`** (1–8) when the endpoint has spare capacity
  (e.g. a multi-GPU self-host). The default `1` is sequential/conservative.
- **Raise `--agentic-timeout`** for verbose models — they emit many tokens and
  tool calls per batch. The symptom of "too low" is repeated `Batch N timed out`
  log lines, the circuit breaker tripping, and an `Agentic enrichment DEGRADED`
  summary.
- **Bound worst-case runtime** with `--agentic-max-retry-seconds` and tune the
  breaker via `--agentic-max-consecutive-failures` (see the table above) so a
  persistently-failing endpoint degrades those components and the scan still
  finishes.

**Worked example** — an open reasoning model served on vLLM (e.g.
`zai-org/GLM-5.2-FP8`) needed thinking disabled **plus**
`--agentic-concurrency 8` **plus** `--agentic-timeout 300` to complete at all;
with thinking left on, every batch exceeded the 120 s default and the run fell
back to the deterministic-only floor:

```bash
cisco-aibom analyze ./my-repo \
  --output-file report.json \
  --llm-provider openai \
  --llm-api-base http://localhost:8000/v1 \
  --llm-model zai-org/GLM-5.2-FP8 \
  --llm-reasoning off \
  --agentic-concurrency 8 \
  --agentic-timeout 300
```

## How Agentic Classification Works

1. **Candidate triage** — After deterministic scanning, all candidates are split into "simple" (registry-confirmable, e.g. known model IDs, manifest dependencies) and "complex" (needs deeper reasoning) tiers.
2. **Locality-aware batching** — Candidates are grouped by directory to provide better code context per batch.
3. **Agent tools** — The agent has access to tools: `read_file_snippet` (inspect source), `search_codebase` (regex-style search across the scanned tree), `trace_data_flow` (follow a symbol's assignments and call sites), `search_package_info` (queries PyPI/npm/Go registries), `analyze_imports`, `lookup_model`, and `resolve_env_var`. It uses these to confirm or reject each candidate.
4. **Structured output** — Each batch returns a structured JSON response with confirmed components, removed false positives, reclassifications, relationships, and risk findings. Kept findings carry `decision_annotation` metadata with justification and evidence references. Raw code snippets are only included when `--include-code-snippets` is enabled.
5. **Caching** — Results are cached by content hash at `~/.aibom/cache/agentic/`. Unchanged components reuse cached results on subsequent runs.

## Circuit Breaker

To protect against runaway API costs, a circuit breaker trips after 3 consecutive batch failures (configurable). When tripped:

- Remaining batches are skipped.
- Skipped components are marked with `agentic_hint: circuit_breaker_tripped`.
- The deterministic detection results are preserved.

## Cache Management

Agentic results are cached on disk at `~/.aibom/cache/agentic/`. To inspect or clear them:

```bash
# Clear both scan cache and agentic cache
cisco-aibom cache clear

# Clear scan cache only (skip agentic)
cisco-aibom cache clear --no-agentic

# List cached agentic entries
cisco-aibom cache list --type agentic

# Inspect one cached agentic entry
cisco-aibom cache get agentic 0123456789ab
```

## Strict Mode

Use `--strict` to suppress all components that would need agentic reasoning. Only exact-match model IDs from the curated known-models list are emitted:

```bash
cisco-aibom analyze ./my-app -o json -O report.json --llm-model gpt-5.4 --strict
```

This is useful for CI/CD pipelines where you want the fastest possible scans with minimal LLM calls.

## Repository Triage (Org-Scale Scanning)

When scanning many repos (`--github-org`, `--discover-repos`), the triage agent explores each repository before deciding whether to deep-scan it:

1. **Directory tree** — Lists the repo structure looking for AI-signal directories.
2. **README / manifests** — Reads key files to understand the project purpose and dependencies.
3. **Package registry lookup** — Queries PyPI/npm/Go for unfamiliar packages via `search_package_info`.
4. **Source sampling** — Greps for AI imports if still uncertain.

The agent biases toward `deep-scan` — skipping an AI repo is far worse than scanning a non-AI repo. No hardcoded package list gates the triage decision.
