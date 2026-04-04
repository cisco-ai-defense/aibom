# Agentic Enrichment Guide

Cisco AI BOM uses a three-tier detection architecture. Tiers 1 and 2 are deterministic and run by default. Tier 3 (agentic enrichment) uses LLM-powered agents to resolve ambiguous detections and is activated by supplying `--llm-model`.

## Three-Tier Detection

| Tier | Method | LLM Required | Examples |
|------|--------|:------------:|---------|
| **1 — Deterministic** | Import analysis, manifest parsing, file-type detection, known SDK patterns | No | `from openai import OpenAI`, `langchain==0.3.1` in requirements.txt, `.safetensors` model files |
| **2 — Cross-Reference** | Multi-file env-var resolution, package index correlation | No | `os.getenv("MODEL_NAME")` resolved via `.env` or `docker-compose.yaml` |
| **3 — Agentic** | LLM reasoning over code context, import chains, and file structure | Yes | Is this `Server()` an MCP server or a web server? Is this `.fit()` call ML training or preprocessing? |

When running without `--llm-model`, the CLI reports how many components were flagged as needing agentic reasoning:

```
Scan complete: 42 components (high confidence)
17 candidates would benefit from agentic reasoning
Run with --llm-model to resolve ambiguous detections
```

## Prerequisites

Install the agentic extra plus the integration package for your LLM provider:

```bash
# OpenAI / Azure OpenAI
uv tool install "cisco-aibom[agentic,llm-openai]"

# AWS Bedrock
uv tool install "cisco-aibom[agentic,llm-aws]"

# Ollama (no extra provider package needed)
uv tool install "cisco-aibom[agentic]"
```

## Supported LLM Providers

### OpenAI

```bash
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-4o \
  --llm-provider openai \
  --llm-api-key $OPENAI_API_KEY
```

### Azure OpenAI

```bash
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-4o \
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
AIBOM_LLM_MODEL=gpt-4o
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
| `--agentic-scope` | `candidates` | Which components to send to the LLM. `candidates` sends only items flagged as `needs_agentic=True` during deterministic scanning. `all` sends every detected component. |
| `--agentic-batch-size` | `5` | Max components grouped into a single LLM invocation. Larger batches reduce API calls but may hit token limits. |
| `--agentic-concurrency` | `1` | Max parallel LLM batches. Increase for faster scans if your provider allows concurrent requests. |
| `--agentic-timeout` | `120` | Wall-clock timeout in seconds per batch. Batches exceeding this are marked as `batch_timeout` and skipped. |
| `--agentic-fast-model` | — | A cheaper/faster model for Tier 1 simple confirmations (e.g. registry lookups). The primary `--llm-model` is used for complex reasoning. |

### Batch sizing guidance

- **Small repos (< 50 components):** Default settings work well.
- **Medium repos (50–200 components):** Consider `--agentic-concurrency 2` and `--agentic-batch-size 10`.
- **Large repos (200+ components):** Use `--agentic-scope candidates` (default), `--agentic-concurrency 4`, and consider `--agentic-fast-model` for a two-tier approach.

## How Agentic Enrichment Works

1. **Classification** — After deterministic scanning, components are split into "simple" (registry-confirmable) and "complex" (needs reasoning) candidates.
2. **Locality-aware batching** — Components are grouped by directory to provide better code context per batch.
3. **Sub-agent dispatch** — For multi-directory scans, independent agents can be dispatched per directory group.
4. **Structured output** — Each batch returns a structured JSON response with enriched components, new components, removed false positives, and risk findings.
5. **Caching** — Results are cached by content hash at `~/.cache/cisco-aibom/agentic/`. Unchanged components reuse cached results on subsequent runs.

## Circuit Breaker

To protect against runaway API costs, a circuit breaker trips after 3 consecutive batch failures (configurable). When tripped:

- Remaining batches are skipped.
- Skipped components are marked with `agentic_hint: circuit_breaker_tripped`.
- The deterministic detection results are preserved.

## Cache Management

Agentic results are cached on disk at `~/.cache/cisco-aibom/agentic/`. To clear:

```bash
# Clear both scan cache and agentic cache
cisco-aibom cache clear

# Clear scan cache only (skip agentic)
cisco-aibom cache clear --no-agentic
```

## Strict Mode

Use `--strict` to suppress all components that would need agentic reasoning. Only high-confidence deterministic detections are emitted:

```bash
cisco-aibom analyze ./my-app -o json -O report.json --strict
```

This is useful for CI/CD pipelines where you want deterministic, fast scans without LLM dependencies.
