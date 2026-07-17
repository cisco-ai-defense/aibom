# AI BOM

[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?logo=discord&logoColor=white)](https://discord.com/invite/nKWtDcXxtx)
[![Cisco AI Defense](https://img.shields.io/badge/Cisco-AI%20Defense-049fd9?logo=cisco&logoColor=white)](https://www.cisco.com/site/us/en/products/security/ai-defense/index.html)
[![AI Security and Safety Framework](https://img.shields.io/badge/AI%20Security-Framework-orange)](https://learn-cloudsecurity.cisco.com/ai-security-framework)

Cisco AI BOM scans codebases, container images, and cloud environments to produce an AI Bill of Materials — a structured inventory of models, agents, tools, MCP servers/clients, datasets, prompts, guardrails, secrets, and other AI assets used in your software. It supports Python, JavaScript/TypeScript, Java, Go, Rust, Ruby, and C#, with deterministic candidate detection, cross-reference resolution, and LLM-powered agentic classification.

## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
- [Agentic Enrichment](#agentic-enrichment)
- [Galileo Observability](#galileo-observability)
- [Container Scanning](#container-scanning)
- [Cross-Repo and Org Scanning](#cross-repo-and-org-scanning)
- [Output Formats](#output-formats)
- [Custom Catalog](#custom-catalog)
- [Policy Engine](#policy-engine)
- [Knowledge Base](#knowledge-base)
- [Environment Variables](#environment-variables)
- [Docker](#docker)
- [Testing](#testing)
- [Further Reading](#further-reading)
- [Troubleshooting](#troubleshooting)

## Features

- **Multi-language analysis** — Python (LibCST), JavaScript/TypeScript, Java, Go, Rust, Ruby, C# (tree-sitter).
- **23 built-in scanners** — model detection, dependency analysis, secret detection, vulnerability scanning (OSV.dev), MCP server/client detection, A2A/remote agent resolution, structural agent detection, ML lifecycle detection, cloud resource scanning, CI/CD pipeline analysis, deployment detection, container scanning, data-file scanning, environment variable resolution, KB enrichment, and more.
- **30 AI component types** — `model`, `llm_endpoint`, `model_endpoint`, `agent`, `agent_proxy`, `tool`, `mcp_server`, `mcp_client`, `mcp_gateway`, `embedding`, `vector_store`, `dataset`, `retriever`, `knowledge_base`, `feature_store`, `memory`, `prompt`, `training_run`, `hyperparameter`, `model_artifact`, `experiment_tracker`, `model_registry`, `data_versioning`, `ml_pipeline`, `guardrail`, `skill`, `observability`, `secret`, `dependency`, `other`.
- **Three-tier detection** — Tier 1 (deterministic high-confidence), Tier 2 (cross-reference resolution), Tier 3 (agentic LLM reasoning). Tier 1 code-level detection is deepest for Python (LibCST); other languages extract imports and literal patterns via tree-sitter and lean more on Tier 3 for confirmation.
- **10 output formats** — Plaintext, JSON, CycloneDX, SARIF, SPDX, HTML dashboard, Markdown, CSV, JUnit, and a live API server.
- **Container image scanning** — Extract and analyze application source code from Docker, Podman, nerdctl, Buildah, Skopeo, or Crane images, with Anchore Syft for SBOM metadata.
- **Cross-repo and org-level scanning** — Scan multiple local repos, GitHub orgs, GitLab groups, or Bitbucket projects, with incremental caching.
- **Agentic classification** — LLM agent (via Deep Agents + LangChain) classifies every scanner candidate, eliminating false positives and enriching confirmed components with concrete identifiers.
- **Policy engine** — YAML-driven pass/fail gates for CI/CD integration (max-risk, required fields, blocked/required component types).
- **Compliance checks** — EU AI Act, OWASP Agentic Top 10, NIST AI RMF advisory mappings.
- **Watch mode** — Real-time file-system monitoring with debounced re-scan and delta reporting.
- **Diff command** — Compare two AIBOM JSON snapshots side-by-side.
- **Benchmark command** — Measure precision/recall/F1 against a labelled ground-truth file.
- **Secret detection** — Integrated Yelp `detect-secrets` for hardcoded API keys, tokens, and credentials.
- **Vulnerability scanning** — OSV.dev API lookups for known CVEs in detected dependencies.
- **Plugin system** — Extend with custom scanners and reporters via Python entry points.
- **Custom catalog** — Register custom AI components, base-class rules, excludes, and relationships via `.aibom.yaml`.
- **Knowledge base** — Curated DuckDB catalog of AI framework symbols with download, verification, and versioned updates.

### Language coverage at a glance

| Language | Dep manifests | Code-level detection | Env-var resolver | Structural / KB / MCP / A2A scanners |
|---|---|---|---|---|
| Python | pip, Poetry, uv, setuptools | LibCST (full) | yes | yes |
| JavaScript / TypeScript | npm, yarn, pnpm | tree-sitter (imports + literals) | yes | no |
| Java | Maven, Gradle | tree-sitter (imports + literals) | yes | no |
| Go | go.mod | tree-sitter (imports + literals) | yes | no |
| Rust | Cargo | tree-sitter (imports + literals) | no | no |
| Ruby | Gemfile | tree-sitter (imports + literals) | yes | no |
| C# | `*.csproj` | tree-sitter (imports + literals) | no | no |

Non-Python code-level detection is focused on dependency imports (matched against a curated allowlist) and inline `model: "..."` literals. Deeper structural signals — agent instantiations, MCP/A2A detection, ReAct-style loops, KB symbol enrichment — are Python-only today. For non-Python code, Tier 3 agentic classification fills most of the gap. See [docs/TECHNICAL_OVERVIEW.md §12](docs/TECHNICAL_OVERVIEW.md#12-language-coverage) for details.

## Repository Layout

```
aibom/   # Python analyzer package + CLI
docs/    # Documentation (CLI reference, guides, API docs)
```

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Docker / Podman (optional, for container image analysis)
- LLM provider credentials (required for `--llm-model`; see [Agentic Classification](#agentic-enrichment))

### Install from PyPI

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Analyze with OpenAI / Azure OpenAI
uv tool install --python 3.13 "cisco-aibom[agentic,llm-openai]"

# Analyze with AWS Bedrock
uv tool install --python 3.13 "cisco-aibom[agentic,llm-aws]"

# Core CLI only (report rendering, cache inspection, KB commands, etc.)
uv tool install --python 3.13 cisco-aibom

# Everything
uv tool install --python 3.13 "cisco-aibom[all]"

# Verify
cisco-aibom --help
```

`cisco-aibom analyze` always requires `--llm-model`. If the required agentic or provider extras are missing, the CLI fails fast with the exact `uv tool install ...` hint for the missing runtime.

### Install from source

```bash
uv tool install --python 3.13 --from git+https://github.com/cisco-ai-defense/aibom cisco-aibom
```

### Local development

```bash
git clone https://github.com/cisco-ai-defense/aibom.git
cd aibom/aibom

uv sync
source .venv/bin/activate

cisco-aibom --help
```

When working from source, you can also use `uv run cisco-aibom ...` or `uv run python -m aibom ...`.

### Extras

The `analyze` command always runs the agentic pipeline, so it requires the `agentic` extra **plus at least one LLM provider extra**. The simplest install is `cisco-aibom[all]`; otherwise pick the provider you plan to use.

**Required for `analyze`:**

| Extra | Installs | Purpose |
|-------|----------|---------|
| `agentic` | Deep Agents, LangChain | LLM-powered agentic enrichment (mandatory for `analyze`) |
| `llm-openai` | `langchain-openai` | OpenAI / Azure OpenAI provider |
| `llm-aws` | `langchain-aws` | AWS Bedrock provider |
| `llm-anthropic` | `langchain-anthropic` | Anthropic Claude provider |
| `llm-google` | `langchain-google-genai` | Google Gemini provider |

**Optional (degrade gracefully if missing):**

| Extra | Installs | Purpose |
|-------|----------|---------|
| `analysis` | `detect-secrets`, `tree-sitter` | Secret detection, multi-language parsing |
| `security` | `cisco-ai-mcp-scanner`, `cisco-ai-skill-scanner` | Cisco security tool integration |
| `observability` | `galileo` | Opt-in, sanitized agentic quality telemetry |
| `cloud` | `boto3`, `google-cloud-aiplatform`, `azure-*` | Cloud resource scanning |
| `all` | All of the above | Full feature set |

## Quick Start

The fastest path from install to seeing assets in AI Inventory:

```bash
# 1. Install with the agentic pipeline + your LLM provider (OpenAI shown)
uv tool install --python 3.13 "cisco-aibom[agentic,llm-openai]"

# 2. Scan a git repo or container image (--llm-model is always required)
cisco-aibom analyze /path/to/repo \
  --output-format json --output-file report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# 3. Upload to AI Defense — assets appear in AI Inventory
cisco-aibom report upload report.json --format json \
  --post-url "$AIBOM_POST_URL" --ai-defense-api-key "$AI_DEFENSE_API_KEY"
```

> **What appears in AI Inventory?** A scan of a **git working tree** or a
> **container image** is automatically attributed (repo URL + commit SHA, or
> image ref + digest) and projected as an inventory asset — no extra flags
> needed. A scan of a plain local directory that is *not* a git checkout still
> uploads successfully, but has no stable identity to project, so it will
> **not** appear in AI Inventory. To populate the inventory, scan a git repo or
> a container image.

More examples:

```bash
# Scan a local project (--llm-model is required)
cisco-aibom analyze /path/to/project -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# Scan a container image
cisco-aibom analyze my-app:latest -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# Scan multiple repos under a directory
cisco-aibom analyze /path/to/repos --discover-repos -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# HTML dashboard
cisco-aibom analyze /path/to/project -o html -O dashboard.html \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# Policy gate for CI
cisco-aibom analyze /path/to/project -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY --policy policy.yaml
```

All LLM options can be set via environment variables (`AIBOM_LLM_MODEL`, `AIBOM_LLM_API_KEY`, etc.) for cleaner commands.

## Commands

| Command | Description |
|---------|-------------|
| `analyze` | Scan source code, container images, or repos and produce an AI BOM. |
| `report` | Render or upload a previously generated JSON report. |
| `watch` | Poll directories for changes and re-scan with delta reporting (deterministic pipeline only). |
| `diff run` | Compare two AIBOM JSON reports side-by-side. |
| `benchmark run` | Measure precision/recall/F1 against ground-truth YAML. |
| `kb download` | Download the latest knowledge base. |
| `kb check` | Check if a newer KB version is available. |
| `kb info` | Display info about the locally installed KB. |
| `kb verify` | Verify KB integrity (SHA-256 checksum). |
| `kb request` | Request a KB build for a specific SDK version. |
| `kb request-status` | Check the status of a KB build request. |
| `kb list-requests` | List all pending KB build requests. |
| `cache clear` | Remove cached scan results and agentic cache. |
| `cache list` | List cached entries by cache type. |
| `cache get` | Inspect a specific cache entry. |
| `plugin list` | List discovered plugins (entry points, MCP servers). |

See [docs/CLI_REFERENCE.md](https://github.com/cisco-ai-defense/aibom/blob/main/docs/CLI_REFERENCE.md) for complete option details.

### Global options

| Option | Env Var | Description |
|--------|---------|-------------|
| `--log-level` | `AIBOM_LOG_LEVEL` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default `INFO`). |

## Agentic Enrichment

The `--llm-model` option (or `AIBOM_LLM_MODEL` env var) is required. The LLM agent acts as the final classifier for every scanner candidate and requires the `agentic` extra plus any provider-specific integration extra (for example `llm-openai` or `llm-aws`):

- Confirms or removes every scanner candidate (no unverified findings)
- Classifies and enriches components with concrete identifiers
- Verifies dependencies against package registries (PyPI, npm, Go)
- Discovers components missed by static analysis

```bash
# OpenAI
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-5.4 --llm-provider openai --llm-api-key $OPENAI_API_KEY

# Azure OpenAI
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-5.4 --llm-provider azure_openai \
  --llm-api-base https://my-endpoint.openai.azure.com \
  --llm-api-key $AZURE_OPENAI_API_KEY --llm-api-version 2024-12-01-preview

# AWS Bedrock
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model us.anthropic.claude-sonnet-4-20250514-v1:0 --llm-provider bedrock

# Local Ollama
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gemma3:12b --llm-provider ollama \
  --llm-api-base http://localhost:11434
```

All LLM options can also be set via environment variables or a `.env` file. See [docs/AGENTIC_MODE.md](https://github.com/cisco-ai-defense/aibom/blob/main/docs/AGENTIC_MODE.md) for the full guide.

### Agentic tuning

| Option | Default | Description |
|--------|---------|-------------|
| `--agentic-batch-size` | `5` | Max components per LLM invocation. |
| `--agentic-concurrency` | `1` | Max parallel LLM batches. |
| `--agentic-timeout` | `120` | Wall-clock seconds per batch before timeout. |
| `--agentic-fast-model` | — | Cheaper model for simple confirmations (model lookups, dependency checks). |
| `--agentic-max-consecutive-failures` | `3` | Circuit-breaker threshold before skipping the rest of a tier. |
| `--agentic-max-retry-seconds` | `1200` | Aggregate wall-clock budget for retries; bounds a failing model so the scan finishes (`0` disables retries). |
| `--llm-reasoning` | `auto` | `off` disables model "thinking" per provider — use for verbose reasoning/self-hosted models that otherwise time out. |
| `--llm-init-kwargs` | — | JSON of provider-specific init kwargs (advanced escape hatch). |
| `--progress` | `auto` | Show live per-stage and per-scanner progress in interactive terminals. |
| `--include-code-snippets` | `off` | Include raw code snippets inside per-finding decision annotations. |

Running a slow self-hosted or verbose reasoning model? See [Self-hosted & reasoning-model tuning](https://github.com/cisco-ai-defense/aibom/blob/main/docs/AGENTIC_MODE.md#self-hosted--reasoning-model-tuning) — disabling "thinking" and raising the timeout/concurrency is usually the difference between real enrichment and a silent deterministic-only fallback.

### Report and cache utilities

```bash
# Render a saved report
cisco-aibom report report.json

# Explicit show form
cisco-aibom report show report.json --raw-json

# Upload an existing JSON report
cisco-aibom report upload report.json --format json \
  --post-url https://example.invalid/aibom/reports \
  --ai-defense-api-key $AI_DEFENSE_API_KEY

# Inspect cache families under the shared cache root
cisco-aibom cache list --type scan
cisco-aibom cache list --type agentic
cisco-aibom cache get scan 0123456789ab
```

All cache families now default under `~/.aibom/cache`, including deterministic scan cache, agentic cache, org cache, model cache, and package metadata cache.

## Galileo Observability

Agentic scans can optionally emit observe-only quality telemetry to Galileo.
The sanitized path is disabled by default and fail-open: it sends allowlisted
decision/guard counters, per-call LLM/tool ordering, status/timing/token data,
and HMAC pseudonyms, but never source code, paths, prompts, model response text,
component names, tool arguments, or tool results. A separate diagnostic
full-trajectory path can send that raw content when every content, identity,
trajectory, destination, and hosted-egress approval gate is enabled.

```bash
uv tool install --python 3.13 \
  "cisco-aibom[agentic,observability,llm-openai]"

export GALILEO_API_KEY="<secret-manager-value>"
export GALILEO_CONSOLE_URL="https://app.galileo.ai"
export GALILEO_API_URL="https://api.galileo.ai"
export GALILEO_PROJECT="<project-name>"
export GALILEO_LOG_STREAM="<log-stream-name>"
export AIBOM_GALILEO_HMAC_KEY="<independent-high-entropy-secret>"
export AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD=true

cisco-aibom analyze /path/to/project --llm-model gpt-5.4 \
  --galileo \
  --galileo-sample-rate 0.05 -o json -O report.json
```

Project provisioning, datasets, metrics, dashboards, alerts, RBAC, and
retention remain operator-managed; the named Galileo project and log stream
must already exist. This integration is hosted-only: telemetry accepts exactly
`https://app.galileo.ai` (and the derived `https://api.galileo.ai` API) and
requires `AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD=true`.

Raw telemetry is disabled by default: `--galileo` alone enables only the
sanitized path, even when raw-approval environment variables are already
present. A reviewed run must also pass `--galileo-full-trajectory`; the raw
callback activates only when the full-content, full-trajectory, exact-identity,
immutable project/log-stream, TLS, and hosted-egress gates all pass. Each live
agent invocation receives a fresh callback and logger, which prevents state from
crossing scans or concurrent batches while retaining one flush per completed or
failed chain. When active, it captures real prompts, responses, tool I/O,
callback metadata, exceptions, and exact repository/component/path identities
and sends them to hosted Galileo. Custom-function experiments remain
pseudonymous by default. See the
[Galileo observability guide](https://github.com/cisco-ai-defense/aibom/blob/main/aibom/docs/galileo-observability.md) for the exact
data contract, trace map, metrics, and rollout procedure.

## Container Scanning

The CLI auto-detects container image references and extracts application source code for analysis.

```bash
# Auto-detect extraction method
cisco-aibom analyze my-app:latest -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# Force a specific extraction tier
cisco-aibom analyze my-app:latest -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY \
  --container-extraction-tier podman
```

Supported tiers: `auto`, `syft`, `docker`, `podman`, `nerdctl`, `buildah`, `crane`, `skopeo`, `tarball`.

See [docs/CONTAINER_SCANNING.md](https://github.com/cisco-ai-defense/aibom/blob/main/docs/CONTAINER_SCANNING.md) for details.

## Cross-Repo and Org Scanning

```bash
# Discover and scan all git repos under a directory
cisco-aibom analyze /path/to/repos --discover-repos -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# Scan a GitHub org (requires GITHUB_TOKEN)
cisco-aibom analyze --github-org my-org --platform-token $GITHUB_TOKEN -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# Scan a GitLab group
cisco-aibom analyze --gitlab-group my-group --platform-token $GITLAB_TOKEN -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# Scan repos from a file (JSON array or newline-delimited)
cisco-aibom analyze --repos-file repos.txt -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# Incremental scan (skip repos with unchanged HEAD)
cisco-aibom analyze /path/to/repos --discover-repos --skip-unchanged -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# Limit and filter
cisco-aibom analyze --github-org my-org --platform-token $GITHUB_TOKEN \
  --max-repos 50 --repo-filter "ml-" --parallel-repos 4 -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY
```

## Output Formats

| Format | Flag | Description |
|--------|------|-------------|
| Plaintext | `-o plaintext` | Human-readable text report. |
| JSON | `-o json` | Structured JSON with full component details. |
| CycloneDX | `-o cyclonedx` | CycloneDX 1.6 BOM (ML-BOM profile). |
| SARIF | `-o sarif` | SARIF v2.1.0 for IDE/CI integration. |
| SPDX | `-o spdx` | SPDX 3.0 with AI and Dataset profiles. |
| HTML | `-o html` | Interactive dashboard with dependency graph and risk heatmap. |
| Markdown | `-o markdown` | Markdown table report. |
| CSV | `-o csv` | Flat CSV for spreadsheet analysis. |
| JUnit | `-o junit` | JUnit XML for CI test result reporting. |
| API | `-o api` | Live FastAPI server at `http://127.0.0.1:8000`. |

All file-based formats require `--output-file` / `-O`.

## Custom Catalog

The built-in DuckDB catalog covers popular AI frameworks, but you can extend it with a `.aibom.yaml` configuration file for custom components, base-class detection rules, exclude patterns, and relationship hints.

```yaml
# .aibom.yaml
components:
  - id: MyLLMWrapper
    concept: model
    label: My Custom LLM
    framework: internal

base_classes:
  - class: BaseTool
    concept: tool

excludes:
  - some_noisy_helper_function

relationship_hints:
  tool_arguments:
    - custom_tools
```

Place `.aibom.yaml` in your project root (auto-discovered) or pass `--custom-catalog /path/to/.aibom.yaml`.

Supported keys: `components`, `base_classes`, `excludes`, `relationship_hints`, `custom_relationships`. See the full reference in [aibom/examples/.aibom.yaml](https://github.com/cisco-ai-defense/aibom/blob/main/aibom/examples/.aibom.yaml).

### Inline annotations

Tag classes and functions directly in source code:

```python
# aibom: concept=guardrail framework=internal
class SafetyFilter:
    ...

class MyRouter:  # aibom: concept=router
    ...
```

## Policy Engine

Define pass/fail gates in a YAML policy file for CI/CD integration:

```yaml
# policy.yaml
max_risk_score: 70
required_fields:
  - model_name
blocked_types:
  - secret
required_types:
  - guardrail
rules:
  - name: no-hardcoded-keys
    field: metadata.secret_type
    operator: not_exists
```

```bash
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY --policy policy.yaml
# Exit code 1 if policy fails
```

## Knowledge Base

The analyzer uses a versioned DuckDB catalog of AI framework symbols.

```bash
# Download the latest KB
cisco-aibom kb download

# Check for updates
cisco-aibom kb check

# Verify integrity
cisco-aibom kb verify

# View info
cisco-aibom kb info
```

Manual download from GitHub Releases (replace `<VERSION>` with the desired KB version, e.g. the latest tag from [Releases](https://github.com/cisco-ai-defense/aibom/releases)):

```bash
VERSION="<VERSION>"
mkdir -p "${HOME}/.aibom/catalogs"
gh release download "${VERSION}" \
  --repo cisco-ai-defense/aibom \
  --pattern "aibom_catalog-${VERSION}.duckdb" \
  --dir "${HOME}/.aibom/catalogs"

export AIBOM_DB_PATH="${HOME}/.aibom/catalogs/aibom_catalog-${VERSION}.duckdb"
```

## Environment Variables

All CLI options with an `envvar` binding can be set via environment variables or a `.env` file. The CLI auto-loads `.env` from the current directory, or you can specify a custom path with `AIBOM_ENV_FILE`.

| Variable | CLI Option | Description |
|----------|-----------|-------------|
| `AIBOM_LOG_LEVEL` | `--log-level` | Logging level (default `INFO`). |
| `AIBOM_LLM_MODEL` | `--llm-model` | LLM model name. |
| `AIBOM_LLM_PROVIDER` | `--llm-provider` | LangChain provider (`openai`, `azure_openai`, `bedrock`, `ollama`, etc.). |
| `AIBOM_LLM_API_KEY` | `--llm-api-key` | LLM API key. |
| `AIBOM_LLM_API_BASE` | `--llm-api-base` | LLM API base URL. |
| `AIBOM_LLM_API_VERSION` | `--llm-api-version` | API version (Azure OpenAI). |
| `AIBOM_POST_URL` | `--post-url` | HTTP endpoint to POST the report to. |
| `AIBOM_POST_TIMEOUT` | `--post-timeout` | Timeout in seconds for POSTing the report (default `30`). |
| `AIBOM_POST_VERIFY_TLS` | `--post-verify-tls` | Verify TLS certificates when POSTing (`true`/`false`). |
| `AI_DEFENSE_API_KEY` | `--ai-defense-api-key` | Cisco AI Defense tenant API key (sent as `x-cisco-ai-defense-tenant-api-key`). |
| `CISCO_AI_DEFENSE_API_KEY` | `--api-key` (kb commands) | Cisco AI Defense tenant API key for `kb request`, `kb request-status`, and `kb list-requests`. |
| `CISCO_AI_DEFENSE_API_BASE` | `--api-base` (kb commands) | Regional Cisco AI Defense API host for `kb request*` commands (e.g. `https://api.security.cisco.com`, `https://api.eu.security.cisco.com`). No default. |
| `CISCO_AIBOM_MANIFEST_URL` | `--url` (`kb download`) | KB manifest URL for `kb download` / `kb check`. No default. |
| `AIBOM_GITHUB_ORG` | `--github-org` | GitHub org for repo discovery. |
| `AIBOM_GITLAB_GROUP` | `--gitlab-group` | GitLab group for repo discovery. |
| `AIBOM_BITBUCKET_PROJECT` | `--bitbucket-project` | Bitbucket project for repo discovery. |
| `AIBOM_PLATFORM_TOKEN` | `--platform-token` | Auth token for GitHub/GitLab/Bitbucket. |
| `AIBOM_DB_PATH` | — | Override path to the DuckDB catalog file. |
| `AIBOM_DB_SHA256` | — | Expected SHA-256 checksum for the catalog. |
| `AIBOM_MANIFEST_PATH` | — | Override path to `manifest.json`. |
| `AIBOM_ENV_FILE` | — | Path to a custom `.env` file. |
| `AIBOM_GALILEO_ENABLED` | `--galileo/--no-galileo` | Enable sanitized Galileo telemetry (default `false`). |
| `AIBOM_GALILEO_SAMPLE_RATE` | `--galileo-sample-rate` | Deterministic ordinary-trace emission rate from `0.0` to `1.0` (default `1.0`; sanitized operational and quality exceptions are retained). |
| `AIBOM_GALILEO_HMAC_KEY` | — | Dedicated secret for stable pseudonyms and deterministic sampling. |
| `AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD` | — | Explicitly approve sanitized or raw telemetry egress to the hosted `https://app.galileo.ai` console (default `false`). |
| `AIBOM_GALILEO_SETUP_BUDGET_S` | — | Bounded Galileo logger/session setup budget in seconds (`>0` through `10`; hosted default `2`). |
| `AIBOM_GALILEO_EVALUATION_PROJECT_ID` | — | Immutable hosted project UUID required by custom-function experiments and full-trajectory callbacks. |
| `AIBOM_GALILEO_EVALUATION_LOG_STREAM_ID` | — | Immutable hosted log-stream UUID required only by the full-trajectory callback. |
| `AIBOM_GALILEO_ALLOW_EXACT_IDENTITIES` | — | Additional opt-in required for exact experiment rows or live/evaluation raw trajectories; default experiments remain pseudonymous. |
| `AIBOM_GALILEO_ALLOW_FULL_CONTENT` | — | Additional opt-in for approved evidence or live/evaluation raw trajectories. |
| `AIBOM_GALILEO_ALLOW_FULL_TRAJECTORY` | — | Final independent opt-in for live/evaluation raw LangChain prompts, responses, tool I/O, metadata, and exceptions. |
| `GALILEO_API_KEY` | — | Galileo SDK credential; required when telemetry is enabled. |
| `GALILEO_CONSOLE_URL` | — | Required hosted console URL; accepts exactly `https://app.galileo.ai` with the public-cloud approval flag. |
| `GALILEO_API_URL` | — | Optional explicit API URL; when set, accepts exactly `https://api.galileo.ai`. |
| `GALILEO_SSL_CONTEXT` | — | Optional Galileo SDK TLS setting; omitted or a true value is accepted, while disabling TLS verification rejects telemetry. |
| `GALILEO_PROJECT` | — | Pre-existing destination Galileo project name. |
| `GALILEO_LOG_STREAM` | — | Pre-existing destination Galileo log-stream name. |

## Docker

A single multi-stage Dockerfile is provided. It installs the `all` extra (which expands to `analysis`, `security`, `agentic`, `observability`, `llm-openai`, `llm-aws`, `llm-anthropic`, `llm-google`, `cloud`) because the `analyze` command requires the agentic pipeline plus at least one LLM provider. Image size is ~800 MB.

```bash
cd aibom

# Build
docker build -t cisco-aibom .

# Run
docker run --rm -v /path/to/project:/workspace cisco-aibom \
  analyze /workspace -o json -O /workspace/report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# Or via docker compose (see aibom/docker-compose.yml for env var forwarding)
SCAN_DIR=/path/to/project docker compose run aibom \
  analyze /workspace -o json -O /workspace/report.json \
  --llm-model gpt-5.4
```

## Testing

```bash
cd aibom
uv run pytest tests -v
```

## Further Reading

- [CLI Reference](https://github.com/cisco-ai-defense/aibom/blob/main/docs/CLI_REFERENCE.md) — Complete command and option reference.
- [Agentic Mode Guide](https://github.com/cisco-ai-defense/aibom/blob/main/docs/AGENTIC_MODE.md) — LLM enrichment setup, providers, and tuning.
- [Galileo Observability](https://github.com/cisco-ai-defense/aibom/blob/main/aibom/docs/galileo-observability.md) — Sanitized production telemetry, evaluation metrics, operator setup, and rollout.
- [Container Scanning Guide](https://github.com/cisco-ai-defense/aibom/blob/main/docs/CONTAINER_SCANNING.md) — Extraction tiers, Syft, and runtime support.
- [API Server](https://github.com/cisco-ai-defense/aibom/blob/main/docs/API_SERVER_README.md) — FastAPI endpoint details for `--output-format api`.
- [Technical Overview](https://github.com/cisco-ai-defense/aibom/blob/main/docs/TECHNICAL_OVERVIEW.md) — Architecture, pipeline stages, and scanner design.

## Troubleshooting

- **DuckDB catalog errors:** Run `cisco-aibom kb download` to fetch the latest catalog, or set `AIBOM_DB_PATH` to point at an existing file. Use `cisco-aibom kb verify` to check integrity.
- **Container extraction fails:** Ensure Docker or an alternative runtime is installed and running. Use `--container-extraction-tier` to force a specific tool. See [docs/CONTAINER_SCANNING.md](https://github.com/cisco-ai-defense/aibom/blob/main/docs/CONTAINER_SCANNING.md).
- **Missing `--llm-model`:** The LLM agent is required. Install the agentic extra (`uv tool install "cisco-aibom[agentic,llm-openai]"`) and supply `--llm-model` or set `AIBOM_LLM_MODEL`. See [docs/AGENTIC_MODE.md](https://github.com/cisco-ai-defense/aibom/blob/main/docs/AGENTIC_MODE.md).
- **LLM provider errors / missing provider package:** Each provider needs its integration extra (`llm-openai`, `llm-aws`, `llm-anthropic`, `llm-google`). If it is missing, the CLI fails fast with the exact `uv tool install "cisco-aibom[agentic,llm-…]"` command to run — including when the model is given by bare name (e.g. `--llm-model gpt-4o`). For Azure OpenAI, `--llm-api-version` is required.
- **Slow scans on large repos:** Use `--timing` to identify bottlenecks. Use `--agentic-fast-model` for a cheaper model on simple confirmations, or increase `--agentic-concurrency` for parallel batches.
- **Missing output files:** `--output-file` / `-O` is required for all file-based formats.
- **Report submission:** Set `AIBOM_POST_URL` and `AI_DEFENSE_API_KEY`. Regional endpoints: US (`api.security.cisco.com`), APJ (`api.apj.security.cisco.com`), EU (`api.eu.security.cisco.com`), UAE (`api.uae.security.cisco.com`).
- **Upload succeeded but nothing appears in AI Inventory:** Only **git** and **container-image** scans are projected into the inventory; the CLI auto-captures their source attribution (repo URL + commit SHA, or image ref + digest). A scan of a plain local path (not a git checkout) uploads successfully but is not projected, because it has no stable asset identity. Re-run the scan against a git working tree or a container image.
