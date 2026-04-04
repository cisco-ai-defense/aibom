# AI BOM

[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?logo=discord&logoColor=white)](https://discord.com/invite/nKWtDcXxtx)
[![Cisco AI Defense](https://img.shields.io/badge/Cisco-AI%20Defense-049fd9?logo=cisco&logoColor=white)](https://www.cisco.com/site/us/en/products/security/ai-defense/index.html)
[![AI Security and Safety Framework](https://img.shields.io/badge/AI%20Security-Framework-orange)](https://learn-cloudsecurity.cisco.com/ai-security-framework)

Cisco AI BOM scans codebases, container images, and cloud environments to produce an AI Bill of Materials — a structured inventory of models, agents, tools, MCP servers/clients, datasets, prompts, guardrails, secrets, and other AI assets used in your software. It supports Python, JavaScript/TypeScript, Java, Go, Rust, Ruby, C#, and PHP, with deterministic detection, cross-reference resolution, and optional LLM-powered agentic enrichment.

## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
- [Agentic Enrichment](#agentic-enrichment)
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

- **Multi-language analysis** — Python (LibCST), JavaScript/TypeScript, Java, Go, Rust, Ruby, C#, PHP (tree-sitter).
- **21 built-in scanners** — model detection, dependency analysis, secret detection, vulnerability scanning (OSV.dev), MCP server/client detection, ML lifecycle detection, cloud resource scanning, CI/CD pipeline analysis, deployment detection, container scanning, data-file scanning, environment variable resolution, KB enrichment, and more.
- **24 AI component types** — `model`, `agent`, `tool`, `mcp_server`, `mcp_client`, `embedding`, `vector_store`, `dataset`, `prompt`, `guardrail`, `memory`, `retriever`, `training_run`, `hyperparameter`, `model_artifact`, `experiment_tracker`, `model_registry`, `data_versioning`, `ml_pipeline`, `skill`, `observability`, `secret`, `dependency`, `other`.
- **Three-tier detection** — Tier 1 (deterministic high-confidence), Tier 2 (cross-reference resolution), Tier 3 (agentic LLM reasoning).
- **10 output formats** — Plaintext, JSON, CycloneDX, SARIF, SPDX, HTML dashboard, Markdown, CSV, JUnit, and a live API server.
- **Container image scanning** — Extract and analyze application source code from Docker, Podman, nerdctl, Buildah, Skopeo, Crane, or Undock images, with Anchore Syft for SBOM metadata.
- **Cross-repo and org-level scanning** — Scan multiple local repos, GitHub orgs, GitLab groups, or Bitbucket projects, with incremental caching.
- **Agentic enrichment** — Optional LLM pass (via Deep Agents + LangChain) to resolve ambiguous detections, extract model names, and classify uncertain components.
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
- LLM provider credentials (optional, for agentic enrichment)

### Install from PyPI

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Core CLI
uv tool install --python 3.13 cisco-aibom

# With agentic enrichment (OpenAI / Azure OpenAI)
uv tool install --python 3.13 "cisco-aibom[agentic,llm-openai]"

# With agentic enrichment (AWS Bedrock)
uv tool install --python 3.13 "cisco-aibom[agentic,llm-aws]"

# Everything
uv tool install --python 3.13 "cisco-aibom[all]"

# Verify
cisco-aibom --help
```

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

### Optional extras

| Extra | Installs | Purpose |
|-------|----------|---------|
| `agentic` | Deep Agents, LangChain | LLM-powered agentic enrichment |
| `llm-openai` | `langchain-openai` | OpenAI / Azure OpenAI provider |
| `llm-aws` | `langchain-aws` | AWS Bedrock provider |
| `analysis` | `detect-secrets`, `tree-sitter` | Secret detection, multi-language parsing |
| `security` | `cisco-ai-mcp-scanner`, `cisco-ai-skill-scanner` | Cisco security tool integration |
| `cloud` | `boto3`, `google-cloud-aiplatform`, `azure-*` | Cloud resource scanning |
| `all` | All of the above | Full feature set |

## Quick Start

```bash
# Scan a local project
cisco-aibom analyze /path/to/project -o json -O report.json

# Scan a container image
cisco-aibom analyze my-app:latest -o json -O report.json

# Scan with agentic enrichment
cisco-aibom analyze /path/to/project -o json -O report.json \
  --llm-model gpt-4o --llm-provider openai --llm-api-key $OPENAI_API_KEY

# Scan multiple repos under a directory
cisco-aibom analyze /path/to/repos --discover-repos -o json -O report.json

# HTML dashboard
cisco-aibom analyze /path/to/project -o html -O dashboard.html

# Policy gate for CI
cisco-aibom analyze /path/to/project -o json -O report.json --policy policy.yaml
```

## Commands

| Command | Description |
|---------|-------------|
| `analyze` | Scan source code, container images, or repos and produce an AI BOM. |
| `report` | Render a previously generated JSON report with Rich formatting. |
| `watch` | Poll directories for changes and re-scan with delta reporting. |
| `diff run` | Compare two AIBOM JSON reports side-by-side. |
| `benchmark run` | Measure precision/recall/F1 against ground-truth YAML. |
| `kb download` | Download the latest knowledge base. |
| `kb check` | Check if a newer KB version is available. |
| `kb info` | Display info about the locally installed KB. |
| `kb verify` | Verify KB integrity (SHA-256 checksum). |
| `kb request` | Request a KB build for a specific SDK version. |
| `cache clear` | Remove cached scan results and agentic cache. |
| `cache list` | List cached scan entries. |
| `plugin list` | List discovered plugins (entry points, MCP servers). |

See [docs/CLI_REFERENCE.md](https://github.com/cisco-ai-defense/aibom/blob/main/docs/CLI_REFERENCE.md) for complete option details.

### Global options

| Option | Env Var | Description |
|--------|---------|-------------|
| `--log-level` | `AIBOM_LOG_LEVEL` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default `INFO`). |

## Agentic Enrichment

When `--llm-model` is supplied (requires `cisco-aibom[agentic]`), the CLI runs a second pass using LLM-powered agents to:

- Resolve ambiguous component classifications
- Extract model names from code context
- Confirm or remove false positives from deterministic detection
- Discover components missed by static analysis

```bash
# OpenAI
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-4o --llm-provider openai --llm-api-key $OPENAI_API_KEY

# Azure OpenAI
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-4o --llm-provider azure_openai \
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
| `--agentic-scope` | `candidates` | `candidates` (only ambiguous items) or `all` (every component). |
| `--agentic-batch-size` | `5` | Max components per LLM invocation. |
| `--agentic-concurrency` | `1` | Max parallel LLM batches. |
| `--agentic-timeout` | `120` | Wall-clock seconds per batch before timeout. |
| `--agentic-fast-model` | — | Cheaper model for simple confirmations. |

## Container Scanning

The CLI auto-detects container image references and extracts application source code for analysis.

```bash
# Auto-detect extraction method
cisco-aibom analyze my-app:latest -o json -O report.json

# Force a specific extraction tier
cisco-aibom analyze my-app:latest -o json -O report.json --container-extraction-tier podman
```

Supported tiers: `auto`, `syft`, `docker`, `podman`, `nerdctl`, `buildah`, `crane`, `skopeo`, `tarball`.

See [docs/CONTAINER_SCANNING.md](https://github.com/cisco-ai-defense/aibom/blob/main/docs/CONTAINER_SCANNING.md) for details.

## Cross-Repo and Org Scanning

```bash
# Discover and scan all git repos under a directory
cisco-aibom analyze /path/to/repos --discover-repos -o json -O report.json

# Scan a GitHub org (requires GITHUB_TOKEN)
cisco-aibom analyze --github-org my-org --platform-token $GITHUB_TOKEN -o json -O report.json

# Scan a GitLab group
cisco-aibom analyze --gitlab-group my-group --platform-token $GITLAB_TOKEN -o json -O report.json

# Scan repos from a file (JSON array or newline-delimited)
cisco-aibom analyze --repos-file repos.txt -o json -O report.json

# Incremental scan (skip repos with unchanged HEAD)
cisco-aibom analyze /path/to/repos --discover-repos --incremental -o json -O report.json

# Limit and filter
cisco-aibom analyze --github-org my-org --platform-token $GITHUB_TOKEN \
  --max-repos 50 --repo-filter "ml-" --parallel-repos 4 -o json -O report.json
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
cisco-aibom analyze ./my-app -o json -O report.json --policy policy.yaml
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

Manual download from GitHub Releases:

```bash
VERSION="0.5.1"
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
| `AI_DEFENSE_API_KEY` | `--ai-defense-api-key` | Cisco AI Defense API key. |
| `AIBOM_GITHUB_ORG` | `--github-org` | GitHub org for repo discovery. |
| `AIBOM_GITLAB_GROUP` | `--gitlab-group` | GitLab group for repo discovery. |
| `AIBOM_BITBUCKET_PROJECT` | `--bitbucket-project` | Bitbucket project for repo discovery. |
| `AIBOM_PLATFORM_TOKEN` | `--platform-token` | Auth token for GitHub/GitLab/Bitbucket. |
| `AIBOM_DB_PATH` | — | Override path to the DuckDB catalog file. |
| `AIBOM_DB_SHA256` | — | Expected SHA-256 checksum for the catalog. |
| `AIBOM_MANIFEST_PATH` | — | Override path to `manifest.json`. |
| `AIBOM_ENV_FILE` | — | Path to a custom `.env` file. |

## Docker

Two Dockerfiles are provided:

| Image | Dockerfile | Extras | Size |
|-------|-----------|--------|------|
| `cisco-aibom` | `Dockerfile` | `analysis`, `security` | ~200 MB |
| `cisco-aibom-agentic` | `Dockerfile.agentic` | All (`analysis`, `security`, `agentic`, `cloud`) | ~800 MB |

```bash
cd aibom

# Build the deterministic image
docker build -t cisco-aibom .

# Build the full agentic image
docker build -f Dockerfile.agentic -t cisco-aibom-agentic .

# Run
docker run --rm -v /path/to/project:/workspace cisco-aibom analyze /workspace -o json -O /workspace/report.json
```

## Testing

```bash
cd aibom
uv run pytest tests -v
```

## Further Reading

- [CLI Reference](https://github.com/cisco-ai-defense/aibom/blob/main/docs/CLI_REFERENCE.md) — Complete command and option reference.
- [Agentic Mode Guide](https://github.com/cisco-ai-defense/aibom/blob/main/docs/AGENTIC_MODE.md) — LLM enrichment setup, providers, and tuning.
- [Container Scanning Guide](https://github.com/cisco-ai-defense/aibom/blob/main/docs/CONTAINER_SCANNING.md) — Extraction tiers, Syft, and runtime support.
- [API Server](https://github.com/cisco-ai-defense/aibom/blob/main/docs/API_SERVER_README.md) — FastAPI endpoint details for `--output-format api`.
- [Technical Overview](https://github.com/cisco-ai-defense/aibom/blob/main/aibom/docs/TECHNICAL_OVERVIEW.md) — Architecture, pipeline stages, and scanner design.

## Troubleshooting

- **DuckDB catalog errors:** Run `cisco-aibom kb download` to fetch the latest catalog, or set `AIBOM_DB_PATH` to point at an existing file. Use `cisco-aibom kb verify` to check integrity.
- **Container extraction fails:** Ensure Docker or an alternative runtime is installed and running. Use `--container-extraction-tier` to force a specific tool. See [docs/CONTAINER_SCANNING.md](https://github.com/cisco-ai-defense/aibom/blob/main/docs/CONTAINER_SCANNING.md).
- **Agentic mode not activating:** Install the agentic extra (`uv tool install "cisco-aibom[agentic,llm-openai]"`) and supply `--llm-model`. See [docs/AGENTIC_MODE.md](https://github.com/cisco-ai-defense/aibom/blob/main/docs/AGENTIC_MODE.md).
- **LLM provider errors:** Ensure `--llm-provider` matches the installed LangChain integration package. For Azure OpenAI, `--llm-api-version` is required.
- **Slow scans on large repos:** Use `--timing` to identify bottlenecks. Consider `--strict` to skip agentic candidates, or `--agentic-scope candidates` (default) to only send ambiguous items.
- **Missing output files:** `--output-file` / `-O` is required for all file-based formats.
- **Report submission:** Set `AIBOM_POST_URL` and `AI_DEFENSE_API_KEY`. Regional endpoints: US (`api.security.cisco.com`), APJ (`api.apj.security.cisco.com`), EU (`api.eu.security.cisco.com`), UAE (`api.uae.security.cisco.com`).
