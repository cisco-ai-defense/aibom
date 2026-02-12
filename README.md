# AI BOM

The AI BOM tool scans codebases and container images to inventory AI framework components (models, agents, tools, prompts, and more). It currently parses Python source code, resolves fully qualified symbols, and matches them against a DuckDB catalog to produce an AI bill of materials (AI BOM). Optional LLM enrichment extracts model names, and a workflow pass annotates components with call-path context.

## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Knowledge Base Configuration](#knowledge-base-configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Output Formats](#output-formats)
- [UI Mode](#ui-mode)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

## Features

- **Static Python analysis:** Uses `libcst` to capture assignments, decorators, type annotations, and context managers.
- **Container image scanning:** Extracts `/app` from Docker images when available, otherwise scans `site-packages`.
- **DuckDB catalog matching:** Maps fully qualified symbols to curated component categories.
- **Workflow context:** Builds a lightweight call graph to show which workflows reach each component.
- **Derived relationships:** Infers `USES_TOOL` and `USES_LLM` links from agent arguments.
- **Optional LLM enrichment:** Uses `litellm` to extract model/embedding names from code snippets.
- **Multiple outputs:** Plaintext, JSON, or a FastAPI UI server.
- **Report submission:** Optional POST of the JSON report with retries.

## Repository Layout

```
aibom/   # Python analyzer package + CLI
ui/      # React UI for exploring results
docs/    # UI/API documentation
```

## Installation

### Prerequisites

- Python 3.11+
- uv (Python package manager, recommended)
- Docker (optional, for container image analysis)
- Node.js 22+ (optional, for the React UI)
- LLM provider API key (optional, for model extraction)

### Installing as a CLI tool

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv

uv tool install --python 3.13 cisco-aibom

# Verify installation
cisco-aibom --help
```

Alternatively, install from source:

```bash
uv tool install --python 3.13 --from git+https://github.com/cisco-ai-defense/aibom cisco-aibom

# Verify installation
cisco-aibom --help
```

### Installing for local development

```bash
git clone https://github.com/cisco-ai-defense/aibom.git
cd aibom/aibom

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv

uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Verify installation
cisco-aibom --help
```

When working from source, you can also run the CLI with `uv run cisco-aibom ...` or `uv run python -m aibom ...`.

## Knowledge Base Configuration

The analyzer uses a local DuckDB catalog described by `manifest.json`.
The DuckDB file is a prebuilt, versioned knowledge-catalog artifact of AI frameworks. It is used as a read-only lookup dataset, with checksum verification for compatibility and integrity.
For users running the packaged CLI (for example via `uv tool install` or `pip`), the packaged manifest provides a default checksum and default catalog location (`~/.aibom/catalogs/aibom_catalog-<version>.duckdb`). You can still override with `AIBOM_DB_PATH` and `AIBOM_DB_SHA256`.
When running from source, execute from the `aibom/` directory or set `AIBOM_MANIFEST_PATH` to point at `aibom/src/aibom/manifest.json`.

### Download the DuckDB artifact from GitHub Releases

```bash
# Set this to the release tag that matches your catalog artifact (example: 0.2.1)
VERSION="<version>"
mkdir -p "${HOME}/.aibom/catalogs"

# Option 1: GitHub CLI
gh release download "${VERSION}" \
  --repo cisco-ai-defense/aibom \
  --pattern "aibom_catalog-${VERSION}.duckdb" \
  --dir "${HOME}/.aibom/catalogs"

# Option 2: direct download URL
curl -fL \
  -o "${HOME}/.aibom/catalogs/aibom_catalog-${VERSION}.duckdb" \
  "https://github.com/cisco-ai-defense/aibom/releases/download/${VERSION}/aibom_catalog-${VERSION}.duckdb"
```

### Provide the DuckDB path to the analyzer

```bash
export AIBOM_DB_PATH="${HOME}/.aibom/catalogs/aibom_catalog-${VERSION}.duckdb"

# Set only if your file is different from the manifest default (for example,
# custom path/version) or if you see a checksum mismatch error:
# export AIBOM_DB_SHA256="<sha256-of-${AIBOM_DB_PATH}>"
```

Compute SHA-256 when needed:

```bash
# macOS
shasum -a 256 "${AIBOM_DB_PATH}"

# Linux
sha256sum "${AIBOM_DB_PATH}"
```

Use only the hash value (first column) as `AIBOM_DB_SHA256`.

Override settings with environment variables:

- `AIBOM_DB_PATH`: local DuckDB file path
- `AIBOM_DB_SHA256`: SHA-256 checksum for the DuckDB file

`AIBOM_DB_PATH` may be absolute or relative. Relative env-var values are resolved from the current working directory; relative `duckdb_file` values in `manifest.json` are resolved from the manifest directory.

## Usage

### Analyze sources

```bash
# Local directory (JSON output)
cisco-aibom analyze /path/to/project --output-format json --output-file report.json

# Container image (JSON output)
cisco-aibom analyze langchain-app:latest --output-format json --output-file report.json

# Multiple images from a JSON list
cisco-aibom analyze --images-file images.json --output-format plaintext --output-file report.txt
```

`--output-file` is required for `plaintext` and `json` output formats.

### Render a JSON report

```bash
cisco-aibom report report.json --raw-json
```

### Optional LLM enrichment

```bash
cisco-aibom analyze /path/to/project \
  --output-format json \
  --output-file report.json \
  --llm-model gpt-3.5-turbo \
  --llm-api-base https://api.openai.com/v1 \
  --llm-api-key $OPENAI_API_KEY
```

Local LLM example:

```bash
cisco-aibom analyze /path/to/project \
  --output-format json \
  --output-file report.json \
  --llm-model ollama_chat/gemma3:12b \
  --llm-api-base http://localhost:11434
```

### Optional report submission

```bash
cisco-aibom analyze /path/to/project \
  --output-format json \
  --output-file report.json \
  --post-url https://api.security.cisco.com/api/ai-defense/v1/aibom/analysis \
  --ai-defense-api-key $AI_DEFENSE_API_KEY
```

You can also set `AIBOM_POST_URL` instead of `--post-url` and `AI_DEFENSE_API_KEY` instead of `--ai-defense-api-key`.

The API key is sent as the `x-cisco-ai-defense-tenant-api-key` header. Use the same path in every region:
`/api/ai-defense/v1/aibom/analysis`.

Choose the base domain for your Cisco AI Defense organization's region:

- US: `https://api.security.cisco.com/api/ai-defense/v1/aibom/analysis`
- APJ: `https://api.apj.security.cisco.com/api/ai-defense/v1/aibom/analysis`
- EU: `https://api.eu.security.cisco.com/api/ai-defense/v1/aibom/analysis`
- UAE: `https://api.uae.security.cisco.com/api/ai-defense/v1/aibom/analysis`

## Testing

```bash
cd aibom
uv run pytest tests -v
```

## Output Formats

### Plaintext output

```text
--- AI BOM Analysis Report ---

--- Results for source: langchain-app:latest ---

[+] Found 4 MODEL:
  - Name: langchain_community.llms.openai.OpenAI
    Model: gpt-3.5-turbo-instruct
    Source: /app/comprehensive_langchain_app.py:32
...
--- End of Report: Found 42 total components across all sources. ---
```

### JSON output

```json
{
  "aibom_analysis": {
    "metadata": {
      "run_id": "...",
      "analyzer_version": "<analyzer-version>",
      "started_at": "2025-01-01T00:00:00Z",
      "completed_at": "2025-01-01T00:00:10Z"
    },
    "sources": {
      "langchain-app:latest": {
        "components": {
          "model": [
            {
              "name": "langchain_community.llms.openai.OpenAI",
              "file_path": "/app/app.py",
              "line_number": 32,
              "category": "model",
              "model_name": "gpt-3.5-turbo",
              "workflows": []
            }
          ]
        },
        "relationships": [
          {
            "source_instance_id": "...",
            "target_instance_id": "...",
            "label": "USES_LLM",
            "source_name": "...",
            "target_name": "...",
            "source_category": "agent",
            "target_category": "model"
          }
        ],
        "workflows": [
          {
            "id": "...",
            "function": "module.flow",
            "file_path": "/app/app.py",
            "line": 10,
            "distance": 0
          }
        ],
        "total_components": 42,
        "total_workflows": 7,
        "summary": {
          "status": "completed",
          "source_kind": "container"
        }
      }
    },
    "summary": {
      "total_sources": 1,
      "total_components": 42,
      "total_relationships": 3,
      "total_workflows": 7,
      "categories": {
        "model": 4,
        "tool": 8
      }
    },
    "errors": []
  }
}
```

## UI Mode

`--output-format ui` starts a FastAPI server that serves the analyzed components:

```bash
cisco-aibom analyze /path/to/project --output-format ui
```

Endpoints:

- `GET /api/components`
- `GET /api/components/types`
- `GET /api/components/{id}`
- `GET /health`

The React UI in `ui/` can connect to this server. See `docs/UI_README.md` and `docs/API_SERVER_README.md` for details.

## Technical Details

- **Parsing:** `libcst` extracts fully qualified names for calls, decorators, type annotations, and context managers.
- **Catalog matching:** Symbols are matched against the DuckDB `component_catalog` table using their fully qualified IDs.
- **Workflow analysis:** The AST-based workflow analyzer associates components with the functions that call into them.
- **Relationships:** Agent arguments are inspected for tool/LLM references to derive `USES_TOOL` and `USES_LLM` links.
- **LLM enrichment:** `litellm` is used only when `--llm-model` is supplied.

## Troubleshooting

- **DuckDB catalog errors:** Ensure the catalog file exists at `AIBOM_DB_PATH` (or `duckdb_file` in manifest) and that `AIBOM_DB_SHA256` (or `duckdb_sha256` in manifest) matches the file checksum. When running from source, execute from `aibom/` or set `AIBOM_MANIFEST_PATH`.
- **Docker issues:** Container analysis requires a working Docker CLI and daemon.
- **LLM configuration errors:** `--llm-api-base` is required whenever `--llm-model` is set.
- **UI server does not start:** If no components are found, the UI server exits early. Verify the target includes AI framework usage.
- **Missing output files:** `--output-file` is mandatory for `plaintext` and `json` formats.
