# AI BOM

The AI BOM tool scans codebases and container images to inventory AI framework components (models, agents, tools, prompts, and more). It currently parses Python source code, resolves fully qualified symbols, and matches them against a DuckDB catalog to produce an AI bill of materials (AI BOM). Optional LLM enrichment extracts model names, and a workflow pass annotates components with call-path context.

## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Knowledge Base Configuration](#knowledge-base-configuration)
- [Usage](#usage)
- [Custom Catalog](#custom-catalog)
- [Testing](#testing)
- [Output Formats](#output-formats)
- [API Mode](#api-mode)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

## Features

- **Static Python analysis:** Uses `libcst` to capture assignments, decorators, type annotations, context managers, class definitions, and inline annotations.
- **Container image scanning:** Extracts `/app` from Docker images when available, otherwise scans `site-packages`.
- **DuckDB catalog matching:** Maps fully qualified symbols to curated component categories.
- **Custom catalog:** Users can register custom AI components, base-class detection rules, exclude patterns, relationship hints, and custom relationship types via a `.aibom.yaml` configuration file.
- **Inline annotations:** Tag classes and functions directly in source code with `# aibom: concept=...` comments for instant recognition.
- **Base class detection:** Automatically categorize classes that inherit from specified base classes.
- **Workflow context:** Builds a lightweight call graph to show which workflows reach each component.
- **Derived relationships:** Infers `USES_TOOL`, `USES_LLM`, `USES_MEMORY`, `USES_RETRIEVER`, `USES_EMBEDDING`, and user-defined relationship links from component arguments.
- **Optional LLM enrichment:** Uses `litellm` to extract model/embedding names from code snippets.
- **Multiple outputs:** Plaintext, JSON, or a FastAPI API server.
- **Report submission:** Optional POST of the JSON report with retries.

## Repository Layout

```
aibom/   # Python analyzer package + CLI
docs/    # API documentation
```

## Installation

### Prerequisites

- Python 3.11+
- uv (Python package manager, recommended)
- Docker (optional, for container image analysis)
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
# Set this to the release tag that matches your catalog artifact (example: 0.4.0)
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

## Custom Catalog

The built-in DuckDB catalog covers popular AI frameworks (LangChain, LangGraph, CrewAI, PyTorch, scikit-learn, etc.), but many teams build custom wrappers, internal tools, or use niche libraries that the catalog does not know about. The **custom catalog** lets you teach the analyzer about these components using three complementary mechanisms:

1. **Configuration file** (`.aibom.yaml`) -- register components, base-class rules, excludes, and relationships declaratively.
2. **Inline annotations** (`# aibom: concept=...`) -- tag individual classes and functions directly in source code.
3. **Base class detection** -- automatically categorize any class that inherits from a specified base class.

### Using a configuration file

Place a `.aibom.yaml` (or `.aibom.yml` / `.aibom.json`) in your project root. The analyzer auto-discovers it, or you can point to it explicitly:

```bash
# Auto-discovery (looks for .aibom.yaml/.yml/.json in the source directory)
cisco-aibom analyze /path/to/project --output-format json --output-file report.json

# Explicit path
cisco-aibom analyze /path/to/project \
  --custom-catalog /path/to/.aibom.yaml \
  --output-format json \
  --output-file report.json
```

### Configuration file reference

A complete `.aibom.yaml` example (also available at `aibom/examples/.aibom.yaml`):

```yaml
# ─── Custom Components ───────────────────────────────────────────────
# Register symbols the built-in catalog does not know about.
# 'id' can be a short class/function name (e.g. MyLLMWrapper) or a
# fully qualified name (e.g. myproject.llm.MyLLMWrapper).
# Short names are matched via suffix matching, so 'MyLLMWrapper' will
# match any qualified name ending in 'MyLLMWrapper'.
components:
  - id: MyLLMWrapper
    concept: model                   # model | agent | tool | memory | ...
    label: My Custom LLM             # human-readable label (optional)
    framework: internal              # framework name (default: "custom")
    metadata:                        # arbitrary key-value pairs (optional)
      owner: ml-team
      version: "2.1"

  - id: myproject.tools.SearchTool
    concept: tool

  - id: SafetyFilter
    concept: guardrail               # custom categories are allowed

  - id: RequestRouter
    concept: router

# ─── Base Class Detection ────────────────────────────────────────────
# Any class that inherits from a listed base is auto-categorized.
base_classes:
  - class: BaseTool
    concept: tool
  - class: mylib.BaseAgent
    concept: agent
  - class: BaseGuardrail
    concept: guardrail

# ─── Exclude Patterns ────────────────────────────────────────────────
# Suppress false positives. Entries whose IDs end with (or equal) these
# strings are filtered out of analysis results.
excludes:
  - langchain.deprecated.OldAgent
  - some_noisy_helper_function

# ─── Extended Relationship Hints ─────────────────────────────────────
# Add argument names that the relationship engine should inspect.
# These are additive -- they extend the built-in hints, not replace them.
relationship_hints:
  tool_arguments:        # extends: tool, tools, skills, abilities
    - custom_tools
    - plugins
  llm_arguments:         # extends: llm, language_model, chat_model, model
    - language_model
  memory_arguments:      # extends: memory, checkpointer, store, saver, ...
    - state_store
  retriever_arguments:   # extends: retriever, retrievers, search, ...
    - doc_search
  embedding_arguments:   # extends: embedding, embeddings, embed, ...
    - vectorizer

# ─── Custom Relationship Types ───────────────────────────────────────
# Define entirely new relationship labels with source/target constraints
# and the argument names that trigger them.
custom_relationships:
  - label: ROUTES_TO
    source_categories: [router]
    target_categories: [agent]
    argument_hints: [routes, destinations]

  - label: GUARDS
    source_categories: [guardrail]
    target_categories: [model, agent]
    argument_hints: [guarded_by, guard]
```

### Inline annotations

Tag classes or functions directly in your source code. The comment must appear on the line immediately above the definition or as a trailing comment on the definition line:

```python
# aibom: concept=guardrail framework=internal
class SafetyFilter:
    """Custom content-safety guardrail."""

    def check(self, text: str) -> bool:
        ...


# aibom: concept=tool label=WebSearch
def search_web(query: str) -> list:
    """Search the web and return results."""
    ...


class MyRouter:  # aibom: concept=router
    """Routes requests to the appropriate agent."""
    ...
```

Supported keys in the annotation: `concept` (required), `framework` (optional, default `"custom"`), `label` (optional).

### Base class detection

When `base_classes` rules are defined in `.aibom.yaml`, the analyzer inspects every class definition in the scanned code. If a class inherits (directly) from a listed base, it is auto-categorized without needing an explicit `components` entry or inline annotation:

```yaml
# .aibom.yaml
base_classes:
  - class: BaseTool
    concept: tool
```

```python
# my_tools.py -- these are automatically detected as "tool" components
class SearchTool(BaseTool):
    ...

class CalculatorTool(BaseTool):
    ...
```

### Precedence

When the same symbol is detected by multiple mechanisms, the following precedence applies (highest first):

1. **Inline annotation** (`# aibom: concept=...`)
2. **Base class rule** (from `.aibom.yaml` `base_classes`)
3. **Custom component entry** (from `.aibom.yaml` `components`)
4. **Supplemental catalog** (built-in LangGraph/CrewAI entries)
5. **DuckDB catalog** (prebuilt knowledge base)

Exclude patterns override all of the above -- a matching exclude always removes the component from results.

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

## API Mode

`--output-format api` starts a FastAPI server that serves the analyzed components:

```bash
cisco-aibom analyze /path/to/project --output-format api
```

Endpoints:

- `GET /api/components`
- `GET /api/components/types`
- `GET /api/components/{id}`
- `GET /health`

See `docs/API_SERVER_README.md` for detailed API usage.

## Technical Details

- **Parsing:** `libcst` extracts fully qualified names for calls, decorators, type annotations, context managers, class definitions (with base classes), and `# aibom:` inline annotations.
- **Catalog matching:** Symbols are matched against the DuckDB `component_catalog` table using suffix matching on their fully qualified IDs. Custom entries from `.aibom.yaml` are merged into this lookup.
- **Custom catalog:** The `custom_catalog` module loads `.aibom.yaml`/`.yml`/`.json` files and provides component entries, base-class rules, exclude patterns, extended relationship hints, and custom relationship types to the categorizer.
- **Inline annotations:** The CST parser extracts `# aibom: concept=...` comments on class and function definitions, which the categorizer uses to create components without requiring catalog entries.
- **Base class detection:** The CST parser captures base classes for every `class` statement. The categorizer matches these against base-class rules from the custom catalog configuration.
- **Workflow analysis:** The AST-based workflow analyzer associates components with the functions that call into them.
- **Relationships:** Agent arguments are inspected for tool/LLM/memory/retriever/embedding references to derive `USES_TOOL`, `USES_LLM`, `USES_MEMORY`, `USES_RETRIEVER`, and `USES_EMBEDDING` links. User-defined relationship types from `.aibom.yaml` `custom_relationships` are also derived.
- **LLM enrichment:** `litellm` is used only when `--llm-model` is supplied.

## Troubleshooting

- **DuckDB catalog errors:** Ensure the catalog file exists at `AIBOM_DB_PATH` (or `duckdb_file` in manifest) and that `AIBOM_DB_SHA256` (or `duckdb_sha256` in manifest) matches the file checksum. When running from source, execute from `aibom/` or set `AIBOM_MANIFEST_PATH`.
- **Docker issues:** Container analysis requires a working Docker CLI and daemon.
- **LLM configuration errors:** `--llm-api-base` is required whenever `--llm-model` is set.
- **API server questions:** Use `docs/API_SERVER_README.md` for API mode behavior and endpoint details.
- **Missing output files:** `--output-file` is mandatory for `plaintext` and `json` formats.
