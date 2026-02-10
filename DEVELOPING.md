# Developing in aibom

This guide is for contributors working on this repository locally.

## Repository Layout

- `aibom/`: Python analyzer package, CLI, tests, and manifest config
- `ui/`: React + Vite frontend for exploring analyzer results
- `docs/`: API/UI usage docs

## Prerequisites

- Python `3.11` to `3.13`
- `uv` (recommended Python package manager)
- Node.js `>=22` (for `ui/`)
- Docker (optional, needed for container-image analysis paths)

## Python Analyzer Development

All Python commands below assume you are in `aibom/`.

```bash
cd aibom
uv sync --group dev
uv run cisco-aibom --help
```

### Common commands

```bash
# Run test suite
uv run pytest

# Run a single test file
uv run pytest tests/test_cli_basic.py -v

# Format and lint
uv run black src tests
uv run isort src tests
uv run flake8 src tests

# Type checking
uv run mypy src
```

## UI Development

All UI commands below assume you are in `ui/`.

```bash
cd ui
npm ci
npm run dev
```

### Common commands

```bash
# Lint
npm run lint

# Production build
npm run build
```

## Local End-to-End Flow (Analyzer API + UI)

1. Start analyzer UI mode in one terminal:

```bash
cd aibom
uv run cisco-aibom analyze /path/to/project --output-format ui
```

2. Start the React UI in another terminal:

```bash
cd ui
cat > .env.local <<'EOF'
VITE_API_BASE_URL=http://127.0.0.1:8000
EOF
npm run dev
```

If `VITE_API_BASE_URL` is not set, UI requests are served by MSW mock handlers.

## Knowledge Base and Versioning Rules

The analyzer uses a DuckDB catalog and manifest under `aibom/`:

- `aibom/src/aibom/manifest.json`
- `aibom/pyproject.toml`

Before running analysis, ensure a local catalog file is available via `AIBOM_DB_PATH` or the `duckdb_file` entry in `aibom/src/aibom/manifest.json`.

When changing analyzer/catalog version:

1. Update `[project].version` in `aibom/pyproject.toml`.
2. Update `analyzer_version` and `schema_version` in `aibom/src/aibom/manifest.json`.
3. Ensure the same version appears in `duckdb_file`.
4. Recompute SHA-256 and update `duckdb_sha256`:

```bash
shasum -a 256 /path/to/aibom_catalog-<version>.duckdb
```

CI (`version-sync.yml`) validates these fields stay in sync.

## Useful Environment Variables

- `AIBOM_MANIFEST_PATH`: override manifest location
- `AIBOM_DB_PATH`: override local DuckDB file path
- `AIBOM_DB_SHA256`: override expected DuckDB checksum
- `AIBOM_POST_URL`: optional report submission endpoint
- `AI_DEFENSE_API_KEY`: API key for report submission

## Pre-commit Hooks

From repo root:

```bash
pre-commit install
pre-commit run --all-files
```

Run hooks before opening a PR to catch formatting and header issues early.
