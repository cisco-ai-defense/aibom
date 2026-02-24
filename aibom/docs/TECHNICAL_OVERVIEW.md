# AI BOM - Technical Overview

## 1. High-Level Architecture

| Layer | Responsibilities |
| --- | --- |
| **CLI (Typer + Rich)** | Provides `analyze` and `report` commands, loads `.env` (or `AIBOM_ENV_FILE`), validates options, configures logging, and renders Rich summaries. |
| **Config + Manifest** | Loads `manifest.json` from `AIBOM_MANIFEST_PATH`, packaged defaults, then current-working-directory fallback; environment overrides support DB path/SHA and post URL. |
| **Knowledge Base Loader** | Resolves a local DuckDB catalog path (from env or manifest), verifies SHA-256, and returns the validated file path. |
| **Source Resolver** | Distinguishes local paths vs Docker images; for images, runs a container and extracts `/app` or `site-packages` into a temp workspace. |
| **Parser (LibCST)** | Extracts assignments, decorators, standalone calls, type annotations, context managers, imports, class definitions (with base classes), `# aibom:` inline annotations, and raw code snippets. |
| **Custom Catalog** | Loads `.aibom.yaml`/`.yml`/`.json` user configuration: custom component entries, base-class rules, exclude patterns, extended relationship hints, and custom relationship types. |
| **Workflow Index (AST)** | Builds a best-effort call graph with function boundaries and callsite metadata for workflow context. |
| **Categorizer + Relationships** | Matches parsed symbols to catalog entries (DuckDB + custom), processes inline annotations and base-class rules, assigns categories, optionally enriches model/tool details via LLM, and derives built-in and user-defined relationship links. |
| **Reporting + API** | Emits plaintext or JSON reports, or starts the FastAPI API server; optional POST of JSON with retries. |

## 2. Execution Flow

1. **Startup** - Load `.env` (from `AIBOM_ENV_FILE` or local defaults), parse CLI args, validate output/LLM options, and apply manifest config for DuckDB path/checksum defaults.  
2. **Knowledge Base** - Call `ensure_local_database()` to resolve and verify the local DuckDB catalog with SHA-256 validation.  
3. **Custom Catalog** - Load `.aibom.yaml`/`.yml`/`.json` (from `--custom-catalog` or auto-discovered in the source directory). Merge custom component entries into the catalog, register exclude patterns, and pass base-class rules, relationship hints, and custom relationship types to the categorizer.
4. **Source Acquisition** - For each source: detect Docker vs local path, extract `/app` or `site-packages` for images, and list `.py` files.
5. **Parsing** - Run the LibCST visitor per file to collect assignments, decorators, type annotations, context managers, imports, class definitions (with base classes), and `# aibom:` inline annotations; parse errors are logged as warnings.
6. **Workflow Index** - Build an AST-based call graph for the source files to provide workflow context (distance, callsite, arguments).
7. **Categorization** - Query the catalog by suffix (DuckDB + custom entries, minus excludes), keep exact symbol matches, assign categories, process inline annotations and base-class rules, attach workflow context, and derive built-in and user-defined relationships. Optional LLM enrichment can add tool descriptions and model names.
8. **Reporting** - Convert container temp paths to container-style paths, build per-source summaries and run metadata, and emit plaintext or JSON reports.
9. **Publishing / API** - Optionally POST the JSON report with retries, or start the in-memory API server (`--output-format api`).
10. **Console Summary** - Render Rich summaries and workflow examples when `--show-summary` is enabled.

## 3. Key Modules

| Module | Notes |
| --- | --- |
| `src/aibom/cli.py` | CLI entrypoint; orchestrates analysis, loads `.env`, resolves manifest settings, loads custom catalog, and renders outputs. |
| `src/aibom/cst_parser.py` | LibCST-based parser for assignments, decorators, annotations, context managers, imports, class definitions (with base classes), `# aibom:` inline annotations, and raw code. |
| `src/aibom/custom_catalog.py` | Loads and validates `.aibom.yaml`/`.yml`/`.json` user configuration files. Parses custom component entries, base-class detection rules, exclude patterns, extended relationship hints, and custom relationship types. Also provides `parse_inline_annotation()` for `# aibom:` comment parsing. |
| `src/aibom/workflow_analyzer.py` | AST-based function index and call graph for workflow context enrichment. |
| `src/aibom/categorizer.py` | Maps observations to catalog entries (DuckDB + supplemental + custom), processes inline annotations and base-class rules, attaches workflows, and derives built-in and user-defined relationships. |
| `src/aibom/catalog_db.py` | DuckDB access layer for catalog lookup; supports custom entry merging and exclude filtering. |
| `src/aibom/db_loader.py` | Manifest/env path resolution and SHA verification of the local catalog. |
| `src/aibom/report_sender.py` | POSTs JSON report payloads with retry/backoff. |
| `src/aibom/api_handler.py` | Converts results to a DataFrame and starts the FastAPI API server. |
| `src/aibom/api/server.py` | FastAPI endpoints for component browsing and health checks. |
| `tests/…` | Coverage for parsing, categorization, workflow indexing, custom catalog loading, and report generation. |

## 4. Workflow Index Essentials

1. **AST Function Indexing** - Records function/method boundaries, qualified names, decorators, class context, and parameter lists.  
2. **Call Edge Recording** - Captures best-effort call relationships within a file, storing callsite line numbers and serialized argument strings.  
3. **Reverse Graph Traversal** - `get_workflow_context` walks callers breadth-first (default depth 4, max 10 entries) and returns `workflow_id`, `distance`, and call metadata.  
4. **Reporting** - Workflow context is attached to components and also summarized per source in the JSON report.

## 5. Knowledge Base (DuckDB) Strategy

* Manifest keys: `duckdb_sha256` and `duckdb_file`, with env overrides (`AIBOM_DB_PATH`, `AIBOM_DB_SHA256`).  
* `duckdb_file` is resolved relative to `manifest.json` when the value is not absolute.  
* Catalog lookup uses `id LIKE %suffix` for parsed symbols; final matches are exact string matches against the parsed qualified names.  
* User-defined custom entries from `.aibom.yaml` are merged into the in-memory catalog with lowest precedence (DuckDB > supplemental > custom). Exclude patterns filter out matching entries from all sources.  

## 6. Report Submission

* `--post-url` triggers a POST of the JSON report; `AIBOM_POST_URL`, `AIBOM_POST_TIMEOUT`, and `AIBOM_POST_VERIFY_TLS` mirror CLI options.  
* Payload includes `run_id`, `analyzer_version`, `submitted_at`, `source_kind`, `sources`, and the report body.  
* Requests use `x-cisco-ai-defense-tenant-api-key` when `--ai-defense-api-key`/`AI_DEFENSE_API_KEY` is set, with retry/backoff for 429/5xx.  

## 7. Outputs

| Output | Description |
| --- | --- |
| Plaintext | Report file listing detected components and their workflow context. |
| JSON | `aibom_analysis` with metadata, per-source components, workflow summaries, relationships, and errors. |
| API | FastAPI server that serves component data (`/api/components`, `/health`). |
| Report Command | `cisco-aibom report` renders JSON summaries and optionally the raw JSON. |
