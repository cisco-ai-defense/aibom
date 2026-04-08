# AI BOM — Technical Overview

## 1. High-Level Architecture

| Layer | Responsibilities |
|-------|------------------|
| **CLI (Typer + Rich)** | Provides `analyze`, `report`, `watch`, `diff`, `benchmark`, `kb`, `cache`, and `plugin` commands. Loads `.env` configuration, validates options, configures logging, and renders Rich summaries. |
| **Config + Manifest** | Loads `manifest.json` from `AIBOM_MANIFEST_PATH`, packaged defaults, or CWD fallback. Environment overrides for DB path, SHA, and post URL. |
| **Knowledge Base Loader** | Resolves a local DuckDB catalog path (from env or manifest), verifies SHA-256, and returns the validated file path. |
| **Source Resolver** | Distinguishes local paths, container images, and remote repos. For images, runs tiered container extraction (Docker/Podman/nerdctl/Buildah/Skopeo/Crane/tarball). For remote repos, clones via platform adapters (GitHub/GitLab/Bitbucket). |
| **Scan Pipeline** | Four-stage orchestrator: Scan → Cross-Ref → Agentic → Assemble. Runs all registered scanners, resolves env-var references, classifies all candidates via the mandatory LLM agent, and applies filtering. |
| **Scanners (21 built-in)** | Pluggable scanner registry with `__init_subclass__` auto-registration. Each scanner implements `scan(ctx) → (components, relationships)`. Covers models, dependencies, secrets, vulnerabilities, MCP, ML lifecycle, cloud, CI/CD, deployments, containers, data files, config, skills, workflows, and more. |
| **Cross-Reference Index** | Builds env-var and package indexes across all scanned files. Resolves env-var references to concrete values (model names, API keys, endpoints) by correlating source code with `.env`, `docker-compose.yaml`, Helm values, and Terraform files. |
| **Agentic Layer** | Mandatory LLM-powered classification via Deep Agents + LangChain. Every scanner candidate is confirmed or rejected by the agent. Two-tier classification (simple/complex candidates), locality-aware batching, sub-agent dispatch, structured output parsing, content-hash caching, and circuit breaker. |
| **Policy Engine** | YAML-driven pass/fail gates: max risk score, required/blocked component types, required fields, and custom rules. |
| **Reporters (10 formats)** | Pluggable reporter registry. Built-in: Plaintext, JSON, CycloneDX (1.6), SARIF (2.1.0), SPDX (3.0), HTML dashboard, Markdown, CSV, JUnit. Plus live FastAPI API server. |
| **Custom Catalog** | Loads `.aibom.yaml`/`.yml`/`.json`: custom component entries, base-class rules, exclude patterns, relationship hints, and custom relationship types. |

## 2. Execution Flow

1. **Startup** — Load `.env` (from `AIBOM_ENV_FILE` or local defaults), parse CLI args, validate options, configure logging.
2. **Knowledge Base** — Call `ensure_local_database()` to resolve and verify the DuckDB catalog with SHA-256 validation.
3. **Custom Catalog** — Load `.aibom.yaml` (from `--custom-catalog` or auto-discovered). Merge custom entries, register excludes, pass base-class rules and relationship hints.
4. **Source Acquisition** — For each source: detect container image vs local path vs remote repo. Extract container images via the tiered extractor. Clone remote repos via platform adapters.
5. **Scan Pipeline Stage 1: Scan** — Run all registered scanners against the source files. Each scanner produces components and relationships. File I/O uses async caching for performance.
6. **Scan Pipeline Stage 2: Cross-Ref** — Build env-var and package indexes. Resolve env-var references (`os.getenv("MODEL_NAME")`) to concrete values by correlating across files, `.env`, docker-compose, Helm, and Terraform.
7. **Scan Pipeline Stage 3: Agentic** — Classify all candidates into simple/complex tiers. Run locality-aware batched LLM classification. The agent confirms, rejects, reclassifies, or enriches each candidate using tools (`read_file_lines`, `search_package_info`). Apply structured output (enrichments, new components, removals, risk findings). Cache results by content hash.
8. **Scan Pipeline Stage 4: Assemble** — Apply `--strict` filtering (drop `needs_agentic` items), collect counts and timing, build the final `PipelineResult`.
9. **Reporting** — Route `PipelineResult` to the selected reporter. Convert container temp paths to container-style paths. Build per-source summaries and run metadata.
10. **Post-Processing** — Optionally POST the JSON report with retries, run policy checks, display compliance advisories, render Rich console summary.

## 3. Scanner Architecture

Scanners use a registry pattern with `__init_subclass__` auto-registration:

```
BaseScanner
├── ModelDetector          — AI model usage (registry lookup, import context)
├── DependencyScanner      — Package manifests (requirements.txt, package.json, go.mod, etc.)
├── SecretDetector         — Hardcoded API keys, tokens (via detect-secrets)
├── VulnScanner            — CVE lookups via OSV.dev API
├── McpDetector            — MCP server/client usage patterns
├── SkillDetector          — AI skill definitions and registrations
├── MLLifecycleDetector    — Training runs, datasets, hyperparameters, model artifacts
├── CloudScanner           — AWS/GCP/Azure AI service resource references
├── CICDScanner            — CI/CD pipeline AI asset references
├── DeploymentDetector     — IaC deployment patterns (Terraform, CloudFormation, Helm, K8s)
├── ContainerScanner       — Dockerfile/Containerfile AI base image detection
├── DataFileScanner        — Model files (.safetensors, .gguf, .onnx) and dataset files (.parquet, .arrow)
├── ModelFileScanner       — Model artifact files on disk
├── ConfigScanner          — Framework config files (LangGraph JSON, pyproject.toml AI sections)
├── MultiLanguageScanner   — JS/TS, Java, Go, Rust, Ruby, C#, PHP (via tree-sitter)
├── EnvVarResolver         — Environment variable extraction from source code
├── KBEnrichmentScanner    — DuckDB knowledge base symbol matching and enrichment
├── WorkflowScanner        — Workflow/call-graph context attachment
├── WorkspaceDepScanner    — Monorepo local path dependency detection
├── ShadowAIDetector       — Shadow AI / unmanaged AI usage patterns
└── (Plugin scanners via entry point: aibom.scanners)
```

Each scanner receives a `ScanContext` with paths, config, and shared state. Scanners run in registration order. The file cache (`file_cache.py`) provides async-compatible read caching to avoid redundant I/O across scanners.

## 4. Key Modules

| Module | Purpose |
|--------|---------|
| `cli.py` | CLI entry point — orchestrates analysis, loads config, renders outputs. |
| `scan_pipeline.py` | Four-stage pipeline orchestrator (Scan → Cross-Ref → Agentic → Assemble). |
| `scanners/__init__.py` | Scanner registry and `run_scanners()` dispatcher. |
| `scanners/base.py` | `BaseScanner` abstract class with `__init_subclass__` auto-registration. |
| `cross_ref.py` | Env-var and package index builder, component resolver, external repo dep detection. |
| `cst_parser.py` | LibCST-based Python parser for assignments, decorators, annotations, imports, class definitions, and inline annotations. |
| `scanners/multi_language_scanner.py` | Tree-sitter-based parser for JS/TS, Java, Go, Rust, Ruby, C#, PHP. |
| `custom_catalog.py` | `.aibom.yaml` loader — custom entries, base-class rules, excludes, relationship hints. |
| `catalog_db.py` | DuckDB access layer for catalog lookup with custom entry merging. |
| `db_loader.py` | Manifest/env path resolution and SHA-256 verification. |
| `agentic/agent.py` | Agentic enrichment core — candidate classification, locality-aware batching, sub-agent dispatch, circuit breaker, content-hash caching. |
| `agentic/tools.py` | LangChain tools for file reading, import analysis, code search, package registry queries (PyPI/npm/Go), and repo triage (directory tree, file snippets). |
| `agentic/middleware.py` | Structured output parser — extracts enrichments, new components, removals, risk findings from LLM JSON responses. |
| `agentic/prompts.py` | System prompts for the agentic enrichment agent. |
| `llm_factory.py` | Centralized `build_chat_model()` — provider resolution, parameter mapping for OpenAI/Azure/Bedrock/Ollama/etc. |
| `llm_client.py` | Lightweight LLM client for semantic parsing (model name extraction). |
| `scanners/container_extractor.py` | Tiered container source extraction (Docker/Podman/nerdctl/Buildah/Skopeo/Crane/tarball). |
| `scanners/model_detector.py` | Model detection via LiteLLM catalog, HuggingFace Hub, and import context. |
| `scanners/dependency_scanner.py` | Multi-format dependency parsing (pip, npm, Maven, Go, Rust, Ruby, C#, PHP). Emits all manifest packages as candidates; `KNOWN_AI_PACKAGES` is a hint, not a gate. |
| `scanners/secret_detector.py` | Secret detection via Yelp `detect-secrets` integration. |
| `scanners/vuln_scanner.py` | Vulnerability scanning via OSV.dev API. |
| `scanners/env_var_resolver.py` | Multi-language env-var extraction (Python, JS/TS, Go, Java, Ruby, C#, Rust, PHP). |
| `scanners/deployment_detector.py` | IaC detection (Terraform, CloudFormation, Azure ARM/Bicep, Helm, K8s manifests). |
| `policy.py` | YAML policy engine — pass/fail gates for CI/CD. |
| `compliance.py` | EU AI Act, OWASP Agentic, NIST AI RMF compliance mappings. |
| `diff.py` | Two-report diff engine (added/removed/changed components). |
| `benchmark.py` | Precision/recall/F1 benchmarking against ground-truth YAML. |
| `watch.py` | File-system polling + debounced re-scan with delta reporting. |
| `risk.py` | Risk scoring and severity classification. |
| `reporters/*.py` | Output format implementations (10 built-in + plugin entry point). |
| `plugins.py` | Plugin discovery via Python entry points (`aibom.scanners`, `aibom.reporters`). |
| `platform_adapters.py` | GitHub/GitLab/Bitbucket API adapters for repo discovery. |
| `multi_repo.py` | Multi-repo scan orchestration with parallel execution. |
| `incremental.py` | Commit-SHA keyed scan caching (`--skip-unchanged`). |
| `repo_triage.py` | Agent-first repository triage — tool-using agent explores repos (directory tree, file reading, codebase search, package registry) to decide deep-scan vs skip. |
| `report_sender.py` | POST JSON reports with retry/backoff. |
| `workflow_analyzer.py` | AST-based function index and call graph for workflow context. |
| `models/enums.py` | `AIComponentType` (24 types), `Severity`, `Confidence` enums. |
| `models/scan.py` | `AIComponent`, `ComponentRelationship`, `RiskFlag`, `ScanContext`, `ScanResult` dataclasses. |

## 5. Component Types

The analyzer recognizes 24 AI component types:

| Type | Description |
|------|-------------|
| `model` | AI/ML models (LLMs, embeddings, classifiers). |
| `agent` | Autonomous AI agents. |
| `tool` | Tools available to agents. |
| `mcp_server` | Model Context Protocol servers. |
| `mcp_client` | Model Context Protocol clients. |
| `embedding` | Embedding models and services. |
| `vector_store` | Vector databases and stores. |
| `dataset` | Datasets used in ML pipelines. |
| `prompt` | Prompt templates and chains. |
| `guardrail` | Safety and content filters. |
| `memory` | Agent memory and state stores. |
| `retriever` | RAG retrievers and search components. |
| `training_run` | Model training/finetuning invocations. |
| `hyperparameter` | ML hyperparameter configurations. |
| `model_artifact` | Serialized model files. |
| `experiment_tracker` | Experiment tracking (MLflow, W&B, etc.). |
| `model_registry` | Model registries and catalogs. |
| `data_versioning` | Dataset versioning tools (DVC, etc.). |
| `ml_pipeline` | ML pipeline orchestrators. |
| `skill` | AI skills and capabilities. |
| `observability` | AI observability tools (LangSmith, Langfuse, etc.). |
| `secret` | Hardcoded API keys, tokens, credentials. |
| `dependency` | AI framework dependencies from package manifests. |
| `other` | Unclassified AI-related components. |

## 6. Output Formats

| Format | Reporter | Standard |
|--------|----------|----------|
| Plaintext | `plaintext_reporter.py` | — |
| JSON | `json_reporter.py` | AIBOM JSON schema |
| CycloneDX | `cyclonedx_reporter.py` | CycloneDX 1.6 (ML-BOM profile) |
| SARIF | `sarif_reporter.py` | SARIF v2.1.0 |
| SPDX | `spdx_reporter.py` | SPDX 3.0 (AI + Dataset profiles) |
| HTML | `html_reporter.py` | Interactive dashboard |
| Markdown | `markdown_reporter.py` | GitHub-flavored Markdown |
| CSV | `csv_reporter.py` | Flat CSV |
| JUnit | `junit_reporter.py` | JUnit XML |
| API | `api_handler.py` + `api/server.py` | FastAPI REST endpoints |

Reporters use a registry pattern. Custom reporters can be added via the `aibom.reporters` entry point group.

## 7. Agentic Enrichment Architecture

The agentic layer is the mandatory final classifier. Scanners generate candidates; the agent is the single source of truth.

1. **Candidate triage** — All candidates are split into "simple" (registry-confirmable: known model IDs, manifest dependencies) and "complex" (ambiguous type, missing model name, multi-file reasoning needed).
2. **Locality-aware batching** — Candidates are grouped by parent directory, then split into batches of configurable size (default 15). This provides coherent code context per batch.
3. **Agent tools** — The agent uses `read_file_lines` (inspect source), `search_package_info` (query PyPI/npm/Go registries for dependency metadata), and code context to make decisions.
4. **Structured output** — Each batch returns JSON with `enriched_components`, `new_components`, `remove_components`, and `risk_findings`. The agent must explicitly confirm or remove every candidate.
5. **Content-hash caching** — Each component's cache key is derived from its file path, line number, name, type, and surrounding code content. Unchanged components reuse cached LLM results.
6. **Circuit breaker** — After N consecutive batch failures, remaining batches are skipped to prevent runaway costs.
7. **Timeout** — Per-batch wall-clock timeout with graceful degradation.

## 8. Cross-Reference Resolution

The cross-ref stage builds two indexes:

- **Env-var index** — Maps environment variable names to their values by scanning `.env`, `docker-compose.yaml`, Helm `values.yaml`, Terraform `*.tfvars`, and CI/CD pipeline definitions.
- **Package index** — Maps package names to installed versions by parsing `requirements.txt`, `poetry.lock`, `package-lock.json`, `go.sum`, etc.

Components referencing env vars (e.g., `os.getenv("MODEL_NAME")`) are resolved to concrete values using the env-var index, with full provenance tracking.

## 9. Knowledge Base Strategy

- Manifest keys: `duckdb_sha256` and `duckdb_file`, with env overrides (`AIBOM_DB_PATH`, `AIBOM_DB_SHA256`).
- `duckdb_file` is resolved relative to `manifest.json` when not absolute.
- Catalog lookup uses suffix matching on fully qualified symbol names.
- The KB is used for enrichment: matching detected symbols against known AI framework components to add category, framework, and metadata.
- Custom entries from `.aibom.yaml` are merged with lowest precedence (DuckDB > supplemental > custom).
- KB management via CLI: `kb download`, `kb check`, `kb info`, `kb verify`, `kb request`.

## 10. Report Submission

- `--post-url` triggers a POST of the JSON report. `AIBOM_POST_URL`, `AIBOM_POST_TIMEOUT`, and `AIBOM_POST_VERIFY_TLS` mirror CLI options.
- Payload includes `run_id`, `analyzer_version`, `submitted_at`, `source_kind`, `sources`, and the report body.
- Requests use `x-cisco-ai-defense-tenant-api-key` when `--ai-defense-api-key` / `AI_DEFENSE_API_KEY` is set, with retry/backoff for 429/5xx.

## 11. Plugin System

Plugins are discovered via Python entry points:

- `aibom.scanners` — Custom scanner classes (must subclass `BaseScanner`).
- `aibom.reporters` — Custom reporter classes (must subclass `BaseReporter`).

The `plugin list` command shows all discovered plugins.
