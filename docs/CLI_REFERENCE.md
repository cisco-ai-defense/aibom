# CLI Reference

Complete reference for the `cisco-aibom` command-line interface.

## Global Options

These options apply to all commands.

| Option | Env Var | Default | Description |
|--------|---------|---------|-------------|
| `--log-level` | `AIBOM_LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `--help` | — | — | Show help and exit. |

## `cisco-aibom analyze`

Scan source code, container images, or repositories to produce an AI Bill of Materials.

```bash
cisco-aibom analyze [OPTIONS] [SOURCES]...
```

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `SOURCES` | One or more source directories, file paths, or container image references to analyze. |

### Source Options

| Option | Env Var | Description |
|--------|---------|-------------|
| `--images-file`, `-f` | — | Path to a JSON file containing a list of container images to scan. |
| `--repos-file` | — | File listing repo paths or git URLs (JSON array or newline-delimited). |
| `--discover-repos` | — | Treat each positional source as a parent directory and discover git repos underneath. |
| `--github-org` | `AIBOM_GITHUB_ORG` | Discover and scan repos from a GitHub org or user. |
| `--gitlab-group` | `AIBOM_GITLAB_GROUP` | Discover and scan repos from a GitLab group. |
| `--bitbucket-project` | `AIBOM_BITBUCKET_PROJECT` | Discover and scan repos from a Bitbucket workspace/project. |
| `--platform-token` | `AIBOM_PLATFORM_TOKEN` | Auth token for GitHub/GitLab/Bitbucket API access. |
| `--repo-filter` | — | Filter discovered repos by name substring. |
| `--repo-topic` | — | Filter discovered repos by topic/tag. |
| `--max-repos` | — | Max repos when using discovery (sorted by last push, most recent first). |
| `--parallel-repos` | — | Number of repositories to scan in parallel (default `1`). |
| `--skip-unchanged` | — | Skip repos whose HEAD is unchanged since last cached scan. Org-cache writes default to `~/.aibom/cache/org` and still read legacy locations for compatibility. |

### Output Options

| Option | Env Var | Description |
|--------|---------|-------------|
| `--output-format`, `-o` | — | Output format: `plaintext`, `json`, `api`, `cyclonedx`, `sarif`, `spdx`, `html`, `markdown`, `csv`, `junit` (default `plaintext`). |
| `--output-file`, `-O` | — | Path to write the report (required for file-based formats). |
| `--validate` | — | Validate output against the format's schema and report errors. |
| `--show-summary` / `--no-show-summary` | — | Display a Rich summary table after analysis (default on). |
| `--timing` | — | Print per-stage and per-scanner timing breakdown. |
| `--progress` / `--no-progress` | — | Show live per-stage and per-scanner progress. Defaults to auto for interactive terminals. |

### Report Submission

| Option | Env Var | Description |
|--------|---------|-------------|
| `--post-url` | `AIBOM_POST_URL` | HTTP endpoint to POST the JSON report to. |
| `--ai-defense-api-key` | `AI_DEFENSE_API_KEY` | API key for Cisco AI Defense endpoints (sent as `x-cisco-ai-defense-tenant-api-key`). |
| `--post-timeout` | `AIBOM_POST_TIMEOUT` | Timeout in seconds for POSTing the report (default `30`). |
| `--post-verify-tls` / `--no-post-verify-tls` | `AIBOM_POST_VERIFY_TLS` | Verify TLS certificates when POSTing (default on). |

### LLM / Agentic Options

| Option | Env Var | Description |
|--------|---------|-------------|
| `--llm-model` | `AIBOM_LLM_MODEL` | **Required.** LLM model name (e.g. `gpt-5.4`, `us.anthropic.claude-sonnet-4-20250514-v1:0`). The LLM agent classifies every scanner candidate and requires `cisco-aibom[agentic]` plus any provider-specific integration extra. |
| `--llm-provider` | `AIBOM_LLM_PROVIDER` | LangChain provider name: `openai`, `azure_openai`, `bedrock`, `anthropic`, `google_genai`, `ollama`, etc. Inferred from the model name if not set. |
| `--llm-api-key` | `AIBOM_LLM_API_KEY` | LLM API key. Optional for local LLMs and AWS Bedrock. |
| `--llm-api-base` | `AIBOM_LLM_API_BASE` | LLM API base URL. |
| `--llm-api-version` | `AIBOM_LLM_API_VERSION` | LLM API version (required for Azure OpenAI). |
| `--agentic-batch-size` | — | Max components per LLM invocation (default `5`). |
| `--agentic-concurrency` | — | Max parallel agentic LLM batches (default `1`). |
| `--agentic-fast-model` | — | Cheaper/faster model for simple confirmations (e.g. dependency checks). |
| `--agentic-timeout` | — | Wall-clock timeout in seconds per agentic batch (default `120`). |
| `--galileo` / `--no-galileo` | `AIBOM_GALILEO_ENABLED` | Emit sampled, privacy-preserving agentic quality telemetry using a sanitized Agent-span hierarchy (disabled by default). Requires the `observability` extra, Galileo credentials and destination variables, and explicit public-cloud approval. |
| `--galileo-sample-rate` | `AIBOM_GALILEO_SAMPLE_RATE` | Deterministic sampling rate from `0.0` to `1.0` for sanitized Galileo batch agent traces (default `1.0` when enabled). |
| `--galileo-full-trajectory` / `--no-galileo-full-trajectory` | — | **Diagnostic only.** Separately emits raw prompts, responses, tool I/O, and exact identities. Disabled by default and activated only with `--galileo` plus every full-content, exact-identity, and public-cloud approval gate. |

See the [Galileo observability guide](../aibom/docs/galileo-observability.md)
for required environment variables, privacy controls, deployment gates, and
the distinction between sanitized telemetry and raw full-trajectory diagnostics.

### Analysis Options

| Option | Description |
|--------|-------------|
| `--custom-catalog` | Path to a custom catalog file (`.aibom.yaml`/`.yml`/`.json`). Auto-discovered from the source directory if not set. |
| `--container-extraction-tier` | Force a container extraction tier: `auto`, `syft`, `docker`, `podman`, `nerdctl`, `buildah`, `crane`, `skopeo`, `tarball` (default `auto`). |
| `--keep-extractions` / `--no-keep-extractions` | Keep extracted container filesystems on disk after the run for inspection, and force retention even on large multi-image runs. Extractions are always kept long enough for cross-source correlation regardless of this flag; runs of more than 20 sources otherwise clean them eagerly to protect temp space. Off by default. |
| `--severity` | Minimum severity of findings to include (default `info`). |
| `--strict` | Only emit high-confidence detections; suppress items needing agentic reasoning. |
| `--fail-on` | Exit non-zero if risk severity meets or exceeds: `critical`, `high`, `medium`, `low`. |
| `--policy` | Path to a YAML policy file. Exits `1` if the policy does not pass. |
| `--compliance` | Advisory compliance mapping: `eu-ai-act`, `owasp-agentic`, `nist-ai-rmf`, or `all`. |
| `--cache-dir` | Shared cache root. Defaults to `~/.aibom/cache` and stores `scan`, `agentic`, `org`, `model`, and `packages` caches beneath it. |
| `--include-code-snippets` / `--no-code-snippets` | Include raw code snippets in per-finding decision annotations. Off by default. |
| `--component-summary` / `--no-component-summary` | When `--output-format=json`, include a flat `component_summary` key in the report listing each non-test component as `{component_type, name, file_path, line_number}`, grouped by source and sorted by `(component_type, name)`. Intended for quick human review and demos; the full structured output is unchanged. Off by default. |
| `--no-network` | Disable anonymous package-liveness freshness requests while retaining local snapshot fields. Also available as `AIBOM_NO_NETWORK`. |
| `--liveness-only-snapshot` | Use only package-liveness fields frozen into the selected knowledge base. Also available as `AIBOM_LIVENESS_ONLY_SNAPSHOT`. |

See [Package liveness freshness](package-freshness.md) for the snapshot,
privacy, retry, and output-field contract.

### Examples

```bash
# Basic scan with JSON output
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# Container image scan
cisco-aibom analyze my-app:latest -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# HTML dashboard
cisco-aibom analyze ./my-app -o html -O dashboard.html \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# CycloneDX BOM
cisco-aibom analyze ./my-app -o cyclonedx -O bom.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# Multi-repo scan with discovery
cisco-aibom analyze /path/to/repos --discover-repos -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# GitHub org scan
cisco-aibom analyze --github-org my-org --platform-token $GITHUB_TOKEN \
  --max-repos 50 --parallel-repos 4 -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY

# Policy gate in CI
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY --policy policy.yaml --fail-on high

# Timing breakdown
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY --timing

# Strict mode (minimal LLM calls)
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY --strict

# Compliance check
cisco-aibom analyze ./my-app -o json -O report.json \
  --llm-model gpt-5.4 --llm-api-key $OPENAI_API_KEY --compliance eu-ai-act
```

---

## `cisco-aibom report`

Show or upload a previously generated JSON report.

```bash
cisco-aibom report REPORT_FILE
cisco-aibom report show REPORT_FILE [--raw-json]
cisco-aibom report upload REPORT_FILE --format json --post-url URL [OPTIONS]
```

The JSON reporter writes `aibom_analysis.metadata.report_schema_version = "1"`. Unversioned legacy JSON reports are still accepted for `report upload`; the CLI warns and synthesizes the current schema version before submitting.

### `report show`

| Argument / Option | Description |
|-------------------|-------------|
| `REPORT_FILE` | Path to a JSON report file. |
| `--raw-json` | Display the raw JSON with syntax highlighting before the summary. |

### `report upload`

| Option | Env Var | Description |
|--------|---------|-------------|
| `--format` | — | Upload format. Only `json` is currently supported. |
| `--post-url` | `AIBOM_POST_URL` | HTTP endpoint to POST the JSON report to. Required for upload. |
| `--ai-defense-api-key` | `AI_DEFENSE_API_KEY` | API key sent as `x-cisco-ai-defense-tenant-api-key`. |
| `--post-timeout` | `AIBOM_POST_TIMEOUT` | Timeout in seconds for POSTing the report (default `30`). |
| `--post-verify-tls` / `--no-post-verify-tls` | `AIBOM_POST_VERIFY_TLS` | Verify TLS certificates when POSTing (default on). |

### Examples

```bash
cisco-aibom report report.json
cisco-aibom report show report.json --raw-json
cisco-aibom report upload report.json --format json \
  --post-url https://example.invalid/aibom/reports \
  --ai-defense-api-key $AI_DEFENSE_API_KEY
```

---

## `cisco-aibom watch`

Poll directories for file-system changes and re-run the deterministic scan pipeline, printing component deltas. This command does not invoke the agentic classification stage; use `analyze` for the full LLM-backed pipeline.

```bash
cisco-aibom watch [OPTIONS] SOURCES...
```

| Argument / Option | Default | Description |
|-------------------|---------|-------------|
| `SOURCES` | — | Source directories or files to poll and re-scan. |
| `--interval` | `2.0` | Seconds between file-system polls. |
| `--debounce` | `0.5` | Seconds to wait after a change before re-scanning. |

### Example

```bash
cisco-aibom watch ./my-app --interval 5 --debounce 1
```

---

## `cisco-aibom diff run`

Compare two AIBOM JSON scan reports and display added, removed, and changed components.

```bash
cisco-aibom diff run [OPTIONS] OLD_REPORT NEW_REPORT
```

| Argument / Option | Default | Description |
|-------------------|---------|-------------|
| `OLD_REPORT` | — | Path to the older JSON report. |
| `NEW_REPORT` | — | Path to the newer JSON report. |
| `--format`, `-f` | `table` | Output format: `table`, `json`, or `markdown`. |

### Example

```bash
cisco-aibom diff run report-v1.json report-v2.json
cisco-aibom diff run report-v1.json report-v2.json --format json
```

---

## `cisco-aibom benchmark run`

Compare scan output against a ground-truth YAML file to measure detection accuracy.

```bash
cisco-aibom benchmark run [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--gt` | Path to ground-truth YAML file (required). |
| `--scan` | Path to scan report JSON (required). |
| `--strict-names` | Match listed names (case-insensitive) when the ground-truth provides names. |
| `--format` | Output format: `table` (default), `json`, or `csv`. |

### Example

```bash
cisco-aibom benchmark run --gt ground-truth.yaml --scan report.json
cisco-aibom benchmark run --gt ground-truth.yaml --scan report.json --format json
```

### Ground-truth YAML format

```yaml
components:
  - type: model
    count: 3
    names:
      - gpt-5.4
      - text-embedding-ada-002
      - gemma-2b
  - type: agent
    count: 2
  - type: tool
    count: 5
```

---

## `cisco-aibom kb`

Manage the AIBOM knowledge base (DuckDB catalog).

### `kb download`

```bash
cisco-aibom kb download [OPTIONS]
```

| Option | Env Var | Description |
|--------|---------|-------------|
| `--version`, `-v` | — | Specific KB version to download (latest if omitted). |
| `--url` | `CISCO_AIBOM_MANIFEST_URL` | Manifest URL. Required: pass `--url` or set the env var. No default is shipped. |

### `kb check`

Check if a newer KB version is available.

```bash
cisco-aibom kb check
```

### `kb info`

Display information about the locally installed KB.

```bash
cisco-aibom kb info
```

For a schema-v2 manifest, this offline command also displays
`schema_version`, `vocabulary_version`, and the 20-concept vocabulary. See
[Schema-v2 concept vocabulary](concept-vocabulary.md).

### `kb verify`

Verify the integrity of the locally installed KB (SHA-256 checksum).

```bash
cisco-aibom kb verify
```

### `kb request`

Request a KB build for a specific SDK version.

```bash
cisco-aibom kb request [OPTIONS]
```

| Option | Env Var | Description |
|--------|---------|-------------|
| `--sdk` | — | SDK name, e.g. `langchain`, `openai` (required). |
| `--version`, `-v` | — | SDK version to request (required). |
| `--language`, `-l` | — | Programming language (default `python`). |
| `--api-key` | `CISCO_AI_DEFENSE_API_KEY` | Cisco AI Defense tenant API key (required). |
| `--api-base` | `CISCO_AI_DEFENSE_API_BASE` | Regional Cisco AI Defense API host (required). Follows the same pattern as `AIBOM_POST_URL`: `https://api.security.cisco.com` (US), `https://api.eu.security.cisco.com` (EU), `https://api.apj.security.cisco.com` (APJ), `https://api.uae.security.cisco.com` (UAE). No default is shipped. |

### `kb request-status`

Check the status of a KB build request previously submitted via `kb request`.

```bash
cisco-aibom kb request-status REQUEST_ID [OPTIONS]
```

| Argument / Option | Env Var | Description |
|-------------------|---------|-------------|
| `REQUEST_ID` | — | Request ID returned by `kb request` (required). |
| `--api-key` | `CISCO_AI_DEFENSE_API_KEY` | Cisco AI Defense tenant API key (required). |
| `--api-base` | `CISCO_AI_DEFENSE_API_BASE` | Regional Cisco AI Defense API host (required). Same regional hosts as `kb request` (e.g. `https://api.security.cisco.com`, `https://api.eu.security.cisco.com`). No default is shipped. |

### `kb list-requests`

List all pending KB build requests for the authenticated tenant.

```bash
cisco-aibom kb list-requests [OPTIONS]
```

| Option | Env Var | Description |
|--------|---------|-------------|
| `--api-key` | `CISCO_AI_DEFENSE_API_KEY` | Cisco AI Defense tenant API key (required). |
| `--api-base` | `CISCO_AI_DEFENSE_API_BASE` | Regional Cisco AI Defense API host (required). Same regional hosts as `kb request` (e.g. `https://api.security.cisco.com`, `https://api.eu.security.cisco.com`). No default is shipped. |

---

## `cisco-aibom cache`

Manage AIBOM cache entries under the shared cache root (`~/.aibom/cache` by default).

### `cache clear`

Remove all cached scan results and (optionally) the agentic enrichment cache.

```bash
cisco-aibom cache clear [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--cache-dir` | `~/.aibom/cache` | Cache root directory. |
| `--include-agentic` / `--no-agentic` | Include | Also clear the agentic enrichment cache. |

### `cache list`

List cached entries for a specific cache family.

```bash
cisco-aibom cache list [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--type` | `scan` | Cache family: `scan`, `agentic`, `org`, `model`, `packages`. |
| `--cache-dir` | `~/.aibom/cache` | Cache root directory. |

### `cache get`

Inspect a specific cache entry by type.

```bash
cisco-aibom cache get CACHE_TYPE ENTRY_REF [OPTIONS]
```

| Argument / Option | Description |
|-------------------|-------------|
| `CACHE_TYPE` | Cache family: `scan`, `agentic`, `org`, `model`, `packages`. |
| `ENTRY_REF` | Entry id, prefix, or logical reference. |
| `--cache-dir` | Cache root directory (default `~/.aibom/cache`). |
| `--sha` | Commit SHA for `org` cache lookups. |
| `--model-id` | Optional model id filter for `model` cache lookups. |
| `--raw-json` | Print the raw cache payload instead of a summary. |

### Cache examples

```bash
cisco-aibom cache list --type scan
cisco-aibom cache list --type agentic
cisco-aibom cache get scan 0123456789ab
cisco-aibom cache get org /path/to/repo --sha deadbeef
```

---

## `cisco-aibom plugin list`

List all discovered plugins (entry points, MCP servers, plugin manifests).

```bash
cisco-aibom plugin list
```

---

## Environment Variable Summary

| Variable | Description |
|----------|-------------|
| `AIBOM_ENV_FILE` | Path to a `.env` file to load. Falls back to `./.env` if present. |
| `AIBOM_LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `AIBOM_LLM_MODEL` | **Required.** LLM model name for agentic classification. |
| `AIBOM_LLM_PROVIDER` | LangChain provider name. |
| `AIBOM_LLM_API_KEY` | LLM API key. |
| `AIBOM_LLM_API_BASE` | LLM API base URL. |
| `AIBOM_LLM_API_VERSION` | LLM API version (Azure OpenAI). |
| `AIBOM_POST_URL` | HTTP endpoint to POST the JSON report to. |
| `AIBOM_POST_TIMEOUT` | POST timeout in seconds. |
| `AIBOM_POST_VERIFY_TLS` | Verify TLS for report POST (`true`/`false`). |
| `AI_DEFENSE_API_KEY` | Cisco AI Defense tenant API key. |
| `AIBOM_GITHUB_ORG` | GitHub org/user for repo discovery. |
| `AIBOM_GITLAB_GROUP` | GitLab group for repo discovery. |
| `AIBOM_BITBUCKET_PROJECT` | Bitbucket workspace/project for repo discovery. |
| `AIBOM_PLATFORM_TOKEN` | Auth token for GitHub/GitLab/Bitbucket APIs. |
| `AIBOM_DB_PATH` | Override path to the DuckDB catalog file. |
| `AIBOM_DB_SHA256` | Expected SHA-256 checksum for the DuckDB catalog. |
| `AIBOM_MANIFEST_PATH` | Override path to `manifest.json`. |
| `AIBOM_NO_NETWORK` | Disable package-liveness freshness requests. |
| `AIBOM_LIVENESS_ONLY_SNAPSHOT` | Use only package-liveness snapshot fields. |
| `CISCO_AIBOM_FRESHNESS_URL` | Optional package-freshness endpoint override. No default. |
| `CISCO_AI_DEFENSE_API_KEY` | API key for KB request commands. |
| `CISCO_AI_DEFENSE_API_BASE` | Regional API base URL for KB request commands. Same regional hosts as `AIBOM_POST_URL` (e.g. `https://api.security.cisco.com`, `https://api.eu.security.cisco.com`). No default. |
| `CISCO_AIBOM_MANIFEST_URL` | KB manifest URL for `kb download` / `kb check`. No default. |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID for cloud scanning. |
