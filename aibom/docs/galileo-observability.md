# Galileo observability for agentic scans

This guide describes the optional Galileo integration implemented by AI BOM.
The sanitized production path is **observe-only and fail-open**: it does not
change findings, block a scan, or configure Galileo resources, and it never
sends repository content. A separate live diagnostic callback can send full
raw trajectories, and the evaluation path is pseudonymous by default. Exact
evaluation rows or full trajectories are available only through separate
identity, full-content, trajectory, immutable-destination, and public-cloud
egress approval gates.

The implementation is in
[`agentic_telemetry.py`](../src/aibom/agentic_telemetry.py). Entity-level
evaluation helpers are in
[`decision_evaluation.py`](../src/aibom/decision_evaluation.py). The versioned
decision-suite schema, approved full-content gate, custom-function experiment
entry point, and optional async callback are in
[`galileo_evaluation.py`](../src/aibom/galileo_evaluation.py).

## Install and enable

Install the `observability` extra together with the agentic runtime and an LLM
provider. For example:

```bash
uv tool install --python 3.13 \
  "cisco-aibom[agentic,observability,llm-openai]"
```

For a source checkout:

```bash
cd aibom
uv sync --extra agentic --extra observability --extra llm-openai
```

Configure the destination and opt in explicitly:

```bash
export GALILEO_API_KEY="<secret-manager-value>"
export GALILEO_CONSOLE_URL="https://app.galileo.ai"
export GALILEO_API_URL="https://api.galileo.ai"
export GALILEO_PROJECT="<project-name>"
export GALILEO_LOG_STREAM="<log-stream-name>"
export AIBOM_GALILEO_HMAC_KEY="<independent-high-entropy-secret>"
export AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD=true
export AIBOM_GALILEO_ENABLED=true
export AIBOM_GALILEO_SAMPLE_RATE=0.05

cisco-aibom analyze /path/to/repository \
  --llm-model gpt-5.4 \
  --galileo \
  --galileo-sample-rate 0.05 \
  -o json -O report.json
```

`AIBOM_GALILEO_SAMPLE_RATE` and `--galileo-sample-rate` accept values from
`0.0` through `1.0`. The CLI value takes precedence when supplied. Production
emission requires an explicit HTTPS `GALILEO_CONSOLE_URL`. For Galileo hosted
cloud, set it to exactly `https://app.galileo.ai` and also set
`AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD=true`. The extra egress flag is intentional:
`--galileo` enables telemetry, while this flag approves the hosted destination.
Every other console origin, non-root path, URL credential, HTTP URL, query, and
fragment is rejected; this integration has no private/custom-cloud mode. The SDK
normally derives `https://api.galileo.ai`; an explicit `GALILEO_API_URL` is
accepted only when it normalizes to that exact API origin. Explicitly disabling
TLS verification through `GALILEO_SSL_CONTEXT` also disables telemetry.
The configured project and log stream must already exist: AI BOM resolves them
before constructing a logger and does not rely on SDK resource creation.

Telemetry remains a no-op when it is disabled, an ordinary trace is unsampled,
the SDK or API key is missing, or the Galileo logger cannot initialize.
Backend and flush failures are caught so a scan continues. Invalid or
incomplete destination/project/log-stream configuration also disables
emission.

Raw telemetry is disabled by default. `--galileo` alone enables only sanitized
telemetry, even when all raw-approval environment variables are already
configured. A reviewed run must additionally pass
`--galileo-full-trajectory`. The raw callback then activates only when the
full-content, full-trajectory, exact-identity, immutable project/log-stream,
verified-TLS, and hosted-egress gates all pass. Callback construction failures
are fail-open: the affected invocation and sanitized telemetry continue.

The optional dependency is intentionally constrained to `galileo==2.4.0`
and `galileo-core==4.4.0`. The pre-ingestion validator checks the exact Galileo
2.4 and galileo-core 4.4.0 node/metrics envelope; a new SDK or core release must
pass the real-SDK hierarchy/privacy tests before either range is widened.

Sanitized hosted logger/resource setup and initial session attachment are each
bounded to a two-second operation budget by default.
`AIBOM_GALILEO_SETUP_BUDGET_S` may override that value from greater than zero
through 10 seconds. A timeout disables sanitized telemetry for the run and never
fails the scan. During shutdown, all queued flushes share one bounded two-second
drain deadline. Starting the drain closes the dispatcher to new traces; queued
flushes that have not started are dropped at the deadline. An SDK request
already in progress cannot be cancelled safely, so it may finish on the daemon
worker, but it cannot delay CLI shutdown or cause another queued trace to start.

### HMAC key requirements

`AIBOM_GALILEO_HMAC_KEY` is not sent as telemetry. It creates stable keyed
pseudonyms and a deterministic sampling cohort; it does not make low-entropy
identifiers anonymous if the key is weak or compromised.

- Generate and store it as a dedicated secret; do not reuse an API key.
- Use the same key across workers that must correlate the same logical scan.
- Rotation intentionally breaks pseudonym correlation and changes the sampled
  cohort.
- If it is omitted, AI BOM creates a process-local random key. Identifiers and
  sample selection will then not be stable across processes or runs.

## Sanitized production data contract

This contract applies only to the sanitized `AgenticTelemetry` path. It does
not apply to the separately gated live/evaluation full-trajectory callback.
Sanitized telemetry is deliberately content-free. AI BOM sends only
allowlisted labels and counters, per-call timing and token counts, status
values, safe version labels, and HMAC-SHA256 pseudonyms truncated to 24 hex
characters.

It does **not** send:

- source code, snippets, prompts, or LLM response text;
- repository, file, directory, or container names and paths;
- component names or raw component identifiers;
- tool arguments, tool results, search queries, or retrieved documents;
- environment-variable values, credentials, secrets, or tracebacks.

For every emitted content field, `input == redacted_input` and, where an output
exists, `output == redacted_output`. Unknown enum-like values are collapsed to
`other`; each callback-observed unknown tool invocation is labeled
`aibom.tool.other`, while aggregate fallback counters combine unknown tools. At
most 128 pseudonymous component identifiers are included in a batch trace.
Each emitted batch trace contains one sanitized
`aibom.agentic.classifier` Agent span around its attempt workflows. This
structural span gives Galileo an agent-typed hierarchy; it does not contain or
enable the separately gated raw LangChain trajectory.

The following data is observable:

| Record | Sanitized fields |
|---|---|
| Batch trace input | HMAC source ID; source-scoped pseudonymous batch/component IDs; stable per-component decision-chain IDs; attempt kind; batch number and size; component-type counts; language counts; tier; cache-hit flag |
| Batch metadata | HMAC source ID, attempt kind, analyzer version, provider, pseudonymous model ID, tier, batch size, cache-hit flag, telemetry-configuration digest in `prompt_version`, response-schema digest in `schema_version` |
| Batch output | Allowlisted decision totals; keep/enrich/remove/reclassify/discover counts sliced by original component type, language, and bounded heuristic-confidence bucket (`low` < 0.5, `medium` < 0.8, `high` ≥ 0.8, or `unknown`; discoveries use emitted dimensions); degraded-candidate count; failure category; schema-valid flag; middleware-guard flag; status |
| Batch agent span | One `aibom.agentic.classifier` parent per emitted batch trace, with `agent_type=classifier`; the same sanitized batch input/output, status, duration, and no additional metadata. It contains no prompt, response, tool I/O, or exact identity content. |
| Attempt workflow | Attempt number/kind; raw model-requested mutation counts; post-middleware final action counts; blocked-action deltas; aggregate allowlisted tool calls/errors/root denials/duration; recovered flag; status; duration |
| LLM span | One span per callback-observed model invocation, in start order, with HMAC call ID, provider, pseudonymous model ID, call start/duration, per-call tokens when supplied, status, and schema/decision-carrier flags. Only the terminal decision-bearing call carries raw mutation counts. A single `mode=aggregate` fallback is used only when callbacks are unavailable. |
| Tool span | One span per callback-observed tool invocation, in start order, with HMAC tool-call ID, allowlisted tool name, call start/duration, and status. An aggregate fallback is used only when callbacks are unavailable; authoritative aggregate guard/error counters remain on the workflow. |
| Source summary | Pseudonymous source ID and kind; `candidate_count_available`; exact post-secret-partition/post-dedup candidate count when available; exact Stage-3 output count; explicit `decision_boundary=agentic_stage_output`; post-assemble final BOM count; degraded count; agentic decisions; token counts; source-pipeline start/duration; status. Whole-scan cache hits set `candidate_count_available=false` instead of mislabeling final BOM size as a candidate count. A no-model skipped source records both deterministic boundaries and zero agentic decisions. |

Every non-sentinel model/deployment label is HMAC-pseudonymized, including
public model families and provider-prefixed labels. URL-like, absolute or
traversing path-like, and secret-shaped values collapse to `other` instead.
Analyzer versions are retained. Project and log-stream names configure the SDK
but are not placed in AIBOM trace payloads. Do not put customer, repository,
credential, or incident data in any of these configuration labels. HMAC
pseudonyms remain correlatable data and must still be covered by the
organization's telemetry policy.

The production payload therefore contains no raw confidential repository
content, but it remains **internal operational telemetry**. Provider family,
analyzer/telemetry/schema versions, type/language distributions, action/risk
counts, reliability outcomes, latency, token use, tool-use shape, and
correlatable pseudonyms can reveal system behavior. Keep the hosted project
access-restricted and do not use customer, repository, incident, username, or
credential text in project/log-stream configuration labels.

### Full-content diagnostic and evaluation modes

> **The raw trajectory callback can transmit full source-derived prompts,
> model responses, retrieved evidence, tool inputs and outputs, callback
> metadata, exception details, and exact repository/component/path identities.
> Do not enable it for a repository unless its data owner and security/privacy
> owner approve the content, destination, retention, access, residency, and
> downstream processing.**

The live diagnostic path is separate from the sanitized logger. It is selected
only when both `--galileo` and the explicit `--galileo-full-trajectory` option
are active. The trajectory option defaults to disabled, so `--galileo` alone
cannot activate the raw callback, and the CLI rejects the trajectory option
when `--galileo` is absent. The explicit CLI option supplies the
call-site approval only after all environment and immutable-destination gates
pass. The raw callback is not governed by `--galileo-sample-rate`: each actual
initial, retry, fallback, or structured-coercion agent invocation receives a
fresh callback and logger and flushes one raw trace when that chain completes or
reports an error. Cache hits and circuit-breaker skips do not create raw
callbacks. No callback is stored in module-global state, so later disabled
scans and concurrent batches cannot inherit or share it.

Raw logger construction and scan-session binding share the configured setup
budget for each invocation. Synchronous setup runs outside the asynchronous
batch event loop. A timeout is fail-open: the invocation continues without a
raw callback, and any callback produced after its deadline is hardened and
discarded instead of being attached late. Scan-session creation permits only
one remote request in flight; later invocations wait for that request and never
launch an overlapping create.

Raw callback loggers have SDK shutdown flushing and Agent Control disabled.
After the callback's one-shot asynchronous flush attempt, any trace still
retained by the SDK is cleared instead of being retried during interpreter
shutdown. The per-logger ingest client owned by the raw callback is then
closed; the shared Galileo API client, which the callback does not own, is not.
When raw mode is requested and all static approval, destination, and immutable
resource gates pass, direct file-tool access is restricted to the invocation's
scan roots so a captured trajectory cannot include files outside the approved
source set. An out-of-root tool request is denied and counted; this safety
boundary means raw diagnostic runs are not guaranteed to be behavior-identical
to ordinary or sanitized-only scans.

The hosted custom-function runner remains pseudonymous by default: exact inputs
stay out of Galileo's dataset, the application receives a deep copy from an
in-process registry, and Galileo receives random per-run entity/name/repository/
path/location/type tokens, stripped metadata, no evidence excerpts, and
pseudonymous experiment labels/tags. This controls Galileo egress only; any
remote model or network call made by the supplied application is caller-owned
egress and requires its own review. This is pseudonymization, not anonymization;
numeric outcomes and timing remain internal telemetry.

Set `exact_identities=True` only for a reviewed exact-row run. That path requires
the literal `approved_fixture=True` plus
`AIBOM_GALILEO_ALLOW_EXACT_IDENTITIES=true`. If the rows contain approved
evidence, `AIBOM_GALILEO_ALLOW_FULL_CONTENT=true` is independently required.
The optional full LangChain callback captures prompts, responses, tool I/O,
callback metadata, and exception details. The live diagnostic and any
evaluation that explicitly uses this callback require
`AIBOM_GALILEO_ALLOW_EXACT_IDENTITIES=true`,
`AIBOM_GALILEO_ALLOW_FULL_CONTENT=true`, and
`AIBOM_GALILEO_ALLOW_FULL_TRAJECTORY=true`. They also require immutable UUIDs in
`AIBOM_GALILEO_EVALUATION_PROJECT_ID` and
`AIBOM_GALILEO_EVALUATION_LOG_STREAM_ID`, construct the logger internally, and
force one new trace plus one flush per chain. The default pseudonymous
custom-function experiment and exact custom-function rows do not use this
callback; their narrower identity/evidence gates are described above.

Every networked evaluation or live raw diagnostic requires the exact hosted
console/API origins, verified TLS, and
`AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD=true`. Externally supplied loggers and
ingestion hooks are rejected because their destinations cannot be verified.
Evidence excerpts are bounded, repo-relative, and local-schema validated, but
their content is deliberately opaque: the data owner must review it for secrets
and PII before exact/full-content approval.

## Trace mapping

The following mapping describes the sanitized path only. Raw diagnostic spans
are emitted separately by Galileo's LangChain callback and are not constrained
by this schema.

One CLI invocation supplies a random scan-batch ID. Its HMAC token becomes the
external ID for a session named `aibom-agentic-scan-<UTC-timestamp>`. All
sampled sources and agentic batches from that invocation attach to that session.
Externally supplied Galileo session IDs are accepted only when they are UUIDv4
values, matching the ingestion contract; an invalid ID disables telemetry
without affecting the scan.

```text
aibom-agentic-scan                         session
├── aibom.agentic.batch                    trace per agentic batch
│   └── aibom.agentic.classifier           agent span (classifier)
│       └── aibom.agentic.<attempt-kind>   workflow span
│           ├── aibom.agentic.llm          actual call 1 (step 1)
│           ├── aibom.tool.<name>           actual call 2 (step 2)
│           └── aibom.agentic.llm          actual call 3 (step 3, decisions)
└── aibom.agentic.source_summary           trace per scanned source
```

Galileo's SDK always uses a Trace as the root; it has no separate
`start_agent_trace` root API. Agent typing is represented by the classifier
Agent span beneath each batch trace. The Console may therefore still title the
overall visualization **Trace Graph**, while showing the typed Agent node and
its nested attempt, LLM, and tool spans. Source-summary records intentionally
remain plain traces and do not contain an Agent span.

Attempt kinds are allowlisted as `initial`, `retry`, `fallback`, `coercion`,
`middleware_validation`, `unknown`, or `other`. Sampled cache hits, plus cache
replays retained for degraded or quality-exception signals, emit a batch trace
with `cache_hit=true` but do not fabricate LLM or tool spans. Successful,
cached, and skipped records use status code 200; other batch/LLM records use
500. Per-call tool spans use 500 on callback error; aggregate fallback tool
spans use 500 when their error or approved-root-denial count is non-zero.
Structured coercion emits its own workflow and LLM span.

Tool names come from one shared allowlist. It includes the AIBOM tools
`analyze_imports`, `list_directory_tree`, `lookup_model`, `read_file_snippet`,
`resolve_env_var`, `search_codebase`, `search_package_info`, and
`trace_data_flow`, plus the Deep Agents built-ins `compact_conversation`,
`edit_file`, `execute`, `glob`, `grep`, `ls`, `read_file`, `task`, `write_file`,
and `write_todos`. Every other name is emitted as `other`.

Callback events are sealed at the batch deadline, so a late completion from an
abandoned synchronous daemon invocation cannot mutate or cross-contaminate an
already concluded trace. Concurrent batches each own an independent callback,
trace handle, Agent span, workflow, and logger. Actual `created_at`, monotonic
duration, sequence, and per-call token carriers are retained; unknown usage
becomes null native token metrics plus `token_usage_missing=true`, not
fabricated zeros.
Response-level provider usage takes precedence over per-generation metadata;
when no response-level carrier exists, each generation's usage is counted
independently even when two choices report equal numeric values.

The decision-bearing LLM span contains raw model-requested mutation counts
(`enrich`, `remove`, `reclassify`, `discover`, relationships, and risks). The
workflow's `final_actions` and the batch output are computed from the
post-middleware product result. `blocked_actions` is the non-negative delta, so
a guard rejection is not mistaken for a model decision. Keep counts exist only
at the final boundary because the response schema has no explicit raw keep list.
`schema_valid` is false whenever a failed batch has no parsed decision carrier,
including timeouts, recursion limits, rate limits, provider outages, refusals,
parse failures, and no-LLM circuit-breaker or retry-budget skips.

Retries and fallbacks are tagged in both batch metadata and their attempt span.
When the retry orchestrator regroups failed candidates, that retry group receives
its own batch trace within the same scan session. Each candidate also carries a
stable HMAC `decision_chain_id`, so an initial, retry, fallback, or no-LLM trace
can be joined across split and merged batches without relying on batch numbers.
The source token namespaces batch, component, and decision-chain identities;
identical repository-relative component IDs in two sources therefore never
collide.

Each sanitized batch trace has one classifier Agent span. There are no retriever
spans and no raw LangChain/OpenAI auto-instrumentation on this path. The Agent
span changes the typed hierarchy Galileo receives, but remains subject to the
same content-free contract and does not make content-based metrics valid.
Sampling decisions are deterministic across repeated scans for the same
original source, tier, and batch number. Batch and source summary records are
still sampled independently. Operationally significant terminal batches and
summaries are retained even when their ordinary cohort was not sampled. Batch
retention covers failures, degradation, circuit breaking, schema/token-accounting
failures, middleware guards, tool errors or approved-root denials, discoveries,
and risk findings. Unsampled handles buffer only the sanitized ordered call shape
needed to build that retained trace; attempts are replayed as sibling workflows
under the batch Agent span. Deferred buffering is bounded at 16 LLM and 32 tool
spans per attempt. If an unusually long loop exceeds the LLM cap, the terminal
schema/decision-bearing call replaces the oldest LLM entry and sequence gaps
make truncation visible.

## Telemetry version dimensions and cache state

The batch trace's `prompt_version` is a non-sensitive 20-character digest used
only to compare Galileo traces produced by materially different agent
configurations. It currently covers the telemetry-version marker and installed
analyzer version; primary and fast models; selected provider initialization
settings; batching, concurrency, timeout, retry, and code-snippet settings;
hashes of the system and coercion prompts; the response schema; the agent-tool
implementation; and the custom agent-signature catalog.

This digest is observational metadata. It does not participate in agentic
verdict-cache identity, so enabling Galileo does not change normal cache keys
or cache-hit behavior. The independent `cache_hit` field reports whether the
existing agentic verdict cache supplied the result.

`schema_version` is the first 20 hex characters of the `AgentResponse`
JSON-schema SHA-256 digest. Internal tier, batch, and cross-repository payload
versions are implementation details and may advance between releases. Always
group or filter quality trends by `analyzer_version`, `prompt_version`,
`schema_version`, `model`, and `cache_hit` before comparing releases.

## Exact evaluation metrics

### Implemented entity-level metrics

`evaluate_decisions(...)` compares exact, de-duplicated component,
relationship, and risk identity sets. `DecisionEvaluationResult.to_galileo_metrics()`
returns these stable flat names:

- `aibom.components.{precision,recall,f1,true_positives,false_positives,false_negatives}`
- `aibom.relationships.{precision,recall,f1,true_positives,false_positives,false_negatives}`
- `aibom.risks.{precision,recall,f1,true_positives,false_positives,false_negatives}`
- `aibom.baseline_components.{precision,recall,f1,true_positives,false_positives,false_negatives}`
- `aibom.baseline_relationships.{precision,recall,f1,true_positives,false_positives,false_negatives}`
- `aibom.net_recall_lift`
- `aibom.relationship_recall_lift`
- `aibom.discoveries.{precision,recall,f1,true_positives,false_positives,false_negatives}`
- `aibom.over_prune_rate`
- `aibom.over_pruned_count`
- `aibom.action_accuracy`
- `aibom.action_macro_f1`
- `aibom.decision_coverage`
- `aibom.reclassification_accuracy`

Baseline, lift, discovery, over-pruning, and inferred action metrics are
present only when their pre-agent deterministic sets are supplied. Action
metrics may also use explicit expected/predicted actions; missing decisions
reduce coverage and macro-F1 rather than being inferred as removals. Explicit
negative risk labels are supported. The production CLI does not upload these
gold-label values.

`build_galileo_decision_metrics()` installs 29 trace-level `LocalMetric`
instances: the 20 primary entity/action/schema metrics plus nine exact
operational-outcome match metrics:

- `aibom.execution.status_accuracy`
- `aibom.execution.schema_validity_accuracy`
- `aibom.execution.abstention_accuracy`
- `aibom.execution.degraded_count_accuracy`
- `aibom.execution.retry_count_accuracy`
- `aibom.execution.fallback_count_accuracy`
- `aibom.execution.cache_hit_accuracy`
- `aibom.execution.tool_error_count_accuracy`
- `aibom.execution.guard_denial_count_accuracy`

Each operational metric is emitted only when that field is labeled in the
case's `expected_execution_outcome`; unlabeled dimensions return `null` and do
not dilute experiment aggregates.

### Operator-created production metrics

The sanitized JSON can support the following custom code metrics without raw
content. These names are recommended operator conventions, not resources
created by AI BOM:

| Metric name | Value |
|---|---|
| `aibom.prod.schema_valid` | `1` when batch output `schema_valid` is true, otherwise `0` |
| `aibom.prod.degraded_candidates` | Batch output `degraded_candidates` |
| `aibom.prod.middleware_guard_triggered` | `1` when the batch guard flag is true |
| `aibom.prod.cache_hit` | `1` when batch metadata `cache_hit` is true |
| `aibom.prod.raw_actions` | Sum the decision-bearing LLM span's requested mutation counts |
| `aibom.prod.blocked_actions` | Sum the attempt workflow's `blocked_actions` object |
| `aibom.prod.tool_errors` | Sum `tool_stats.*.errors` on the attempt workflow; use tool-span output only for the aggregate-fallback mode |
| `aibom.prod.tool_guard_denials` | Sum `tool_stats.*.guard_denials` on the attempt workflow |
| `aibom.prod.token_usage_missing` | `1` when an actual LLM span reports no token carrier |
| `aibom.prod.batch_success` | `1` for `success`, `cache_hit`, or `skipped`; otherwise `0` |
| `aibom.prod.degradation_rate` | Source `degraded_candidate_count / candidate_count` only when `candidate_count_available=true` and the denominator is non-zero; otherwise null |

Also use Galileo's system latency, status-code, and token aggregations. Cost is
not present in the sanitized AIBOM payload: model identifiers are pseudonymized
and no price or cost field is emitted. If cost is required, join the pseudonym
to an approved price table outside the trace payload or add a reviewed numeric
calculation within the access-restricted environment.
Do not enable Context Adherence, Correctness, Prompt Injection, PII, or other
content judges on sanitized production traces: they would evaluate aggregate
JSON rather than the source/model exchange and produce misleading results.
The sanitized classifier Agent span is structural only; it does not supply the
prompt, response, tool, or retrieval content those judges require.

In an approved full-content experiment only, useful native metrics include
Context Adherence, Completeness, Ground Truth Adherence, Instruction
Adherence, Action Completion, Agent Flow, Tool Selection Quality, Tool Error,
Agent Efficiency, Prompt Injection, and PII. The experiment harness must emit
the appropriate full-content LLM, tool, and retriever spans; the sanitized
production path does not do so, while the separately gated raw callback does.

Operators must define and approve evaluation acceptance criteria against their
own held-out labels. Content-based judge sampling cannot run on sanitized
production traces. Sampling rates and promotion thresholds for a separately
approved full-content evaluation stream must be selected through documented
privacy, cost, and quality review. Promote reviewer corrections into a new
immutable suite version rather than modifying history.

## Golden dataset schema

Golden datasets are **operator-managed and are not uploaded automatically**.
Use immutable, reviewed fixtures in the versioned
`aibom.galileo.decision_suite.v1` schema. Each case has one deterministic
candidate batch, a complete action map, exact final/discovered entities,
optional deterministic baseline edges, expected relationship/risk labels,
optional expected execution outcomes, slice metadata, and optional approved
evidence. The legacy `candidate`/`expected_action` pair remains accepted for
single-candidate fixtures, but new datasets should use the batch contract:

```json
{
  "schema_version": "aibom.galileo.decision_suite.v1",
  "metadata": {"dataset_version": "example-v1"},
  "cases": [{
    "case_id": "batch-001-router",
    "candidates": [{
      "name": "router",
      "component_type": "tool",
      "repository": "example/router",
      "source_path": "src/router.py",
      "line_number": 12,
      "stable_case_id": "case-001-router"
    }, {
      "name": "gpt-5",
      "component_type": "model",
      "repository": "example/router",
      "source_path": "src/models.py",
      "line_number": 4,
      "stable_case_id": "case-001-model"
    }],
    "expected_actions": {
      "case-001-router": {
        "action": "reclassify",
        "target_type": "agent"
      },
      "case-001-model": {"action": "keep"}
    },
    "expected_components": [{
      "name": "router",
      "component_type": "agent",
      "repository": "example/router",
      "source_path": "src/router.py",
      "line_number": 12,
      "stable_case_id": "case-001-router"
    }, {
      "name": "gpt-5",
      "component_type": "model",
      "repository": "example/router",
      "source_path": "src/models.py",
      "line_number": 4,
      "stable_case_id": "case-001-model"
    }],
    "expected_discovered_components": [],
    "deterministic_relationships": [],
    "expected_relationships": [{
      "relationship_type": "uses_model",
      "source_case_id": "case-001-router",
      "target_case_id": "case-001-model"
    }],
    "expected_risks": [{
      "case_id": "case-001-router",
      "risk_type": "unresolved_model_reference",
      "severity": "high",
      "expected_present": true
    }],
    "expected_execution_outcome": {
      "status": "success",
      "schema_valid": true,
      "abstained": false,
      "degraded_candidate_count": 0,
      "retry_count": 0,
      "fallback_count": 0,
      "cache_hit": false,
      "tool_error_count": 0,
      "guard_denial_count": 0
    },
    "approved_evidence": [],
    "metadata": {"language": "python", "category": "reclassification"}
  }]
}
```

The application may return a native `PipelineResult`-like object or the
canonical output envelope. For example, a sanitized provider-outage result is:

```json
{
  "schema_version": "aibom.galileo.decision_output.v1",
  "schema_valid": false,
  "final_components": [],
  "relationships": [],
  "risk_flags": [],
  "actions": null,
  "execution_outcome": {
    "status": "provider_outage",
    "schema_valid": false,
    "degraded_candidate_count": 2,
    "retry_count": 2,
    "fallback_count": 1,
    "cache_hit": false,
    "tool_error_count": 0,
    "guard_denial_count": 0
  }
}
```

`validate_decision_suite(...)` rejects unknown fields, missing repository or
source-path identity, absolute/traversing paths, duplicate labels, malformed
reclassifications, discoveries already present in the deterministic set,
relationships with unknown endpoints, expected risks without severity, empty
operational-outcome labels, oversized evidence, and secret-shaped metadata
keys. `deterministic_relationships` may also be supplied as
`baseline_relationships`; `None` means unlabeled while explicit `[]` means a
labeled empty graph. `build_galileo_experiment_rows(...)` creates canonical
Galileo 2.4 rows without SDK import or network access.
`run_galileo_custom_function_experiment(...)` then uses Galileo's current
custom-function path—not the legacy AIBOM benchmark—to run the supplied
application function. In the default mode, exact rows never become Galileo's
hosted dataset: a per-run random registry pseudonymizes component, case, action,
edge, and risk identities; names; repositories; paths; non-zero line locations;
and semantic type labels. It removes arbitrary metadata, reason codes, runtime
IDs, and all approved-evidence excerpts. Experiment name/group/tag values are
pseudonymized too. The exact fixture remains in the runner's process, the
application receives a fresh deep copy for each call, and an unknown hosted case
token produces a schema-invalid result without invoking the application. The
runner cannot control application-side model or network egress; operators must
separately approve and constrain the supplied application.

Native `PipelineResult`-like output is first projected against the in-process
exact fixture: runtime identifiers are mapped back to fixture-stable IDs and absolute
paths/application exception text are discarded. The canonical result is then
mapped into the same hosted pseudonym namespace. Because this is a consistent
injective renaming, deterministic TP/FP/FN, lift, action, relationship, risk,
and reliability metric values are unchanged; automated tests compare all 29
metric outputs between exact and pseudonymous runs. Random per-run mappings
intentionally prevent cross-run entity correlation. The supplied application
function must create an isolated empty agentic cache for every model/prompt
variant; the helper cannot enforce this application-level obligation.

Wrap native four-stage results with
`adapt_pipeline_result_for_galileo(result, execution_outcome=...,
actions=..., dataset_input=...)` when a case labels reliability or substantive
enrichment. The adapter automatically carries the authoritative
`agentic_degraded_count`, derives `success`/`degraded`, and marks total
degradation as abstention when the network-free dataset input is supplied.
Degraded passthroughs are omitted from inferred actions, so they reduce
decision coverage instead of earning false keep credit. The experiment
application must supply facts that `PipelineResult` does not expose, including
provider-specific terminal status, schema validity, retry/fallback, cache,
tool-error, and guard-denial counts. It must also supply explicit actions for
`enrich`, because the content-minimized entity identity cannot distinguish
substantive enrichment from keep. Unlabeled or unavailable fields remain
absent rather than being guessed.

Store fixture content outside the suite by default. If approved evidence is
needed for Agent Flow, Tool Selection, or `AIBOM Evidence Grounding`, add
bounded `approved_evidence` excerpts and use the exact/full-content mode; the
default pseudonymous mode deliberately removes the excerpts, so content judges
cannot run there. `create_galileo_async_callback(...)` is used by both approved
fixture evaluation and the live diagnostic path; it rejects external
logger/ingestion-hook overrides.

```bash
export AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD=true
export AIBOM_GALILEO_EVALUATION_PROJECT_ID="<project-uuid>"

# Required for exact experiment rows and every raw trajectory:
export AIBOM_GALILEO_ALLOW_EXACT_IDENTITIES=true

# Required when experiment rows include evidence and for every raw trajectory:
export AIBOM_GALILEO_ALLOW_FULL_CONTENT=true

# Add only for the raw LangChain trajectory callback:
export AIBOM_GALILEO_ALLOW_FULL_TRAJECTORY=true
export AIBOM_GALILEO_EVALUATION_LOG_STREAM_ID="<log-stream-uuid>"

# Be explicit when running a reviewed live raw trajectory:
cisco-aibom analyze /approved/repository --llm-model gpt-5.4 \
  --galileo --galileo-full-trajectory \
  -o json -O approved-trajectory-report.json

# Optional explicit assertion of the default sanitized-only behavior:
cisco-aibom analyze /path/to/repository --llm-model gpt-5.4 \
  --galileo --no-galileo-full-trajectory \
  -o json -O sanitized-report.json
```

## Operator setup: project, dashboard, and alerts

This repository contains no setup automation for projects, log streams,
metrics, datasets, dashboards, alerts, webhooks, annotation queues, or
retention rules. A Galileo tenant operator must provision or verify these
resources and review the resulting setup; do not rely on any SDK-side implicit
creation as production configuration. Hosted Galileo does not make the project
publicly readable: keep tenant access restricted and explicitly review sharing.

Recommended project layout:

- one access-restricted hosted project for AIBOM agentic quality;
- separate operator-named sanitized streams for evaluation, pre-production
  validation, and production;
- an immutable project UUID for pseudonymous experiments, plus a separately
  pinned log-stream UUID and stricter access boundary for exact/full-content
  trajectories;
- dashboard filters for trace name, analyzer/telemetry/schema version, provider,
  model, tier, cache hit, source kind, and status.

Recommended dashboard sections:

1. Volume and outcomes: sessions, batch/source traces, statuses, cache-hit rate,
   and decision counts.
2. Quality: schema validity, degraded candidates/rate, guard triggers, and the
   offline entity metrics listed above. Use the batch output's bounded
   action-by-type, action-by-language, and action-by-confidence maps for exact
   removal and reclassification drift slices, including mixed batches.
3. Reliability and usage: p50/p95 per-call/batch latency, tokens, workflow
   `tool_stats` calls/errors/guard denials, callback tool status, and status
   codes. Group retry and fallback outcomes by `source_id` plus
   `decision_chain_id`, not by batch ordinal, because retries may be regrouped.
   Show cost only when an operator-approved external calculation is configured.
4. Release comparison: analyzer, telemetry configuration, schema, model, and
   provider, with cache-hit status as a separate slice.

Starting alert rules should notify, not block:

- a source where `candidate_count_available=true` and
  `degraded_candidate_count == candidate_count > 0`;
- any `circuit_breaker`, failed, timeout, refusal, or provider-outage status;
- any schema-invalid output that leaves a source degraded;
- any actual LLM span with `token_usage_missing=true`;
- any `middleware_guard_triggered=true` or tool `guard_denials>0`;
- any privacy-contract canary failure reported by CI or an approved
  pre-ingestion validator (the rejected payload must not be ingested); and
- any tool status 500 or `tool_errors>0`.

After an operator-defined baseline period, add warnings for material deviations
in discovery, high-confidence removal, structured recovery, latency, token use,
and approved cost calculations. Define windows, minimum sample sizes, and
thresholds for each environment through the organization's monitoring and risk
review. The high-confidence removal signal is available in the
`decisions_by_confidence.removed.high` batch counter. Reviewed recall lift and
cost still require approved review/dataset or pricing joins outside the
sanitized trace itself.

Route notifications to an approved email, Slack, or incident webhook and use
the alert deduplication key in downstream routing. Calibrate windows and
thresholds on staging data before production.

## Known integration boundaries

- The retry orchestrator can regroup failed candidates. Retries and fallbacks
  remain separate batch traces rather than child workflows of the original
  batch trace; stable source-scoped `decision_chain_id` values provide explicit
  candidate-level lineage across those trace boundaries.
- The custom experiment helper requires, but cannot itself enforce, an isolated
  empty AIBOM agentic cache for every model/prompt variant.
- Per-run random experiment pseudonyms deliberately cannot be joined across
  runs. Exact SME annotation/replay needs the independently approved exact mode
  or a separate local fixture registry; do not weaken hosted pseudonymization
  merely to make identifiers stable.
- Operator-supplied metric objects and their display names are outside the row
  pseudonymizer. Keep custom metric names and configuration generic, and do not
  embed repository, customer, incident, or credential data in them.
- The full callback observes the entire LangChain trajectory, not only the
  bounded `approved_evidence` array. Its separate exact/content/trajectory gates
  are necessary but do not replace fixture-root isolation, tool allowlisting,
  secret/PII review, RBAC, or retention controls.

## Retention and RBAC assumptions

The runtime assumes only that the configured API key can write to the named
project and log stream. It does not verify tenant, residency, retention,
encryption, exports, SSO, or reader permissions.

Before enabling telemetry, operators must:

- keep the project private and grant least-privilege roles;
- restrict API-key ownership and rotation to the service/operator boundary;
- give production write access only to the scanner identity;
- limit exports and dashboard/trace access to approved reviewers;
- set the shortest organization-approved retention period and document
  deletion/backup behavior;
- confirm residency, encryption, subprocessors, model-judge egress, and
  contractual controls; and
- periodically audit project membership, keys, webhooks, and retained traces.

Sanitization reduces exposure; it does not waive privacy, security, or records
management requirements.

Set retention independently for each stream according to the organization's
approved policy. Full-content evaluation data should use the shortest retention
period permitted for the approved use case.

## Observe-only rollout

1. **Disabled baseline:** keep the default `--no-galileo`; provision the hosted
   tenant project, metrics, RBAC, retention, dashboards, and alerts.
2. **Non-production validation:** enable sanitized telemetry with a stable HMAC
   key. Inspect trace payloads and exports to verify that no raw content or
   identifiers appear.
3. **Production canary:** use a low, approved AIBOM sampling rate. Keep alerts
   notification-only and compare scan outputs with telemetry disabled/enabled.
4. **Broader observation:** raise sampling only after privacy and cost review.
   Establish baselines by telemetry configuration and analyzer release, with
   cache-hit status as a separate slice.
5. **Evaluation loop:** run golden-dataset experiments through the default
   pseudonymous hosted path and verify metric parity locally. Enable exact rows
   or raw trajectories only for separately approved review cases; use
   entity-level metrics to gate releases outside production telemetry.

The Galileo SDK's Agent Control bridge is explicitly disabled for every
AIBOM-constructed sanitized or raw logger; there is no automated remediation
path. An alert must lead to manual investigation. Disable emission immediately with
`--no-galileo` or `AIBOM_GALILEO_ENABLED=false`; scanning remains functional
when Galileo is unavailable.

## Galileo references

- [Logging basics](https://docs.galileo.ai/sdk-api/logging/logging-basics)
- [Galileo Logger](https://docs.galileo.ai/sdk-api/logging/galileo-logger)
- [Sessions](https://docs.galileo.ai/concepts/logging/sessions/sessions-overview)
- [Metrics overview](https://docs.galileo.ai/concepts/metrics/overview)
- [Agent Flow](https://docs.galileo.ai/concepts/metrics/agentic/agent-flow)
- [Tool Selection Quality](https://docs.galileo.ai/concepts/metrics/agentic/tool-selection-quality)
- [Datasets](https://docs.galileo.ai/sdk-api/experiments/datasets)
- [Alerts](https://docs.galileo.ai/how-to-guides/basics/set-up-alerts-on-logs)
- [Access control](https://docs.galileo.ai/concepts/access-control)
