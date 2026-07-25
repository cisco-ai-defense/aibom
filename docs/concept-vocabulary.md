# Schema-v2 concept vocabulary

The schema-v2 knowledge base classifies every catalog entry with one of the
20 concepts below. `cisco-aibom kb info` displays the schema and vocabulary
versions, plus this concept list, when the locally installed manifest declares
`schema_version: 2`. The command is fully offline.

| Concept | Allowed subconcepts |
|---|---|
| `model` | `chat`, `completion`, `structured_output`, `multimodal_vision`, `multimodal_audio`, `multimodal_video`, `code_generation` |
| `embedding` | `dense`, `sparse`, `colbert`, `multilingual` |
| `reranker` | `cross_encoder`, `api_reranker`, `bm25` |
| `agent` | `react`, `plan_and_execute`, `structured_chat`, `multi_agent_supervisor`, `openai_functions`, `tool_calling` |
| `tool` | `function_call`, `retrieval`, `calculator`, `code_execution`, `web_search`, `file_ops`, `mcp_bridge` |
| `skill` | None |
| `prompt` | `template`, `few_shot`, `hub_pull`, `system`, `partial_template` |
| `vector_store` | `managed_cloud`, `self_hosted`, `in_memory`, `hybrid_search` |
| `retriever` | `bm25`, `tfidf`, `multi_query`, `parent_doc`, `contextual_compression`, `self_query`, `ensemble` |
| `memory` | `buffer`, `summary`, `entity`, `vector_store_backed`, `window` |
| `guardrail` | `input_filter`, `output_filter`, `policy_nemo`, `policy_guardrails_ai`, `content_safety` |
| `evaluator` | `ragas`, `deepeval`, `llm_as_judge`, `custom_metric` |
| `mcp_server` | `stdio`, `sse`, `http_streamable`, `fastmcp` |
| `mcp_client` | `stdio`, `sse`, `http_streamable`, `multi_server` |
| `dataset` | None |
| `training_run` | `sft`, `dpo`, `grpo`, `reward_modeling`, `lora`, `qlora`, `full_finetune` |
| `model_artifact` | `hf_hub_push`, `hf_hub_pull`, `checkpoint`, `lora_adapter`, `peft_adapter`, `quantized_gguf` |
| `observability` | `tracer`, `callback_handler`, `span_exporter`, `cost_tracker` |
| `framework_core` | None |
| `document_loader` | `pdf`, `web`, `confluence`, `s3`, `github`, `notion`, `gdrive`, `sql`, `csv`, `office_docs` |

## Exclusions

`chain` and `text_splitter` are intentionally not schema-v2 concepts. A
framework symbol with one of those roles must be represented by another
allowed concept when appropriate; it must not extend the vocabulary implicitly.

## Evolution policy

- Adding a subconcept does not change the schema version. It increments the
  vocabulary version.
- Adding a concept increments the minor schema version and the vocabulary
  version.
- Renaming or removing a concept, or changing its meaning, increments the
  major schema version and requires a coordinated cutover.

## CLI compatibility

The packaged CLI 1.x manifest and its schema-v1 DuckDB artifact remain
supported. If a 1.x CLI is pointed at a schema-v2 manifest, it exits that KB
load path with an upgrade-required message instead of attempting to read an
incompatible catalog. That message applies only to the selected schema-v2
knowledge base; it does not invalidate an existing schema-v1 artifact.

## Release checks and rollback

Before activating a schema-v2 manifest:

1. Run a representative scan against a frozen schema-v1 fixture and confirm
   its output is unchanged.
2. Run `cisco-aibom kb info` against the schema-v2 candidate and confirm the
   reported schema, vocabulary version, and concept count.
3. Exercise the 1.x compatibility path and confirm the upgrade-required
   message is concise and does not include a traceback.

After activation, monitor KB download, checksum, and unsupported-schema
failures separately. A rise in unsupported-schema failures can indicate that
older clients are selecting the new manifest. Roll back by restoring the
previous manifest selection; the frozen schema-v1 artifact itself remains
valid.
