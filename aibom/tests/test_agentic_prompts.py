from aibom.agentic.prompts import AIBOM_AGENT_SYSTEM_PROMPT


def _normalized_aibom_prompt() -> str:
    """Single-line lowercase prompt for stable substring checks."""
    return " ".join(AIBOM_AGENT_SYSTEM_PROMPT.lower().split())


class TestAgenticPromptHardening:
    def test_dependency_prompt_rejects_bare_version_names(self) -> None:
        assert "0.18.0" in AIBOM_AGENT_SYSTEM_PROMPT
        assert "3.9.3" in AIBOM_AGENT_SYSTEM_PROMPT
        assert "0.59b0" in AIBOM_AGENT_SYSTEM_PROMPT
        assert "dependency names that are only version tokens" in AIBOM_AGENT_SYSTEM_PROMPT

    def test_dependency_prompt_prefers_package_identifier_from_context(self) -> None:
        assert "prefer the package identifier from the file context" in AIBOM_AGENT_SYSTEM_PROMPT
        assert "fuzzywuzzy" in AIBOM_AGENT_SYSTEM_PROMPT

    def test_embedding_prompt_rejects_helper_and_template_false_positives(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "storage paths, file templates, copy jobs, and helper functions are not embedding assets" in prompt
        assert "s3 key templates" in prompt
        assert "class name alone is not enough to confirm an embedding" in prompt

    def test_cache_replay_prompt_preserves_validated_relationships(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "treat replayed/cache-restored findings the same way you would treat fresh reasoning" in prompt
        assert "preserve it" in prompt
        assert "validated cross-component links" in prompt

    def test_cache_replay_prompt_rejects_only_spurious_extras(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "partial-cache results contain relationships or risk findings that conflict with the current code context" in prompt
        assert "reject only the spurious extras" in prompt
        assert "do not silently drop validated cross-component links" in prompt

    def test_dependency_prompt_rejects_generic_non_ai_packages(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "requests" in prompt
        assert "lodash" in prompt
        assert "github.com/google/uuid" in prompt
        assert "generic infra, db, telemetry, auth, test, or utility packages" in prompt

    def test_prompt_distinguishes_remove_vs_reclassify(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "use `remove_components` when the row is not an ai asset at all" in prompt
        assert "use `reclassify_components` when the row is ai-relevant but typed incorrectly" in prompt
        assert "do not leave a wrong type in `enriched_components` without a matching `reclassify_components` entry" in prompt

    def test_prompt_rejects_prompt_plumbing_non_ai_secrets_and_metric_constants(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "load_prompt" in prompt
        assert "all_messages" in prompt
        assert "dialog" in prompt
        assert "ld_api_key" in prompt
        assert "guardrail_input_token_count" in prompt

    def test_prompt_rejects_generic_helper_prompt_kwargs(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "render_prompt(prompt=...)" in prompt
        assert "requestbuilder.create(messages=payload)" in prompt
        assert "real ai client or agent framework call" in prompt

    def test_prompt_rejects_vector_db_files_and_test_only_agents(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "data_level0.bin" in prompt
        assert "link_lists.bin" in prompt
        assert "fakeagentrouter" in prompt
        assert "testagentloop" in prompt

    def test_prompt_rejects_kb_method_helper_matches_without_store_evidence(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "create_store_adapter" in prompt
        assert "build_index_client" in prompt
        assert "kb method matches are usually operations, not assets" in prompt
        assert "concrete store/client instance, endpoint, or persisted asset identifier" in prompt

    def test_prompt_handles_conflicting_vector_backend_selectors(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "vector_db_type" in prompt
        assert "weaviate_endpoint" in prompt
        assert "explicit backend selection beats provider-name hints" in prompt
        assert "treat the row as ambiguous" in prompt

    def test_prompt_preserves_deployment_ids_without_inventing_canonical_models(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "prod-chat-gpt4o" in prompt
        assert "embed-prod-westus" in prompt
        assert "do not invent a canonical public model name without supporting evidence" in prompt
