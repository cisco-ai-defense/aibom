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
        assert "feature-flag api keys" in prompt
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
        assert "prod-chat-gpt4o-westus" in prompt
        assert "embed-prod-westus" in prompt
        assert "org/custom-model/stable" in prompt
        assert "do not remove them" in prompt
        assert "do not reclassify them" in prompt
        assert "do not replace them with" in prompt
        assert "middleware enforces this as a hard safety rail" in prompt

    def test_prompt_defines_agent_three_conditions(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "llm-driven control flow" in prompt
        assert "tool / action execution" in prompt
        assert "iterative loop" in prompt
        assert "all three" in prompt

    def test_prompt_lists_agent_positive_patterns(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "react loops" in prompt or "react loop" in prompt
        assert "agentexecutor" in prompt
        assert "create_react_agent" in prompt
        assert "agent_proxy" in prompt or "agent proxies" in prompt

    def test_prompt_lists_agent_negative_patterns(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "a class that calls an llm once" in prompt
        assert "sequentialchain" in prompt
        assert "the name is misleading" in prompt

    def test_prompt_agent_verification_procedure(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "read_file_snippet" in prompt
        assert "verification procedure" in prompt
        assert "source module" in prompt

    def test_prompt_agent_relationship_discovery(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "agent relationship discovery procedure" in prompt
        assert "client.chat.completions.create" in prompt
        assert "uses_vector_store" in prompt

    def test_prompt_read_file_snippet_tool(self) -> None:
        assert "read_file_snippet" in AIBOM_AGENT_SYSTEM_PROMPT
        assert "inspect class definitions" in AIBOM_AGENT_SYSTEM_PROMPT

    def test_prompt_forbids_verdicts_for_other_detected_components(self) -> None:
        prompt = _normalized_aibom_prompt()
        assert "read-only context" in prompt
        assert (
            "you must not emit `remove_components`, `reclassify_components`, "
            "or `enriched_components` entries for any `instance_id` that "
            "appears in `other_detected_components`"
        ) in prompt
        assert (
            "only ids that appear in `enrich_these` are valid targets"
        ) in prompt

    def test_model_section_has_value_literal_removal_rule(self) -> None:
        """Fix 10: section 4 (Model verification) must enumerate the
        deployment-artifact value forms that should always be removed,
        even when ``lookup_model`` returns ``found: true``. Mirrors the
        existing dependency-version rule for the model class.
        """
        prompt = _normalized_aibom_prompt()
        for needle in [
            "value-literal removal",
            "pure version literal",
            "pure numeric token",
            "ipv4 literal",
            "bare environment marker",
            ".image_tag",
            "_threshold",
            "env.environment",
        ]:
            assert needle in prompt, (
                f"Fix 10 prompt vocabulary missing: {needle!r}"
            )

    def test_model_section_acknowledges_lookup_model_can_lie(self) -> None:
        """Fix 10: the load-bearing instruction is that a registry hit
        does not override the value-literal removal rule. Without this
        line the agent would defer to ``lookup_model`` and keep
        ``3.10.2``-style garbage.
        """
        prompt = _normalized_aibom_prompt()
        assert "even if ``lookup_model`` reports" in prompt or (
            "even if ``lookup_model`` returns" in prompt
        ) or (
            "even if `lookup_model` reports" in prompt
        ), (
            "model section must contain the 'even if lookup_model …' "
            "override clause that justifies removing registry-known "
            "values when they're deployment artifacts"
        )

    def test_iid_verbatim_rule_forbids_invented_line_numbers(self) -> None:
        """Fix 13: an agent that picks a "likely" line number — e.g.
        the start of a class body it just read via ``read_file_snippet``
        — silently breaks downstream removal. Rule 4 in ABSOLUTE RULES
        must spell out that line numbers are the exact scanner-emitted
        source line, copied character-for-character from
        ``enrich_these``.
        """
        prompt = _normalized_aibom_prompt()
        for needle in [
            "never invent or guess a line number",
            "scanner emitted the candidate",
            "do not pick \"a likely line\"",
            "verified the candidate at via `read_file_snippet`",
            "copy it character-for-character",
        ]:
            assert needle in prompt, (
                f"Fix 13 iid-verbatim vocabulary missing: {needle!r}"
            )

    def test_iid_verbatim_rule_forbids_fabricated_paths(self) -> None:
        """Fix 13: when the same logical concept appears in many
        ``values*.yaml`` files, the agent must not synthesise a
        per-file ``instance_id`` — the orchestrator already fans out a
        single removal via the consolidation key. Cite the multi-file
        ``values*.yaml`` case so the rule generalises beyond Python
        sources.
        """
        prompt = _normalized_aibom_prompt()
        for needle in [
            "never fabricate a path",
            "values*.yaml",
            "consolidation key",
            "you do not need to (and must not) emit a separate verdict for each file",
        ]:
            assert needle in prompt, (
                f"Fix 13 path-fabrication vocabulary missing: {needle!r}"
            )

    def test_iid_verbatim_rule_directs_other_detected_to_no_op(self) -> None:
        """Fix 13: if the agent looks at ``other_detected_components``
        and concludes "this should be removed", the right action is to
        do nothing in the current batch — the orchestrator schedules
        each candidate in its own batch where it will appear in
        ``enrich_these``. The middleware redirect via consolidation key
        is a safety net for the agent's mistakes; the prompt must not
        encourage the agent to lean on it.
        """
        prompt = _normalized_aibom_prompt()
        assert (
            "if you believe a candidate should be removed but it is in "
            "`other_detected_components` (not `enrich_these`), do not emit a "
            "verdict for it"
        ) in prompt, (
            "Fix 13 must instruct the agent to no-op on "
            "other_detected_components rather than risk emitting a "
            "fabricated iid"
        )
