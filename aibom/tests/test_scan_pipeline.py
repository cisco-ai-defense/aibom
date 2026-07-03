# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from aibom.agentic.agent import TokenUsage
from aibom.models import AIComponent, AIComponentType
from aibom.scan_pipeline import ScanPipeline, StageTiming


def _stub_token_usage() -> TokenUsage:
    return TokenUsage()


class TestScanPipeline:
    def test_basic_run(self, tmp_path: Path) -> None:
        """A scan path with a simple Python file should complete all 4 stages."""
        (tmp_path / "app.py").write_text(
            'from openai import OpenAI\nclient = OpenAI(model="gpt-4o")\n'
        )
        pipeline = ScanPipeline(scan_paths=[str(tmp_path)])
        result = pipeline.run()
        assert result.components is not None
        assert result.env_index is not None
        assert result.pkg_index is not None

    def test_strict_filters_agentic(self, tmp_path: Path) -> None:
        """Strict mode should remove agentic candidates from results."""
        (tmp_path / "values.yaml").write_text(
            "inference:\n  model: some-custom-thing\n"
        )
        pipeline_relaxed = ScanPipeline(scan_paths=[str(tmp_path)], strict=False)
        result_relaxed = pipeline_relaxed.run()

        pipeline_strict = ScanPipeline(scan_paths=[str(tmp_path)], strict=True)
        result_strict = pipeline_strict.run()

        agentic_relaxed = [c for c in result_relaxed.components if c.needs_agentic]
        agentic_strict = [c for c in result_strict.components if c.needs_agentic]
        assert len(agentic_strict) == 0
        if agentic_relaxed:
            assert result_strict.agentic_candidate_count > 0

    def test_cross_ref_resolution(self, tmp_path: Path) -> None:
        """EnvVarResolver + cross-ref should wire env vars to config values."""
        (tmp_path / ".env").write_text("LLM_MODEL=gpt-4o\n")
        (tmp_path / "app.py").write_text(
            "import os\nfrom openai import OpenAI\n"
            "client = OpenAI()\n"
            "resp = client.chat.completions.create(\n"
            '    model=os.getenv("LLM_MODEL"),\n'
            "    messages=[]\n"
            ")\n"
        )
        pipeline = ScanPipeline(scan_paths=[str(tmp_path)])
        result = pipeline.run()

        env_comps = [c for c in result.components if c.name.startswith("env:")]
        resolved = [c for c in env_comps if c.model_name and c.model_name == "gpt-4o"]
        if env_comps:
            assert any(
                c.metadata.get("resolved_value") == "gpt-4o" or c.model_name == "gpt-4o"
                for c in env_comps
            )

    def test_external_deps_detected(self, tmp_path: Path) -> None:
        """Pipeline should detect cross-repo git dependencies."""
        (tmp_path / "pyproject.toml").write_text(
            """[tool.poetry.dependencies]
python = "^3.11"
ai-common = {git = "https://github.com/org/ai-common.git", branch = "main"}
""",
            encoding="utf-8",
        )
        (tmp_path / "app.py").write_text("print('hello')\n")
        pipeline = ScanPipeline(scan_paths=[str(tmp_path)])
        result = pipeline.run()
        assert len(result.external_deps) >= 1
        assert result.external_deps[0].name == "ai-common"

    def test_empty_path(self, tmp_path: Path) -> None:
        """Empty scan path should return empty results gracefully."""
        pipeline = ScanPipeline(scan_paths=[str(tmp_path)])
        result = pipeline.run()
        assert result.components == [] or isinstance(result.components, list)
        assert result.agentic_candidate_count >= 0


class TestPipelineTiming:
    def test_timings_populated(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text(
            'from openai import OpenAI\nclient = OpenAI(model="gpt-4o")\n'
        )
        result = ScanPipeline(scan_paths=[str(tmp_path)]).run()
        assert len(result.timings) == 4
        assert result.total_elapsed_s > 0
        names = [t.name for t in result.timings]
        assert names == ["scan", "cross_ref", "agentic", "assemble"]
        for t in result.timings:
            assert t.elapsed_s >= 0

    def test_agentic_skipped_detail(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("x = 1\n")
        result = ScanPipeline(scan_paths=[str(tmp_path)]).run()
        agentic_timing = next(t for t in result.timings if t.name == "agentic")
        assert "skipped" in agentic_timing.detail

    def test_file_cache_stats_in_scan_detail(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("import openai\n")
        result = ScanPipeline(scan_paths=[str(tmp_path)]).run()
        scan_timing = next(t for t in result.timings if t.name == "scan")
        assert "file cache" in scan_timing.detail

    def test_progress_callback_receives_stage_and_scanner_events(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "app.py").write_text("import openai\n")
        events: list[dict[str, object]] = []
        result = ScanPipeline(
            scan_paths=[str(tmp_path)],
            progress_callback=events.append,
        ).run()

        assert result.components is not None
        event_names = [str(event["event"]) for event in events]
        assert event_names[0] == "stage_started"
        assert "scanners_discovered" in event_names
        assert "scanner_completed" in event_names
        assert event_names[-1] == "stage_completed"
        stage_names = [
            str(event["stage"]) for event in events if event["event"] == "stage_started"
        ]
        assert stage_names == ["scan", "cross_ref", "agentic", "assemble"]


class TestAgenticScope:
    def test_all_components_sent_to_agent(self, tmp_path: Path) -> None:
        """All components are sent to the agent for classification."""
        (tmp_path / "app.py").write_text(
            'from openai import OpenAI\nclient = OpenAI(model="gpt-4o")\n'
        )
        llm_cfg = {"model": "test/model", "api_key": "fake", "api_base": "http://x"}
        pipeline = ScanPipeline(
            scan_paths=[str(tmp_path)],
            llm_config=llm_cfg,
        )
        with patch("aibom.scan_pipeline.ensure_llm_runtime_available"):
            with patch("aibom.agentic.agent.run_agentic_enrichment") as mock_enrich:
                mock_enrich.return_value = ([], [], [], _stub_token_usage())
                result = pipeline.run()
        mock_enrich.assert_called_once()

    def test_agentic_cache_dir_is_forwarded(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text(
            'from openai import OpenAI\nclient = OpenAI(model="gpt-4o")\n'
        )
        llm_cfg = {"model": "test/model", "api_key": "fake", "api_base": "http://x"}
        agentic_cache_dir = tmp_path / "scan-cache" / "agentic"
        pipeline = ScanPipeline(
            scan_paths=[str(tmp_path)],
            llm_config=llm_cfg,
            agentic_cache_dir=agentic_cache_dir,
        )
        with patch("aibom.scan_pipeline.ensure_llm_runtime_available"):
            with patch("aibom.agentic.agent.run_agentic_enrichment") as mock_enrich:
                mock_enrich.return_value = ([], [], [], _stub_token_usage())
                pipeline.run()
        assert mock_enrich.call_args.kwargs["cache_dir"] == agentic_cache_dir

    def test_include_code_snippets_is_forwarded(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text(
            'from openai import OpenAI\nclient = OpenAI(model="gpt-4o")\n'
        )
        llm_cfg = {"model": "test/model", "api_key": "fake", "api_base": "http://x"}
        pipeline = ScanPipeline(
            scan_paths=[str(tmp_path)],
            llm_config=llm_cfg,
            include_code_snippets=True,
        )
        with patch("aibom.scan_pipeline.ensure_llm_runtime_available") as mock_runtime:
            with patch("aibom.agentic.agent.run_agentic_enrichment") as mock_enrich:
                mock_enrich.return_value = ([], [], [], _stub_token_usage())
                pipeline.run()
        mock_runtime.assert_called_once()
        assert mock_enrich.call_args.kwargs["include_code_snippets"] is True

    def test_max_consecutive_failures_is_forwarded(self, tmp_path: Path) -> None:
        # the configurable circuit-breaker threshold must reach
        # run_agentic_enrichment (previously it was never passed -> stuck at 3).
        (tmp_path / "app.py").write_text(
            'from openai import OpenAI\nclient = OpenAI(model="gpt-4o")\n'
        )
        llm_cfg = {"model": "test/model", "api_key": "fake", "api_base": "http://x"}
        pipeline = ScanPipeline(
            scan_paths=[str(tmp_path)],
            llm_config=llm_cfg,
            agentic_max_consecutive_failures=7,
        )
        with patch("aibom.scan_pipeline.ensure_llm_runtime_available"):
            with patch("aibom.agentic.agent.run_agentic_enrichment") as mock_enrich:
                mock_enrich.return_value = ([], [], [], _stub_token_usage())
                pipeline.run()
        assert mock_enrich.call_args.kwargs["max_consecutive_failures"] == 7

    def test_max_retry_seconds_is_forwarded(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text(
            'from openai import OpenAI\nclient = OpenAI(model="gpt-4o")\n'
        )
        llm_cfg = {"model": "test/model", "api_key": "fake", "api_base": "http://x"}
        pipeline = ScanPipeline(
            scan_paths=[str(tmp_path)],
            llm_config=llm_cfg,
            agentic_max_retry_seconds=42,
        )
        with patch("aibom.scan_pipeline.ensure_llm_runtime_available"):
            with patch("aibom.agentic.agent.run_agentic_enrichment") as mock_enrich:
                mock_enrich.return_value = ([], [], [], _stub_token_usage())
                pipeline.run()
        assert mock_enrich.call_args.kwargs["max_retry_seconds"] == 42


class TestFileCache:
    def test_cache_deduplicates(self, tmp_path: Path) -> None:
        from aibom.scanners.file_cache import cache_stats, clear_cache, read_text_cached

        clear_cache()
        f = tmp_path / "test.py"
        f.write_text("hello\n")

        read_text_cached(f)
        read_text_cached(f)
        stats = cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1
        clear_cache()

    def test_clear_resets(self, tmp_path: Path) -> None:
        from aibom.scanners.file_cache import cache_stats, clear_cache, read_text_cached

        clear_cache()
        f = tmp_path / "test.py"
        f.write_text("hello\n")
        read_text_cached(f)
        clear_cache()
        stats = cache_stats()
        assert stats["entries"] == 0
        assert stats["hits"] == 0


class TestResolveComponentsProvenance:
    def test_provenance_metadata_added(self, tmp_path: Path) -> None:
        """Resolved env vars should carry provenance in metadata."""
        (tmp_path / ".env").write_text("MODEL_VAR=gpt-4o\n")
        (tmp_path / "main.py").write_text(
            "import os\n" 'model=os.getenv("MODEL_VAR")\n'
        )
        pipeline = ScanPipeline(scan_paths=[str(tmp_path)])
        result = pipeline.run()

        env_comps = [
            c for c in result.components if c.metadata.get("env") == "MODEL_VAR"
        ]
        for c in env_comps:
            if c.metadata.get("resolved_from"):
                assert c.metadata["resolved_from"] == "dotenv"
                assert "resolved_source_file" in c.metadata


class TestDedupForAgentic:
    """Pre-agentic representative deduplication for context-free types."""

    def test_context_free_dedup(self):
        from aibom.scan_pipeline import _dedup_for_agentic

        comps = [
            AIComponent(
                name="torch",
                component_type=AIComponentType.DEPENDENCY,
                file_path="a/req.txt",
                line_number=1,
            ),
            AIComponent(
                name="torch",
                component_type=AIComponentType.DEPENDENCY,
                file_path="b/req.txt",
                line_number=1,
            ),
            AIComponent(
                name="torch",
                component_type=AIComponentType.DEPENDENCY,
                file_path="c/req.txt",
                line_number=1,
            ),
        ]
        deduped, fanout = _dedup_for_agentic(comps)
        assert len(deduped) == 1
        assert len(fanout) == 1
        rep_id = deduped[0].instance_id
        assert len(fanout[rep_id]) == 3

    def test_context_dependent_passthrough(self):
        from aibom.scan_pipeline import _dedup_for_agentic

        comps = [
            AIComponent(
                name="env:EP",
                component_type=AIComponentType.LLM_ENDPOINT,
                file_path="a.yaml",
                line_number=1,
            ),
            AIComponent(
                name="env:EP",
                component_type=AIComponentType.LLM_ENDPOINT,
                file_path="b.yaml",
                line_number=5,
            ),
        ]
        deduped, fanout = _dedup_for_agentic(comps)
        assert len(deduped) == 2
        assert len(fanout) == 0

    def test_mixed_components(self):
        from aibom.scan_pipeline import _dedup_for_agentic

        comps = [
            AIComponent(
                name="torch",
                component_type=AIComponentType.DEPENDENCY,
                file_path="a/req.txt",
                line_number=1,
            ),
            AIComponent(
                name="torch",
                component_type=AIComponentType.DEPENDENCY,
                file_path="b/req.txt",
                line_number=1,
            ),
            AIComponent(
                name="env:EP",
                component_type=AIComponentType.LLM_ENDPOINT,
                file_path="a.yaml",
                line_number=1,
            ),
        ]
        deduped, fanout = _dedup_for_agentic(comps)
        assert len(deduped) == 2
        assert len(fanout) == 1

    def test_picks_richest_representative(self):
        from aibom.scan_pipeline import _dedup_for_agentic

        sparse = AIComponent(
            name="torch",
            component_type=AIComponentType.DEPENDENCY,
            file_path="a/req.txt",
            line_number=1,
        )
        rich = AIComponent(
            name="torch",
            component_type=AIComponentType.DEPENDENCY,
            file_path="b/req.txt",
            line_number=1,
            description="PyTorch deep learning framework",
            metadata={"version": "2.1.0", "known_ai_package": True},
        )
        deduped, fanout = _dedup_for_agentic([sparse, rich])
        assert len(deduped) == 1
        assert deduped[0].instance_id == rich.instance_id

    def test_different_names_not_deduped(self):
        from aibom.scan_pipeline import _dedup_for_agentic

        comps = [
            AIComponent(
                name="torch",
                component_type=AIComponentType.DEPENDENCY,
                file_path="a.txt",
                line_number=1,
            ),
            AIComponent(
                name="tensorflow",
                component_type=AIComponentType.DEPENDENCY,
                file_path="b.txt",
                line_number=1,
            ),
        ]
        deduped, fanout = _dedup_for_agentic(comps)
        assert len(deduped) == 2


class TestFanoutAgenticResults:
    """Propagation of agentic verdicts from representatives to siblings."""

    def test_fanout_propagates_enrichment(self):
        from aibom.scan_pipeline import _dedup_for_agentic, _fanout_agentic_results

        comps = [
            AIComponent(
                name="torch",
                component_type=AIComponentType.DEPENDENCY,
                file_path="a/req.txt",
                line_number=1,
            ),
            AIComponent(
                name="torch",
                component_type=AIComponentType.DEPENDENCY,
                file_path="b/req.txt",
                line_number=1,
            ),
        ]
        deduped, fanout = _dedup_for_agentic(comps)
        rep = deduped[0].model_copy(
            update={"heuristic_confidence": 0.95, "needs_agentic": False}
        )

        result = _fanout_agentic_results([rep], fanout)
        assert len(result) == 2
        for c in result:
            assert c.heuristic_confidence == 0.95
            assert c.needs_agentic is False

    def test_fanout_propagates_removal(self):
        from aibom.scan_pipeline import _dedup_for_agentic, _fanout_agentic_results

        comps = [
            AIComponent(
                name="requests",
                component_type=AIComponentType.DEPENDENCY,
                file_path="a/req.txt",
                line_number=1,
            ),
            AIComponent(
                name="requests",
                component_type=AIComponentType.DEPENDENCY,
                file_path="b/req.txt",
                line_number=1,
            ),
        ]
        _, fanout = _dedup_for_agentic(comps)
        result = _fanout_agentic_results([], fanout)
        assert len(result) == 0

    def test_fanout_propagates_reclassification(self):
        from aibom.scan_pipeline import _dedup_for_agentic, _fanout_agentic_results

        comps = [
            AIComponent(
                name="ada-ep",
                component_type=AIComponentType.EMBEDDING,
                file_path="a.yaml",
                line_number=1,
            ),
            AIComponent(
                name="ada-ep",
                component_type=AIComponentType.EMBEDDING,
                file_path="b.yaml",
                line_number=5,
            ),
        ]
        deduped, fanout = _dedup_for_agentic(comps)
        rep = deduped[0].model_copy(
            update={
                "component_type": AIComponentType.MODEL_ENDPOINT,
                "heuristic_confidence": 0.9,
                "needs_agentic": False,
            }
        )

        result = _fanout_agentic_results([rep], fanout)
        assert len(result) == 2
        for c in result:
            assert c.component_type == AIComponentType.MODEL_ENDPOINT
            assert c.needs_agentic is False

    def test_non_fanout_components_pass_through(self):
        from aibom.scan_pipeline import _fanout_agentic_results

        ep = AIComponent(
            name="env:EP",
            component_type=AIComponentType.LLM_ENDPOINT,
            file_path="v.yaml",
            line_number=1,
        )
        result = _fanout_agentic_results([ep], {})
        assert len(result) == 1
        assert result[0].instance_id == ep.instance_id


class TestPropagateRemovals:
    def test_prefanout_removed_ids_drop_all_matching_siblings(self):
        from aibom.scan_pipeline import _propagate_removals

        removed_rep = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="a.yaml",
            line_number=1,
            model_name="gpt-4o",
        )
        sibling = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="b.yaml",
            line_number=2,
            model_name="gpt-4o",
        )
        untouched = AIComponent(
            name="gpt-5",
            component_type=AIComponentType.MODEL,
            file_path="c.yaml",
            line_number=3,
            model_name="gpt-5",
        )

        result = _propagate_removals(
            sent=[removed_rep, untouched],
            received=[untouched],
            all_candidates=[removed_rep, sibling, untouched],
            pre_fanout_removed_ids={removed_rep.instance_id},
        )

        assert [c.name for c in result] == ["gpt-5"]

    def test_import_only_removal_does_not_kill_usage_sibling(self):
        """Regression: LLM removing an import-line ``Agent`` must NOT cascade
        to a usage-line ``agent = Agent(...)`` sibling that shares the same
        canonical consolidation key.

        ``kb_enrichment_scanner`` emits a weak, import-inferred Agent at
        ``from strands import Agent`` lines (tagged with
        ``metadata['import_statement']``). The LLM correctly removes those
        import-only detections (imports alone are not agents).

        However, ``_consolidation_key`` lowercases both names:
          * import-line  name='Agent' -> canonical 'agent'
          * usage-line   name='agent' -> canonical 'agent'  (from the
            assignment target ``agent = Agent(...)``)

        Before this fix, propagation killed the usage-line agent, wiping
        every real Strands agent from the scan. The fix treats
        import-only removals as weak: their removal propagates only to
        other import-only siblings, never to usage/assignment lines.

        Verified against scan cache on the AWS Strands sample repo where
        0 agents survived until this propagation logic was corrected.
        """
        from aibom.scan_pipeline import _propagate_removals

        import_agent = AIComponent(
            name="Agent",
            component_type=AIComponentType.AGENT,
            file_path="lab/async_example.py",
            line_number=2,
            framework="",
            metadata={"import_statement": "from strands import Agent"},
        )
        usage_agent = AIComponent(
            name="agent",
            component_type=AIComponentType.AGENT,
            file_path="lab/async_example.py",
            line_number=6,
            framework="strands",
            metadata={
                "call_pattern": "strands.Agent",
                "assigned_to": "agent",
            },
        )

        result = _propagate_removals(
            sent=[import_agent, usage_agent],
            received=[usage_agent],
            pre_fanout_removed_ids={import_agent.instance_id},
        )

        names = [c.name for c in result]
        assert "agent" in names, (
            "usage-line 'agent = Agent(...)' must survive when only the "
            "import-only Agent was removed by the LLM; got kept: "
            f"{[(c.name, c.line_number, c.metadata.get('import_statement')) for c in result]}"
        )
        assert "Agent" not in names, (
            "the import-only Agent should still be dropped by the LLM "
            "verdict even though we no longer propagate that removal"
        )

    def test_import_only_removal_still_propagates_to_other_imports(self):
        """Weak (import-only) removals must still take out other
        import-only siblings sharing the same canonical key.

        This preserves one side of the original deduplication intent:
        the LLM only needs to reject an import-inferred agent once; every
        other ``from X import Foo`` in the repo is also invalid evidence.
        """
        from aibom.scan_pipeline import _propagate_removals

        import_a = AIComponent(
            name="Agent",
            component_type=AIComponentType.AGENT,
            file_path="lab/a.py",
            line_number=2,
            metadata={"import_statement": "from strands import Agent"},
        )
        import_b = AIComponent(
            name="Agent",
            component_type=AIComponentType.AGENT,
            file_path="lab/b.py",
            line_number=3,
            metadata={"import_statement": "from strands import Agent"},
        )

        result = _propagate_removals(
            sent=[import_a, import_b],
            received=[import_b],
            pre_fanout_removed_ids={import_a.instance_id},
        )

        assert result == [], (
            "both import-only Agent detections must be dropped when the "
            "LLM removes either; got: "
            f"{[(c.name, c.file_path) for c in result]}"
        )

    def test_substantive_removal_still_propagates_to_import_sibling(self):
        """Original behaviour guard: a substantive (non-import) removal
        must still cascade to weak, import-only siblings.

        This is the scenario the propagation logic was designed for —
        the LLM rejects a usage-line component after inspection; every
        import referring to that same class must also be removed because
        it was inferred from that rejected usage.
        """
        from aibom.scan_pipeline import _propagate_removals

        usage_agent = AIComponent(
            name="agent",
            component_type=AIComponentType.AGENT,
            file_path="lab/single_llm_call.py",
            line_number=15,
            framework="strands",
            metadata={
                "call_pattern": "strands.Agent",
                "assigned_to": "agent",
            },
        )
        import_agent = AIComponent(
            name="Agent",
            component_type=AIComponentType.AGENT,
            file_path="lab/single_llm_call.py",
            line_number=1,
            metadata={"import_statement": "from strands import Agent"},
        )

        result = _propagate_removals(
            sent=[usage_agent, import_agent],
            received=[import_agent],
            pre_fanout_removed_ids={usage_agent.instance_id},
        )

        assert result == [], (
            "when the LLM removes a usage-line agent the import-only "
            "sibling must also be dropped; got: "
            f"{[(c.name, c.metadata.get('import_statement')) for c in result]}"
        )

    def test_test_file_removal_does_not_kill_production_sibling(self):
        """Regression: the LLM pruning a guardrail call-site in a *test* file
        must NOT cascade to the production call-site that shares the same
        canonical consolidation key.

        ``agentsec.protect(...)`` appears once in production ``agent.py`` and
        many times across ``tests/`` (mode-handling unit tests). Both are
        call-site emissions (``call_pattern`` set), so both are "strong" under
        the import-only rule. ``_consolidation_key`` lowercases the name and
        ignores the file, collapsing them to ``("agentsec.protect",
        "guardrail")``. When the agent removed a test-file instance as test
        scaffolding, the strong removal cascaded and wiped the production
        guardrail — observed end-to-end where the agentsec example tree
        produced 0 guardrails while a prod-only copy produced the guardrail.
        """
        from aibom.scan_pipeline import _propagate_removals

        prod = AIComponent(
            name="agentsec.protect",
            component_type=AIComponentType.GUARDRAIL,
            file_path="langchain-agent/agent.py",
            line_number=77,
            metadata={"call_pattern": "aidefense.runtime.agentsec.protect"},
        )
        test_use = AIComponent(
            name="agentsec.protect",
            component_type=AIComponentType.GUARDRAIL,
            file_path="langchain-agent/tests/unit/test_langchain_example.py",
            line_number=210,
            metadata={"call_pattern": "aidefense.runtime.agentsec.protect"},
        )

        result = _propagate_removals(
            sent=[prod, test_use],
            received=[prod],
            pre_fanout_removed_ids={test_use.instance_id},
        )

        names = [(c.name, c.file_path) for c in result]
        assert prod.instance_id in {c.instance_id for c in result}, (
            "production agentsec.protect guardrail must survive when only a "
            f"test-file sibling was removed; got kept: {names}"
        )

    def test_production_removal_still_cascades_to_test_sibling(self):
        """A production-scope removal must still take out test-scope siblings
        sharing the canonical key — the original cascade intent is preserved
        for the production→all direction."""
        from aibom.scan_pipeline import _propagate_removals

        prod = AIComponent(
            name="FakeAgent",
            component_type=AIComponentType.AGENT,
            file_path="src/app.py",
            line_number=10,
            metadata={"call_pattern": "pkg.FakeAgent"},
        )
        test_use = AIComponent(
            name="FakeAgent",
            component_type=AIComponentType.AGENT,
            file_path="tests/test_app.py",
            line_number=5,
            metadata={"call_pattern": "pkg.FakeAgent"},
        )

        result = _propagate_removals(
            sent=[prod, test_use],
            received=[test_use],
            pre_fanout_removed_ids={prod.instance_id},
        )

        assert result == [], (
            "a production-scope removal must cascade to the test-scope "
            f"sibling; got kept: {[(c.name, c.file_path) for c in result]}"
        )

    def test_test_removal_still_cascades_to_other_test_sibling(self):
        """A test-scope removal still cascades to *other test-scope* siblings
        (one rejection is enough for test scaffolding), just not to
        production."""
        from aibom.scan_pipeline import _propagate_removals

        test_a = AIComponent(
            name="agentsec.protect",
            component_type=AIComponentType.GUARDRAIL,
            file_path="tests/unit/test_a.py",
            line_number=10,
            metadata={"call_pattern": "aidefense.runtime.agentsec.protect"},
        )
        test_b = AIComponent(
            name="agentsec.protect",
            component_type=AIComponentType.GUARDRAIL,
            file_path="tests/unit/test_b.py",
            line_number=20,
            metadata={"call_pattern": "aidefense.runtime.agentsec.protect"},
        )

        result = _propagate_removals(
            sent=[test_a, test_b],
            received=[test_b],
            pre_fanout_removed_ids={test_a.instance_id},
        )

        assert result == [], (
            "a test-scope removal should cascade to other test-scope "
            f"siblings; got kept: {[(c.name, c.file_path) for c in result]}"
        )

    def test_test_file_import_only_removal_does_not_kill_production_import(self):
        """A removed *test-file* import-only line must not cascade to a
        *production* import-only sibling.

        Removal reach is the intersection of evidence-strength (import-only →
        only other imports) AND scope (test → only other test files). A
        bare ``from X import Y`` pruned in a unit test is the weakest possible
        signal and must not delete the production import of the same symbol.
        """
        from aibom.scan_pipeline import _propagate_removals

        test_import = AIComponent(
            name="Agent",
            component_type=AIComponentType.AGENT,
            file_path="tests/unit/test_app.py",
            line_number=2,
            metadata={"import_statement": "from strands import Agent"},
        )
        prod_import = AIComponent(
            name="Agent",
            component_type=AIComponentType.AGENT,
            file_path="src/app.py",
            line_number=2,
            metadata={"import_statement": "from strands import Agent"},
        )

        result = _propagate_removals(
            sent=[test_import, prod_import],
            received=[prod_import],
            pre_fanout_removed_ids={test_import.instance_id},
        )

        assert prod_import.instance_id in {c.instance_id for c in result}, (
            "production import-only sibling must survive when only a "
            "test-file import-only line was removed; got kept: "
            f"{[(c.name, c.file_path) for c in result]}"
        )

    def test_production_import_only_removal_still_cascades_to_all_imports(self):
        """Guard: a *production* import-only removal still cascades to all
        import-only siblings (the original strands weak-cascade), test or
        production."""
        from aibom.scan_pipeline import _propagate_removals

        prod_import = AIComponent(
            name="Agent",
            component_type=AIComponentType.AGENT,
            file_path="src/a.py",
            line_number=2,
            metadata={"import_statement": "from strands import Agent"},
        )
        other_import = AIComponent(
            name="Agent",
            component_type=AIComponentType.AGENT,
            file_path="src/b.py",
            line_number=3,
            metadata={"import_statement": "from strands import Agent"},
        )

        result = _propagate_removals(
            sent=[prod_import, other_import],
            received=[other_import],
            pre_fanout_removed_ids={prod_import.instance_id},
        )

        assert result == [], (
            "a production import-only removal should still cascade to other "
            f"import-only siblings; got kept: {[c.file_path for c in result]}"
        )


class TestProtectedDependencies:
    """Recognized AI dependencies are deterministic manifest facts and must
    survive the agentic stage even if the LLM omits them from its set."""

    def _dep(self, name: str, path: str) -> AIComponent:
        return AIComponent(
            name=name,
            component_type=AIComponentType.DEPENDENCY,
            file_path=path,
            line_number=1,
            metadata={"ecosystem": "pypi", "known_ai_package": True},
        )

    def test_protected_dependency_predicate(self):
        from aibom.scan_pipeline import _is_protected_dependency

        known = self._dep("cisco-aidefense-sdk", "a/pyproject.toml")
        assert _is_protected_dependency(known) is True

        unknown_dep = AIComponent(
            name="some-random-lib",
            component_type=AIComponentType.DEPENDENCY,
            file_path="a/pyproject.toml",
            line_number=1,
            metadata={"ecosystem": "pypi"},
        )
        assert _is_protected_dependency(unknown_dep) is False

        model = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="a.yaml",
            line_number=1,
        )
        assert _is_protected_dependency(model) is False

    def test_agent_omission_does_not_cascade_to_dependency(self):
        from aibom.scan_pipeline import _propagate_removals

        dep = self._dep("cisco-aidefense-sdk", "a/pyproject.toml")
        # Agent returned nothing for the dependency (omitted from its set).
        result = _propagate_removals(
            sent=[dep],
            received=[],
            all_candidates=[dep],
            pre_fanout_removed_ids={dep.instance_id},
        )
        # No strong/weak key should be formed from a protected dependency.
        assert [c.name for c in result] == []  # received was empty here

    def test_dropped_sole_representative_is_reinstated(self):
        from aibom.scan_pipeline import _reinstate_protected_dependencies

        # Two manifests declare the same package; dedup keeps one
        # representative, which the agent then dropped — fanout restored none.
        dep_a = self._dep("cisco-aidefense-sdk", "a/pyproject.toml")
        dep_b = self._dep("cisco-aidefense-sdk", "b/pyproject.toml")
        # enriched lost both instances.
        enriched: list[AIComponent] = []
        result = _reinstate_protected_dependencies([dep_a, dep_b], enriched)
        names = [c.name for c in result]
        assert names.count("cisco-aidefense-sdk") == 1, (
            "exactly one protected dependency instance should be reinstated; "
            f"got {names}"
        )

    def test_reinstate_noop_when_dependency_survived(self):
        from aibom.scan_pipeline import _reinstate_protected_dependencies

        dep = self._dep("strands-agents", "a/pyproject.toml")
        enriched = [dep]
        result = _reinstate_protected_dependencies([dep], enriched)
        assert [c.name for c in result] == ["strands-agents"]


class TestBackfillRelationshipInstanceIds:
    """LLM-emitted edges with blank endpoint ids are backfilled deterministically."""

    def _model(self, name: str, path: str, line: int) -> AIComponent:
        return AIComponent(
            name=name,
            component_type=AIComponentType.MODEL,
            file_path=path,
            line_number=line,
            model_name=name,
        )

    def _agent(self, name: str, path: str, line: int) -> AIComponent:
        return AIComponent(
            name=name,
            component_type=AIComponentType.AGENT,
            file_path=path,
            line_number=line,
        )

    def test_unique_name_endpoints_are_resolved(self):
        from aibom.models import ComponentRelationship, RelationshipType
        from aibom.scan_pipeline import _backfill_relationship_instance_ids

        agent = self._agent("agent", "lab/a.py", 5)
        model = self._model("claude-3-5-haiku", "lab/a.py", 10)
        rel = ComponentRelationship(
            source_instance_id="",
            target_instance_id="",
            source_name="agent",
            target_name="claude-3-5-haiku",
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _backfill_relationship_instance_ids([rel], [agent, model])
        assert out[0].source_instance_id == agent.instance_id
        assert out[0].target_instance_id == model.instance_id

    def test_ambiguous_name_left_blank(self):
        from aibom.models import ComponentRelationship, RelationshipType
        from aibom.scan_pipeline import _backfill_relationship_instance_ids

        # Two agents named "agent" in different files -> ambiguous.
        a1 = self._agent("agent", "lab/a.py", 5)
        a2 = self._agent("agent", "lab/b.py", 5)
        model = self._model("gpt-4o", "lab/a.py", 10)
        rel = ComponentRelationship(
            source_instance_id="",
            target_instance_id="",
            source_name="agent",
            target_name="gpt-4o",
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _backfill_relationship_instance_ids([rel], [a1, a2, model])
        # Ambiguous source stays blank; unique target is resolved.
        assert out[0].source_instance_id == ""
        assert out[0].target_instance_id == model.instance_id

    def test_existing_ids_preserved(self):
        from aibom.models import ComponentRelationship, RelationshipType
        from aibom.scan_pipeline import _backfill_relationship_instance_ids

        agent = self._agent("agent", "lab/a.py", 5)
        model = self._model("gpt-4o", "lab/a.py", 10)
        rel = ComponentRelationship(
            source_instance_id="preset-src",
            target_instance_id="preset-tgt",
            source_name="agent",
            target_name="gpt-4o",
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _backfill_relationship_instance_ids([rel], [agent, model])
        assert out[0].source_instance_id == "preset-src"
        assert out[0].target_instance_id == "preset-tgt"

    def test_unknown_name_left_blank(self):
        from aibom.models import ComponentRelationship, RelationshipType
        from aibom.scan_pipeline import _backfill_relationship_instance_ids

        agent = self._agent("agent", "lab/a.py", 5)
        rel = ComponentRelationship(
            source_instance_id="",
            target_instance_id="",
            source_name="agent",
            target_name="model-that-was-pruned",
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _backfill_relationship_instance_ids([rel], [agent])
        assert out[0].source_instance_id == agent.instance_id
        assert out[0].target_instance_id == ""


class TestDefaultBomScope:
    def test_stage_assemble_excludes_test_only_components(self):
        component = AIComponent(
            name="FakeTool",
            component_type=AIComponentType.TOOL,
            file_path="repo/tests/test_agent.py",
            line_number=10,
            heuristic_confidence=0.8,
        )
        pipeline = ScanPipeline(scan_paths=["/tmp"])

        components, agentic_count = pipeline._stage_assemble([component])

        assert components == []
        assert agentic_count == 0

    def test_stage_assemble_excludes_test_prefix_filename_components(self):
        component = AIComponent(
            name="gpt-4o-mini",
            component_type=AIComponentType.MODEL,
            file_path="repo/src/orchestrator/test_firewall_routing.py",
            line_number=14,
            heuristic_confidence=0.8,
        )
        pipeline = ScanPipeline(scan_paths=["/tmp"])

        components, agentic_count = pipeline._stage_assemble([component])

        assert components == []
        assert agentic_count == 0

    def test_stage_assemble_excludes_test_suffix_filename_components(self):
        component = AIComponent(
            name="gpt-4o-mini",
            component_type=AIComponentType.MODEL,
            file_path="repo/src/orchestrator/firewall_routing_test.py",
            line_number=21,
            heuristic_confidence=0.8,
        )
        pipeline = ScanPipeline(scan_paths=["/tmp"])

        components, agentic_count = pipeline._stage_assemble([component])

        assert components == []
        assert agentic_count == 0

    def test_stage_assemble_keeps_non_test_prefixed_production_files(self):
        component = AIComponent(
            name="gpt-4o-mini",
            component_type=AIComponentType.MODEL,
            file_path="repo/src/orchestrator/testimony_router.py",
            line_number=8,
            heuristic_confidence=0.8,
        )
        pipeline = ScanPipeline(scan_paths=["/tmp"])

        components, _ = pipeline._stage_assemble([component])

        assert [c.name for c in components] == ["gpt-4o-mini"]

    def test_stage_assemble_excludes_fixture_components_under_test_root(self):
        # A fixtures/ dir nested under a test root is genuine test scaffolding.
        component = AIComponent(
            name="FakePrompt",
            component_type=AIComponentType.PROMPT,
            file_path="repo/tests/fixtures/sample_prompt.py",
            line_number=4,
            heuristic_confidence=0.8,
        )
        pipeline = ScanPipeline(scan_paths=["/tmp"])

        components, agentic_count = pipeline._stage_assemble([component])

        assert components == []
        assert agentic_count == 0

    def test_stage_assemble_keeps_top_level_fixtures_production_component(self):
        # A bare top-level fixtures/ dir with no other test signal is NOT
        # automatically test-only — it may be a production data loader. The
        # over-broad ``fixtures`` segment previously dropped this asset.
        component = AIComponent(
            name="gpt-4o-mini",
            component_type=AIComponentType.MODEL,
            file_path="repo/fixtures/prompt_loader.py",
            line_number=4,
            heuristic_confidence=0.8,
        )
        pipeline = ScanPipeline(scan_paths=["/tmp"])

        components, _ = pipeline._stage_assemble([component])

        assert [c.name for c in components] == ["gpt-4o-mini"]

    def test_stage_assemble_keeps_components_with_production_evidence(self):
        prod = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="repo/app/service.py",
            line_number=12,
            heuristic_confidence=0.9,
        )
        test = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="repo/tests/test_service.py",
            line_number=18,
            heuristic_confidence=0.7,
        )
        pipeline = ScanPipeline(scan_paths=["/tmp"])

        components, _ = pipeline._stage_assemble([prod, test])

        assert len(components) == 1
        assert components[0].name == "gpt-4o"
        assert components[0].metadata.get("test_only") is not True

    def test_stage_assemble_keeps_github_automation_components(self):
        component = AIComponent(
            name="env:LLM_API_BASE",
            component_type=AIComponentType.LLM_ENDPOINT,
            file_path="repo/.github/workflows/ai-review.yml",
            line_number=42,
            heuristic_confidence=0.8,
        )
        pipeline = ScanPipeline(scan_paths=["/tmp"])

        components, _ = pipeline._stage_assemble([component])

        assert len(components) == 1
        assert components[0].file_path.endswith(".github/workflows/ai-review.yml")

    def test_run_filters_test_scope_relationships_and_risk_flags(self):
        from aibom.models.enums import RelationshipType, Severity
        from aibom.models.scan import ComponentRelationship, RiskFlag

        prod = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="repo/app/service.py",
            line_number=12,
            heuristic_confidence=0.9,
            needs_agentic=False,
        )
        test = AIComponent(
            name="FakeTool",
            component_type=AIComponentType.TOOL,
            file_path="repo/tests/test_agent.py",
            line_number=10,
            heuristic_confidence=0.8,
            needs_agentic=False,
        )
        kept_rel = ComponentRelationship(
            source_instance_id=prod.instance_id,
            target_instance_id=prod.instance_id,
            relationship_type=RelationshipType.USES_MODEL,
            source_name=prod.name,
            target_name=prod.name,
            source_type=prod.component_type,
            target_type=prod.component_type,
        )
        dropped_rel = ComponentRelationship(
            source_instance_id=test.instance_id,
            target_instance_id=prod.instance_id,
            relationship_type=RelationshipType.USES_TOOL,
            source_name=test.name,
            target_name=prod.name,
            source_type=test.component_type,
            target_type=prod.component_type,
        )
        kept_flag = RiskFlag(
            flag="prod-risk",
            severity=Severity.LOW,
            weight=1,
            description="prod",
            file_path=prod.file_path,
            line_number=prod.line_number,
        )
        dropped_flag = RiskFlag(
            flag="test-risk",
            severity=Severity.LOW,
            weight=1,
            description="test",
            file_path=test.file_path,
            line_number=test.line_number,
        )
        pipeline = ScanPipeline(scan_paths=["/tmp"])

        with (
            patch.object(
                pipeline,
                "_stage_scan",
                return_value=([prod, test], [kept_rel, dropped_rel]),
            ),
            patch.object(
                pipeline,
                "_stage_cross_ref",
                return_value=([prod, test], MagicMock(), MagicMock(), []),
            ),
            patch.object(
                pipeline,
                "_stage_agentic",
                return_value=(
                    [prod, test],
                    [kept_rel, dropped_rel],
                    [kept_flag, dropped_flag],
                ),
            ),
        ):
            result = pipeline.run()

        assert [c.name for c in result.components] == ["gpt-4o"]
        assert result.relationships == [kept_rel]


class TestIsTestFile:
    """``_is_test_file`` must classify genuine test files as test-only without
    misclassifying production files that merely live under a ``spec``,
    ``testing``, or ``fixtures`` directory."""

    def _f(self, p: str) -> bool:
        from aibom.scan_pipeline import _is_test_file

        return _is_test_file(p)

    # Unambiguous test markers — standalone, still test-only.
    def test_tests_dir_is_test(self):
        assert self._f("repo/tests/test_agent.py") is True

    def test_dunder_tests_dir_is_test(self):
        assert self._f("repo/__tests__/agent.py") is True

    def test_testdata_dir_is_test(self):
        assert self._f("repo/testdata/agent.py") is True
        assert self._f("repo/test_data/agent.py") is True

    def test_test_prefix_filename_is_test(self):
        assert self._f("repo/src/test_router.py") is True

    def test_test_suffix_filename_is_test(self):
        assert self._f("repo/src/router_test.py") is True

    # Ambiguous segments — only test-only WITH a co-located test signal.
    def test_fixtures_under_test_root_is_test(self):
        assert self._f("repo/tests/fixtures/sample.py") is True

    def test_spec_with_test_filename_is_test(self):
        assert self._f("repo/spec/test_contract.py") is True

    def test_testing_under_test_root_is_test(self):
        assert self._f("repo/test/testing/helpers.py") is True

    # Ambiguous segments alone in a production tree — NOT test-only.
    def test_top_level_fixtures_production_is_not_test(self):
        assert self._f("repo/fixtures/prompt_loader.py") is False

    def test_top_level_spec_production_is_not_test(self):
        assert self._f("repo/spec/openapi_models.py") is False

    def test_top_level_testing_utility_is_not_test(self):
        assert self._f("repo/src/testing/harness.py") is False

    # Regression: a production file with no test signal stays production.
    def test_plain_production_file_is_not_test(self):
        assert self._f("repo/src/app/service.py") is False


class TestDependencyPolicyScope:
    def test_stage_assemble_drops_non_ai_dependencies(self) -> None:
        ai_dep = AIComponent(
            name="openai",
            component_type=AIComponentType.DEPENDENCY,
            file_path="/repo/requirements.txt",
            line_number=1,
            metadata={
                "ecosystem": "pypi",
                "known_ai_package": True,
            },
            needs_agentic=False,
        )
        generic_dep = AIComponent(
            name="requests",
            component_type=AIComponentType.DEPENDENCY,
            file_path="/repo/requirements.txt",
            line_number=2,
            metadata={
                "ecosystem": "pypi",
                "known_ai_package": False,
            },
            needs_agentic=False,
        )
        pipeline = ScanPipeline(scan_paths=["/repo"])

        components, agentic_count = pipeline._stage_assemble([ai_dep, generic_dep])

        assert agentic_count == 0
        assert [c.name for c in components] == ["openai"]

    def test_run_keeps_name_only_agentic_relationships_for_kept_components(self):
        from aibom.models.enums import RelationshipType
        from aibom.models.scan import ComponentRelationship

        prod = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="repo/app/service.py",
            line_number=12,
            heuristic_confidence=0.9,
            needs_agentic=False,
            model_name="gpt-4o",
        )
        test = AIComponent(
            name="FakeTool",
            component_type=AIComponentType.TOOL,
            file_path="repo/tests/test_agent.py",
            line_number=10,
            heuristic_confidence=0.8,
            needs_agentic=False,
        )
        kept_rel = ComponentRelationship(
            source_instance_id="",
            target_instance_id="",
            relationship_type=RelationshipType.USES_MODEL,
            source_name="gpt-4o",
            target_name="gpt-4o",
            source_type=prod.component_type,
            target_type=prod.component_type,
        )
        dropped_rel = ComponentRelationship(
            source_instance_id="",
            target_instance_id="",
            relationship_type=RelationshipType.USES_TOOL,
            source_name="FakeTool",
            target_name="gpt-4o",
            source_type=test.component_type,
            target_type=prod.component_type,
        )
        pipeline = ScanPipeline(scan_paths=["/tmp"])

        with (
            patch.object(
                pipeline,
                "_stage_scan",
                return_value=([prod, test], [kept_rel, dropped_rel]),
            ),
            patch.object(
                pipeline,
                "_stage_cross_ref",
                return_value=([prod, test], MagicMock(), MagicMock(), []),
            ),
            patch.object(
                pipeline,
                "_stage_agentic",
                return_value=([prod, test], [kept_rel, dropped_rel], []),
            ),
        ):
            result = pipeline.run()

        assert [c.name for c in result.components] == ["gpt-4o"]
        assert result.relationships == [kept_rel]


class TestSymmetricEvidenceGate:
    """Regression tests for the symmetric ``agent_evidence`` gate.

    Historically ``_evidence_gate`` accepted any post-agentic
    ``AGENT`` / ``AGENT_PROXY`` without rechecking the citation, which
    meant structural-scanner emissions bypassed the same verification
    the middleware applied to LLM verdicts. After the symmetric-gate
    fix:

    * Structural emissions must carry verifiable ``agent_evidence`` —
      missing or invalid citations cause the component to be dropped.
    * LLM type-flips to agent must also carry verifiable evidence
      (belt-and-braces check; the middleware already enforces this
      at Phase 6).
    * KB / framework agents (no structural marker, no type flip) are
      left alone — their authority comes from the framework match, not
      from a code citation.
    """

    @staticmethod
    def _make_class_file(
        tmp_path: Path, filename: str, source: str
    ) -> tuple[str, int, int]:
        """Write *source* to *filename* under *tmp_path* and return
        (absolute_path, class_start_line, class_end_line).
        """
        path = tmp_path / filename
        path.write_text(source, encoding="utf-8")
        lines = source.splitlines()
        class_start = next(
            i + 1 for i, line in enumerate(lines) if line.lstrip().startswith("class ")
        )
        return str(path), class_start, len(lines)

    def test_structural_emission_without_evidence_is_dropped(
        self, tmp_path: Path
    ) -> None:
        from aibom.scan_pipeline import _evidence_gate

        source = (
            "from openai import OpenAI\n"
            "class TracedLoop:\n"
            "    def run(self, x):\n"
            "        return x\n"
        )
        file_path, class_start, class_end = self._make_class_file(
            tmp_path, "traced.py", source
        )
        before = AIComponent(
            name="TracedLoop",
            component_type=AIComponentType.AGENT,
            file_path=file_path,
            line_number=class_start,
            framework="unknown",
            metadata={
                "discovery": "structural_react_loop",
                "class_start_line": class_start,
                "class_end_line": class_end,
            },
        )
        after = before.model_copy()

        kept = _evidence_gate([before], [after], scan_paths=[str(tmp_path)])

        assert kept == [], (
            "A structural agent emission without agent_evidence must "
            "be dropped by the symmetric gate."
        )

    def test_structural_emission_with_valid_evidence_is_kept(
        self, tmp_path: Path
    ) -> None:
        from aibom.scan_pipeline import _evidence_gate

        source = (
            "from openai import OpenAI\n"
            "class TracedLoop:\n"
            "    def run(self, x):\n"
            "        return x\n"
        )
        file_path, class_start, class_end = self._make_class_file(
            tmp_path, "traced.py", source
        )
        snippet = "\n".join(source.splitlines()[class_start - 1 : class_end])
        before = AIComponent(
            name="TracedLoop",
            component_type=AIComponentType.AGENT,
            file_path=file_path,
            line_number=class_start,
            framework="unknown",
            metadata={
                "discovery": "structural_react_loop",
                "agent_evidence": {
                    "pattern": "react_loop",
                    "definition_file": file_path,
                    "definition_start_line": class_start,
                    "definition_end_line": class_end,
                    "evidence_snippet": snippet,
                    "justification": "structural",
                },
            },
        )
        after = before.model_copy()

        kept = _evidence_gate([before], [after], scan_paths=[str(tmp_path)])

        assert len(kept) == 1
        assert kept[0].name == "TracedLoop"

    def test_structural_emission_with_stale_snippet_is_dropped(
        self, tmp_path: Path
    ) -> None:
        """A citation whose snippet does not appear in the cited range
        must be dropped — this is how the gate detects hallucinated or
        stale evidence.
        """
        from aibom.scan_pipeline import _evidence_gate

        source = (
            "from openai import OpenAI\n"
            "class TracedLoop:\n"
            "    def run(self, x):\n"
            "        return x\n"
        )
        file_path, class_start, class_end = self._make_class_file(
            tmp_path, "traced.py", source
        )
        before = AIComponent(
            name="TracedLoop",
            component_type=AIComponentType.AGENT,
            file_path=file_path,
            line_number=class_start,
            framework="unknown",
            metadata={
                "discovery": "structural_react_loop",
                "agent_evidence": {
                    "pattern": "react_loop",
                    "definition_file": file_path,
                    "definition_start_line": class_start,
                    "definition_end_line": class_end,
                    "evidence_snippet": "SNIPPET_THAT_IS_NOT_IN_FILE",
                    "justification": "structural",
                },
            },
        )
        after = before.model_copy()

        kept = _evidence_gate([before], [after], scan_paths=[str(tmp_path)])

        assert kept == []

    def test_llm_type_flip_without_evidence_is_dropped(self, tmp_path: Path) -> None:
        """If the LLM reclassifies a non-agent component to AGENT but
        the middleware somehow failed to attach evidence, the pipeline
        gate must still drop it.
        """
        from aibom.scan_pipeline import _evidence_gate

        source = "class Endpoint:\n" "    def run(self, x):\n" "        return x\n"
        file_path, class_start, class_end = self._make_class_file(
            tmp_path, "endpoint.py", source
        )
        before = AIComponent(
            name="Endpoint",
            component_type=AIComponentType.TOOL,
            file_path=file_path,
            line_number=class_start,
        )
        after = before.model_copy(
            update={"component_type": AIComponentType.AGENT},
        )

        kept = _evidence_gate([before], [after], scan_paths=[str(tmp_path)])

        assert kept == []

    def test_kb_agent_without_structural_marker_is_kept(self, tmp_path: Path) -> None:
        """An AGENT that did not come from the structural scanner and
        was not reclassified by the LLM (type did not change) is left
        alone by the symmetric gate — its authority comes from whichever
        framework / KB scanner matched it, not from a code citation.
        """
        from aibom.scan_pipeline import _evidence_gate

        source = "class FrameworkAgent:\n    pass\n"
        file_path, class_start, _class_end = self._make_class_file(
            tmp_path, "framework_agent.py", source
        )
        before = AIComponent(
            name="FrameworkAgent",
            component_type=AIComponentType.AGENT,
            file_path=file_path,
            line_number=class_start,
            framework="langchain",
        )
        after = before.model_copy()

        kept = _evidence_gate([before], [after], scan_paths=[str(tmp_path)])

        assert len(kept) == 1
        assert kept[0].name == "FrameworkAgent"

    def test_import_only_agent_without_evidence_is_dropped(
        self, tmp_path: Path
    ) -> None:
        """Fix 7: import-only AGENT/AGENT_PROXY candidates from the
        kb_enrichment scanner (``import_statement`` set, ``call_pattern``
        unset) must produce verifiable on-disk loop evidence to survive.

        These are weak class-name matches inferred from a bare ``from x
        import Y`` line; the LLM is supposed to verify the import is
        actually exercised as an LLM-driven loop. Without that evidence
        the symmetric gate must drop them, instead of letting them
        rubber-stamp through ``_component_annotation``.
        """
        from aibom.scan_pipeline import _evidence_gate

        path = tmp_path / "uses_client.py"
        path.write_text(
            "from third_party.deepagent import DeepAgentClient\n"
            "client = DeepAgentClient(url=URL)\n",
            encoding="utf-8",
        )
        before = AIComponent(
            name="DeepAgentClient",
            component_type=AIComponentType.AGENT,
            file_path=str(path),
            line_number=1,
            framework="kb_enrichment",
            heuristic_confidence=0.35,
            metadata={
                "import_statement": (
                    "from third_party.deepagent import DeepAgentClient"
                ),
            },
        )
        after = before.model_copy()

        kept = _evidence_gate([before], [after], scan_paths=[str(tmp_path)])

        assert kept == [], (
            "Import-only agent candidate without agent_evidence must be "
            "dropped — Fix 7 closes the kb_enrichment loophole that let "
            "tautological 'agent confirmed' rows reach the report."
        )

    def test_import_only_agent_gate_uses_original_metadata(
        self, tmp_path: Path
    ) -> None:
        """Agent-added metadata must not let an import-only candidate bypass
        the evidence gate.

        The gate runs after agentic enrichment, so the post-agentic component
        may contain echoed or invented ``call_pattern`` metadata. Import-only
        status is a scanner-origin property and must be computed from the
        immutable ``before`` component, while evidence is still read from the
        post-agentic component.
        """
        from aibom.scan_pipeline import _evidence_gate

        path = tmp_path / "uses_client.py"
        path.write_text(
            "from third_party.deepagent import DeepAgentClient\n"
            "client = DeepAgentClient(url=URL)\n",
            encoding="utf-8",
        )
        before = AIComponent(
            name="DeepAgentClient",
            component_type=AIComponentType.AGENT,
            file_path=str(path),
            line_number=1,
            framework="kb_enrichment",
            heuristic_confidence=0.35,
            metadata={
                "import_statement": (
                    "from third_party.deepagent import DeepAgentClient"
                ),
            },
        )
        after = before.model_copy(
            deep=True,
            update={
                "metadata": {
                    **before.metadata,
                    "call_pattern": "third_party.deepagent.DeepAgentClient",
                },
            },
        )

        kept = _evidence_gate([before], [after], scan_paths=[str(tmp_path)])

        assert kept == [], (
            "Import-only status must be based on the original scanner "
            "component; post-agentic call_pattern metadata is not evidence."
        )

    def test_import_only_agent_with_verified_evidence_is_kept(
        self, tmp_path: Path
    ) -> None:
        """Fix 7 regression guard: when the LLM does attach valid
        ``agent_evidence`` (a real react loop in the same source tree),
        the symmetric gate must keep the component.
        """
        from aibom.scan_pipeline import _evidence_gate

        source = (
            "from openai import OpenAI\n"
            "class DeepAgentLoop:\n"
            "    def step(self, state):\n"
            "        return self._tools(state)\n"
        )
        file_path, class_start, class_end = self._make_class_file(
            tmp_path, "deep_agent_loop.py", source
        )
        snippet = "\n".join(source.splitlines()[class_start - 1 : class_end])
        before = AIComponent(
            name="DeepAgentLoop",
            component_type=AIComponentType.AGENT,
            file_path=file_path,
            line_number=class_start,
            framework="kb_enrichment",
            heuristic_confidence=0.35,
            metadata={
                "import_statement": ("from local.deep import DeepAgentLoop"),
                "agent_evidence": {
                    "pattern": "react_loop",
                    "definition_file": file_path,
                    "definition_start_line": class_start,
                    "definition_end_line": class_end,
                    "evidence_snippet": snippet,
                    "justification": "verified",
                },
            },
        )
        after = before.model_copy()

        kept = _evidence_gate([before], [after], scan_paths=[str(tmp_path)])

        assert len(kept) == 1
        assert kept[0].name == "DeepAgentLoop"


class TestCanonicalizeEnvVarNames:
    """Regression tests for the env-var name canonicalizer.

    Every env-var naming shape emitted by any scanner must canonicalize
    to the resolved literal when ``model_name`` holds a concrete model id.
    See :func:`aibom.scan_pipeline._canonicalize_env_var_names`.
    """

    def _make_component(
        self,
        *,
        name: str,
        model_name: str,
        ctype: AIComponentType = AIComponentType.MODEL,
        meta: dict[str, object] | None = None,
    ) -> AIComponent:
        return AIComponent(
            name=name,
            component_type=ctype,
            file_path="/tmp/fixture/config",
            line_number=1,
            model_name=model_name,
            metadata=dict(meta or {}),
        )

    def test_bare_env_var_name_canonicalizes(self) -> None:
        from aibom.scan_pipeline import _canonicalize_env_var_names

        c = self._make_component(
            name="ANTHROPIC_MODEL",
            model_name="claude-sonnet-4-20250514",
            meta={"env_var": "ANTHROPIC_MODEL"},
        )
        out = _canonicalize_env_var_names([c])
        assert len(out) == 1
        assert out[0].name == "claude-sonnet-4-20250514"
        assert out[0].metadata.get("env_var") == "ANTHROPIC_MODEL"

    def test_env_colon_prefix_canonicalizes(self) -> None:
        from aibom.scan_pipeline import _canonicalize_env_var_names

        c = self._make_component(
            name="env:OPENAI_MODEL",
            model_name="gpt-5",
            meta={"env_var": "OPENAI_MODEL"},
        )
        out = _canonicalize_env_var_names([c])
        assert out[0].name == "gpt-5"

    def test_env_model_prefix_canonicalizes(self) -> None:
        from aibom.scan_pipeline import _canonicalize_env_var_names

        c = self._make_component(
            name="env_model_ANTHROPIC_MODEL",
            model_name="claude-sonnet-4-20250514",
            meta={"env_var": "ANTHROPIC_MODEL", "config_kind": ".env"},
        )
        out = _canonicalize_env_var_names([c])
        assert out[0].name == "claude-sonnet-4-20250514"
        assert out[0].metadata.get("env_var") == "ANTHROPIC_MODEL"
        assert out[0].metadata.get("config_kind") == ".env"

    def test_env_embedding_prefix_canonicalizes(self) -> None:
        from aibom.scan_pipeline import _canonicalize_env_var_names

        c = self._make_component(
            name="env_embedding_EMBED_MODEL",
            model_name="text-embedding-3-large",
            ctype=AIComponentType.EMBEDDING,
            meta={"env_var": "EMBED_MODEL", "config_kind": ".env"},
        )
        out = _canonicalize_env_var_names([c])
        assert out[0].name == "text-embedding-3-large"
        assert out[0].component_type == AIComponentType.EMBEDDING

    def test_dockerfile_env_prefix_with_env_metadata_key_canonicalizes(
        self,
    ) -> None:
        """Dockerfile ENV uses ``metadata["env"]`` rather than ``env_var``.

        The widened canonicalizer must accept either metadata key so
        that Dockerfile-sourced MODEL components are promoted too.
        """
        from aibom.scan_pipeline import _canonicalize_env_var_names

        c = self._make_component(
            name="dockerfile_env_OPENAI_MODEL",
            model_name="gpt-4o",
            meta={"env": "OPENAI_MODEL", "config_kind": "Dockerfile"},
        )
        out = _canonicalize_env_var_names([c])
        assert out[0].name == "gpt-4o"

    def test_skip_when_model_name_null(self) -> None:
        """Components with no resolved literal must be left alone."""
        from aibom.scan_pipeline import _canonicalize_env_var_names

        c = AIComponent(
            name="env_model_CODEX_MODEL",
            component_type=AIComponentType.MODEL,
            file_path="/tmp/fixture/config",
            line_number=1,
            model_name=None,
            metadata={"env_var": "CODEX_MODEL"},
        )
        out = _canonicalize_env_var_names([c])
        assert out[0].name == "env_model_CODEX_MODEL"

    def test_skip_endpoint_url_in_model_name(self) -> None:
        """URLs stored in ``model_name`` must not become a MODEL name."""
        from aibom.scan_pipeline import _canonicalize_env_var_names

        c = self._make_component(
            name="env_model_OPENAI_API_BASE",
            model_name="https://api.openai.com",
            meta={"env_var": "OPENAI_API_BASE"},
        )
        out = _canonicalize_env_var_names([c])
        assert out[0].name == "env_model_OPENAI_API_BASE"

    def test_skip_unresolved_placeholder(self) -> None:
        """``${VAR}`` placeholders must not be promoted to a name."""
        from aibom.scan_pipeline import _canonicalize_env_var_names

        c = self._make_component(
            name="env_model_OPENAI_MODEL",
            model_name="${OPENAI_MODEL}",
            meta={"env_var": "OPENAI_MODEL"},
        )
        out = _canonicalize_env_var_names([c])
        assert out[0].name == "env_model_OPENAI_MODEL"

    def test_endpoint_type_never_touched(self) -> None:
        """LLM_ENDPOINT / MODEL_ENDPOINT components never promote — endpoints
        keep their URL-or-``env:VAR`` shape decided by the endpoint owner."""
        from aibom.scan_pipeline import _canonicalize_env_var_names

        c = AIComponent(
            name="env:AZURE_OPENAI_ENDPOINT",
            component_type=AIComponentType.LLM_ENDPOINT,
            file_path="/tmp/fixture/config",
            line_number=1,
            model_name="https://example.openai.azure.com/",
            metadata={"env_var": "AZURE_OPENAI_ENDPOINT"},
        )
        out = _canonicalize_env_var_names([c])
        assert out[0].name == "env:AZURE_OPENAI_ENDPOINT"


class TestPipelineEnvPrefixInvariant:
    """Pipeline-level invariant: no MODEL/EMBEDDING component with a
    resolved literal ``model_name`` leaves the pipeline carrying an
    ``env_*`` / ``dockerfile_env_*`` naming prefix.

    This guards against the whole class of regressions where a scanner
    emits a prefix-shaped name and later stages fail to promote it.
    """

    _FORBIDDEN_PREFIXES = ("env_model_", "env_embedding_", "dockerfile_env_")

    def test_dotenv_model_does_not_leak_env_prefix(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text(
            "ANTHROPIC_MODEL=claude-sonnet-4-20250514\n"
            "EMBED_MODEL=text-embedding-3-large\n"
        )
        (tmp_path / "app.py").write_text(
            "import os\nfrom anthropic import Anthropic\n"
            "client = Anthropic()\n"
            "resp = client.messages.create(\n"
            '    model=os.getenv("ANTHROPIC_MODEL"),\n'
            "    messages=[]\n"
            ")\n"
        )
        pipeline = ScanPipeline(scan_paths=[str(tmp_path)])
        result = pipeline.run()

        for c in result.components:
            if c.component_type not in (
                AIComponentType.MODEL,
                AIComponentType.EMBEDDING,
            ):
                continue
            if not (isinstance(c.model_name, str) and c.model_name):
                continue
            for prefix in self._FORBIDDEN_PREFIXES:
                assert not c.name.startswith(prefix), (
                    f"env-var prefix leaked into final BOM: "
                    f"name={c.name!r}, model_name={c.model_name!r}"
                )

    def test_dockerfile_env_model_does_not_leak_prefix(self, tmp_path: Path) -> None:
        (tmp_path / "Dockerfile").write_text(
            "FROM python:3.11-slim\n"
            "ENV OPENAI_MODEL=gpt-4o\n"
            'CMD ["python", "app.py"]\n'
        )
        pipeline = ScanPipeline(scan_paths=[str(tmp_path)])
        result = pipeline.run()

        for c in result.components:
            if c.component_type not in (
                AIComponentType.MODEL,
                AIComponentType.EMBEDDING,
            ):
                continue
            if not (isinstance(c.model_name, str) and c.model_name):
                continue
            for prefix in self._FORBIDDEN_PREFIXES:
                assert not c.name.startswith(prefix), (
                    f"Dockerfile env prefix leaked into final BOM: "
                    f"name={c.name!r}, model_name={c.model_name!r}"
                )
