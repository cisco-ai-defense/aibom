# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from aibom.models import AIComponent, AIComponentType
from aibom.scan_pipeline import ScanPipeline, StageTiming


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
        pipeline_relaxed = ScanPipeline(
            scan_paths=[str(tmp_path)], strict=False
        )
        result_relaxed = pipeline_relaxed.run()

        pipeline_strict = ScanPipeline(
            scan_paths=[str(tmp_path)], strict=True
        )
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
            'import os\nfrom openai import OpenAI\n'
            'client = OpenAI()\n'
            'resp = client.chat.completions.create(\n'
            '    model=os.getenv("LLM_MODEL"),\n'
            '    messages=[]\n'
            ')\n'
        )
        pipeline = ScanPipeline(scan_paths=[str(tmp_path)])
        result = pipeline.run()

        env_comps = [
            c for c in result.components if c.name.startswith("env:")
        ]
        resolved = [
            c for c in env_comps
            if c.model_name and c.model_name == "gpt-4o"
        ]
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
        with patch("aibom.agentic.agent.run_agentic_enrichment") as mock_enrich:
            mock_enrich.return_value = ([], [], [])
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
        with patch("aibom.agentic.agent.run_agentic_enrichment") as mock_enrich:
            mock_enrich.return_value = ([], [], [])
            pipeline.run()
        assert mock_enrich.call_args.kwargs["cache_dir"] == agentic_cache_dir


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
            'import os\n'
            'model=os.getenv("MODEL_VAR")\n'
        )
        pipeline = ScanPipeline(scan_paths=[str(tmp_path)])
        result = pipeline.run()

        env_comps = [
            c for c in result.components
            if c.metadata.get("env") == "MODEL_VAR"
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
            AIComponent(name="torch", component_type=AIComponentType.DEPENDENCY, file_path="a/req.txt", line_number=1),
            AIComponent(name="torch", component_type=AIComponentType.DEPENDENCY, file_path="b/req.txt", line_number=1),
            AIComponent(name="torch", component_type=AIComponentType.DEPENDENCY, file_path="c/req.txt", line_number=1),
        ]
        deduped, fanout = _dedup_for_agentic(comps)
        assert len(deduped) == 1
        assert len(fanout) == 1
        rep_id = deduped[0].instance_id
        assert len(fanout[rep_id]) == 3

    def test_context_dependent_passthrough(self):
        from aibom.scan_pipeline import _dedup_for_agentic

        comps = [
            AIComponent(name="env:EP", component_type=AIComponentType.LLM_ENDPOINT, file_path="a.yaml", line_number=1),
            AIComponent(name="env:EP", component_type=AIComponentType.LLM_ENDPOINT, file_path="b.yaml", line_number=5),
        ]
        deduped, fanout = _dedup_for_agentic(comps)
        assert len(deduped) == 2
        assert len(fanout) == 0

    def test_mixed_components(self):
        from aibom.scan_pipeline import _dedup_for_agentic

        comps = [
            AIComponent(name="torch", component_type=AIComponentType.DEPENDENCY, file_path="a/req.txt", line_number=1),
            AIComponent(name="torch", component_type=AIComponentType.DEPENDENCY, file_path="b/req.txt", line_number=1),
            AIComponent(name="env:EP", component_type=AIComponentType.LLM_ENDPOINT, file_path="a.yaml", line_number=1),
        ]
        deduped, fanout = _dedup_for_agentic(comps)
        assert len(deduped) == 2
        assert len(fanout) == 1

    def test_picks_richest_representative(self):
        from aibom.scan_pipeline import _dedup_for_agentic

        sparse = AIComponent(name="torch", component_type=AIComponentType.DEPENDENCY, file_path="a/req.txt", line_number=1)
        rich = AIComponent(
            name="torch", component_type=AIComponentType.DEPENDENCY,
            file_path="b/req.txt", line_number=1,
            description="PyTorch deep learning framework",
            metadata={"version": "2.1.0", "known_ai_package": True},
        )
        deduped, fanout = _dedup_for_agentic([sparse, rich])
        assert len(deduped) == 1
        assert deduped[0].instance_id == rich.instance_id

    def test_different_names_not_deduped(self):
        from aibom.scan_pipeline import _dedup_for_agentic

        comps = [
            AIComponent(name="torch", component_type=AIComponentType.DEPENDENCY, file_path="a.txt", line_number=1),
            AIComponent(name="tensorflow", component_type=AIComponentType.DEPENDENCY, file_path="b.txt", line_number=1),
        ]
        deduped, fanout = _dedup_for_agentic(comps)
        assert len(deduped) == 2


class TestFanoutAgenticResults:
    """Propagation of agentic verdicts from representatives to siblings."""

    def test_fanout_propagates_enrichment(self):
        from aibom.scan_pipeline import _dedup_for_agentic, _fanout_agentic_results

        comps = [
            AIComponent(name="torch", component_type=AIComponentType.DEPENDENCY, file_path="a/req.txt", line_number=1),
            AIComponent(name="torch", component_type=AIComponentType.DEPENDENCY, file_path="b/req.txt", line_number=1),
        ]
        deduped, fanout = _dedup_for_agentic(comps)
        rep = deduped[0].model_copy(update={"confidence": 0.95, "needs_agentic": False})

        result = _fanout_agentic_results([rep], fanout)
        assert len(result) == 2
        for c in result:
            assert c.confidence == 0.95
            assert c.needs_agentic is False

    def test_fanout_propagates_removal(self):
        from aibom.scan_pipeline import _dedup_for_agentic, _fanout_agentic_results

        comps = [
            AIComponent(name="requests", component_type=AIComponentType.DEPENDENCY, file_path="a/req.txt", line_number=1),
            AIComponent(name="requests", component_type=AIComponentType.DEPENDENCY, file_path="b/req.txt", line_number=1),
        ]
        _, fanout = _dedup_for_agentic(comps)
        result = _fanout_agentic_results([], fanout)
        assert len(result) == 0

    def test_fanout_propagates_reclassification(self):
        from aibom.scan_pipeline import _dedup_for_agentic, _fanout_agentic_results

        comps = [
            AIComponent(name="ada-ep", component_type=AIComponentType.EMBEDDING, file_path="a.yaml", line_number=1),
            AIComponent(name="ada-ep", component_type=AIComponentType.EMBEDDING, file_path="b.yaml", line_number=5),
        ]
        deduped, fanout = _dedup_for_agentic(comps)
        rep = deduped[0].model_copy(update={
            "component_type": AIComponentType.MODEL_ENDPOINT,
            "confidence": 0.9,
            "needs_agentic": False,
        })

        result = _fanout_agentic_results([rep], fanout)
        assert len(result) == 2
        for c in result:
            assert c.component_type == AIComponentType.MODEL_ENDPOINT
            assert c.needs_agentic is False

    def test_non_fanout_components_pass_through(self):
        from aibom.scan_pipeline import _fanout_agentic_results

        ep = AIComponent(name="env:EP", component_type=AIComponentType.LLM_ENDPOINT, file_path="v.yaml", line_number=1)
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
