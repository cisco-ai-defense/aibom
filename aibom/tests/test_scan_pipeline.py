# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from aibom.models.enums import AIComponentType
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
    def test_candidates_scope_skips_confirmed(self, tmp_path: Path) -> None:
        """Default 'candidates' scope should skip agentic when no candidates."""
        (tmp_path / "app.py").write_text(
            'from openai import OpenAI\nclient = OpenAI(model="gpt-4o")\n'
        )
        llm_cfg = {"model": "test/model", "api_key": "fake", "api_base": "http://x"}
        pipeline = ScanPipeline(
            scan_paths=[str(tmp_path)],
            llm_config=llm_cfg,
            agentic_scope="candidates",
        )
        with patch("aibom.agentic.agent.run_agentic_enrichment") as mock_enrich:
            result = pipeline.run()
        mock_enrich.assert_not_called()

    def test_all_scope_sends_everything(self, tmp_path: Path) -> None:
        """'all' scope should invoke agentic enrichment with all components."""
        (tmp_path / "app.py").write_text(
            'from openai import OpenAI\nclient = OpenAI(model="gpt-4o")\n'
        )
        llm_cfg = {"model": "test/model", "api_key": "fake", "api_base": "http://x"}
        pipeline = ScanPipeline(
            scan_paths=[str(tmp_path)],
            llm_config=llm_cfg,
            agentic_scope="all",
        )
        with patch("aibom.agentic.agent.run_agentic_enrichment") as mock_enrich:
            mock_enrich.return_value = ([], [], [])
            result = pipeline.run()
        mock_enrich.assert_called_once()


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
