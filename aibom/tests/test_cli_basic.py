# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from aibom.cli import app
from aibom.scan_pipeline import PipelineResult

runner = CliRunner()


def test_analyze_requires_output_file_for_json():
    result = runner.invoke(app, ["analyze", "src", "--output-format", "json"])
    assert result.exit_code != 0


def test_analyze_rejects_invalid_output_format():
    result = runner.invoke(app, ["analyze", "src", "--output-format", "bad"])
    assert result.exit_code != 0
    assert "Invalid output format" in result.output


def test_analyze_rejects_legacy_ui_output_format():
    result = runner.invoke(app, ["analyze", "src", "--output-format", "ui"])
    assert result.exit_code != 0
    assert "Invalid output format" in result.output


def test_analyze_rejects_schema_v2_manifest_without_traceback(
    monkeypatch,
    tmp_path,
):
    source_dir = tmp_path / "repo"
    source_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "duckdb_file": "candidate.duckdb",
                "duckdb_sha256": "unused",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AIBOM_MANIFEST_PATH", str(manifest))

    result = runner.invoke(
        app,
        [
            "analyze",
            str(source_dir),
            "--llm-model",
            "not-used",
        ],
    )

    assert result.exit_code == 1
    assert "Knowledge base upgrade required" in result.output
    assert "requires cisco-aibom 2.x" in result.output
    assert "Traceback" not in result.output


def test_kb_request_submits_one_package_with_schema_v2_contract():
    response = {
        "request": {
            "request_id": "00000000-0000-4000-8000-000000000001",
            "state": "queued",
            "disposition": "ready_in_build",
        },
        "coalesced": False,
        "quota_remaining": 9,
        "rate_limit": {"limit": "10", "remaining": "9", "reset": "1234"},
    }
    with patch(
        "aibom.cli.KBManager.request_build",
        return_value=response,
    ) as request_build:
        result = runner.invoke(
            app,
            [
                "kb",
                "request",
                "--ecosystem",
                "pypi",
                "--package",
                "example-sdk",
                "--symbol",
                "example_sdk.Client",
                "--api-key",
                "test-key",
                "--api-base",
                "https://api.example.com",
            ],
        )

    assert result.exit_code == 0
    request_build.assert_called_once_with(
        ecosystem="pypi",
        package_name="example-sdk",
        symbols=["example_sdk.Client"],
        api_key="test-key",
        api_base="https://api.example.com",
    )
    assert "ready_in_build" in result.output
    assert "Quota remaining: 9" in result.output


def test_kb_request_reads_grouped_coverage_gaps_as_one_bulk_request(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "aibom_analysis": {
                    "coverage_gaps": {
                        "packages": [
                            {
                                "ecosystem": "pypi",
                                "package_name": "example-sdk",
                                "symbols": ["example_sdk.Client"],
                            },
                            {
                                "ecosystem": "npm",
                                "package_name": "example-ai",
                                "symbols": ["ExampleAgent"],
                            },
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    response = {
        "request_ids": ["00000000-0000-4000-8000-000000000002"],
        "duplicates": [],
        "rejected": [],
        "quota_remaining": 8,
        "rate_limit": {},
    }
    with patch(
        "aibom.cli.KBManager.request_bulk",
        return_value=response,
    ) as request_bulk:
        result = runner.invoke(
            app,
            [
                "kb",
                "request",
                "--from-scan",
                str(report),
                "--api-key",
                "test-key",
                "--api-base",
                "https://api.example.com",
            ],
        )

    assert result.exit_code == 0
    requests = request_bulk.call_args.args[0]
    assert requests == [
        {
            "ecosystem": "npm",
            "package_name": "example-ai",
            "symbols": ["ExampleAgent"],
        },
        {
            "ecosystem": "pypi",
            "package_name": "example-sdk",
            "symbols": ["example_sdk.Client"],
        },
    ]


def test_analyze_defaults_cache_root_for_scan_and_agentic(tmp_path):
    source_dir = tmp_path / "repo"
    source_dir.mkdir()
    (source_dir / "app.py").write_text("print('hello')\n", encoding="utf-8")
    report = tmp_path / "report.json"
    shared_root = tmp_path / "shared-cache"
    seen: dict[str, object] = {}

    def fake_pipeline_run(self):
        seen["agentic_cache_dir"] = self.agentic_cache_dir
        return PipelineResult(
            components=[],
            relationships=[],
            agentic_risk_flags=[],
            agentic_candidate_count=0,
            external_deps=[],
            timings=[],
            total_elapsed_s=0.0,
        )

    with patch("aibom.cli.ensure_llm_runtime_available"):
        with patch("aibom.cli.resolve_cache_root", return_value=shared_root):
            with patch("aibom.scan_cache.load_cached", return_value=None) as mock_load:
                with patch("aibom.scan_cache.save_cached") as mock_save:
                    with patch(
                        "aibom.scan_pipeline.ScanPipeline.run", fake_pipeline_run
                    ):
                        result = runner.invoke(
                            app,
                            [
                                "analyze",
                                str(source_dir),
                                "--output-format",
                                "json",
                                "--output-file",
                                str(report),
                                "--llm-model",
                                "test-model",
                            ],
                        )

    assert result.exit_code == 0
    assert mock_load.call_args.args[0] == shared_root / "scan"
    assert mock_save.call_args.args[0] == shared_root / "scan"
    assert seen["agentic_cache_dir"] == shared_root / "agentic"


def test_analyze_component_summary_flag_adds_key_to_json_report(tmp_path):
    from aibom.models import AIComponent, AIComponentType

    source_dir = tmp_path / "repo"
    source_dir.mkdir()
    (source_dir / "app.py").write_text("print('hello')\n", encoding="utf-8")
    report = tmp_path / "report.json"

    detected = [
        AIComponent(
            name="router_agent",
            component_type=AIComponentType.AGENT,
            file_path=str(source_dir / "app.py"),
            line_number=12,
        ),
        AIComponent(
            name="calc_tool",
            component_type=AIComponentType.TOOL,
            file_path=str(source_dir / "app.py"),
            line_number=25,
        ),
    ]

    def fake_pipeline_run(self):
        return PipelineResult(
            components=list(detected),
            relationships=[],
            agentic_risk_flags=[],
            agentic_candidate_count=0,
            external_deps=[],
            timings=[],
            total_elapsed_s=0.0,
        )

    with patch("aibom.cli.ensure_llm_runtime_available"):
        with patch("aibom.scan_cache.load_cached", return_value=None):
            with patch("aibom.scan_cache.save_cached"):
                with patch("aibom.scan_pipeline.ScanPipeline.run", fake_pipeline_run):
                    result = runner.invoke(
                        app,
                        [
                            "analyze",
                            str(source_dir),
                            "--output-format",
                            "json",
                            "--output-file",
                            str(report),
                            "--component-summary",
                            "--llm-model",
                            "test-model",
                        ],
                    )

    assert result.exit_code == 0, result.output
    data = json.loads(report.read_text())
    analysis = data["aibom_analysis"]
    assert "component_summary" in analysis
    source_summaries = analysis["component_summary"]
    assert len(source_summaries) == 1
    entries = next(iter(source_summaries.values()))
    assert entries == [
        {
            "component_type": "agent",
            "name": "router_agent",
            "file_path": str(source_dir / "app.py"),
            "line_number": 12,
        },
        {
            "component_type": "tool",
            "name": "calc_tool",
            "file_path": str(source_dir / "app.py"),
            "line_number": 25,
        },
    ]


def test_analyze_without_component_summary_flag_omits_key(tmp_path):
    from aibom.models import AIComponent, AIComponentType

    source_dir = tmp_path / "repo"
    source_dir.mkdir()
    (source_dir / "app.py").write_text("print('hello')\n", encoding="utf-8")
    report = tmp_path / "report.json"

    def fake_pipeline_run(self):
        return PipelineResult(
            components=[
                AIComponent(
                    name="router_agent",
                    component_type=AIComponentType.AGENT,
                    file_path=str(source_dir / "app.py"),
                    line_number=12,
                )
            ],
            relationships=[],
            agentic_risk_flags=[],
            agentic_candidate_count=0,
            external_deps=[],
            timings=[],
            total_elapsed_s=0.0,
        )

    with patch("aibom.cli.ensure_llm_runtime_available"):
        with patch("aibom.scan_cache.load_cached", return_value=None):
            with patch("aibom.scan_cache.save_cached"):
                with patch("aibom.scan_pipeline.ScanPipeline.run", fake_pipeline_run):
                    result = runner.invoke(
                        app,
                        [
                            "analyze",
                            str(source_dir),
                            "--output-format",
                            "json",
                            "--output-file",
                            str(report),
                            "--llm-model",
                            "test-model",
                        ],
                    )

    assert result.exit_code == 0, result.output
    data = json.loads(report.read_text())
    assert "component_summary" not in data["aibom_analysis"]


@patch("aibom.cli.ensure_llm_runtime_available", return_value=None)
@patch("aibom.multi_repo.is_git_url", return_value=True)
@patch("aibom.multi_repo.ClonedRepo")
def test_analyze_records_clone_failures_in_json_output(
    mock_cloned_repo, _mock_is_git_url, _mock_preflight, tmp_path
):
    report = tmp_path / "report.json"
    mock_cloned_repo.return_value.__enter__.side_effect = RuntimeError("network down")

    result = runner.invoke(
        app,
        [
            "analyze",
            "https://github.com/acme/bad-repo.git",
            "--output-format",
            "json",
            "--output-file",
            str(report),
            "--llm-model",
            "test-model",
            "--llm-api-base",
            "http://localhost:11434",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(report.read_text())
    analysis = data["aibom_analysis"]
    assert analysis["metadata"]["error_count"] == 1
    assert analysis["metadata"]["sources_analyzed"] == 1
    assert analysis["metadata"]["sources_with_errors"] == 1
    assert analysis["metadata"]["status"] == "failed"
    assert analysis["errors"] == ["Clone failed: network down"]


def test_analyze_scan_cache_hit_finalizes_per_source_status(tmp_path):
    """Regression: a scan_cache hit must mark the per-source ``summary.status``
    as a terminal value (``completed``).

    Previously the cache-hit branch in ``aibom.cli.analyze`` ``continue``-d
    out of the scan loop without updating ``source_summary``, so the source
    landed in the report with ``status="in_progress"`` even though the run
    overall had ``status="completed"``. That violates the producer contract
    that every per-source status must be terminal at submission time.
    """
    source_dir = tmp_path / "repo"
    source_dir.mkdir()
    (source_dir / "app.py").write_text("print('hello')\n", encoding="utf-8")
    report = tmp_path / "report.json"
    cached_payload = {
        "_v2": True,
        "components": [],
        "relationships": [],
        "_agentic_risk_flags": [],
        "_agentic_candidate_count": 0,
    }
    telemetry = MagicMock(enabled=True)

    def _explode(self):
        raise AssertionError(
            "scan_cache hit must short-circuit ScanPipeline.run; pipeline ran"
        )

    with patch(
        "aibom.agentic_telemetry.create_agentic_telemetry",
        return_value=telemetry,
    ):
        with patch("aibom.cli.ensure_llm_runtime_available"):
            with patch("aibom.scan_cache.load_cached", return_value=cached_payload):
                with patch("aibom.scan_cache.save_cached"):
                    with patch("aibom.scan_pipeline.ScanPipeline.run", _explode):
                        result = runner.invoke(
                            app,
                            [
                                "analyze",
                                str(source_dir),
                                "--output-format",
                                "json",
                                "--output-file",
                                str(report),
                                "--llm-model",
                                "test-model",
                            ],
                        )

    assert result.exit_code == 0, result.output
    data = json.loads(report.read_text(encoding="utf-8"))
    sources = data["aibom_analysis"]["sources"]
    assert sources, "expected at least one source in report"
    terminal = {"completed", "completed_with_errors", "failed", "skipped"}
    for src_key, src in sources.items():
        status = src["summary"]["status"]
        assert status in terminal, (
            f"source {src_key!r} status={status!r} is not terminal "
            "after scan_cache hit"
        )
        assert (
            status == "completed"
        ), f"expected completed after scan_cache hit, got {status!r}"
        assert src["summary"][
            "last_generated_at"
        ], f"source {src_key!r} missing last_generated_at after cache hit"
    summary = telemetry.record_summary.call_args.kwargs
    assert summary["status"] == "cache_hit"
    assert summary["candidate_count"] == 0
    assert summary["candidate_count_available"] is False
    assert summary["final_component_count"] == 0


def test_analyze_pipeline_exception_emits_failed_source_summary(tmp_path):
    source_dir = tmp_path / "repo"
    source_dir.mkdir()
    (source_dir / "app.py").write_text("print('hello')\n", encoding="utf-8")
    report = tmp_path / "report.json"
    telemetry = MagicMock(enabled=True)

    with patch(
        "aibom.agentic_telemetry.create_agentic_telemetry",
        return_value=telemetry,
    ):
        with patch("aibom.cli.ensure_llm_runtime_available"):
            with patch("aibom.scan_cache.load_cached", return_value=None):
                with patch("aibom.scan_cache.save_cached"):
                    with patch(
                        "aibom.scan_pipeline.ScanPipeline.run",
                        side_effect=RuntimeError("scanner failed"),
                    ):
                        result = runner.invoke(
                            app,
                            [
                                "analyze",
                                str(source_dir),
                                "--output-format",
                                "json",
                                "--output-file",
                                str(report),
                                "--llm-model",
                                "test-model",
                            ],
                        )

    assert result.exit_code != 0
    telemetry.record_summary.assert_called_once_with(
        source_id=str(source_dir),
        source_kind="directory",
        status="failed",
    )
    telemetry.drain.assert_called_once_with(timeout_s=2.0)


def test_analyze_unexpected_source_discovery_failure_emits_one_summary(tmp_path):
    source_dir = tmp_path / "repo"
    source_dir.mkdir()
    report = tmp_path / "report.json"
    telemetry = MagicMock(enabled=True)

    with patch(
        "aibom.agentic_telemetry.create_agentic_telemetry",
        return_value=telemetry,
    ):
        with patch("aibom.cli.ensure_llm_runtime_available"):
            with patch(
                "aibom.multi_repo.is_git_url",
                side_effect=OSError("source discovery failed"),
            ):
                result = runner.invoke(
                    app,
                    [
                        "analyze",
                        str(source_dir),
                        "--output-format",
                        "json",
                        "--output-file",
                        str(report),
                        "--llm-model",
                        "test-model",
                    ],
                )

    assert result.exit_code != 0
    telemetry.record_summary.assert_called_once_with(
        source_id=str(source_dir),
        source_kind="unknown",
        status="failed",
    )
    telemetry.drain.assert_called_once_with(timeout_s=2.0)
