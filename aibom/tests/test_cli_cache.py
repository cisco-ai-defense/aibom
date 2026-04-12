from __future__ import annotations

import json
import time
from pathlib import Path

from typer.testing import CliRunner

from aibom.cli import app
from aibom.models import AIComponent, AIComponentType, ScanResult, SourceResult
from aibom.scan_cache import save_cached

runner = CliRunner()


def test_cache_list_defaults_to_scan_type_under_root(tmp_path: Path):
    save_cached(
        tmp_path / "scan",
        "scan-key-1234567890abcdef",
        {
            "_v2": True,
            "components": [],
            "relationships": [],
            "_agentic_risk_flags": [],
            "_agentic_candidate_count": 0,
        },
    )

    result = runner.invoke(app, ["cache", "list", "--cache-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "scan-key-12345678" in result.output


def test_cache_list_supports_agentic_type(tmp_path: Path):
    cache_file = tmp_path / "agentic" / "tier_abc123.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {
                "_tier_cache_version": 1,
                "tier_enriched": [],
                "tier_new": [],
                "tier_rels": [],
                "tier_flags": [],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["cache", "list", "--type", "agentic", "--cache-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "tier" in result.output.lower()


def test_cache_get_scan_entry_by_prefix(tmp_path: Path):
    key = "scan-key-1234567890abcdef"
    save_cached(
        tmp_path / "scan",
        key,
        {
            "_v2": True,
            "components": [{"name": "router_agent", "component_type": "agent"}],
            "relationships": [],
            "_agentic_risk_flags": [],
            "_agentic_candidate_count": 0,
        },
    )

    result = runner.invoke(
        app,
        ["cache", "get", "scan", "scan-key-1234", "--cache-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert key in result.output
    assert "router_agent" in result.output


def test_cache_get_scan_prefix_ambiguity_fails(tmp_path: Path):
    for key in ("shared-prefix-aaa111", "shared-prefix-bbb222"):
        save_cached(
            tmp_path / "scan",
            key,
            {
                "_v2": True,
                "components": [],
                "relationships": [],
                "_agentic_risk_flags": [],
                "_agentic_candidate_count": 0,
            },
        )

    result = runner.invoke(
        app,
        ["cache", "get", "scan", "shared-prefix", "--cache-dir", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "ambiguous" in result.output.lower()


def test_cache_get_org_entry_by_repo_path_and_sha(tmp_path: Path):
    repo_path = tmp_path / "repos" / "service-a"
    repo_path.mkdir(parents=True, exist_ok=True)
    sha = "deadbeef"

    from aibom.incremental import _repo_bucket_key

    cache_file = tmp_path / "org" / _repo_bucket_key(str(repo_path)) / f"{sha}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            ScanResult(
                metadata={"run_id": "run-1"},
                sources=[
                    SourceResult(
                        path=str(repo_path),
                        components=[
                            AIComponent(
                                name="router_agent",
                                component_type=AIComponentType.AGENT,
                                file_path=str(repo_path / "app.py"),
                                line_number=12,
                            )
                        ],
                        relationships=[],
                    )
                ],
                errors=[],
            ).model_dump(mode="json")
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "cache",
            "get",
            "org",
            str(repo_path),
            "--sha",
            sha,
            "--cache-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert str(repo_path) in result.output
    assert sha in result.output


def test_cache_get_model_entry(tmp_path: Path):
    cache_file = tmp_path / "model" / "model_catalog.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {
                "_ts": time.time(),
                "models": {
                    "gpt-5.4": {"provider": "openai", "source": "model_catalog"}
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["cache", "get", "model", "model_catalog.json", "--cache-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "model_catalog.json" in result.output
    assert "gpt-5.4" in result.output


def test_cache_get_package_entry(tmp_path: Path):
    cache_file = tmp_path / "packages" / "pypi" / "openai.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"name": "openai", "summary": "OpenAI SDK"}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["cache", "get", "packages", "pypi/openai", "--cache-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "pypi/openai" in result.output
    assert "OpenAI SDK" in result.output
