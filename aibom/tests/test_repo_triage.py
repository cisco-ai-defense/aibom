# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from aibom.repo_triage import RepoTriager, TriageResult


class TestDeterministicTriage:
    def test_repo_with_ai_package_in_requirements(self, tmp_path: Path):
        repo = tmp_path / "ml-app"
        repo.mkdir()
        (repo / "requirements.txt").write_text("flask==2.0\nlangchain==0.3.1\nrequests\n")
        triager = RepoTriager()
        results = triager.triage_repos([str(repo)])
        assert len(results) == 1
        assert results[0].decision == "deep-scan"
        assert results[0].method == "deterministic"
        assert "langchain" in results[0].reason

    def test_repo_with_ai_package_in_pyproject(self, tmp_path: Path):
        repo = tmp_path / "agent-svc"
        repo.mkdir()
        (repo / "pyproject.toml").write_text(
            '[project]\ndependencies = ["openai>=1.0", "pydantic"]\n'
        )
        triager = RepoTriager()
        results = triager.triage_repos([str(repo)])
        assert results[0].decision == "deep-scan"

    def test_repo_with_ai_package_in_package_json(self, tmp_path: Path):
        repo = tmp_path / "js-ai"
        repo.mkdir()
        (repo / "package.json").write_text(
            '{"dependencies": {"@langchain/core": "^0.2"}}\n'
        )
        triager = RepoTriager()
        results = triager.triage_repos([str(repo)])
        assert results[0].decision == "deep-scan"

    def test_repo_without_ai_packages_needs_clone(self, tmp_path: Path):
        repo = tmp_path / "web-app"
        repo.mkdir()
        (repo / "requirements.txt").write_text("flask==2.0\nrequests\nsqlalchemy\n")
        triager = RepoTriager()
        results = triager.triage_repos([str(repo)])
        assert results[0].decision == "needs-clone"
        assert "agentic mode not available" in results[0].reason

    def test_nonexistent_repo_skipped(self, tmp_path: Path):
        triager = RepoTriager()
        results = triager.triage_repos([str(tmp_path / "missing")])
        assert results[0].decision == "skip"
        assert "does not exist" in results[0].reason

    def test_multiple_repos_mixed(self, tmp_path: Path):
        ai_repo = tmp_path / "ai"
        ai_repo.mkdir()
        (ai_repo / "requirements.txt").write_text("transformers\n")

        web_repo = tmp_path / "web"
        web_repo.mkdir()
        (web_repo / "requirements.txt").write_text("django\n")

        triager = RepoTriager()
        results = triager.triage_repos([str(ai_repo), str(web_repo)])
        decisions = {Path(r.repo_path).name: r.decision for r in results}
        assert decisions["ai"] == "deep-scan"
        assert decisions["web"] == "needs-clone"

    def test_empty_repo_needs_clone(self, tmp_path: Path):
        repo = tmp_path / "empty"
        repo.mkdir()
        triager = RepoTriager()
        results = triager.triage_repos([str(repo)])
        assert results[0].decision == "needs-clone"


class TestBuildRepoSummary:
    def test_includes_readme(self, tmp_path: Path):
        repo = tmp_path / "with-readme"
        repo.mkdir()
        (repo / "README.md").write_text("# AI Project\nThis uses transformers.")
        triager = RepoTriager()
        summary = triager._build_repo_summary(repo)
        assert "README" in summary
        assert "AI Project" in summary

    def test_includes_manifest(self, tmp_path: Path):
        repo = tmp_path / "with-manifest"
        repo.mkdir()
        (repo / "requirements.txt").write_text("openai==1.5\n")
        triager = RepoTriager()
        summary = triager._build_repo_summary(repo)
        assert "requirements.txt" in summary
        assert "openai" in summary

    def test_lists_top_level_files_if_no_readme_or_manifest(self, tmp_path: Path):
        repo = tmp_path / "bare"
        repo.mkdir()
        (repo / "main.py").write_text("pass")
        (repo / "config.yaml").write_text("key: val")
        triager = RepoTriager()
        summary = triager._build_repo_summary(repo)
        assert "Top-level files" in summary
