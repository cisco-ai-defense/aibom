# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from aibom.repo_triage import RepoTriager, TriageResult


class TestTriageNonexistent:
    def test_nonexistent_repo_skipped(self, tmp_path: Path):
        triager = RepoTriager()
        results = triager.triage_repos([str(tmp_path / "missing")])
        assert results[0].decision == "skip"
        assert "does not exist" in results[0].reason


class TestTriageNoLLM:
    def test_no_llm_defaults_to_deep_scan(self, tmp_path: Path):
        repo = tmp_path / "some-repo"
        repo.mkdir()
        triager = RepoTriager(llm_config=None)
        results = triager.triage_repos([str(repo)])
        assert results[0].decision == "deep-scan"
        assert "safe bias" in results[0].reason


class TestTriageAgentIntegration:
    def test_agent_deep_scan(self, tmp_path: Path):
        repo = tmp_path / "ai-app"
        repo.mkdir()

        triager = RepoTriager(llm_config={"model": "test-model", "api_key": "k"})
        with patch.object(triager, "_triage_single") as mock_triage:
            mock_triage.return_value = TriageResult(
                str(repo), "deep-scan", "uses openai",
                evidence=["requirements.txt"], method="agentic",
            )
            results = triager.triage_repos([str(repo)])

        assert results[0].decision == "deep-scan"
        assert results[0].method == "agentic"

    def test_agent_skip(self, tmp_path: Path):
        repo = tmp_path / "web-app"
        repo.mkdir()

        triager = RepoTriager(llm_config={"model": "test-model", "api_key": "k"})
        with patch.object(triager, "_triage_single") as mock_triage:
            mock_triage.return_value = TriageResult(
                str(repo), "skip", "pure web framework", method="agentic",
            )
            results = triager.triage_repos([str(repo)])

        assert results[0].decision == "skip"
        assert results[0].method == "agentic"

    def test_agent_failure_defaults_deep_scan(self, tmp_path: Path):
        repo = tmp_path / "broken"
        repo.mkdir()

        triager = RepoTriager(llm_config={"model": "test-model", "api_key": "k"})
        with patch.object(triager, "_invoke_with_timeout", side_effect=RuntimeError("LLM timeout")), \
             patch("aibom.llm_factory.build_chat_model", return_value=MagicMock()), \
             patch("deepagents.create_deep_agent", return_value=MagicMock()):
            results = triager.triage_repos([str(repo)])

        assert results[0].decision == "deep-scan"
        assert "failed" in results[0].reason

    def test_multiple_repos(self, tmp_path: Path):
        existing = tmp_path / "exists"
        existing.mkdir()
        missing = tmp_path / "nope"

        triager = RepoTriager(llm_config=None)
        results = triager.triage_repos([str(existing), str(missing)])
        assert len(results) == 2
        assert results[0].decision == "deep-scan"
        assert results[1].decision == "skip"


class TestTriageSingleNoExtras:
    def test_missing_agentic_extras_defaults_deep_scan(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        triager = RepoTriager(llm_config={"model": "test", "api_key": "k"})
        with patch.dict("sys.modules", {"deepagents": None}):
            result = triager._triage_single(str(repo))
        assert result.decision == "deep-scan"


class TestParseResult:
    def test_structured_response_dict(self):
        triager = RepoTriager()
        result = {
            "structured_response": {
                "decision": "skip",
                "reason": "no AI",
                "evidence": ["README.md"],
            }
        }
        tr = triager._parse_result("/repo", result)
        assert tr.decision == "skip"
        assert tr.evidence == ["README.md"]

    def test_message_json_fallback(self):
        triager = RepoTriager()
        result = {
            "messages": [
                MagicMock(content='{"decision": "needs-clone", "reason": "opaque repo"}')
            ]
        }
        tr = triager._parse_result("/repo", result)
        assert tr.decision == "needs-clone"

    def test_invalid_decision_defaults_deep_scan(self):
        triager = RepoTriager()
        result = {
            "structured_response": {
                "decision": "maybe",
                "reason": "unsure",
            }
        }
        tr = triager._parse_result("/repo", result)
        assert tr.decision == "deep-scan"

    def test_unparseable_response_defaults_deep_scan(self):
        triager = RepoTriager()
        result = {"messages": [MagicMock(content="I'm not valid JSON")]}
        tr = triager._parse_result("/repo", result)
        assert tr.decision == "deep-scan"

    def test_empty_result_defaults_deep_scan(self):
        triager = RepoTriager()
        result = {}
        tr = triager._parse_result("/repo", result)
        assert tr.decision == "deep-scan"
