# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
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
                str(repo),
                "deep-scan",
                "uses openai",
                evidence=["requirements.txt"],
                method="agentic",
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
                str(repo),
                "skip",
                "pure web framework",
                method="agentic",
            )
            results = triager.triage_repos([str(repo)])

        assert results[0].decision == "skip"
        assert results[0].method == "agentic"

    def test_triage_passes_max_tokens_to_model(self, tmp_path: Path):
        repo = tmp_path / "ai-app"
        repo.mkdir()

        fake_deepagents = MagicMock()
        triager = RepoTriager(
            llm_config={"model": "test-model", "api_key": "k", "max_tokens": 4242}
        )
        with (
            patch(
                "aibom.llm_factory.build_chat_model", return_value=MagicMock()
            ) as mock_build,
            patch.object(
                triager, "_invoke_with_timeout", side_effect=RuntimeError("stop")
            ),
            patch.dict("sys.modules", {"deepagents": fake_deepagents}),
        ):
            triager.triage_repos([str(repo)])

        _, kwargs = mock_build.call_args
        assert kwargs.get("max_tokens") == 4242

    def test_agent_failure_defaults_deep_scan(self, tmp_path: Path):
        repo = tmp_path / "broken"
        repo.mkdir()

        fake_deepagents = MagicMock()
        triager = RepoTriager(llm_config={"model": "test-model", "api_key": "k"})
        with (
            patch.object(
                triager, "_invoke_with_timeout", side_effect=RuntimeError("LLM timeout")
            ),
            patch("aibom.llm_factory.build_chat_model", return_value=MagicMock()),
            patch.dict("sys.modules", {"deepagents": fake_deepagents}),
        ):
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


class TestTriageInvokeTimeout:
    """A hung triage agent.invoke must be bounded by the daemon-thread deadline
    helper and fail open to deep-scan, never wedging the scan."""

    def test_invoke_with_timeout_bounds_hung_agent(self) -> None:
        never = threading.Event()  # never set -> invoke blocks forever

        def hung_invoke(*_a: object, **_k: object) -> object:
            never.wait()
            return {"messages": []}

        agent = MagicMock()
        agent.invoke.side_effect = hung_invoke

        triager = RepoTriager()
        with patch("aibom.repo_triage._TRIAGE_TIMEOUT_S", 1):
            start = time.monotonic()
            from aibom.agentic.agent import _InvokeTimeout

            try:
                triager._invoke_with_timeout(agent, "triage this repo")
                raised = None
            except _InvokeTimeout as exc:
                raised = exc
            elapsed = time.monotonic() - start

        assert raised is not None, "expected _InvokeTimeout on a hung invoke"
        assert elapsed < 10, f"triage invoke hung for {elapsed:.1f}s"

    def test_hung_triage_degrades_to_deep_scan(self, tmp_path: Path) -> None:
        repo = tmp_path / "ai-app"
        repo.mkdir()

        never = threading.Event()

        def hung_invoke(*_a: object, **_k: object) -> object:
            never.wait()
            return {"messages": []}

        agent = MagicMock()
        agent.invoke.side_effect = hung_invoke
        fake_deepagents = MagicMock()
        fake_deepagents.create_deep_agent.return_value = agent

        triager = RepoTriager(llm_config={"model": "test-model", "api_key": "k"})
        with (
            patch("aibom.repo_triage._TRIAGE_TIMEOUT_S", 1),
            patch("aibom.llm_factory.build_chat_model", return_value=MagicMock()),
            patch("aibom.agentic.tools.build_triage_tools", return_value=[]),
            patch.dict("sys.modules", {"deepagents": fake_deepagents}),
        ):
            start = time.monotonic()
            results = triager.triage_repos([str(repo)])
            elapsed = time.monotonic() - start

        assert elapsed < 15, f"triage hung for {elapsed:.1f}s"
        assert results[0].decision == "deep-scan"
        assert "failed" in results[0].reason


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
                MagicMock(
                    content='{"decision": "needs-clone", "reason": "opaque repo"}'
                )
            ]
        }
        tr = triager._parse_result("/repo", result)
        assert tr.decision == "needs-clone"

    def test_message_list_content_fallback(self):
        # LangChain-style list content (thinking + text blocks) must be
        # normalized to text before JSON scanning, not break the parser.
        triager = RepoTriager()
        msg = MagicMock()
        msg.content = [
            {"type": "thinking", "thinking": "let me look..."},
            {"type": "text", "text": '{"decision": "skip", "reason": "no AI"}'},
        ]
        tr = triager._parse_result("/repo", {"messages": [msg]})
        assert tr.decision == "skip"
        assert tr.reason == "no AI"

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
