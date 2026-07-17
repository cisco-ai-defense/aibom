# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

from aibom.agentic.agent import (
    AgentResponse,
    _agentic_cache_namespace,
    _AgenticResultCache,
    _cross_repo_cache_key,
    _scan_evidence_digest,
)


def _namespace(
    *,
    model: str = "model-a",
    provider: str = "openai",
    reasoning: str = "auto",
    api_key: str = "super-secret-key",
    max_concurrent: int = 1,
    timeout_s: int = 120,
    max_consecutive_failures: int = 3,
    max_retry_seconds: int = 300,
    decision_context_digest: str = "context-a",
) -> str:
    return _agentic_cache_namespace(
        model_string=model,
        fast_model=None,
        llm_config={
            "provider": provider,
            "reasoning": reasoning,
            "api_key": api_key,
        },
        batch_size=5,
        max_concurrent=max_concurrent,
        timeout_s=timeout_s,
        max_consecutive_failures=max_consecutive_failures,
        max_retry_seconds=max_retry_seconds,
        include_code_snippets=False,
        agent_signature_catalog=None,
        decision_context_digest=decision_context_digest,
    )


def test_agentic_cache_namespace_is_stable_and_non_sensitive() -> None:
    first = _namespace()
    second = _namespace(api_key="a-different-secret")

    assert first == second
    assert len(first) == 20
    assert "secret" not in first


def test_agentic_cache_namespace_changes_with_verdict_inputs() -> None:
    baseline = _namespace()

    assert _namespace(model="model-b") != baseline
    assert _namespace(provider="bedrock") != baseline
    assert _namespace(reasoning="off") != baseline
    assert _namespace(max_concurrent=2) != baseline
    assert _namespace(timeout_s=30) != baseline
    assert _namespace(max_consecutive_failures=1) != baseline
    assert _namespace(max_retry_seconds=0) != baseline
    assert _namespace(decision_context_digest="context-b") != baseline


def test_namespaced_cache_does_not_replay_other_model_entries(tmp_path) -> None:
    cache_dir = tmp_path / "agentic"
    first = _AgenticResultCache(cache_dir, namespace=_namespace(model="model-a"))
    first.put("component-key", {"verdict": "keep"})

    same_model = _AgenticResultCache(cache_dir, namespace=_namespace(model="model-a"))
    other_model = _AgenticResultCache(cache_dir, namespace=_namespace(model="model-b"))

    assert same_model.get("component-key") == {"verdict": "keep"}
    assert other_model.get("component-key") is None


def test_namespace_changes_with_prompt_schema_tools_and_analyzer() -> None:
    baseline = _namespace()

    with patch(
        "aibom.agentic.agent.AIBOM_AGENT_SYSTEM_PROMPT",
        "changed-system-prompt",
    ):
        assert _namespace() != baseline

    with patch.object(
        AgentResponse, "model_json_schema", return_value={"schema": "changed"}
    ):
        assert _namespace() != baseline

    with patch("aibom.agentic.agent.Path.read_bytes", return_value=b"changed-tools"):
        assert _namespace() != baseline

    with patch(
        "aibom.agentic.agent.resolve_package_version",
        return_value="changed-analyzer",
    ):
        assert _namespace() != baseline


def test_namespace_changes_with_batch_and_tool_configuration() -> None:
    baseline = _namespace()
    with_snippets = _agentic_cache_namespace(
        model_string="model-a",
        fast_model="fast-model",
        llm_config={
            "provider": "openai",
            "reasoning": "auto",
            "init_kwargs": {"tool_choice": "required"},
        },
        batch_size=2,
        max_concurrent=2,
        timeout_s=60,
        max_consecutive_failures=2,
        max_retry_seconds=90,
        include_code_snippets=True,
        agent_signature_catalog=None,
    )

    assert with_snippets != baseline


def test_cross_repo_cache_key_versions_model_configuration_without_secrets() -> None:
    per_repo = {
        "repo-a": {"components": [], "_unresolved_env_vars": ["MODEL_NAME"]},
        "repo-b": {"components": [], "_unresolved_env_vars": []},
    }
    baseline = _cross_repo_cache_key(
        "model-a",
        per_repo,
        {"provider": "openai", "reasoning": "auto", "api_key": "secret-a"},
    )

    assert baseline == _cross_repo_cache_key(
        "model-a",
        per_repo,
        {"provider": "openai", "reasoning": "auto", "api_key": "secret-b"},
    )
    assert baseline != _cross_repo_cache_key(
        "model-b", per_repo, {"provider": "openai", "reasoning": "auto"}
    )
    assert baseline != _cross_repo_cache_key(
        "model-a", per_repo, {"provider": "bedrock", "reasoning": "auto"}
    )
    assert baseline != _cross_repo_cache_key(
        "model-a", per_repo, {"provider": "openai", "reasoning": "off"}
    )


def test_scan_evidence_digest_changes_for_unrelated_tool_visible_file(tmp_path) -> None:
    config = tmp_path / "deployment.yaml"
    config.write_text("model: gpt-5\n")
    before = _scan_evidence_digest([str(tmp_path)])

    config.write_text("model: gpt-4\n")

    assert _scan_evidence_digest([str(tmp_path)]) != before
