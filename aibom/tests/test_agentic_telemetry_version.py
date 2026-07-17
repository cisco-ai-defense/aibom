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

from aibom.agentic.agent import AgentResponse, _agentic_telemetry_version


def _version(
    *,
    model: str = "model-a",
    provider: str = "openai",
    reasoning: str = "auto",
    api_key: str = "super-secret-key",
    max_concurrent: int = 1,
    timeout_s: int = 120,
    max_consecutive_failures: int = 3,
    max_retry_seconds: int = 300,
) -> str:
    return _agentic_telemetry_version(
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
    )


def test_agentic_telemetry_version_is_stable_and_non_sensitive() -> None:
    first = _version()
    second = _version(api_key="a-different-secret")

    assert first == second
    assert len(first) == 20
    assert "secret" not in first


def test_agentic_telemetry_version_changes_with_configuration() -> None:
    baseline = _version()

    assert _version(model="model-b") != baseline
    assert _version(provider="bedrock") != baseline
    assert _version(reasoning="off") != baseline
    assert _version(max_concurrent=2) != baseline
    assert _version(timeout_s=30) != baseline
    assert _version(max_consecutive_failures=1) != baseline
    assert _version(max_retry_seconds=0) != baseline


def test_telemetry_version_changes_with_prompt_schema_tools_and_analyzer() -> None:
    baseline = _version()

    with patch(
        "aibom.agentic.agent.AIBOM_AGENT_SYSTEM_PROMPT",
        "changed-system-prompt",
    ):
        assert _version() != baseline

    with patch.object(
        AgentResponse, "model_json_schema", return_value={"schema": "changed"}
    ):
        assert _version() != baseline

    with patch("aibom.agentic.agent.Path.read_bytes", return_value=b"changed-tools"):
        assert _version() != baseline

    with patch(
        "aibom.agentic.agent.resolve_package_version",
        return_value="changed-analyzer",
    ):
        assert _version() != baseline


def test_telemetry_version_changes_with_batch_and_tool_configuration() -> None:
    baseline = _version()
    with_snippets = _agentic_telemetry_version(
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
