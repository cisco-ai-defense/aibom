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

"""Focused tests for content-free, per-invocation agent telemetry."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

import aibom.agentic.agent as agent_module
import aibom.agentic.sanitized_trace as sanitized_trace_module
import aibom.agentic_telemetry as telemetry_module
from aibom.agentic.agent import _BatchTelemetryContext
from aibom.agentic.sanitized_trace import SanitizedAgentCallback
from aibom.agentic_telemetry import (
    GalileoTelemetryConfig,
    create_agentic_telemetry,
)
from aibom.models import AIComponent, AIComponentType


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, event: str, **kwargs: Any) -> object:
        self.calls.append((event, kwargs))
        return object()

    def start_trace(self, **kwargs: Any) -> object:
        return self._record("start_trace", **kwargs)

    def add_agent_span(self, **kwargs: Any) -> object:
        return self._record("add_agent_span", **kwargs)

    def add_workflow_span(self, **kwargs: Any) -> object:
        return self._record("add_workflow_span", **kwargs)

    def add_llm_span(self, **kwargs: Any) -> object:
        return self._record("add_llm_span", **kwargs)

    def add_tool_span(self, **kwargs: Any) -> object:
        return self._record("add_tool_span", **kwargs)

    def conclude(self, **kwargs: Any) -> object:
        return self._record("conclude", **kwargs)

    def flush(self, **kwargs: Any) -> list[object]:
        self._record("flush", **kwargs)
        return []

    def disable_agent_control(self) -> None:
        return None


def _config(sample_rate: float = 1.0) -> GalileoTelemetryConfig:
    return GalileoTelemetryConfig(
        enabled=True,
        sample_rate=sample_rate,
        project="example-project",
        log_stream="test",
    )


def _llm_result(
    *,
    content: str = "",
    prompt_tokens: int = 11,
    completion_tokens: int = 3,
    total_tokens: int = 14,
    cached_tokens: int = 4,
) -> Any:
    message = SimpleNamespace(
        content=content,
        usage_metadata={
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "input_token_details": {"cache_read": cached_tokens},
        },
        response_metadata={"content_canary": content},
    )
    generation = SimpleNamespace(message=message, text=content)
    return SimpleNamespace(generations=[[generation]], llm_output={})


def _record_llm_tool_llm(
    collector: SanitizedAgentCallback,
    *,
    content: str = "private-content-canary",
) -> tuple[str, str, str]:
    first_id = "11111111-1111-4111-8111-111111111111"
    tool_id = "22222222-2222-4222-8222-222222222222"
    final_id = "33333333-3333-4333-8333-333333333333"
    collector.on_chat_model_start(
        {"name": "private-deployment", "description": content},
        [[{"role": "user", "content": content}]],
        run_id=UUID(first_id),
    )
    collector.on_llm_end(
        _llm_result(content=content),
        run_id=UUID(first_id),
    )
    collector.on_tool_start(
        {"name": "search_codebase", "description": content},
        content,
        run_id=UUID(tool_id),
        inputs={"absolute_path": content},
    )
    collector.on_tool_end(
        {"source": content, "result": content},
        run_id=UUID(tool_id),
    )
    collector.on_chat_model_start(
        {"name": "private-deployment", "description": content},
        [[{"role": "tool", "content": content}]],
        run_id=UUID(final_id),
    )
    collector.on_llm_end(
        _llm_result(
            content=content,
            prompt_tokens=7,
            completion_tokens=2,
            total_tokens=9,
            cached_tokens=0,
        ),
        run_id=UUID(final_id),
    )
    return first_id, tool_id, final_id


def test_callback_discards_prompts_responses_tool_content_and_errors() -> None:
    canaries = (
        "/Users/alice/customer-secret-repository/private.py",
        "sk-private-token-canary",
        "alice@example.com",
        "-----BEGIN PRIVATE KEY-----",
        "customer-private-endpoint.internal",
    )
    content = " | ".join(canaries)
    collector = SanitizedAgentCallback()

    _record_llm_tool_llm(collector, content=content)
    failed_tool_id = UUID("44444444-4444-4444-8444-444444444444")
    collector.on_tool_start(
        {"name": content, "description": content},
        content,
        run_id=failed_tool_id,
    )
    collector.on_tool_error(RuntimeError(content), run_id=failed_tool_id)
    calls = collector.seal()

    retained = json.dumps(
        {
            "calls": [asdict(call) for call in calls],
            "collector_state": vars(collector),
        },
        default=str,
        sort_keys=True,
    )
    for canary in canaries:
        assert canary not in retained
    assert [call.tool_name for call in calls if call.kind == "tool"] == [
        "search_codebase",
        "other",
    ]
    assert collector._active == {}


def test_callback_preserves_order_per_call_tokens_timestamps_and_durations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic_values = iter((10.0, 10.25, 10.5, 10.75, 11.0, 11.5, 12.0))
    monkeypatch.setattr(
        sanitized_trace_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    collector = SanitizedAgentCallback()

    ids = _record_llm_tool_llm(collector)
    calls = collector.seal()

    assert [call.kind for call in calls] == ["llm", "tool", "llm"]
    assert [call.sequence for call in calls] == [1, 2, 3]
    assert [call.call_id for call in calls] == list(ids)
    assert [call.status for call in calls] == ["success", "success", "success"]
    assert [call.duration_s for call in calls] == pytest.approx([0.25, 0.25, 0.5])
    assert calls[0].prompt_tokens == 11
    assert calls[0].completion_tokens == 3
    assert calls[0].total_tokens == 14
    assert calls[0].cached_tokens == 4
    assert calls[1].total_tokens == 0
    assert calls[2].prompt_tokens == 7
    assert calls[2].completion_tokens == 2
    assert calls[2].total_tokens == 9
    assert all(call.created_at.tzinfo is timezone.utc for call in calls)
    assert [call.created_at for call in calls] == sorted(
        call.created_at for call in calls
    )


def test_callback_does_not_duplicate_response_level_usage_across_choices() -> None:
    collector = SanitizedAgentCallback()
    run_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    usage = {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(content="first", usage_metadata=usage)
                ),
                ChatGeneration(
                    message=AIMessage(content="second", usage_metadata=usage)
                ),
            ]
        ],
        llm_output={"token_usage": usage},
    )

    collector.on_chat_model_start({}, [[]], run_id=run_id)
    collector.on_llm_end(response, run_id=run_id)

    call = collector.seal()[0]
    assert (
        call.prompt_tokens,
        call.completion_tokens,
        call.total_tokens,
    ) == (10, 4, 14)


def test_callback_sums_equal_choice_usage_without_response_level_carrier() -> None:
    collector = SanitizedAgentCallback()
    run_id = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
    usage = {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(content="first", usage_metadata=usage)
                ),
                ChatGeneration(
                    message=AIMessage(content="second", usage_metadata=dict(usage))
                ),
            ]
        ],
        llm_output={},
    )

    collector.on_chat_model_start({}, [[]], run_id=run_id)
    collector.on_llm_end(response, run_id=run_id)

    call = collector.seal()[0]
    assert (
        call.prompt_tokens,
        call.completion_tokens,
        call.total_tokens,
    ) == (20, 8, 28)


@pytest.mark.parametrize(
    "tool_name",
    [
        "compact_conversation",
        "edit_file",
        "execute",
        "glob",
        "grep",
        "ls",
        "read_file",
        "task",
        "write_file",
        "write_todos",
    ],
)
def test_callback_preserves_allowlisted_deep_agent_tool_names(tool_name: str) -> None:
    collector = SanitizedAgentCallback()
    run_id = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")

    collector.on_tool_start({"name": tool_name}, "private", run_id=run_id)
    collector.on_tool_end("private", run_id=run_id)

    assert collector.seal()[0].tool_name == tool_name


def test_timeout_seal_converts_unfinished_calls_and_ignores_late_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic_values = iter((20.0, 21.5, 22.0, 23.0, 24.0))
    monkeypatch.setattr(
        sanitized_trace_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    collector = SanitizedAgentCallback()
    run_id = UUID("55555555-5555-4555-8555-555555555555")
    collector.on_chat_model_start({}, [[{"content": "secret"}]], run_id=run_id)

    sealed = collector.seal(unfinished_status="timeout")
    collector.on_llm_end(
        _llm_result(content="late-private-response", total_tokens=999),
        run_id=run_id,
    )
    collector.on_tool_start(
        {"name": "search_codebase"},
        "late-private-tool-input",
        run_id=UUID("66666666-6666-4666-8666-666666666666"),
    )
    sealed_again = collector.seal(unfinished_status="failed")

    assert sealed_again == sealed
    assert len(sealed_again) == 1
    assert sealed_again[0].kind == "llm"
    assert sealed_again[0].status == "timeout"
    assert sealed_again[0].duration_s == pytest.approx(1.5)
    assert sealed_again[0].total_tokens == 0


def test_detailed_spans_emit_ordered_pseudonymous_calls_and_terminal_decisions() -> (
    None
):
    logger = _RecordingLogger()
    client = create_agentic_telemetry(
        _config(),
        logger_factory=lambda _project, _stream: logger,
        hmac_key="detailed-span-test-key",
    )
    context = _BatchTelemetryContext(
        telemetry=client,
        tier="complex",
        provider="openai",
        model="private-deployment",
        prompt_version="prompt-v1",
        schema_version="schema-v1",
        analyzer_version="analyzer-v1",
        source_id="/private/customer-repository",
    )
    trace = client.start_batch(
        batch_id="private-batch",
        source_id=context.source_id,
        component_ids=["private-component"],
    )
    attempt = trace.start_attempt(kind="initial", attempt_number=1)
    collector = SanitizedAgentCallback()
    raw_call_ids = _record_llm_tool_llm(
        collector, content="private-prompt-and-tool-canary"
    )
    captured_calls = collector.seal()

    has_llm, has_tool = agent_module._record_detailed_calls(
        attempt,
        collector,
        context=context,
        status="success",
        schema_valid=True,
        raw_data={"remove_components": ["private-component"]},
    )
    attempt.finish(
        status="success",
        raw_decisions={"removed": 1},
        final_decisions={"removed": 1},
    )
    trace.finish(status="success", decisions={"removed": 1})

    assert has_llm and has_tool
    assert client.drain(1.0)
    leaves = [
        values
        for event, values in logger.calls
        if event in {"add_llm_span", "add_tool_span"}
    ]
    assert [values["step_number"] for values in leaves] == [1, 2, 3]
    assert [values["created_at"] for values in leaves] == [
        call.created_at for call in captured_calls
    ]
    assert [values["duration_ns"] for values in leaves] == [
        int(call.duration_s * 1_000_000_000) for call in captured_calls
    ]
    assert [values["name"] for values in leaves] == [
        "aibom.agentic.llm",
        "aibom.tool.search_codebase",
        "aibom.agentic.llm",
    ]
    llm_spans = [values for event, values in logger.calls if event == "add_llm_span"]
    assert llm_spans[0]["num_input_tokens"] == 11
    assert llm_spans[0]["num_output_tokens"] == 3
    assert llm_spans[0]["total_tokens"] == 14
    assert llm_spans[0]["metadata"]["cached_tokens"] == 4
    assert json.loads(llm_spans[0]["output"])["decision_carrier"] is False
    final_output = json.loads(llm_spans[1]["output"])
    assert final_output["decision_carrier"] is True
    assert final_output["decisions"]["removed"] == 1
    assert llm_spans[1]["num_input_tokens"] == 7
    assert llm_spans[1]["num_output_tokens"] == 2
    serialized = json.dumps(logger.calls, default=str, sort_keys=True)
    for raw_value in (
        *raw_call_ids,
        "private-prompt-and-tool-canary",
        "private-deployment",
        "/private/customer-repository",
        "private-component",
        "private-batch",
    ):
        assert raw_value not in serialized


def test_finish_batch_separates_raw_model_actions_middleware_delta_and_final_result() -> (
    None
):
    logger = _RecordingLogger()
    client = create_agentic_telemetry(
        _config(),
        logger_factory=lambda _project, _stream: logger,
        hmac_key="middleware-delta-test-key",
    )
    context = _BatchTelemetryContext(
        telemetry=client,
        tier="complex",
        provider="anthropic",
        model="private-deployment",
        prompt_version="prompt-v1",
        schema_version="schema-v1",
        analyzer_version="analyzer-v1",
        source_id="/private/customer-repository",
    )
    component = AIComponent(
        name="private-component-name",
        component_type=AIComponentType.DEPENDENCY,
        file_path="/private/customer-repository/src/private.py",
        line_number=7,
    )
    trace = client.start_batch(
        batch_id="private-batch",
        source_id=context.source_id,
        component_ids=[component.instance_id],
    )
    attempt = trace.start_attempt(kind="initial", attempt_number=1)
    collector = SanitizedAgentCallback()
    run_id = UUID("77777777-7777-4777-8777-777777777777")
    collector.on_chat_model_start({}, [[{"content": "private"}]], run_id=run_id)
    collector.on_llm_end(_llm_result(content="private"), run_id=run_id)
    raw_data = {
        "enriched_components": [],
        "new_components": [],
        "remove_components": [component.instance_id],
        "reclassify_components": [],
        "new_relationships": [],
        "risk_findings": [],
    }

    agent_module._finish_batch_trace(
        trace,
        attempt,
        context=context,
        batch=[component],
        # Middleware retained the deterministic candidate, blocking removal.
        output=([component], [], [], [], False),
        result={},
        duration_s=0.2,
        tool_stats={},
        raw_data=raw_data,
        call_collector=collector,
    )

    assert client.drain(1.0)
    workflows = [
        values["name"] for event, values in logger.calls if event == "add_workflow_span"
    ]
    assert workflows == [
        "aibom.agentic.initial",
        "aibom.agentic.middleware_validation",
    ]
    llm = next(values for event, values in logger.calls if event == "add_llm_span")
    assert json.loads(llm["output"])["decisions"]["removed"] == 1
    conclusions = [
        json.loads(values["output"])
        for event, values in logger.calls
        if event == "conclude"
    ]
    attempt_result, middleware_result, trace_result = conclusions
    for result in (attempt_result, middleware_result):
        assert result["raw_actions"]["removed"] == 1
        assert result["final_actions"]["removed"] == 0
        assert result["final_actions"]["kept"] == 1
        assert result["blocked_actions"]["removed"] == 1
    assert middleware_result["status"] == "degraded"
    assert middleware_result["recovered"] is True
    assert trace_result["decisions"]["kept"] == 1
    assert trace_result["decisions"]["removed"] == 0
    assert trace_result["middleware_guard_triggered"] is True
    serialized = json.dumps(logger.calls, default=str, sort_keys=True)
    for raw_value in (
        component.instance_id,
        component.name,
        component.file_path,
        "private-deployment",
        "/private/customer-repository",
    ):
        assert raw_value not in serialized


def test_unsampled_deferred_trace_replays_detailed_children_in_trajectory_order() -> (
    None
):
    logger = _RecordingLogger()
    client = create_agentic_telemetry(
        _config(0.0),
        logger_factory=lambda _project, _stream: logger,
        hmac_key="deferred-order-test-key",
    )
    trace = client.start_batch(
        batch_id="private-batch",
        source_id="/private/customer-repository",
        component_ids=["private-component"],
    )
    attempt = trace.start_attempt(kind="initial", attempt_number=1)
    first_at = datetime(2026, 7, 15, 10, 0, 0, tzinfo=timezone.utc)
    tool_at = datetime(2026, 7, 15, 10, 0, 1, tzinfo=timezone.utc)
    final_at = datetime(2026, 7, 15, 10, 0, 2, tzinfo=timezone.utc)
    attempt.record_llm(
        provider="openai",
        model="private-deployment",
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
        call_id="private-llm-call-1",
        sequence=1,
        created_at=first_at,
        mode="per_call",
        decision_carrier=False,
        schema_expected=False,
    )
    attempt.record_tool_call(
        name="read_file_snippet",
        call_id="private-tool-call-2",
        sequence=2,
        created_at=tool_at,
        duration_s=0.1,
    )
    attempt.record_llm(
        provider="openai",
        model="private-deployment",
        prompt_tokens=8,
        completion_tokens=3,
        total_tokens=11,
        decisions={"discovered": 1},
        call_id="private-llm-call-3",
        sequence=3,
        created_at=final_at,
        mode="per_call",
        decision_carrier=True,
    )
    attempt.finish(
        status="success",
        raw_decisions={"discovered": 1},
        final_decisions={"discovered": 1},
    )
    trace.finish(status="success", decisions={"discovered": 1})

    assert client.drain(1.0)
    assert [event for event, _ in logger.calls] == [
        "start_trace",
        "add_agent_span",
        "add_workflow_span",
        "add_llm_span",
        "add_tool_span",
        "add_llm_span",
        "conclude",
        "conclude",
        "flush",
    ]
    leaves = [
        values
        for event, values in logger.calls
        if event in {"add_llm_span", "add_tool_span"}
    ]
    assert [values["step_number"] for values in leaves] == [1, 2, 3]
    assert [values["created_at"] for values in leaves] == [
        first_at,
        tool_at,
        final_at,
    ]
    serialized = json.dumps(logger.calls, default=str, sort_keys=True)
    for raw_value in (
        "private-batch",
        "/private/customer-repository",
        "private-component",
        "private-deployment",
        "private-llm-call-1",
        "private-tool-call-2",
        "private-llm-call-3",
    ):
        assert raw_value not in serialized


def test_detailed_hierarchy_passes_real_galileo_ingestion_validation() -> None:
    galileo = pytest.importorskip("galileo", reason="optional observability extra")
    captured: list[Any] = []
    logger = galileo.GalileoLogger(
        project="local-test",
        log_stream="local-test",
        mode="batch",
        ingestion_hook=captured.append,
    )
    client = create_agentic_telemetry(
        _config(),
        logger_factory=lambda _project, _stream: logger,
        hmac_key="real-sdk-detailed-test-key",
    )
    trace = client.start_batch(
        batch_id="private-batch",
        source_id="/private/customer-repository",
        component_ids=["private-component"],
    )
    attempt = trace.start_attempt(kind="initial", attempt_number=1)
    now = datetime.now(timezone.utc)
    attempt.record_llm(
        provider="openai",
        model="private-deployment",
        prompt_tokens=5,
        completion_tokens=2,
        total_tokens=7,
        call_id="private-llm-call-1",
        sequence=1,
        created_at=now,
        duration_s=0.01,
        mode="per_call",
        decision_carrier=False,
        schema_expected=False,
    )
    attempt.record_tool_call(
        name="search_codebase",
        call_id="private-tool-call-2",
        sequence=2,
        created_at=now,
        duration_s=0.01,
    )
    attempt.record_llm(
        provider="openai",
        model="private-deployment",
        prompt_tokens=4,
        completion_tokens=1,
        total_tokens=5,
        decisions={"kept": 1},
        call_id="private-llm-call-3",
        sequence=3,
        created_at=now,
        duration_s=0.01,
        mode="per_call",
        decision_carrier=True,
    )
    attempt.finish(
        status="success",
        duration_s=0.03,
        raw_decisions={"kept": 1},
        final_decisions={"kept": 1},
    )
    trace.finish(status="success", duration_s=0.04, decisions={"kept": 1})

    assert client.drain(2.0)
    assert len(captured) == 1
    request = captured[0].model_dump(mode="json", exclude_none=True)
    assert len(request["traces"]) == 1
    root = request["traces"][0]
    assert root["name"] == "aibom.agentic.batch"
    assert len(root["spans"]) == 1
    agent_span = root["spans"][0]
    assert agent_span["type"] == "agent"
    assert agent_span["name"] == "aibom.agentic.classifier"
    assert agent_span["agent_type"] == "classifier"
    assert agent_span["input"] == agent_span["redacted_input"] == root["input"]
    assert agent_span["output"] == agent_span["redacted_output"] == root["output"]
    workflow = agent_span["spans"][0]
    assert workflow["name"] == "aibom.agentic.initial"
    assert [span["name"] for span in workflow["spans"]] == [
        "aibom.agentic.llm",
        "aibom.tool.search_codebase",
        "aibom.agentic.llm",
    ]
    assert [span["step_number"] for span in workflow["spans"]] == [1, 2, 3]
    serialized = json.dumps(request, sort_keys=True)
    for raw_value in (
        "private-batch",
        "/private/customer-repository",
        "private-component",
        "private-deployment",
        "private-llm-call-1",
        "private-tool-call-2",
        "private-llm-call-3",
    ):
        assert raw_value not in serialized
    assert telemetry_module._is_uuid4("11111111-1111-4111-8111-111111111111")
