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

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import aibom.agentic.agent as agent_module
import aibom.agentic_telemetry as telemetry_module
from aibom.agentic.agent import _BatchTelemetryContext
from aibom.agentic.middleware import AIBOMScannerMiddleware
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


class _Message:
    content = json.dumps(
        {
            "enriched_components": [],
            "new_components": [],
            "remove_components": [],
            "reclassify_components": [],
            "new_relationships": [],
            "risk_findings": [],
        }
    )
    usage_metadata = {
        "input_tokens": 11,
        "output_tokens": 3,
        "total_tokens": 14,
    }
    response_metadata: dict[str, Any] = {}


class _Agent:
    def __init__(self) -> None:
        self.calls = 0
        self.callback_configs: list[list[Any]] = []

    def invoke(self, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        self.callback_configs.append(
            list((kwargs.get("config") or {}).get("callbacks", []))
        )
        return {"messages": [_Message()]}

    async def ainvoke(self, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        self.callback_configs.append(
            list((kwargs.get("config") or {}).get("callbacks", []))
        )
        await asyncio.sleep(0)
        return {"messages": [_Message()]}


def _telemetry(
    loggers: list[_RecordingLogger],
) -> Any:
    config = GalileoTelemetryConfig(
        enabled=True,
        sample_rate=1.0,
        project="example-project",
        log_stream="test",
    )

    def factory(_project: str, _stream: str) -> _RecordingLogger:
        logger = _RecordingLogger()
        loggers.append(logger)
        return logger

    return create_agentic_telemetry(
        config,
        logger_factory=factory,
        hmac_key="integration-test-key",
    )


def _component(path: Path, name: str) -> AIComponent:
    return AIComponent(
        name=name,
        component_type=AIComponentType.DEPENDENCY,
        file_path=str(path),
        line_number=1,
    )


def test_enrichment_wires_sanitized_batch_trace_and_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "customer-secret.py"
    source.write_text(
        "API_TOKEN = 'sk-private-token-canary'\n"
        "OWNER = 'alice@example.com'\n"
        "PRIVATE_KEY = '-----BEGIN PRIVATE KEY-----'\n"
        "RESOLVED_ENV = 'customer-private-endpoint.internal'\n",
        encoding="utf-8",
    )
    component = _component(source, "private-component-name")
    agent = _Agent()
    loggers: list[_RecordingLogger] = []
    telemetry = _telemetry(loggers)

    monkeypatch.setattr(agent_module, "_build_model", lambda *_args: object())
    monkeypatch.setattr(agent_module, "_close_model_clients", lambda *_args: None)
    monkeypatch.setattr(agent_module, "create_aibom_agent", lambda *_args, **_kw: agent)

    kwargs = {
        "model_string": "test-model",
        "deterministic_components": [component],
        "deterministic_relationships": [],
        "scan_paths": [str(tmp_path)],
        "cache_dir": tmp_path / "cache",
        "telemetry": telemetry,
    }
    first, _, _, _ = agent_module.run_agentic_enrichment(**kwargs)
    cached, _, _, _ = agent_module.run_agentic_enrichment(**kwargs)
    assert telemetry.drain(1.0)

    assert [item.model_dump(mode="json") for item in cached] == [
        item.model_dump(mode="json") for item in first
    ]
    assert agent.calls == 1
    assert len(loggers) == 2
    first_events = [name for name, _ in loggers[0].calls]
    cached_events = [name for name, _ in loggers[1].calls]
    assert "add_llm_span" in first_events
    assert "add_llm_span" not in cached_events
    cache_trace = next(
        values for name, values in loggers[1].calls if name == "start_trace"
    )
    assert cache_trace["metadata"]["cache_hit"] is True
    assert cache_trace["metadata"]["attempt_kind"] == "initial"
    assert json.loads(cache_trace["input"])["attempt_kind"] == "initial"

    serialized = json.dumps(
        [logger.calls for logger in loggers], default=str, sort_keys=True
    )
    for secret in (
        str(tmp_path),
        str(source),
        component.name,
        "sk-private-token-canary",
        "alice@example.com",
        "-----BEGIN PRIVATE KEY-----",
        "customer-private-endpoint.internal",
        "API_TOKEN",
    ):
        assert secret not in serialized


def test_parallel_batches_use_independent_non_mixed_loggers(tmp_path: Path) -> None:
    components = [
        _component(tmp_path / "first-private.py", "first-private"),
        _component(tmp_path / "second-private.py", "second-private"),
    ]
    loggers: list[_RecordingLogger] = []
    telemetry = _telemetry(loggers)
    context = _BatchTelemetryContext(
        telemetry=telemetry,
        tier="complex",
        provider="openai",
        model="test-model",
        prompt_version="11111111111111111111",
        schema_version="22222222222222222222",
        analyzer_version="1.0.0",
    )

    asyncio.run(
        agent_module._run_batches_parallel(
            _Agent(),
            AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)]),
            [[components[0]], [components[1]]],
            [],
            [str(tmp_path)],
            all_components=components,
            max_concurrent=2,
            telemetry_context=context,
        )
    )
    assert telemetry.drain(1.0)

    assert len(loggers) == 2
    trace_inputs: list[dict[str, Any]] = []
    for logger in loggers:
        names = [name for name, _ in logger.calls]
        assert names.count("start_trace") == 1
        assert names.count("add_agent_span") == 1
        assert names.count("add_workflow_span") == 1
        assert names.count("add_llm_span") == 1
        trace = next(values for name, values in logger.calls if name == "start_trace")
        trace_inputs.append(json.loads(trace["input"]))

    component_tokens = [item["component_ids"][0] for item in trace_inputs]
    assert len(set(component_tokens)) == 2
    serialized = json.dumps(trace_inputs, sort_keys=True)
    for component in components:
        assert component.name not in serialized
        assert component.file_path not in serialized


def test_circuit_breaker_trace_has_no_fake_llm_span(tmp_path: Path) -> None:
    component = _component(tmp_path / "private.py", "private")
    loggers: list[_RecordingLogger] = []
    telemetry = _telemetry(loggers)
    context = _BatchTelemetryContext(
        telemetry=telemetry,
        tier="complex",
        provider="openai",
        model="test-model",
        prompt_version="11111111111111111111",
        schema_version="22222222222222222222",
        analyzer_version="1.0.0",
    )

    agent_module._record_no_llm_trace(
        context,
        [component],
        2,
        3,
        status="circuit_breaker",
        failure_hint="circuit_breaker_tripped",
    )
    assert telemetry.drain(1.0)

    names = [name for name, _ in loggers[0].calls]
    assert names == ["start_trace", "add_agent_span", "conclude", "flush"]
    conclusion = next(values for name, values in loggers[0].calls if name == "conclude")
    output = json.loads(conclusion["output"])
    assert output["status"] == "circuit_breaker"
    assert output["schema_valid"] is False


def test_structured_coercion_has_its_own_workflow_and_llm_span(
    tmp_path: Path,
) -> None:
    pytest.importorskip("langchain_core.messages")

    class ProseMessage:
        content = "unstructured findings"
        usage_metadata = {
            "input_tokens": 7,
            "output_tokens": 2,
            "total_tokens": 9,
        }
        response_metadata: dict[str, Any] = {}

    class CoercionMessage:
        content = ""
        usage_metadata = {
            "input_tokens": 5,
            "output_tokens": 1,
            "total_tokens": 6,
        }
        response_metadata: dict[str, Any] = {}

    model = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = {
        "raw": CoercionMessage(),
        "parsed": agent_module.AgentResponse(),
        "parsing_error": None,
    }
    model.with_structured_output.return_value = structured

    agent = MagicMock()
    agent.invoke.return_value = {"messages": [ProseMessage()]}
    agent.needs_coercion = True
    agent.aibom_chat_model = model

    loggers: list[_RecordingLogger] = []
    telemetry = _telemetry(loggers)
    context = _BatchTelemetryContext(
        telemetry=telemetry,
        tier="complex",
        provider="openai",
        model="test-model",
        prompt_version="11111111111111111111",
        schema_version="22222222222222222222",
        analyzer_version="1.0.0",
    )
    component = _component(tmp_path / "private.py", "private")
    raw_callbacks = [object(), object()]
    callback_factory = MagicMock(side_effect=raw_callbacks)

    output = agent_module._run_batch(
        agent,
        AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)]),
        [component],
        [],
        [str(tmp_path)],
        1,
        1,
        telemetry_context=context,
        invoke_callback_factory=callback_factory,
    )
    assert telemetry.drain(1.0)

    assert output[-1] is False
    workflow_names = [
        values["name"]
        for name, values in loggers[0].calls
        if name == "add_workflow_span"
    ]
    assert workflow_names == ["aibom.agentic.initial", "aibom.agentic.coercion"]
    llm_spans = [values for name, values in loggers[0].calls if name == "add_llm_span"]
    assert len(llm_spans) == 2
    assert {span["total_tokens"] for span in llm_spans} == {6, 9}
    assert callback_factory.call_count == 2
    assert raw_callbacks[0] in agent.invoke.call_args.kwargs["config"]["callbacks"]
    assert structured.invoke.call_args.kwargs["config"]["callbacks"] == [
        raw_callbacks[1]
    ]


def test_real_galileo_structured_coercion_workflows_share_agent_parent(
    tmp_path: Path,
) -> None:
    galileo = pytest.importorskip("galileo", reason="optional observability extra")
    pytest.importorskip("langchain_core.messages")

    class ProseMessage:
        content = "unstructured findings"
        usage_metadata = {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}
        response_metadata: dict[str, Any] = {}

    class CoercionMessage:
        content = ""
        usage_metadata = {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
        response_metadata: dict[str, Any] = {}

    model = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = {
        "raw": CoercionMessage(),
        "parsed": agent_module.AgentResponse(),
        "parsing_error": None,
    }
    model.with_structured_output.return_value = structured
    agent = MagicMock()
    agent.invoke.return_value = {"messages": [ProseMessage()]}
    agent.needs_coercion = True
    agent.aibom_chat_model = model

    captured: list[Any] = []
    logger = galileo.GalileoLogger(
        project="test",
        log_stream="test",
        mode="batch",
        ingestion_hook=captured.append,
    )
    telemetry = create_agentic_telemetry(
        GalileoTelemetryConfig(
            enabled=True,
            sample_rate=1.0,
            project="test",
            log_stream="test",
        ),
        logger_factory=lambda _project, _stream: logger,
        hmac_key="coercion-sibling-test",
    )
    context = _BatchTelemetryContext(
        telemetry=telemetry,
        tier="complex",
        provider="openai",
        model="private-deployment",
        prompt_version="11111111111111111111",
        schema_version="22222222222222222222",
        analyzer_version="1.0.0",
    )

    output = agent_module._run_batch(
        agent,
        AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)]),
        [_component(tmp_path / "private.py", "private")],
        [],
        [str(tmp_path)],
        1,
        1,
        telemetry_context=context,
    )

    assert output[-1] is False
    assert telemetry.drain(2.0)
    assert len(captured) == 1
    trace = captured[0].model_dump(mode="json", exclude_none=True)["traces"][0]
    agent_span = trace["spans"][0]
    assert agent_span["type"] == "agent"
    assert agent_span["name"] == "aibom.agentic.classifier"
    assert agent_span["agent_type"] == "classifier"
    assert [span["name"] for span in agent_span["spans"]] == [
        "aibom.agentic.initial",
        "aibom.agentic.coercion",
    ]
    assert all(
        child["name"] == "aibom.agentic.llm"
        for workflow in agent_span["spans"]
        for child in workflow["spans"]
    )


def test_sync_invoke_propagates_tool_stats_into_sanitized_span(
    tmp_path: Path,
) -> None:
    from aibom.agentic.tools import (
        read_file_snippet_impl,
        reset_allowed_search_roots,
        set_allowed_search_roots,
    )

    source = tmp_path / "agent.py"
    source.write_text("class Agent:\n    pass\n", encoding="utf-8")

    class ToolAgent(_Agent):
        def invoke(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            read_file_snippet_impl(str(source))
            return super().invoke()

    loggers: list[_RecordingLogger] = []
    telemetry = _telemetry(loggers)
    context = _BatchTelemetryContext(
        telemetry=telemetry,
        tier="complex",
        provider="openai",
        model="test-model",
        prompt_version="11111111111111111111",
        schema_version="22222222222222222222",
        analyzer_version="1.0.0",
    )
    roots_token = set_allowed_search_roots([str(tmp_path)])
    try:
        output = agent_module._run_batch(
            ToolAgent(),
            AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)]),
            [_component(source, "private")],
            [],
            [str(tmp_path)],
            1,
            1,
            telemetry_context=context,
        )
    finally:
        reset_allowed_search_roots(roots_token)
    assert telemetry.drain(1.0)

    assert output[-1] is False
    tool_span = next(
        values for name, values in loggers[0].calls if name == "add_tool_span"
    )
    assert tool_span["name"] == "aibom.tool.read_file_snippet"
    assert tool_span["metadata"] == {
        "calls": 1,
        "errors": 0,
        "guard_denials": 0,
    }
    serialized = json.dumps(tool_span, sort_keys=True)
    assert str(source) not in serialized


def test_disabled_and_broken_telemetry_leave_agentic_output_identical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BrokenLogger(_RecordingLogger):
        def start_trace(self, **kwargs: Any) -> object:
            raise RuntimeError("ingestion unavailable")

    source = tmp_path / "app.py"
    source.write_text("import openai\n", encoding="utf-8")
    component = _component(source, "openai")
    agent = _Agent()
    monkeypatch.setattr(agent_module, "_build_model", lambda *_args: object())
    monkeypatch.setattr(agent_module, "_close_model_clients", lambda *_args: None)
    monkeypatch.setattr(agent_module, "create_aibom_agent", lambda *_args, **_kw: agent)

    disabled = create_agentic_telemetry(
        GalileoTelemetryConfig(
            enabled=False,
            project="example-project",
            log_stream="test",
        )
    )
    broken = create_agentic_telemetry(
        GalileoTelemetryConfig(
            enabled=True,
            project="example-project",
            log_stream="test",
        ),
        logger_factory=lambda _project, _stream: BrokenLogger(),
    )

    def run(cache_name: str, telemetry: Any) -> tuple[Any, ...]:
        components, relationships, flags, usage = agent_module.run_agentic_enrichment(
            model_string="test-model",
            deterministic_components=[component],
            deterministic_relationships=[],
            scan_paths=[str(tmp_path)],
            cache_dir=tmp_path / cache_name,
            telemetry=telemetry,
        )
        return (
            [item.model_dump(mode="json") for item in components],
            [item.model_dump(mode="json") for item in relationships],
            [item.model_dump(mode="json") for item in flags],
            usage,
        )

    baseline = run("baseline-cache", None)
    assert run("disabled-cache", disabled) == baseline
    assert run("broken-cache", broken) == baseline


def test_real_galileo_manual_logger_flushes_only_aibom_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    galileo = pytest.importorskip("galileo", reason="optional observability extra")
    disable_calls: dict[int, int] = {}
    original_disable = galileo.GalileoLogger.disable_agent_control

    def tracked_disable(logger: Any) -> None:
        disable_calls[id(logger)] = disable_calls.get(id(logger), 0) + 1
        original_disable(logger)

    monkeypatch.setattr(
        galileo.GalileoLogger,
        "disable_agent_control",
        tracked_disable,
    )
    captured_safe: list[Any] = []
    safe_logger = galileo.GalileoLogger(
        project="test",
        log_stream="test",
        mode="batch",
        ingestion_hook=captured_safe.append,
    )
    safe = create_agentic_telemetry(
        GalileoTelemetryConfig(
            enabled=True,
            sample_rate=1.0,
            project="test",
            log_stream="test",
        ),
        logger_factory=lambda _project, _stream: safe_logger,
        hmac_key="real-sdk-test-key",
    )

    batch = safe.start_batch(
        batch_id="private-batch",
        component_ids=["private-component"],
        model="private-deployment-name",
    )
    attempt = batch.start_attempt(kind="initial", attempt_number=1)
    attempt.record_llm(
        provider="openai",
        model="private-deployment-name",
        prompt_tokens=5,
        completion_tokens=2,
        total_tokens=7,
        decisions={"enriched": 1},
    )
    attempt.record_tools(
        {"search_codebase": {"calls": 1, "errors": 0, "total_s": 0.01}}
    )
    attempt.finish(status="success")
    batch.finish(status="success", decisions={"enriched": 1})

    assert safe.drain(2.0)
    assert disable_calls[id(safe_logger)] >= 2
    assert len(captured_safe) == 1
    safe_payload = captured_safe[0].model_dump(mode="json", exclude_none=True)
    serialized_safe = json.dumps(safe_payload, sort_keys=True)
    assert "private-batch" not in serialized_safe
    assert "private-component" not in serialized_safe
    assert "private-deployment-name" not in serialized_safe
    assert safe_payload["traces"][0]["name"] == "aibom.agentic.batch"
    safe_agent = safe_payload["traces"][0]["spans"][0]
    assert safe_agent["type"] == "agent"
    assert safe_agent["name"] == "aibom.agentic.classifier"
    assert safe_agent["agent_type"] == "classifier"
    span_names = [
        span["name"] for workflow in safe_agent["spans"] for span in workflow["spans"]
    ]
    assert span_names == ["aibom.agentic.llm", "aibom.tool.search_codebase"]

    captured_rogue: list[Any] = []
    rogue_logger = galileo.GalileoLogger(
        project="test",
        log_stream="test",
        mode="batch",
        ingestion_hook=captured_rogue.append,
    )
    rogue = create_agentic_telemetry(
        GalileoTelemetryConfig(
            enabled=True,
            sample_rate=1.0,
            project="test",
            log_stream="test",
        ),
        logger_factory=lambda _project, _stream: rogue_logger,
        hmac_key="real-sdk-test-key",
    )
    trace = rogue.start_batch(batch_id="private-batch")
    rogue_logger.add_tool_span(
        input="raw customer source",
        redacted_input="raw customer source",
        output="raw tool result",
        redacted_output="raw tool result",
        name="external.raw-tool",
    )
    trace.finish(status="success")

    assert rogue.drain(2.0)
    assert captured_rogue == []
    assert rogue_logger.traces == []


def test_real_galileo_loggers_share_an_explicit_uuid4_session() -> None:
    galileo = pytest.importorskip("galileo", reason="optional observability extra")
    session_id = "11111111-1111-4111-8111-111111111111"
    captured: list[Any] = []
    loggers: list[Any] = []

    def factory(_project: str, _stream: str) -> Any:
        logger = galileo.GalileoLogger(
            project="test",
            log_stream="test",
            mode="batch",
            ingestion_hook=captured.append,
        )
        loggers.append(logger)
        return logger

    client = create_agentic_telemetry(
        GalileoTelemetryConfig(
            enabled=True,
            sample_rate=1.0,
            project="test",
            log_stream="test",
        ),
        galileo_session_id=session_id,
        logger_factory=factory,
        hmac_key="real-session-test-key",
    )

    first = client.start_batch(
        batch_id="first-private-batch",
        source_id="first-private-source",
        component_ids=["shared-relative-component"],
    )
    second = client.start_batch(
        batch_id="second-private-batch",
        source_id="second-private-source",
        component_ids=["shared-relative-component"],
    )
    first.finish(status="success")
    second.finish(status="success")

    assert client.drain(2.0)
    assert len(loggers) == 2
    assert len(captured) == 2
    assert all(str(request.session_id) == session_id for request in captured)
    serialized = json.dumps(
        [request.model_dump(mode="json", exclude_none=True) for request in captured],
        sort_keys=True,
    )
    for raw_value in (
        "first-private-batch",
        "second-private-batch",
        "first-private-source",
        "second-private-source",
        "shared-relative-component",
    ):
        assert raw_value not in serialized


def test_real_galileo_validator_rejects_mutated_dataset_identity_and_metadata() -> None:
    galileo = pytest.importorskip("galileo", reason="optional observability extra")
    logger = galileo.GalileoLogger(
        project="test",
        log_stream="test",
        mode="batch",
        ingestion_hook=lambda _request: None,
    )
    client = create_agentic_telemetry(
        GalileoTelemetryConfig(
            enabled=True,
            sample_rate=1.0,
            project="test",
            log_stream="test",
        ),
        logger_factory=lambda _project, _stream: logger,
        hmac_key="mutation-test-key",
    )
    captured: list[Any] = []
    client._flush_dispatcher.submit = lambda item: captured.append(item) or True

    batch = client.start_batch(
        batch_id="private-batch",
        model="private-model",
        analyzer_version="1.0.0",
        prompt_version="11111111111111111111",
        schema_version="22222222222222222222",
    )
    attempt = batch.start_attempt()
    attempt.record_llm(
        provider="openai",
        model="private-model",
        prompt_tokens=3,
        completion_tokens=2,
        total_tokens=5,
    )
    attempt.record_tools(
        {"search_codebase": {"calls": 1, "errors": 0, "total_s": 0.01}}
    )
    attempt.finish()
    batch.finish()

    assert captured == [logger]
    assert telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    trace = logger.traces[0]

    for field_name in ("dataset_input", "dataset_output"):
        setattr(trace, field_name, "RAW_CUSTOMER_SECRET")
        assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
        setattr(trace, field_name, None)

    trace.dataset_metadata = {"private": "RAW_CUSTOMER_SECRET"}
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    trace.dataset_metadata = {}

    original_external_id = trace.external_id
    trace.external_id = "RAW_CUSTOMER_SECRET"
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    trace.external_id = original_external_id

    original_metadata = dict(trace.user_metadata)
    for field_name in (
        "model",
        "batch_size",
        "cache_hit",
        "analyzer_version",
        "prompt_version",
        "schema_version",
    ):
        trace.user_metadata[field_name] = "RAW_CUSTOMER_SECRET"
        assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
        trace.user_metadata = dict(original_metadata)

    agent_span = trace.spans[0]
    original_agent_type = agent_span.agent_type
    agent_span.agent_type = "planner"
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    agent_span.agent_type = original_agent_type

    original_agent_input = agent_span.input
    original_agent_redacted_input = agent_span.redacted_input
    agent_span.input = "RAW_CUSTOMER_SECRET"
    agent_span.redacted_input = "RAW_CUSTOMER_SECRET"
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    agent_span.input = original_agent_input
    agent_span.redacted_input = original_agent_redacted_input

    workflow = agent_span.spans[0]
    llm_span, tool_span = workflow.spans

    agent_span.spans.append(tool_span)
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    agent_span.spans.pop()

    original_tokens = llm_span.metrics.num_input_tokens
    llm_span.metrics.num_input_tokens = "RAW_CUSTOMER_SECRET"
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    llm_span.metrics.num_input_tokens = original_tokens

    setattr(
        llm_span.metrics,
        "RAW_METRIC_KEY",
        {"explanation": "RAW_METRIC_SECRET"},
    )
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    delattr(llm_span.metrics, "RAW_METRIC_KEY")

    original_input_tool_call_id = llm_span.input[0].tool_call_id
    llm_span.input[0].tool_call_id = "RAW_TOOL_CALL_ID_SECRET"
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    llm_span.input[0].tool_call_id = original_input_tool_call_id

    original_output_role = llm_span.output.role
    llm_span.output.role = "RAW_OUTPUT_ROLE_SECRET"
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    llm_span.output.role = original_output_role

    llm_span.tools = [{"description": "RAW_CUSTOMER_SECRET"}]
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    llm_span.tools = None

    original_duration = tool_span.metrics.duration_ns
    tool_span.metrics.duration_ns = "RAW_CUSTOMER_SECRET"
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    tool_span.metrics.duration_ns = original_duration

    tool_span.tool_call_id = "RAW_CUSTOMER_SECRET"
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    tool_span.tool_call_id = None

    logger._session_external_id = "RAW_SESSION_EXTERNAL_SECRET"
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    logger._session_external_id = None

    logger.local_metrics = [{"explanation": "RAW_LOCAL_METRIC_SECRET"}]
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    logger.local_metrics = None

    logger.experiment_id = "RAW_EXPERIMENT_SECRET"
    assert not telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
    logger.experiment_id = None

    assert telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)


def test_raw_callback_does_not_leak_to_later_disabled_scan(tmp_path: Path) -> None:
    agent = _Agent()
    middleware = AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)])
    first = _component(tmp_path / "first.py", "first")
    second = _component(tmp_path / "second.py", "second")
    raw_callback = object()

    agent_module._run_batch(
        agent,
        middleware,
        [first],
        [],
        [str(tmp_path)],
        1,
        1,
        invoke_callback_factory=lambda: raw_callback,
    )
    agent_module._run_batch(
        agent,
        middleware,
        [second],
        [],
        [str(tmp_path)],
        1,
        1,
    )

    assert agent.callback_configs[0] == [raw_callback]
    assert agent.callback_configs[1] == []


def test_overlapping_scans_keep_raw_callbacks_isolated(tmp_path: Path) -> None:
    first_agent = _Agent()
    second_agent = _Agent()
    first_raw = object()
    second_raw = object()

    async def run_overlapping() -> None:
        await asyncio.gather(
            agent_module._run_batch_async(
                first_agent,
                AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)]),
                [_component(tmp_path / "first.py", "first")],
                [],
                [str(tmp_path)],
                1,
                1,
                invoke_callback_factory=lambda: first_raw,
            ),
            agent_module._run_batch_async(
                second_agent,
                AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)]),
                [_component(tmp_path / "second.py", "second")],
                [],
                [str(tmp_path)],
                1,
                1,
                invoke_callback_factory=lambda: second_raw,
            ),
        )

    asyncio.run(run_overlapping())

    assert first_agent.callback_configs == [[first_raw]]
    assert second_agent.callback_configs == [[second_raw]]


def test_parallel_batches_receive_fresh_raw_callbacks(tmp_path: Path) -> None:
    agent = _Agent()
    created: list[Any] = []

    def callback_factory() -> Any:
        callback = object()
        created.append(callback)
        return callback

    components = [
        _component(tmp_path / "first.py", "first"),
        _component(tmp_path / "second.py", "second"),
    ]
    asyncio.run(
        agent_module._run_batches_parallel(
            agent,
            AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)]),
            [[components[0]], [components[1]]],
            [],
            [str(tmp_path)],
            all_components=components,
            max_concurrent=2,
            invoke_callback_factory=callback_factory,
        )
    )

    assert len(created) == 2
    assert len({id(callback) for callback in created}) == 2
    assert len(agent.callback_configs) == 2
    assert all(len(callbacks) == 1 for callbacks in agent.callback_configs)
    assert {id(callbacks[0]) for callbacks in agent.callback_configs} == {
        id(callback) for callback in created
    }


def test_raw_callback_factory_failure_preserves_sanitized_telemetry(
    tmp_path: Path,
) -> None:
    agent = _Agent()
    loggers: list[_RecordingLogger] = []
    telemetry = _telemetry(loggers)
    context = _BatchTelemetryContext(
        telemetry=telemetry,
        tier="complex",
        provider="openai",
        model="test-model",
        prompt_version="11111111111111111111",
        schema_version="22222222222222222222",
        analyzer_version="1.0.0",
    )

    def broken_factory() -> Any:
        raise RuntimeError("callback setup failed")

    output = agent_module._run_batch(
        agent,
        AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)]),
        [_component(tmp_path / "private.py", "private")],
        [],
        [str(tmp_path)],
        1,
        1,
        telemetry_context=context,
        invoke_callback_factory=broken_factory,
    )

    assert output[-1] is False
    assert len(agent.callback_configs) == 1
    assert len(agent.callback_configs[0]) == 1
    assert isinstance(
        agent.callback_configs[0][0],
        agent_module.SanitizedAgentCallback,
    )
    assert telemetry.drain(1.0)
    assert [name for name, _ in loggers[0].calls][-2:] == ["conclude", "flush"]


def test_failed_scan_callback_does_not_leak_to_next_scan(tmp_path: Path) -> None:
    class FailingAgent(_Agent):
        def invoke(self, *_args: Any, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            self.callback_configs.append(
                list((kwargs.get("config") or {}).get("callbacks", []))
            )
            raise RuntimeError("provider failed")

    failing_agent = FailingAgent()
    raw_callback = object()
    failed = agent_module._run_batch(
        failing_agent,
        AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)]),
        [_component(tmp_path / "failed.py", "failed")],
        [],
        [str(tmp_path)],
        1,
        1,
        invoke_callback_factory=lambda: raw_callback,
    )

    next_agent = _Agent()
    succeeded = agent_module._run_batch(
        next_agent,
        AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)]),
        [_component(tmp_path / "next.py", "next")],
        [],
        [str(tmp_path)],
        1,
        1,
    )

    assert failed[-1] is True
    assert succeeded[-1] is False
    assert failing_agent.callback_configs == [[raw_callback]]
    assert next_agent.callback_configs == [[]]


def test_retry_and_fallback_forward_raw_callback_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    component = _component(tmp_path / "component.py", "component")
    callback_factory = MagicMock()
    received: list[Any] = []
    responses = [
        (
            [component.model_copy(update={"agentic_hint": "provider_outage"})],
            [],
            [],
            [],
            True,
        ),
        (
            [component.model_copy(update={"agentic_hint": ""})],
            [],
            [],
            [],
            False,
        ),
    ]

    def fake_run_batch(*_args: Any, **kwargs: Any) -> tuple[Any, ...]:
        received.append(kwargs.get("invoke_callback_factory"))
        return responses.pop(0)

    monkeypatch.setattr(agent_module, "_run_batch", fake_run_batch)
    monkeypatch.setattr(agent_module, "_RETRY_COOLDOWN_S", 0)
    agent_module._run_tier(
        object(),
        AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)]),
        [component],
        [],
        [str(tmp_path)],
        1,
        1,
        [component],
        None,
        retry_deadline=agent_module.time.monotonic() + 10,
        invoke_callback_factory=callback_factory,
    )

    assert received == [callback_factory, callback_factory]

    fallback_received: list[Any] = []

    def fake_fallback_batch(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        fallback_received.append(kwargs.get("invoke_callback_factory"))
        return args[2], [], [], [], False

    monkeypatch.setattr(agent_module, "_run_batch", fake_fallback_batch)
    agent_module._strategy_fallback_pass(
        object(),
        AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)]),
        [component.model_copy(update={"agentic_hint": "no_usable_output"})],
        [],
        [str(tmp_path)],
        [component],
        batch_size=1,
        timeout_s=10,
        max_consecutive_failures=1,
        invoke_callback_factory=callback_factory,
    )

    assert fallback_received == [callback_factory]
