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

import json
import ssl
import threading
import time
from typing import Any

import pytest

import aibom.agentic_telemetry as telemetry_module
from aibom.agentic_telemetry import (
    GalileoTelemetryConfig,
    Pseudonymizer,
    create_agentic_telemetry,
)


def _unsafe_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


class FakeLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.agent_control_disable_calls = 0

    def _record(self, event: str, **kwargs: Any) -> object:
        self.calls.append((event, kwargs))
        return object()

    def start_session(self, **kwargs: Any) -> str:
        self._record("start_session", **kwargs)
        return "11111111-1111-4111-8111-111111111111"

    def set_session(self, session_id: str) -> None:
        self._record("set_session", session_id=session_id)

    def start_trace(self, **kwargs: Any) -> object:
        return self._record("start_trace", **kwargs)

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
        self.agent_control_disable_calls += 1


def _config(sample_rate: float = 1.0) -> GalileoTelemetryConfig:
    return GalileoTelemetryConfig(
        enabled=True,
        sample_rate=sample_rate,
        project="example-project",
        log_stream="test",
    )


def _serialized_calls(logger: FakeLogger) -> str:
    return json.dumps(logger.calls, default=str, sort_keys=True)


def test_config_from_env_and_invalid_rate_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIBOM_GALILEO_ENABLED", "yes")
    monkeypatch.setenv("AIBOM_GALILEO_SAMPLE_RATE", "0.25")
    monkeypatch.setenv("GALILEO_PROJECT", "quality")
    monkeypatch.setenv("GALILEO_LOG_STREAM", "canary")

    config = GalileoTelemetryConfig.from_env()

    assert config.configured
    assert config.sample_rate == 0.25
    assert config.project == "quality"
    assert config.log_stream == "canary"
    assert "API" not in repr(config)

    monkeypatch.setenv("AIBOM_GALILEO_SAMPLE_RATE", "not-a-number")
    assert not GalileoTelemetryConfig.from_env().configured


def test_disabled_or_missing_credentials_never_imports_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = False

    def fail_import(_name: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("optional SDK must not be imported")

    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.setattr(telemetry_module.importlib, "import_module", fail_import)

    disabled = create_agentic_telemetry(
        GalileoTelemetryConfig(
            enabled=False,
            sample_rate=1.0,
            project="quality",
            log_stream="prod",
        )
    )
    missing_key = create_agentic_telemetry(_config())

    assert not disabled.enabled
    assert not disabled.start_batch(batch_id="batch").active
    assert not missing_key.enabled
    assert not missing_key.start_batch(batch_id="batch").active
    assert not imported


@pytest.mark.parametrize(
    "console_url",
    [
        None,
        "",
        "http://galileo.customer.internal",
        "https://galileo.customer.internal",
        "https://app.galileo.ai",
        "https://APP.GALILEO.AI./",
        "https://galileo.ai",
        "https://api.galileo.ai",
        "https://customer.galileo.ai",
        "https://user:password@app.galileo.ai",
        "https://app.galileo.ai/project/customer",
        "https://app.galileo.ai?tenant=customer",
        "https://app.galileo.ai#fragment",
        "not-a-url",
    ],
)
def test_default_factory_rejects_unapproved_or_invalid_console(
    monkeypatch: pytest.MonkeyPatch,
    console_url: str | None,
) -> None:
    imported = False

    def fail_import(_name: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("invalid destinations must not import the SDK")

    monkeypatch.setenv("GALILEO_API_KEY", "configured")
    monkeypatch.delenv("AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD", raising=False)
    if console_url is None:
        monkeypatch.delenv("GALILEO_CONSOLE_URL", raising=False)
    else:
        monkeypatch.setenv("GALILEO_CONSOLE_URL", console_url)
    monkeypatch.setattr(telemetry_module.importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError, match="approved hosted Galileo"):
        telemetry_module._default_logger_factory("project", "stream")
    client = create_agentic_telemetry(_config())

    assert not client.enabled
    assert not client.start_batch(batch_id="batch").active
    assert not imported


@pytest.mark.parametrize(
    "console_url",
    [
        "https://galileo.ai",
        "https://api.galileo.ai",
        "https://customer.galileo.ai",
        "https://galileo.customer.internal",
        "https://app.galileo.ai:8443",
        "https://app.galileo.ai/project/customer",
        "https://app.galileo.ai?tenant=customer",
    ],
)
def test_public_opt_in_accepts_only_the_hosted_console_origin(
    monkeypatch: pytest.MonkeyPatch,
    console_url: str,
) -> None:
    imported = False

    def fail_import(_name: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("invalid hosted destinations must not import the SDK")

    monkeypatch.setenv("GALILEO_API_KEY", "configured")
    monkeypatch.setenv("GALILEO_CONSOLE_URL", console_url)
    monkeypatch.setenv("AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD", "true")
    monkeypatch.setattr(telemetry_module.importlib, "import_module", fail_import)

    client = create_agentic_telemetry(_config())

    assert not client.enabled
    assert not client.start_batch(batch_id="batch").active
    assert not imported


@pytest.mark.parametrize(
    "api_url",
    [
        "http://api.galileo.ai",
        "https://attacker.example",
        "https://api.galileo.ai:8443",
        "https://api.galileo.ai/v1",
        "https://api.galileo.ai?tenant=customer",
        "https://user:password@api.galileo.ai",
    ],
)
def test_hosted_api_override_is_rejected_before_sdk_import(
    monkeypatch: pytest.MonkeyPatch,
    api_url: str,
) -> None:
    imported = False

    def fail_import(_name: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("invalid API origins must not import the SDK")

    monkeypatch.setenv("GALILEO_API_KEY", "configured")
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv("GALILEO_API_URL", api_url)
    monkeypatch.setenv("AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD", "true")
    monkeypatch.setattr(telemetry_module.importlib, "import_module", fail_import)

    client = create_agentic_telemetry(_config())

    assert not client.enabled
    assert not client.start_batch(batch_id="batch").active
    assert not imported


@pytest.mark.parametrize(
    "tls_setting", ["0", "false", "NO", " off ", "f", "n", "invalid"]
)
def test_disabled_tls_verification_is_rejected_before_sdk_import(
    monkeypatch: pytest.MonkeyPatch,
    tls_setting: str,
) -> None:
    imported = False

    def fail_import(_name: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("unsafe TLS settings must not import the SDK")

    monkeypatch.setenv("GALILEO_API_KEY", "configured")
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv("GALILEO_SSL_CONTEXT", tls_setting)
    monkeypatch.setenv("AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD", "true")
    monkeypatch.setattr(telemetry_module.importlib, "import_module", fail_import)

    client = create_agentic_telemetry(_config())

    assert not client.enabled
    assert not client.start_batch(batch_id="batch").active
    assert not imported


def test_missing_sdk_is_lazy_and_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def missing_sdk(_name: str) -> Any:
        nonlocal attempts
        attempts += 1
        raise ModuleNotFoundError("galileo")

    monkeypatch.setenv("GALILEO_API_KEY", "configured-but-never-logged")
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv("AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD", "true")
    monkeypatch.setattr(telemetry_module.importlib, "import_module", missing_sdk)
    client = create_agentic_telemetry(_config())

    assert attempts == 0
    assert not client.start_batch(batch_id="batch").active
    assert attempts == 1
    assert not client.enabled
    assert not client.start_batch(batch_id="second").active
    assert attempts == 1


@pytest.mark.parametrize(
    ("api_url", "ssl_context"),
    [
        ("https://attacker.example/", True),
        ("https://api.galileo.ai/", False),
        ("https://api.galileo.ai/", _unsafe_ssl_context()),
    ],
)
def test_default_factory_rejects_preloaded_unsafe_sdk_settings(
    monkeypatch: pytest.MonkeyPatch,
    api_url: str,
    ssl_context: bool | ssl.SSLContext,
) -> None:
    resource_lookup_attempted = False

    def fake_import(name: str) -> Any:
        nonlocal resource_lookup_attempted
        if name == "galileo.config":
            instance = type(
                "ConfigInstance",
                (),
                {
                    "console_url": "https://app.galileo.ai/",
                    "api_url": api_url,
                    "ssl_context": ssl_context,
                },
            )()
            config = type("Config", (), {"_instance": instance})
            return type("ConfigModule", (), {"GalileoPythonConfig": config})
        resource_lookup_attempted = True
        raise AssertionError("resource lookup must not use the stale SDK destination")

    monkeypatch.setenv("GALILEO_API_KEY", "configured")
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv("AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD", "true")
    monkeypatch.setattr(telemetry_module.importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError, match="does not match"):
        telemetry_module._default_logger_factory("project", "stream")
    assert not resource_lookup_attempted


@pytest.mark.parametrize(
    ("console_url", "api_url"),
    [
        ("https://app.galileo.ai", None),
        ("https://APP.GALILEO.AI.:443/", "https://API.GALILEO.AI.:443/"),
    ],
)
def test_default_factory_uses_existing_resource_ids_and_never_creates(
    monkeypatch: pytest.MonkeyPatch,
    console_url: str,
    api_url: str | None,
) -> None:
    project_gets = 0
    stream_gets = 0
    logger_kwargs: list[dict[str, Any]] = []

    class Resource:
        def __init__(self, resource_id: str) -> None:
            self.id = resource_id

    class Projects:
        def get(self, **kwargs: Any) -> Resource:
            nonlocal project_gets
            project_gets += 1
            assert kwargs == {"name": "example-project"}
            return Resource("project-id")

        def create(self, **_kwargs: Any) -> None:
            raise AssertionError("telemetry must not create a project")

    class LogStreams:
        def get(self, **kwargs: Any) -> Resource:
            nonlocal stream_gets
            stream_gets += 1
            assert kwargs == {"name": "test", "project_id": "project-id"}
            return Resource("stream-id")

        def create(self, **_kwargs: Any) -> None:
            raise AssertionError("telemetry must not create a log stream")

    class Logger(FakeLogger):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            logger_kwargs.append(kwargs)
            self.project_id = kwargs["project_id"]
            self.log_stream_id = kwargs["log_stream_id"]

    def fake_import(name: str) -> Any:
        if name == "galileo.config":
            return type(
                "ConfigModule",
                (),
                {"GalileoPythonConfig": type("Config", (), {"_instance": None})},
            )
        if name == "galileo.projects":
            return type("ProjectsModule", (), {"Projects": Projects})
        if name == "galileo.log_streams":
            return type("LogStreamsModule", (), {"LogStreams": LogStreams})
        if name == "galileo":
            return type("GalileoModule", (), {"GalileoLogger": Logger})
        raise AssertionError(name)

    monkeypatch.setenv("GALILEO_API_KEY", "configured")
    monkeypatch.setenv("GALILEO_CONSOLE_URL", console_url)
    monkeypatch.setenv("AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD", "true")
    if api_url is None:
        monkeypatch.delenv("GALILEO_API_URL", raising=False)
    else:
        monkeypatch.setenv("GALILEO_API_URL", api_url)
    monkeypatch.setattr(telemetry_module.importlib, "import_module", fake_import)
    client = create_agentic_telemetry(_config())

    first = client.start_batch(batch_id="first")
    second = client.start_batch(batch_id="second")

    assert first.active and second.active
    assert project_gets == 1
    assert stream_gets == 1
    assert logger_kwargs == [
        {"project_id": "project-id", "log_stream_id": "stream-id", "mode": "batch"},
        {"project_id": "project-id", "log_stream_id": "stream-id", "mode": "batch"},
    ]
    assert all("project" not in kwargs for kwargs in logger_kwargs)
    assert all("log_stream" not in kwargs for kwargs in logger_kwargs)


def test_public_console_singleton_matches_normalized_explicit_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = type(
        "Config",
        (),
        {
            "_instance": type(
                "ConfigInstance",
                (),
                {
                    "console_url": "https://app.galileo.ai/",
                    "api_url": "https://api.galileo.ai/",
                },
            )()
        },
    )
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://APP.GALILEO.AI.:443/")
    monkeypatch.setenv("GALILEO_API_URL", "https://API.GALILEO.AI.:443/")
    monkeypatch.setenv("AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD", "true")
    monkeypatch.setattr(
        telemetry_module.importlib,
        "import_module",
        lambda name: (
            type("ConfigModule", (), {"GalileoPythonConfig": config})
            if name == "galileo.config"
            else pytest.fail(f"unexpected import: {name}")
        ),
    )

    assert telemetry_module._loaded_galileo_destination_matches()


def test_setup_budget_is_larger_for_hosted_cloud_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.delenv("AIBOM_GALILEO_SETUP_BUDGET_S", raising=False)
    assert telemetry_module._resolve_setup_budget_s() == 2.0

    monkeypatch.setenv("AIBOM_GALILEO_SETUP_BUDGET_S", "5")
    assert telemetry_module._resolve_setup_budget_s() == 5.0

    for invalid in ("0", "11", "1e300", "invalid"):
        monkeypatch.setenv("AIBOM_GALILEO_SETUP_BUDGET_S", invalid)
        assert telemetry_module._resolve_setup_budget_s() == 2.0


def test_default_factory_disables_when_resources_are_not_preprovisioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creates = 0

    class Projects:
        def get(self, **_kwargs: Any) -> None:
            return None

        def create(self, **_kwargs: Any) -> None:
            nonlocal creates
            creates += 1

    def fake_import(name: str) -> Any:
        if name == "galileo.config":
            return type(
                "ConfigModule",
                (),
                {"GalileoPythonConfig": type("Config", (), {"_instance": None})},
            )
        if name == "galileo.projects":
            return type("ProjectsModule", (), {"Projects": Projects})
        raise AssertionError(f"unexpected import after missing project: {name}")

    monkeypatch.setenv("GALILEO_API_KEY", "configured")
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv("AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD", "true")
    monkeypatch.setattr(telemetry_module.importlib, "import_module", fake_import)
    client = create_agentic_telemetry(_config())

    assert not client.start_batch(batch_id="batch").active
    assert not client.enabled
    assert creates == 0


def test_independent_loggers_share_pseudonymous_session_and_sanitize_payloads() -> None:
    loggers: list[FakeLogger] = []

    def factory(_project: str, _stream: str) -> FakeLogger:
        logger = FakeLogger()
        loggers.append(logger)
        return logger

    raw_session = "/Users/alice/AcmeSecret/repository"
    raw_batch = "/private/source.py:42"
    raw_component = "customer-secret-component"
    client = create_agentic_telemetry(
        _config(),
        session_external_id=raw_session,
        logger_factory=factory,
        hmac_key="stable-test-key",
    )

    first = client.start_batch(
        batch_id=raw_batch,
        source_id=raw_session,
        tier="complex",
        batch_num=1,
        total_batches=2,
        component_ids=[raw_component],
        component_type_counts={"model": 1, raw_component: 9},
        language_counts={"python": 1, raw_session: 1},
        provider="openai",
        model="/Users/alice/private-model",
        analyzer_version="1.0.7",
        prompt_version="prompt-v2",
        schema_version="3",
    )
    second = client.start_batch(
        batch_id="second-private-batch",
        component_ids=["another-private-name"],
    )

    assert first.active and second.active
    assert len(loggers) == 2
    assert all(logger.agent_control_disable_calls >= 1 for logger in loggers)
    assert any(name == "start_session" for name, _ in loggers[0].calls)
    set_session = [kwargs for name, kwargs in loggers[1].calls if name == "set_session"]
    assert set_session == [{"session_id": "11111111-1111-4111-8111-111111111111"}]

    payload = _serialized_calls(loggers[0]) + _serialized_calls(loggers[1])
    for forbidden in (
        raw_session,
        raw_batch,
        raw_component,
        "another-private-name",
        "/Users/alice/private-model",
    ):
        assert forbidden not in payload
    assert "batch_" in payload
    assert "component_" in payload
    assert '"model": "other"' in payload
    first_trace = next(
        kwargs for name, kwargs in loggers[0].calls if name == "start_trace"
    )
    assert json.loads(first_trace["input"])["component_type_counts"]["other"] == 9


def test_source_scoped_identity_and_component_lineage_are_stable_and_private() -> None:
    loggers: list[FakeLogger] = []

    def factory(_project: str, _stream: str) -> FakeLogger:
        logger = FakeLogger()
        loggers.append(logger)
        return logger

    client = create_agentic_telemetry(
        _config(),
        logger_factory=factory,
        hmac_key="source-lineage-test-key",
    )
    raw_source = "/Users/alice/AcmeSecret/repository"
    other_source = "/Users/bob/OtherSecret/repository"
    raw_component = "src/agent.py:model:customer-secret-component"
    raw_batch = "complex:initial:1:private-batch"

    initial = client.start_batch(
        batch_id=raw_batch,
        source_id=raw_source,
        attempt_kind="initial",
        component_ids=[raw_component],
    )
    retry = client.start_batch(
        batch_id="complex:retry:1:private-batch",
        source_id=raw_source,
        attempt_kind="retry",
        component_ids=[raw_component],
    )
    other = client.start_batch(
        batch_id=raw_batch,
        source_id=other_source,
        attempt_kind="initial",
        component_ids=[raw_component],
    )

    for trace in (initial, retry, other):
        trace.finish(status="success")
    assert client.drain(1.0)

    starts = [
        next(values for event, values in logger.calls if event == "start_trace")
        for logger in loggers
    ]
    payloads = [json.loads(values["input"]) for values in starts]

    assert payloads[0]["attempt_kind"] == "initial"
    assert payloads[1]["attempt_kind"] == "retry"
    assert payloads[0]["source_id"] == payloads[1]["source_id"]
    assert payloads[0]["source_id"] != payloads[2]["source_id"]
    assert payloads[0]["component_ids"] == payloads[1]["component_ids"]
    assert payloads[0]["component_ids"] != payloads[2]["component_ids"]
    assert payloads[0]["decision_chain_ids"] == payloads[1]["decision_chain_ids"]
    assert payloads[0]["decision_chain_ids"] != payloads[2]["decision_chain_ids"]
    assert len(payloads[0]["decision_chain_ids"]) == len(payloads[0]["component_ids"])
    assert payloads[0]["batch_id"] != payloads[1]["batch_id"]
    assert payloads[0]["batch_id"] != payloads[2]["batch_id"]
    assert starts[0]["metadata"]["source_id"] == payloads[0]["source_id"]
    assert starts[1]["metadata"]["attempt_kind"] == "retry"

    serialized = json.dumps(
        [logger.calls for logger in loggers], default=str, sort_keys=True
    )
    for confidential_value in (
        raw_source,
        other_source,
        raw_component,
        raw_batch,
        "private-batch",
        "customer-secret-component",
    ):
        assert confidential_value not in serialized


def test_non_v4_external_session_id_is_rejected_before_logger_creation() -> None:
    factory_calls = 0

    def factory(_project: str, _stream: str) -> FakeLogger:
        nonlocal factory_calls
        factory_calls += 1
        return FakeLogger()

    client = create_agentic_telemetry(
        _config(),
        galileo_session_id="11111111-1111-1111-8111-111111111111",
        logger_factory=factory,
    )

    assert not client.enabled
    assert not client.start_batch(batch_id="private-batch").active
    assert factory_calls == 0
    assert telemetry_module._is_uuid("11111111-1111-1111-8111-111111111111")
    assert not telemetry_module._is_uuid4("11111111-1111-1111-8111-111111111111")
    assert telemetry_module._is_uuid4("11111111-1111-4111-8111-111111111111")


def test_batch_attempt_emits_only_aggregate_manual_spans() -> None:
    logger = FakeLogger()
    client = create_agentic_telemetry(
        _config(), logger_factory=lambda _project, _stream: logger, hmac_key="key"
    )
    batch = client.start_batch(
        batch_id="batch-1",
        component_ids=["component-1", "component-2"],
        component_type_counts={"agent": 1, "model": 1},
    )
    attempt = batch.start_attempt(kind="retry", attempt_number=2)
    attempt.record_llm(
        provider="anthropic",
        model="claude-sonnet-4",
        status="success",
        duration_s=1.25,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        cached_tokens=40,
        decisions={"kept": 1, "removed": 1, "raw_output": 999},
    )
    attempt.record_tools(
        {
            "search_codebase": {
                "calls": 2,
                "errors": 0,
                "guard_denials": 1,
                "total_s": 0.5,
            },
            "/private/evil-tool": {"calls": 1, "errors": 1, "total_s": 0.1},
        }
    )
    attempt.finish(status="success", duration_s=1.3)
    batch.finish(
        status="success",
        duration_s=1.4,
        decisions={
            "kept": 1,
            "removed": 1,
            "discovered": 2,
            "enriched": 4,
            "relationships": 3,
        },
        decisions_by_component_type={
            "kept": {"agent": 1},
            "removed": {"model": 1, "customer-private-type": 9},
        },
        decisions_by_confidence={
            "kept": {"medium": 1},
            "removed": {"high": 1, "customer-private-confidence": 9},
        },
        decisions_by_language={
            "kept": {"python": 1},
            "removed": {"typescript": 1, "customer-private-language": 9},
        },
    )
    assert client.drain(1.0)

    names = [name for name, _ in logger.calls]
    assert names == [
        "start_trace",
        "add_workflow_span",
        "add_llm_span",
        "add_tool_span",
        "add_tool_span",
        "conclude",
        "conclude",
        "flush",
    ]
    tool_calls = [kwargs for name, kwargs in logger.calls if name == "add_tool_span"]
    assert {item["name"] for item in tool_calls} == {
        "aibom.tool.other",
        "aibom.tool.search_codebase",
    }
    guarded = next(
        item for item in tool_calls if item["name"] == "aibom.tool.search_codebase"
    )
    assert guarded["status_code"] == 500
    assert json.loads(guarded["output"])["guard_denials"] == 1
    payload = _serialized_calls(logger)
    assert "/private/evil-tool" not in payload
    assert "raw_output" not in payload
    assert "search_query" not in payload
    assert "num_input_tokens" in payload
    assert "input_tokens" not in payload.replace("num_input_tokens", "")
    batch_output = next(
        kwargs["output"]
        for name, kwargs in logger.calls
        if name == "conclude" and "degraded_candidates" in kwargs["output"]
    )
    parsed_batch_output = json.loads(batch_output)
    assert parsed_batch_output["decisions"]["enriched"] == 4
    assert parsed_batch_output["decisions_by_component_type"]["kept"] == {"agent": 1}
    assert parsed_batch_output["decisions_by_component_type"]["removed"] == {
        "model": 1,
        "other": 9,
    }
    assert parsed_batch_output["decisions_by_language"]["removed"] == {
        "other": 9,
        "typescript": 1,
    }
    assert parsed_batch_output["decisions_by_confidence"]["removed"] == {
        "high": 1,
        "other": 9,
    }
    for name, kwargs in logger.calls:
        if name in {
            "start_trace",
            "add_workflow_span",
            "add_llm_span",
            "add_tool_span",
        }:
            assert kwargs["input"] == kwargs["redacted_input"]
        if name in {"add_llm_span", "add_tool_span"}:
            assert kwargs["output"] == kwargs["redacted_output"]


def test_zero_sample_rate_and_logger_failures_are_noops() -> None:
    created = 0

    def factory(_project: str, _stream: str) -> FakeLogger:
        nonlocal created
        created += 1
        return FakeLogger()

    unsampled = create_agentic_telemetry(_config(0.0), logger_factory=factory)
    ordinary = unsampled.start_batch(batch_id="never")
    assert ordinary.active
    ordinary.finish(status="success")
    assert created == 0

    class BrokenLogger(FakeLogger):
        def start_trace(self, **kwargs: Any) -> object:
            raise RuntimeError("backend unavailable")

    broken = create_agentic_telemetry(
        _config(), logger_factory=lambda _project, _stream: BrokenLogger()
    )
    assert not broken.start_batch(batch_id="safe").active
    broken.record_summary(source_id="safe")


def test_extreme_numeric_telemetry_values_are_clamped_and_fail_open() -> None:
    logger = FakeLogger()
    client = create_agentic_telemetry(
        _config(), logger_factory=lambda _project, _stream: logger, hmac_key="key"
    )
    enormous = 10**5000

    batch = client.start_batch(batch_id="extreme")
    attempt = batch.start_attempt()
    attempt.record_llm(
        duration_s=1e308,
        prompt_tokens=enormous,
        completion_tokens=enormous,
        total_tokens=enormous,
    )
    attempt.record_tools({"search_codebase": {"calls": enormous, "total_s": 1e308}})
    attempt.finish(duration_s=1e308)
    batch.finish(duration_s=1e308)
    client.record_summary(
        source_id="source",
        duration_s=1e308,
        prompt_tokens=enormous,
        completion_tokens=enormous,
        cached_tokens=enormous,
    )

    assert client.drain(1.0)
    assert "9223372036854775807" in _serialized_calls(logger)


def test_summary_trace_pseudonymizes_source_and_allowlists_results() -> None:
    logger = FakeLogger()
    client = create_agentic_telemetry(
        _config(),
        session_external_id="private-session",
        logger_factory=lambda _project, _stream: logger,
        hmac_key="key",
    )
    raw_source = "/Users/alice/customer-repository"
    client.record_summary(
        source_id=raw_source,
        source_kind=raw_source,
        status="provider_outage",
        candidate_count=8,
        final_component_count=7,
        degraded_candidate_count=2,
        duration_s=4.5,
        prompt_tokens=20,
        completion_tokens=10,
        cached_tokens=5,
        decisions={"kept": 5, "secret_decision": 100},
    )
    assert client.drain(1.0)

    names = [name for name, _ in logger.calls]
    assert names == ["start_session", "start_trace", "conclude", "flush"]
    payload = _serialized_calls(logger)
    assert raw_source not in payload
    assert "private-session" not in payload
    assert "secret_decision" not in payload
    assert "source_" in payload
    assert '"source_kind": "other"' in payload


def test_pseudonymizer_is_stable_and_sampling_is_deterministic() -> None:
    first = Pseudonymizer("shared-key")
    second = Pseudonymizer("shared-key")

    assert first.token("sensitive", prefix="component") == second.token(
        "sensitive", prefix="component"
    )
    assert "sensitive" not in first.token("sensitive", prefix="component")
    assert first.selected("batch", 0.5) == first.selected("batch", 0.5)
    assert not first.selected("batch", 0.0)
    assert first.selected("batch", 1.0)


def test_slow_sdk_setup_is_bounded_and_fails_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    monkeypatch.setenv("AIBOM_GALILEO_SETUP_BUDGET_S", "0.01")

    def slow_factory(_project: str, _stream: str) -> FakeLogger:
        release.wait(1.0)
        return FakeLogger()

    client = create_agentic_telemetry(_config(), logger_factory=slow_factory)
    started = time.monotonic()
    trace = client.start_batch(batch_id="slow-backend")
    elapsed = time.monotonic() - started
    release.set()

    assert not trace.active
    assert elapsed < 0.1
    assert not client.enabled


def test_slow_flush_runs_off_the_scan_path_and_drain_is_finite() -> None:
    release = threading.Event()

    class SlowFlushLogger(FakeLogger):
        def flush(self, **kwargs: Any) -> list[object]:
            self._record("flush", **kwargs)
            release.wait(1.0)
            return []

    logger = SlowFlushLogger()
    client = create_agentic_telemetry(
        _config(), logger_factory=lambda _project, _stream: logger
    )
    trace = client.start_batch(batch_id="batch")

    started = time.monotonic()
    trace.finish(status="success")
    elapsed = time.monotonic() - started

    assert elapsed < 0.1
    assert not client.drain(0.01)
    release.set()

    assert client.drain(1.0)
    assert any(name == "flush" for name, _ in logger.calls)


def test_drain_timeout_drops_queue_and_rejects_new_flushes() -> None:
    release = threading.Event()
    flush_started = threading.Event()

    class SlowFlushLogger(FakeLogger):
        def flush(self, **kwargs: Any) -> list[object]:
            self._record("flush", **kwargs)
            flush_started.set()
            release.wait(1.0)
            return []

    first = SlowFlushLogger()
    queued = FakeLogger()
    after_close = FakeLogger()
    loggers = iter((first, queued, after_close))
    client = create_agentic_telemetry(
        _config(), logger_factory=lambda _project, _stream: next(loggers)
    )

    client.start_batch(batch_id="first").finish(status="success")
    assert flush_started.wait(0.2)
    client.start_batch(batch_id="queued").finish(status="success")

    assert not client.drain(0.01)
    client.start_batch(batch_id="after-close").finish(status="success")
    release.set()
    assert client.drain(1.0)

    assert any(name == "flush" for name, _ in first.calls)
    assert not any(name == "flush" for name, _ in queued.calls)
    assert not any(name == "flush" for name, _ in after_close.calls)


def test_flush_drain_uses_one_global_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = telemetry_module._FlushDispatcher()
    dispatcher._pending = 2
    clock = [100.0]
    waits: list[float] = []

    class ProgressCondition:
        def __enter__(self) -> ProgressCondition:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def wait(self, timeout: float | None = None) -> None:
            assert timeout is not None
            waits.append(timeout)
            if len(waits) == 1:
                # One flush completes after 60% of the shutdown budget. The
                # next wait must receive only the original deadline's balance.
                dispatcher._pending -= 1
                clock[0] += 0.6
            else:
                clock[0] += timeout

    dispatcher._condition = ProgressCondition()  # type: ignore[assignment]
    monkeypatch.setattr(telemetry_module.time, "monotonic", lambda: clock[0])

    assert not dispatcher.drain(1.0)
    assert waits == pytest.approx([1.0, 0.4])


def test_flush_exception_is_fail_open_and_drain_completes() -> None:
    class RaisingFlushLogger(FakeLogger):
        def flush(self, **kwargs: Any) -> list[object]:
            self._record("flush", **kwargs)
            raise RuntimeError("ingestion unavailable")

    logger = RaisingFlushLogger()
    client = create_agentic_telemetry(
        _config(), logger_factory=lambda _project, _stream: logger
    )
    trace = client.start_batch(batch_id="batch")

    # Neither queue submission nor the worker-side SDK exception reaches the
    # scan. The pending counter must still settle so shutdown remains bounded.
    trace.finish(status="success")

    assert client.drain(1.0)
    assert any(name == "flush" for name, _ in logger.calls)


def test_sampling_is_stable_across_sessions_and_can_use_source_key() -> None:
    first = create_agentic_telemetry(
        _config(0.37),
        session_external_id="first-invocation",
        logger_factory=lambda _project, _stream: FakeLogger(),
        hmac_key="stable-key",
    )
    second = create_agentic_telemetry(
        _config(0.37),
        session_external_id="second-invocation",
        logger_factory=lambda _project, _stream: FakeLogger(),
        hmac_key="stable-key",
    )

    assert first._sampled("same-source") == second._sampled("same-source")

    first_trace = first.start_batch(
        batch_id="ephemeral-first",
        sample_key="stable/repository/source",
        batch_num=2,
    )
    second_trace = second.start_batch(
        batch_id="ephemeral-second",
        sample_key="stable/repository/source",
        batch_num=2,
    )
    assert first_trace.active == second_trace.active


def test_unsampled_terminal_batch_is_retained_without_raw_or_llm_data() -> None:
    loggers: list[FakeLogger] = []

    def factory(_project: str, _stream: str) -> FakeLogger:
        logger = FakeLogger()
        loggers.append(logger)
        return logger

    client = create_agentic_telemetry(
        _config(0.0), logger_factory=factory, hmac_key="stable-key"
    )
    ordinary = client.start_batch(
        batch_id="raw-private-ordinary",
        component_ids=["private-component"],
    )
    assert ordinary.active
    ordinary.finish(status="success")
    assert not loggers

    terminal = client.start_batch(
        batch_id="raw-private-terminal",
        component_ids=["private-component"],
        model="private-customer-deployment",
    )
    assert terminal.active
    assert terminal.start_attempt().active
    terminal.finish(
        status="degraded",
        degraded_candidates=1,
        failure_hint="total_agentic_degradation",
        decisions={"enriched": 2},
    )

    assert client.drain(1.0)
    assert len(loggers) == 1
    names = [name for name, _ in loggers[0].calls]
    assert names == [
        "start_trace",
        "add_workflow_span",
        "conclude",
        "conclude",
        "flush",
    ]
    serialized = _serialized_calls(loggers[0])
    assert "raw-private-terminal" not in serialized
    assert "private-component" not in serialized
    assert "private-customer-deployment" not in serialized
    conclusion = next(
        values
        for name, values in loggers[0].calls
        if name == "conclude" and "decisions" in values["output"]
    )
    assert json.loads(conclusion["output"])["decisions"]["enriched"] == 2


@pytest.mark.parametrize(
    "finish_kwargs",
    [
        {"decisions": {"discovered": 1}},
        {"decisions": {"risk_findings": 1}},
        {"middleware_guard_triggered": True},
    ],
)
def test_unsampled_quality_and_guard_signals_are_force_retained(
    finish_kwargs: dict[str, Any],
) -> None:
    logger = FakeLogger()
    client = create_agentic_telemetry(
        _config(0.0),
        logger_factory=lambda _project, _stream: logger,
        hmac_key="key",
    )

    client.start_batch(batch_id="private-batch").finish(
        status="success", **finish_kwargs
    )

    assert client.drain(1.0)
    assert [name for name, _ in logger.calls] == [
        "start_trace",
        "conclude",
        "flush",
    ]


def test_unsampled_attempt_buffers_only_sanitized_aggregates_and_replays_siblings() -> (
    None
):
    logger = FakeLogger()
    client = create_agentic_telemetry(
        _config(0.0),
        logger_factory=lambda _project, _stream: logger,
        hmac_key="key",
    )
    batch = client.start_batch(
        batch_id="raw-private-batch",
        component_ids=["raw-private-component"],
        model="openai:gpt-acme",
    )
    missing_tokens = batch.start_attempt(kind="initial", attempt_number=1)
    missing_tokens.record_llm(
        provider="openai",
        model="claude-customer/private",
        status="success",
        decisions={"kept": 1},
    )
    missing_tokens.finish(status="success")
    guarded_tool = batch.start_attempt(kind="retry", attempt_number=2)
    guarded_tool.record_tools(
        {
            "search_codebase": {
                "calls": 2,
                "errors": 1,
                "guard_denials": 1,
                "total_s": 0.2,
            }
        }
    )
    guarded_tool.finish(status="success", recovered=True)

    batch.finish(status="success", decisions={"kept": 1})

    assert client.drain(1.0)
    assert [name for name, _ in logger.calls] == [
        "start_trace",
        "add_workflow_span",
        "add_llm_span",
        "conclude",
        "add_workflow_span",
        "add_tool_span",
        "conclude",
        "conclude",
        "flush",
    ]
    assert telemetry_module._validate_fake_logger(logger)
    serialized = _serialized_calls(logger)
    for raw_value in (
        "raw-private-batch",
        "raw-private-component",
        "openai:gpt-acme",
        "claude-customer/private",
    ):
        assert raw_value not in serialized
    llm = next(values for name, values in logger.calls if name == "add_llm_span")
    assert llm["model"].startswith("model_")
    assert json.loads(llm["output"])["token_usage_missing"] is True
    tools = next(values for name, values in logger.calls if name == "add_tool_span")
    assert json.loads(tools["output"]) == {"errors": 1, "guard_denials": 1}


def test_unsampled_deferred_hierarchy_passes_real_galileo_validator() -> None:
    galileo = pytest.importorskip("galileo")
    accepted: list[bool] = []

    def factory(_project: str, _stream: str) -> Any:
        return galileo.GalileoLogger(
            project="local-test",
            log_stream="local-test",
            mode="batch",
            ingestion_hook=lambda _request: None,
        )

    client = create_agentic_telemetry(
        _config(0.0), logger_factory=factory, hmac_key="key"
    )
    client._flush_dispatcher.submit = lambda logger: (
        accepted.append(
            telemetry_module._logger_contains_only_sanitized_aibom_spans(logger)
        )
        or True
    )
    batch = client.start_batch(batch_id="private-batch", model="gpt-private")
    first = batch.start_attempt(kind="initial", attempt_number=1)
    first.record_llm(model="gpt-private", decisions={"discovered": 1})
    first.finish(status="success")
    second = batch.start_attempt(kind="retry", attempt_number=2)
    second.record_tools({"search_codebase": {"calls": 1, "errors": 1, "total_s": 0.1}})
    second.finish(status="success", recovered=True)

    batch.finish(status="success", decisions={"discovered": 1})

    assert accepted == [True]


def test_terminal_summary_overrides_trace_sampling() -> None:
    logger = FakeLogger()
    client = create_agentic_telemetry(
        _config(0.0),
        logger_factory=lambda _project, _stream: logger,
        hmac_key="key",
    )

    client.record_summary(source_id="private-source", status="provider_outage")

    assert client.drain(1.0)
    assert [name for name, _ in logger.calls] == [
        "start_trace",
        "conclude",
        "flush",
    ]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("https://customer.internal/deployments/gpt", "other"),
        (r"C:\\Users\\alice\\private-deployment", "other"),
        ("/srv/private/deployment", "other"),
        ("gpt-5.5", "model"),
        ("claude-customer/private", "model"),
        ("openai:gpt-acme", "model"),
    ],
)
def test_model_labels_reject_urls_and_paths(model: str, expected: str) -> None:
    logger = FakeLogger()
    client = create_agentic_telemetry(
        _config(),
        logger_factory=lambda _project, _stream: logger,
        hmac_key="model-key",
    )

    trace = client.start_batch(batch_id="batch", model=model)

    assert trace.active
    started = next(values for name, values in logger.calls if name == "start_trace")
    emitted = started["metadata"]["model"]
    if expected == "model":
        assert emitted.startswith("model_")
    else:
        assert emitted == expected
    assert model not in _serialized_calls(logger)


def test_private_model_deployment_is_stably_pseudonymized() -> None:
    first_logger = FakeLogger()
    second_logger = FakeLogger()
    first = create_agentic_telemetry(
        _config(),
        logger_factory=lambda _project, _stream: first_logger,
        hmac_key="model-key",
    )
    second = create_agentic_telemetry(
        _config(),
        logger_factory=lambda _project, _stream: second_logger,
        hmac_key="model-key",
    )

    first.start_batch(batch_id="first", model="customer-prod-deployment")
    second.start_batch(batch_id="second", model="customer-prod-deployment")

    first_model = next(
        values["metadata"]["model"]
        for name, values in first_logger.calls
        if name == "start_trace"
    )
    second_model = next(
        values["metadata"]["model"]
        for name, values in second_logger.calls
        if name == "start_trace"
    )
    assert first_model == second_model
    assert first_model.startswith("model_")
    assert "customer-prod-deployment" not in _serialized_calls(first_logger)


@pytest.mark.parametrize(
    "value",
    ["gpt-5.5", "claude-customer/private", "openai:gpt-acme", "model_customer"],
)
def test_model_validator_rejects_every_raw_model_label(value: str) -> None:
    assert not telemetry_module._safe_model_value(value)


def test_model_validator_accepts_only_sentinels_and_hmac_tokens() -> None:
    assert telemetry_module._safe_model_value("unknown")
    assert telemetry_module._safe_model_value("other")
    assert telemetry_module._safe_model_value("model_0123456789abcdef01234567")


def test_raw_or_non_aibom_span_causes_entire_flush_to_be_dropped() -> None:
    logger = FakeLogger()
    client = create_agentic_telemetry(
        _config(), logger_factory=lambda _project, _stream: logger
    )
    trace = client.start_batch(batch_id="batch")
    logger.add_tool_span(
        input="raw customer source",
        redacted_input="raw customer source",
        output="raw tool output",
        redacted_output="raw tool output",
        name="external.raw-tool",
    )

    trace.finish(status="success")

    assert client.drain(1.0)
    assert not any(name == "flush" for name, _ in logger.calls)


def test_agent_control_disable_failure_disables_telemetry() -> None:
    class UnsafeLogger(FakeLogger):
        def disable_agent_control(self) -> None:
            raise RuntimeError("bridge remains active")

    logger = UnsafeLogger()
    client = create_agentic_telemetry(
        _config(), logger_factory=lambda _project, _stream: logger
    )

    assert not client.start_batch(batch_id="batch").active
    assert not client.enabled
    assert not any(name == "flush" for name, _ in logger.calls)


def test_flush_dispatcher_never_flushes_loggers_concurrently() -> None:
    release = threading.Event()
    lock = threading.Lock()
    active_flushes = 0
    maximum_flushes = 0

    class ControlledLogger(FakeLogger):
        def flush(self, **kwargs: Any) -> list[object]:
            nonlocal active_flushes, maximum_flushes
            with lock:
                active_flushes += 1
                maximum_flushes = max(maximum_flushes, active_flushes)
            self._record("flush", **kwargs)
            release.wait(1.0)
            with lock:
                active_flushes -= 1
            return []

    client = create_agentic_telemetry(
        _config(), logger_factory=lambda _project, _stream: ControlledLogger()
    )
    for number in range(8):
        client.start_batch(batch_id=f"batch-{number}").finish(status="success")

    assert not client.drain(0.01)
    release.set()
    assert client.drain(2.0)
    assert maximum_flushes == 1
