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

"""Tests for agent factory and enrichment pipeline.

These tests do NOT require deepagents/langchain to be installed --
they test the helper functions and mock the external dependencies.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from aibom.agentic.agent import (
    AgentEvidence,
    AgentResponse,
    TokenUsage,
    _batch_corr_id,
    _batch_corr_metadata,
    _batch_decision_breakdowns,
    _batch_decision_counts,
    _build_context_message,
    _build_rate_limiter,
    _EnrichedComponent,
    _extract_structured_response,
    _resolve_message_usage,
    _RiskFinding,
)
from aibom.models import (
    AIComponent,
    AIComponentType,
    ComponentRelationship,
    DecisionAnnotation,
    EvidenceLocation,
)

# ``deepagents`` is an optional (agentic) extra and is NOT installed in CI's
# ``uv sync --group dev`` env. Tests that exercise the real ``create_aibom_agent``
# (which imports deepagents) are skipped there, matching this module's design of
# not requiring deepagents/langchain; helper-level tests below run everywhere.
_HAS_DEEPAGENTS = importlib.util.find_spec("deepagents") is not None
# ``langchain_core`` ships with the same agentic extra. A few helper-level tests
# call ``_coerce_structured`` directly, which imports ``langchain_core.messages``
# at call time, so they are gated the same way (skipped in the CI ``--group dev``
# env). The pure-Python helper tests around them still run everywhere.
_HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None
# ``langchain`` (the agents/middleware framework) ships with the same agentic
# extra. The Bedrock system-prompt cache middleware subclasses
# ``langchain.agents.middleware.AgentMiddleware``, so tests that build or exercise
# it are gated on ``langchain``; the provider-gating tests that expect ``[]`` need
# no import and run everywhere.
_HAS_LANGCHAIN_AGENTS = importlib.util.find_spec("langchain") is not None
# ``langchain_aws`` (the ``llm-aws`` extra) ships ``BedrockPromptCachingMiddleware``,
# which the Converse (``ChatBedrockConverse``) path reuses. Tests that assert that
# concrete middleware is built are gated on it.
_HAS_LANGCHAIN_AWS = importlib.util.find_spec("langchain_aws") is not None


@pytest.fixture(autouse=True)
def _isolate_agentic_cache():
    """Prevent on-disk agentic cache from leaking between tests."""
    with patch("aibom.agentic.agent._default_agentic_cache_dir", return_value=None):
        yield


def test_batch_correlation_is_invocation_scoped_and_phase_pairable():
    with patch(
        "aibom.agentic.agent.secrets.token_hex",
        side_effect=["a" * 32, "b" * 32],
    ) as token_hex:
        first = _batch_corr_id("initial", 1, correlate=True)
        second = _batch_corr_id("initial", 1, correlate=True)
        disabled = _batch_corr_id("initial", 1, correlate=False)

    assert first == f"{'a' * 32}:initial1"
    assert second == f"{'b' * 32}:initial1"
    assert first != second
    assert disabled is None
    assert token_hex.call_count == 2
    assert _batch_corr_metadata(first, "agent") == {
        "aibom_batch_corr_id": first,
        "aibom_phase": "agent",
    }
    assert _batch_corr_metadata(first, "coercion") == {
        "aibom_batch_corr_id": first,
        "aibom_phase": "coercion",
    }


def test_middleware_guard_emits_a_sibling_validation_workflow():
    from aibom.agentic.agent import _finish_batch_trace

    component = AIComponent(
        name="router",
        component_type=AIComponentType.AGENT,
        file_path="src/router.py",
        line_number=4,
    )
    trace = MagicMock()
    attempt = MagicMock()
    middleware_attempt = MagicMock()
    trace.start_attempt.return_value = middleware_attempt

    output = _finish_batch_trace(
        trace,
        attempt,
        context=None,
        batch=[component],
        output=([component], [], [], [], False),
        result=None,
        duration_s=0.1,
        tool_stats={},
        raw_data={"remove_components": [{"instance_id": component.instance_id}]},
    )

    assert output[0] == [component]
    trace.start_attempt.assert_called_once_with(
        kind="middleware_validation", attempt_number=1
    )
    middleware_attempt.finish.assert_called_once_with(
        status="degraded",
        duration_s=0.0,
        recovered=True,
        raw_decisions={
            "enriched": 0,
            "removed": 1,
            "reclassified": 0,
            "discovered": 0,
            "relationships": 0,
            "risk_findings": 0,
        },
        final_decisions={
            "kept": 1,
            "enriched": 0,
            "removed": 0,
            "reclassified": 0,
            "discovered": 0,
            "relationships": 0,
            "risk_findings": 0,
            "degraded": 0,
        },
        blocked_decisions={
            "enriched": 0,
            "removed": 1,
            "reclassified": 0,
            "discovered": 0,
            "relationships": 0,
            "risk_findings": 0,
        },
    )
    assert trace.finish.call_args.kwargs["middleware_guard_triggered"] is True
    assert trace.finish.call_args.kwargs["decisions_by_component_type"]["kept"] == {
        "agent": 1
    }
    assert trace.finish.call_args.kwargs["decisions_by_language"]["kept"] == {
        "python": 1
    }


@pytest.mark.parametrize(
    "failure_hint",
    [
        "batch_timeout",
        "batch_recursion_limit",
        "provider_outage",
        "rate_limited",
    ],
)
def test_failed_batch_without_structured_output_is_schema_invalid(failure_hint):
    from aibom.agentic.agent import _finish_batch_trace

    component = AIComponent(
        name="router",
        component_type=AIComponentType.AGENT,
        file_path="src/router.py",
        line_number=4,
        agentic_hint=failure_hint,
    )
    trace = MagicMock()
    attempt = MagicMock()

    _finish_batch_trace(
        trace,
        attempt,
        context=None,
        batch=[component],
        output=([component], [], [], [], True),
        result=None,
        duration_s=0.1,
        tool_stats={},
    )

    assert attempt.record_llm.call_args.kwargs["schema_valid"] is False
    assert trace.finish.call_args.kwargs["schema_valid"] is False


def test_coercion_workflow_is_finalized_at_the_post_middleware_boundary():
    from aibom.agentic.agent import _finish_batch_trace

    component = AIComponent(
        name="router",
        component_type=AIComponentType.AGENT,
        file_path="src/router.py",
        line_number=4,
    )
    trace = MagicMock()
    initial_attempt = MagicMock()
    coercion_attempt = MagicMock()
    coercion_attempt.active = True

    _finish_batch_trace(
        trace,
        initial_attempt,
        context=None,
        batch=[component],
        output=([component], [], [], [], False),
        result=None,
        duration_s=0.1,
        tool_stats={},
        raw_data={"remove_components": [{"instance_id": component.instance_id}]},
        attempt_already_finished=True,
        coercion_attempt=coercion_attempt,
        coercion_duration_s=0.04,
    )

    initial_attempt.finish.assert_not_called()
    coercion_attempt.finish.assert_called_once()
    concluded = coercion_attempt.finish.call_args.kwargs
    assert concluded["status"] == "degraded"
    assert concluded["duration_s"] == 0.04
    assert concluded["recovered"] is True
    assert concluded["raw_decisions"]["removed"] == 1
    assert concluded["final_decisions"]["removed"] == 0
    assert concluded["final_decisions"]["kept"] == 1
    assert concluded["blocked_decisions"]["removed"] == 1


def test_batch_decision_breakdowns_attribute_original_and_discovered_slices():
    router = AIComponent(
        name="router",
        component_type=AIComponentType.TOOL,
        file_path="src/router.py",
        line_number=4,
    )
    removed_model = AIComponent(
        name="old-model",
        component_type=AIComponentType.MODEL,
        file_path="src/client.js",
        line_number=2,
    )
    reclassified_router = router.model_copy(
        update={"component_type": AIComponentType.AGENT}
    )
    discovered_model = AIComponent(
        name="new-model",
        component_type=AIComponentType.MODEL,
        file_path="src/new.ts",
        line_number=8,
    )

    by_type, by_language, by_confidence = _batch_decision_breakdowns(
        [router, removed_model],
        [reclassified_router],
        [discovered_model],
    )

    assert by_type["reclassified"] == {"tool": 1}
    assert by_type["removed"] == {"model": 1}
    assert by_type["discovered"] == {"model": 1}
    assert by_language["reclassified"] == {"python": 1}
    assert by_language["removed"] == {"javascript": 1}
    assert by_language["discovered"] == {"typescript": 1}
    assert by_confidence["reclassified"] == {"high": 1}
    assert by_confidence["removed"] == {"high": 1}
    assert by_confidence["discovered"] == {"high": 1}


def test_degraded_passthrough_is_an_abstention_not_a_keep() -> None:
    component = AIComponent(
        name="router",
        component_type=AIComponentType.AGENT,
        file_path="src/router.py",
        line_number=4,
    )
    degraded = component.model_copy(update={"agentic_hint": "batch_timeout"})

    decisions = _batch_decision_counts([component], [degraded], [], [], [])
    by_type, by_language, by_confidence = _batch_decision_breakdowns(
        [component], [degraded], []
    )

    assert decisions["kept"] == 0
    assert decisions["degraded"] == 1
    assert all(not counts for counts in by_type.values())
    assert all(not counts for counts in by_language.values())
    assert all(not counts for counts in by_confidence.values())


class TestBuildContextMessage:
    def test_includes_components_and_paths(self):
        comps = [
            AIComponent(
                name="gpt-4o",
                component_type=AIComponentType.MODEL,
                file_path="app.py",
                line_number=10,
                model_name="gpt-4o",
            )
        ]
        msg = _build_context_message(comps, [], ["/tmp/repo"])
        assert "gpt-4o" in msg
        assert "/tmp/repo" in msg
        assert "deterministic scan results" in msg.lower()

    def test_full_context_separates_batch_from_others(self):
        batch_comp = AIComponent(
            name="dataset",
            component_type=AIComponentType.DATASET,
            file_path="data.py",
            line_number=5,
        )
        other_comp = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="app.py",
            line_number=10,
            model_name="gpt-4o",
        )
        msg = _build_context_message(
            [batch_comp],
            [],
            ["/tmp"],
            all_components=[batch_comp, other_comp],
        )
        json_start = msg.index("```json\n") + 8
        json_end = msg.index("\n```", json_start)
        data = json.loads(msg[json_start:json_end])
        assert len(data["enrich_these"]) == 1
        assert data["enrich_these"][0]["name"] == "dataset"
        assert data["enrich_these"][0]["ENRICH"] is True
        assert len(data["other_detected_components"]) == 1
        assert data["other_detected_components"][0]["name"] == "gpt-4o"
        assert "ENRICH" not in data["other_detected_components"][0]

    def test_json_is_parseable(self):
        comps = [
            AIComponent(
                name="test",
                component_type=AIComponentType.TOOL,
                file_path="t.py",
                line_number=1,
            )
        ]
        msg = _build_context_message(comps, [], ["/code"])
        json_start = msg.index("```json\n") + 8
        json_end = msg.index("\n```", json_start)
        data = json.loads(msg[json_start:json_end])
        assert len(data["enrich_these"]) == 1
        assert data["enrich_these"][0]["ENRICH"] is True
        assert data["scan_paths"] == ["/code"]
        assert data["other_detected_components"] == []

    def test_includes_code_context_for_real_file(self, tmp_path):
        src = tmp_path / "example.py"
        src.write_text(
            "import openai\nclient = openai.OpenAI()\nresult = client.chat.completions.create(model='gpt-4o')\n"
        )
        comps = [
            AIComponent(
                name="gpt-4o",
                component_type=AIComponentType.MODEL,
                file_path=str(src),
                line_number=3,
                model_name="gpt-4o",
            )
        ]
        msg = _build_context_message(comps, [], [str(tmp_path)])
        json_start = msg.index("```json\n") + 8
        json_end = msg.index("\n```", json_start)
        data = json.loads(msg[json_start:json_end])
        comp = data["enrich_these"][0]
        assert "code_context" in comp
        assert "import openai" in comp["code_context"]
        assert "model='gpt-4o'" in comp["code_context"]

    def test_no_code_context_for_missing_file(self):
        comps = [
            AIComponent(
                name="x",
                component_type=AIComponentType.MODEL,
                file_path="/nonexistent/path.py",
                line_number=1,
            )
        ]
        msg = _build_context_message(comps, [], ["/tmp"])
        json_start = msg.index("```json\n") + 8
        json_end = msg.index("\n```", json_start)
        data = json.loads(msg[json_start:json_end])
        assert "code_context" not in data["enrich_these"][0]


class TestExtractStructuredResponse:
    def test_prefers_structured_response(self):
        result = {
            "structured_response": {"enriched_components": [], "new_components": []},
            "messages": [MagicMock()],
        }
        data = _extract_structured_response(result)
        assert data == {"enriched_components": [], "new_components": []}

    def test_falls_back_to_json_message(self):
        mock_msg = MagicMock()
        mock_msg.content = '{"enriched_components": []}'
        result = {"messages": [mock_msg]}
        data = _extract_structured_response(result)
        assert data == {"enriched_components": []}

    def test_returns_none_for_empty(self):
        assert _extract_structured_response({"messages": []}) is None
        assert _extract_structured_response({}) is None


class TestExtractStructuredResponseCarriers:
    """Provider-agnostic carriers beyond ``message.content``.

    Many models return the structured answer via tool-call args, a parsed
    object in ``additional_kwargs``, or list-form content blocks rather than
    a JSON string in ``message.content``. The extractor must recover all of
    these so non-OpenAI-native output is not silently dropped.
    """

    class _Msg:
        """Minimal stand-in for a LangChain ``AIMessage`` (no langchain dep)."""

        def __init__(self, content="", tool_calls=None, additional_kwargs=None):
            self.content = content
            self.tool_calls = [] if tool_calls is None else tool_calls
            self.additional_kwargs = (
                {} if additional_kwargs is None else additional_kwargs
            )

    def test_extracts_from_tool_call_args(self):
        # function_calling: the answer is a dict in tool_calls[].args, content empty
        msg = self._Msg(
            content="",
            tool_calls=[
                {
                    "name": "AgentResponse",
                    "args": {
                        "enriched_components": [],
                        "new_components": [{"name": "x"}],
                    },
                    "id": "call_1",
                }
            ],
        )
        data = _extract_structured_response({"messages": [msg]})
        assert data == {"enriched_components": [], "new_components": [{"name": "x"}]}

    def test_extracts_from_additional_kwargs_parsed_dict(self):
        # json_schema: the parsed object lands in additional_kwargs, content empty
        msg = self._Msg(
            content="", additional_kwargs={"parsed": {"enriched_components": []}}
        )
        data = _extract_structured_response({"messages": [msg]})
        assert data == {"enriched_components": []}

    def test_extracts_from_additional_kwargs_parsed_model(self):
        from pydantic import BaseModel as _BM

        class _Parsed(_BM):
            enriched_components: list = []
            new_components: list = []

        msg = self._Msg(content="", additional_kwargs={"parsed": _Parsed()})
        data = _extract_structured_response({"messages": [msg]})
        assert data == {"enriched_components": [], "new_components": []}

    def test_parses_list_form_text_content(self):
        # Anthropic/Bedrock/Gemini return content as a list of blocks
        msg = self._Msg(content=[{"type": "text", "text": '{"new_components": []}'}])
        data = _extract_structured_response({"messages": [msg]})
        assert data == {"new_components": []}

    def test_list_form_strips_non_text_blocks(self):
        msg = self._Msg(
            content=[
                {"type": "thinking", "thinking": "let me reason about this..."},
                {"type": "text", "text": '{"enriched_components": []}'},
            ]
        )
        data = _extract_structured_response({"messages": [msg]})
        assert data == {"enriched_components": []}

    def test_tool_call_args_preferred_over_content(self):
        # When both are present, the structured tool-call args win over free text
        msg = self._Msg(
            content='{"new_components": [{"name": "from_content"}]}',
            tool_calls=[
                {
                    "name": "AgentResponse",
                    "args": {"new_components": [{"name": "from_tool"}]},
                    "id": "call_1",
                }
            ],
        )
        data = _extract_structured_response({"messages": [msg]})
        assert data == {"new_components": [{"name": "from_tool"}]}

    def test_recovers_char_doubled_content(self):
        # Gateways may echo character-doubled content ("hheelllloo"); when the
        # de-doubled form is valid JSON, recover it.
        original = '{"enriched_components": [], "new_components": []}'
        doubled = "".join(ch * 2 for ch in original)
        msg = self._Msg(content=doubled)
        data = _extract_structured_response({"messages": [msg]})
        assert data == {"enriched_components": [], "new_components": []}

    def test_unrepairable_content_returns_none(self):
        msg = self._Msg(content="this is not json at all")
        assert _extract_structured_response({"messages": [msg]}) is None

    def test_ignores_non_agentresponse_tool_call(self):
        # A normal tool invocation (not the structured-output tool) must not be
        # mistaken for the agent's response.
        msg = self._Msg(
            content="",
            tool_calls=[
                {"name": "search_codebase", "args": {"query": "foo"}, "id": "t1"}
            ],
        )
        assert _extract_structured_response({"messages": [msg]}) is None

    def test_falls_through_to_content_when_tool_call_not_response(self):
        # Non-response tool call ignored, but valid JSON content still parsed.
        msg = self._Msg(
            content='{"new_components": []}',
            tool_calls=[
                {"name": "lookup_model", "args": {"name": "gpt-4o"}, "id": "t1"}
            ],
        )
        assert _extract_structured_response({"messages": [msg]}) == {
            "new_components": []
        }

    def test_rejects_non_object_json_array(self):
        msg = self._Msg(content="[1, 2, 3]")
        assert _extract_structured_response({"messages": [msg]}) is None

    def test_rejects_non_object_json_scalar(self):
        msg = self._Msg(content='"just a string"')
        assert _extract_structured_response({"messages": [msg]}) is None


class TestFailureClassification:
    """A batch failure must be classified into a precise, distinct hint."""

    def _exc(self, **attrs):
        exc = Exception("boom")
        for k, v in attrs.items():
            setattr(exc, k, v)
        return exc

    def test_classify_rate_limited(self):
        from aibom.agentic.agent import _classify_failure_hint

        assert _classify_failure_hint(self._exc(status_code=429)) == "rate_limited"

    def test_classify_provider_outage(self):
        from aibom.agentic.agent import _classify_failure_hint

        assert _classify_failure_hint(self._exc(status_code=503)) == "provider_outage"
        assert _classify_failure_hint(self._exc(status_code=500)) == "provider_outage"

    def test_classify_structured_output_parse_error(self):
        from aibom.agentic.agent import _classify_failure_hint

        # StructuredOutputValidationError carries an ai_message attribute.
        exc = self._exc(ai_message=object())
        assert _classify_failure_hint(exc) == "structured_output_parse_error"

    def test_classify_generic_unknown_stays_recursion_limit(self):
        from aibom.agentic.agent import _classify_failure_hint

        assert _classify_failure_hint(RuntimeError("x")) == "batch_recursion_limit"

    def test_refusal_present(self):
        from aibom.agentic.agent import _refusal_present

        class _M:
            def __init__(self, ak):
                self.additional_kwargs = ak

        assert _refusal_present({"messages": [_M({"refusal": "no"})]}) is True
        assert _refusal_present({"messages": [_M({})]}) is False
        assert _refusal_present({"messages": []}) is False


class TestBuildModelMaxTokens:
    @patch("aibom.agentic.agent._build_rate_limiter", return_value=None)
    @patch("aibom.llm_factory.build_chat_model")
    def test_no_default_max_tokens(self, mock_build, _mock_rl):
        # No cap by default: omit max_tokens so the model generates up to its
        # context limit (a fixed default would truncate or starve input).
        from aibom.agentic.agent import _build_model

        _build_model("some-model", {"provider": "openai"})
        _, kwargs = mock_build.call_args
        assert kwargs.get("max_tokens") is None

    @patch("aibom.agentic.agent._build_rate_limiter", return_value=None)
    @patch("aibom.llm_factory.build_chat_model")
    def test_llm_config_overrides_max_tokens(self, mock_build, _mock_rl):
        from aibom.agentic.agent import _build_model

        _build_model("some-model", {"max_tokens": 123})
        _, kwargs = mock_build.call_args
        assert kwargs.get("max_tokens") == 123


class TestStructuredOutputRecovery:
    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_recovers_from_structured_output_validation_error(
        self, mock_create, _mock_build, _mock_close
    ):
        from aibom.agentic.agent import run_agentic_enrichment

        class _ParseErr(Exception):
            def __init__(self, ai_message):
                super().__init__("parse failed")
                self.ai_message = ai_message

        class _Msg:
            def __init__(self):
                self.content = ""
                self.tool_calls = [
                    {
                        "name": "AgentResponse",
                        "args": {"enriched_components": [], "new_components": []},
                        "id": "call_1",
                    }
                ]
                self.additional_kwargs = {}
                self.usage_metadata = {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                }

        mock_create.return_value = MagicMock(
            invoke=MagicMock(side_effect=_ParseErr(_Msg()))
        )
        comp = AIComponent(
            name="test-model",
            component_type=AIComponentType.MODEL,
            file_path="x.py",
            line_number=1,
            model_name="gpt-4o",
        )
        comps, _rels, _flags, usage = run_agentic_enrichment(
            model_string="bad-model",
            deterministic_components=[comp],
            deterministic_relationships=[],
            scan_paths=["/tmp"],
        )
        assert len(comps) == 1
        # Recovered from the exception's ai_message -> not degraded.
        assert not comps[0].agentic_hint
        # Token usage from the recovered ai_message is counted, not lost.
        assert usage.total_tokens == 15


class TestStrategyFallback:
    """A component that yields no usable output under the primary structured-
    output strategy is recovered by re-running via the alternate (fallback)
    agent; a clean batch never triggers the fallback."""

    def test_fallback_callsite_preserves_attempt_kind_and_context(self):
        from aibom.agentic.agent import _strategy_fallback_pass
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        degraded = AIComponent(
            name="test-model",
            component_type=AIComponentType.MODEL,
            file_path="x.py",
            line_number=1,
            model_name="gpt-4o",
            needs_agentic=False,
            agentic_hint="no_usable_output",
        )
        recovered = degraded.model_copy(
            update={"needs_agentic": False, "agentic_hint": ""}
        )
        telemetry_context = MagicMock()

        with patch(
            "aibom.agentic.agent._run_batch",
            return_value=([recovered], [], [], [], False),
        ) as run_batch:
            _strategy_fallback_pass(
                MagicMock(),
                AIBOMScannerMiddleware(allowed_roots=["/tmp"]),
                [degraded],
                [],
                ["/tmp"],
                None,
                batch_size=5,
                timeout_s=30,
                max_consecutive_failures=3,
                telemetry_context=telemetry_context,
            )

        call = run_batch.call_args
        assert call.kwargs["attempt_kind"] == "fallback"
        assert call.kwargs["attempt_number"] == 3
        assert call.kwargs["telemetry_context"] is telemetry_context

    def test_fallback_recovers_no_usable_output(self):
        from aibom.agentic.agent import _strategy_fallback_pass
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        degraded = AIComponent(
            name="test-model",
            component_type=AIComponentType.MODEL,
            file_path="x.py",
            line_number=1,
            model_name="gpt-4o",
            needs_agentic=False,
            agentic_hint="no_usable_output",
        )
        fallback_agent = MagicMock()
        msg = MagicMock()
        msg.content = json.dumps({"enriched_components": [], "new_components": []})
        fallback_agent.invoke.return_value = {"messages": [msg]}

        enriched, _new, _rels, _flags, _artifacts = _strategy_fallback_pass(
            fallback_agent,
            AIBOMScannerMiddleware(allowed_roots=["/tmp"]),
            [degraded],
            [],
            ["/tmp"],
            None,
            batch_size=5,
            timeout_s=30,
            max_consecutive_failures=3,
        )
        assert len(enriched) == 1
        assert not enriched[0].agentic_hint  # recovered, no longer degraded
        fallback_agent.invoke.assert_called_once()

    def test_no_targets_is_noop(self):
        from aibom.agentic.agent import _strategy_fallback_pass
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        clean = AIComponent(
            name="m",
            component_type=AIComponentType.MODEL,
            file_path="x.py",
            line_number=1,
            model_name="gpt-4o",
        )
        fallback_agent = MagicMock()
        enriched, _n, _r, _f, _a = _strategy_fallback_pass(
            fallback_agent,
            AIBOMScannerMiddleware(allowed_roots=["/tmp"]),
            [clean],
            [],
            ["/tmp"],
            None,
            batch_size=5,
            timeout_s=30,
            max_consecutive_failures=3,
        )
        assert enriched == [clean]
        fallback_agent.invoke.assert_not_called()

    def test_fallback_removal_drops_component(self):
        # A successful fallback can REMOVE a component (middleware omits it from
        # the returned list). The merge must drop it, not keep the stale
        # degraded original.
        from aibom.agentic.agent import _strategy_fallback_pass
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        degraded = AIComponent(
            name="not-really-ai",
            component_type=AIComponentType.MODEL,
            file_path="x.py",
            line_number=1,
            model_name="gpt-4o",
            needs_agentic=False,
            agentic_hint="no_usable_output",
        )
        fallback_agent = MagicMock()
        msg = MagicMock()
        msg.content = json.dumps(
            {
                "enriched_components": [],
                "new_components": [],
                "remove_components": [
                    {
                        "instance_id": degraded.instance_id,
                        "reason": "not an AI component",
                    }
                ],
            }
        )
        fallback_agent.invoke.return_value = {"messages": [msg]}

        enriched, _n, _r, _f, _a = _strategy_fallback_pass(
            fallback_agent,
            AIBOMScannerMiddleware(allowed_roots=["/tmp"]),
            [degraded],
            [],
            ["/tmp"],
            None,
            batch_size=5,
            timeout_s=30,
            max_consecutive_failures=3,
        )
        assert enriched == []  # removed by fallback, not kept as degraded


class TestLazyImport:
    def test_aibom_import_does_not_import_deepagents(self):
        """Importing aibom.agentic should NOT trigger deepagents import."""
        import importlib
        import sys

        mods_before = set(sys.modules.keys())
        importlib.import_module("aibom.agentic")
        mods_after = set(sys.modules.keys())
        new_mods = mods_after - mods_before
        deepagent_mods = [m for m in new_mods if "deepagent" in m.lower()]
        assert (
            deepagent_mods == []
        ), f"Importing aibom.agentic pulled in deepagents: {deepagent_mods}"

    def test_aibom_import_does_not_import_langchain(self):
        """Importing aibom.agentic should NOT trigger langchain import."""
        import importlib
        import sys

        mods_before = set(sys.modules.keys())
        importlib.import_module("aibom.agentic")
        mods_after = set(sys.modules.keys())
        new_mods = mods_after - mods_before
        langchain_mods = [m for m in new_mods if "langchain" in m.lower()]
        assert (
            langchain_mods == []
        ), f"Importing aibom.agentic pulled in langchain: {langchain_mods}"


class TestRunAgenticEnrichment:
    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    def test_disabled_telemetry_skips_version_work(
        self, _mock_build, _mock_close, tmp_path
    ):
        from aibom.agentic.agent import run_agentic_enrichment

        telemetry = MagicMock(enabled=False)
        with patch("aibom.agentic.agent._agentic_telemetry_version") as version:
            run_agentic_enrichment(
                "test-model",
                [],
                [],
                [str(tmp_path)],
                cache_dir=tmp_path / "cache",
                telemetry=telemetry,
            )

        version.assert_not_called()

    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    def test_all_tool_root_guards_activate_only_for_raw_callback_factory(
        self, _mock_build, _mock_close, tmp_path
    ):
        from aibom.agentic.agent import run_agentic_enrichment

        def unmarked_factory():
            return None

        def raw_factory():
            return None

        setattr(raw_factory, "_aibom_strict_tool_roots", True)
        with (
            patch("aibom.agentic.tools.set_strict_tool_root_enforcement") as set_strict,
            patch("aibom.agentic.tools.reset_strict_tool_root_enforcement"),
        ):
            run_agentic_enrichment(
                "test-model", [], [], [str(tmp_path)], cache_dir=tmp_path / "cache"
            )
            run_agentic_enrichment(
                "test-model",
                [],
                [],
                [str(tmp_path)],
                cache_dir=tmp_path / "cache",
                invoke_callback_factory=unmarked_factory,
            )
            run_agentic_enrichment(
                "test-model",
                [],
                [],
                [str(tmp_path)],
                cache_dir=tmp_path / "cache",
                invoke_callback_factory=raw_factory,
            )

        assert [call.args[0] for call in set_strict.call_args_list] == [
            False,
            False,
            True,
        ]

    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent._AgenticResultCache")
    def test_explicit_cache_dir_overrides_default_agentic_cache(
        self,
        mock_cache_cls,
        _mock_build,
        _mock_close,
        tmp_path,
    ):
        from aibom.agentic.agent import run_agentic_enrichment

        cache_dir = tmp_path / "agentic-cache"
        with patch(
            "aibom.agentic.agent._default_agentic_cache_dir",
            side_effect=AssertionError("default cache dir should not be used"),
        ):
            comps, rels, flags, _usage = run_agentic_enrichment(
                model_string="test-model",
                deterministic_components=[],
                deterministic_relationships=[],
                scan_paths=["/tmp"],
                cache_dir=cache_dir,
            )

        mock_cache_cls.assert_called_once_with(
            cache_dir,
            fallback_dirs=[],
        )
        assert comps == []
        assert rels == []
        assert flags == []

    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_merges_agent_output_into_components(
        self, mock_create, _mock_build, _mock_close
    ):
        from aibom.agentic.agent import run_agentic_enrichment

        agent_response = json.dumps(
            {
                "enriched_components": [],
                "new_components": [
                    {
                        "name": "agent-found-model",
                        "component_type": "model",
                        "file_path": "new.py",
                        "line_number": 1,
                        "framework": "openai",
                        "model_name": "gpt-5",
                    }
                ],
                "new_relationships": [],
                "risk_findings": [],
            }
        )

        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = agent_response
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create.return_value = mock_agent

        det_comps = [
            AIComponent(
                name="existing",
                component_type=AIComponentType.DEPENDENCY,
                file_path="app.py",
                line_number=1,
            )
        ]
        comps, rels, flags, _usage = run_agentic_enrichment(
            model_string="test-model",
            deterministic_components=det_comps,
            deterministic_relationships=[],
            scan_paths=["/tmp"],
        )
        assert len(comps) == 2
        names = {c.name for c in comps}
        assert "existing" in names
        assert "agent-found-model" in names

    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_cached_rerun_replays_new_components(
        self, mock_create, _mock_build, _mock_close, tmp_path
    ):
        from aibom.agentic.agent import run_agentic_enrichment

        agent_response = json.dumps(
            {
                "enriched_components": [],
                "new_components": [
                    {
                        "name": "agent-found-model",
                        "component_type": "model",
                        "file_path": "new.py",
                        "line_number": 1,
                        "framework": "openai",
                        "model_name": "gpt-5",
                    }
                ],
                "new_relationships": [],
                "risk_findings": [],
            }
        )

        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = agent_response
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create.return_value = mock_agent

        det_comps = [
            AIComponent(
                name="existing",
                component_type=AIComponentType.DEPENDENCY,
                file_path="app.py",
                line_number=1,
            )
        ]
        scan_root = tmp_path / "repo"
        scan_root.mkdir()

        with patch(
            "aibom.agentic.agent._default_agentic_cache_dir",
            return_value=tmp_path / "agentic-cache",
        ):
            fresh, _, _, _ = run_agentic_enrichment(
                model_string="test-model",
                deterministic_components=det_comps,
                deterministic_relationships=[],
                scan_paths=[str(scan_root)],
            )
            cached, _, _, _ = run_agentic_enrichment(
                model_string="test-model",
                deterministic_components=det_comps,
                deterministic_relationships=[],
                scan_paths=[str(scan_root)],
            )

        assert {c.name for c in fresh} == {"existing", "agent-found-model"}
        assert {c.name for c in cached} == {c.name for c in fresh}
        assert mock_agent.invoke.call_count == 1

    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_partial_cached_rerun_replays_batch_findings_once(
        self, mock_create, _mock_build, _mock_close, tmp_path
    ):
        from aibom.agentic.agent import run_agentic_enrichment

        first_msg = MagicMock()
        first_msg.content = json.dumps(
            {
                "enriched_components": [],
                "new_components": [
                    {
                        "name": "agent-found-model",
                        "component_type": "model",
                        "file_path": "new.py",
                        "line_number": 1,
                        "framework": "openai",
                        "model_name": "gpt-5",
                    }
                ],
                "new_relationships": [
                    {
                        "source_name": "existing-a",
                        "target_name": "existing-b",
                        "relationship_type": "USES_MODEL",
                    }
                ],
                "risk_findings": [
                    {
                        "flag": "cache_replayed",
                        "description": "cached finding",
                        "file_path": "app_a.py",
                        "line_number": 1,
                        "severity": "low",
                    }
                ],
            }
        )
        second_msg = MagicMock()
        second_msg.content = json.dumps(
            {
                "enriched_components": [],
                "new_components": [],
                "new_relationships": [],
                "risk_findings": [],
            }
        )
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [
            {"messages": [first_msg]},
            {"messages": [second_msg]},
        ]
        mock_create.return_value = mock_agent

        det_comps_first = [
            AIComponent(
                name="existing-a",
                component_type=AIComponentType.DEPENDENCY,
                file_path="app_a.py",
                line_number=1,
            ),
            AIComponent(
                name="existing-b",
                component_type=AIComponentType.DEPENDENCY,
                file_path="app_b.py",
                line_number=1,
            ),
        ]
        det_comps_second = det_comps_first + [
            AIComponent(
                name="existing-c",
                component_type=AIComponentType.DEPENDENCY,
                file_path="app_c.py",
                line_number=1,
            )
        ]
        scan_root = tmp_path / "repo"
        scan_root.mkdir()
        cache_dir = tmp_path / "agentic-cache"

        fresh_components, fresh_rels, fresh_flags, _ = run_agentic_enrichment(
            model_string="test-model",
            deterministic_components=det_comps_first,
            deterministic_relationships=[],
            scan_paths=[str(scan_root)],
            cache_dir=cache_dir,
        )
        cached_components, cached_rels, cached_flags, _ = run_agentic_enrichment(
            model_string="test-model",
            deterministic_components=det_comps_second,
            deterministic_relationships=[],
            scan_paths=[str(scan_root)],
            cache_dir=cache_dir,
        )

        assert {c.name for c in fresh_components} == {
            "existing-a",
            "existing-b",
            "agent-found-model",
        }
        assert {c.name for c in cached_components} == {
            "existing-a",
            "existing-b",
            "existing-c",
            "agent-found-model",
        }
        assert sum(c.name == "agent-found-model" for c in cached_components) == 1
        assert len(cached_rels) == 1
        assert cached_rels[0].source_name == "existing-a"
        assert cached_rels[0].target_name == "existing-b"
        assert len(cached_flags) == 1
        assert cached_flags[0].flag == "cache_replayed"
        assert len(fresh_rels) == 1
        assert len(fresh_flags) == 1
        assert mock_agent.invoke.call_count == 2

    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_cross_repo_coordination_rerun_uses_cache(
        self, mock_create, _mock_build, _mock_close, tmp_path
    ):
        from types import SimpleNamespace

        from aibom.agentic.agent import run_cross_repo_coordination

        mock_msg = MagicMock()
        mock_msg.content = json.dumps(
            {
                "new_relationships": [
                    {
                        "source_name": "repo-a-model",
                        "target_name": "repo-b-endpoint",
                        "relationship_type": "USES_MODEL",
                    }
                ],
                "risk_findings": [
                    {
                        "flag": "cross_repo_env_var_mismatch",
                        "description": "backend mismatch",
                        "severity": "medium",
                    }
                ],
            }
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create.return_value = mock_agent

        per_repo_results = {
            "/repo-a": {
                "components": [
                    AIComponent(
                        name="repo-a-model",
                        component_type=AIComponentType.MODEL,
                        file_path="/repo-a/app.py",
                        line_number=10,
                        model_name="gpt-4o-mini",
                    )
                ],
                "_unresolved_env_vars": [],
            },
            "/repo-b": {
                "components": [
                    AIComponent(
                        name="repo-b-endpoint",
                        component_type=AIComponentType.LLM_ENDPOINT,
                        file_path="/repo-b/values.yaml",
                        line_number=20,
                    )
                ],
                "_unresolved_env_vars": [],
            },
        }

        with (
            patch(
                "aibom.agentic.cross_repo.cross_repo_summary_tool",
                return_value=json.dumps(
                    {
                        "shared_env_vars": [],
                        "shared_packages": [],
                        "unresolved_env_var_refs": [],
                    }
                ),
            ),
            patch(
                "aibom.agentic.cross_repo.build_cross_repo_tools",
                return_value=[MagicMock()],
            ),
            patch(
                "aibom.cross_ref.build_env_index",
                return_value=SimpleNamespace(env={}),
            ),
        ):
            fresh_rels, fresh_flags = run_cross_repo_coordination(
                model_string="test-model",
                per_repo_results=per_repo_results,
                cache_dir=tmp_path / "agentic-cache",
            )
            cached_rels, cached_flags = run_cross_repo_coordination(
                model_string="test-model",
                per_repo_results=per_repo_results,
                cache_dir=tmp_path / "agentic-cache",
            )

        assert len(fresh_rels) == 1
        assert len(cached_rels) == 1
        assert fresh_rels[0].source_name == cached_rels[0].source_name
        assert fresh_rels[0].target_name == cached_rels[0].target_name
        assert len(fresh_flags) == 1
        assert len(cached_flags) == 1
        assert fresh_flags[0].flag == cached_flags[0].flag
        assert mock_agent.invoke.call_count == 1

    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_handles_agent_failure_gracefully(
        self, mock_create, _mock_build, _mock_close
    ):
        from aibom.agentic.agent import run_agentic_enrichment

        mock_create.return_value = MagicMock(
            invoke=MagicMock(side_effect=RuntimeError("LLM unavailable"))
        )
        comp = AIComponent(
            name="test-model",
            component_type=AIComponentType.MODEL,
            file_path="x.py",
            line_number=1,
            model_name="gpt-4o",
        )
        comps, rels, flags, _usage = run_agentic_enrichment(
            model_string="bad-model",
            deterministic_components=[comp],
            deterministic_relationships=[],
            scan_paths=["/tmp"],
        )
        assert len(comps) == 1
        assert comps[0].name == "test-model"

    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_batching_splits_large_input(self, mock_create, _mock_build, _mock_close):
        from aibom.agentic.agent import run_agentic_enrichment

        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = json.dumps(
            {
                "enriched_components": [],
                "new_components": [],
                "new_relationships": [],
                "risk_findings": [],
            }
        )
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_msg]})
        mock_create.return_value = mock_agent

        comps = [
            AIComponent(
                name=f"model-{i}",
                component_type=AIComponentType.MODEL,
                file_path=f"f{i}.py",
                line_number=i,
                model_name=f"gpt-{i}",
            )
            for i in range(32)
        ]
        result_comps, _, _, _ = run_agentic_enrichment(
            model_string="test-model",
            deterministic_components=comps,
            deterministic_relationships=[],
            scan_paths=["/tmp"],
            batch_size=15,
            max_concurrent=3,
        )
        assert mock_agent.ainvoke.call_count >= 2  # 15 + 15 + 2, parallel

    def test_async_batch_callback_setup_does_not_block_event_loop(self):
        import asyncio
        import threading
        import time

        from aibom.agentic.agent import _run_batch_async
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        callback_started = threading.Event()
        release_callback = threading.Event()

        def blocking_callback_factory():
            callback_started.set()
            release_callback.wait()
            return None

        class FakeAgent:
            async def ainvoke(self, _message, config=None):
                return {
                    "structured_response": {
                        "enriched_components": [],
                        "new_components": [],
                        "new_relationships": [],
                        "risk_findings": [],
                    }
                }

        component = AIComponent(
            name="test-model",
            component_type=AIComponentType.MODEL,
            file_path="x.py",
            line_number=1,
            model_name="gpt-4o",
        )

        async def exercise():
            started = time.monotonic()
            batch_task = asyncio.create_task(
                _run_batch_async(
                    FakeAgent(),
                    AIBOMScannerMiddleware(allowed_roots=["/tmp"]),
                    [component],
                    [],
                    ["/tmp"],
                    1,
                    1,
                    timeout_s=1,
                    invoke_callback_factory=blocking_callback_factory,
                )
            )
            while not callback_started.is_set():
                await asyncio.sleep(0.001)

            # This coroutine can run while the synchronous factory is still
            # blocked only when callback construction is off the event loop.
            assert time.monotonic() - started < 0.9
            assert not batch_task.done()
            release_callback.set()
            await asyncio.wait_for(batch_task, timeout=1)

        safety_release = threading.Timer(1.0, release_callback.set)
        safety_release.daemon = True
        safety_release.start()
        try:
            asyncio.run(exercise())
        finally:
            release_callback.set()
            safety_release.cancel()

    def test_async_batch_raw_coercion_setup_does_not_block_event_loop(self):
        import asyncio
        import threading
        import time

        from aibom.agentic.agent import _run_batch_async
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        coercion_started = threading.Event()
        release_coercion = threading.Event()

        def blocking_resolve(*_args, **_kwargs):
            coercion_started.set()
            release_coercion.wait()
            return {
                "enriched_components": [],
                "new_components": [],
                "new_relationships": [],
                "risk_findings": [],
            }

        class FakeAgent:
            needs_coercion = True

            async def ainvoke(self, _message, config=None):
                return {"messages": []}

        component = AIComponent(
            name="test-model",
            component_type=AIComponentType.MODEL,
            file_path="x.py",
            line_number=1,
            model_name="gpt-4o",
        )

        async def exercise():
            started = time.monotonic()
            with patch(
                "aibom.agentic.agent._resolve_batch_data",
                side_effect=blocking_resolve,
            ):
                batch_task = asyncio.create_task(
                    _run_batch_async(
                        FakeAgent(),
                        AIBOMScannerMiddleware(allowed_roots=["/tmp"]),
                        [component],
                        [],
                        ["/tmp"],
                        1,
                        1,
                        timeout_s=1,
                        invoke_callback_factory=lambda: None,
                    )
                )
                while not coercion_started.is_set():
                    await asyncio.sleep(0.001)

                assert time.monotonic() - started < 0.9
                assert not batch_task.done()
                release_coercion.set()
                await asyncio.wait_for(batch_task, timeout=1)

        safety_release = threading.Timer(1.0, release_coercion.set)
        safety_release.daemon = True
        safety_release.start()
        try:
            asyncio.run(exercise())
        finally:
            release_coercion.set()
            safety_release.cancel()

    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_single_batch_uses_sequential(self, mock_create, _mock_build, _mock_close):
        """A single batch should use invoke (sequential), not ainvoke."""
        from aibom.agentic.agent import run_agentic_enrichment

        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = json.dumps(
            {
                "enriched_components": [],
                "new_components": [],
                "new_relationships": [],
                "risk_findings": [],
            }
        )
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create.return_value = mock_agent

        comps = [
            AIComponent(
                name="gpt-4o",
                component_type=AIComponentType.MODEL,
                file_path="a.py",
                line_number=1,
                model_name="gpt-4o",
            )
        ]
        run_agentic_enrichment(
            model_string="test-model",
            deterministic_components=comps,
            deterministic_relationships=[],
            scan_paths=["/tmp"],
            batch_size=5,
        )
        assert mock_agent.invoke.call_count == 1

    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_tiered_model_uses_fast_for_simple(
        self, mock_create, mock_build, _mock_close
    ):
        from aibom.agentic.agent import run_agentic_enrichment

        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = json.dumps(
            {
                "enriched_components": [],
                "new_components": [],
                "new_relationships": [],
                "risk_findings": [],
            }
        )
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create.return_value = mock_agent

        simple = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="a.py",
            line_number=1,
            model_name="gpt-4o",
        )
        complex_ = AIComponent(
            name="some-agent",
            component_type=AIComponentType.AGENT,
            file_path="b.py",
            line_number=5,
        )
        run_agentic_enrichment(
            model_string="expensive-model",
            deterministic_components=[simple, complex_],
            deterministic_relationships=[],
            scan_paths=["/tmp"],
            fast_model="cheap-model",
        )
        calls = mock_create.call_args_list
        assert calls[0][0][0] == "cheap-model"
        assert calls[1][0][0] == "expensive-model"


class TestToolStats:
    def test_tool_stats_isolated_per_reset(self):
        from aibom.agentic.tools import (
            _get_stats_dict,
            _reset_tool_stats,
            get_tool_stats,
        )

        _reset_tool_stats()
        assert get_tool_stats() == {}
        stats = _get_stats_dict()
        stats["test_tool"] = {"calls": 1, "total_s": 0.5, "errors": 0}
        assert get_tool_stats()["test_tool"]["calls"] == 1

        _reset_tool_stats()
        assert get_tool_stats() == {}

    def test_track_tool_decorator(self):
        from aibom.agentic.tools import _reset_tool_stats, _track_tool, get_tool_stats

        _reset_tool_stats()

        @_track_tool("my_tool")
        def dummy(x):
            return x * 2

        assert dummy(5) == 10
        stats = get_tool_stats()
        assert "my_tool" in stats
        assert stats["my_tool"]["calls"] == 1
        assert stats["my_tool"]["errors"] == 0


class TestLocalityAwareBatching:
    """Locality-aware batching groups co-located components."""

    def test_groups_by_directory(self):
        from aibom.agentic.agent import _locality_aware_batches

        comps = [
            AIComponent(
                name="a",
                component_type=AIComponentType.MODEL,
                file_path="/repo/dir1/a.py",
                line_number=1,
            ),
            AIComponent(
                name="b",
                component_type=AIComponentType.MODEL,
                file_path="/repo/dir1/b.py",
                line_number=2,
            ),
            AIComponent(
                name="c",
                component_type=AIComponentType.MODEL,
                file_path="/repo/dir2/c.py",
                line_number=1,
            ),
            AIComponent(
                name="d",
                component_type=AIComponentType.MODEL,
                file_path="/repo/dir2/d.py",
                line_number=2,
            ),
        ]
        batches = _locality_aware_batches(comps, batch_size=3)
        assert len(batches) == 2
        dirs_b0 = {str(Path(c.file_path).parent) for c in batches[0]}
        assert len(dirs_b0) <= 2

    def test_single_dir_stays_together(self):
        from aibom.agentic.agent import _locality_aware_batches

        comps = [
            AIComponent(
                name=f"m{i}",
                component_type=AIComponentType.MODEL,
                file_path=f"/repo/pkg/{i}.py",
                line_number=i,
            )
            for i in range(4)
        ]
        batches = _locality_aware_batches(comps, batch_size=5)
        assert len(batches) == 1
        assert len(batches[0]) == 4

    def test_respects_batch_size(self):
        from aibom.agentic.agent import _locality_aware_batches

        comps = [
            AIComponent(
                name=f"m{i}",
                component_type=AIComponentType.MODEL,
                file_path=f"/repo/pkg/{i}.py",
                line_number=i,
            )
            for i in range(7)
        ]
        batches = _locality_aware_batches(comps, batch_size=3)
        assert all(len(b) <= 3 for b in batches)
        assert sum(len(b) for b in batches) == 7


class TestAgenticResultCache:
    """Content-hash result cache for agentic enrichment."""

    def test_cache_miss_then_hit(self, tmp_path):
        from aibom.agentic.agent import _AgenticResultCache, _component_cache_key

        cache = _AgenticResultCache(tmp_path / "cache")
        comp = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="a.py",
            line_number=1,
            model_name="gpt-4o",
        )
        key = _component_cache_key(comp)
        assert cache.get(key) is None

        cache.put(key, {"enriched_components": [], "new_components": []})
        assert cache.get(key) is not None

    def test_partition_splits_cached_and_uncached(self, tmp_path):
        from aibom.agentic.agent import _AgenticResultCache, _component_cache_key

        cache = _AgenticResultCache(tmp_path / "cache")
        c1 = AIComponent(
            name="a",
            component_type=AIComponentType.MODEL,
            file_path="a.py",
            line_number=1,
            model_name="gpt-4o",
        )
        c2 = AIComponent(
            name="b",
            component_type=AIComponentType.MODEL,
            file_path="b.py",
            line_number=1,
            model_name="gpt-5",
        )
        cache.put(_component_cache_key(c1), {"enriched_components": []})

        cached, uncached = cache.partition([c1, c2])
        assert len(cached) == 1
        assert cached[0].name == "a"
        assert len(uncached) == 1
        assert uncached[0].name == "b"

    def test_disk_persistence(self, tmp_path):
        from aibom.agentic.agent import _AgenticResultCache, _component_cache_key

        cache_dir = tmp_path / "cache"
        cache1 = _AgenticResultCache(cache_dir)
        comp = AIComponent(
            name="x",
            component_type=AIComponentType.MODEL,
            file_path="x.py",
            line_number=1,
        )
        key = _component_cache_key(comp)
        cache1.put(key, {"enriched_components": [], "test": True})

        cache2 = _AgenticResultCache(cache_dir)
        assert cache2.get(key) is not None
        assert cache2.get(key)["test"] is True

    def test_apply_cached_replays_full_component_snapshot(self, tmp_path):
        from aibom.agentic.agent import _AgenticResultCache, _component_cache_key
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        cache = _AgenticResultCache(tmp_path / "cache")
        before = AIComponent(
            name="ada-ep",
            component_type=AIComponentType.EMBEDDING,
            file_path="cfg.yaml",
            line_number=3,
            description="legacy description",
            framework="legacy",
            metadata={"old": True},
            heuristic_confidence=0.2,
            agentic_hint="stale_hint",
        )
        after = before.model_copy(
            update={
                "component_type": AIComponentType.MODEL_ENDPOINT,
                "description": "",
                "framework": "openai",
                "metadata": {"verified": True},
                "heuristic_confidence": 0.97,
                "agentic_hint": "",
                "needs_agentic": False,
            }
        )
        cache.put(
            _component_cache_key(before),
            {
                "cached_component": after.model_dump(mode="json"),
                "enriched_components": [],
                "new_components": [],
                "remove_components": [],
                "reclassify_components": [],
                "new_relationships": [],
                "risk_findings": [],
            },
        )

        enriched, new, rels, flags = cache.apply_cached(
            [before], AIBOMScannerMiddleware()
        )

        assert new == []
        assert rels == []
        assert flags == []
        assert len(enriched) == 1
        assert enriched[0].component_type == AIComponentType.MODEL_ENDPOINT
        assert enriched[0].description == ""
        assert enriched[0].framework == "openai"
        assert enriched[0].metadata == {"verified": True}
        assert enriched[0].heuristic_confidence == 0.97
        assert enriched[0].agentic_hint == ""
        assert enriched[0].needs_agentic is False

    def test_apply_cached_hydrates_component_snippet_when_enabled(self, tmp_path):
        from aibom.agentic.agent import _AgenticResultCache, _component_cache_key
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        source = tmp_path / "service.py"
        source.write_text(
            "setup()\n" "agent = RouterAgent()\n" "agent.run(task)\n",
            encoding="utf-8",
        )

        cache = _AgenticResultCache(tmp_path / "cache")
        before = AIComponent(
            name="router_agent",
            component_type=AIComponentType.AGENT,
            file_path=str(source),
            line_number=2,
        )
        after = before.model_copy(
            update={
                "decision_annotation": DecisionAnnotation(
                    decision="confirmed",
                    justification="The code instantiates and runs the agent.",
                    evidence_kinds=["code_context"],
                    evidence_locations=[
                        EvidenceLocation(
                            file_path=str(source),
                            start_line=2,
                            end_line=3,
                            role="primary",
                        )
                    ],
                ),
                "needs_agentic": False,
            }
        )
        cache.put(
            _component_cache_key(before),
            {
                "cached_component": after.model_dump(mode="json"),
                "enriched_components": [],
                "new_components": [],
                "remove_components": [],
                "reclassify_components": [],
                "new_relationships": [],
                "risk_findings": [],
            },
        )

        enriched, new, rels, flags = cache.apply_cached(
            [before],
            AIBOMScannerMiddleware(include_code_snippets=True),
        )

        assert new == []
        assert rels == []
        assert flags == []
        assert enriched[0].decision_annotation is not None
        assert enriched[0].decision_annotation.code_snippet is not None
        assert enriched[0].decision_annotation.code_snippet.text == (
            "agent = RouterAgent()\nagent.run(task)\n"
        )


class TestRunTier:
    @patch("aibom.agentic.agent._RETRY_COOLDOWN_S", 0)
    def test_retry_callsite_preserves_attempt_kind_and_context(self):
        from aibom.agentic.agent import _run_tier
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        component = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="app.py",
            line_number=1,
            model_name="gpt-4o",
        )
        degraded = component.model_copy(
            update={"needs_agentic": False, "agentic_hint": "batch_timeout"}
        )
        recovered = component.model_copy(
            update={"needs_agentic": False, "agentic_hint": ""}
        )
        telemetry_context = MagicMock()

        with patch(
            "aibom.agentic.agent._run_batch",
            side_effect=[
                ([degraded], [], [], [], True),
                ([recovered], [], [], [], False),
            ],
        ) as run_batch:
            enriched, _, _, _ = _run_tier(
                agent=MagicMock(),
                middleware=AIBOMScannerMiddleware(),
                components=[component],
                relationships=[],
                scan_paths=["/tmp"],
                batch_size=4,
                max_concurrent=1,
                all_components=[component],
                cache=None,
                telemetry_context=telemetry_context,
            )

        assert enriched == [recovered]
        assert run_batch.call_count == 2
        retry_call = run_batch.call_args_list[1]
        assert retry_call.kwargs["attempt_kind"] == "retry"
        assert retry_call.kwargs["attempt_number"] == 2
        assert retry_call.kwargs["telemetry_context"] is telemetry_context

    @patch("aibom.agentic.agent._RETRY_COOLDOWN_S", 0)
    def test_timeout_retry_repartitions_into_smaller_batches(self):
        from aibom.agentic.agent import _run_tier
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        components = [
            AIComponent(
                name=f"model-{index}",
                component_type=AIComponentType.MODEL,
                file_path="app.py",
                line_number=index,
                model_name=f"model-{index}",
            )
            for index in range(1, 5)
        ]
        degraded = [
            component.model_copy(
                update={
                    "needs_agentic": True,
                    "agentic_hint": "batch_timeout",
                }
            )
            for component in components
        ]

        def run_batch(*args, **kwargs):
            batch = args[2]
            if kwargs.get("attempt_kind") == "retry":
                recovered = [
                    component.model_copy(
                        update={"needs_agentic": False, "agentic_hint": ""}
                    )
                    for component in batch
                ]
                return recovered, [], [], [], False
            return degraded, [], [], [], True

        with patch(
            "aibom.agentic.agent._run_batch",
            side_effect=run_batch,
        ) as run_batch_mock:
            enriched, _, _, _ = _run_tier(
                agent=MagicMock(),
                middleware=AIBOMScannerMiddleware(),
                components=components,
                relationships=[],
                scan_paths=["/tmp"],
                batch_size=4,
                max_concurrent=1,
                all_components=components,
                cache=None,
            )

        assert len(enriched) == 4
        retry_calls = [
            call
            for call in run_batch_mock.call_args_list
            if call.kwargs.get("attempt_kind") == "retry"
        ]
        assert [len(call.args[2]) for call in retry_calls] == [2, 2]

    def test_memo_hit_emits_cache_trace_without_running_the_agent(self):
        from aibom.agentic.agent import _DecisionMemo, _run_tier
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        component = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="app.py",
            line_number=1,
            model_name="gpt-4o",
        )
        resolved = component.model_copy(
            update={"heuristic_confidence": 0.97, "needs_agentic": False}
        )
        memo = _DecisionMemo()
        memo.record(component, resolved)
        agent = MagicMock()
        telemetry_context = MagicMock()

        with patch("aibom.agentic.agent._record_cache_hit_trace") as record_trace:
            enriched, new, rels, flags = _run_tier(
                agent=agent,
                middleware=AIBOMScannerMiddleware(),
                components=[component],
                relationships=[],
                scan_paths=["/tmp"],
                batch_size=4,
                max_concurrent=1,
                all_components=[component],
                cache=None,
                memo=memo,
                telemetry_context=telemetry_context,
            )

        agent.invoke.assert_not_called()
        assert new == []
        assert rels == []
        assert flags == []
        assert enriched == [resolved]
        record_trace.assert_called_once_with(
            telemetry_context,
            [component],
            [resolved],
            [],
            [],
            [],
        )

    def test_tier_cache_hit_populates_memo(self, tmp_path):
        from aibom.agentic.agent import (
            _AgenticResultCache,
            _build_tier_cache_payload,
            _DecisionMemo,
            _run_tier,
            _tier_cache_key,
        )
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        comp = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="app.py",
            line_number=1,
            model_name="gpt-4o",
        )
        cached = comp.model_copy(
            update={"heuristic_confidence": 0.99, "needs_agentic": False}
        )
        cache = _AgenticResultCache(tmp_path / "cache")
        cache.put(
            _tier_cache_key([comp]), _build_tier_cache_payload([cached], [], [], [])
        )

        memo = _DecisionMemo()
        agent = MagicMock()
        enriched, new, rels, flags = _run_tier(
            agent=agent,
            middleware=AIBOMScannerMiddleware(),
            components=[comp],
            relationships=[],
            scan_paths=["/tmp"],
            batch_size=4,
            max_concurrent=1,
            all_components=[comp],
            cache=cache,
            memo=memo,
        )

        agent.invoke.assert_not_called()
        assert new == []
        assert rels == []
        assert flags == []
        assert len(enriched) == 1
        assert enriched[0].heuristic_confidence == 0.99
        assert memo.lookup(comp) == {"action": "keep", "heuristic_confidence": 0.99}

    def test_cache_only_fast_path_populates_memo(self, tmp_path):
        from aibom.agentic.agent import (
            _AgenticResultCache,
            _component_cache_key,
            _DecisionMemo,
            _run_tier,
        )
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        comp = AIComponent(
            name="ada-ep",
            component_type=AIComponentType.EMBEDDING,
            file_path="cfg.yaml",
            line_number=7,
        )
        cached = comp.model_copy(
            update={
                "component_type": AIComponentType.MODEL_ENDPOINT,
                "heuristic_confidence": 0.91,
                "needs_agentic": False,
            }
        )
        cache = _AgenticResultCache(tmp_path / "cache")
        cache.put(
            _component_cache_key(comp),
            {
                "cached_component": cached.model_dump(mode="json"),
                "enriched_components": [],
                "new_components": [],
                "remove_components": [],
                "reclassify_components": [],
                "new_relationships": [],
                "risk_findings": [],
            },
        )

        memo = _DecisionMemo()
        agent = MagicMock()
        enriched, new, rels, flags = _run_tier(
            agent=agent,
            middleware=AIBOMScannerMiddleware(),
            components=[comp],
            relationships=[],
            scan_paths=["/tmp"],
            batch_size=4,
            max_concurrent=1,
            all_components=[comp],
            cache=cache,
            memo=memo,
        )

        agent.invoke.assert_not_called()
        assert new == []
        assert rels == []
        assert flags == []
        assert len(enriched) == 1
        assert enriched[0].component_type == AIComponentType.MODEL_ENDPOINT
        assert enriched[0].heuristic_confidence == 0.91
        assert memo.lookup(comp) == {
            "action": "reclassify",
            "new_type": "model_endpoint",
            "heuristic_confidence": 0.91,
        }


class TestDecisionMemo:
    """Intra-run decision memoization for context-free types."""

    def test_memo_keys_context_free_types(self):
        from aibom.agentic.agent import _DecisionMemo

        memo = _DecisionMemo()
        dep = AIComponent(
            name="torch",
            component_type=AIComponentType.DEPENDENCY,
            file_path="req.txt",
            line_number=1,
        )
        model = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="a.py",
            line_number=1,
            model_name="gpt-4o",
        )
        emb = AIComponent(
            name="ada-002",
            component_type=AIComponentType.EMBEDDING,
            file_path="b.py",
            line_number=1,
        )
        artifact = AIComponent(
            name="model.onnx",
            component_type=AIComponentType.MODEL_ARTIFACT,
            file_path="c.py",
            line_number=1,
        )

        assert memo._key(dep) is not None
        assert memo._key(model) is not None
        assert memo._key(emb) is not None
        assert memo._key(artifact) is not None

    def test_memo_skips_context_dependent_types(self):
        from aibom.agentic.agent import _DecisionMemo

        memo = _DecisionMemo()
        endpoint = AIComponent(
            name="env:ENDPOINT",
            component_type=AIComponentType.LLM_ENDPOINT,
            file_path="v.yaml",
            line_number=5,
        )
        prompt = AIComponent(
            name="my-prompt",
            component_type=AIComponentType.PROMPT,
            file_path="p.py",
            line_number=1,
        )
        secret = AIComponent(
            name="api-key",
            component_type=AIComponentType.SECRET,
            file_path="s.py",
            line_number=1,
        )

        assert memo._key(endpoint) is None
        assert memo._key(prompt) is None
        assert memo._key(secret) is None

    def test_record_and_lookup_keep(self):
        from aibom.agentic.agent import _DecisionMemo

        memo = _DecisionMemo()
        before = AIComponent(
            name="torch",
            component_type=AIComponentType.DEPENDENCY,
            file_path="req.txt",
            line_number=1,
        )
        after = before.model_copy(
            update={
                "heuristic_confidence": 0.95,
                "needs_agentic": False,
                "decision_annotation": DecisionAnnotation(
                    decision="confirmed",
                    justification="The dependency is declared.",
                ),
            }
        )

        memo.record(before, after)
        verdict = memo.lookup(before)
        assert verdict is not None
        assert verdict["action"] == "keep"
        assert verdict["heuristic_confidence"] == 0.95
        replayed = memo.apply([before])[0]
        assert replayed.decision_annotation is not None
        assert replayed.decision_annotation.decision == "confirmed"

    def test_record_and_lookup_remove(self):
        from aibom.agentic.agent import _DecisionMemo

        memo = _DecisionMemo()
        before = AIComponent(
            name="requests",
            component_type=AIComponentType.DEPENDENCY,
            file_path="req.txt",
            line_number=5,
        )

        memo.record(before, None)
        verdict = memo.lookup(before)
        assert verdict is not None
        assert verdict["action"] == "remove"

    def test_record_and_lookup_reclassify(self):
        from aibom.agentic.agent import _DecisionMemo

        memo = _DecisionMemo()
        before = AIComponent(
            name="ada-ep",
            component_type=AIComponentType.EMBEDDING,
            file_path="cfg.yaml",
            line_number=3,
        )
        after = before.model_copy(
            update={
                "component_type": AIComponentType.MODEL_ENDPOINT,
                "heuristic_confidence": 0.9,
            }
        )

        memo.record(before, after)
        verdict = memo.lookup(before)
        assert verdict is not None
        assert verdict["action"] == "reclassify"
        assert verdict["new_type"] == "model_endpoint"
        assert verdict["heuristic_confidence"] == 0.9

    def test_partition_separates_hits_and_misses(self):
        from aibom.agentic.agent import _DecisionMemo

        memo = _DecisionMemo()
        c1 = AIComponent(
            name="torch",
            component_type=AIComponentType.DEPENDENCY,
            file_path="a.txt",
            line_number=1,
        )
        memo.record(
            c1,
            c1.model_copy(update={"heuristic_confidence": 0.9, "needs_agentic": False}),
        )

        c1_dup = AIComponent(
            name="torch",
            component_type=AIComponentType.DEPENDENCY,
            file_path="c.txt",
            line_number=3,
        )
        c2 = AIComponent(
            name="flask",
            component_type=AIComponentType.DEPENDENCY,
            file_path="b.txt",
            line_number=2,
        )
        hits, misses = memo.partition([c1_dup, c2])
        assert len(hits) == 1
        assert hits[0].name == "torch"
        assert len(misses) == 1
        assert misses[0].name == "flask"

    def test_apply_keeps(self):
        from aibom.agentic.agent import _DecisionMemo

        memo = _DecisionMemo()
        c = AIComponent(
            name="torch",
            component_type=AIComponentType.DEPENDENCY,
            file_path="r.txt",
            line_number=1,
        )
        memo.record(c, c.model_copy(update={"heuristic_confidence": 0.95}))

        result = memo.apply([c])
        assert len(result) == 1
        assert result[0].heuristic_confidence == 0.95
        assert result[0].needs_agentic is False

    def test_apply_removes(self):
        from aibom.agentic.agent import _DecisionMemo

        memo = _DecisionMemo()
        c = AIComponent(
            name="requests",
            component_type=AIComponentType.DEPENDENCY,
            file_path="r.txt",
            line_number=1,
        )
        memo.record(c, None)

        result = memo.apply([c])
        assert len(result) == 0

    def test_apply_reclassifies(self):
        from aibom.agentic.agent import _DecisionMemo

        memo = _DecisionMemo()
        before = AIComponent(
            name="ada-ep",
            component_type=AIComponentType.EMBEDDING,
            file_path="x.yaml",
            line_number=1,
        )
        after = before.model_copy(
            update={
                "component_type": AIComponentType.MODEL_ENDPOINT,
                "heuristic_confidence": 0.85,
            }
        )
        memo.record(before, after)

        dup = AIComponent(
            name="ada-ep",
            component_type=AIComponentType.EMBEDDING,
            file_path="y.yaml",
            line_number=5,
        )
        result = memo.apply([dup])
        assert len(result) == 1
        assert result[0].component_type == AIComponentType.MODEL_ENDPOINT
        assert result[0].heuristic_confidence == 0.85
        assert result[0].needs_agentic is False

    def test_context_dependent_skips_memo(self):
        from aibom.agentic.agent import _DecisionMemo

        memo = _DecisionMemo()
        ep = AIComponent(
            name="env:WEAVIATE_EP",
            component_type=AIComponentType.LLM_ENDPOINT,
            file_path="v.yaml",
            line_number=1,
        )
        memo.record(
            ep, ep.model_copy(update={"component_type": AIComponentType.VECTOR_STORE})
        )

        assert memo.lookup(ep) is None
        assert len(memo) == 0

    def test_len(self):
        from aibom.agentic.agent import _DecisionMemo

        memo = _DecisionMemo()
        assert len(memo) == 0
        c = AIComponent(
            name="torch",
            component_type=AIComponentType.DEPENDENCY,
            file_path="r.txt",
            line_number=1,
        )
        memo.record(c, c.model_copy(update={"heuristic_confidence": 0.9}))
        assert len(memo) == 1


class TestSubAgentGrouping:
    """Sub-agent dispatch groups components by scan root."""

    def test_groups_by_scan_root(self):
        from aibom.agentic.agent import _group_by_top_dir

        comps = [
            AIComponent(
                name="a",
                component_type=AIComponentType.MODEL,
                file_path="/repo1/src/a.py",
                line_number=1,
            ),
            AIComponent(
                name="b",
                component_type=AIComponentType.MODEL,
                file_path="/repo1/src/b.py",
                line_number=2,
            ),
            AIComponent(
                name="c",
                component_type=AIComponentType.MODEL,
                file_path="/repo2/lib/c.py",
                line_number=1,
            ),
        ]
        groups = _group_by_top_dir(comps, scan_paths=["/repo1", "/repo2"])
        assert len(groups) == 2

    def test_single_root_single_group(self):
        from aibom.agentic.agent import _group_by_top_dir

        comps = [
            AIComponent(
                name=f"m{i}",
                component_type=AIComponentType.MODEL,
                file_path=f"/repo/d{i}/f.py",
                line_number=i,
            )
            for i in range(5)
        ]
        groups = _group_by_top_dir(comps, scan_paths=["/repo"])
        assert len(groups) == 1


class TestAgentEvidence:
    """Schema coverage for the :class:`AgentEvidence` model."""

    def test_all_fields_have_defaults(self):
        from aibom.agentic.agent import AgentEvidence

        ev = AgentEvidence()
        assert ev.pattern == "other"
        assert ev.definition_file == ""
        assert ev.definition_start_line == 0
        assert ev.definition_end_line == 0
        assert ev.evidence_snippet == ""
        assert ev.justification == ""

    def test_round_trip_preserves_fields(self):
        from aibom.agentic.agent import AgentEvidence

        ev = AgentEvidence(
            pattern="react_loop",
            definition_file="/repo/src/agent.py",
            definition_start_line=10,
            definition_end_line=42,
            evidence_snippet="while not done:\n    tool = llm.invoke(...)",
            justification="Explicit while loop with LLM and tool dispatch",
        )
        payload = ev.model_dump()
        restored = AgentEvidence.model_validate(payload)
        assert restored == ev

    def test_rejects_unknown_pattern(self):
        from pydantic import ValidationError

        from aibom.agentic.agent import AgentEvidence

        with pytest.raises(ValidationError):
            AgentEvidence(pattern="not_a_real_pattern")

    def test_all_patterns_accepted(self):
        from aibom.agentic.agent import AgentEvidence

        valid_patterns = (
            "framework_agent",
            "react_loop",
            "framework_inheritance",
            "a2a_server",
            "remote_proxy",
            "other",
        )
        for pat in valid_patterns:
            ev = AgentEvidence(pattern=pat)
            assert ev.pattern == pat


class TestAgentEvidenceOnClassifications:
    """``agent_evidence`` is an optional field on each classification schema."""

    def test_enriched_component_agent_evidence_defaults_to_none(self):
        from aibom.agentic.agent import _EnrichedComponent

        ec = _EnrichedComponent(instance_id="x")
        assert ec.agent_evidence is None

    def test_enriched_component_accepts_agent_evidence(self):
        from aibom.agentic.agent import AgentEvidence, _EnrichedComponent

        ev = AgentEvidence(pattern="framework_agent", definition_file="a.py")
        ec = _EnrichedComponent(instance_id="x", agent_evidence=ev)
        assert ec.agent_evidence is not None
        assert ec.agent_evidence.pattern == "framework_agent"

    def test_new_component_accepts_agent_evidence(self):
        from aibom.agentic.agent import AgentEvidence, _NewComponent

        ev = AgentEvidence(pattern="a2a_server", definition_file="a2a.py")
        nc = _NewComponent(
            name="my-agent",
            component_type="agent",
            file_path="a2a.py",
            line_number=1,
            agent_evidence=ev,
        )
        assert nc.agent_evidence is not None
        assert nc.agent_evidence.pattern == "a2a_server"

    def test_reclassify_component_accepts_agent_evidence(self):
        from aibom.agentic.agent import AgentEvidence, _ReclassifyComponent

        ev = AgentEvidence(pattern="react_loop", definition_file="loop.py")
        rc = _ReclassifyComponent(instance_id="y", new_type="agent", agent_evidence=ev)
        assert rc.agent_evidence is not None
        assert rc.agent_evidence.pattern == "react_loop"

    def test_agent_response_full_round_trip_with_evidence(self):
        from aibom.agentic.agent import AgentEvidence, AgentResponse

        ev = AgentEvidence(
            pattern="framework_inheritance",
            definition_file="/repo/worker.py",
            definition_start_line=1,
            definition_end_line=50,
            evidence_snippet="class MyAgent(LangChainAgent):\n    ...",
            justification="Subclasses LangChain agent base class",
        )
        resp = AgentResponse(
            reclassify_components=[
                {
                    "instance_id": "a",
                    "new_type": "agent",
                    "agent_evidence": ev.model_dump(),
                },
            ],
        )
        assert resp.reclassify_components[0].agent_evidence == ev

        payload = resp.model_dump()
        restored = AgentResponse.model_validate(payload)
        assert restored.reclassify_components[0].agent_evidence == ev


class TestA2ARelationshipTypes:
    """The two A2A-specific relationship types exist and round-trip."""

    def test_invokes_a2a_agent_exists(self):
        from aibom.models import RelationshipType

        assert RelationshipType.INVOKES_A2A_AGENT.value == "INVOKES_A2A_AGENT"

    def test_exposes_a2a_agent_exists(self):
        from aibom.models import RelationshipType

        assert RelationshipType.EXPOSES_A2A_AGENT.value == "EXPOSES_A2A_AGENT"

    def test_a2a_types_round_trip_via_string(self):
        from aibom.models import RelationshipType

        assert (
            RelationshipType("INVOKES_A2A_AGENT") is RelationshipType.INVOKES_A2A_AGENT
        )
        assert (
            RelationshipType("EXPOSES_A2A_AGENT") is RelationshipType.EXPOSES_A2A_AGENT
        )


class TestAgenticResultCacheAtomicWrite:
    """``_AgenticResultCache`` must write cache entries atomically so an
    interrupted run never leaves a half-written ``.json`` that a later resume
    reads as a partial/corrupt entry."""

    def test_successful_put_writes_complete_json_only(self, tmp_path: Path):
        from aibom.agentic.agent import _AgenticResultCache

        cache = _AgenticResultCache(cache_dir=tmp_path)
        cache.put("k1", {"a": 1, "b": [1, 2, 3]})

        # The committed entry is complete and parseable JSON.
        files = list(tmp_path.glob("*.json"))
        assert [f.name for f in files] == ["k1.json"]
        assert json.loads(files[0].read_text(encoding="utf-8")) == {
            "a": 1,
            "b": [1, 2, 3],
        }
        # No temp/partial artifacts left behind.
        assert list(tmp_path.glob("*.tmp")) == []
        assert list(tmp_path.glob("*.json.*")) == []

    def test_write_is_atomic_via_temp_then_rename(self, tmp_path: Path, monkeypatch):
        """``put`` must stage to a temp file and atomically rename into place,
        so a crash during the commit never leaves a partial ``key.json``. We
        intercept ``os.replace`` (the atomic commit) to fail; the target file
        must not exist afterwards (only a leftover temp at worst)."""
        import os as _os

        from aibom.agentic.agent import _AgenticResultCache

        cache = _AgenticResultCache(cache_dir=tmp_path)

        calls: dict[str, int] = {"replace": 0}

        def failing_replace(src, dst):
            calls["replace"] += 1
            raise OSError("simulated crash during atomic commit")

        monkeypatch.setattr(_os, "replace", failing_replace)

        # put() swallows OSError today; the contract we assert is that the
        # atomic commit was attempted and no corrupt target file remains.
        cache.put("k1", {"a": 1})

        assert (
            calls["replace"] == 1
        ), "put() must commit via os.replace (atomic temp-file rename)"
        assert not (
            tmp_path / "k1.json"
        ).exists(), "a failed commit must not leave a partial/corrupt k1.json"

    def test_resume_skips_truncated_cache_entry(self, tmp_path: Path):
        from aibom.agentic.agent import _AgenticResultCache

        # Simulate a killed run that left a truncated (invalid) JSON file.
        (tmp_path / "good.json").write_text('{"ok": true}', encoding="utf-8")
        (tmp_path / "partial.json").write_text('{"ok": tr', encoding="utf-8")

        cache = _AgenticResultCache(cache_dir=tmp_path)

        # Good entry loads; truncated entry is skipped (cache miss), not fatal.
        assert cache.get("good") == {"ok": True}
        assert cache.get("partial") is None


class TestApplyBatchFindingsFailOpen:
    """applying a batch's structured output must fail OPEN per
    batch. Any unexpected middleware error degrades only that batch instead of
    propagating up and aborting the whole agentic stage."""

    def _batch(self):
        return [
            AIComponent(
                name="c",
                component_type=AIComponentType.MODEL,
                file_path="a.py",
                line_number=1,
                instance_id="c_a.py_1",
            )
        ]

    def test_degrades_when_middleware_raises(self):
        from aibom.agentic import agent as agent_mod
        from aibom.agentic.agent import _apply_batch_findings

        batch = self._batch()
        mw = MagicMock()
        mw.extract_findings_from_dict.side_effect = AttributeError(
            "'str' object has no attribute 'get'"
        )

        enriched, new_c, new_r, rf, degraded = _apply_batch_findings(
            mw, batch, {"enriched_components": ["stray"]}, batch_num=1
        )

        assert degraded is True
        assert new_c == [] and new_r == [] and rf == []
        assert len(enriched) == 1
        # Degraded with a retryable hint so _collect_failed re-queues it for
        # the retry pass (mirrors batch_timeout / no_usable_output semantics).
        assert enriched[0].agentic_hint == "structured_output_parse_error"
        assert enriched[0].agentic_hint in agent_mod._RETRYABLE_HINTS

    def test_happy_path_passes_through(self):
        from aibom.agentic.agent import _apply_batch_findings

        batch = self._batch()
        mw = MagicMock()
        mw.extract_findings_from_dict.return_value = ([], [], [])
        mw.apply_enrichments_from_dict.return_value = batch

        enriched, new_c, new_r, rf, degraded = _apply_batch_findings(
            mw, batch, {"enriched_components": []}, batch_num=1
        )

        assert degraded is False
        assert enriched == batch
        assert new_c == [] and new_r == [] and rf == []


class TestAccumulateTokenUsage:
    """usage must be recovered from ``response_metadata`` for
    providers that don't populate the standardized ``usage_metadata`` field
    (Bedrock Invoke path; Azure gpt-5.3-codex)."""

    class _Msg:
        """Minimal AIMessage stand-in carrying the usage fields we read."""

        def __init__(self, usage_metadata=None, response_metadata=None):
            self.usage_metadata = usage_metadata
            self.response_metadata = response_metadata

    def _usage(self, *messages):
        from aibom.agentic import agent as agent_mod

        agent_mod._reset_token_usage()
        agent_mod._accumulate_token_usage({"messages": list(messages)})
        u = agent_mod.get_token_usage()
        return (u.prompt_tokens, u.completion_tokens, u.total_tokens)

    def test_usage_metadata_only(self):
        msg = self._Msg(
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            }
        )
        assert self._usage(msg) == (100, 20, 120)

    def test_openai_azure_token_usage_fallback(self):
        # gpt-5.3-codex: no usage_metadata; usage under response_metadata.
        msg = self._Msg(
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 10,
                    "total_tokens": 60,
                }
            }
        )
        assert self._usage(msg) == (50, 10, 60)

    def test_bedrock_response_metadata_usage_fallback(self):
        # ChatBedrock Converse-style usage block.
        msg = self._Msg(
            response_metadata={"usage": {"input_tokens": 200, "output_tokens": 40}}
        )
        # total synthesized when the provider omits it.
        assert self._usage(msg) == (200, 40, 240)

    def test_bedrock_invocation_metrics_fallback(self):
        # ChatBedrock Invoke-path metrics block.
        msg = self._Msg(
            response_metadata={
                "amazon-bedrock-invocationMetrics": {
                    "inputTokenCount": 300,
                    "outputTokenCount": 60,
                }
            }
        )
        assert self._usage(msg) == (300, 60, 360)

    def test_prefers_usage_metadata_no_double_count(self):
        # Both carriers present (LangChain often derives one from the other):
        # count only once.
        msg = self._Msg(
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                }
            },
        )
        assert self._usage(msg) == (100, 20, 120)

    def test_empty_usage_metadata_falls_back(self):
        msg = self._Msg(
            usage_metadata={},
            response_metadata={
                "token_usage": {"prompt_tokens": 7, "completion_tokens": 3}
            },
        )
        assert self._usage(msg) == (7, 3, 10)

    def test_mixed_messages_summed(self):
        m1 = self._Msg(
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            }
        )
        m2 = self._Msg(
            response_metadata={
                "amazon-bedrock-invocationMetrics": {
                    "inputTokenCount": 300,
                    "outputTokenCount": 60,
                }
            }
        )
        assert self._usage(m1, m2) == (400, 80, 480)


class TestAllBatchesFailed:
    """a total agentic failure (e.g. every batch rejected by the
    provider) must be distinguishable from 'ran fine, found nothing to add' so
    the run can report a degraded status instead of 'enrichment complete'."""

    def _comp(self, iid, hint=""):
        return AIComponent(
            name=iid,
            component_type=AIComponentType.MODEL,
            file_path="a.py",
            line_number=1,
            instance_id=iid,
            agentic_hint=hint,
        )

    def test_all_degraded_no_findings_is_failure(self):
        from aibom.agentic.agent import _all_batches_failed

        enriched = [
            self._comp("c1", "batch_timeout"),
            self._comp("c2", "batch_recursion_limit"),
        ]
        assert _all_batches_failed(enriched, [], [], []) is True

    def test_partial_success_is_not_failure(self):
        from aibom.agentic.agent import _all_batches_failed

        enriched = [self._comp("c1", "batch_timeout"), self._comp("c2", "")]
        assert _all_batches_failed(enriched, [], [], []) is False

    def test_all_ok_is_not_failure(self):
        from aibom.agentic.agent import _all_batches_failed

        enriched = [self._comp("c1", ""), self._comp("c2", "")]
        assert _all_batches_failed(enriched, [], [], []) is False

    def test_findings_present_is_not_failure(self):
        from aibom.agentic.agent import _all_batches_failed

        # Even if all inputs degraded, discovering something means the layer
        # produced usable output.
        enriched = [self._comp("c1", "batch_timeout")]
        new = [self._comp("new1", "")]
        assert _all_batches_failed(enriched, new, [], []) is False

    def test_empty_is_not_failure(self):
        from aibom.agentic.agent import _all_batches_failed

        assert _all_batches_failed([], [], [], []) is False

    @patch("aibom.agentic.agent._RETRY_COOLDOWN_S", 0)
    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_run_reports_degraded_when_every_batch_fails(
        self, mock_create, _mb, _mc, caplog
    ):
        """End-to-end: a run where the provider rejects every batch logs a
        DEGRADED warning and NOT 'enrichment complete'."""
        import logging

        from aibom.agentic.agent import run_agentic_enrichment

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("provider rejected request")
        mock_create.return_value = mock_agent

        comp = AIComponent(
            name="x",
            component_type=AIComponentType.AGENT,
            file_path="b.py",
            line_number=1,
        )
        with caplog.at_level(logging.WARNING, logger="aibom.agentic.agent"):
            comps, _, _, _ = run_agentic_enrichment(
                model_string="m",
                deterministic_components=[comp],
                deterministic_relationships=[],
                scan_paths=["/tmp"],
                timeout_s=5,
            )

        assert "DEGRADED" in caplog.text
        assert "Agentic enrichment complete" not in caplog.text


class TestPartialDegradedWarning:
    """When SOME (but not all) components fail enrichment, the run still logs
    'enrichment complete' — but must ALSO warn that N components were left
    degraded, so a raised --agentic-concurrency/--agentic-rate-limit that
    silently drops components does not read as a clean run."""

    def _comp(self, iid, hint=""):
        return AIComponent(
            name=iid,
            component_type=AIComponentType.MODEL,
            file_path="a.py",
            line_number=1,
            instance_id=iid,
            agentic_hint=hint,
        )

    def test_count_degraded_counts_only_hinted(self):
        from aibom.agentic.agent import _count_degraded

        comps = [
            self._comp("c1", "batch_timeout"),
            self._comp("c2", ""),
            self._comp("c3", "retry_failed"),
        ]
        assert _count_degraded(comps) == 2

    def test_count_degraded_zero_when_all_clean(self):
        from aibom.agentic.agent import _count_degraded

        comps = [self._comp("c1", ""), self._comp("c2", "")]
        assert _count_degraded(comps) == 0

    def test_dominant_degraded_hint_returns_most_common(self):
        from aibom.agentic.agent import _dominant_degraded_hint

        comps = [
            self._comp("c1", "batch_timeout"),
            self._comp("c2", "batch_timeout"),
            self._comp("c3", "retry_failed"),
            self._comp("c4", ""),
        ]
        assert _dominant_degraded_hint(comps) == "batch_timeout"

    def test_dominant_degraded_hint_none_when_all_clean(self):
        from aibom.agentic.agent import _dominant_degraded_hint

        assert _dominant_degraded_hint([self._comp("c1", "")]) is None

    @patch("aibom.agentic.agent._RETRY_COOLDOWN_S", 0)
    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_run_warns_on_partial_degradation(self, mock_create, _mb, _mc, caplog):
        """A run that enriches one component but leaves another degraded logs
        BOTH 'enrichment complete' and a degraded-components warning."""
        import logging

        from aibom.agentic.agent import _DEGRADED_LOAD_HINTS, run_agentic_enrichment

        # Batch 1 succeeds (discovers a component so the run is not a total
        # failure); batch 2 raises -> its component is left degraded.
        agent_response = json.dumps(
            {
                "enriched_components": [],
                "new_components": [
                    {
                        "name": "discovered",
                        "component_type": "tool",
                        "file_path": "b.py",
                        "line_number": 2,
                    }
                ],
                "new_relationships": [],
                "risk_findings": [],
            }
        )
        good_msg = MagicMock()
        good_msg.content = agent_response
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [
            {"messages": [good_msg]},
            TimeoutError("batch 2 timed out"),
        ]
        mock_create.return_value = mock_agent

        comps = [
            AIComponent(
                name="ok",
                component_type=AIComponentType.AGENT,
                file_path="b.py",
                line_number=1,
                instance_id="ok",
            ),
            AIComponent(
                name="fails",
                component_type=AIComponentType.AGENT,
                file_path="b.py",
                line_number=3,
                instance_id="fails",
            ),
        ]
        with caplog.at_level(logging.WARNING, logger="aibom.agentic.agent"):
            run_agentic_enrichment(
                model_string="m",
                deterministic_components=comps,
                deterministic_relationships=[],
                scan_paths=["/tmp"],
                batch_size=1,
                timeout_s=5,
                max_retry_seconds=0,
            )

        assert "left degraded" in caplog.text
        # load-related hints belong to the remediation set
        assert "batch_timeout" in _DEGRADED_LOAD_HINTS


class TestDiscoveryWiringIsProviderAgnostic:
    """Bedrock/Anthropic (Opus 4.8) report 0 new components on every
    repo while the OpenAI path finds many. This asserts the discovery path IS
    wired provider-agnostically — ``new_components`` are extracted from the SAME
    tool-call carrier that ``enriched_components`` use, and enrichment demonstrably
    works on Bedrock. So a uniform 0-discovery is a model-behavior signal (Opus
    confirms/prunes but does not propose new components — consistent with the
    benchmark, where every model except GPT-5.5 discovers 0), NOT an aibom
    extraction gap. Definitive root cause requires a live Bedrock-vs-OpenAI probe
    (see the e2e plan); this fixture guards against a real extraction regression.
    """

    class _ToolMsg:
        """AIMessage stand-in for a function-calling (tool_use) response, the
        carrier Anthropic/Bedrock use for structured output."""

        def __init__(self, args):
            self.tool_calls = [{"name": "AgentResponse", "args": args}]
            self.content = ""
            self.additional_kwargs = {}

    def test_discovery_and_enrichment_share_the_same_carrier(self):
        from aibom.agentic.agent import _extract_structured_response
        from aibom.agentic.middleware import AIBOMScannerMiddleware

        existing = [
            AIComponent(
                name="known",
                component_type=AIComponentType.MODEL,
                file_path="a.py",
                line_number=1,
                instance_id="known_a.py_1",
            )
        ]
        # One AgentResponse tool call carrying BOTH an enrichment and a newly
        # discovered component — exactly what Bedrock/Anthropic emit.
        args = {
            "enriched_components": [
                {"instance_id": "known_a.py_1", "updates": {"model_name": "gpt-4o"}}
            ],
            "new_components": [
                {
                    "name": "secret-model",
                    "component_type": "model",
                    "file_path": "b.py",
                    "line_number": 2,
                    "model_name": "gpt-5",
                }
            ],
        }
        data = _extract_structured_response({"messages": [self._ToolMsg(args)]})
        assert data is not None

        mw = AIBOMScannerMiddleware()
        new_comps, _, _ = mw.extract_findings_from_dict(data)
        enriched = mw.apply_enrichments_from_dict(existing, data)

        # Enrichment works on this carrier (proven on Bedrock in the eval) ...
        assert enriched[0].model_name == "gpt-4o"
        # ... and discovery is read from the very same dict — not dropped.
        assert any(c.name == "secret-model" for c in new_comps)


class TestTwoPhaseStructuredOutput:
    """provider-general two-phase decouple. Phase 1 runs the tool
    loop UNFORCED (no response_format -> tool_choice=auto -> natural termination
    on every provider); Phase 2 coerces the transcript into AgentResponse via a
    single tool-less with_structured_output call."""

    class _Msg:
        def __init__(self, usage_metadata=None, content=""):
            self.usage_metadata = usage_metadata
            self.response_metadata = None
            self.content = content

    @pytest.mark.skipif(
        not _HAS_DEEPAGENTS, reason="requires deepagents (agentic extra)"
    )
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("deepagents.create_deep_agent")
    def test_create_agent_defaults_to_unforced_phase1(self, mock_cda, _mb):
        from aibom.agentic.agent import create_aibom_agent

        graph = MagicMock()
        mock_cda.return_value = graph
        bundle = create_aibom_agent("m")

        # Phase 1: no forced structured-output tool.
        assert mock_cda.call_args.kwargs["response_format"] is None
        # Bundle carries the chat model for Phase 2 and flags coercion needed.
        assert bundle.needs_coercion is True
        assert bundle.aibom_chat_model is not None
        # Bundle transparently proxies invoke/ainvoke to the graph.
        bundle.invoke("x")
        graph.invoke.assert_called_once_with("x")

    @pytest.mark.skipif(
        not _HAS_DEEPAGENTS, reason="requires deepagents (agentic extra)"
    )
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("deepagents.create_deep_agent")
    def test_explicit_response_format_not_coerced(self, mock_cda, _mb):
        from aibom.agentic.agent import create_aibom_agent

        mock_cda.return_value = MagicMock()
        sentinel = object()
        bundle = create_aibom_agent("m", model=MagicMock(), response_format=sentinel)
        assert mock_cda.call_args.kwargs["response_format"] is sentinel
        assert bundle.needs_coercion is False

    @pytest.mark.skipif(
        not _HAS_LANGCHAIN, reason="requires langchain_core (agentic extra)"
    )
    def test_coerce_structured_returns_dict_and_counts_tokens(self):
        from aibom.agentic import agent as agent_mod
        from aibom.agentic.agent import AgentResponse, _coerce_structured

        agent_mod._reset_token_usage()
        model = MagicMock()
        raw = self._Msg(
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            }
        )
        parsed = AgentResponse(
            new_components=[{"name": "x", "component_type": "model"}]
        )
        structured = MagicMock()
        structured.invoke.return_value = {
            "raw": raw,
            "parsed": parsed,
            "parsing_error": None,
        }
        model.with_structured_output.return_value = structured
        raw_callback = object()
        callback_factory = MagicMock(return_value=raw_callback)

        data = _coerce_structured(
            model,
            [self._Msg(content="findings")],
            invoke_callback_factory=callback_factory,
        )

        assert isinstance(data, dict)
        assert data["new_components"][0]["name"] == "x"
        # include_raw lets us keep Phase-2 token accounting.
        assert agent_mod.get_token_usage().total_tokens == 12
        # with_structured_output must be tool-less (single coercion call).
        model.with_structured_output.assert_called_once()
        callback_factory.assert_called_once_with()
        structured.invoke.assert_called_once_with(
            ANY,
            config={"callbacks": [raw_callback]},
        )

    @pytest.mark.skipif(
        not _HAS_LANGCHAIN, reason="requires langchain_core (agentic extra)"
    )
    def test_successful_coercion_can_defer_workflow_until_middleware(self):
        from aibom.agentic.agent import AgentResponse, _coerce_structured

        model = MagicMock()
        parsed = AgentResponse(
            remove_components=[{"instance_id": "candidate-1", "reason": "invalid"}]
        )
        structured = MagicMock()
        structured.invoke.return_value = {
            "raw": self._Msg(
                usage_metadata={
                    "input_tokens": 8,
                    "output_tokens": 2,
                    "total_tokens": 10,
                }
            ),
            "parsed": parsed,
            "parsing_error": None,
        }
        model.with_structured_output.return_value = structured
        attempt = MagicMock()

        data = _coerce_structured(
            model,
            [self._Msg(content="findings")],
            telemetry_attempt=attempt,
            defer_successful_workflow=True,
        )

        assert data is not None
        assert data["remove_components"][0]["instance_id"] == "candidate-1"
        attempt.record_llm.assert_called_once()
        attempt.finish.assert_not_called()

    @pytest.mark.skipif(
        not _HAS_LANGCHAIN, reason="requires langchain_core (agentic extra)"
    )
    def test_coerce_structured_none_on_failure(self):
        from aibom.agentic.agent import _coerce_structured

        model = MagicMock()
        model.with_structured_output.side_effect = RuntimeError("no structured out")
        assert _coerce_structured(model, [self._Msg()]) is None

    def test_resolve_batch_data_extractor_first_then_coerce(self):
        from aibom.agentic.agent import _resolve_batch_data

        # Phase-1 unforced agent whose transcript has NO parseable structured
        # output -> falls through to Phase-2 coercion.
        agent = MagicMock()
        agent.needs_coercion = True
        coerced = {"enriched_components": [], "new_components": []}
        callback_factory = MagicMock()
        with patch(
            "aibom.agentic.agent._coerce_structured", return_value=coerced
        ) as mock_coerce:
            data = _resolve_batch_data(
                agent,
                {"messages": [self._Msg(content="prose")]},
                invoke_callback_factory=callback_factory,
            )
        assert data == coerced
        mock_coerce.assert_called_once()
        assert (
            mock_coerce.call_args.kwargs["invoke_callback_factory"] is callback_factory
        )

    def test_resolve_batch_data_skips_coercion_when_structured_present(self):
        from aibom.agentic.agent import _resolve_batch_data

        # A result that already carries a usable structured response (e.g. the
        # ProviderStrategy fallback agent) must NOT trigger a Phase-2 call.
        agent = MagicMock()
        agent.needs_coercion = True
        good = {"enriched_components": [{"instance_id": "a"}], "new_components": []}
        result = {"structured_response": good, "messages": [self._Msg()]}
        with patch("aibom.agentic.agent._coerce_structured") as mock_coerce:
            data = _resolve_batch_data(agent, result)
        assert data == good
        mock_coerce.assert_not_called()


class TestStructuredOutputCapabilityGate:
    """capability gate: ProviderStrategy-capable OpenAI-family models
    keep single-pass native structured output (baseline, full fidelity); every
    other provider/model uses the two-phase decouple."""

    import pytest as _pytest

    @_pytest.mark.parametrize(
        "provider,model_id,inloop",
        [
            ("openai", "gpt-5.5", True),
            ("openai", "gpt-4o", True),
            ("azure_openai", "gpt-5.3-codex", True),
            ("openai", "o3-mini", True),
            (None, "gpt-4o", True),  # LangChain-inferred OpenAI
            ("openai", "zai-org/GLM-5.2-FP8", False),  # vLLM open model
            ("openai", "mistral-7b-instruct", False),  # vLLM open model
            ("bedrock", "us.anthropic.claude-opus-4-8", False),
            ("anthropic", "claude-opus-4-8", False),
            ("google_genai", "gemini-2.5-pro", False),
            ("ollama", "llama3.1", False),
        ],
    )
    def test_inloop_capability(self, provider, model_id, inloop):
        from aibom.agentic.agent import _supports_inloop_structured_output

        assert _supports_inloop_structured_output(provider, model_id) is inloop

    def test_agent_response_format_gpt_is_native(self):
        from aibom.agentic.agent import AgentResponse, _agent_response_format

        assert (
            _agent_response_format("gpt-5.5", {"provider": "openai"}) is AgentResponse
        )

    def test_agent_response_format_bedrock_is_two_phase(self):
        from aibom.agentic.agent import _agent_response_format

        # None -> create_aibom_agent runs unforced (two-phase).
        assert (
            _agent_response_format(
                "us.anthropic.claude-opus-4-8", {"provider": "bedrock"}
            )
            is None
        )


@pytest.mark.skipif(
    not _HAS_LANGCHAIN, reason="requires langchain_core (agentic extra)"
)
class TestRateLimiterConfig:
    """The agentic request rate must be configurable, default 1/sec."""

    def test_defaults_unchanged(self):
        rl = _build_rate_limiter()
        assert rl.requests_per_second == 1.0
        assert rl.max_bucket_size == 10

    def test_accepts_configured_rate_and_bucket(self):
        rl = _build_rate_limiter(requests_per_second=5.0, max_bucket_size=20)
        assert rl.requests_per_second == 5.0
        assert rl.max_bucket_size == 20

    def test_build_model_threads_rate_from_llm_config(self, monkeypatch):
        from aibom.agentic.agent import _build_model

        captured = {}

        def fake_build_chat_model(model_string, **kwargs):
            captured["rate_limiter"] = kwargs.get("rate_limiter")
            return MagicMock()

        monkeypatch.setattr("aibom.llm_factory.build_chat_model", fake_build_chat_model)
        _build_model("gpt-5.5", {"rate_limit_rps": 5.0})
        assert captured["rate_limiter"].requests_per_second == 5.0

    def test_build_model_defaults_to_one_rps_when_unset(self, monkeypatch):
        from aibom.agentic.agent import _build_model

        captured = {}

        def fake_build_chat_model(model_string, **kwargs):
            captured["rate_limiter"] = kwargs.get("rate_limiter")
            return MagicMock()

        monkeypatch.setattr("aibom.llm_factory.build_chat_model", fake_build_chat_model)
        _build_model("gpt-5.5", {})
        assert captured["rate_limiter"].requests_per_second == 1.0


class TestCachedTokenAccounting:
    """Cache-read tokens must be captured so prompt-cache savings
    are measurable (Azure/OpenAI cache automatically for stable prompts)."""

    def _msg(self, usage_metadata=None, response_metadata=None):
        m = MagicMock()
        m.usage_metadata = usage_metadata
        m.response_metadata = response_metadata or {}
        return m

    def test_token_usage_sums_cached_tokens(self):
        a = TokenUsage(prompt_tokens=100, cached_tokens=80)
        b = TokenUsage(prompt_tokens=50, cached_tokens=30)
        a.add(b)
        assert a.cached_tokens == 110

    def test_reads_cache_read_from_usage_metadata(self):
        msg = self._msg(
            usage_metadata={
                "input_tokens": 12000,
                "output_tokens": 40,
                "total_tokens": 12040,
                "input_token_details": {"cache_read": 8000},
            }
        )
        prompt, completion, total, cached = _resolve_message_usage(msg)
        assert prompt == 12000
        assert cached == 8000

    def test_reads_cached_tokens_from_response_metadata(self):
        msg = self._msg(
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 5000,
                    "completion_tokens": 20,
                    "total_tokens": 5020,
                    "prompt_tokens_details": {"cached_tokens": 3000},
                }
            }
        )
        prompt, completion, total, cached = _resolve_message_usage(msg)
        assert prompt == 5000
        assert cached == 3000

    def test_no_cache_fields_yields_zero_cached(self):
        msg = self._msg(
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
            }
        )
        _, _, _, cached = _resolve_message_usage(msg)
        assert cached == 0


class TestNullTolerantDefaults:
    """LLMs routinely emit an explicit ``null`` for a field that has a default
    instead of omitting it. Pydantic rejects ``null`` for a non-Optional
    defaulted field, and because the whole batch is validated against a single
    ``AgentResponse`` schema, one stray ``null`` would discard every component
    in the batch and force a cooldown retry. The structured-output models must
    coerce an explicit ``null`` on a defaulted field back to its default."""

    def test_null_scalar_coerced_to_default(self):
        ev = AgentEvidence(evidence_snippet=None, justification=None)
        assert ev.evidence_snippet == ""
        assert ev.justification == ""

    def test_null_literal_coerced_to_default(self):
        ev = AgentEvidence(pattern=None)
        assert ev.pattern == "other"

    def test_null_int_coerced_to_default(self):
        ev = AgentEvidence(definition_start_line=None)
        assert ev.definition_start_line == 0

    def test_null_list_field_coerced_to_default(self):
        resp = AgentResponse(enriched_components=None, risk_findings=None)
        assert resp.enriched_components == []
        assert resp.risk_findings == []

    def test_null_dict_field_coerced_to_default(self):
        comp = _EnrichedComponent(instance_id="x", updates=None)
        assert comp.updates == {}

    def test_explicitly_optional_field_still_accepts_none(self):
        # ``model_name`` is genuinely ``str | None`` and its default is None,
        # so an explicit null must be preserved, not coerced.
        comp = _EnrichedComponent(instance_id="x", decision_annotation=None)
        assert comp.decision_annotation is None

    def test_provided_value_is_untouched(self):
        finding = _RiskFinding(severity="high", flag="secret")
        assert finding.severity == "high"
        assert finding.flag == "secret"

    def test_one_null_does_not_invalidate_whole_batch(self):
        # A single component carrying a stray ``null`` must not discard the
        # other components in the same batch response.
        resp = AgentResponse.model_validate(
            {
                "risk_findings": [
                    {"flag": "a", "severity": None},
                    {"flag": "b", "severity": "high"},
                ]
            }
        )
        assert len(resp.risk_findings) == 2
        assert resp.risk_findings[0].severity == "info"
        assert resp.risk_findings[1].severity == "high"

    def test_nested_null_in_batch_component_is_tolerated(self):
        resp = AgentResponse.model_validate(
            {
                "enriched_components": [
                    {
                        "instance_id": "x",
                        "agent_evidence": {"evidence_snippet": None},
                    }
                ]
            }
        )
        assert resp.enriched_components[0].agent_evidence.evidence_snippet == ""

    def test_mixed_batch_with_null_yields_all_finding_kinds(self):
        # A realistic full batch: a stray ``null`` on a defaulted field in one
        # entry must not discard the enrichments, new components, relationships,
        # or risk findings that share the same response.
        resp = AgentResponse.model_validate(
            {
                "enriched_components": [
                    {"instance_id": "keep_me", "agent_evidence": None}
                ],
                "new_components": [{"name": "n1", "framework": None}],
                "new_relationships": [
                    {"source_name": "a", "target_name": "b", "source_type": None}
                ],
                "risk_findings": [{"flag": "leak", "severity": None}],
            }
        )
        assert resp.enriched_components[0].instance_id == "keep_me"
        assert resp.new_components[0].framework == ""
        assert resp.new_relationships[0].source_type == ""
        assert resp.risk_findings[0].severity == "info"

    @pytest.mark.skipif(
        not _HAS_LANGCHAIN, reason="requires langchain_core (agentic extra)"
    )
    def test_langchain_parser_tolerates_null_on_defaulted_field(self):
        # The production structured-output path validates the model's raw JSON
        # through langchain's parser (``response_format=AgentResponse``), not a
        # direct ``model_validate`` call. Drive that real entrypoint to prove the
        # before-validator is wired into it and a stray null no longer discards
        # the sibling entries in the batch.
        from langchain_core.output_parsers import PydanticOutputParser

        parser = PydanticOutputParser(pydantic_object=AgentResponse)
        resp = parser.parse(
            '{"risk_findings": [{"flag": "a", "severity": null}, '
            '{"flag": "b", "severity": "high"}]}'
        )
        assert isinstance(resp, AgentResponse)
        assert len(resp.risk_findings) == 2
        assert resp.risk_findings[0].severity == "info"
        assert resp.risk_findings[1].severity == "high"


class TestPromptCachingMiddleware:
    """Explicit Bedrock prompt-caching wiring.

    Anthropic-on-Bedrock does not cache automatically. langchain_aws's built-in
    ``BedrockPromptCachingMiddleware`` places the breakpoint on the LAST message,
    but aibom re-sends an identical ~12k-token system prompt + tool schema with a
    DIFFERENT last message on every batch, so that breakpoint never matches across
    batches (verified live: cached_tokens=0). Tagging the stable SYSTEM block with
    ``cache_control`` instead caches the whole tools+system prefix every batch
    shares (verified live: cross-call cache_read). The gate is strictly by resolved
    provider: only the Bedrock family gets the middleware. Native Anthropic is
    already cached by deepagents' built-in ``AnthropicPromptCachingMiddleware`` (so
    aibom adds nothing there); OpenAI/Azure stay untouched (server-side automatic
    caching).
    """

    @pytest.mark.parametrize(
        "model_string,provider",
        [
            ("gpt-5.5", "openai"),
            ("openai/gpt-5.5", None),
            ("gpt-5.5", "azure_openai"),
            # Native Anthropic — deepagents already injects the Anthropic caching
            # middleware, so aibom must NOT add its own (no duplicate breakpoints).
            ("claude-opus-4-8", "anthropic"),
            ("claude-opus-4-8", None),
            ("gemini-2.5-pro", "google_genai"),
            ("llama3", "ollama"),
        ],
    )
    def test_no_middleware_for_non_bedrock_providers(self, model_string, provider):
        from aibom.agentic.agent import _prompt_caching_middleware

        llm_config = {"provider": provider} if provider else None
        assert _prompt_caching_middleware(None, model_string, llm_config) == []

    @pytest.mark.skipif(
        not _HAS_LANGCHAIN_AGENTS, reason="requires langchain (agentic extra)"
    )
    @pytest.mark.parametrize(
        "model_string,provider",
        [
            ("bedrock/us.anthropic.claude-opus-4-8", None),
            ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "bedrock"),
        ],
    )
    def test_bedrock_gets_single_caching_middleware(self, model_string, provider):
        from langchain.agents.middleware import AgentMiddleware

        from aibom.agentic.agent import _prompt_caching_middleware

        llm_config = {"provider": provider} if provider else None
        mw = _prompt_caching_middleware(None, model_string, llm_config)

        assert len(mw) == 1
        assert isinstance(mw[0], AgentMiddleware)
        assert hasattr(mw[0], "wrap_model_call")

    @pytest.mark.skipif(
        not _HAS_LANGCHAIN_AGENTS, reason="requires langchain (agentic extra)"
    )
    def test_bedrock_gated_by_model_object_when_provider_absent(self):
        # Regression: the tier runners call create_aibom_agent with a pre-built
        # model and a BARE model id (``us.anthropic.…``, no ``bedrock/`` prefix)
        # and WITHOUT llm_config, so the provider is resolvable only from the
        # model object. The gate must detect Bedrock from the built ChatBedrock
        # instance, not just the string/config — otherwise caching silently
        # never engages (observed live as cached_tokens=0).
        from aibom.agentic.agent import _prompt_caching_middleware

        fake_bedrock = type(
            "ChatBedrock", (), {"model_id": "us.anthropic.claude-sonnet-5"}
        )()
        mw = _prompt_caching_middleware(
            fake_bedrock, "us.anthropic.claude-sonnet-5", None
        )
        assert len(mw) == 1

        fake_openai = type("ChatOpenAI", (), {"model_name": "gpt-5.5"})()
        assert _prompt_caching_middleware(fake_openai, "gpt-5.5", None) == []

    @pytest.mark.skipif(
        not _HAS_LANGCHAIN_AWS, reason="requires langchain_aws (llm-aws extra)"
    )
    def test_bedrock_converse_uses_builtin_cachepoint_middleware(self):
        # ChatBedrockConverse (Converse API) expresses caching with a ``cachePoint``
        # block, not the InvokeModel ``cache_control`` shape. The built-in
        # BedrockPromptCachingMiddleware passes the setting via model_settings and
        # ChatBedrockConverse serializes a system+tools cachePoint (surviving
        # deepagents' block stripping), so Converse reuses it — detected both by
        # model class and by the ``bedrock_converse`` provider.
        from langchain_aws.middleware import BedrockPromptCachingMiddleware

        from aibom.agentic.agent import _prompt_caching_middleware

        fake_converse = type(
            "ChatBedrockConverse", (), {"model_id": "us.anthropic.claude-sonnet-5"}
        )()
        by_class = _prompt_caching_middleware(
            fake_converse, "us.anthropic.claude-sonnet-5", None
        )
        assert len(by_class) == 1
        assert isinstance(by_class[0], BedrockPromptCachingMiddleware)

        by_provider = _prompt_caching_middleware(
            None, "bedrock_converse/us.anthropic.claude-sonnet-5", None
        )
        assert len(by_provider) == 1
        assert isinstance(by_provider[0], BedrockPromptCachingMiddleware)

    def _make_request(self, model_id, system_prompt):
        from langchain.agents.middleware import ModelRequest
        from langchain_core.messages import HumanMessage

        model = MagicMock()
        model.model_id = model_id
        return ModelRequest(
            model=model,
            messages=[HumanMessage(content="batch-specific candidates")],
            system_prompt=system_prompt,
        )

    @pytest.mark.skipif(
        not _HAS_LANGCHAIN_AGENTS, reason="requires langchain (agentic extra)"
    )
    def test_tags_system_block_with_cache_control_for_bedrock_claude(self):
        from aibom.agentic.agent import _prompt_caching_middleware

        mw = _prompt_caching_middleware(
            None, "bedrock/us.anthropic.claude-opus-4-8", None
        )[0]
        request = self._make_request("us.anthropic.claude-opus-4-8", "SYSTEM PREFIX")

        seen = {}
        mw.wrap_model_call(request, lambda req: seen.setdefault("req", req))
        got = seen["req"]

        # System prefix rewritten to a content-block list whose (last) block
        # carries an ephemeral cache_control breakpoint — this is what makes the
        # stable tools+system prefix cacheable across batches.
        assert isinstance(got.system_message.content, list)
        last = got.system_message.content[-1]
        assert last["type"] == "text"
        assert last["text"] == "SYSTEM PREFIX"
        assert last["cache_control"]["type"] == "ephemeral"
        # The variable, per-batch messages must be left untouched.
        assert got.messages == request.messages

    @pytest.mark.skipif(
        not _HAS_LANGCHAIN_AGENTS, reason="requires langchain (agentic extra)"
    )
    def test_preserves_system_content_block_boundaries(self):
        # The deep agent delivers the system message as a LIST of content blocks.
        # We must not flatten it (``sm.text`` would collapse boundaries and drop
        # non-text blocks) — copy every block and tag only the LAST text block.
        from langchain.agents.middleware import ModelRequest
        from langchain_core.messages import HumanMessage, SystemMessage

        from aibom.agentic.agent import _prompt_caching_middleware

        mw = _prompt_caching_middleware(
            None, "bedrock/us.anthropic.claude-opus-4-8", None
        )[0]
        model = MagicMock()
        model.model_id = "us.anthropic.claude-opus-4-8"
        request = ModelRequest(
            model=model,
            messages=[HumanMessage(content="candidates")],
            system_message=SystemMessage(
                content=[
                    {"type": "text", "text": "AIBOM SYSTEM"},
                    {"type": "text", "text": "BASE PROMPT"},
                ]
            ),
        )

        seen = {}
        mw.wrap_model_call(request, lambda req: seen.setdefault("req", req))
        content = seen["req"].system_message.content

        # Both blocks preserved (not flattened into one).
        assert [b["text"] for b in content] == ["AIBOM SYSTEM", "BASE PROMPT"]
        # cache_control only on the LAST text block; earlier blocks untouched.
        assert "cache_control" not in content[0]
        assert content[1]["cache_control"]["type"] == "ephemeral"

    @pytest.mark.skipif(
        not _HAS_LANGCHAIN_AGENTS, reason="requires langchain (agentic extra)"
    )
    def test_passthrough_for_unsupported_bedrock_model(self):
        # A non-Claude/Nova Bedrock model (e.g. Llama) must not be poked with a
        # cache_control block it may reject — the system prompt passes through
        # unchanged (still a plain string).
        from aibom.agentic.agent import _prompt_caching_middleware

        llama = "bedrock/meta.llama3-70b-instruct-v1:0"
        mw = _prompt_caching_middleware(None, llama, None)[0]
        request = self._make_request("meta.llama3-70b-instruct-v1:0", "SYSTEM PREFIX")

        seen = {}
        mw.wrap_model_call(request, lambda req: seen.setdefault("req", req))
        assert seen["req"].system_message.content == "SYSTEM PREFIX"

    @pytest.mark.skipif(
        not _HAS_LANGCHAIN_AGENTS, reason="requires langchain (agentic extra)"
    )
    def test_passthrough_when_no_system_message(self):
        from langchain.agents.middleware import ModelRequest
        from langchain_core.messages import HumanMessage

        from aibom.agentic.agent import _prompt_caching_middleware

        mw = _prompt_caching_middleware(
            None, "bedrock/us.anthropic.claude-opus-4-8", None
        )[0]
        model = MagicMock()
        model.model_id = "us.anthropic.claude-opus-4-8"
        request = ModelRequest(
            model=model, messages=[HumanMessage(content="x")], system_prompt=None
        )

        seen = {}
        mw.wrap_model_call(request, lambda req: seen.setdefault("req", req))
        assert seen["req"].system_message is None

    def test_bedrock_graceful_when_langchain_missing(self):
        # If the agentic extra is absent, the Bedrock path must degrade to a no-op
        # (empty list), never raise — a scan without caching still works.
        import sys

        from aibom.agentic.agent import _prompt_caching_middleware

        with patch.dict(sys.modules, {"langchain.agents.middleware": None}):
            assert (
                _prompt_caching_middleware(
                    None, "bedrock/us.anthropic.claude-opus-4-8", None
                )
                == []
            )

    @pytest.mark.skipif(
        not (_HAS_DEEPAGENTS and _HAS_LANGCHAIN_AGENTS),
        reason="requires deepagents + langchain (agentic extra)",
    )
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("deepagents.create_deep_agent")
    def test_create_aibom_agent_forwards_caching_middleware(self, mock_cda, _mb):
        from langchain.agents.middleware import AgentMiddleware

        from aibom.agentic.agent import create_aibom_agent

        mock_cda.return_value = MagicMock()

        create_aibom_agent(
            "bedrock/us.anthropic.claude-opus-4-8",
            llm_config={"provider": "bedrock"},
        )
        bedrock_mw = mock_cda.call_args.kwargs["middleware"]
        assert len(bedrock_mw) == 1
        assert isinstance(bedrock_mw[0], AgentMiddleware)

        mock_cda.reset_mock()
        create_aibom_agent("gpt-5.5", llm_config={"provider": "openai"})
        assert mock_cda.call_args.kwargs["middleware"] == []


def test_raw_decision_counts_ignore_annotation_only_enrichment() -> None:
    from aibom.agentic.agent import _raw_decision_counts

    counts = _raw_decision_counts(
        {
            "enriched_components": [
                {
                    "instance_id": "candidate-1",
                    "updates": {},
                    "decision_annotation": {"decision": "keep"},
                },
                {
                    "instance_id": "candidate-2",
                    "updates": {"framework": "langgraph"},
                },
            ]
        }
    )

    assert counts["enriched"] == 1
