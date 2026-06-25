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

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aibom.agentic.agent import _build_context_message, _extract_structured_response
from aibom.models import (
    AIComponent,
    AIComponentType,
    ComponentRelationship,
    DecisionAnnotation,
    EvidenceLocation,
)


@pytest.fixture(autouse=True)
def _isolate_agentic_cache():
    """Prevent on-disk agentic cache from leaking between tests."""
    with patch("aibom.agentic.agent._default_agentic_cache_dir", return_value=None):
        yield


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
    def test_passes_generous_default_max_tokens(self, mock_build, _mock_rl):
        from aibom.agentic.agent import _DEFAULT_AGENTIC_MAX_TOKENS, _build_model

        _build_model("some-model", {"provider": "openai"})
        _, kwargs = mock_build.call_args
        assert kwargs.get("max_tokens") == _DEFAULT_AGENTIC_MAX_TOKENS

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
        comps, _rels, _flags, _usage = run_agentic_enrichment(
            model_string="bad-model",
            deterministic_components=[comp],
            deterministic_relationships=[],
            scan_paths=["/tmp"],
        )
        assert len(comps) == 1
        # Recovered from the exception's ai_message -> not degraded.
        assert not comps[0].agentic_hint


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

        mock_cache_cls.assert_called_once_with(cache_dir, fallback_dirs=[])
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

        with patch(
            "aibom.agentic.agent._default_agentic_cache_dir",
            return_value=tmp_path / "agentic-cache",
        ):
            fresh, _, _, _ = run_agentic_enrichment(
                model_string="test-model",
                deterministic_components=det_comps,
                deterministic_relationships=[],
                scan_paths=["/tmp"],
            )
            cached, _, _, _ = run_agentic_enrichment(
                model_string="test-model",
                deterministic_components=det_comps,
                deterministic_relationships=[],
                scan_paths=["/tmp"],
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
        cache_dir = tmp_path / "agentic-cache"

        fresh_components, fresh_rels, fresh_flags, _ = run_agentic_enrichment(
            model_string="test-model",
            deterministic_components=det_comps_first,
            deterministic_relationships=[],
            scan_paths=["/tmp"],
            cache_dir=cache_dir,
        )
        cached_components, cached_rels, cached_flags, _ = run_agentic_enrichment(
            model_string="test-model",
            deterministic_components=det_comps_second,
            deterministic_relationships=[],
            scan_paths=["/tmp"],
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
            update={"heuristic_confidence": 0.95, "needs_agentic": False}
        )

        memo.record(before, after)
        verdict = memo.lookup(before)
        assert verdict is not None
        assert verdict["action"] == "keep"
        assert verdict["heuristic_confidence"] == 0.95

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
