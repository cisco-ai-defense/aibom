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

"""Tests for :mod:`aibom.agentic.evidence_injection` and the agent-
evidence fields that :func:`_component_to_summary` wires into the LLM
prompt.

These tests validate the end-to-end path:

1. :func:`build_dossier_index` discovers candidate Python files from a
   list of deterministic :class:`AIComponent` entries, parses them with
   :func:`aibom.cst_parser.parse_source_code`, and produces a keyed
   index of :class:`AgentEvidenceDossier` objects.
2. :func:`lookup_dossier` returns the dossier whose class span contains
   the component's source line.
3. :func:`_component_to_summary` injects both the truncated class body
   and the structured dossier into the prompt summary when — and only
   when — a component is an ENRICH target **and** the dossier index has
   a matching entry.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from aibom.agentic import evidence_injection
from aibom.agentic.agent import (
    _MAX_CLASS_BODY_CHARS,
    _build_context_message,
    _component_to_summary,
)
from aibom.agentic.evidence_injection import (
    _MAX_PY_FILE_SIZE_BYTES,
    build_dossier_index,
    lookup_dossier,
)
from aibom.models import AIComponent
from aibom.models.enums import AIComponentType
from aibom.scanners.agent_evidence_builder import AgentEvidenceDossier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_AGENT_SOURCE = textwrap.dedent(
    """\
    from langchain.agents import AgentExecutor


    class MyAgent(AgentExecutor):
        def run(self, query: str) -> str:
            while self.should_continue():
                plan = self.plan(query)
                action = self.pick_tool(plan)
                observation = self.execute(action)
                self.observations.append(observation)
            return self.finalize()
    """
)


_TOOL_SOURCE = textwrap.dedent(
    """\
    def add(a: int, b: int) -> int:
        return a + b
    """
)


def _write_py(tmp_path: Path, name: str, body: str) -> Path:
    """Materialize *body* at ``tmp_path/name`` and return the path."""
    file = tmp_path / name
    file.write_text(body)
    return file


def _make_component(
    *,
    name: str,
    component_type: AIComponentType,
    file_path: Path | str | None = None,
    line_number: int | None = None,
) -> AIComponent:
    kwargs: dict = {
        "name": name,
        "component_type": component_type,
    }
    if file_path is not None:
        kwargs["file_path"] = str(file_path)
    if line_number is not None:
        kwargs["line_number"] = line_number
    return AIComponent(**kwargs)


# ---------------------------------------------------------------------------
# build_dossier_index / lookup_dossier
# ---------------------------------------------------------------------------


class TestBuildDossierIndex:
    def test_empty_component_list_returns_empty_index(self):
        assert build_dossier_index([]) == {}

    def test_non_candidate_types_are_skipped(self, tmp_path):
        py = _write_py(tmp_path, "agent.py", _AGENT_SOURCE)

        comps = [
            _make_component(
                name="MyAgent",
                component_type=AIComponentType.MODEL,
                file_path=py,
                line_number=4,
            ),
            _make_component(
                name="MyAgent",
                component_type=AIComponentType.TOOL,
                file_path=py,
                line_number=4,
            ),
        ]

        assert build_dossier_index(comps) == {}

    def test_agent_candidate_produces_dossier_with_class_body(self, tmp_path):
        py = _write_py(tmp_path, "agent.py", _AGENT_SOURCE)

        comp = _make_component(
            name="MyAgent",
            component_type=AIComponentType.AGENT,
            file_path=py,
            line_number=4,
        )

        index = build_dossier_index([comp])

        assert len(index) == 1
        (key_file, key_start_line), dossier = next(iter(index.items()))
        assert key_file == str(py)
        assert key_start_line == 4
        assert dossier.class_name == "MyAgent"
        assert dossier.class_start_line == 4
        assert dossier.class_end_line >= 11
        assert "class MyAgent(AgentExecutor):" in dossier.class_body_source

    def test_mcp_and_agent_proxy_candidates_are_parsed(self, tmp_path):
        py = _write_py(tmp_path, "mcp_client.py", _AGENT_SOURCE)

        comps = [
            _make_component(
                name="MyAgent",
                component_type=AIComponentType.MCP_CLIENT,
                file_path=py,
                line_number=4,
            ),
            _make_component(
                name="MyAgent",
                component_type=AIComponentType.AGENT_PROXY,
                file_path=py,
                line_number=4,
            ),
            _make_component(
                name="MyAgent",
                component_type=AIComponentType.MCP_SERVER,
                file_path=py,
                line_number=4,
            ),
        ]

        index = build_dossier_index(comps)
        assert len(index) == 1

    def test_files_with_no_classes_produce_no_dossiers(self, tmp_path):
        py = _write_py(tmp_path, "tools.py", _TOOL_SOURCE)
        comp = _make_component(
            name="add",
            component_type=AIComponentType.AGENT,
            file_path=py,
            line_number=1,
        )

        assert build_dossier_index([comp]) == {}

    def test_missing_file_is_silently_skipped(self, tmp_path):
        comp = _make_component(
            name="Phantom",
            component_type=AIComponentType.AGENT,
            file_path=tmp_path / "does_not_exist.py",
            line_number=1,
        )
        assert build_dossier_index([comp]) == {}

    def test_non_python_extension_is_skipped(self, tmp_path):
        not_py = _write_py(tmp_path, "config.yaml", "key: value\n")
        comp = _make_component(
            name="Phantom",
            component_type=AIComponentType.AGENT,
            file_path=not_py,
            line_number=1,
        )
        assert build_dossier_index([comp]) == {}

    def test_oversized_file_is_skipped(self, tmp_path, monkeypatch):
        py = _write_py(tmp_path, "agent.py", _AGENT_SOURCE)
        comp = _make_component(
            name="MyAgent",
            component_type=AIComponentType.AGENT,
            file_path=py,
            line_number=4,
        )

        monkeypatch.setattr(evidence_injection, "_MAX_PY_FILE_SIZE_BYTES", 10)

        assert build_dossier_index([comp]) == {}

    def test_same_file_is_parsed_only_once(self, tmp_path, monkeypatch):
        py = _write_py(tmp_path, "agent.py", _AGENT_SOURCE)

        calls: list[str] = []
        real_parse = evidence_injection.parse_source_code

        def _spy(file_path: str, source: str):
            calls.append(file_path)
            return real_parse(file_path, source)

        monkeypatch.setattr(evidence_injection, "parse_source_code", _spy)

        comps = [
            _make_component(
                name=f"MyAgent#{i}",
                component_type=AIComponentType.AGENT,
                file_path=py,
                line_number=4 + i,
            )
            for i in range(5)
        ]

        build_dossier_index(comps)
        assert calls.count(str(py)) == 1


class TestLookupDossier:
    def _index_with_agent(self, tmp_path: Path) -> tuple[Path, dict]:
        py = _write_py(tmp_path, "agent.py", _AGENT_SOURCE)
        comp = _make_component(
            name="MyAgent",
            component_type=AIComponentType.AGENT,
            file_path=py,
            line_number=4,
        )
        return py, build_dossier_index([comp])

    def test_returns_none_for_empty_index(self):
        comp = _make_component(
            name="x",
            component_type=AIComponentType.AGENT,
            file_path="a.py",
            line_number=1,
        )
        assert lookup_dossier(comp, None) is None
        assert lookup_dossier(comp, {}) is None

    def test_returns_none_when_component_missing_file_or_line(self, tmp_path):
        _, index = self._index_with_agent(tmp_path)

        no_file = _make_component(
            name="MyAgent",
            component_type=AIComponentType.AGENT,
            line_number=4,
        )
        no_line = _make_component(
            name="MyAgent",
            component_type=AIComponentType.AGENT,
            file_path="some.py",
        )

        assert lookup_dossier(no_file, index) is None
        assert lookup_dossier(no_line, index) is None

    def test_line_inside_class_span_matches(self, tmp_path):
        py, index = self._index_with_agent(tmp_path)

        for line in (4, 5, 6, 10, 11):
            comp = _make_component(
                name="MyAgent",
                component_type=AIComponentType.AGENT,
                file_path=py,
                line_number=line,
            )
            dossier = lookup_dossier(comp, index)
            assert isinstance(dossier, AgentEvidenceDossier), (
                f"Expected dossier for line {line}, got None"
            )
            assert dossier.class_name == "MyAgent"

    def test_line_outside_class_span_returns_none(self, tmp_path):
        py, index = self._index_with_agent(tmp_path)
        comp = _make_component(
            name="Unrelated",
            component_type=AIComponentType.AGENT,
            file_path=py,
            line_number=1,
        )
        assert lookup_dossier(comp, index) is None

    def test_different_file_returns_none(self, tmp_path):
        _, index = self._index_with_agent(tmp_path)
        other_py = _write_py(tmp_path, "other.py", _AGENT_SOURCE)
        comp = _make_component(
            name="MyAgent",
            component_type=AIComponentType.AGENT,
            file_path=other_py,
            line_number=4,
        )
        assert lookup_dossier(comp, index) is None


# ---------------------------------------------------------------------------
# _component_to_summary — dossier injection
# ---------------------------------------------------------------------------


class TestComponentToSummaryDossierInjection:
    def test_no_injection_when_index_is_none(self, tmp_path):
        py = _write_py(tmp_path, "agent.py", _AGENT_SOURCE)
        comp = _make_component(
            name="MyAgent",
            component_type=AIComponentType.AGENT,
            file_path=py,
            line_number=4,
        )

        entry = _component_to_summary(
            comp, enrich_target=True, dossier_index=None
        )
        assert "class_body_source" not in entry
        assert "agent_evidence_dossier" not in entry

    def test_no_injection_when_not_enrich_target(self, tmp_path):
        py = _write_py(tmp_path, "agent.py", _AGENT_SOURCE)
        comp = _make_component(
            name="MyAgent",
            component_type=AIComponentType.AGENT,
            file_path=py,
            line_number=4,
        )
        index = build_dossier_index([comp])
        assert index  # sanity: index is populated

        entry = _component_to_summary(
            comp, enrich_target=False, dossier_index=index
        )
        assert "class_body_source" not in entry
        assert "agent_evidence_dossier" not in entry

    def test_injection_happens_for_enrich_target_with_matching_dossier(
        self, tmp_path
    ):
        py = _write_py(tmp_path, "agent.py", _AGENT_SOURCE)
        comp = _make_component(
            name="MyAgent",
            component_type=AIComponentType.AGENT,
            file_path=py,
            line_number=5,
        )
        index = build_dossier_index([comp])

        entry = _component_to_summary(
            comp, enrich_target=True, dossier_index=index
        )

        assert "class_body_source" in entry
        assert "class MyAgent(AgentExecutor):" in entry["class_body_source"]
        assert "class_body_truncated" not in entry

        dossier_payload = entry["agent_evidence_dossier"]
        assert dossier_payload["class_name"] == "MyAgent"
        assert dossier_payload["file_path"] == str(py)
        assert dossier_payload["class_start_line"] == 4
        for key in (
            "framework_matches",
            "protocol_matches",
            "react_loop_matches",
            "anti_pattern_matches",
            "has_direct_agent_evidence",
        ):
            assert key in dossier_payload

    def test_no_injection_when_component_outside_any_dossier_span(
        self, tmp_path
    ):
        agent_py = _write_py(tmp_path, "agent.py", _AGENT_SOURCE)
        helper_comp = _make_component(
            name="MyAgent",
            component_type=AIComponentType.AGENT,
            file_path=agent_py,
            line_number=4,
        )
        index = build_dossier_index([helper_comp])

        other_comp = _make_component(
            name="Unrelated",
            component_type=AIComponentType.AGENT,
            file_path=agent_py,
            line_number=1,
        )

        entry = _component_to_summary(
            other_comp, enrich_target=True, dossier_index=index
        )
        assert "class_body_source" not in entry
        assert "agent_evidence_dossier" not in entry

    def test_large_class_body_is_truncated(self, tmp_path, monkeypatch):
        huge_body_lines = [
            f"        self.log_{i} = 'x' * 200"
            for i in range(200)
        ]
        huge_source = (
            "from langchain.agents import AgentExecutor\n"
            "\n"
            "\n"
            "class HugeAgent(AgentExecutor):\n"
            "    def __init__(self):\n"
            + "\n".join(huge_body_lines)
            + "\n"
        )
        py = _write_py(tmp_path, "huge.py", huge_source)
        comp = _make_component(
            name="HugeAgent",
            component_type=AIComponentType.AGENT,
            file_path=py,
            line_number=4,
        )

        # Lower the cap so the fixture stays small while still proving the
        # truncation branch runs. We patch via the public constant in
        # aibom.agentic.agent to ensure we're exercising the same guard
        # the production code path uses.
        monkeypatch.setattr(
            "aibom.agentic.agent._MAX_CLASS_BODY_CHARS", 256
        )

        index = build_dossier_index([comp])
        assert index, "precondition: dossier must have been built"
        dossier = next(iter(index.values()))
        assert len(dossier.class_body_source) > 256, (
            f"precondition: class body must exceed patched cap; "
            f"got {len(dossier.class_body_source)} bytes"
        )

        entry = _component_to_summary(
            comp, enrich_target=True, dossier_index=index
        )

        assert "class_body_source" in entry, entry
        assert entry.get("class_body_truncated") is True
        assert len(entry["class_body_source"]) == 256
        assert "class HugeAgent(AgentExecutor):" in entry["class_body_source"]

    def test_module_constant_is_sane(self):
        assert _MAX_CLASS_BODY_CHARS > 0
        assert _MAX_PY_FILE_SIZE_BYTES > 0


# ---------------------------------------------------------------------------
# End-to-end: _build_context_message carries the dossier into the JSON prompt
# ---------------------------------------------------------------------------


class TestBuildContextMessageWithDossier:
    def test_enrich_targets_receive_dossier_payload_in_prompt(self, tmp_path):
        py = _write_py(tmp_path, "agent.py", _AGENT_SOURCE)
        enrich = _make_component(
            name="MyAgent",
            component_type=AIComponentType.AGENT,
            file_path=py,
            line_number=4,
        )
        non_enrich = _make_component(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path=py,
            line_number=1,
        )

        index = build_dossier_index([enrich, non_enrich])

        msg = _build_context_message(
            [enrich],
            [],
            [str(tmp_path)],
            all_components=[enrich, non_enrich],
            dossier_index=index,
        )

        json_start = msg.index("```json\n") + 8
        json_end = msg.index("\n```", json_start)
        data = json.loads(msg[json_start:json_end])

        assert len(data["enrich_these"]) == 1
        enriched = data["enrich_these"][0]
        assert enriched["ENRICH"] is True
        assert "class_body_source" in enriched
        assert "class MyAgent(AgentExecutor):" in enriched["class_body_source"]
        assert enriched["agent_evidence_dossier"]["class_name"] == "MyAgent"

        assert len(data["other_detected_components"]) == 1
        other = data["other_detected_components"][0]
        assert "class_body_source" not in other
        assert "agent_evidence_dossier" not in other

    def test_missing_dossier_index_does_not_break_prompt(self, tmp_path):
        py = _write_py(tmp_path, "agent.py", _AGENT_SOURCE)
        comp = _make_component(
            name="MyAgent",
            component_type=AIComponentType.AGENT,
            file_path=py,
            line_number=4,
        )

        msg = _build_context_message(
            [comp], [], [str(tmp_path)], dossier_index=None
        )

        json_start = msg.index("```json\n") + 8
        json_end = msg.index("\n```", json_start)
        data = json.loads(msg[json_start:json_end])

        assert len(data["enrich_these"]) == 1
        enriched = data["enrich_these"][0]
        assert "class_body_source" not in enriched
        assert "agent_evidence_dossier" not in enriched
