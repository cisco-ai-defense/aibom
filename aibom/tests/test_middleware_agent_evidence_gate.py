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

"""Tests for the agent-evidence verification gate in
:class:`aibom.agentic.middleware.AIBOMScannerMiddleware`.

The gate rejects agent / agent-proxy verdicts whose citations cannot be
re-verified against the on-disk source — catching LLM hallucinations.
Three insertion points are exercised:

1. New components via ``new_components``.
2. Reclassifications via ``reclassify_components``.
3. Type-changing enrichments via ``enriched_components`` (where the
   ``component_type`` update is stripped while other updates survive).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from aibom.agentic.middleware import (
    AIBOMScannerMiddleware,
    _normalize_ws,
    _verify_agent_evidence,
)
from aibom.models import AIComponent, AIComponentType


# ---------------------------------------------------------------------------
# Fixtures
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


@pytest.fixture()
def agent_py(tmp_path: Path) -> Path:
    """Write a minimal, on-disk agent fixture and return its path."""
    path = tmp_path / "agent.py"
    path.write_text(_AGENT_SOURCE)
    return path


def _valid_evidence(agent_py: Path) -> dict:
    """Return an ``agent_evidence`` dict that matches *agent_py*."""
    return {
        "pattern": "framework_agent",
        "definition_file": str(agent_py),
        "definition_start_line": 4,
        "definition_end_line": 11,
        "evidence_snippet": "class MyAgent(AgentExecutor):",
        "justification": "Inherits from LangChain AgentExecutor.",
    }


# ---------------------------------------------------------------------------
# _verify_agent_evidence: unit semantics
# ---------------------------------------------------------------------------


class TestVerifyAgentEvidence:
    def test_valid_evidence_passes(self, agent_py: Path) -> None:
        ok, reason = _verify_agent_evidence(
            _valid_evidence(agent_py),
            allowed_roots=[str(agent_py.parent)],
        )
        assert ok
        assert reason == ""

    @pytest.mark.parametrize("bad", [None, "", 0, [], {}])
    def test_missing_or_empty_evidence_fails(self, bad) -> None:
        ok, reason = _verify_agent_evidence(bad, allowed_roots=[])
        assert not ok
        assert reason == "missing agent_evidence"

    def test_non_dict_evidence_fails(self) -> None:
        ok, reason = _verify_agent_evidence("not a dict", allowed_roots=[])
        assert not ok
        assert reason == "agent_evidence is not an object"

    def test_invalid_pattern_fails(self, agent_py: Path) -> None:
        ev = _valid_evidence(agent_py)
        ev["pattern"] = "definitely_not_a_real_pattern"
        ok, reason = _verify_agent_evidence(
            ev, allowed_roots=[str(agent_py.parent)]
        )
        assert not ok
        assert reason.startswith("invalid pattern")

    def test_empty_definition_file_fails(self, agent_py: Path) -> None:
        ev = _valid_evidence(agent_py)
        ev["definition_file"] = ""
        ok, reason = _verify_agent_evidence(
            ev, allowed_roots=[str(agent_py.parent)]
        )
        assert not ok
        assert reason == "empty definition_file"

    def test_file_outside_allowed_roots_fails(
        self, agent_py: Path, tmp_path: Path
    ) -> None:
        forbidden_root = tmp_path / "forbidden"
        forbidden_root.mkdir()
        ev = _valid_evidence(agent_py)
        ok, reason = _verify_agent_evidence(
            ev, allowed_roots=[str(forbidden_root)]
        )
        assert not ok
        assert reason == "file outside allowed roots"

    def test_file_not_found_fails(self, tmp_path: Path) -> None:
        ev = {
            "pattern": "framework_agent",
            "definition_file": str(tmp_path / "phantom.py"),
            "definition_start_line": 1,
            "definition_end_line": 2,
            "evidence_snippet": "class X",
        }
        ok, reason = _verify_agent_evidence(ev, allowed_roots=[str(tmp_path)])
        assert not ok
        assert reason == "file not found"

    @pytest.mark.parametrize(
        "start,end",
        [
            (0, 2),
            (-1, 2),
            (2, 1),
            (1, 9999),
        ],
    )
    def test_stale_numeric_line_range_is_repaired(
        self, agent_py: Path, start, end
    ) -> None:
        ev = _valid_evidence(agent_py)
        ev["definition_start_line"] = start
        ev["definition_end_line"] = end
        ok, reason = _verify_agent_evidence(
            ev, allowed_roots=[str(agent_py.parent)]
        )
        assert ok
        assert reason == ""
        assert ev["definition_start_line"] == 4
        assert ev["definition_end_line"] == 4

    @pytest.mark.parametrize("start,end", [("1", 2), (1, "2")])
    def test_non_integer_line_range_fails(
        self, agent_py: Path, start, end
    ) -> None:
        ev = _valid_evidence(agent_py)
        ev["definition_start_line"] = start
        ev["definition_end_line"] = end
        ok, reason = _verify_agent_evidence(
            ev, allowed_roots=[str(agent_py.parent)]
        )
        assert not ok
        assert reason == "invalid line range"

    def test_missing_snippet_fails(self, agent_py: Path) -> None:
        ev = _valid_evidence(agent_py)
        ev["evidence_snippet"] = "    \n\t"
        ok, reason = _verify_agent_evidence(
            ev, allowed_roots=[str(agent_py.parent)]
        )
        assert not ok
        assert reason == "missing evidence_snippet"

    def test_snippet_not_in_range_fails(self, agent_py: Path) -> None:
        ev = _valid_evidence(agent_py)
        ev["evidence_snippet"] = "class ThisClassDoesNotExist(Agent):"
        ok, reason = _verify_agent_evidence(
            ev, allowed_roots=[str(agent_py.parent)]
        )
        assert not ok
        assert reason == "snippet not found in cited file"

    def test_unique_snippet_outside_cited_range_is_repaired(
        self, agent_py: Path
    ) -> None:
        ev = _valid_evidence(agent_py)
        ev["definition_start_line"] = 1
        ev["definition_end_line"] = 2
        ev["evidence_snippet"] = "class MyAgent(AgentExecutor):"
        ok, reason = _verify_agent_evidence(
            ev, allowed_roots=[str(agent_py.parent)]
        )
        assert ok
        assert reason == ""
        assert ev["definition_start_line"] == 4
        assert ev["definition_end_line"] == 4

    def test_ambiguous_snippet_outside_range_is_rejected(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "agents.py"
        source.write_text(
            "first = Agent()\nsecond = Agent()\n",
            encoding="utf-8",
        )
        ev = {
            "pattern": "framework_agent",
            "definition_file": str(source),
            "definition_start_line": 99,
            "definition_end_line": 99,
            "evidence_snippet": "Agent()",
        }

        ok, reason = _verify_agent_evidence(
            ev, allowed_roots=[str(tmp_path)]
        )

        assert not ok
        assert reason == "snippet match is ambiguous in cited file"

    def test_whitespace_tolerant_match(self, agent_py: Path) -> None:
        ev = _valid_evidence(agent_py)
        ev["evidence_snippet"] = (
            "class    MyAgent(AgentExecutor):\n"
            "\t\tdef run(self, query: str) -> str:"
        )
        ok, reason = _verify_agent_evidence(
            ev, allowed_roots=[str(agent_py.parent)]
        )
        assert ok
        assert reason == ""

    def test_empty_allowed_roots_skips_root_check(self, agent_py: Path) -> None:
        ok, reason = _verify_agent_evidence(
            _valid_evidence(agent_py), allowed_roots=[]
        )
        assert ok
        assert reason == ""

    def test_normalize_ws_collapses_runs(self) -> None:
        assert _normalize_ws("a\t\tb\n\nc  d") == "a b c d"
        assert _normalize_ws("   leading") == "leading"
        assert _normalize_ws("trailing   ") == "trailing"


# ---------------------------------------------------------------------------
# Integration: new_components path
# ---------------------------------------------------------------------------


class TestGateOnNewComponents:
    def test_new_agent_with_valid_evidence_is_kept(self, agent_py: Path) -> None:
        mw = AIBOMScannerMiddleware(allowed_roots=[str(agent_py.parent)])
        evidence = _valid_evidence(agent_py)
        evidence["definition_start_line"] = 1
        evidence["definition_end_line"] = 2
        data = {
            "new_components": [
                {
                    "name": "MyAgent",
                    "component_type": "agent",
                    "file_path": str(agent_py),
                    "line_number": 4,
                    "agent_evidence": evidence,
                }
            ],
        }
        components, _rels, _flags = mw.extract_findings_from_dict(data)
        assert [c.name for c in components] == ["MyAgent"]
        assert components[0].component_type == AIComponentType.AGENT
        stored = components[0].metadata["agent_evidence"]
        assert stored["definition_start_line"] == 4
        assert stored["definition_end_line"] == 4
        assert stored["definition_file"] == str(agent_py)
        assert "evidence_snippet" not in stored

    @pytest.mark.parametrize("comp_type", ["agent", "agent_proxy"])
    def test_new_agent_without_evidence_is_rejected(
        self, agent_py: Path, comp_type: str
    ) -> None:
        mw = AIBOMScannerMiddleware(allowed_roots=[str(agent_py.parent)])
        data = {
            "new_components": [
                {
                    "name": "Ghost",
                    "component_type": comp_type,
                    "file_path": str(agent_py),
                    "line_number": 4,
                }
            ],
        }
        components, _rels, _flags = mw.extract_findings_from_dict(data)
        assert components == []

    def test_new_agent_with_hallucinated_snippet_is_rejected(
        self, agent_py: Path
    ) -> None:
        mw = AIBOMScannerMiddleware(allowed_roots=[str(agent_py.parent)])
        ev = _valid_evidence(agent_py)
        ev["evidence_snippet"] = "class HallucinatedAgent(SomethingMadeUp):"
        data = {
            "new_components": [
                {
                    "name": "Ghost",
                    "component_type": "agent",
                    "agent_evidence": ev,
                }
            ],
        }
        components, _rels, _flags = mw.extract_findings_from_dict(data)
        assert components == []

    def test_new_non_agent_components_bypass_the_gate(
        self, agent_py: Path
    ) -> None:
        """Models / tools / MCP servers do not require agent_evidence."""
        mw = AIBOMScannerMiddleware(allowed_roots=[str(agent_py.parent)])
        data = {
            "new_components": [
                {
                    "name": "anthropic-claude-sonnet-4",
                    "component_type": "model",
                    "file_path": str(agent_py),
                    "line_number": 4,
                },
                {
                    "name": "mcp_fs_tool",
                    "component_type": "tool",
                    "file_path": str(agent_py),
                    "line_number": 4,
                },
            ],
        }
        components, _rels, _flags = mw.extract_findings_from_dict(data)
        names = sorted(c.name for c in components)
        assert names == ["anthropic-claude-sonnet-4", "mcp_fs_tool"]


# ---------------------------------------------------------------------------
# Integration: reclassify_components path
# ---------------------------------------------------------------------------


def _existing(
    name: str, comp_type: AIComponentType, *, file_path: str, line: int = 1
) -> AIComponent:
    return AIComponent(
        name=name,
        component_type=comp_type,
        file_path=file_path,
        line_number=line,
    )


class TestGateOnReclassify:
    def test_reclassify_to_agent_with_valid_evidence_applies(
        self, agent_py: Path
    ) -> None:
        mw = AIBOMScannerMiddleware(allowed_roots=[str(agent_py.parent)])
        existing = _existing(
            "MyAgent", AIComponentType.OTHER, file_path=str(agent_py), line=4
        )
        data = {
            "reclassify_components": [
                {
                    "instance_id": existing.instance_id,
                    "new_type": "agent",
                    "reason": "ReAct loop + tool dispatch",
                    "agent_evidence": _valid_evidence(agent_py),
                }
            ],
        }
        result = mw.apply_enrichments_from_dict([existing], data)
        assert len(result) == 1
        assert result[0].component_type == AIComponentType.AGENT
        stored = result[0].metadata["agent_evidence"]
        assert stored["definition_file"] == str(agent_py)
        assert "evidence_snippet" not in stored

    def test_reclassify_to_agent_without_evidence_is_rejected(
        self, agent_py: Path
    ) -> None:
        mw = AIBOMScannerMiddleware(allowed_roots=[str(agent_py.parent)])
        existing = _existing(
            "NotAnAgent", AIComponentType.OTHER, file_path=str(agent_py), line=4
        )
        data = {
            "reclassify_components": [
                {
                    "instance_id": existing.instance_id,
                    "new_type": "agent",
                    "reason": "vibes",
                }
            ],
        }
        result = mw.apply_enrichments_from_dict([existing], data)
        assert len(result) == 1
        assert result[0].component_type == AIComponentType.OTHER

    def test_reclassify_to_non_agent_type_bypasses_the_gate(
        self, agent_py: Path
    ) -> None:
        mw = AIBOMScannerMiddleware(allowed_roots=[str(agent_py.parent)])
        existing = _existing(
            "Helper", AIComponentType.OTHER, file_path=str(agent_py), line=1
        )
        data = {
            "reclassify_components": [
                {
                    "instance_id": existing.instance_id,
                    "new_type": "tool",
                    "reason": "It's a tool.",
                }
            ],
        }
        result = mw.apply_enrichments_from_dict([existing], data)
        assert len(result) == 1
        assert result[0].component_type == AIComponentType.TOOL


# ---------------------------------------------------------------------------
# Integration: enriched_components path — strip component_type, keep rest
# ---------------------------------------------------------------------------


class TestGateOnEnrichmentUpdates:
    def test_enrichment_type_change_to_agent_without_evidence_is_stripped(
        self, agent_py: Path
    ) -> None:
        mw = AIBOMScannerMiddleware(allowed_roots=[str(agent_py.parent)])
        existing = _existing(
            "MaybeAgent", AIComponentType.OTHER, file_path=str(agent_py), line=4
        )
        data = {
            "enriched_components": [
                {
                    "instance_id": existing.instance_id,
                    "updates": {
                        "component_type": "agent",
                        "framework": "langchain",
                        "metadata": {"source": "kept"},
                    },
                }
            ],
        }
        result = mw.apply_enrichments_from_dict([existing], data)
        assert len(result) == 1
        comp = result[0]
        assert comp.component_type == AIComponentType.OTHER
        assert comp.framework == "langchain"
        assert comp.metadata.get("source") == "kept"

    def test_enrichment_type_change_to_agent_with_valid_evidence_applies(
        self, agent_py: Path
    ) -> None:
        mw = AIBOMScannerMiddleware(allowed_roots=[str(agent_py.parent)])
        existing = _existing(
            "MyAgent", AIComponentType.OTHER, file_path=str(agent_py), line=4
        )
        data = {
            "enriched_components": [
                {
                    "instance_id": existing.instance_id,
                    "updates": {
                        "component_type": "agent",
                        "framework": "langchain",
                    },
                    "agent_evidence": _valid_evidence(agent_py),
                }
            ],
        }
        result = mw.apply_enrichments_from_dict([existing], data)
        assert len(result) == 1
        comp = result[0]
        assert comp.component_type == AIComponentType.AGENT
        assert comp.framework == "langchain"
        stored = comp.metadata["agent_evidence"]
        assert stored["definition_file"] == str(agent_py)
        assert "evidence_snippet" not in stored

    def test_enrichment_without_type_change_bypasses_the_gate(
        self, agent_py: Path
    ) -> None:
        mw = AIBOMScannerMiddleware(allowed_roots=[str(agent_py.parent)])
        existing = _existing(
            "gpt-4o", AIComponentType.MODEL, file_path=str(agent_py), line=4
        )
        data = {
            "enriched_components": [
                {
                    "instance_id": existing.instance_id,
                    "updates": {
                        "framework": "openai",
                        "metadata": {"provider": "openai"},
                    },
                }
            ],
        }
        result = mw.apply_enrichments_from_dict([existing], data)
        assert len(result) == 1
        comp = result[0]
        assert comp.component_type == AIComponentType.MODEL
        assert comp.framework == "openai"
        assert comp.metadata.get("provider") == "openai"

    def test_enrichment_type_change_to_non_agent_bypasses_the_gate(
        self, agent_py: Path
    ) -> None:
        mw = AIBOMScannerMiddleware(allowed_roots=[str(agent_py.parent)])
        existing = _existing(
            "embedder", AIComponentType.OTHER, file_path=str(agent_py), line=4
        )
        data = {
            "enriched_components": [
                {
                    "instance_id": existing.instance_id,
                    "updates": {
                        "component_type": "embedding",
                        "embedding_model": "text-embedding-3-small",
                    },
                }
            ],
        }
        result = mw.apply_enrichments_from_dict([existing], data)
        assert len(result) == 1
        assert result[0].component_type == AIComponentType.EMBEDDING
