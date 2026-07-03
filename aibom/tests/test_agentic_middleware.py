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

"""Tests for AIBOMScannerMiddleware."""

from __future__ import annotations

import json
import logging

import pytest

from aibom.agentic.middleware import (
    AIBOMScannerMiddleware,
    _parse_iid_name_prefix,
)
from aibom.models import AIComponent, AIComponentType, DetectionSource


@pytest.fixture
def mw():
    return AIBOMScannerMiddleware()


SAMPLE_AGENT_OUTPUT = json.dumps({
    "enriched_components": [
        {
            "instance_id": "comp1_app.py_10",
            "updates": {
                "model_name": "gpt-4o-2024-08-06",
                "metadata": {"license": "proprietary", "deprecated": False},
            },
        }
    ],
    "new_components": [
        {
            "name": "secret-model",
            "component_type": "model",
            "file_path": "hidden.py",
            "line_number": 5,
            "framework": "openai",
            "model_name": "gpt-5",
            "metadata": {"discovered_by": "agent"},
        }
    ],
    "new_relationships": [
        {
            "source_name": "my_agent",
            "target_name": "search_tool",
            "relationship_type": "USES_TOOL",
        }
    ],
    "risk_findings": [
        {
            "flag": "deprecated_model",
            "description": "gpt-3.5-turbo is deprecated",
            "file_path": "old.py",
            "line_number": 3,
            "severity": "medium",
        }
    ],
})


class TestExtractFindings:
    def test_extracts_new_components(self, mw):
        comps, rels, flags = mw.extract_findings(SAMPLE_AGENT_OUTPUT)
        assert len(comps) == 1
        assert comps[0].name == "secret-model"
        assert comps[0].component_type == AIComponentType.MODEL
        assert comps[0].detection_source == DetectionSource.AGENTIC
        assert comps[0].model_name == "gpt-5"

    def test_extracts_relationships(self, mw):
        _, rels, _ = mw.extract_findings(SAMPLE_AGENT_OUTPUT)
        assert len(rels) == 1
        assert rels[0].source_name == "my_agent"
        assert rels[0].target_name == "search_tool"

    def test_extracts_risk_flags(self, mw):
        _, _, flags = mw.extract_findings(SAMPLE_AGENT_OUTPUT)
        assert len(flags) == 1
        assert flags[0].flag == "deprecated_model"
        assert flags[0].severity.value == "medium"

    def test_handles_no_json(self, mw):
        comps, rels, flags = mw.extract_findings("No JSON here, just text.")
        assert comps == []
        assert rels == []
        assert flags == []

    def test_handles_malformed_json(self, mw):
        comps, rels, flags = mw.extract_findings('{"broken: json}')
        assert comps == []

    def test_handles_empty_sections(self, mw):
        output = json.dumps({
            "enriched_components": [],
            "new_components": [],
            "new_relationships": [],
            "risk_findings": [],
        })
        comps, rels, flags = mw.extract_findings(output)
        assert comps == [] and rels == [] and flags == []

    def test_extracts_decision_annotations_for_relationships_and_risk_flags(self, mw):
        output = json.dumps(
            {
                "new_components": [],
                "new_relationships": [
                    {
                        "source_name": "planner_agent",
                        "target_name": "search_tool",
                        "relationship_type": "USES_TOOL",
                        "decision_annotation": {
                            "decision": "derived",
                            "justification": "The planner calls the search tool in the same workflow.",
                            "evidence_kinds": ["relationship_context"],
                            "evidence_locations": [
                                {
                                    "file_path": "/repo/app.py",
                                    "start_line": 12,
                                    "end_line": 16,
                                    "role": "source",
                                }
                            ],
                        },
                    }
                ],
                "risk_findings": [
                    {
                        "flag": "unpinned_model",
                        "description": "The configured model identifier is not version-pinned.",
                        "file_path": "/repo/app.py",
                        "line_number": 18,
                        "severity": "medium",
                        "decision_annotation": {
                            "decision": "flagged",
                            "justification": "The model name is generic and lacks an immutable revision.",
                            "evidence_kinds": ["code_context"],
                            "evidence_locations": [
                                {
                                    "file_path": "/repo/app.py",
                                    "start_line": 18,
                                    "end_line": 18,
                                    "role": "trigger",
                                }
                            ],
                        },
                    }
                ],
            }
        )

        _, rels, flags = mw.extract_findings(output)

        assert rels[0].decision_annotation is not None
        assert rels[0].decision_annotation.decision == "derived"
        assert flags[0].decision_annotation is not None
        assert flags[0].decision_annotation.justification.startswith("The model name")

    def test_extracts_code_snippet_only_when_enabled(self, tmp_path):
        source = tmp_path / "app.py"
        source.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

        output = json.dumps(
            {
                "new_components": [
                    {
                        "name": "router_agent",
                        "component_type": "agent",
                        "file_path": str(source),
                        "line_number": 2,
                        "framework": "custom",
                        "agent_evidence": {
                            "pattern": "framework_agent",
                            "definition_file": str(source),
                            "definition_start_line": 2,
                            "definition_end_line": 3,
                            "evidence_snippet": "line 2",
                            "justification": "Test fixture",
                        },
                        "decision_annotation": {
                            "decision": "added",
                            "justification": "The code declares and uses a request routing agent.",
                            "evidence_kinds": ["code_context"],
                            "evidence_locations": [
                                {
                                    "file_path": str(source),
                                    "start_line": 2,
                                    "end_line": 3,
                                    "role": "primary",
                                }
                            ],
                        },
                    }
                ]
            }
        )

        without_snippets = AIBOMScannerMiddleware(include_code_snippets=False)
        with_snippets = AIBOMScannerMiddleware(include_code_snippets=True)

        comps_without, _, _ = without_snippets.extract_findings(output)
        comps_with, _, _ = with_snippets.extract_findings(output)

        assert comps_without[0].decision_annotation is not None
        assert comps_without[0].decision_annotation.code_snippet is None
        assert comps_with[0].decision_annotation is not None
        assert comps_with[0].decision_annotation.code_snippet is not None
        assert comps_with[0].decision_annotation.code_snippet.text == "line 2\nline 3\n"


class TestApplyEnrichments:
    def test_updates_existing_component(self, mw):
        existing = [
            AIComponent(
                name="my-model",
                component_type=AIComponentType.MODEL,
                file_path="app.py",
                line_number=10,
                instance_id="comp1_app.py_10",
            )
        ]
        enriched = mw.apply_enrichments(existing, SAMPLE_AGENT_OUTPUT)
        assert len(enriched) == 1
        assert enriched[0].model_name == "gpt-4o-2024-08-06"
        assert enriched[0].metadata["license"] == "proprietary"

    def test_preserves_unmatched_components(self, mw):
        existing = [
            AIComponent(
                name="other",
                component_type=AIComponentType.TOOL,
                file_path="x.py",
                line_number=1,
                instance_id="other_x.py_1",
            )
        ]
        enriched = mw.apply_enrichments(existing, SAMPLE_AGENT_OUTPUT)
        assert len(enriched) == 1
        assert enriched[0].name == "other"

    def test_merges_metadata(self, mw):
        existing = [
            AIComponent(
                name="my-model",
                component_type=AIComponentType.MODEL,
                file_path="app.py",
                line_number=10,
                instance_id="comp1_app.py_10",
                metadata={"existing_key": "keep_me"},
            )
        ]
        enriched = mw.apply_enrichments(existing, SAMPLE_AGENT_OUTPUT)
        assert enriched[0].metadata["existing_key"] == "keep_me"
        assert enriched[0].metadata["license"] == "proprietary"

    def test_preserves_scanner_set_metadata_keys(self, mw, caplog):
        """Regression: keys produced by deterministic scanners
        (``ecosystem``, ``manifest``, ``known_ai_package``,
        ``vulnerabilities``, ``risk_flag``) must NOT be stripped by
        ``_sanitize_metadata`` and must NOT trigger the
        ``"Stripped unknown metadata key"`` WARNING. They are not LLM
        hallucinations — the LLM is permitted to echo them."""
        import logging

        caplog.set_level(logging.WARNING, logger="aibom.agentic.middleware")
        existing = [
            AIComponent(
                name="openai",
                component_type=AIComponentType.MODEL,
                file_path="app.py",
                line_number=10,
                instance_id="openai_app.py_10",
            )
        ]
        output = json.dumps(
            {
                "enriched_components": [
                    {
                        "instance_id": "openai_app.py_10",
                        "updates": {
                            "metadata": {
                                "ecosystem": "pypi",
                                "manifest": "requirements.txt",
                                "known_ai_package": True,
                                "vulnerabilities": [{"id": "GHSA-xxxx"}],
                                "risk_flag": {"level": "medium"},
                            }
                        },
                    }
                ]
            }
        )

        enriched = mw.apply_enrichments(existing, output)

        assert enriched[0].metadata["ecosystem"] == "pypi"
        assert enriched[0].metadata["manifest"] == "requirements.txt"
        assert enriched[0].metadata["known_ai_package"] is True
        assert enriched[0].metadata["vulnerabilities"] == [{"id": "GHSA-xxxx"}]
        assert enriched[0].metadata["risk_flag"] == {"level": "medium"}

        offending = [
            r.getMessage()
            for r in caplog.records
            if "Stripped unknown metadata key" in r.getMessage()
        ]
        assert offending == [], (
            f"middleware stripped scanner-set metadata keys: {offending}"
        )

    def test_no_enrichments_returns_copy(self, mw):
        existing = [
            AIComponent(
                name="a", component_type=AIComponentType.MODEL,
                file_path="a.py", line_number=1,
            )
        ]
        output = json.dumps({"enriched_components": []})
        result = mw.apply_enrichments(existing, output)
        assert len(result) == 1
        assert result[0].name == "a"

    def test_applies_component_decision_annotation(self, mw):
        existing = [
            AIComponent(
                name="my-model",
                component_type=AIComponentType.MODEL,
                file_path="app.py",
                line_number=10,
                instance_id="comp1_app.py_10",
            )
        ]
        output = json.dumps(
            {
                "enriched_components": [
                    {
                        "instance_id": "comp1_app.py_10",
                        "updates": {},
                        "decision_annotation": {
                            "decision": "confirmed",
                            "justification": "The constructor argument shows this is an active model dependency.",
                            "evidence_kinds": ["code_context"],
                            "evidence_locations": [
                                {
                                    "file_path": "app.py",
                                    "start_line": 10,
                                    "end_line": 10,
                                    "role": "primary",
                                }
                            ],
                        },
                    }
                ]
            }
        )

        enriched = mw.apply_enrichments(existing, output)

        assert enriched[0].decision_annotation is not None
        assert enriched[0].decision_annotation.decision == "confirmed"
        assert "active model dependency" in enriched[0].decision_annotation.justification


class TestRemoveComponents:
    def test_removes_flagged_components(self, mw):
        existing = [
            AIComponent(
                name="LLMChain", component_type=AIComponentType.MODEL,
                file_path="app.py", line_number=10,
                instance_id="LLMChain_app.py_10",
            ),
            AIComponent(
                name="gpt-4o", component_type=AIComponentType.MODEL,
                file_path="app.py", line_number=5,
                instance_id="gpt-4o_app.py_5",
            ),
        ]
        output = json.dumps({
            "enriched_components": [],
            "new_components": [],
            "remove_components": [
                {
                    "instance_id": "LLMChain_app.py_10",
                    "reason": "LLMChain is a chain, not a model",
                }
            ],
            "reclassify_components": [],
            "new_relationships": [],
            "risk_findings": [],
        })
        result = mw.apply_enrichments(existing, output)
        assert len(result) == 1
        assert result[0].name == "gpt-4o"

    def test_remove_nonexistent_id_is_harmless(self, mw):
        existing = [
            AIComponent(
                name="x", component_type=AIComponentType.TOOL,
                file_path="a.py", line_number=1,
            )
        ]
        output = json.dumps({
            "remove_components": [{"instance_id": "nonexistent_id", "reason": "test"}],
        })
        result = mw.apply_enrichments(existing, output)
        assert len(result) == 1


class TestReclassifyComponents:
    def test_reclassifies_component_type(self, mw):
        existing = [
            AIComponent(
                name="ConversationBufferMemory",
                component_type=AIComponentType.VECTOR_STORE,
                file_path="app.py", line_number=93,
                instance_id="mem_app.py_93",
            )
        ]
        output = json.dumps({
            "reclassify_components": [
                {
                    "instance_id": "mem_app.py_93",
                    "new_type": "memory",
                    "reason": "This is a memory system, not a vector store",
                }
            ],
        })
        result = mw.apply_enrichments(existing, output)
        assert len(result) == 1
        assert result[0].component_type == AIComponentType.MEMORY

    def test_invalid_type_is_ignored(self, mw):
        existing = [
            AIComponent(
                name="x", component_type=AIComponentType.TOOL,
                file_path="a.py", line_number=1,
                instance_id="x_a.py_1",
            )
        ]
        output = json.dumps({
            "reclassify_components": [
                {"instance_id": "x_a.py_1", "new_type": "invalid_type", "reason": "test"}
            ],
        })
        result = mw.apply_enrichments(existing, output)
        assert result[0].component_type == AIComponentType.TOOL

    def test_remove_and_reclassify_together(self, mw):
        existing = [
            AIComponent(
                name="bad", component_type=AIComponentType.MODEL,
                file_path="a.py", line_number=1, instance_id="bad_a.py_1",
            ),
            AIComponent(
                name="fix", component_type=AIComponentType.VECTOR_STORE,
                file_path="a.py", line_number=2, instance_id="fix_a.py_2",
            ),
            AIComponent(
                name="keep", component_type=AIComponentType.TOOL,
                file_path="a.py", line_number=3, instance_id="keep_a.py_3",
            ),
        ]
        output = json.dumps({
            "remove_components": [{"instance_id": "bad_a.py_1", "reason": "false positive"}],
            "reclassify_components": [
                {"instance_id": "fix_a.py_2", "new_type": "retriever", "reason": "is a retriever"}
            ],
        })
        result = mw.apply_enrichments(existing, output)
        assert len(result) == 2
        assert result[0].name == "fix"
        assert result[0].component_type == AIComponentType.RETRIEVER
        assert result[1].name == "keep"


class TestOutOfBatchVerdicts:
    """Verdicts referencing instance_ids that are not in the current
    batch must be dropped with a WARNING.

    The agent sees ``other_detected_components`` for context but is not
    authorized to act on them. Without this guard, verdicts for
    out-of-batch ids silently evaporate because the apply loop only
    walks the current batch, which produced the bug where
    ``DuoAssistantAPI``'s reclassify-to-``tool`` was logged but never
    applied.
    """

    def test_out_of_batch_remove_is_dropped(self, mw, caplog):
        existing = [
            AIComponent(
                name="keep", component_type=AIComponentType.TOOL,
                file_path="a.py", line_number=1, instance_id="keep_a.py_1",
            ),
        ]
        output = json.dumps({
            "remove_components": [
                {"instance_id": "not_in_batch_a.py_99", "reason": "x"}
            ],
        })
        with caplog.at_level("WARNING"):
            result = mw.apply_enrichments(existing, output)
        assert len(result) == 1
        assert result[0].name == "keep"
        assert any(
            "out-of-batch remove" in rec.getMessage()
            for rec in caplog.records
        )

    def test_out_of_batch_reclassify_is_dropped(self, mw, caplog):
        existing = [
            AIComponent(
                name="keep", component_type=AIComponentType.TOOL,
                file_path="a.py", line_number=1, instance_id="keep_a.py_1",
            ),
        ]
        output = json.dumps({
            "reclassify_components": [
                {
                    "instance_id": "not_in_batch_a.py_99",
                    "new_type": "tool",
                    "reason": "x",
                }
            ],
        })
        with caplog.at_level("WARNING"):
            result = mw.apply_enrichments(existing, output)
        assert len(result) == 1
        assert result[0].component_type == AIComponentType.TOOL
        assert any(
            "out-of-batch reclassify" in rec.getMessage()
            for rec in caplog.records
        )

    def test_out_of_batch_enrichment_is_dropped(self, mw, caplog):
        existing = [
            AIComponent(
                name="keep", component_type=AIComponentType.TOOL,
                file_path="a.py", line_number=1, instance_id="keep_a.py_1",
            ),
        ]
        output = json.dumps({
            "enriched_components": [
                {
                    "instance_id": "not_in_batch_a.py_99",
                    "updates": {"model_name": "gpt-5"},
                }
            ],
        })
        with caplog.at_level("WARNING"):
            result = mw.apply_enrichments(existing, output)
        assert len(result) == 1
        assert result[0].metadata == {}
        assert any(
            "out-of-batch enrichment" in rec.getMessage()
            for rec in caplog.records
        )


class TestOutOfBatchRemoveRedirect:
    """Fix 14: an out-of-batch ``remove_components`` verdict whose iid
    does not appear in ``existing`` is no longer dropped silently. The
    middleware parses the ``name`` prefix from the iid, looks for an
    in-batch sibling whose canonical ``(name, type)`` matches, and
    redirects the removal to that sibling's consolidation key. The
    scan-pipeline-level :func:`_propagate_removals` then fans the
    decision out to every other instance of the same logical asset.

    Without this redirect, an agent that emits a removal for the same
    logical concept it sees in ``other_detected_components`` (or one
    that invents a line number for a candidate it just verified) would
    have its decision silently evaporate.
    """

    @pytest.fixture
    def mw(self):
        return AIBOMScannerMiddleware()

    def test_out_of_batch_remove_redirects_to_in_batch_sibling(
        self, mw, caplog,
    ):
        in_batch = AIComponent(
            name="dev", component_type=AIComponentType.MODEL,
            file_path="/repo/charts/svc-a/values.yaml", line_number=10,
            instance_id="dev_/repo/charts/svc-a/values.yaml_10",
        )
        existing = [
            in_batch,
            AIComponent(
                name="gpt-4o", component_type=AIComponentType.MODEL,
                file_path="/repo/app.py", line_number=42,
                instance_id="gpt-4o_/repo/app.py_42",
            ),
        ]
        output = json.dumps({
            "remove_components": [
                {
                    "instance_id": "dev_/repo/charts/svc-b/values.yaml_77",
                    "reason": "bare environment marker, not a model",
                }
            ],
        })
        with caplog.at_level("INFO"):
            result = mw.apply_enrichments(existing, output)

        names = {c.name for c in result}
        assert "dev" not in names, (
            "Out-of-batch remove of 'dev' should have been redirected "
            "to the in-batch 'dev' sibling and removed via consolidation key"
        )
        assert names == {"gpt-4o"}
        assert any(
            "remove redirected" in rec.getMessage().lower()
            for rec in caplog.records
        ), "redirect must be logged at INFO level"

    def test_out_of_batch_remove_with_no_sibling_is_dropped(
        self, mw, caplog,
    ):
        existing = [
            AIComponent(
                name="gpt-4o", component_type=AIComponentType.MODEL,
                file_path="/repo/app.py", line_number=42,
                instance_id="gpt-4o_/repo/app.py_42",
            ),
        ]
        output = json.dumps({
            "remove_components": [
                {
                    "instance_id": "ghost_/repo/missing.py_99",
                    "reason": "agent hallucinated this iid",
                }
            ],
        })
        with caplog.at_level("WARNING"):
            result = mw.apply_enrichments(existing, output)

        assert len(result) == 1
        assert result[0].name == "gpt-4o"
        assert any(
            "no in-batch sibling has canonical name 'ghost'"
            in rec.getMessage()
            for rec in caplog.records
        ), "truly hallucinated iids must still be dropped with a warning"

    def test_out_of_batch_remove_redirects_relative_path_iid(
        self, mw, caplog,
    ):
        existing = [
            AIComponent(
                name="gpt-4o", component_type=AIComponentType.MODEL,
                file_path="src/app.ts", line_number=12,
                instance_id="gpt-4o_src/app.ts_12",
            ),
            AIComponent(
                name="keep", component_type=AIComponentType.TOOL,
                file_path="src/tool.ts", line_number=7,
                instance_id="keep_src/tool.ts_7",
            ),
        ]
        output = json.dumps({
            "remove_components": [
                {
                    "instance_id": "gpt-4o_src/other_app.ts_99",
                    "reason": "same model candidate in another relative path",
                }
            ],
        })
        with caplog.at_level("INFO"):
            result = mw.apply_enrichments(existing, output)

        assert {c.name for c in result} == {"keep"}
        assert any(
            "remove redirected" in rec.getMessage().lower()
            for rec in caplog.records
        ), "relative-path out-of-batch iids must redirect via sibling name"

    def test_parse_iid_name_prefix_uses_longest_relative_sibling_name(self):
        assert _parse_iid_name_prefix(
            "simple_skills_re_src/foo_bar.ts_12",
            ["simple", "simple_skills", "simple_skills_re"],
        ) == "simple_skills_re"

    def test_parse_iid_name_prefix_keeps_unknown_relative_iid_unparseable(self):
        assert _parse_iid_name_prefix(
            "unknown_src/foo_bar.ts_12",
            ["gpt-4o", "claude-3"],
        ) is None

    def test_out_of_batch_remove_with_unparseable_iid_is_dropped(
        self, mw, caplog,
    ):
        existing = [
            AIComponent(
                name="gpt-4o", component_type=AIComponentType.MODEL,
                file_path="/repo/app.py", line_number=42,
                instance_id="gpt-4o_/repo/app.py_42",
            ),
        ]
        output = json.dumps({
            "remove_components": [
                {
                    "instance_id": "not_in_canonical_format_at_all",
                    "reason": "agent garbled the iid",
                }
            ],
        })
        with caplog.at_level("WARNING"):
            result = mw.apply_enrichments(existing, output)

        assert len(result) == 1
        assert any(
            "unparseable out-of-batch remove" in rec.getMessage()
            for rec in caplog.records
        )

    def test_out_of_batch_remove_ambiguous_across_types_is_dropped(
        self, mw, caplog,
    ):
        existing = [
            AIComponent(
                name="orchestrator", component_type=AIComponentType.AGENT,
                file_path="/repo/a.py", line_number=10,
                instance_id="orchestrator_/repo/a.py_10",
            ),
            AIComponent(
                name="orchestrator", component_type=AIComponentType.TOOL,
                file_path="/repo/b.py", line_number=20,
                instance_id="orchestrator_/repo/b.py_20",
            ),
        ]
        output = json.dumps({
            "remove_components": [
                {
                    "instance_id": "orchestrator_/repo/elsewhere.py_99",
                    "reason": "...",
                }
            ],
        })
        with caplog.at_level("WARNING"):
            result = mw.apply_enrichments(existing, output)

        assert len(result) == 2
        assert any(
            "is ambiguous across types" in rec.getMessage()
            for rec in caplog.records
        )

    def test_in_batch_remove_still_applied_directly(self, mw, caplog):
        existing = [
            AIComponent(
                name="bogus", component_type=AIComponentType.MODEL,
                file_path="/repo/x.py", line_number=1,
                instance_id="bogus_/repo/x.py_1",
            ),
            AIComponent(
                name="keep", component_type=AIComponentType.TOOL,
                file_path="/repo/y.py", line_number=2,
                instance_id="keep_/repo/y.py_2",
            ),
        ]
        output = json.dumps({
            "remove_components": [
                {
                    "instance_id": "bogus_/repo/x.py_1",
                    "reason": "false positive",
                }
            ],
        })
        with caplog.at_level("INFO"):
            result = mw.apply_enrichments(existing, output)
        assert {c.name for c in result} == {"keep"}
        assert any(
            "Agent removed component bogus_/repo/x.py_1"
            in rec.getMessage()
            for rec in caplog.records
        )
        assert not any(
            "remove redirected" in rec.getMessage().lower()
            for rec in caplog.records
        ), "in-batch removes must NOT route through the redirect path"


class TestReclassifyWiring:
    """Applied reclassify verdicts must leave the component in a state
    that downstream post-agentic gates can recognise as 'agent touched'.

    Specifically: ``agentic_confidence`` must be set, ``needs_agentic``
    must flip to False, and any ``agent_evidence`` supplied by the LLM
    must land in ``metadata`` for the symmetric gate to consume.
    """

    def test_reclassify_sets_agentic_confidence(self, mw):
        existing = [
            AIComponent(
                name="Foo", component_type=AIComponentType.MODEL,
                file_path="a.py", line_number=1, instance_id="Foo_a.py_1",
            ),
        ]
        output = json.dumps({
            "reclassify_components": [
                {"instance_id": "Foo_a.py_1", "new_type": "tool", "reason": "x"}
            ],
        })
        result = mw.apply_enrichments(existing, output)
        assert result[0].component_type == AIComponentType.TOOL
        assert result[0].agentic_confidence == 0.8
        assert result[0].needs_agentic is False


class TestParseJson:
    def test_parses_pure_json(self, mw):
        text = '{"key": "value"}'
        result = mw._parse_json(text)
        assert result == {"key": "value"}

    def test_rejects_json_with_surrounding_text(self, mw):
        text = 'Here is the result:\n{"key": "value"}\nDone.'
        assert mw._parse_json(text) is None

    def test_returns_none_for_no_json(self, mw):
        assert mw._parse_json("no json here") is None


class TestMalformedListItems:
    """partial/malformed structured output can put a bare string
    (or other non-dict) into a list the middleware iterates. Every ``for item
    in data.get(...)`` loop must skip non-dict elements instead of raising
    ``AttributeError: 'str' object has no attribute 'get'`` (which previously
    aborted the whole agentic stage)."""

    def _existing(self):
        return [
            AIComponent(
                name="my-model",
                component_type=AIComponentType.MODEL,
                file_path="app.py",
                line_number=10,
                instance_id="comp1_app.py_10",
            )
        ]

    def test_apply_enrichments_skips_non_dict_enriched_items(self, mw):
        # Mirrors the exact malformed shape from the ticket: stray string
        # elements interleaved with one valid enrichment dict.
        data = {
            "enriched_components": [
                "agent_evidence: null},{",
                {
                    "instance_id": "comp1_app.py_10",
                    "updates": {"model_name": "gpt-4o-2024-08-06"},
                },
                "instance_id:langchain-op...pyproject.toml_13",
            ]
        }
        enriched = mw.apply_enrichments_from_dict(self._existing(), data)
        assert len(enriched) == 1
        assert enriched[0].model_name == "gpt-4o-2024-08-06"

    def test_apply_enrichments_skips_non_dict_remove_and_reclassify(self, mw):
        data = {
            "remove_components": ["stray", 123, {"instance_id": "comp1_app.py_10"}],
            "reclassify_components": [
                "stray",
                {"instance_id": "comp1_app.py_10", "new_type": "embedding"},
            ],
        }
        # Must not raise; the valid removal wins so the component is dropped.
        enriched = mw.apply_enrichments_from_dict(self._existing(), data)
        assert enriched == []

    def test_extract_findings_skips_non_dict_items(self, mw):
        data = {
            "new_components": [
                "stray string",
                {
                    "name": "secret-model",
                    "component_type": "model",
                    "file_path": "hidden.py",
                    "line_number": 5,
                    "model_name": "gpt-5",
                },
            ],
            "new_relationships": [
                42,
                {
                    "source_name": "my_agent",
                    "target_name": "search_tool",
                    "relationship_type": "USES_TOOL",
                },
            ],
            "risk_findings": [
                None,
                {"flag": "deprecated_model", "description": "x", "severity": "medium"},
            ],
        }
        comps, rels, flags = mw.extract_findings_from_dict(data)
        assert [c.name for c in comps] == ["secret-model"]
        assert len(rels) == 1
        assert len(flags) == 1

    def test_skipped_non_dict_items_are_logged(self, mw, caplog):
        # Silently dropping malformed items hides a provider emitting garbage;
        # each skip must be observable at debug level, naming the field.
        data = {
            "new_components": ["stray string"],
            "new_relationships": [42],
            "risk_findings": [None],
        }
        with caplog.at_level(logging.DEBUG, logger="aibom.agentic.middleware"):
            mw.extract_findings_from_dict(data)
        skipped = [
            r.getMessage()
            for r in caplog.records
            if "non-dict" in r.getMessage().lower()
        ]
        # One debug line per malformed field, each identifying the field name.
        assert any("new_components" in m for m in skipped)
        assert any("new_relationships" in m for m in skipped)
        assert any("risk_findings" in m for m in skipped)
