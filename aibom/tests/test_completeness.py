# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the completeness scoring engine."""

import pytest

from aibom.completeness import compute_completeness_score, CompletenessReport
from aibom.structures import CategorizationOutput, ComponentRelationship


def _make_output(components=None, relationships=None):
    return CategorizationOutput(
        components=components or {},
        relationships=relationships or [],
    )


class TestCompletenessScoring:
    def test_empty_output_scores_zero(self):
        report = compute_completeness_score(_make_output())
        assert report.score == 0.0
        assert report.warnings == []

    def test_agent_with_model_scores_high(self):
        output = _make_output(
            components={
                "agent": [{"name": "MyAgent", "instance_id": "a1", "category": "agent"}],
                "model": [{"name": "GPT4", "instance_id": "m1", "category": "model"}],
            },
            relationships=[
                ComponentRelationship(
                    source_instance_id="a1",
                    target_instance_id="m1",
                    label="USES_LLM",
                    source_name="MyAgent",
                    target_name="GPT4",
                    source_category="agent",
                    target_category="model",
                ),
            ],
        )
        report = compute_completeness_score(output)
        assert report.score > 50.0

    def test_agent_without_model_warns(self):
        output = _make_output(
            components={
                "agent": [{"name": "OrphanAgent", "instance_id": "a1", "category": "agent"}],
            },
        )
        report = compute_completeness_score(output)
        assert any("model" in w.lower() for w in report.warnings)

    def test_agent_without_prompt_warns(self):
        output = _make_output(
            components={
                "agent": [{"name": "NoPrmpt", "instance_id": "a1", "category": "agent"}],
                "model": [{"name": "GPT4", "instance_id": "m1", "category": "model"}],
            },
            relationships=[
                ComponentRelationship(
                    source_instance_id="a1",
                    target_instance_id="m1",
                    label="USES_LLM",
                    source_name="NoPrmpt",
                    target_name="GPT4",
                    source_category="agent",
                    target_category="model",
                ),
            ],
        )
        report = compute_completeness_score(output)
        assert any("prompt" in w.lower() for w in report.warnings)

    def test_orphaned_model_warns(self):
        output = _make_output(
            components={
                "model": [{"name": "StandaloneModel", "instance_id": "m1", "category": "model"}],
            },
        )
        report = compute_completeness_score(output)
        assert any("not referenced" in w.lower() or "model" in w.lower() for w in report.warnings)

    def test_retriever_without_embedding_warns(self):
        output = _make_output(
            components={
                "retriever": [{"name": "MyRetriever", "instance_id": "r1", "category": "retriever"}],
            },
        )
        report = compute_completeness_score(output)
        assert report.score < 100.0

    def test_perfect_score(self):
        output = _make_output(
            components={
                "agent": [{"name": "Agent", "instance_id": "a1", "category": "agent"}],
                "model": [{"name": "LLM", "instance_id": "m1", "category": "model"}],
                "tool": [{"name": "Search", "instance_id": "t1", "category": "tool"}],
                "prompt": [{"name": "Prompt", "instance_id": "p1", "category": "prompt"}],
                "memory": [{"name": "Mem", "instance_id": "mem1", "category": "memory"}],
            },
            relationships=[
                ComponentRelationship("a1", "m1", "USES_LLM", "Agent", "LLM", "agent", "model"),
                ComponentRelationship("a1", "t1", "USES_TOOL", "Agent", "Search", "agent", "tool"),
                ComponentRelationship("a1", "p1", "USES_PROMPT", "Agent", "Prompt", "agent", "prompt"),
                ComponentRelationship("a1", "mem1", "USES_MEMORY", "Agent", "Mem", "agent", "memory"),
            ],
        )
        report = compute_completeness_score(output)
        assert report.score == 100.0

    def test_completeness_report_structure(self):
        output = _make_output(
            components={
                "agent": [{"name": "A", "instance_id": "a1", "category": "agent"}],
            },
        )
        report = compute_completeness_score(output)
        assert isinstance(report, CompletenessReport)
        d = report.to_dict()
        assert "score" in d
        assert "warnings" in d
        assert "breakdown" in d

    def test_orphaned_tool_warns(self):
        output = _make_output(
            components={
                "tool": [{"name": "Search", "instance_id": "t1", "category": "tool"}],
            },
        )
        report = compute_completeness_score(output)
        assert any("not referenced" in w.lower() for w in report.warnings)

    def test_orphaned_memory_warns(self):
        output = _make_output(
            components={
                "memory": [{"name": "MemSaver", "instance_id": "mem1", "category": "memory"}],
            },
        )
        report = compute_completeness_score(output)
        assert any("not referenced" in w.lower() for w in report.warnings)

    def test_orphaned_prompt_warns(self):
        output = _make_output(
            components={
                "prompt": [{"name": "MyPrompt", "instance_id": "p1", "category": "prompt"}],
            },
        )
        report = compute_completeness_score(output)
        assert any("not referenced" in w.lower() for w in report.warnings)

    def test_no_components_no_concepts_scores_zero(self):
        report = compute_completeness_score(_make_output(components={}))
        assert report.score == 0.0
        assert report.warnings == []
