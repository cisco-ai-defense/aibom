# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""No-LLM runs drop model candidates that only a classifier could resolve."""

from __future__ import annotations

from aibom.models import AIComponent, AIComponentType
from aibom.scan_pipeline import _drop_unresolved_model_candidates


def _model(name: str, **kwargs) -> AIComponent:
    return AIComponent(
        name=name,
        component_type=AIComponentType.MODEL,
        file_path="app.py",
        line_number=1,
        **kwargs,
    )


class TestDropUnresolvedModelCandidates:
    def test_unresolved_candidates_are_dropped(self) -> None:
        comps = [
            _model("WebSearch", metadata={"partial_kb_id": "genai.types.WebSearch"}),
            _model(
                "langchain_openai.ChatOpenAI",
                metadata={"kb_id": "langchain_openai.ChatOpenAI"},
            ),
            _model(
                "OpenAIModel", metadata={"detection_method": "bare_provider_client"}
            ),
        ]
        assert _drop_unresolved_model_candidates(comps) == []

    def test_resolved_and_other_rows_are_kept(self) -> None:
        comps = [
            _model("gpt-4o", model_name="gpt-4o"),
            _model(
                "langchain_openai.ChatOpenAI",
                model_name="gpt-4o-mini",
                metadata={"kb_id": "langchain_openai.ChatOpenAI"},
            ),
            _model("all-minilm-l6-v2", metadata={"detection_method": "regex"}),
            AIComponent(
                name="langchain_openai.OpenAIEmbeddings",
                component_type=AIComponentType.EMBEDDING,
                file_path="app.py",
                line_number=2,
                metadata={"kb_id": "langchain_openai.OpenAIEmbeddings"},
            ),
        ]
        assert _drop_unresolved_model_candidates(comps) == comps
