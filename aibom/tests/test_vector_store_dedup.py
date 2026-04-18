# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aibom.models import AIComponent, AIComponentType
from aibom.scan_pipeline import _consolidate_vector_stores


class TestConsolidateVectorStores:
    def test_multiple_weaviate_wrappers_merge_to_one(self) -> None:
        comps = [
            AIComponent(
                name="WeaviateRetriever",
                component_type=AIComponentType.VECTOR_STORE,
                file_path="a.py",
                line_number=1,
                heuristic_confidence=0.7,
            ),
            AIComponent(
                name="WeaviateVectorStore",
                component_type=AIComponentType.VECTOR_STORE,
                file_path="b.py",
                line_number=2,
                heuristic_confidence=0.9,
            ),
            AIComponent(
                name="WeaviateManager",
                component_type=AIComponentType.VECTOR_STORE,
                file_path="c.py",
                line_number=3,
                heuristic_confidence=0.5,
            ),
        ]
        out = _consolidate_vector_stores(comps)
        assert len(out) == 1
        w = out[0]
        assert w.name == "WeaviateVectorStore"
        assert w.metadata.get("store_technology") == "weaviate"
        assert w.metadata.get("consolidated_count") == 3
        ev = w.metadata.get("evidence") or []
        files = {(e["file"], e["line"]) for e in ev}
        assert ("a.py", 1) in files
        assert ("c.py", 3) in files

    def test_mixed_technologies_remain_separate(self) -> None:
        comps = [
            AIComponent(
                name="WeaviateRetriever",
                component_type=AIComponentType.VECTOR_STORE,
                file_path="w.py",
                line_number=1,
            ),
            AIComponent(
                name="PineconeIndex",
                component_type=AIComponentType.VECTOR_STORE,
                file_path="p.py",
                line_number=1,
            ),
        ]
        out = _consolidate_vector_stores(comps)
        assert len(out) == 2
        techs = {c.metadata.get("store_technology") for c in out}
        assert techs == {"weaviate", "pinecone"}

    def test_non_matching_names_unchanged(self) -> None:
        comps = [
            AIComponent(
                name="CustomVectorBackend",
                component_type=AIComponentType.VECTOR_STORE,
                file_path="x.py",
                line_number=1,
            ),
        ]
        out = _consolidate_vector_stores(comps)
        assert len(out) == 1
        assert out[0].metadata.get("store_technology") is None

    def test_evidence_accumulates_locations(self) -> None:
        comps = [
            AIComponent(
                name="qdrant_client",
                component_type=AIComponentType.VECTOR_STORE,
                file_path="one.py",
                line_number=10,
                heuristic_confidence=0.8,
            ),
            AIComponent(
                name="QdrantWrapper",
                component_type=AIComponentType.VECTOR_STORE,
                file_path="two.py",
                line_number=20,
                heuristic_confidence=0.6,
            ),
        ]
        out = _consolidate_vector_stores(comps)
        assert len(out) == 1
        m = out[0].metadata
        assert m.get("store_technology") == "qdrant"
        ev = m.get("evidence") or []
        assert any(e["file"] == "two.py" and e["line"] == 20 for e in ev)

    def test_store_technology_on_singleton(self) -> None:
        comps = [
            AIComponent(
                name="MilvusCollection",
                component_type=AIComponentType.VECTOR_STORE,
                file_path="m.py",
                line_number=5,
            ),
        ]
        out = _consolidate_vector_stores(comps)
        assert len(out) == 1
        assert out[0].metadata.get("store_technology") == "milvus"

    def test_metadata_store_technology_overrides_name_hint(self) -> None:
        comps = [
            AIComponent(
                name="env:WEAVIATE_ENDPOINT",
                component_type=AIComponentType.VECTOR_STORE,
                file_path="values.yaml",
                line_number=12,
                metadata={"store_technology": "chromadb"},
            ),
        ]
        out = _consolidate_vector_stores(comps)
        assert len(out) == 1
        assert out[0].metadata.get("store_technology") == "chromadb"
