# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aibom.models import AIComponent, AIComponentType
from aibom.scan_pipeline import _consolidate_vector_stores


class TestConsolidateVectorStores:
    def test_rows_at_one_site_merge_to_one(self) -> None:
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
                file_path="a.py",
                line_number=1,
                heuristic_confidence=0.9,
                metadata={"evidence": [{"file": "c.py", "line": 3}]},
            ),
        ]
        out = _consolidate_vector_stores(comps)
        assert len(out) == 1
        w = out[0]
        assert w.name == "WeaviateVectorStore"
        assert w.metadata.get("store_technology") == "weaviate"
        assert w.metadata.get("consolidated_count") == 2
        ev = w.metadata.get("evidence") or []
        files = {(e["file"], e["line"]) for e in ev}
        assert ("c.py", 3) in files

    def test_same_technology_at_different_sites_stays_separate(self) -> None:
        comps = [
            AIComponent(
                name="langchain_community.vectorstores.FAISS.load_local",
                component_type=AIComponentType.VECTOR_STORE,
                file_path=path,
                line_number=line,
            )
            for path, line in (("retriever.py", 52), ("indexer.py", 336))
        ]
        out = _consolidate_vector_stores(comps)
        assert len(out) == 2
        assert {(c.file_path, c.line_number) for c in out} == {
            ("retriever.py", 52),
            ("indexer.py", 336),
        }
        assert all(c.metadata.get("store_technology") == "faiss" for c in out)
        assert all("consolidated_count" not in c.metadata for c in out)

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
                file_path="one.py",
                line_number=10,
                heuristic_confidence=0.6,
                metadata={"evidence": [{"file": "two.py", "line": 20}]},
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
