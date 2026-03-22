# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from aibom.cross_ref import CrossRefIndex, EnvVarEntry
from aibom.dep_graph import DependencyGraph, RepoEdge
from aibom.models import AIComponent, ComponentRelationship
from aibom.models.enums import AIComponentType, RelationshipType


def _dep(name: str) -> AIComponent:
    return AIComponent(name=name, component_type=AIComponentType.DEPENDENCY)


class TestDependencyGraph:
    def test_shared_packages(self, tmp_path: Path) -> None:
        repo_a = tmp_path / "repo_a"
        repo_b = tmp_path / "repo_b"
        repo_a.mkdir()
        repo_b.mkdir()
        g = DependencyGraph()
        g.add_repo(str(repo_a), [_dep("numpy"), _dep("pandas")])
        g.add_repo(str(repo_b), [_dep("numpy"), _dep("scipy")])
        edges = g.build()
        shared = [e for e in edges if e.edge_type == "shared_package"]
        assert len(shared) == 1
        assert shared[0].details["package"] == "numpy"
        assert {shared[0].source_repo, shared[0].target_repo} == {str(repo_a), str(repo_b)}

    def test_env_var_flows(self, tmp_path: Path) -> None:
        repo_a = tmp_path / "repo_a"
        repo_b = tmp_path / "repo_b"
        repo_a.mkdir()
        repo_b.mkdir()
        env_file = repo_a / ".env"
        env_file.write_text("MODEL_NAME=gpt-4\n", encoding="utf-8")
        xref = CrossRefIndex()
        xref.env["MODEL_NAME"] = [
            EnvVarEntry(
                name="MODEL_NAME",
                value="gpt-4",
                source_type="dotenv",
                source_path=str(env_file),
            )
        ]
        g = DependencyGraph()
        g.add_repo(str(repo_a), [], cross_ref=xref)
        g.add_repo(
            str(repo_b),
            [
                AIComponent(
                    name="cfg",
                    component_type=AIComponentType.OTHER,
                    metadata={"env": "MODEL_NAME"},
                )
            ],
        )
        edges = g.build()
        flows = [e for e in edges if e.edge_type == "env_var_flow"]
        assert len(flows) == 1
        assert flows[0].details["env_var"] == "MODEL_NAME"
        assert flows[0].source_repo == str(repo_a)
        assert flows[0].target_repo == str(repo_b)

    def test_no_edges_single_repo(self, tmp_path: Path) -> None:
        repo = tmp_path / "solo"
        repo.mkdir()
        g = DependencyGraph()
        g.add_repo(str(repo), [_dep("numpy")])
        assert g.build() == []

    def test_to_relationships(self, tmp_path: Path) -> None:
        repo_a = tmp_path / "repo_a"
        repo_b = tmp_path / "repo_b"
        repo_a.mkdir()
        repo_b.mkdir()
        g = DependencyGraph()
        g.add_repo(str(repo_a), [_dep("x")])
        g.add_repo(str(repo_b), [_dep("x")])
        g.build()
        rels = g.to_relationships()
        assert len(rels) == 1
        r = rels[0]
        assert isinstance(r, ComponentRelationship)
        assert r.relationship_type == RelationshipType.CUSTOM
        assert r.label == "shared_package"
        assert r.source_instance_id == "repo:repo_a"
        assert r.target_instance_id == "repo:repo_b"
        assert r.source_name == "repo_a"
        assert r.target_name == "repo_b"

    def test_to_dict(self, tmp_path: Path) -> None:
        repo_a = tmp_path / "repo_a"
        repo_b = tmp_path / "repo_b"
        repo_a.mkdir()
        repo_b.mkdir()
        g = DependencyGraph()
        g.add_repo(str(repo_a), [_dep("numpy")])
        g.add_repo(str(repo_b), [_dep("numpy")])
        g.build()
        d = g.to_dict()
        json.dumps(d)
        assert d["total_repos"] == 2
        assert d["total_edges"] == 1
        assert len(d["repos"]) == 2
        assert d["edges"][0]["type"] == "shared_package"
        assert d["edges"][0]["details"] == {"package": "numpy"}

    def test_empty_graph(self) -> None:
        g = DependencyGraph()
        assert g.build() == []
        assert g.to_relationships() == []
        d = g.to_dict()
        assert d["repos"] == []
        assert d["edges"] == []
        assert d["total_repos"] == 0
        assert d["total_edges"] == 0

    def test_multiple_shared_packages(self, tmp_path: Path) -> None:
        repo_a = tmp_path / "repo_a"
        repo_b = tmp_path / "repo_b"
        repo_c = tmp_path / "repo_c"
        for p in (repo_a, repo_b, repo_c):
            p.mkdir()
        g = DependencyGraph()
        g.add_repo(str(repo_a), [_dep("p1"), _dep("p2")])
        g.add_repo(str(repo_b), [_dep("p1"), _dep("p2")])
        g.add_repo(str(repo_c), [_dep("p1"), _dep("p2")])
        edges = g.build()
        shared = [e for e in edges if e.edge_type == "shared_package"]
        assert len(shared) == 6
        by_pkg: dict[str, list[RepoEdge]] = {}
        for e in shared:
            by_pkg.setdefault(e.details["package"], []).append(e)
        assert set(by_pkg.keys()) == {"p1", "p2"}
        assert len(by_pkg["p1"]) == 3
        assert len(by_pkg["p2"]) == 3
