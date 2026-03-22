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

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cross_ref import CrossRefIndex
from .models import AIComponent, ComponentRelationship
from .models.enums import AIComponentType, RelationshipType

_LOGGER = logging.getLogger(__name__)


@dataclass
class RepoNode:
    """A repository in the dependency graph."""

    path: str
    name: str
    components: list[AIComponent] = field(default_factory=list)
    packages: set[str] = field(default_factory=set)
    env_vars_defined: set[str] = field(default_factory=set)
    env_vars_referenced: set[str] = field(default_factory=set)


@dataclass
class RepoEdge:
    """A directed edge between two repos."""

    source_repo: str
    target_repo: str
    edge_type: str  # "shared_package", "env_var_flow", "api_contract", "config_ref"
    details: dict[str, Any] = field(default_factory=dict)


class DependencyGraph:
    """Cross-repo dependency graph builder."""

    def __init__(self) -> None:
        self._repos: dict[str, RepoNode] = {}
        self._edges: list[RepoEdge] = []

    def add_repo(
        self,
        path: str,
        components: list[AIComponent],
        cross_ref: CrossRefIndex | None = None,
    ) -> None:
        """Register a repo and its scan results."""
        name = Path(path).name
        node = RepoNode(path=path, name=name, components=list(components))

        for c in components:
            if c.component_type == AIComponentType.DEPENDENCY:
                node.packages.add(c.name)

        for c in components:
            env = c.metadata.get("env") or c.metadata.get("config_key")
            if isinstance(env, str):
                node.env_vars_referenced.add(env)

        if cross_ref:
            root = Path(path).resolve()
            for var_name, entries in cross_ref.env.items():
                for entry in entries:
                    try:
                        src = Path(entry.source_path).resolve()
                    except OSError:
                        _LOGGER.debug(
                            "Skipping env entry with unreadable path: %s",
                            entry.source_path,
                        )
                        continue
                    if src.is_relative_to(root):
                        node.env_vars_defined.add(var_name)

        self._repos[path] = node

    def build(self) -> list[RepoEdge]:
        """Compute edges between repos."""
        self._edges = []
        self._find_shared_packages()
        self._find_env_var_flows()
        return list(self._edges)

    def _find_shared_packages(self) -> None:
        """Find packages used by multiple repos."""
        pkg_repos: dict[str, list[str]] = defaultdict(list)
        for path, node in self._repos.items():
            for pkg in node.packages:
                pkg_repos[pkg].append(path)

        for pkg, repos in sorted(pkg_repos.items()):
            if len(repos) < 2:
                continue
            repos_sorted = sorted(repos)
            for i, src in enumerate(repos_sorted):
                for tgt in repos_sorted[i + 1 :]:
                    self._edges.append(
                        RepoEdge(
                            source_repo=src,
                            target_repo=tgt,
                            edge_type="shared_package",
                            details={"package": pkg},
                        )
                    )

    def _find_env_var_flows(self) -> None:
        """Find env vars defined in one repo and referenced in another."""
        repo_paths = sorted(self._repos.keys())
        for src_path in repo_paths:
            src_node = self._repos[src_path]
            for tgt_path in repo_paths:
                if src_path == tgt_path:
                    continue
                tgt_node = self._repos[tgt_path]
                shared = src_node.env_vars_defined & tgt_node.env_vars_referenced
                for var in sorted(shared):
                    self._edges.append(
                        RepoEdge(
                            source_repo=src_path,
                            target_repo=tgt_path,
                            edge_type="env_var_flow",
                            details={"env_var": var},
                        )
                    )

    def to_relationships(self) -> list[ComponentRelationship]:
        """Convert graph edges to AIBOM ComponentRelationship objects."""
        rels: list[ComponentRelationship] = []
        for edge in self._edges:
            src_name = Path(edge.source_repo).name
            tgt_name = Path(edge.target_repo).name
            rels.append(
                ComponentRelationship(
                    source_instance_id=f"repo:{src_name}",
                    target_instance_id=f"repo:{tgt_name}",
                    relationship_type=RelationshipType.CUSTOM,
                    label=edge.edge_type,
                    source_name=src_name,
                    target_name=tgt_name,
                )
            )
        return rels

    def to_dict(self) -> dict[str, Any]:
        """Export graph as serializable dict for JSON output."""
        return {
            "repos": [
                {
                    "path": n.path,
                    "name": n.name,
                    "component_count": len(n.components),
                    "packages": sorted(n.packages),
                    "env_vars_defined": sorted(n.env_vars_defined),
                    "env_vars_referenced": sorted(n.env_vars_referenced),
                }
                for n in sorted(self._repos.values(), key=lambda x: x.path)
            ],
            "edges": [
                {
                    "source": Path(e.source_repo).name,
                    "target": Path(e.target_repo).name,
                    "type": e.edge_type,
                    "details": e.details,
                }
                for e in self._edges
            ],
            "total_repos": len(self._repos),
            "total_edges": len(self._edges),
        }
