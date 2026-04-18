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

"""Tests for cross_repo_links filtering functions."""

from __future__ import annotations

from aibom.cross_repo_links import (
    _filter_intra_repo_links,
    _filter_quality_bar,
    _resolve_colon_prefixed_repo_paths,
)
from aibom.models.enums import CrossRepoLinkType
from aibom.models.scan import CrossRepoLink, RepoOccurrence


def _link(identifier: str, occurrences: list[RepoOccurrence]) -> CrossRepoLink:
    return CrossRepoLink(
        link_type=CrossRepoLinkType.ENV_VAR_BINDING,
        identifier=identifier,
        occurrences=occurrences,
    )


def _custom_link(identifier: str, occurrences: list[RepoOccurrence]) -> CrossRepoLink:
    return CrossRepoLink(
        link_type=CrossRepoLinkType.CUSTOM,
        identifier=identifier,
        occurrences=occurrences,
        evidence="LLM-derived: USES_MODEL",
    )


class TestFilterIntraRepoLinks:
    def test_keeps_multi_repo_links(self):
        link = _link("MY_VAR", [
            RepoOccurrence(repo_path="/repo-a", role="producer"),
            RepoOccurrence(repo_path="/repo-b", role="consumer"),
        ])
        result = _filter_intra_repo_links([link])
        assert len(result) == 1

    def test_removes_single_repo_links(self):
        link = _link("MY_VAR", [
            RepoOccurrence(repo_path="/repo-a", role="producer"),
            RepoOccurrence(repo_path="/repo-a", role="consumer"),
        ])
        result = _filter_intra_repo_links([link])
        assert len(result) == 0

    def test_removes_llm_derived_intra_repo_links(self):
        link = _custom_link("AgentA->ModelB", [
            RepoOccurrence(repo_path="/repo-a", component_name="AgentA", role="source"),
            RepoOccurrence(repo_path="/repo-a", component_name="ModelB", role="target"),
        ])
        result = _filter_intra_repo_links([link])
        assert len(result) == 0

    def test_keeps_llm_derived_cross_repo_links(self):
        link = _custom_link("AgentA->ModelB", [
            RepoOccurrence(repo_path="/repo-a", component_name="AgentA", role="source"),
            RepoOccurrence(repo_path="/repo-b", component_name="ModelB", role="target"),
        ])
        result = _filter_intra_repo_links([link])
        assert len(result) == 1


class TestFilterQualityBar:
    def test_keeps_links_with_repo_path(self):
        link = _link("MY_VAR", [
            RepoOccurrence(repo_path="/repo-a", role="producer"),
        ])
        result = _filter_quality_bar([link])
        assert len(result) == 1

    def test_removes_links_without_repo_path(self):
        link = _link("MY_VAR", [
            RepoOccurrence(repo_path="", role="producer"),
        ])
        result = _filter_quality_bar([link])
        assert len(result) == 0

    def test_removes_llm_derived_links_without_repo_path(self):
        link = _custom_link("AgentA->ModelB", [
            RepoOccurrence(repo_path="", component_name="AgentA", role="source"),
            RepoOccurrence(repo_path="", component_name="ModelB", role="target"),
        ])
        result = _filter_quality_bar([link])
        assert len(result) == 0

    def test_keeps_llm_derived_links_with_partial_repo_path(self):
        link = _custom_link("AgentA->ModelB", [
            RepoOccurrence(repo_path="/repo-a", component_name="AgentA", role="source"),
            RepoOccurrence(repo_path="", component_name="ModelB", role="target"),
        ])
        result = _filter_quality_bar([link])
        assert len(result) == 1


class TestResolveColonPrefixedRepoPaths:
    def test_resolves_colon_names(self):
        link = _link("MY_VAR", [
            RepoOccurrence(repo_path=":my-repo:", role="producer"),
            RepoOccurrence(repo_path="/scan/path/my-repo", role="consumer"),
        ])
        result = _resolve_colon_prefixed_repo_paths([link], ["/scan/path/my-repo"])
        assert result[0].occurrences[0].repo_path == "/scan/path/my-repo"

    def test_leaves_normal_paths_unchanged(self):
        link = _link("MY_VAR", [
            RepoOccurrence(repo_path="/scan/path/my-repo", role="producer"),
        ])
        result = _resolve_colon_prefixed_repo_paths([link], ["/scan/path/my-repo"])
        assert result[0].occurrences[0].repo_path == "/scan/path/my-repo"
