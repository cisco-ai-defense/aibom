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

import subprocess
from pathlib import Path

import pytest

from aibom.source_attribution import (
    SOURCE_KIND_CONTAINER_IMAGE,
    SOURCE_KIND_GIT,
    SOURCE_KIND_LOCAL_PATH,
    canonicalize_image_ref,
    canonicalize_source_ref,
    detect_source_kind,
)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git("init", cwd=path)
    _git("config", "user.email", "dev@example.com", cwd=path)
    _git("config", "user.name", "Dev", cwd=path)


class TestDetectSourceKind:
    def test_git_working_tree_is_git(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        assert detect_source_kind(str(tmp_path)) == SOURCE_KIND_GIT

    def test_nested_dir_in_git_tree_is_git(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        nested = tmp_path / "pkg" / "sub"
        nested.mkdir(parents=True)
        assert detect_source_kind(str(nested)) == SOURCE_KIND_GIT

    def test_plain_directory_is_local_path(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        assert detect_source_kind(str(plain)) == SOURCE_KIND_LOCAL_PATH

    def test_container_image_flag_short_circuits(self, tmp_path: Path) -> None:
        assert (
            detect_source_kind("my-app:latest", is_container_image=True)
            == SOURCE_KIND_CONTAINER_IMAGE
        )

    def test_nonexistent_path_is_local_path(self) -> None:
        assert detect_source_kind("/no/such/path/here") == SOURCE_KIND_LOCAL_PATH

    def test_detection_is_deterministic(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        first = detect_source_kind(str(tmp_path))
        second = detect_source_kind(str(tmp_path))
        assert first == second == SOURCE_KIND_GIT


class TestCanonicalizeGitRemote:
    # Every spelling of the same repo must collapse to one identical string.
    EQUIVALENT = [
        "git@github.com:org/repo.git",
        "git@github.com:org/repo",
        "https://github.com/org/repo",
        "https://github.com/org/repo.git",
        "https://github.com/org/repo/",
        "https://user:token@github.com/org/repo.git",
        "ssh://git@github.com/org/repo.git",
        "https://GitHub.com/org/repo.git",
        "git://github.com/org/repo.git",
    ]

    @pytest.mark.parametrize("remote", EQUIVALENT)
    def test_all_spellings_collapse(self, remote: str) -> None:
        assert canonicalize_source_ref(remote, SOURCE_KIND_GIT) == "github.com/org/repo"

    def test_one_identical_string_across_set(self) -> None:
        results = {canonicalize_source_ref(r, SOURCE_KIND_GIT) for r in self.EQUIVALENT}
        assert results == {"github.com/org/repo"}

    def test_path_case_preserved_host_lowered(self) -> None:
        # Host lowercased; path stays case-sensitive (org/repo can be mixed).
        assert (
            canonicalize_source_ref("https://GitLab.com/Org/Repo.git", SOURCE_KIND_GIT)
            == "gitlab.com/Org/Repo"
        )

    def test_self_hosted_with_port(self) -> None:
        assert (
            canonicalize_source_ref(
                "https://git.internal.example:8443/team/svc.git",
                SOURCE_KIND_GIT,
            )
            == "git.internal.example/team/svc"
        )

    def test_empty_returns_empty(self) -> None:
        assert canonicalize_source_ref("", SOURCE_KIND_GIT) == ""


class TestCanonicalizeImageRef:
    def test_tag_stripped_and_hub_defaults(self) -> None:
        assert canonicalize_image_ref("redis:7") == "docker.io/library/redis"

    def test_namespaced_hub_image(self) -> None:
        assert (
            canonicalize_image_ref("bitnami/redis:latest") == "docker.io/bitnami/redis"
        )

    def test_digest_stripped(self) -> None:
        assert (
            canonicalize_image_ref("ghcr.io/org/app@sha256:" + "a" * 64)
            == "ghcr.io/org/app"
        )

    def test_registry_with_port(self) -> None:
        assert (
            canonicalize_image_ref("registry.example.com:5000/team/app:v1.2")
            == "registry.example.com:5000/team/app"
        )

    def test_uppercase_registry_lowered(self) -> None:
        assert canonicalize_image_ref("GHCR.io/Org/App:latest") == "ghcr.io/Org/App"

    def test_via_canonicalize_source_ref(self) -> None:
        assert (
            canonicalize_source_ref("redis:7", SOURCE_KIND_CONTAINER_IMAGE)
            == "docker.io/library/redis"
        )
