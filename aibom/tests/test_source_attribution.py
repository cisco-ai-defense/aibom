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

from aibom.source_attribution import (
    SOURCE_KIND_CONTAINER_IMAGE,
    SOURCE_KIND_GIT,
    SOURCE_KIND_LOCAL_PATH,
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
