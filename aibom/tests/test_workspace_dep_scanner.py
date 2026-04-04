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

import json
from pathlib import Path

import pytest

from aibom.models.enums import AIComponentType, DetectionSource
from aibom.models.scan import ScanContext
from aibom.scanners.workspace_dep_scanner import WorkspaceDepScanner


@pytest.fixture
def scanner() -> WorkspaceDepScanner:
    return WorkspaceDepScanner()


class TestWorkspaceDepScanner:
    def test_poetry_path_dependency(
        self, tmp_path: Path, scanner: WorkspaceDepScanner,
    ) -> None:
        local = tmp_path / "libpkg"
        local.mkdir()
        (tmp_path / "pyproject.toml").write_text(
            '[tool.poetry.dependencies]\npython = "^3.11"\n'
            'mypkg = { path = "./libpkg", develop = true }\n',
            encoding="utf-8",
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        c = comps[0]
        assert c.name == "mypkg"
        assert c.component_type == AIComponentType.DEPENDENCY
        assert c.detection_source == DetectionSource.DEPENDENCY_MANIFEST
        assert c.metadata["local"] is True
        assert c.metadata["ecosystem"] == "python"
        assert c.metadata["local_path"] == str(local.resolve())

    def test_pip_editable_install(
        self, tmp_path: Path, scanner: WorkspaceDepScanner,
    ) -> None:
        sub = tmp_path / "submod"
        sub.mkdir()
        (tmp_path / "requirements.txt").write_text(
            "-e ./submod\n",
            encoding="utf-8",
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].metadata["local"] is True
        assert comps[0].metadata["ecosystem"] == "python"
        assert comps[0].metadata["local_path"] == str(sub.resolve())

    def test_uv_workspace_source(
        self, tmp_path: Path, scanner: WorkspaceDepScanner,
    ) -> None:
        member = tmp_path / "ws_member"
        member.mkdir()
        (tmp_path / "pyproject.toml").write_text(
            "[tool.uv.sources]\n"
            "ws_member = { workspace = true }\n",
            encoding="utf-8",
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        c = comps[0]
        assert c.name == "ws_member"
        assert c.metadata["local"] is True
        assert c.metadata["ecosystem"] == "python"
        assert c.metadata["local_path"] == str(member.resolve())

    def test_npm_file_dependency(
        self, tmp_path: Path, scanner: WorkspaceDepScanner,
    ) -> None:
        pkgdir = tmp_path / "local_pkg"
        pkgdir.mkdir()
        (tmp_path / "package.json").write_text(
            json.dumps({"dependencies": {"foo": "file:./local_pkg"}}),
            encoding="utf-8",
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        c = comps[0]
        assert c.name == "foo"
        assert c.metadata["local"] is True
        assert c.metadata["ecosystem"] == "node"
        assert c.metadata["local_path"] == str(pkgdir.resolve())

    def test_go_mod_replace(
        self, tmp_path: Path, scanner: WorkspaceDepScanner,
    ) -> None:
        moddir = tmp_path / "vendor" / "fork"
        moddir.mkdir(parents=True)
        (tmp_path / "go.mod").write_text(
            "module example.com/x\n\ngo 1.22\n\n"
            "replace example.com/foo => ./vendor/fork v0.0.0\n",
            encoding="utf-8",
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        c = comps[0]
        assert c.component_type == AIComponentType.DEPENDENCY
        assert c.metadata["local"] is True
        assert c.metadata["ecosystem"] == "go"
        assert c.metadata["local_path"] == str(moddir.resolve())
