# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from aibom.cross_ref import detect_external_repo_deps


class TestPoetryGitDeps:
    def test_poetry_git_dependency(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            """[tool.poetry.dependencies]
python = "^3.11"
ai-common-py = {git = "https://github.com/org/ai-common-py.git", branch = "main", subdirectory = "models"}
""",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].name == "ai-common-py"
        assert deps[0].dep_type == "git"
        assert "ai-common-py" in deps[0].url_or_path
        assert deps[0].subdirectory == "models"
        assert deps[0].branch == "main"

    def test_poetry_path_dependency(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            """[tool.poetry.dependencies]
shared-lib = {path = "../../shared-lib"}
""",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].dep_type == "path"
        assert deps[0].escapes_root is True

    def test_poetry_dev_deps(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            """[tool.poetry.dev-dependencies]
test-utils = {git = "https://github.com/org/test-utils.git"}
""",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].name == "test-utils"

    def test_poetry_group_deps(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            """[tool.poetry.group.dev.dependencies]
lint-rules = {git = "https://github.com/org/lint-rules.git", tag = "v1.0"}
""",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].tag == "v1.0"


class TestUvSources:
    def test_uv_git_source(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            """[project]
dependencies = ["ai-common"]

[tool.uv.sources]
ai-common = {git = "https://github.com/org/ai-common.git", branch = "main"}
""",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].name == "ai-common"
        assert deps[0].dep_type == "git"
        assert deps[0].branch == "main"

    def test_uv_path_source(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            """[project]
dependencies = ["local-pkg"]

[tool.uv.sources]
local-pkg = {path = "../local-pkg"}
""",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].dep_type == "path"
        assert deps[0].escapes_root is True

    def test_uv_with_subdirectory(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            """[tool.uv.sources]
ml-lib = {git = "https://github.com/org/mono.git", subdirectory = "packages/ml"}
""",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].subdirectory == "packages/ml"


class TestRequirementsTxtGit:
    def test_git_plus_https(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text(
            "git+https://github.com/org/ai-common-py.git@main#egg=ai-common-py\n",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].dep_type == "git"
        assert deps[0].name == "ai-common-py"
        assert deps[0].branch == "main"

    def test_editable_git(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text(
            "-e git+https://github.com/org/sdk.git@v2.0#egg=sdk\n",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].dep_type == "git"
        assert deps[0].name == "sdk"

    def test_editable_local_path(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text(
            "-e ../shared-lib\n",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].dep_type == "editable"
        assert deps[0].escapes_root is True

    def test_normal_deps_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text(
            "langchain>=0.2.0\nopenai==1.30.0\n",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 0


class TestPackageJsonGit:
    def test_git_dep(self, tmp_path: Path) -> None:
        payload = {
            "dependencies": {
                "ai-utils": "git+https://github.com/org/ai-utils.git#main",
                "express": "^4.18.0",
            }
        }
        (tmp_path / "package.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].name == "ai-utils"
        assert deps[0].dep_type == "git"

    def test_github_shorthand(self, tmp_path: Path) -> None:
        payload = {"dependencies": {"tool": "github:org/tool#v2"}}
        (tmp_path / "package.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1

    def test_file_dep(self, tmp_path: Path) -> None:
        payload = {"dependencies": {"local": "file:../local-pkg"}}
        (tmp_path / "package.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].dep_type == "path"
        assert deps[0].escapes_root is True


class TestGoModReplace:
    def test_local_replace(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text(
            """module github.com/org/svc
go 1.22
require github.com/org/common v0.0.0
replace github.com/org/common => ../common
""",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].dep_type == "path"
        assert deps[0].url_or_path == "../common"
        assert deps[0].escapes_root is True

    def test_remote_replace(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text(
            """module github.com/org/svc
replace github.com/old/pkg => github.com/new/pkg v1.0.0
""",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].dep_type == "git"
        assert deps[0].url_or_path == "github.com/new/pkg"

    def test_no_replace(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text(
            """module github.com/org/svc
go 1.22
require github.com/gin-gonic/gin v1.9.0
""",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 0


class TestPathEscaping:
    def test_path_inside_repo_not_flagged(self, tmp_path: Path) -> None:
        sub = tmp_path / "packages" / "lib"
        sub.mkdir(parents=True)
        (tmp_path / "pyproject.toml").write_text(
            """[tool.poetry.dependencies]
lib = {path = "packages/lib"}
""",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].escapes_root is False

    def test_absolute_path_flagged(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            """[tool.poetry.dependencies]
lib = {path = "/opt/shared/lib"}
""",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].escapes_root is True

    def test_git_deps_not_flagged(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            """[tool.poetry.dependencies]
lib = {git = "https://github.com/org/lib.git"}
""",
            encoding="utf-8",
        )
        deps = detect_external_repo_deps([str(tmp_path)])
        assert len(deps) == 1
        assert deps[0].escapes_root is False
