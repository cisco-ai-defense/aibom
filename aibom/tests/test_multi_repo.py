# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from aibom.multi_repo import (
    ClonedRepo,
    discover_repos,
    is_git_url,
    read_repos_file,
)


class TestIsGitUrl:
    def test_https(self) -> None:
        assert is_git_url("https://github.com/org/repo.git")

    def test_ssh(self) -> None:
        assert is_git_url("git@github.com:org/repo.git")

    def test_git_protocol(self) -> None:
        assert is_git_url("git://example.com/repo.git")

    def test_ssh_protocol(self) -> None:
        assert is_git_url("ssh://git@example.com/repo.git")

    def test_local_path(self) -> None:
        assert not is_git_url("/home/user/repo")

    def test_relative_path(self) -> None:
        assert not is_git_url("./repo")

    def test_empty(self) -> None:
        assert not is_git_url("")


class TestDiscoverRepos:
    def test_finds_git_repos(self, tmp_path: Path) -> None:
        (tmp_path / "repo-a" / ".git").mkdir(parents=True)
        (tmp_path / "repo-b" / ".git").mkdir(parents=True)
        (tmp_path / "not-a-repo").mkdir()

        repos = discover_repos(tmp_path)
        names = [r.name for r in repos]
        assert "repo-a" in names
        assert "repo-b" in names
        assert "not-a-repo" not in names

    def test_nested_repos(self, tmp_path: Path) -> None:
        (tmp_path / "org" / "project" / ".git").mkdir(parents=True)
        repos = discover_repos(tmp_path)
        assert len(repos) == 1
        assert repos[0].name == "project"

    def test_respects_max_depth(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "d" / ".git"
        deep.mkdir(parents=True)
        assert discover_repos(tmp_path, max_depth=2) == []
        assert len(discover_repos(tmp_path, max_depth=4)) == 1

    def test_stops_at_git_root(self, tmp_path: Path) -> None:
        (tmp_path / "parent" / ".git").mkdir(parents=True)
        (tmp_path / "parent" / "nested" / ".git").mkdir(parents=True)
        repos = discover_repos(tmp_path)
        assert len(repos) == 1
        assert repos[0].name == "parent"

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        assert discover_repos(tmp_path / "no-such") == []

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert discover_repos(tmp_path) == []


class TestReadReposFile:
    def test_json_array(self, tmp_path: Path) -> None:
        f = tmp_path / "repos.json"
        f.write_text(json.dumps(["/repo/a", "/repo/b"]))
        assert read_repos_file(f) == ["/repo/a", "/repo/b"]

    def test_newline_delimited(self, tmp_path: Path) -> None:
        f = tmp_path / "repos.txt"
        f.write_text("/repo/a\n/repo/b\n\n# comment\n/repo/c\n")
        result = read_repos_file(f)
        assert result == ["/repo/a", "/repo/b", "/repo/c"]

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert read_repos_file(f) == []

    def test_json_with_git_urls(self, tmp_path: Path) -> None:
        urls = ["https://github.com/org/a.git", "git@github.com:org/b.git"]
        f = tmp_path / "urls.json"
        f.write_text(json.dumps(urls))
        assert read_repos_file(f) == urls

    def test_mixed_text(self, tmp_path: Path) -> None:
        f = tmp_path / "mixed.txt"
        f.write_text("# local repos\n/local/repo\nhttps://github.com/org/repo\n")
        result = read_repos_file(f)
        assert len(result) == 2


class TestClonedRepo:
    @patch("aibom.multi_repo.subprocess.run")
    def test_clone_success(self, mock_run: object, tmp_path: Path) -> None:
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0, stdout="", stderr=""
        )
        with patch("aibom.multi_repo.tempfile.mkdtemp", return_value=str(tmp_path)):
            with ClonedRepo("https://github.com/org/repo.git") as path:
                assert path == tmp_path

    @patch("aibom.multi_repo.subprocess.run")
    def test_clone_failure_raises(self, mock_run: object) -> None:
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=128, stdout="", stderr="fatal: not found"
        )
        with pytest.raises(RuntimeError, match="git clone failed"):
            with ClonedRepo("https://github.com/org/bad.git") as _:
                pass

    @patch("aibom.multi_repo.subprocess.run")
    def test_branch_option(self, mock_run: object, tmp_path: Path) -> None:
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0, stdout="", stderr=""
        )
        with patch("aibom.multi_repo.tempfile.mkdtemp", return_value=str(tmp_path)):
            with ClonedRepo("https://github.com/org/repo.git", branch="dev") as _:
                call_args = mock_run.call_args[0][0]  # type: ignore[attr-defined]
                assert "--branch" in call_args
                assert "dev" in call_args
