# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aibom.scan_cache import (
    _CACHE_VERSION,
    cache_info,
    cache_key,
    clear_cache,
    load_cached,
    save_cached,
)


def _init_git_repo(
    root: Path, *, remote_url: str, commit_msg: str = "initial"
) -> None:
    """Initialize a minimal git repo under *root* so :func:`_git_info` succeeds.

    The repo is self-contained (``user.email`` / ``user.name`` set only
    locally) and has a single commit so ``git rev-parse HEAD`` works.
    The supplied *remote_url* is attached as ``origin`` but never
    fetched from.
    """
    env = {"HOME": str(root), "GIT_CONFIG_NOSYSTEM": "1"}

    def _run(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**env},
        )

    subprocess.run(
        ["git", "init", "-q", "-b", "main"],
        cwd=root,
        check=True,
        env={**env},
    )
    _run("config", "user.email", "ci@example.invalid")
    _run("config", "user.name", "CI")
    _run("config", "commit.gpgsign", "false")
    _run("remote", "add", "origin", remote_url)
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    _run("add", "seed.txt")
    _run("commit", "-q", "-m", commit_msg)


class TestCacheKey:
    def test_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "app.py"
        f.write_text("x = 1\n")
        k1 = cache_key([str(tmp_path)])
        k2 = cache_key([str(tmp_path)])
        assert k1 == k2

    def test_changes_on_modification(self, tmp_path: Path) -> None:
        f = tmp_path / "app.py"
        f.write_text("x = 1\n")
        k1 = cache_key([str(tmp_path)])
        f.write_text("x = 2\n")
        k2 = cache_key([str(tmp_path)])
        assert k1 != k2

    def test_changes_when_analysis_settings_change(self, tmp_path: Path) -> None:
        f = tmp_path / "app.py"
        f.write_text("x = 1\n")
        k1 = cache_key([str(tmp_path)], {"strict": False, "llm_config": {"model": "gpt-5.4"}})
        k2 = cache_key([str(tmp_path)], {"strict": True, "llm_config": {"model": "gpt-5.4"}})
        assert k1 != k2

    def test_analysis_settings_are_order_insensitive(self, tmp_path: Path) -> None:
        f = tmp_path / "app.py"
        f.write_text("x = 1\n")
        s1 = {"strict": True, "llm_config": {"model": "gpt-5.4", "provider": "azure_openai"}}
        s2 = {"llm_config": {"provider": "azure_openai", "model": "gpt-5.4"}, "strict": True}
        assert cache_key([str(tmp_path)], s1) == cache_key([str(tmp_path)], s2)


class TestSaveLoadClear:
    def test_round_trip(self, tmp_path: Path) -> None:
        cd = tmp_path / "cache"
        data = {"components": [{"name": "test"}]}
        save_cached(cd, "abc123", data)
        loaded = load_cached(cd, "abc123")
        assert loaded is not None
        assert loaded["components"] == [{"name": "test"}]
        assert loaded["_cache_version"] == _CACHE_VERSION

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        cd = tmp_path / "cache"
        assert load_cached(cd, "nonexistent") is None

    def test_clear_removes_all(self, tmp_path: Path) -> None:
        cd = tmp_path / "cache"
        save_cached(cd, "a", {"x": 1})
        save_cached(cd, "b", {"x": 2})
        removed = clear_cache(cd)
        assert removed == 2
        assert load_cached(cd, "a") is None

    def test_clear_empty_dir(self, tmp_path: Path) -> None:
        cd = tmp_path / "cache"
        assert clear_cache(cd) == 0


class TestCacheInfo:
    def test_lists_entries(self, tmp_path: Path) -> None:
        cd = tmp_path / "cache"
        save_cached(cd, "entry1", {"data": "hello"})
        entries = cache_info(cd)
        assert len(entries) == 1
        assert entries[0]["key"] == "entry1"
        assert "cached_at" in entries[0]

    def test_empty_returns_empty(self, tmp_path: Path) -> None:
        cd = tmp_path / "cache"
        assert cache_info(cd) == []


class TestFallbackOnCorruptPrimary:
    def test_corrupt_primary_falls_through_to_legacy(self, tmp_path: Path) -> None:
        primary = tmp_path / "primary"
        legacy = tmp_path / "legacy"
        primary.mkdir(parents=True)
        legacy.mkdir(parents=True)

        save_cached(legacy, "abc", {"components": [{"name": "good"}]})

        corrupt_file = primary / "abc.json"
        corrupt_file.write_text("{INVALID JSON", encoding="utf-8")

        result = load_cached(primary, "abc", search_dirs=[legacy])
        assert result is not None
        assert result["components"] == [{"name": "good"}]

    def test_version_mismatch_primary_falls_through_to_legacy(self, tmp_path: Path) -> None:
        primary = tmp_path / "primary"
        legacy = tmp_path / "legacy"
        primary.mkdir(parents=True)

        save_cached(legacy, "abc", {"components": [{"name": "good"}]})

        stale = primary / "abc.json"
        stale.write_text('{"_cache_version": 999}', encoding="utf-8")

        result = load_cached(primary, "abc", search_dirs=[legacy])
        assert result is not None
        assert result["components"] == [{"name": "good"}]


@pytest.fixture
def multi_service_repo(tmp_path: Path) -> Path:
    """A single git repo with two sibling service directories.

    Emulates the ``full-repo`` / ``sub-dir`` / ``single-file`` scan modes
    all operating inside the same repository revision. The repo has a
    fake ``origin`` URL so :func:`_git_info` returns a real
    ``(url, sha, relpath)`` triple without any network access.
    """
    repo_root = tmp_path / "multi_service"
    repo_root.mkdir()
    service_a = repo_root / "service_a"
    service_a.mkdir()
    (service_a / "main.py").write_text("x = 1\n", encoding="utf-8")
    service_b = repo_root / "service_b"
    service_b.mkdir()
    (service_b / "main.py").write_text("y = 2\n", encoding="utf-8")
    _init_git_repo(
        repo_root,
        remote_url="https://example.invalid/org/multi-service.git",
    )
    return repo_root


class TestCacheKeyScanModes:
    """Full-repo, sub-dir, and file-level scans must produce distinct keys.

    Regression coverage for the cache-collision bug where any path
    inside the same git repo hashed to the same key because only
    ``(url, sha)`` were used. The fix includes the path relative to
    the git top-level in the key.
    """

    def test_sibling_subdirs_have_different_keys(
        self, multi_service_repo: Path
    ) -> None:
        repo = multi_service_repo
        k_a = cache_key([str(repo / "service_a")])
        k_b = cache_key([str(repo / "service_b")])
        assert k_a != k_b

    def test_full_repo_scan_differs_from_subdir_scan(
        self, multi_service_repo: Path
    ) -> None:
        repo = multi_service_repo
        k_root = cache_key([str(repo)])
        k_sub = cache_key([str(repo / "service_a")])
        assert k_root != k_sub

    def test_file_level_scan_differs_from_parent_subdir_scan(
        self, multi_service_repo: Path
    ) -> None:
        repo = multi_service_repo
        k_file = cache_key([str(repo / "service_a" / "main.py")])
        k_subdir = cache_key([str(repo / "service_a")])
        assert k_file != k_subdir

    def test_same_subdir_scanned_twice_hits_same_key(
        self, multi_service_repo: Path
    ) -> None:
        repo = multi_service_repo
        target = str(repo / "service_a")
        assert cache_key([target]) == cache_key([target])

    def test_multipath_scan_is_stable(
        self, multi_service_repo: Path
    ) -> None:
        repo = multi_service_repo
        k1 = cache_key([
            str(repo / "service_a"),
            str(repo / "service_b"),
        ])
        k2 = cache_key([
            str(repo / "service_b"),
            str(repo / "service_a"),
        ])
        assert k1 == k2
