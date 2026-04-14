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

from aibom.incremental import OrgCache, incremental_scan
from aibom.models import AIComponent, AIComponentType, ScanResult, SourceResult


def _git_commit(repo: Path) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@e.co"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "t"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "c"],
        check=True,
        capture_output=True,
    )


def _make_scan(name: str = "x") -> ScanResult:
    return ScanResult(
        metadata={},
        sources=[
            SourceResult(
                path="/p",
                components=[
                    AIComponent(name=name, component_type=AIComponentType.MODEL),
                ],
                relationships=[],
            )
        ],
    )


def test_cache_stores_and_retrieves(tmp_path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "f.txt").write_text("1", encoding="utf-8")
    _git_commit(repo)

    cache = OrgCache(base_dir=tmp_path / "cache")
    sr = _make_scan("a")
    cache.store(str(repo.resolve()), sr)
    got = cache.get_cached(str(repo.resolve()))
    assert got is not None
    assert got.sources[0].components[0].name == "a"


def test_cache_miss_on_sha_mismatch(tmp_path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "f.txt").write_text("1", encoding="utf-8")
    _git_commit(repo)

    cache = OrgCache(base_dir=tmp_path / "cache")
    cache.store(str(repo.resolve()), _make_scan("v1"))
    (repo / "f.txt").write_text("2", encoding="utf-8")
    _git_commit(repo)
    assert cache.get_cached(str(repo.resolve())) is None


def test_cache_hit_on_same_sha(tmp_path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "f.txt").write_text("1", encoding="utf-8")
    _git_commit(repo)

    cache = OrgCache(base_dir=tmp_path / "cache")
    cache.store(str(repo.resolve()), _make_scan("stable"))
    assert cache.get_cached(str(repo.resolve())) is not None
    assert cache.get_cached(str(repo.resolve())) is not None


def test_non_git_directory_skips_caching(tmp_path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    cache = OrgCache(base_dir=tmp_path / "cache")

    def scan_fn(p: str) -> ScanResult:
        return _make_scan("live")

    out = incremental_scan([str(plain)], scan_fn, cache)
    assert len(out) == 1
    path, sr, cached = out[0]
    assert cached is False
    assert sr.sources[0].components[0].name == "live"


def test_corrupt_primary_falls_through_to_fallback(tmp_path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "f.txt").write_text("1", encoding="utf-8")
    _git_commit(repo)

    primary = tmp_path / "primary"
    legacy = tmp_path / "legacy"
    legacy_cache = OrgCache(base_dir=legacy)
    legacy_cache.store(str(repo.resolve()), _make_scan("from-legacy"))

    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()

    corrupt_dir = primary / OrgCache(base_dir=primary)._repo_dir(
        str(repo.resolve()), primary
    ).relative_to(primary)
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    (corrupt_dir / f"{sha}.json").write_text("{BAD", encoding="utf-8")

    cache = OrgCache.__new__(OrgCache)
    cache.base_dir = primary
    cache.fallback_dirs = [legacy]

    result = cache.get_cached(str(repo.resolve()))
    assert result is not None
    assert result.sources[0].components[0].name == "from-legacy"


def test_incremental_scan_cache_hit(tmp_path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "f.txt").write_text("1", encoding="utf-8")
    _git_commit(repo)

    cache = OrgCache(base_dir=tmp_path / "cache")
    calls: list[str] = []

    def scan_fn(p: str) -> ScanResult:
        calls.append(p)
        return _make_scan("once")

    rs = str(repo.resolve())
    incremental_scan([rs], scan_fn, cache)
    out = incremental_scan([rs], scan_fn, cache)
    assert len(out) == 1
    assert out[0][2] is True
    assert calls == [rs]
