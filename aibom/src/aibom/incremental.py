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

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Optional

from .cache_paths import cache_read_dirs, ensure_cache_dir
from .models import ScanResult


def _is_git_repo(repo_path: Path) -> bool:
    return (repo_path / ".git").exists()


def is_git_repo(repo_path: str | Path) -> bool:
    return _is_git_repo(Path(repo_path))


def _repo_bucket_key(repo_path: str) -> str:
    return hashlib.sha256(os.path.abspath(repo_path).encode()).hexdigest()


class OrgCache:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        if base_dir is None:
            read_dirs = cache_read_dirs("org")
            self.base_dir = ensure_cache_dir("org")
            self.fallback_dirs = [p for p in read_dirs if p != self.base_dir]
        else:
            self.base_dir = base_dir
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.fallback_dirs = []

    def _repo_dir(self, repo_path: str, base_dir: Optional[Path] = None) -> Path:
        key = _repo_bucket_key(repo_path)
        return (base_dir or self.base_dir) / key

    @staticmethod
    def _get_head_sha(repo_path: str) -> Optional[str]:
        p = Path(repo_path)
        if not _is_git_repo(p):
            return None
        try:
            out = subprocess.run(
                ["git", "-C", str(p.resolve()), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            return out.stdout.strip() or None
        except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
            return None

    def get_cached(self, repo_path: str) -> Optional[ScanResult]:
        sha = self._get_head_sha(repo_path)
        if not sha:
            return None
        candidate_dirs = [self.base_dir, *self.fallback_dirs]
        for base_dir in candidate_dirs:
            cache_file = self._repo_dir(repo_path, base_dir) / f"{sha}.json"
            if not cache_file.is_file():
                continue
            try:
                raw = json.loads(cache_file.read_text(encoding="utf-8"))
                return ScanResult.model_validate(raw)
            except (json.JSONDecodeError, OSError, ValueError):
                return None
        return None

    def store(self, repo_path: str, result: ScanResult) -> None:
        sha = self._get_head_sha(repo_path)
        if not sha:
            return
        rdir = self._repo_dir(repo_path)
        rdir.mkdir(parents=True, exist_ok=True)
        path = rdir / f"{sha}.json"
        path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )


def incremental_scan(
    paths: list[str],
    scan_fn: Callable[[str], ScanResult],
    cache: OrgCache,
) -> list[tuple[str, ScanResult, bool]]:
    out: list[tuple[str, ScanResult, bool]] = []
    for path in paths:
        p = Path(path).resolve()
        ps = str(p)
        if not _is_git_repo(p):
            out.append((ps, scan_fn(ps), False))
            continue
        cached = cache.get_cached(ps)
        if cached is not None:
            out.append((ps, cached, True))
            continue
        result = scan_fn(ps)
        cache.store(ps, result)
        out.append((ps, result, False))
    return out
