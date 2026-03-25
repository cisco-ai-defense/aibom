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

"""Optional on-disk cache for scan results.

Keyed by ``repo_url@commit_sha`` or ``path@mtime_hash`` so repeated scans
of the same codebase at the same revision can skip the full pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

_LOGGER = logging.getLogger(__name__)

_CACHE_VERSION = 1


def _git_info(path: str) -> tuple[str, str] | None:
    """Return (remote_url, commit_sha) if *path* is inside a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=path,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return url, sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _mtime_hash(path: str) -> str:
    """Compute a fast hash of mtimes for all files under *path*."""
    h = hashlib.sha256()
    root = Path(path)
    if root.is_file():
        h.update(f"{root}:{root.stat().st_mtime_ns}".encode())
    else:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                try:
                    h.update(f"{f}:{f.stat().st_mtime_ns}".encode())
                except OSError:
                    continue
    return h.hexdigest()[:16]


def cache_key(scan_paths: list[str]) -> str:
    """Derive a cache key from the scan paths."""
    parts: list[str] = []
    for p in sorted(scan_paths):
        info = _git_info(p)
        if info:
            url, sha = info
            parts.append(f"{url}@{sha}")
        else:
            parts.append(f"{p}@{_mtime_hash(p)}")
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def load_cached(cache_dir: Path, key: str) -> Optional[dict[str, Any]]:
    """Load a cached scan result. Returns None on miss or version mismatch."""
    p = _cache_path(cache_dir, key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("_cache_version") != _CACHE_VERSION:
            _LOGGER.debug("Cache version mismatch for %s", key)
            return None
        _LOGGER.info("Cache hit: %s (cached %s)", key[:12], data.get("_cached_at", "?"))
        return data
    except (json.JSONDecodeError, OSError) as exc:
        _LOGGER.debug("Cache load error for %s: %s", key, exc)
        return None


def save_cached(cache_dir: Path, key: str, data: dict[str, Any]) -> None:
    """Persist a scan result to the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **data,
        "_cache_version": _CACHE_VERSION,
        "_cache_key": key,
        "_cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    p = _cache_path(cache_dir, key)
    p.write_text(json.dumps(payload, default=str), encoding="utf-8")
    _LOGGER.info("Cached result: %s → %s", key[:12], p)


def clear_cache(cache_dir: Path) -> int:
    """Remove all cached scan results. Returns count of files removed."""
    if not cache_dir.exists():
        return 0
    count = 0
    for f in cache_dir.glob("*.json"):
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    return count


def cache_info(cache_dir: Path) -> list[dict[str, Any]]:
    """List all cached entries with metadata."""
    if not cache_dir.exists():
        return []
    entries = []
    for f in sorted(cache_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            entries.append({
                "key": data.get("_cache_key", f.stem),
                "cached_at": data.get("_cached_at", "unknown"),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "path": str(f),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return entries
