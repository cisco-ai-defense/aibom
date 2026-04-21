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

Keyed by ``repo_url@commit_sha:relpath`` or ``path@mtime_hash`` so
repeated scans of the same codebase at the same revision can skip the
full pipeline. The ``relpath`` component is the scan target's path
relative to the git top-level, which ensures that two sibling
subdirectories (or individual files) inside the *same* repository
produce distinct cache keys. For a scan whose target is the repo root
itself, the relpath is ``"."``.
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

_CACHE_VERSION = 3


def _git_info(path: str) -> tuple[str, str, str] | None:
    """Return ``(remote_url, commit_sha, relpath_from_toplevel)`` if *path*
    is inside a git repo, else ``None``.

    *path* may point at a directory (scan of a sub-tree), a single file
    (file-level scan), or the repo root (full-repo scan). The returned
    ``relpath`` is computed with :meth:`pathlib.Path.resolve` on both
    ends so symlinked checkouts resolve identically on each side.
    When the target is the repo root, the relpath is ``"."`` rather
    than an empty string so the downstream cache-key format never loses
    its separator.
    """
    try:
        target = Path(path)
    except (TypeError, ValueError):
        return None
    if not target.exists():
        return None
    cwd = str(target if target.is_dir() else target.parent)
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        toplevel_raw = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if not toplevel_raw:
        return None
    try:
        rel = target.resolve().relative_to(Path(toplevel_raw).resolve())
    except (OSError, ValueError):
        return None
    relpath = rel.as_posix() or "."
    return url, sha, relpath


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


def _normalize_settings(value: Any) -> Any:
    """Convert cache settings into a deterministic, JSON-serializable shape."""
    if isinstance(value, dict):
        return {str(k): _normalize_settings(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_settings(v) for v in value]
    if isinstance(value, Path):
        try:
            return str(value.resolve())
        except OSError:
            return str(value)
    if hasattr(value, "value"):
        return getattr(value, "value")
    return value


def cache_key(scan_paths: list[str], settings: dict[str, Any] | None = None) -> str:
    """Derive a cache key from the scan paths and analysis settings."""
    parts: list[str] = []
    for p in sorted(scan_paths):
        info = _git_info(p)
        if info:
            url, sha, relpath = info
            parts.append(f"{url}@{sha}:{relpath}")
        else:
            parts.append(f"{p}@{_mtime_hash(p)}")
    combined = json.dumps(
        {
            "paths": parts,
            "settings": _normalize_settings(settings or {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def load_cached(
    cache_dir: Path,
    key: str,
    *,
    search_dirs: list[Path] | None = None,
) -> Optional[dict[str, Any]]:
    """Load a cached scan result. Returns None on miss or version mismatch."""
    dirs = [cache_dir]
    if search_dirs:
        dirs.extend(search_dirs)

    seen: set[str] = set()
    for directory in dirs:
        resolved = str(directory)
        if resolved in seen:
            continue
        seen.add(resolved)
        p = _cache_path(directory, key)
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("_cache_version") != _CACHE_VERSION:
                _LOGGER.debug("Cache version mismatch for %s in %s", key, directory)
                continue
            _LOGGER.info("Cache hit: %s (cached %s)", key[:12], data.get("_cached_at", "?"))
            return data
        except (json.JSONDecodeError, OSError) as exc:
            _LOGGER.debug("Cache load error for %s in %s: %s", key, directory, exc)
            continue
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


def cache_info(
    cache_dir: Path,
    *,
    search_dirs: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """List all cached entries with metadata."""
    dirs = [cache_dir]
    if search_dirs:
        dirs.extend(search_dirs)
    entries = []
    seen_files: set[str] = set()
    for directory in dirs:
        if not directory.exists():
            continue
        for f in sorted(directory.glob("*.json")):
            if str(f) in seen_files:
                continue
            seen_files.add(str(f))
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
