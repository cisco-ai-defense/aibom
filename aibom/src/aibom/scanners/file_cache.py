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

"""Thread-safe file content cache shared across scanners.

Multiple scanners read the same files during a single pipeline run.
This cache deduplicates I/O so each file is read at most once.
Provides both sync (``read_text_cached``) and async (``read_text_cached_async``)
entry points; both share the same cache.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

_lock = threading.Lock()
_cache: dict[str, str] = {}
_miss_count = 0
_hit_count = 0


def read_text_cached(path: Path | str, *, encoding: str = "utf-8", errors: str = "replace") -> str:
    """Read a file's text content, returning a cached copy on subsequent calls."""
    global _miss_count, _hit_count
    key = str(path)
    with _lock:
        if key in _cache:
            _hit_count += 1
            return _cache[key]

    text = Path(key).read_text(encoding=encoding, errors=errors)
    with _lock:
        _cache[key] = text
        _miss_count += 1
    return text


async def read_text_cached_async(
    path: Path | str, *, encoding: str = "utf-8", errors: str = "replace",
) -> str:
    """Async variant — checks the shared cache, falls back to ``aiofiles``."""
    global _miss_count, _hit_count
    key = str(path)
    with _lock:
        if key in _cache:
            _hit_count += 1
            return _cache[key]

    try:
        import aiofiles
        async with aiofiles.open(key, encoding=encoding, errors=errors) as f:
            text = await f.read()
    except ImportError:
        text = await asyncio.to_thread(Path(key).read_text, encoding=encoding, errors=errors)

    with _lock:
        _cache[key] = text
        _miss_count += 1
    return text


async def warm_cache_async(
    paths: list[Path | str],
    *,
    encoding: str = "utf-8",
    errors: str = "replace",
    concurrency: int = 64,
) -> int:
    """Pre-populate the cache by reading *paths* concurrently with aiofiles.

    Returns the number of files successfully cached.  Files already in the
    cache are skipped.  This should be called once at pipeline start so that
    all subsequent sync ``read_text_cached`` calls are instant cache hits.
    """
    uncached = []
    with _lock:
        for p in paths:
            key = str(p)
            if key not in _cache:
                uncached.append(key)

    if not uncached:
        return 0

    sem = asyncio.Semaphore(concurrency)
    loaded = 0

    async def _read_one(key: str) -> None:
        nonlocal loaded
        async with sem:
            try:
                import aiofiles
                async with aiofiles.open(key, encoding=encoding, errors=errors) as f:
                    text = await f.read()
            except ImportError:
                text = await asyncio.to_thread(
                    Path(key).read_text, encoding=encoding, errors=errors,
                )
            except OSError:
                return
            global _miss_count
            with _lock:
                if key not in _cache:
                    _cache[key] = text
                    _miss_count += 1
                    loaded += 1

    await asyncio.gather(*[_read_one(k) for k in uncached])
    return loaded


def cache_stats() -> dict[str, int]:
    """Return hit/miss statistics for diagnostics."""
    with _lock:
        return {"hits": _hit_count, "misses": _miss_count, "entries": len(_cache)}


def clear_cache() -> None:
    """Reset the cache between runs (useful for testing)."""
    global _miss_count, _hit_count
    with _lock:
        _cache.clear()
        _miss_count = 0
        _hit_count = 0
