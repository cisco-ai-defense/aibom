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


_PYTHON_SUFFIXES: frozenset[str] = frozenset({".py", ".ipynb"})


def is_python_source(path: Path | str) -> bool:
    """Return True if *path* is a Python source file or Jupyter notebook."""
    return Path(path).suffix.lower() in _PYTHON_SUFFIXES


def read_python_source(path: Path | str) -> str:
    """Read Python source from a ``.py`` file or extract code cells from ``.ipynb``.

    For ``.py`` files this is identical to ``read_text_cached``.
    For ``.ipynb`` files the notebook is parsed and code cells are
    concatenated into a virtual Python source string.  The result is
    cached so repeated calls across scanners pay no extra I/O.
    """
    p = Path(path)
    if p.suffix.lower() == ".ipynb":
        cache_key = f"__notebook_python__:{p}"
        with _lock:
            if cache_key in _cache:
                _hit_count_inc()
                return _cache[cache_key]
        from ..notebook_parser import extract_code_from_notebook

        text = extract_code_from_notebook(p)
        with _lock:
            _cache[cache_key] = text
            _miss_count_inc()
        return text
    return read_text_cached(p)


def _hit_count_inc() -> None:
    global _hit_count
    _hit_count += 1


def _miss_count_inc() -> None:
    global _miss_count
    _miss_count += 1


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
