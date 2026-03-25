# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aibom.scanners.file_cache import (
    cache_stats,
    clear_cache,
    read_text_cached,
    read_text_cached_async,
    warm_cache_async,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()
    yield
    clear_cache()


class TestReadTextCached:
    def test_returns_file_content(self, tmp_path: Path):
        f = tmp_path / "hello.py"
        f.write_text("print('hi')")
        assert read_text_cached(f) == "print('hi')"

    def test_second_read_is_cache_hit(self, tmp_path: Path):
        f = tmp_path / "a.py"
        f.write_text("x = 1")
        read_text_cached(f)
        stats_before = cache_stats()
        read_text_cached(f)
        stats_after = cache_stats()
        assert stats_after["hits"] == stats_before["hits"] + 1
        assert stats_after["misses"] == stats_before["misses"]

    def test_accepts_string_path(self, tmp_path: Path):
        f = tmp_path / "b.py"
        f.write_text("y = 2")
        assert read_text_cached(str(f)) == "y = 2"

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_text_cached(tmp_path / "nonexistent.py")

    def test_replace_errors_by_default(self, tmp_path: Path):
        f = tmp_path / "binary.py"
        f.write_bytes(b"hello \xff world")
        text = read_text_cached(f)
        assert "hello" in text
        assert "\ufffd" in text


class TestReadTextCachedAsync:
    def test_async_read(self, tmp_path: Path):
        f = tmp_path / "async.py"
        f.write_text("async_content")
        result = asyncio.run(read_text_cached_async(f))
        assert result == "async_content"

    def test_async_uses_shared_cache(self, tmp_path: Path):
        f = tmp_path / "shared.py"
        f.write_text("shared")
        read_text_cached(f)
        stats = cache_stats()
        asyncio.run(read_text_cached_async(f))
        assert cache_stats()["hits"] == stats["hits"] + 1


class TestWarmCacheAsync:
    def test_warm_cache_preloads_files(self, tmp_path: Path):
        files = []
        for i in range(5):
            f = tmp_path / f"file{i}.py"
            f.write_text(f"content_{i}")
            files.append(f)

        loaded = asyncio.run(warm_cache_async(files))
        assert loaded == 5
        assert cache_stats()["entries"] == 5
        for f in files:
            assert read_text_cached(f) == f.read_text()

    def test_warm_skips_already_cached(self, tmp_path: Path):
        f = tmp_path / "existing.py"
        f.write_text("already")
        read_text_cached(f)

        loaded = asyncio.run(warm_cache_async([f]))
        assert loaded == 0

    def test_warm_skips_unreadable_files(self, tmp_path: Path):
        good = tmp_path / "good.py"
        good.write_text("ok")
        bad = tmp_path / "bad.py"

        loaded = asyncio.run(warm_cache_async([good, bad]))
        assert loaded == 1

    def test_warm_empty_list(self):
        loaded = asyncio.run(warm_cache_async([]))
        assert loaded == 0


class TestClearCache:
    def test_clear_resets_stats(self, tmp_path: Path):
        f = tmp_path / "x.py"
        f.write_text("x")
        read_text_cached(f)
        assert cache_stats()["entries"] == 1
        clear_cache()
        assert cache_stats() == {"hits": 0, "misses": 0, "entries": 0}
