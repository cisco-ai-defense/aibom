# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from aibom.scan_cache import (
    cache_info,
    cache_key,
    clear_cache,
    load_cached,
    save_cached,
)


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
        data = {"_v2": True, "components": [{"name": "test"}]}
        save_cached(cd, "abc123", data)
        loaded = load_cached(cd, "abc123")
        assert loaded is not None
        assert loaded["components"] == [{"name": "test"}]
        assert loaded["_cache_version"] == 2

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
