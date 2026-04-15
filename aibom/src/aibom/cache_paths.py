from __future__ import annotations

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_dir

CacheType = Literal["scan", "agentic", "org", "model", "packages"]

_CACHE_TYPES: tuple[CacheType, ...] = ("scan", "agentic", "org", "model", "packages")


def default_cache_root() -> Path:
    return Path.home() / ".aibom" / "cache"


def resolve_cache_root(root: Path | None = None) -> Path:
    return (root or default_cache_root()).expanduser()


def cache_dir(cache_type: CacheType, root: Path | None = None) -> Path:
    return resolve_cache_root(root) / cache_type


def cache_read_dirs(cache_type: CacheType, root: Path | None = None) -> list[Path]:
    base_root = resolve_cache_root(root)
    primary = cache_dir(cache_type, base_root)
    dirs: list[Path] = [primary]

    if cache_type == "scan":
        legacy = base_root
        if legacy != primary:
            dirs.append(legacy)
        return dirs

    if root is not None:
        return dirs

    legacy_map = {
        "agentic": Path.home() / ".cache" / "cisco-aibom" / "agentic",
        "org": Path.home() / ".cache" / "cisco-aibom" / "org-cache",
        "model": Path(user_cache_dir("aibom")) / "model_cache",
        "packages": Path.home() / ".cache" / "cisco-aibom" / "packages",
    }
    legacy = legacy_map.get(cache_type)
    if legacy and legacy != primary:
        dirs.append(legacy)
    return dirs


def ensure_cache_dir(cache_type: CacheType, root: Path | None = None) -> Path:
    directory = cache_dir(cache_type, root)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cache_types() -> tuple[CacheType, ...]:
    return _CACHE_TYPES
