from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cache_paths import CacheType, cache_dir, cache_read_dirs, resolve_cache_root
from .incremental import OrgCache, _repo_bucket_key


def _unique_files(directories: list[Path], pattern: str) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob(pattern)):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return files


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _format_size_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    return f"{round(size_bytes / 1024, 1)} KB"


def _agentic_subtype(payload: dict[str, Any]) -> str:
    if "_tier_cache_version" in payload:
        return "tier"
    if "_batch_cache_version" in payload:
        return "batch"
    if "_cross_repo_cache_version" in payload:
        return "cross_repo"
    if "cached_component" in payload:
        return "component"
    return "unknown"


def _scan_entries(cache_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in _unique_files(cache_read_dirs("scan", cache_root), "*.json"):
        payload = _load_json(path)
        if not payload:
            continue
        components = payload.get("components", [])
        relationships = payload.get("relationships", [])
        flags = payload.get("_agentic_risk_flags", [])
        entries.append(
            {
                "id": payload.get("_cache_key", path.stem),
                "path": path,
                "cached_at": payload.get("_cached_at", "unknown"),
                "subtype": "scan",
                "size": _format_size_bytes(path.stat().st_size),
                "detail": (
                    f"{len(components)} comps, {len(relationships)} rels, "
                    f"{len(flags)} flags"
                ),
                "payload": payload,
            }
        )
    return entries


def _agentic_entries(cache_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in _unique_files(cache_read_dirs("agentic", cache_root), "*.json"):
        payload = _load_json(path)
        if not payload:
            continue
        subtype = _agentic_subtype(payload)
        detail = "cached component snapshot"
        if subtype == "tier":
            detail = (
                f"{len(payload.get('tier_enriched', []))} enriched, "
                f"{len(payload.get('tier_new', []))} new, "
                f"{len(payload.get('tier_rels', []))} rels, "
                f"{len(payload.get('tier_flags', []))} flags"
            )
        elif subtype == "batch":
            detail = (
                f"{len(payload.get('batch_new', []))} new, "
                f"{len(payload.get('batch_rels', []))} rels, "
                f"{len(payload.get('batch_flags', []))} flags"
            )
        elif subtype == "cross_repo":
            detail = (
                f"{len(payload.get('cross_repo_rels', []))} rels, "
                f"{len(payload.get('cross_repo_flags', []))} flags"
            )
        entries.append(
            {
                "id": path.stem,
                "path": path,
                "cached_at": payload.get("_cached_at", "unknown"),
                "subtype": subtype,
                "size": _format_size_bytes(path.stat().st_size),
                "detail": detail,
                "payload": payload,
            }
        )
    return entries


def _org_entries(cache_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    base_dirs = cache_read_dirs("org", cache_root)
    repo_dirs = _unique_files(base_dirs, "*")
    for repo_dir in repo_dirs:
        if not repo_dir.is_dir():
            continue
        for path in sorted(repo_dir.glob("*.json")):
            payload = _load_json(path)
            if not payload:
                continue
            sources = payload.get("sources", [])
            repo_path = ""
            if isinstance(sources, list) and sources:
                repo_path = str(sources[0].get("path", ""))
            component_count = sum(
                len(source.get("components", []))
                for source in sources
                if isinstance(source, dict)
            )
            entries.append(
                {
                    "id": f"{repo_path}@{path.stem}" if repo_path else path.stem,
                    "path": path,
                    "cached_at": "unknown",
                    "subtype": "org",
                    "size": _format_size_bytes(path.stat().st_size),
                    "detail": f"{len(sources)} sources, {component_count} comps",
                    "repo_path": repo_path,
                    "sha": path.stem,
                    "payload": payload,
                }
            )
    return entries


def _model_entries(cache_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in _unique_files(cache_read_dirs("model", cache_root), "*.json"):
        payload = _load_json(path)
        if not payload:
            continue
        models = payload.get("models", {})
        entries.append(
            {
                "id": path.name,
                "path": path,
                "cached_at": payload.get("_ts", "unknown"),
                "subtype": "model",
                "size": _format_size_bytes(path.stat().st_size),
                "detail": f"{len(models)} models",
                "payload": payload,
            }
        )
    return entries


def _package_entries(cache_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for base_dir in cache_read_dirs("packages", cache_root):
        if not base_dir.exists():
            continue
        for ecosystem_dir in sorted(base_dir.iterdir()):
            if not ecosystem_dir.is_dir():
                continue
            for path in sorted(ecosystem_dir.glob("*.json")):
                payload = _load_json(path)
                if not payload:
                    continue
                entry_id = f"{ecosystem_dir.name}/{path.stem}"
                entries.append(
                    {
                        "id": entry_id,
                        "path": path,
                        "cached_at": "unknown",
                        "subtype": ecosystem_dir.name,
                        "size": _format_size_bytes(path.stat().st_size),
                        "detail": payload.get("summary", "") or payload.get("name", ""),
                        "payload": payload,
                    }
                )
    return entries


def list_cache_entries(cache_type: CacheType, cache_root: Path | None = None) -> list[dict[str, Any]]:
    root = resolve_cache_root(cache_root)
    if cache_type == "scan":
        return _scan_entries(root)
    if cache_type == "agentic":
        return _agentic_entries(root)
    if cache_type == "org":
        return _org_entries(root)
    if cache_type == "model":
        return _model_entries(root)
    if cache_type == "packages":
        return _package_entries(root)
    raise ValueError(f"Unsupported cache type: {cache_type}")


def _resolve_prefix(entries: list[dict[str, Any]], entry_ref: str) -> dict[str, Any]:
    exact = [entry for entry in entries if entry["id"] == entry_ref]
    if exact:
        return exact[0]
    matches = [entry for entry in entries if str(entry["id"]).startswith(entry_ref)]
    if not matches:
        raise FileNotFoundError(f"No cache entry matches '{entry_ref}'")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous cache entry reference '{entry_ref}'")
    return matches[0]


def get_cache_entry(
    cache_type: CacheType,
    entry_ref: str,
    *,
    cache_root: Path | None = None,
    sha: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    root = resolve_cache_root(cache_root)

    if cache_type in {"scan", "agentic", "model", "packages"}:
        entry = _resolve_prefix(list_cache_entries(cache_type, root), entry_ref)
        payload = entry["payload"]
        if cache_type == "model" and model_id:
            models = payload.get("models", {})
            if model_id not in models:
                raise FileNotFoundError(
                    f"Model id '{model_id}' not present in {entry['id']}"
                )
            entry = {
                **entry,
                "detail": f"{model_id}: {models[model_id]}",
                "payload": {model_id: models[model_id]},
            }
        return entry

    if cache_type == "org":
        repo_path = str(Path(entry_ref).expanduser())
        resolved_sha = sha or OrgCache._get_head_sha(repo_path)
        if not resolved_sha:
            raise FileNotFoundError(
                "Could not resolve a commit SHA for the requested org cache entry"
            )
        candidate_dirs = cache_read_dirs("org", root)
        for base_dir in candidate_dirs:
            path = base_dir / _repo_bucket_key(repo_path) / f"{resolved_sha}.json"
            payload = _load_json(path)
            if payload:
                sources = payload.get("sources", [])
                component_count = sum(
                    len(source.get("components", []))
                    for source in sources
                    if isinstance(source, dict)
                )
                return {
                    "id": f"{repo_path}@{resolved_sha}",
                    "path": path,
                    "cached_at": "unknown",
                    "subtype": "org",
                    "size": _format_size_bytes(path.stat().st_size),
                    "detail": f"{len(sources)} sources, {component_count} comps",
                    "repo_path": repo_path,
                    "sha": resolved_sha,
                    "payload": payload,
                }
        raise FileNotFoundError(
            f"No org cache entry matches '{repo_path}' at sha '{resolved_sha}'"
        )

    raise ValueError(f"Unsupported cache type: {cache_type}")
