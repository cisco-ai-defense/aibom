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

import json
import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models.enums import AIComponentType
from .models.scan import AIComponent
from .scanners.dependency_scanner import (
    AI_PACKAGES,
    _GO_REQUIRE,
    _normalize_pypi_name,
    _REQUIREMENT_NAME,
)

_ENV_SUBST = re.compile(r"\$\{([^}]+)\}")

_COMPOSE_NAMES = frozenset(
    {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}
)


@dataclass
class EnvVarEntry:
    name: str
    value: str
    source_type: str
    source_path: str = ""


@dataclass
class CrossRefIndex:
    env: dict[str, list[EnvVarEntry]] = field(default_factory=dict)
    packages: set[str] = field(default_factory=set)


def _add_env(
    index: CrossRefIndex,
    name: str,
    value: str,
    source_type: str,
    source_path: str,
) -> None:
    entry = EnvVarEntry(
        name=name,
        value=value,
        source_type=source_type,
        source_path=source_path,
    )
    index.env.setdefault(name, []).append(entry)


def _strip_quotes(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def _parse_dotenv(content: str, source_path: str, index: CrossRefIndex) -> None:
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        val = _strip_quotes(rest)
        _add_env(index, key, val, "dotenv", source_path)


_TFVAR_ASSIGN = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*(?:#.*)?$"
)


def _parse_tfvars(content: str, source_path: str, index: CrossRefIndex) -> None:
    for line in content.splitlines():
        m = _TFVAR_ASSIGN.match(line)
        if not m:
            continue
        key = m.group(1)
        raw_val = m.group(2).strip()
        if raw_val.startswith('"') and raw_val.endswith('"'):
            val = _strip_quotes(raw_val)
        elif raw_val.startswith("'") and raw_val.endswith("'"):
            val = _strip_quotes(raw_val)
        else:
            val = raw_val.split("#", 1)[0].strip()
        _add_env(index, key, val, "terraform-tfvars", source_path)


def _walk_yaml_leaves(
    obj: Any,
    index: CrossRefIndex,
    source_path: str,
    source_type: str,
) -> None:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                _add_env(index, str(k), "" if v is None else str(v), source_type, source_path)
            elif isinstance(v, Mapping):
                _walk_yaml_leaves(v, index, source_path, source_type)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, (str, int, float, bool)) or item is None:
                        _add_env(
                            index,
                            str(k),
                            "" if item is None else str(item),
                            source_type,
                            source_path,
                        )
                    else:
                        _walk_yaml_leaves(item, index, source_path, source_type)


def _parse_compose_env(
    env_block: Any,
    source_path: str,
    index: CrossRefIndex,
) -> None:
    if env_block is None:
        return
    if isinstance(env_block, Mapping):
        for k, v in env_block.items():
            _add_env(
                index,
                str(k),
                "" if v is None else str(v),
                "docker-compose",
                source_path,
            )
        return
    if isinstance(env_block, list):
        for item in env_block:
            if not isinstance(item, str) or "=" not in item:
                continue
            ek, _, ev = item.partition("=")
            _add_env(index, ek.strip(), ev.strip(), "docker-compose", source_path)


def _parse_docker_compose(path: Path, index: CrossRefIndex) -> None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return
    if not isinstance(data, Mapping):
        return
    services = data.get("services")
    if not isinstance(services, Mapping):
        return
    rel = str(path)
    for svc in services.values():
        if not isinstance(svc, Mapping):
            continue
        _parse_compose_env(svc.get("environment"), rel, index)


def _parse_k8s_configmap(path: Path, index: CrossRefIndex) -> None:
    try:
        docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return
    rel = str(path)
    for doc in docs:
        if not isinstance(doc, Mapping):
            continue
        if str(doc.get("kind", "")).lower() != "configmap":
            continue
        data = doc.get("data")
        if not isinstance(data, Mapping):
            continue
        for k, v in data.items():
            _add_env(
                index,
                str(k),
                "" if v is None else str(v),
                "k8s-configmap",
                rel,
            )


def _parse_helm_values(path: Path, index: CrossRefIndex) -> None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return
    _walk_yaml_leaves(data, index, str(path), "helm-values")


def _scan_env_paths(paths: Iterable[str]) -> CrossRefIndex:
    index = CrossRefIndex()
    for root in paths:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            name = path.name
            rel = str(path)
            if name == ".env":
                try:
                    _parse_dotenv(path.read_text(encoding="utf-8"), rel, index)
                except (OSError, UnicodeDecodeError):
                    continue
            elif name in _COMPOSE_NAMES:
                _parse_docker_compose(path, index)
            elif name == "terraform.tfvars" or name.endswith(".tfvars"):
                try:
                    _parse_tfvars(path.read_text(encoding="utf-8"), rel, index)
                except (OSError, UnicodeDecodeError):
                    continue
            elif name == "values.yaml":
                _parse_helm_values(path, index)
            elif name.endswith((".yaml", ".yml")):
                _parse_k8s_configmap(path, index)
    return index


def build_env_index(paths: Iterable[str]) -> CrossRefIndex:
    return _scan_env_paths(paths)


def _pypi_candidate_ai(name: str) -> bool:
    norm = _normalize_pypi_name(name)
    return norm in AI_PACKAGES["pypi"]


def _npm_candidate_ai(name: str) -> bool:
    return name in AI_PACKAGES["npm"]


def _go_candidate_ai(module: str) -> bool:
    if module in AI_PACKAGES["go"]:
        return True
    return module == "github.com/openai/openai-go"


def _parse_requirements(content: str, index: CrossRefIndex) -> None:
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _REQUIREMENT_NAME.match(s)
        if not m:
            continue
        raw_name = m.group(1)
        if _pypi_candidate_ai(raw_name):
            index.packages.add(_normalize_pypi_name(raw_name))


def _parse_package_json(path: Path, index: CrossRefIndex) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    deps = data.get("dependencies")
    if isinstance(deps, Mapping):
        for name in deps:
            if _npm_candidate_ai(name):
                index.packages.add(name)
    dev = data.get("devDependencies")
    if isinstance(dev, Mapping):
        for name in dev:
            if _npm_candidate_ai(name):
                index.packages.add(name)


def _pyproject_dep_strings(project: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    deps = project.get("dependencies")
    if isinstance(deps, list):
        for item in deps:
            if isinstance(item, str):
                out.append(item)
    opt = project.get("optional-dependencies")
    if isinstance(opt, Mapping):
        for group in opt.values():
            if isinstance(group, list):
                for item in group:
                    if isinstance(item, str):
                        out.append(item)
    return out


def _parse_pyproject(path: Path, index: CrossRefIndex) -> None:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return
    project = data.get("project")
    if not isinstance(project, Mapping):
        return
    for spec in _pyproject_dep_strings(project):
        m = _REQUIREMENT_NAME.match(spec.strip())
        if not m:
            continue
        raw_name = m.group(1)
        if _pypi_candidate_ai(raw_name):
            index.packages.add(_normalize_pypi_name(raw_name))


def _parse_go_mod(content: str, index: CrossRefIndex) -> None:
    in_require = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("require ("):
            in_require = True
            continue
        if in_require and stripped == ")":
            in_require = False
            continue
        if in_require:
            m = _GO_REQUIRE.match(stripped)
            if m:
                mod = m.group(1)
                if _go_candidate_ai(mod):
                    index.packages.add(mod)
        else:
            if stripped.startswith("require "):
                rest = stripped[len("require ") :].strip()
                m = _GO_REQUIRE.match(rest)
                if m:
                    mod = m.group(1)
                    if _go_candidate_ai(mod):
                        index.packages.add(mod)


def _scan_package_paths(paths: Iterable[str]) -> CrossRefIndex:
    index = CrossRefIndex()
    for root in paths:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            name = path.name
            if name == "requirements.txt":
                try:
                    _parse_requirements(path.read_text(encoding="utf-8"), index)
                except (OSError, UnicodeDecodeError):
                    continue
            elif name == "package.json":
                _parse_package_json(path, index)
            elif name == "pyproject.toml":
                _parse_pyproject(path, index)
            elif name == "go.mod":
                try:
                    _parse_go_mod(path.read_text(encoding="utf-8"), index)
                except (OSError, UnicodeDecodeError):
                    continue
    return index


def build_package_index(paths: Iterable[str]) -> CrossRefIndex:
    return _scan_package_paths(paths)


def _env_lookup(index: CrossRefIndex, key: str) -> str | None:
    entries = index.env.get(key)
    if not entries:
        return None
    return entries[0].value


def _substitute_placeholders(model_name: str, index: CrossRefIndex) -> str:
    def repl(match: re.Match[str]) -> str:
        k = match.group(1).strip()
        v = _env_lookup(index, k)
        return v if v is not None else match.group(0)

    return _ENV_SUBST.sub(repl, model_name)


def _try_registry(model_name: str) -> dict[str, Any] | None:
    """Attempt registry lookup; returns None on import failure."""
    try:
        from .scanners.model_detector import _registry_lookup
        return _registry_lookup(model_name)
    except Exception:  # noqa: BLE001
        return None


def resolve_components(
    components: Iterable[AIComponent],
    env_index: CrossRefIndex,
) -> list[AIComponent]:
    out: list[AIComponent] = []
    for c in components:
        new_c = c
        meta = dict(c.metadata) if c.metadata else {}

        env_key = meta.get("env") or meta.get("config_key")
        resolved_from: EnvVarEntry | None = None

        if isinstance(env_key, str):
            entries = env_index.env.get(env_key)
            if entries:
                resolved_from = entries[0]
                val = resolved_from.value

                provenance = {
                    "resolved_from": resolved_from.source_type,
                    "resolved_source_file": resolved_from.source_path,
                    "resolved_env_var": env_key,
                    "resolved_value": val,
                }
                meta.update(provenance)

                if c.component_type == AIComponentType.VECTOR_STORE:
                    meta["index_name"] = val
                    new_c = c.model_copy(update={
                        "confidence": max(c.confidence, 0.8),
                        "needs_agentic": False,
                        "metadata": meta,
                    })
                elif c.model_name is None and val:
                    new_c = c.model_copy(update={
                        "model_name": val,
                        "metadata": meta,
                    })
            elif c.needs_agentic:
                meta["unresolved_env_var"] = env_key
                new_c = c.model_copy(update={
                    "agentic_hint": (
                        f"env var {env_key} used as {meta.get('env_context', 'unknown')}; "
                        f"not found in any config source — may require "
                        f"additional repos or runtime configuration"
                    ),
                    "metadata": meta,
                })

        mn = new_c.model_name
        if isinstance(mn, str) and "${" in mn:
            sub = _substitute_placeholders(mn, env_index)
            if sub != mn:
                meta_sub = dict(new_c.metadata) if new_c.metadata else {}
                meta_sub["placeholder_resolved"] = True
                new_c = new_c.model_copy(update={
                    "model_name": sub,
                    "metadata": meta_sub,
                })
                mn = sub

        if resolved_from and mn and not _ENV_SUBST.search(mn):
            reg_hit = _try_registry(mn)
            if reg_hit:
                reg_meta = dict(new_c.metadata) if new_c.metadata else {}
                reg_meta["registry_source"] = reg_hit.get("source", "model_catalog")
                reg_meta["provider"] = reg_hit.get("provider", "unknown")
                new_c = new_c.model_copy(update={
                    "confidence": max(new_c.confidence, 0.85),
                    "needs_agentic": False,
                    "agentic_hint": "",
                    "metadata": reg_meta,
                })
            else:
                agentic_meta = dict(new_c.metadata) if new_c.metadata else {}
                agentic_meta["registry_source"] = "none"
                new_c = new_c.model_copy(update={
                    "confidence": max(new_c.confidence, 0.5),
                    "needs_agentic": True,
                    "agentic_hint": (
                        f"Resolved '{mn}' from env var "
                        f"{resolved_from.name} ({resolved_from.source_type}), "
                        f"but not found in model registry"
                    ),
                    "metadata": agentic_meta,
                })

        out.append(new_c)
    return out


# ---------------------------------------------------------------------------
# Cross-repo dependency detection
# ---------------------------------------------------------------------------


@dataclass
class ExternalRepoDep:
    """A dependency that references code outside the current repository."""

    name: str
    source_file: str
    dep_type: str  # "git", "path", "editable"
    url_or_path: str
    subdirectory: str = ""
    branch: str = ""
    tag: str = ""
    escapes_root: bool = False


_GIT_URL_RE = re.compile(
    r"git\+(?:https?|ssh|git)://[^\s#]+|git@[^\s#:]+"
)
_POETRY_GIT_RE = re.compile(
    r"""^\[tool\.poetry\.(?:dependencies|dev-dependencies|group\.\w+\.dependencies)\]""",
    re.MULTILINE,
)
_UV_SOURCES_HEADER_RE = re.compile(
    r"""^\[tool\.uv\.sources\]""", re.MULTILINE,
)


def _parse_poetry_git_deps(
    data: dict[str, Any], source_path: str
) -> list[ExternalRepoDep]:
    """Extract Poetry git and path dependencies from parsed pyproject.toml."""
    deps: list[ExternalRepoDep] = []
    sections = [
        data.get("tool", {}).get("poetry", {}).get("dependencies", {}),
        data.get("tool", {}).get("poetry", {}).get("dev-dependencies", {}),
    ]
    groups = data.get("tool", {}).get("poetry", {}).get("group", {})
    if isinstance(groups, Mapping):
        for grp in groups.values():
            if isinstance(grp, Mapping):
                sections.append(grp.get("dependencies", {}))

    for section in sections:
        if not isinstance(section, Mapping):
            continue
        for pkg_name, spec in section.items():
            if not isinstance(spec, Mapping):
                continue
            git_url = spec.get("git")
            local_path = spec.get("path")
            if git_url:
                deps.append(ExternalRepoDep(
                    name=pkg_name,
                    source_file=source_path,
                    dep_type="git",
                    url_or_path=git_url,
                    subdirectory=spec.get("subdirectory", ""),
                    branch=spec.get("branch", ""),
                    tag=spec.get("tag", spec.get("rev", "")),
                ))
            elif local_path:
                deps.append(ExternalRepoDep(
                    name=pkg_name,
                    source_file=source_path,
                    dep_type="path",
                    url_or_path=local_path,
                ))
    return deps


def _parse_uv_sources(
    data: dict[str, Any], source_path: str
) -> list[ExternalRepoDep]:
    """Extract uv [tool.uv.sources] git and path references."""
    deps: list[ExternalRepoDep] = []
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    if not isinstance(sources, Mapping):
        return deps
    for pkg_name, spec in sources.items():
        if not isinstance(spec, Mapping):
            continue
        git_url = spec.get("git")
        local_path = spec.get("path")
        if git_url:
            deps.append(ExternalRepoDep(
                name=pkg_name,
                source_file=source_path,
                dep_type="git",
                url_or_path=git_url,
                subdirectory=spec.get("subdirectory", ""),
                branch=spec.get("branch", spec.get("rev", "")),
                tag=spec.get("tag", ""),
            ))
        elif local_path:
            deps.append(ExternalRepoDep(
                name=pkg_name,
                source_file=source_path,
                dep_type="path",
                url_or_path=local_path,
            ))
    return deps


_REQ_GIT_RE = re.compile(
    r"""^(?:-e\s+)?git\+(?P<url>https?://[^\s@#]+|ssh://[^\s@#]+|git@[^\s@#]+)"""
    r"""(?:@(?P<ref>[^\s#]+))?"""
    r"""(?:#egg=(?P<egg>[^\s&]+))?""",
)
_REQ_PATH_EDITABLE_RE = re.compile(
    r"""^-e\s+(?P<path>[./][^\s]+)"""
)


def _parse_requirements_git(
    content: str, source_path: str
) -> list[ExternalRepoDep]:
    deps: list[ExternalRepoDep] = []
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _REQ_GIT_RE.match(s)
        if m:
            deps.append(ExternalRepoDep(
                name=m.group("egg") or "",
                source_file=source_path,
                dep_type="git",
                url_or_path=m.group("url"),
                branch=m.group("ref") or "",
            ))
            continue
        m = _REQ_PATH_EDITABLE_RE.match(s)
        if m:
            deps.append(ExternalRepoDep(
                name="",
                source_file=source_path,
                dep_type="editable",
                url_or_path=m.group("path"),
            ))
    return deps


def _parse_package_json_git(
    data: dict[str, Any], source_path: str
) -> list[ExternalRepoDep]:
    deps: list[ExternalRepoDep] = []
    for section_key in ("dependencies", "devDependencies"):
        section = data.get(section_key)
        if not isinstance(section, Mapping):
            continue
        for pkg_name, spec in section.items():
            if not isinstance(spec, str):
                continue
            if spec.startswith("git+") or spec.startswith("github:"):
                deps.append(ExternalRepoDep(
                    name=pkg_name,
                    source_file=source_path,
                    dep_type="git",
                    url_or_path=spec,
                ))
            elif spec.startswith("file:"):
                deps.append(ExternalRepoDep(
                    name=pkg_name,
                    source_file=source_path,
                    dep_type="path",
                    url_or_path=spec[len("file:"):],
                ))
    return deps


_GO_REPLACE_RE = re.compile(
    r"""^replace\s+(\S+)\s+=>\s+(\S+)(?:\s+(\S+))?$""", re.MULTILINE
)


def _parse_go_mod_replace(
    content: str, source_path: str
) -> list[ExternalRepoDep]:
    deps: list[ExternalRepoDep] = []
    for m in _GO_REPLACE_RE.finditer(content):
        original = m.group(1)
        replacement = m.group(2)
        if replacement.startswith("./") or replacement.startswith("../"):
            deps.append(ExternalRepoDep(
                name=original,
                source_file=source_path,
                dep_type="path",
                url_or_path=replacement,
            ))
        elif not replacement.startswith("."):
            deps.append(ExternalRepoDep(
                name=original,
                source_file=source_path,
                dep_type="git",
                url_or_path=replacement,
            ))
    return deps


def _flag_escaping_paths(
    deps: list[ExternalRepoDep], repo_roots: list[Path]
) -> None:
    """Set ``escapes_root=True`` for path deps that leave all repo roots."""
    for dep in deps:
        if dep.dep_type not in ("path", "editable"):
            continue
        dep_path = Path(dep.url_or_path)
        if dep_path.is_absolute():
            dep.escapes_root = True
            continue
        manifest_dir = Path(dep.source_file).parent
        resolved = (manifest_dir / dep_path).resolve()
        inside = False
        for root in repo_roots:
            try:
                resolved.relative_to(root.resolve())
                inside = True
                break
            except ValueError:
                continue
        if not inside:
            dep.escapes_root = True


def detect_external_repo_deps(paths: Iterable[str]) -> list[ExternalRepoDep]:
    """Scan manifests under *paths* for cross-repo git/path dependencies."""
    deps: list[ExternalRepoDep] = []
    repo_roots = [Path(p) for p in paths]

    for root_str in paths:
        root = Path(root_str)
        if not root.exists():
            continue
        for fpath in root.rglob("*"):
            if not fpath.is_file():
                continue
            name = fpath.name
            sp = str(fpath)

            if name == "pyproject.toml":
                try:
                    data = tomllib.loads(fpath.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
                    continue
                deps.extend(_parse_poetry_git_deps(data, sp))
                deps.extend(_parse_uv_sources(data, sp))

            elif name in ("requirements.txt", "requirements-dev.txt"):
                try:
                    content = fpath.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                deps.extend(_parse_requirements_git(content, sp))

            elif name == "package.json":
                try:
                    data = json.loads(fpath.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                deps.extend(_parse_package_json_git(data, sp))

            elif name == "go.mod":
                try:
                    content = fpath.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                deps.extend(_parse_go_mod_replace(content, sp))

    _flag_escaping_paths(deps, repo_roots)
    return deps
