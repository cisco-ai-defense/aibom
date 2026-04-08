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

"""Cross-repo and IaC reasoning helpers for the agentic pipeline."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..cross_ref import CrossRefIndex, EnvVarEntry, build_env_index, build_package_index
from ..models import AIComponent
from ..models.enums import AIComponentType

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "AIComponent",
    "AIComponentType",
    "CrossRepoSummaryArgs",
    "ResolveEnvVarArgs",
    "ResolveIaCRefArgs",
    "cross_repo_summary_tool",
    "resolve_env_var_tool",
    "resolve_iac_ref_tool",
]

from ..utils.path_filter import should_skip_dir

_TF_DEFAULT_STR = re.compile(
    r"default\s*=\s*(\"([^\"\\]*(?:\\.[^\"\\]*)*)\"|'([^'\\]*(?:\\.[^'\\]*)*)')",
    re.MULTILINE,
)
_TF_DEFAULT_HEREDOC = re.compile(r"default\s*=\s*<<[-\w]*\n([\s\S]*?)\n\s*[-\w]*", re.MULTILINE)
_TF_LOCAL_ASSIGN_STR = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\"([^\"\\]*(?:\\.[^\"\\]*)*)\"|'([^'\\]*(?:\\.[^'\\]*)*)')",
    re.MULTILINE,
)
_HELM_VALUES_REF = re.compile(
    r"\{\{\s*\.Values\.([A-Za-z0-9_.-]+)\s*\}\}",
    re.IGNORECASE,
)
_ARM_PARAM_REF = re.compile(r"parameters\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", re.IGNORECASE)
_ARM_RESOURCE_REF = re.compile(r"reference\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", re.IGNORECASE)
_PY_ENV_REFS = [
    re.compile(r"os\.getenv\s*\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"),
    re.compile(r"os\.environ\.get\s*\(\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"os\.environ\s*\[\s*['\"]([^'\"]+)['\"]\s*\]"),
]
_JS_ENV_REFS = [
    re.compile(r"process\.env\.([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"process\.env\s*\[\s*['\"]([^'\"]+)['\"]\s*\]"),
]
_GO_ENV_REF = re.compile(r"os\.Getenv\s*\(\s*[\"]([^\"]+)[\"]\s*\)")


class ResolveEnvVarArgs(BaseModel):
    var_name: str = Field(description="Environment variable name to resolve")
    scan_paths: list[str] = Field(description="Paths to search for env var definitions")


class ResolveIaCRefArgs(BaseModel):
    ref_expression: str = Field(
        description="IaC reference expression (e.g., var.model_name, !Ref ModelParam)"
    )
    iac_type: str = Field(description="IaC type: terraform, cloudformation, arm, helm")
    scan_paths: list[str] = Field(description="Paths to search")


class CrossRepoSummaryArgs(BaseModel):
    scan_paths: list[str] = Field(description="All repo paths being scanned")


def _should_skip_path(path: Path) -> bool:
    return any(should_skip_dir(p) for p in path.parts)


def _iter_files(paths: list[str], suffixes: tuple[str, ...] | None = None) -> Iterator[Path]:
    for root in paths:
        base = Path(root)
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or _should_skip_path(p):
                continue
            if suffixes is not None and not str(p).endswith(suffixes):
                continue
            yield p


def _root_prefixes_for_file(file_path: str, roots: list[str]) -> list[str]:
    fp = Path(file_path)
    try:
        fp_r = fp.resolve()
    except OSError:
        fp_r = fp
    out: list[str] = []
    for r in roots:
        try:
            br = Path(r).resolve()
            fp_r.relative_to(br)
            out.append(r)
        except (OSError, ValueError):
            continue
    return out


def _extract_tf_block_body(text: str, var_name: str) -> str | None:
    token = f'variable "{var_name}"'
    idx = text.find(token)
    if idx < 0:
        return None
    brace = text.find("{", idx + len(token))
    if brace < 0:
        return None
    depth = 0
    i = brace
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : i]
        i += 1
    return None


def _first_string_default(block: str) -> str | None:
    m = _TF_DEFAULT_STR.search(block)
    if not m:
        m2 = _TF_DEFAULT_HEREDOC.search(block)
        if m2:
            return m2.group(1).strip()
        return None
    inner = m.group(2) or m.group(3) or ""
    return inner


def _resolve_terraform_var(var_name: str, paths: list[str]) -> dict[str, Any]:
    for path in _iter_files(paths, (".tf",)):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _LOGGER.debug("read tf failed %s: %s", path, exc)
            continue
        body = _extract_tf_block_body(text, var_name)
        if body is None:
            continue
        val = _first_string_default(body)
        if val is not None:
            return {
                "resolved": True,
                "value": val,
                "source_file": str(path),
                "source_type": "terraform_variable",
            }
    return {"resolved": False, "value": None, "source_file": None, "source_type": None}


def _resolve_terraform_local(local_name: str, paths: list[str]) -> dict[str, Any]:
    for path in _iter_files(paths, (".tf",)):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _LOGGER.debug("read tf failed %s: %s", path, exc)
            continue
        for m in _TF_LOCAL_ASSIGN_STR.finditer(text):
            name = m.group(1)
            if name != local_name:
                continue
            raw = m.group(2) or m.group(3) or ""
            return {
                "resolved": True,
                "value": raw,
                "source_file": str(path),
                "source_type": "terraform_local",
            }
    return {"resolved": False, "value": None, "source_file": None, "source_type": None}


def _parse_cfn_template(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            data = yaml.safe_load(raw)
    except (json.JSONDecodeError, yaml.YAMLError, UnicodeDecodeError):
        return None
    if not isinstance(data, Mapping):
        return None
    if (
        "AWSTemplateFormatVersion" not in data
        and "Transform" not in data
        and "Resources" not in data
    ):
        return None
    return dict(data)


def _resolve_cfn_ref(logical_id: str, paths: list[str]) -> dict[str, Any]:
    for path in _iter_files(paths, (".yaml", ".yml", ".json")):
        tmpl = _parse_cfn_template(path)
        if tmpl is None:
            continue
        params = tmpl.get("Parameters")
        if isinstance(params, Mapping) and logical_id in params:
            p = params[logical_id]
            if isinstance(p, Mapping):
                default = p.get("Default")
                return {
                    "resolved": default is not None,
                    "value": default,
                    "source_file": str(path),
                    "source_type": "cloudformation_parameter",
                    "logical_id": logical_id,
                }
        resources = tmpl.get("Resources")
        if isinstance(resources, Mapping) and logical_id in resources:
            r = resources[logical_id]
            rtype = r.get("Type") if isinstance(r, Mapping) else None
            return {
                "resolved": True,
                "value": None,
                "resource_type": rtype,
                "source_file": str(path),
                "source_type": "cloudformation_resource",
                "logical_id": logical_id,
            }
    return {
        "resolved": False,
        "value": None,
        "source_file": None,
        "source_type": None,
        "logical_id": logical_id,
    }


def _navigate_values(obj: Any, parts: list[str]) -> Any:
    cur: Any = obj
    for part in parts:
        if not part:
            continue
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _resolve_helm_value(key_path: str, paths: list[str]) -> dict[str, Any]:
    parts = [p for p in key_path.split(".") if p]
    for path in _iter_files(paths):
        if path.name != "values.yaml":
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, yaml.YAMLError, UnicodeDecodeError):
            continue
        val = _navigate_values(data, parts)
        if val is not None:
            return {
                "resolved": True,
                "value": val,
                "source_file": str(path),
                "source_type": "helm_values",
            }
    return {"resolved": False, "value": None, "source_file": None, "source_type": None}


def _parse_arm_template(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, Mapping):
        return None
    schema = str(data.get("$schema", "")).lower()
    keys = {k.lower() for k in data}
    if "schema.management.azure.com" not in schema and "resources" not in keys:
        return None
    return dict(data)


def _resolve_arm_parameter(param_name: str, paths: list[str]) -> dict[str, Any]:
    for path in _iter_files(paths, (".json",)):
        tmpl = _parse_arm_template(path)
        if tmpl is None:
            continue
        params = tmpl.get("parameters")
        if not isinstance(params, Mapping):
            continue
        block = params.get(param_name)
        if isinstance(block, Mapping) and "defaultValue" in block:
            return {
                "resolved": True,
                "value": block.get("defaultValue"),
                "source_file": str(path),
                "source_type": "arm_parameter",
            }
    return {"resolved": False, "value": None, "source_file": None, "source_type": None}


def _resolve_arm_resource(resource_name: str, paths: list[str]) -> dict[str, Any]:
    for path in _iter_files(paths, (".json",)):
        tmpl = _parse_arm_template(path)
        if tmpl is None:
            continue
        resources = tmpl.get("resources")
        if isinstance(resources, list):
            for res in resources:
                if not isinstance(res, Mapping):
                    continue
                if res.get("name") == resource_name or res.get("name") == f"[{resource_name}]":
                    return {
                        "resolved": True,
                        "value": res.get("type"),
                        "source_file": str(path),
                        "source_type": "arm_resource",
                        "resource_name": resource_name,
                    }
        elif isinstance(resources, Mapping) and resource_name in resources:
            res = resources[resource_name]
            return {
                "resolved": True,
                "value": res.get("type") if isinstance(res, Mapping) else None,
                "source_file": str(path),
                "source_type": "arm_resource",
                "resource_name": resource_name,
            }
    return {
        "resolved": False,
        "value": None,
        "source_file": None,
        "source_type": None,
        "resource_name": resource_name,
    }


def _parse_cfn_logical_id(ref: str) -> str | None:
    s = ref.strip()
    m = re.match(r"!Ref\s+([A-Za-z0-9_-]+)", s)
    if m:
        return m.group(1)
    m = re.match(r"Ref\s*:\s*([A-Za-z0-9_-]+)", s, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"Fn::Ref\s*:\s*([A-Za-z0-9_-]+)", s)
    if m:
        return m.group(1)
    m = re.search(r"\"Ref\"\s*:\s*\"([^\"]+)\"", s)
    if m:
        return m.group(1)
    m = re.search(r"'Ref'\s*:\s*'([^']+)'", s)
    if m:
        return m.group(1)
    return None


def _collect_code_env_refs(paths: list[str]) -> set[str]:
    refs: set[str] = set()
    exts = (
        ".py",
        ".pyw",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".mjs",
        ".cjs",
        ".go",
    )
    for path in _iter_files(paths):
        if not str(path).endswith(exts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if path.suffix == ".go":
            for m in _GO_ENV_REF.finditer(text):
                refs.add(m.group(1))
            continue
        if path.suffix in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
            for rx in _JS_ENV_REFS:
                for m in rx.finditer(text):
                    refs.add(m.group(1))
            continue
        for rx in _PY_ENV_REFS:
            for m in rx.finditer(text):
                refs.add(m.group(1))
    return refs


def resolve_env_var_tool(var_name: str, scan_paths: list[str]) -> str:
    try:
        index = build_env_index(scan_paths)
        entries = index.env.get(var_name)
        if not entries:
            return json.dumps(
                {
                    "resolved": False,
                    "var_name": var_name,
                    "value": None,
                    "source_file": None,
                    "source_type": None,
                    "message": "not found",
                }
            )
        first: EnvVarEntry = entries[0]
        alts = [
            {
                "value": e.value,
                "source_file": e.source_path,
                "source_type": e.source_type,
            }
            for e in entries[1:8]
        ]
        return json.dumps(
            {
                "resolved": True,
                "var_name": var_name,
                "value": first.value,
                "source_file": first.source_path,
                "source_type": first.source_type,
                "alternates": alts,
            }
        )
    except Exception as exc:
        _LOGGER.exception("resolve_env_var_tool failed")
        return json.dumps({"resolved": False, "error": str(exc), "var_name": var_name})


def resolve_iac_ref_tool(ref_expression: str, iac_type: str, scan_paths: list[str]) -> str:
    try:
        kind = iac_type.strip().lower()
        ref = ref_expression.strip()
        if kind == "terraform":
            if ref.startswith("var."):
                var_n = ref[4:].strip()
                out = _resolve_terraform_var(var_n, scan_paths)
                return json.dumps(out)
            if ref.startswith("local."):
                loc_n = ref[6:].strip()
                out = _resolve_terraform_local(loc_n, scan_paths)
                return json.dumps(out)
            if ref.startswith("module."):
                return json.dumps(
                    {
                        "resolved": False,
                        "reason": "module reference not deterministically resolvable",
                        "ref_expression": ref,
                        "iac_type": kind,
                    }
                )
            return json.dumps(
                {
                    "resolved": False,
                    "reason": "unsupported terraform reference shape",
                    "ref_expression": ref,
                    "message": "needs LLM reasoning",
                }
            )
        if kind == "cloudformation":
            lid = _parse_cfn_logical_id(ref)
            if not lid and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", ref):
                lid = ref
            if not lid:
                return json.dumps(
                    {
                        "resolved": False,
                        "reason": "could not parse logical id",
                        "ref_expression": ref,
                        "message": "needs LLM reasoning",
                    }
                )
            out = _resolve_cfn_ref(lid, scan_paths)
            return json.dumps(out)
        if kind == "arm":
            mp = _ARM_PARAM_REF.search(ref)
            if mp:
                out = _resolve_arm_parameter(mp.group(1), scan_paths)
                return json.dumps(out)
            mr = _ARM_RESOURCE_REF.search(ref)
            if mr:
                out = _resolve_arm_resource(mr.group(1), scan_paths)
                return json.dumps(out)
            return json.dumps(
                {
                    "resolved": False,
                    "reason": "unrecognized ARM expression",
                    "ref_expression": ref,
                    "message": "needs LLM reasoning",
                }
            )
        if kind == "helm":
            m = _HELM_VALUES_REF.search(ref)
            if m:
                key_path = m.group(1)
            else:
                key_path = ref
                if ref.lower().startswith(".values."):
                    key_path = ref[8:].strip()
            out = _resolve_helm_value(key_path, scan_paths)
            if not out.get("resolved"):
                out["message"] = "needs LLM reasoning"
            return json.dumps(out)
        return json.dumps(
            {
                "resolved": False,
                "error": f"unknown iac_type: {iac_type}",
                "ref_expression": ref,
            }
        )
    except Exception as exc:
        _LOGGER.exception("resolve_iac_ref_tool failed")
        return json.dumps({"resolved": False, "error": str(exc), "ref_expression": ref_expression})


def cross_repo_summary_tool(scan_paths: list[str]) -> str:
    try:
        env_index = build_env_index(scan_paths)
        shared_env: list[dict[str, Any]] = []
        for name, entries in env_index.env.items():
            roots: set[str] = set()
            for e in entries:
                for rp in _root_prefixes_for_file(e.source_path, scan_paths):
                    roots.add(rp)
            if len(roots) > 1:
                shared_env.append(
                    {
                        "name": name,
                        "paths": sorted(roots),
                        "definitions": len(entries),
                    }
                )
        shared_env.sort(key=lambda x: x["name"])

        pkg_roots: dict[str, set[str]] = {}
        for root in scan_paths:
            sub: CrossRefIndex = build_package_index([root])
            for pkg in sub.packages:
                pkg_roots.setdefault(pkg, set()).add(root)
        shared_pkgs = sorted([p for p, rs in pkg_roots.items() if len(rs) > 1])

        code_refs = _collect_code_env_refs(scan_paths)
        defined = set(env_index.env.keys())
        unresolved = sorted(code_refs - defined)

        return json.dumps(
            {
                "shared_env_vars": shared_env,
                "shared_packages": shared_pkgs,
                "unresolved_env_var_refs": unresolved,
            }
        )
    except Exception as exc:
        _LOGGER.exception("cross_repo_summary_tool failed")
        return json.dumps({"error": str(exc)})


class GetRepoComponentsArgs(BaseModel):
    repo_name: str = Field(description="Repository name (as shown in the overview)")


class _CoordResolveEnvVarArgs(BaseModel):
    var_name: str = Field(description="Environment variable name to resolve")


class _CoordResolveIaCRefArgs(BaseModel):
    ref_expression: str = Field(
        description="IaC reference expression (e.g., var.model_name, !Ref ModelParam)"
    )
    iac_type: str = Field(description="IaC type: terraform, cloudformation, arm, helm")


def build_cross_repo_tools(
    per_repo_results: dict[str, dict[str, Any]],
    scan_paths: list[str],
) -> list:
    """Build LangChain StructuredTools for the cross-repo coordinator agent.

    The ``get_repo_components`` tool is a closure over *per_repo_results* so
    the LLM can pull full component data for any repo on demand.

    The env-var and IaC tools close over *scan_paths* so the LLM does not
    need to supply them.
    """
    from langchain_core.tools import StructuredTool

    def _get_repo_components(repo_name: str) -> str:
        data = per_repo_results.get(repo_name)
        if data is None:
            for key in per_repo_results:
                if repo_name in key or key in repo_name:
                    data = per_repo_results[key]
                    break
        if data is None:
            return json.dumps({
                "error": f"Repo '{repo_name}' not found. "
                f"Available: {list(per_repo_results.keys())}",
            })
        components = data.get("components", [])
        serialized = []
        for c in components:
            if hasattr(c, "model_dump"):
                serialized.append(c.model_dump(mode="json"))
            elif isinstance(c, dict):
                serialized.append(c)
        return json.dumps({
            "repo": repo_name,
            "component_count": len(serialized),
            "components": serialized,
            "unresolved_env_vars": data.get("_unresolved_env_vars", []),
        }, default=str)

    def _resolve_env(var_name: str) -> str:
        return resolve_env_var_tool(var_name, scan_paths)

    def _resolve_iac(ref_expression: str, iac_type: str) -> str:
        return resolve_iac_ref_tool(ref_expression, iac_type, scan_paths)

    return [
        StructuredTool.from_function(
            name="get_repo_components",
            description=(
                "Get full component data for a specific repository. "
                "Returns all components with complete metadata, file paths, "
                "model names, and unresolved env vars for that repo."
            ),
            func=_get_repo_components,
            args_schema=GetRepoComponentsArgs,
        ),
        StructuredTool.from_function(
            name="resolve_env_var",
            description=(
                "Search for an environment variable definition across all "
                "scanned repositories. Checks .env files, docker-compose, "
                "Terraform tfvars, Helm values.yaml, and other config files."
            ),
            func=_resolve_env,
            args_schema=_CoordResolveEnvVarArgs,
        ),
        StructuredTool.from_function(
            name="resolve_iac_ref",
            description=(
                "Resolve an Infrastructure-as-Code reference (Terraform var, "
                "Helm value, CloudFormation Ref, ARM parameter) to its "
                "concrete value across all scanned repositories."
            ),
            func=_resolve_iac,
            args_schema=_CoordResolveIaCRefArgs,
        ),
    ]
