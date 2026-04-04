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
import logging
import re
import tomllib
from pathlib import Path

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource
from .base import BaseScanner

_LOGGER = logging.getLogger(__name__)

_REQ_EDITABLE = re.compile(r"^\s*-e\s+([^\s#]+)")
_REQ_REL_PATH = re.compile(
    r"^\s*((?:\./|\.\./)[^\s#]+)\s*(?:#.*)?$"
)


class WorkspaceDepScanner(BaseScanner):
    name = "workspace_dep_scanner"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext,
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components: list[AIComponent] = []
        idx = context.file_index()

        for entry in idx.get(".toml", []):
            if entry.path.name != "pyproject.toml":
                continue
            components.extend(_scan_pyproject(entry.path))

        for entry in idx.get(".txt", []):
            if entry.path.name != "requirements.txt":
                continue
            components.extend(_scan_requirements_txt(entry.path))

        for entry in idx.get(".json", []):
            if entry.path.name != "package.json":
                continue
            components.extend(_scan_package_json(entry.path))

        for entry in idx.get(".mod", []):
            if entry.path.name != "go.mod":
                continue
            components.extend(_scan_go_mod(entry.path))

        return components, []


def _dep_component(
    *,
    name: str,
    file_path: str,
    line_number: int,
    local_path: str,
    ecosystem: str,
) -> AIComponent:
    return AIComponent(
        name=name,
        component_type=AIComponentType.DEPENDENCY,
        file_path=file_path,
        line_number=line_number,
        framework="",
        detection_source=DetectionSource.DEPENDENCY_MANIFEST,
        confidence=1.0,
        metadata={
            "local": True,
            "local_path": local_path,
            "ecosystem": ecosystem,
        },
    )


def _scan_pyproject(path: Path) -> list[AIComponent]:
    out: list[AIComponent] = []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _LOGGER.debug("Failed to parse %s", path, exc_info=True)
        return out

    base = path.parent
    tool = data.get("tool") if isinstance(data.get("tool"), dict) else {}

    poetry = tool.get("poetry") if isinstance(tool.get("poetry"), dict) else {}
    pdeps = poetry.get("dependencies") if isinstance(poetry.get("dependencies"), dict) else {}
    for pkg, spec in pdeps.items():
        if pkg == "python" or not isinstance(spec, dict):
            continue
        p = spec.get("path")
        if isinstance(p, str) and p.strip():
            resolved = (base / p).resolve()
            out.append(
                _dep_component(
                    name=str(pkg),
                    file_path=str(path),
                    line_number=0,
                    local_path=str(resolved),
                    ecosystem="python",
                )
            )

    groups = poetry.get("group") if isinstance(poetry.get("group"), dict) else {}
    for _gname, gdata in groups.items():
        if not isinstance(gdata, dict):
            continue
        gdeps = gdata.get("dependencies")
        if not isinstance(gdeps, dict):
            continue
        for pkg, spec in gdeps.items():
            if not isinstance(spec, dict):
                continue
            p = spec.get("path")
            if isinstance(p, str) and p.strip():
                resolved = (base / p).resolve()
                out.append(
                    _dep_component(
                        name=str(pkg),
                        file_path=str(path),
                        line_number=0,
                        local_path=str(resolved),
                        ecosystem="python",
                    )
                )

    uv = tool.get("uv") if isinstance(tool.get("uv"), dict) else {}
    sources = uv.get("sources") if isinstance(uv.get("sources"), dict) else {}
    for pkg, spec in sources.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("workspace") is True:
            resolved = _resolve_uv_workspace_member(base, str(pkg))
            out.append(
                _dep_component(
                    name=str(pkg),
                    file_path=str(path),
                    line_number=0,
                    local_path=str(resolved),
                    ecosystem="python",
                )
            )

    return out


def _resolve_uv_workspace_member(base: Path, pkg: str) -> Path:
    candidates = [
        base / pkg,
        base / "packages" / pkg,
        base.parent / pkg,
    ]
    for c in candidates:
        try:
            if c.exists():
                return c.resolve()
        except OSError:
            continue
    return base.resolve()


def _scan_requirements_txt(path: Path) -> list[AIComponent]:
    out: list[AIComponent] = []
    base = path.parent
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _REQ_EDITABLE.match(line)
        if m:
            ref = m.group(1).strip().strip('"').strip("'")
            if ref.startswith("./") or ref.startswith("../"):
                resolved = (base / ref).resolve()
                name = Path(ref).name.rstrip("/") or ref
                out.append(
                    _dep_component(
                        name=name,
                        file_path=str(path),
                        line_number=i,
                        local_path=str(resolved),
                        ecosystem="python",
                    )
                )
            continue
        m2 = _REQ_REL_PATH.match(line)
        if m2:
            ref = m2.group(1).strip().strip('"').strip("'")
            if ref.startswith("./") or ref.startswith("../"):
                resolved = (base / ref).resolve()
                name = Path(ref.rstrip("/")).name or ref
                out.append(
                    _dep_component(
                        name=name,
                        file_path=str(path),
                        line_number=i,
                        local_path=str(resolved),
                        ecosystem="python",
                    )
                )

    return out


def _scan_package_json(path: Path) -> list[AIComponent]:
    out: list[AIComponent] = []
    base = path.parent
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _LOGGER.debug("Failed to parse %s", path, exc_info=True)
        return out

    if not isinstance(data, dict):
        return out

    for section in (
        "dependencies",
        "devDependencies",
        "optionalDependencies",
        "peerDependencies",
    ):
        block = data.get(section)
        if not isinstance(block, dict):
            continue
        for pkg, ver in block.items():
            if not isinstance(ver, str):
                continue
            v = ver.strip()
            if v.startswith("file:"):
                rel = v[5:].lstrip()
            elif v.startswith("link:"):
                rel = v[5:].lstrip()
            else:
                continue
            resolved = (base / rel).resolve()
            out.append(
                _dep_component(
                    name=str(pkg),
                    file_path=str(path),
                    line_number=0,
                    local_path=str(resolved),
                    ecosystem="node",
                )
            )

    return out


def _scan_go_mod(path: Path) -> list[AIComponent]:
    out: list[AIComponent] = []
    base = path.parent
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped.startswith("replace "):
            continue
        if "=>" not in line:
            continue
        left_s, right_s = line.split("=>", 1)
        left = left_s.replace("replace", "", 1).strip()
        right = right_s.split("//", 1)[0].strip().strip('"')
        local_ref: str | None = None
        for tok in right.split():
            if tok.startswith("./") or tok.startswith("../"):
                local_ref = tok
                break
        if local_ref is None:
            continue
        resolved = (base / local_ref).resolve()
        name = left.split()[-1] if left else local_ref
        out.append(
            _dep_component(
                name=name,
                file_path=str(path),
                line_number=i,
                local_path=str(resolved),
                ecosystem="go",
            )
        )

    return out
