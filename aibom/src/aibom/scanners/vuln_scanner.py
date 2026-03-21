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
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource, Severity
from .base import BaseScanner
from .dependency_scanner import DependencyScanner

_LOGGER = logging.getLogger(__name__)

vuln_provider_registry: list[type["BaseVulnProvider"]] = []

_OSV_ECOSYSTEM: dict[str, str] = {
    "pypi": "PyPI",
    "npm": "npm",
    "go": "Go",
    "maven": "Maven",
    "cargo": "crates.io",
    "crates.io": "crates.io",
    "rubygems": "RubyGems",
    "nuget": "NuGet",
}

_DISPLAY_ECOSYSTEM: dict[str, str] = {
    "pypi": "PyPI",
    "npm": "npm",
    "go": "Go",
    "maven": "Maven",
    "cargo": "crates.io",
    "rubygems": "RubyGems",
    "nuget": "NuGet",
}

_SEVERITY_WEIGHTS: dict[Severity, int] = {
    Severity.CRITICAL: 25,
    Severity.HIGH: 20,
    Severity.MEDIUM: 10,
    Severity.LOW: 5,
    Severity.INFO: 1,
}


class Vulnerability(BaseModel):
    id: str
    summary: str
    severity: Severity
    cvss_score: float = 0.0
    affected_package: str
    affected_version: str
    fixed_version: str = ""
    source: str
    url: str = ""


class PackageRef(BaseModel):
    name: str
    version: str
    ecosystem: str


def _package_ref_key(ref: PackageRef) -> str:
    return f"{ref.ecosystem}:{ref.name}@{ref.version}"


def _normalize_internal_ecosystem(raw: str) -> str:
    return raw.strip().lower()


def _to_display_ecosystem(internal: str) -> str:
    return _DISPLAY_ECOSYSTEM.get(_normalize_internal_ecosystem(internal), internal)


def _to_osv_ecosystem(ecosystem: str) -> Optional[str]:
    eco = _normalize_internal_ecosystem(ecosystem)
    if eco in _OSV_ECOSYSTEM:
        return _OSV_ECOSYSTEM[eco]
    if ecosystem in _OSV_ECOSYSTEM.values():
        return ecosystem
    return _OSV_ECOSYSTEM.get(eco)


def _parse_severity_label(label: str) -> Severity:
    s = label.strip().lower()
    if s in ("critical", "crit"):
        return Severity.CRITICAL
    if s in ("high",):
        return Severity.HIGH
    if s in ("medium", "med", "moderate"):
        return Severity.MEDIUM
    if s in ("low",):
        return Severity.LOW
    return Severity.INFO


def _max_severity(a: Severity, b: Severity) -> Severity:
    return a if a.numeric >= b.numeric else b


def _osv_fixed_version(vuln: dict[str, Any]) -> str:
    for aff in vuln.get("affected") or []:
        if not isinstance(aff, dict):
            continue
        for r in aff.get("ranges") or []:
            if not isinstance(r, dict):
                continue
            for ev in r.get("events") or []:
                if isinstance(ev, dict) and "fixed" in ev:
                    fx = ev["fixed"]
                    if fx is not None:
                        return str(fx)
    return ""


def _osv_severity_and_score(vuln: dict[str, Any]) -> tuple[Severity, float]:
    best: Severity = Severity.INFO
    score = 0.0
    for entry in vuln.get("severity") or []:
        if not isinstance(entry, dict):
            continue
        raw_score = entry.get("score")
        if raw_score is not None:
            try:
                sc = float(str(raw_score).split()[0])
                score = max(score, sc)
                if sc >= 9.0:
                    best = _max_severity(best, Severity.CRITICAL)
                elif sc >= 7.0:
                    best = _max_severity(best, Severity.HIGH)
                elif sc >= 4.0:
                    best = _max_severity(best, Severity.MEDIUM)
                elif sc > 0:
                    best = _max_severity(best, Severity.LOW)
            except (TypeError, ValueError):
                continue
    if best == Severity.INFO and vuln.get("database_specific"):
        ds = vuln["database_specific"]
        if isinstance(ds, dict):
            sev = ds.get("severity") or ds.get("cvss_severity")
            if isinstance(sev, str):
                best = _max_severity(best, _parse_severity_label(sev))
    return best, score


def _osv_references(vuln: dict[str, Any]) -> str:
    for ref in vuln.get("references") or []:
        if isinstance(ref, dict):
            u = ref.get("url")
            if isinstance(u, str) and u:
                return u
    return ""


def _grype_severity(v: dict[str, Any]) -> tuple[Severity, float]:
    label = v.get("severity")
    if isinstance(label, str):
        return _parse_severity_label(label), 0.0
    cvss = v.get("cvss")
    if isinstance(cvss, list) and cvss:
        first = cvss[0]
        if isinstance(first, dict):
            m = first.get("metrics")
            if isinstance(m, dict):
                bm = m.get("baseMetrics")
                if isinstance(bm, dict):
                    b = bm.get("baseScore")
                    if b is not None:
                        try:
                            sc = float(b)
                            return _parse_severity_label(
                                str(bm.get("baseSeverity", "MEDIUM"))
                            ), sc
                        except (TypeError, ValueError):
                            pass
    return Severity.INFO, 0.0


def _grype_fixed_version(v: dict[str, Any]) -> str:
    fix = v.get("fix")
    if isinstance(fix, dict):
        versions = fix.get("versions")
        if isinstance(versions, list) and versions:
            return str(versions[0])
    return ""


def _grype_url(v: dict[str, Any]) -> str:
    urls = v.get("urls")
    if isinstance(urls, list) and urls:
        u = urls[0]
        if isinstance(u, str):
            return u
    return ""


def _grype_type_to_ecosystem(type_name: str) -> str:
    t = type_name.lower()
    mapping = {
        "python": "PyPI",
        "npm": "npm",
        "go-module": "Go",
        "go": "Go",
        "java-archive": "Maven",
        "jenkins-plugin": "Maven",
        "rust-crate": "crates.io",
        "ruby-gem": "RubyGems",
        "dotnet": "NuGet",
    }
    return mapping.get(t, type_name)


class BaseVulnProvider(ABC):
    name: str = ""
    supported_ecosystems: set[str] = set()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            vuln_provider_registry.append(cls)

    @abstractmethod
    def query(self, package: str, version: str, ecosystem: str) -> list[Vulnerability]:
        ...

    @abstractmethod
    def batch_query(
        self, packages: list[PackageRef]
    ) -> dict[str, list[Vulnerability]]:
        ...


class OsvProvider(BaseVulnProvider):
    name = "osv"
    supported_ecosystems = {
        "PyPI",
        "npm",
        "Go",
        "Maven",
        "crates.io",
        "RubyGems",
        "NuGet",
    }
    _BATCH_URL = "https://api.osv.dev/v1/querybatch"

    def query(self, package: str, version: str, ecosystem: str) -> list[Vulnerability]:
        ref = PackageRef(name=package, version=version, ecosystem=ecosystem)
        return self.batch_query([ref]).get(_package_ref_key(ref), [])

    def batch_query(
        self, packages: list[PackageRef]
    ) -> dict[str, list[Vulnerability]]:
        result: dict[str, list[Vulnerability]] = {}
        query_entries: list[tuple[str, PackageRef]] = []
        queries: list[dict[str, Any]] = []
        for ref in packages:
            osv_eco = _to_osv_ecosystem(ref.ecosystem)
            if not osv_eco:
                continue
            key = _package_ref_key(ref)
            query_entries.append((key, ref))
            queries.append(
                {
                    "package": {"name": ref.name, "ecosystem": osv_eco},
                    "version": ref.version,
                },
            )
        if not queries:
            return result
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(self._BATCH_URL, json={"queries": queries})
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            _LOGGER.warning("OSV batch query failed: %s", e)
            return result
        results = data.get("results") or []
        for i, block in enumerate(results):
            if i >= len(query_entries):
                break
            key, ref = query_entries[i]
            vulns: list[Vulnerability] = []
            for vuln in block.get("vulns") or []:
                if not isinstance(vuln, dict):
                    continue
                vid = str(vuln.get("id", ""))
                if not vid:
                    continue
                sev, cvss = _osv_severity_and_score(vuln)
                details = vuln.get("summary") or vuln.get("details") or ""
                vulns.append(
                    Vulnerability(
                        id=vid,
                        summary=str(details)[:2000],
                        severity=sev,
                        cvss_score=cvss,
                        affected_package=ref.name,
                        affected_version=ref.version,
                        fixed_version=_osv_fixed_version(vuln),
                        source="osv",
                        url=_osv_references(vuln),
                    ),
                )
            result[key] = vulns
        return result


class GrypeProvider(BaseVulnProvider):
    name = "grype"
    supported_ecosystems = set(OsvProvider.supported_ecosystems)

    def __init__(self, sbom_path: str) -> None:
        self._sbom_path = sbom_path

    def query(self, package: str, version: str, ecosystem: str) -> list[Vulnerability]:
        ref = PackageRef(name=package, version=version, ecosystem=ecosystem)
        return self.batch_query([ref]).get(_package_ref_key(ref), [])

    def batch_query(
        self, packages: list[PackageRef]
    ) -> dict[str, list[Vulnerability]]:
        result: dict[str, list[Vulnerability]] = {
            _package_ref_key(p): [] for p in packages
        }
        if not shutil.which("grype"):
            return result
        try:
            proc = subprocess.run(
                ["grype", f"sbom:{self._sbom_path}", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            _LOGGER.warning("Grype execution failed: %s", e)
            return result
        if proc.returncode != 0:
            _LOGGER.warning("Grype exited %s: %s", proc.returncode, proc.stderr[:500])
            return result
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            _LOGGER.warning("Grype JSON parse error: %s", e)
            return result
        want = {_package_ref_key(p): p for p in packages}
        for match in data.get("matches") or []:
            if not isinstance(match, dict):
                continue
            vuln = match.get("vulnerability")
            art = match.get("artifact")
            if not isinstance(vuln, dict) or not isinstance(art, dict):
                continue
            name = str(art.get("name") or "")
            version = str(art.get("version") or "")
            eco = _grype_type_to_ecosystem(str(art.get("type") or ""))
            ref = PackageRef(name=name, version=version, ecosystem=eco)
            key = _package_ref_key(ref)
            if key not in want:
                continue
            vid = str(vuln.get("id") or "")
            if not vid:
                continue
            sev, cvss = _grype_severity(vuln)
            result[key].append(
                Vulnerability(
                    id=vid,
                    summary=str(vuln.get("description") or vid)[:2000],
                    severity=sev,
                    cvss_score=cvss,
                    affected_package=name,
                    affected_version=version,
                    fixed_version=_grype_fixed_version(vuln),
                    source="grype",
                    url=_grype_url(vuln),
                ),
            )
        return result


def _components_to_package_refs(
    components: list[AIComponent],
) -> tuple[list[PackageRef], dict[str, AIComponent]]:
    refs: list[PackageRef] = []
    by_key: dict[str, AIComponent] = {}
    for comp in components:
        internal = comp.metadata.get("ecosystem")
        if not isinstance(internal, str):
            continue
        ver = comp.sdk_version
        if not ver:
            spec = comp.metadata.get("version_spec")
            ver = str(spec).strip() if spec else "*"
        if not ver:
            ver = "*"
        eco = _to_display_ecosystem(internal)
        ref = PackageRef(name=comp.name, version=ver, ecosystem=eco)
        key = _package_ref_key(ref)
        if key not in by_key:
            by_key[key] = comp
            refs.append(ref)
    return refs, by_key


def _component_from_config_dict(item: dict[str, Any]) -> AIComponent:
    name = str(item["name"])
    raw_eco = str(item.get("ecosystem", "pypi"))
    ver = str(item.get("version") or item.get("sdk_version") or "*")
    return AIComponent(
        name=name,
        component_type=AIComponentType.DEPENDENCY,
        file_path=str(item.get("file_path", "")),
        line_number=int(item.get("line_number", 0) or 0),
        sdk_version=ver if ver != "*" else None,
        detection_source=DetectionSource.DEPENDENCY_MANIFEST,
        metadata={
            "ecosystem": _normalize_internal_ecosystem(raw_eco),
            "version_spec": str(item.get("version_spec", ver)),
            "manifest": str(item.get("manifest", "")),
        },
    )


def _resolve_packages(
    context: ScanContext,
) -> tuple[list[PackageRef], dict[str, AIComponent]]:
    raw = context.config.get("detected_packages") or []
    by_key: dict[str, PackageRef] = {}
    comp_by_key: dict[str, AIComponent] = {}

    if raw:
        for item in raw:
            if isinstance(item, AIComponent):
                if item.component_type != AIComponentType.DEPENDENCY:
                    continue
                refs_one, cmap = _components_to_package_refs([item])
                for r in refs_one:
                    k = _package_ref_key(r)
                    if k not in by_key:
                        by_key[k] = r
                        comp_by_key[k] = cmap[k]
            elif isinstance(item, PackageRef):
                comp = AIComponent(
                    name=item.name,
                    component_type=AIComponentType.DEPENDENCY,
                    file_path="",
                    line_number=0,
                    sdk_version=item.version if item.version != "*" else None,
                    detection_source=DetectionSource.DEPENDENCY_MANIFEST,
                    metadata={
                        "ecosystem": _normalize_internal_ecosystem(item.ecosystem),
                        "version_spec": item.version,
                    },
                )
                k = _package_ref_key(item)
                if k not in by_key:
                    by_key[k] = item
                    comp_by_key[k] = comp
            elif isinstance(item, dict) and item.get("name"):
                comp = _component_from_config_dict(item)
                refs_one, cmap = _components_to_package_refs([comp])
                for r in refs_one:
                    k = _package_ref_key(r)
                    if k not in by_key:
                        by_key[k] = r
                        comp_by_key[k] = cmap[k]
        return list(by_key.values()), comp_by_key

    dep = DependencyScanner()
    comps, _ = dep.scan(context)
    return _components_to_package_refs(comps)


class VulnScanner(BaseScanner):
    name = "vuln_scanner"

    def supports(self, context: ScanContext) -> bool:
        if not context.paths:
            return False
        if context.config.get("dependencies") is False:
            return False
        return True

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        refs, key_to_comp = _resolve_packages(context)
        if not refs:
            return [], []

        providers: list[BaseVulnProvider] = [OsvProvider()]
        sbom_path = context.config.get("sbom_path") or context.config.get(
            "cyclonedx_sbom"
        )
        if isinstance(sbom_path, str) and sbom_path.strip() and shutil.which("grype"):
            providers.append(GrypeProvider(sbom_path.strip()))

        merged: dict[str, list[Vulnerability]] = {}
        for prov in providers:
            batch = prov.batch_query(refs)
            for k, vulns in batch.items():
                merged.setdefault(k, []).extend(vulns)

        out: list[AIComponent] = []
        for ref in refs:
            key = _package_ref_key(ref)
            base = key_to_comp.get(key)
            vulns = merged.get(key, [])
            if not vulns:
                continue
            max_sev = Severity.INFO
            for v in vulns:
                max_sev = _max_severity(max_sev, v.severity)
            weight = _SEVERITY_WEIGHTS.get(max_sev, 1)
            desc = f"{len(vulns)} known vulnerabilities in {ref.name} {ref.version}"
            if base:
                meta = dict(base.metadata)
                file_path = base.file_path
                line_no = base.line_number
                det_src = base.detection_source
                name = base.name
                sdk_ver = base.sdk_version
                framework = base.framework
            else:
                meta = {
                    "ecosystem": _normalize_internal_ecosystem(ref.ecosystem),
                    "version_spec": ref.version,
                }
                file_path = ""
                line_no = 0
                det_src = DetectionSource.API
                name = ref.name
                sdk_ver = ref.version if ref.version != "*" else None
                framework = ""
            meta["vulnerabilities"] = [v.model_dump(mode="json") for v in vulns]
            meta["risk_flag"] = {
                "flag": f"vulnerable_dependency:{ref.name}",
                "severity": max_sev.value,
                "weight": weight,
                "description": desc,
            }
            out.append(
                AIComponent(
                    name=name,
                    component_type=AIComponentType.DEPENDENCY,
                    file_path=file_path,
                    line_number=line_no,
                    framework=framework,
                    detection_source=det_src,
                    description=desc,
                    sdk_version=sdk_ver,
                    metadata=meta,
                ),
            )
        return out, []
