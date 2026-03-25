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
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pathspec import PathSpec

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource, Severity
from .base import BaseScanner
from .file_cache import read_text_cached

_LOGGER = logging.getLogger(__name__)

try:
    from detect_secrets import settings as _detect_secrets_settings  # noqa: F401
    from detect_secrets.core.scan import scan_file as _ds_scan_file

    _HAS_DETECT_SECRETS = True
except ImportError:
    _HAS_DETECT_SECRETS = False
    _ds_scan_file = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AIKeyPattern:
    regex: re.Pattern[str]
    name: str
    severity: Severity
    confidence: float
    line_predicate: Optional[Callable[[str], bool]] = None


def _line_has_cohere(line: str) -> bool:
    return "cohere" in line.lower()


def _line_has_together(line: str) -> bool:
    return "together" in line.lower()


def _line_has_mistral(line: str) -> bool:
    return "mistral" in line.lower()


def _line_has_azure_openai(line: str) -> bool:
    low = line.lower()
    return "azure" in low and "openai" in low


AI_KEY_PATTERNS: list[AIKeyPattern] = [
    AIKeyPattern(
        re.compile(r"\bsk-(?!ant-)[a-zA-Z0-9]{20,}\b"),
        "openai_api_key",
        Severity.HIGH,
        0.88,
    ),
    AIKeyPattern(
        re.compile(r"\bsk-ant-[a-zA-Z0-9-]{20,}\b"),
        "anthropic_api_key",
        Severity.HIGH,
        0.9,
    ),
    AIKeyPattern(
        re.compile(r"\bhf_[a-zA-Z0-9]{20,}\b"),
        "huggingface_api_key",
        Severity.HIGH,
        0.88,
    ),
    AIKeyPattern(
        re.compile(r"\b[a-zA-Z0-9]{40}\b"),
        "cohere_api_key",
        Severity.MEDIUM,
        0.42,
        _line_has_cohere,
    ),
    AIKeyPattern(
        re.compile(r"\bAIza[a-zA-Z0-9_-]{35}\b"),
        "google_ai_api_key",
        Severity.HIGH,
        0.87,
    ),
    AIKeyPattern(
        re.compile(r"\br8_[a-zA-Z0-9]{40}\b"),
        "replicate_api_token",
        Severity.HIGH,
        0.9,
    ),
    AIKeyPattern(
        re.compile(r"\b[a-fA-F0-9]{64}\b"),
        "together_api_key",
        Severity.MEDIUM,
        0.48,
        _line_has_together,
    ),
    AIKeyPattern(
        re.compile(r"\b[a-zA-Z0-9]{32}\b"),
        "mistral_api_key",
        Severity.MEDIUM,
        0.45,
        _line_has_mistral,
    ),
    AIKeyPattern(
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "aws_access_key_id",
        Severity.HIGH,
        0.92,
    ),
    AIKeyPattern(
        re.compile(r"\b[a-fA-F0-9]{32}\b"),
        "azure_openai_key",
        Severity.MEDIUM,
        0.5,
        _line_has_azure_openai,
    ),
]


from ..utils.path_filter import should_skip_dir

_SKIP_NAME_SUFFIXES = (".pyc", ".lock", ".min.js")


def _slugify_type(label: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
    return s or "secret"


def _path_should_skip(rel_posix: str, name: str) -> bool:
    parts = rel_posix.split("/")
    if any(should_skip_dir(p) for p in parts):
        return True
    low = name.lower()
    if low.endswith(_SKIP_NAME_SUFFIXES):
        return True
    return False


def _load_exclude_spec(context: ScanContext) -> Optional[PathSpec]:
    if not context.exclude_patterns:
        return None
    return PathSpec.from_lines("gitwildmatch", context.exclude_patterns)


def _is_excluded(abs_file: Path, root: Path, spec: Optional[PathSpec]) -> bool:
    if not spec:
        return False
    try:
        rel = abs_file.relative_to(root).as_posix()
    except ValueError:
        rel = abs_file.as_posix()
    return spec.match_file(rel)


def _iter_scan_files(context: ScanContext) -> Iterator[tuple[Path, Path]]:
    idx = context.file_index()
    if idx:
        for entries in idx.values():
            for entry in entries:
                try:
                    rel = entry.path.relative_to(entry.root).as_posix()
                except ValueError:
                    rel = entry.path.name
                if _path_should_skip(rel, entry.path.name):
                    continue
                yield entry.path, entry.root
        return

    spec = _load_exclude_spec(context)
    for root_str in context.paths:
        root = Path(root_str).resolve()
        if root.is_file():
            if _path_should_skip(root.name, root.name):
                continue
            if _is_excluded(root, root.parent, spec):
                continue
            yield root, root.parent
            continue
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dp = Path(dirpath)
            rel_root = dp.relative_to(root).as_posix() if dp != root else ""
            dirnames[:] = [
                d
                for d in dirnames
                if not should_skip_dir(d)
                and not _path_should_skip(
                    "/".join(filter(None, [rel_root, d])), d
                )
            ]
            for fn in filenames:
                file_path = dp / fn
                rel = file_path.relative_to(root).as_posix()
                if _path_should_skip(rel, fn):
                    continue
                if _is_excluded(file_path, root, spec):
                    continue
                yield file_path, root


def _read_text_lines(path: Path) -> Optional[list[str]]:
    try:
        text = read_text_cached(path)
    except OSError as e:
        _LOGGER.debug("Unreadable file %s: %s", path, e)
        return None
    if "\x00" in text[:8192]:
        return None
    return text.splitlines()


def _regex_scan_line(
    line: str, line_number: int, file_path: str, out: dict[tuple[str, int], AIComponent]
) -> None:
    for pat in AI_KEY_PATTERNS:
        pred = pat.line_predicate
        if pred is not None and not pred(line):
            continue
        if not pat.regex.search(line):
            continue
        is_low_confidence = pat.confidence < 0.6
        comp = AIComponent(
            name=pat.name,
            component_type=AIComponentType.SECRET,
            file_path=file_path,
            line_number=line_number,
            detection_source=DetectionSource.CODE_ANALYSIS,
            confidence=pat.confidence,
            needs_agentic=is_low_confidence,
            agentic_hint=f"Generic hex pattern for {pat.name}; needs context verification" if is_low_confidence else "",
            description=f"Potential {pat.name} detected",
            text=None,
            metadata={
                "secret_type": pat.name,
                "detection_method": "regex",
            },
        )
        _merge_component(out, comp)


def _merge_component(bucket: dict[tuple[str, int], AIComponent], comp: AIComponent) -> None:
    key = (comp.file_path, comp.line_number)
    prev = bucket.get(key)
    if prev is None or comp.confidence > prev.confidence:
        bucket[key] = comp


def _detect_secrets_scan_file(abs_path: Path, out: dict[tuple[str, int], AIComponent]) -> None:
    if not _HAS_DETECT_SECRETS or _ds_scan_file is None:
        return
    try:
        for secret in _ds_scan_file(str(abs_path)):
            stype = _slugify_type(secret.type)
            line_no = int(secret.line_number or 0) or 1
            file_path = str(abs_path)
            comp = AIComponent(
                name=stype,
                component_type=AIComponentType.SECRET,
                file_path=file_path,
                line_number=line_no,
                detection_source=DetectionSource.CODE_ANALYSIS,
                confidence=0.72,
                description=f"Potential {stype} detected",
                text=None,
                metadata={
                    "secret_type": stype,
                    "detection_method": "detect_secrets",
                },
            )
            _merge_component(out, comp)
    except Exception:
        _LOGGER.debug("detect_secrets failed for %s", abs_path, exc_info=True)


def _run_gitleaks(source: Path, out: dict[tuple[str, int], AIComponent]) -> None:
    if not shutil.which("gitleaks"):
        return
    report_path: Optional[str] = None
    raw = ""
    try:
        fd, report_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        subprocess.run(
            [
                "gitleaks",
                "detect",
                "--source",
                str(source),
                "--report-format",
                "json",
                "--report-path",
                report_path,
                "--no-git",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        raw = Path(report_path).read_text(encoding="utf-8")
    except Exception:
        _LOGGER.debug("gitleaks failed for %s", source, exc_info=True)
        return
    finally:
        if report_path:
            try:
                Path(report_path).unlink(missing_ok=True)
            except OSError:
                pass

    try:
        findings: Any = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(findings, list):
        return
    for item in findings:
        if not isinstance(item, dict):
            continue
        file_path = item.get("File") or item.get("file")
        line = item.get("StartLine") or item.get("Line") or item.get("line")
        rule = item.get("RuleID") or item.get("Description") or item.get("reason") or "secret"
        if not file_path or line is None:
            continue
        try:
            line_no = int(line)
        except (TypeError, ValueError):
            continue
        stype = _slugify_type(str(rule))
        abs_path = str(Path(str(file_path)).resolve())
        comp = AIComponent(
            name=stype,
            component_type=AIComponentType.SECRET,
            file_path=abs_path,
            line_number=line_no,
            detection_source=DetectionSource.CODE_ANALYSIS,
            confidence=0.65,
            description=f"Potential {stype} detected",
            text=None,
            metadata={
                "secret_type": stype,
                "detection_method": "gitleaks",
            },
        )
        _merge_component(out, comp)


def _run_trufflehog(source: Path, out: dict[tuple[str, int], AIComponent]) -> None:
    if not shutil.which("trufflehog"):
        return
    try:
        r = subprocess.run(
            ["trufflehog", "filesystem", str(source), "--json", "--no-update"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        text = r.stdout or ""
    except Exception:
        _LOGGER.debug("trufflehog failed for %s", source, exc_info=True)
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        det = item.get("DetectorName") or item.get("detector_name") or "secret"
        stype = _slugify_type(str(det))
        meta = item.get("SourceMetadata") or {}
        if isinstance(meta, dict):
            data = meta.get("Data") or meta
        else:
            data = {}
        if not isinstance(data, dict):
            data = {}
        file_path = data.get("path") or data.get("filename") or data.get("File")
        line_no = data.get("line") or data.get("Line")
        if not file_path:
            continue
        try:
            ln = int(line_no) if line_no is not None else 1
        except (TypeError, ValueError):
            ln = 1
        abs_path = str(Path(str(file_path)).resolve())
        comp = AIComponent(
            name=stype,
            component_type=AIComponentType.SECRET,
            file_path=abs_path,
            line_number=ln,
            detection_source=DetectionSource.CODE_ANALYSIS,
            confidence=0.65,
            description=f"Potential {stype} detected",
            text=None,
            metadata={
                "secret_type": stype,
                "detection_method": "trufflehog",
            },
        )
        _merge_component(out, comp)


class SecretDetector(BaseScanner):
    name = "secret_detector"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        bucket: dict[tuple[str, int], AIComponent] = {}
        seen_files: set[Path] = set()

        for file_path, _root in _iter_scan_files(context):
            resolved = file_path.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            fp_str = str(resolved)

            lines = _read_text_lines(resolved)
            if lines is None:
                continue
            for i, line in enumerate(lines, start=1):
                _regex_scan_line(line, i, fp_str, bucket)

            if _HAS_DETECT_SECRETS:
                _detect_secrets_scan_file(resolved, bucket)

        for root_str in context.paths:
            src = Path(root_str).resolve()
            if src.exists():
                try:
                    _run_gitleaks(src, bucket)
                except Exception:
                    _LOGGER.debug("gitleaks wrapper error", exc_info=True)
                try:
                    _run_trufflehog(src, bucket)
                except Exception:
                    _LOGGER.debug("trufflehog wrapper error", exc_info=True)

        return list(bucket.values()), []
