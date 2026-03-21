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

import re
from typing import IO
from xml.etree import ElementTree as ET

from ..models import AIComponent, RiskFlag, ScanResult, Severity
from .base import BaseReporter


def _junit_threshold(result: ScanResult) -> Severity:
    raw = result.metadata.get("junit_failure_threshold")
    if raw is None:
        return Severity.MEDIUM
    if isinstance(raw, Severity):
        return raw
    try:
        return Severity(str(raw).lower())
    except ValueError:
        return Severity.MEDIUM


def _flag_matches_component(flag: RiskFlag, comp: AIComponent) -> bool:
    if flag.file_path != comp.file_path:
        return False
    if flag.line_number == 0:
        return True
    return flag.line_number == comp.line_number


def _safe_name(prefix: str, key: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", key, flags=re.UNICODE)
    s = s.strip("_") or "item"
    return f"{prefix}_{s}"[:200]


class JunitReporter(BaseReporter):
    name = "junit"
    file_extension = ".xml"

    def render(self, result: ScanResult, output: IO[str]) -> None:
        threshold = _junit_threshold(result)
        flags = result.risk.flags
        components = result.all_components

        flagged: set[str] = set()
        for comp in components:
            if any(_flag_matches_component(f, comp) for f in flags):
                flagged.add(comp.instance_id)

        root = ET.Element("testsuites")
        suite = ET.SubElement(root, "testsuite", name="aibom")

        failures = 0
        idx = 0
        for f in flags:
            idx += 1
            tc = ET.SubElement(
                suite,
                "testcase",
                classname="aibom.risk",
                name=_safe_name("risk", f"{f.flag}_{idx}"),
                time="0",
            )
            if f.severity >= threshold:
                failures += 1
                msg = (
                    f"{f.flag}: {f.description} "
                    f"(severity={f.severity.value})"
                )
                fl = ET.SubElement(
                    tc,
                    "failure",
                    message=msg,
                    type=f.severity.value,
                )
                fl.text = msg

        for comp in components:
            if comp.instance_id in flagged:
                continue
            ET.SubElement(
                suite,
                "testcase",
                classname="aibom.component",
                name=_safe_name("component", comp.instance_id),
                time="0",
            )

        tests = len(flags) + sum(
            1 for c in components if c.instance_id not in flagged
        )
        suite.set("tests", str(tests))
        suite.set("failures", str(failures))
        suite.set("errors", "0")

        ET.ElementTree(root).write(
            output,
            encoding="unicode",
            xml_declaration=True,
        )
