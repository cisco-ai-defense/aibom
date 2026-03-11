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

"""JavaScript / TypeScript source parser using a Node.js subprocess.

Invokes ``js_parser/parse.js`` via ``node`` and deserialises the JSON
output into the same :class:`CodeAnalysisResult` used by the Python parser.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .structures import (
    AssignmentObservation,
    CallObservation,
    ClassDefObservation,
    CodeAnalysisResult,
    DecoratorObservation,
)

LOGGER = logging.getLogger(__name__)

_PARSER_SCRIPT = Path(__file__).resolve().parent / "js_parser" / "parse.js"

JS_EXTENSIONS = frozenset({".js", ".jsx", ".ts", ".tsx"})


def is_js_file(path: Path) -> bool:
    return path.suffix.lower() in JS_EXTENSIONS


def _node_available() -> bool:
    return shutil.which("node") is not None


def parse_js_source_code(
    file_path: str,
    source_code: str,  # noqa: ARG001 – kept for interface parity with parse_source_code
    *,
    timeout: int = 30,
) -> CodeAnalysisResult:
    """Parse a JS/TS file and return observations.

    Falls back to an empty result on any error (missing Node.js,
    parse failure, etc.) so the rest of the pipeline can proceed.
    """
    empty = CodeAnalysisResult(file_path=file_path)

    if not _node_available():
        LOGGER.warning("Node.js not found – skipping JS/TS file %s", file_path)
        return empty

    if not _PARSER_SCRIPT.is_file():
        LOGGER.warning("JS parser script not found at %s", _PARSER_SCRIPT)
        return empty

    try:
        proc = subprocess.run(
            ["node", str(_PARSER_SCRIPT), file_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        LOGGER.warning("JS parser timed out for %s", file_path)
        return empty
    except FileNotFoundError:
        LOGGER.warning("Node.js not found – skipping JS/TS file %s", file_path)
        return empty

    if proc.returncode != 0:
        LOGGER.debug("JS parser stderr for %s: %s", file_path, proc.stderr)
        return empty

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        LOGGER.warning("JS parser returned invalid JSON for %s", file_path)
        return empty

    return _deserialise(data, file_path)


def _deserialise(data: dict, fallback_path: str) -> CodeAnalysisResult:
    """Convert the raw JSON dict into a ``CodeAnalysisResult``."""
    fp = data.get("file_path", fallback_path)
    result = CodeAnalysisResult(file_path=fp)

    for raw in data.get("assignments", []):
        call_data = raw.get("call", {})
        call = CallObservation(
            qualified_name=call_data.get("qualified_name", ""),
            arguments=call_data.get("arguments", {}),
            line_number=call_data.get("line_number", 0),
            raw_code=call_data.get("raw_code", ""),
        )
        result.assignments.append(
            AssignmentObservation(
                target_qualified_name=raw.get("target_qualified_name", ""),
                call=call,
                line_number=raw.get("line_number", 0),
            )
        )

    for raw in data.get("calls", []):
        result.calls.append(
            CallObservation(
                qualified_name=raw.get("qualified_name", ""),
                arguments=raw.get("arguments", {}),
                line_number=raw.get("line_number", 0),
                raw_code=raw.get("raw_code", ""),
            )
        )

    for raw in data.get("decorators", []):
        result.decorators.append(
            DecoratorObservation(
                decorator_qualified_name=raw.get("decorator_qualified_name", ""),
                decorated_function_name=raw.get("decorated_function_name", ""),
                line_number=raw.get("line_number", 0),
                instance_variable=raw.get("instance_variable"),
            )
        )

    for raw in data.get("class_defs", []):
        result.class_defs.append(
            ClassDefObservation(
                class_name=raw.get("class_name", ""),
                qualified_name=raw.get("qualified_name"),
                base_classes=raw.get("base_classes", []),
                line_number=raw.get("line_number", 0),
                aibom_annotation=raw.get("aibom_annotation"),
            )
        )

    result.imports = data.get("imports", [])
    return result
