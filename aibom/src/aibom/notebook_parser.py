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

"""Extract Python source code from Jupyter notebooks for analysis.

Reads ``.ipynb`` files, extracts all ``code`` cells, and concatenates
them into a single virtual Python source string that can be fed through
the existing LibCST parser.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

LOGGER = logging.getLogger(__name__)


def extract_code_from_notebook(notebook_path: Union[str, Path]) -> str:
    """Read a Jupyter notebook and return concatenated Python code cells.

    Each cell is separated by a blank line so that line numbers roughly
    correspond to the original notebook structure.  IPython magic lines
    (``%`` / ``!``) are commented out to avoid LibCST parse errors.

    Args:
        notebook_path: Path to the ``.ipynb`` file.

    Returns:
        A string containing all code cells concatenated together.
    """
    notebook_path = Path(notebook_path)
    try:
        data = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Failed to read notebook %s: %s", notebook_path, exc)
        return ""

    cells = data.get("cells", [])
    code_blocks: list[str] = []

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source_lines = cell.get("source", [])
        if isinstance(source_lines, list):
            source = "".join(source_lines)
        else:
            source = str(source_lines)

        cleaned_lines: list[str] = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("%", "!")):
                cleaned_lines.append(f"# MAGIC: {line}")
            else:
                cleaned_lines.append(line)

        code_blocks.append("\n".join(cleaned_lines))

    return "\n\n".join(code_blocks)
