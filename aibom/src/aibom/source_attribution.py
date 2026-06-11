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

"""Source-attribution capture for the AI BOM.

When the CLI scans a repository or container image, a downstream backend
ingests the resulting BOM and projects each discovered component as an
inventory asset. To keep an asset stable across re-scans, the backend derives a
deterministic identity from three source-attribution fields the CLI emits on
every BOM:

* ``source_kind`` — ``git`` for a git working tree, ``container_image`` for an
  OCI image, or ``local-path`` for a plain directory that is not a checkout.
* ``source_ref_canonical`` — a single canonical string per source (normalized
  git remote URL or registry path) so the same source spelled different ways
  resolves to one identity.
* ``source_ref_version`` — a point-in-time version stamp (git commit SHA or
  image manifest digest).

All three are produced deterministically at scan time; the customer's CI is not
in the loop, so the CLI is the source of truth.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

# Source-kind values.
SOURCE_KIND_GIT = "git"
SOURCE_KIND_CONTAINER_IMAGE = "container_image"
SOURCE_KIND_LOCAL_PATH = "local-path"

_GIT_TIMEOUT_S = 5


def _is_git_working_tree(path: Path) -> bool:
    """Return True when *path* resolves inside a git working tree."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def detect_source_kind(
    source: str, *, is_container_image: Optional[bool] = None
) -> str:
    """Detect the ``source_kind`` for a scan target.

    Args:
        source: The scan target — a filesystem path (git working tree or plain
            directory) or an OCI image reference.
        is_container_image: When the caller already knows the target is an OCI
            image (e.g. the CLI resolved it via the container-extraction path),
            pass ``True`` to short-circuit detection. ``None`` means "decide
            from the path".

    Returns:
        ``"git"`` if the target is a git working tree, ``"container_image"`` if
        it is an OCI image, otherwise ``"local-path"``.
    """
    if is_container_image:
        return SOURCE_KIND_CONTAINER_IMAGE

    path = Path(source)
    if path.exists() and _is_git_working_tree(path):
        return SOURCE_KIND_GIT

    return SOURCE_KIND_LOCAL_PATH
