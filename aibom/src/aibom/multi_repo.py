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

"""Multi-repo discovery, remote cloning, and repos-file parsing.

Supports:
- Auto-detecting git repos under a parent directory
- Reading repo paths/URLs from a text or JSON file
- Shallow-cloning a remote git URL for ephemeral scanning
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

_GIT_URL_RE = re.compile(
    r"^(https?://|git@|ssh://|git://)"
)


def is_git_url(value: str) -> bool:
    return bool(_GIT_URL_RE.match(value.strip()))


def discover_repos(parent: Path, max_depth: int = 3) -> list[Path]:
    """Find all git repositories under *parent* up to *max_depth* levels.

    Returns sorted list of repo root paths (directories containing ``.git``).
    """
    found: list[Path] = []
    if not parent.is_dir():
        return found

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        git_dir = current / ".git"
        if git_dir.exists():
            found.append(current)
            return
        try:
            children = sorted(current.iterdir())
        except PermissionError:
            return
        for child in children:
            if child.is_dir() and child.name not in (
                ".git", "node_modules", ".venv", "venv", "__pycache__",
            ):
                _walk(child, depth + 1)

    _walk(parent, 0)
    return sorted(found)


def read_repos_file(path: Path) -> list[str]:
    """Parse a repos file (JSON array or newline-delimited text).

    Blank lines and lines starting with ``#`` are ignored in text mode.
    """
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    if content.startswith("["):
        try:
            entries = json.loads(content)
            if isinstance(entries, list):
                return [str(e).strip() for e in entries if str(e).strip()]
        except json.JSONDecodeError:
            pass

    return [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


class ClonedRepo:
    """Context manager for a shallow-cloned remote repository."""

    def __init__(self, url: str, *, branch: str | None = None):
        self.url = url
        self.branch = branch
        self._tmpdir: str | None = None
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self._tmpdir = tempfile.mkdtemp(prefix="aibom_clone_")
        cmd = [
            "git", "clone", "--depth=1", "--single-branch",
        ]
        if self.branch:
            cmd.extend(["--branch", self.branch])
        cmd.extend([self.url, self._tmpdir])

        _LOGGER.debug("Cloning %s → %s", self.url, self._tmpdir)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            raise RuntimeError(
                f"git clone failed (rc={result.returncode}): {result.stderr.strip()}"
            )
        self.path = Path(self._tmpdir)
        return self.path

    def __exit__(self, *exc: object) -> None:
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            _LOGGER.debug("Cleaned up clone: %s", self._tmpdir)
            self._tmpdir = None
            self.path = None
