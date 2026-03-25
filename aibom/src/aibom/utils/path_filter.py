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

"""Shared directory/file skip logic used across all scanners."""

from __future__ import annotations

SKIP_DIR_NAMES: frozenset[str] = frozenset({
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
    "site-packages",
})


def should_skip_dir(name: str) -> bool:
    """Return True if a directory should be excluded from scanning.

    Matches exact names in SKIP_DIR_NAMES as well as vendored virtualenv
    patterns (``*_venv``, ``*-venv``) and egg-info directories.
    """
    if name in SKIP_DIR_NAMES:
        return True
    low = name.lower()
    return (
        low.endswith("_venv")
        or low.endswith("-venv")
        or low.endswith(".egg-info")
    )
