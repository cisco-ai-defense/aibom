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

import re
import subprocess
from pathlib import Path
from typing import Optional

# Source-kind values.
SOURCE_KIND_GIT = "git"
SOURCE_KIND_CONTAINER_IMAGE = "container_image"
SOURCE_KIND_LOCAL_PATH = "local-path"

_GIT_TIMEOUT_S = 5

# scp-style git remote, e.g. ``git@github.com:org/repo.git``.
_SCP_LIKE_RE = re.compile(r"^[^/@]+@([^:/]+):(.+)$")
# ``scheme://[user[:pass]@]host[:port]/path`` for git remotes.
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://(?P<rest>.*)$")


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


def canonicalize_git_remote(remote_url: str) -> str:
    """Canonicalize a git remote URL to one stable ``host/path`` string.

    The same repository can be expressed many ways
    (``git@github.com:org/repo.git``, ``https://github.com/org/repo``, with or
    without a trailing ``.git``, mixed host casing, embedded credentials, a
    trailing slash). A downstream backend derives a deterministic asset
    identity from this string, so every equivalent spelling must collapse to a
    single canonical form.

    Canonicalization: drop the scheme, strip any ``user[:pass]@`` credentials,
    lowercase the host, drop a ``:port``, normalize path separators to ``/``,
    remove a trailing ``.git``, and drop a trailing slash. The result is
    ``host/path`` lowercased on the host only (paths stay case-sensitive).
    """
    if not remote_url:
        return ""

    raw = remote_url.strip()
    host = ""
    path = ""

    scp = _SCP_LIKE_RE.match(raw)
    url = _URL_RE.match(raw)
    if scp:
        host = scp.group(1)
        path = scp.group(2)
    elif url:
        rest = url.group("rest")
        # Strip credentials.
        if "@" in rest:
            rest = rest.rsplit("@", 1)[1]
        if "/" in rest:
            authority, path = rest.split("/", 1)
        else:
            authority, path = rest, ""
        host = authority
    else:
        # Bare ``host:path`` or ``host/path`` (no scheme, not scp/user form).
        if ":" in raw and "/" not in raw.split(":", 1)[0]:
            host, path = raw.split(":", 1)
        elif "/" in raw:
            host, path = raw.split("/", 1)
        else:
            host, path = raw, ""

    # Drop a :port from the host.
    if ":" in host:
        host = host.split(":", 1)[0]
    host = host.lower()

    path = path.replace("\\", "/").strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]

    return f"{host}/{path}" if path else host


def canonicalize_image_ref(image_ref: str) -> str:
    """Canonicalize a container image reference to a stable registry path.

    Strips a digest (``@sha256:...``) and a tag (``:tag``), normalizes the
    registry host to lowercase, and applies Docker Hub defaults
    (``library/`` for single-segment official images; an implicit
    ``docker.io`` registry). The version (tag/digest) is captured separately as
    ``source_ref_version`` and intentionally not part of the canonical ref.
    """
    if not image_ref:
        return ""

    ref = image_ref.strip()

    # Drop digest.
    if "@" in ref:
        ref = ref.split("@", 1)[0]

    # Split off an optional registry host (first segment containing '.' or
    # ':' or equal to 'localhost').
    first, sep, remainder = ref.partition("/")
    if sep and ("." in first or ":" in first or first == "localhost"):
        registry = first.lower()
        repo = remainder
    else:
        registry = "docker.io"
        repo = ref

    # Drop a tag from the final path segment (but not from a registry port).
    if ":" in repo:
        repo_head, _, repo_tag = repo.rpartition(":")
        # A ':' that is part of the path (tag) has no '/' after it.
        if "/" not in repo_tag:
            repo = repo_head

    repo = repo.replace("\\", "/").strip("/")

    # Docker Hub official images get the implicit ``library/`` namespace.
    if registry == "docker.io" and "/" not in repo and repo:
        repo = f"library/{repo}"

    return f"{registry}/{repo}" if repo else registry


def canonicalize_source_ref(source_ref: str, source_kind: str) -> str:
    """Canonicalize a source reference according to its ``source_kind``."""
    if source_kind == SOURCE_KIND_CONTAINER_IMAGE:
        return canonicalize_image_ref(source_ref)
    return canonicalize_git_remote(source_ref)


def capture_git_remote(path: str) -> Optional[str]:
    """Return the ``origin`` remote URL for the git working tree at *path*.

    Returns ``None`` when the path is not a git tree or has no ``origin``
    remote. The raw URL is returned as-is; canonicalization is the caller's
    responsibility via :func:`canonicalize_source_ref`.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    url = result.stdout.strip()
    return url or None


def capture_git_head_sha(path: str) -> Optional[str]:
    """Return the full commit SHA of ``HEAD`` for the git tree at *path*.

    Returns ``None`` when the path is not a git tree or ``HEAD`` cannot be
    resolved (e.g. a freshly initialized repo with no commits).
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def capture_source_ref_version(
    source: str,
    source_kind: str,
    *,
    image_digest: Optional[str] = None,
) -> Optional[str]:
    """Capture the point-in-time ``source_ref_version`` for a scan target.

    For a git source this is the full ``HEAD`` commit SHA; for a container
    image it is the manifest digest supplied by the caller (the
    container-extraction path already resolves it). Returns ``None`` when no
    version can be determined — the field is a best-effort version stamp, not
    part of the stable identity hash.
    """
    if source_kind == SOURCE_KIND_CONTAINER_IMAGE:
        digest = (image_digest or "").strip()
        return digest or None
    if source_kind == SOURCE_KIND_GIT:
        return capture_git_head_sha(source)
    return None
