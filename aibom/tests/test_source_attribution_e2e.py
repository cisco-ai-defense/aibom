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

"""End-to-end source-attribution test.

Exercises the full attribution path the way a downstream backend consumes it:
scan a real git working tree, render the BOM JSON through the production
reporter, and assert that the emitted ``metadata`` block carries the same
``(source_kind, source_ref_canonical, source_ref_version)`` the CLI computed —
and that those values, which a backend hashes into a stable per-asset identity,
are identical across a repeat scan of the same source (no duplicate asset).

The staging ingestion backend that projects assets lives outside this repo, so
the "projection triple match" is asserted here against the BOM the backend
would ingest, and identity stability is asserted by re-rendering the same
source and comparing the triple byte-for-byte.
"""

from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path

from aibom.models import AIComponent, AIComponentType, ScanResult, SourceResult
from aibom.reporters import JsonReporter
from aibom.source_attribution import (
    canonicalize_source_ref,
    capture_source_ref_version,
)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _make_repo(path: Path) -> str:
    """Create a git working tree with an origin remote and one commit."""
    path.mkdir(parents=True, exist_ok=True)
    _git("init", cwd=path)
    _git("config", "user.email", "dev@example.com", cwd=path)
    _git("config", "user.name", "Dev", cwd=path)
    _git("remote", "add", "origin", "https://github.com/example-org/svc.git", cwd=path)
    (path / "agent.py").write_text(
        "from langgraph.prebuilt import create_react_agent\n"
        "agent = create_react_agent(model=llm, tools=tools)\n",
        encoding="utf-8",
    )
    _git("add", "agent.py", cwd=path)
    _git("commit", "-m", "initial", cwd=path)
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _scan_result_for(path: str) -> ScanResult:
    # A minimal but realistic single-source result; the reporter derives the
    # attribution block from the path on disk.
    comp = AIComponent(
        name="agent",
        component_type=AIComponentType.AGENT,
        file_path=str(Path(path) / "agent.py"),
        line_number=2,
        framework="langgraph",
    )
    return ScanResult(
        metadata={"analyzer_version": "test", "run_id": "e2e-001"},
        sources=[SourceResult(path=path, components=[comp], relationships=[])],
    )


def _render_triple(path: str) -> dict:
    buf = StringIO()
    JsonReporter().render(_scan_result_for(path), buf)
    data = json.loads(buf.getvalue())
    src = next(iter(data["aibom_analysis"]["sources"].values()))
    return src["metadata"]


class TestSourceAttributionEndToEnd:
    def test_emitted_triple_matches_computed_values(self, tmp_path: Path) -> None:
        head = _make_repo(tmp_path)

        meta = _render_triple(str(tmp_path))

        # The BOM the backend ingests carries the attribution the CLI computed.
        assert meta["source_kind"] == "git"
        assert meta["source_ref_canonical"] == canonicalize_source_ref(
            "https://github.com/example-org/svc.git", "git"
        )
        assert meta["source_ref_canonical"] == "github.com/example-org/svc"
        assert meta["source_ref_version"] == head
        assert meta["source_ref_version"] == capture_source_ref_version(
            str(tmp_path), "git"
        )

    def test_identity_triple_stable_across_repeat_scan(self, tmp_path: Path) -> None:
        _make_repo(tmp_path)

        first = _render_triple(str(tmp_path))
        second = _render_triple(str(tmp_path))

        # The identity-bearing fields must be byte-for-byte identical across a
        # repeat scan of the same source so the backend de-duplicates the asset.
        identity_keys = ("source_kind", "source_ref_canonical", "source_ref_version")
        assert {k: first[k] for k in identity_keys} == {
            k: second[k] for k in identity_keys
        }

    def test_two_distinct_repos_have_distinct_canonical_refs(
        self, tmp_path: Path
    ) -> None:
        repo_a = tmp_path / "a"
        repo_b = tmp_path / "b"
        _make_repo(repo_a)
        repo_b.mkdir()
        _git("init", cwd=repo_b)
        _git("config", "user.email", "dev@example.com", cwd=repo_b)
        _git("config", "user.name", "Dev", cwd=repo_b)
        _git(
            "remote",
            "add",
            "origin",
            "git@github.com:example-org/other.git",
            cwd=repo_b,
        )
        (repo_b / "f.py").write_text("x = 1\n", encoding="utf-8")
        _git("add", "f.py", cwd=repo_b)
        _git("commit", "-m", "init", cwd=repo_b)

        meta_a = _render_triple(str(repo_a))
        meta_b = _render_triple(str(repo_b))

        assert meta_a["source_ref_canonical"] == "github.com/example-org/svc"
        assert meta_b["source_ref_canonical"] == "github.com/example-org/other"
        assert meta_a["source_ref_canonical"] != meta_b["source_ref_canonical"]
