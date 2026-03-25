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

"""Agentic repo triage for org-scale scanning.

Instead of blindly scanning every repository or truncating with --max-repos,
this module lets the LLM reason about each repo's manifests and README to
decide: **deep-scan**, **skip**, or **needs-clone**.

Deterministic fast-path: If a manifest contains any known AI package from
``dependency_scanner.AI_PACKAGES``, the repo is auto-promoted to deep-scan
without spending an LLM call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_LOGGER = logging.getLogger(__name__)

_TRIAGE_SYSTEM_PROMPT = """\
You are an AI-BOM repo triage assistant.  Given a repo's manifest files and
README, decide whether the repository contains AI/ML assets worth scanning.

Respond ONLY with a JSON object:
{"decision": "deep-scan" | "skip" | "needs-clone", "reason": "<one sentence>"}

- **deep-scan**: The repo clearly uses AI/ML frameworks, models, agents, or
  MCP servers.
- **skip**: The repo has no AI/ML relevance (pure infrastructure, frontend, etc.).
- **needs-clone**: Not enough info in manifests/README to decide; a full clone
  is required for deeper analysis.
"""


@dataclass
class TriageResult:
    repo_path: str
    decision: str  # "deep-scan" | "skip" | "needs-clone"
    reason: str
    method: str = ""  # "deterministic" or "agentic"


@dataclass
class RepoTriager:
    """Triage repos for AI relevance using a deterministic + agentic approach."""

    llm_config: Optional[dict] = None
    results: list[TriageResult] = field(default_factory=list)

    def triage_repos(self, repo_paths: list[str]) -> list[TriageResult]:
        from .scanners.dependency_scanner import AI_PACKAGES

        all_ai_pkgs: set[str] = set()
        for ecosystem_pkgs in AI_PACKAGES.values():
            all_ai_pkgs.update(p.lower() for p in ecosystem_pkgs)

        deterministic: list[TriageResult] = []
        needs_llm: list[str] = []

        for repo_path in repo_paths:
            root = Path(repo_path)
            if not root.exists():
                deterministic.append(
                    TriageResult(repo_path, "skip", "path does not exist", "deterministic")
                )
                continue

            manifest_ai = self._check_manifests_for_ai(root, all_ai_pkgs)
            if manifest_ai:
                deterministic.append(
                    TriageResult(
                        repo_path, "deep-scan",
                        f"manifest contains AI packages: {', '.join(sorted(manifest_ai)[:5])}",
                        "deterministic",
                    )
                )
            else:
                needs_llm.append(repo_path)

        agentic: list[TriageResult] = []
        if needs_llm and self.llm_config:
            agentic = self._agentic_triage(needs_llm)
        elif needs_llm:
            for rp in needs_llm:
                agentic.append(
                    TriageResult(
                        rp, "needs-clone",
                        "no AI packages in manifest; agentic mode not available",
                        "deterministic",
                    )
                )

        self.results = deterministic + agentic
        return self.results

    def _check_manifests_for_ai(
        self, root: Path, all_ai_pkgs: set[str]
    ) -> set[str]:
        """Fast structural check: parse manifests for known AI packages."""
        found: set[str] = set()

        manifest_names = {
            "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
            "pipfile", "poetry.lock", "uv.lock",
            "package.json", "package-lock.json",
            "go.mod", "cargo.toml", "gemfile",
            "pom.xml", "build.gradle", "build.gradle.kts",
        }

        for child in root.iterdir():
            if child.name.lower() in manifest_names:
                try:
                    text = child.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                text_lower = text.lower()
                for pkg in all_ai_pkgs:
                    if pkg in text_lower:
                        found.add(pkg)
            if found:
                break

        return found

    def _agentic_triage(self, repo_paths: list[str]) -> list[TriageResult]:
        """Use LLM to reason about repos that lack obvious AI manifest entries."""
        results: list[TriageResult] = []

        try:
            from litellm import completion
        except ImportError:
            _LOGGER.warning("litellm not available; falling back to needs-clone for all")
            return [
                TriageResult(rp, "needs-clone", "litellm unavailable", "deterministic")
                for rp in repo_paths
            ]

        summaries: list[tuple[str, str]] = []
        for rp in repo_paths:
            root = Path(rp)
            summary = self._build_repo_summary(root)
            summaries.append((rp, summary))

        batch_prompt = "Triage the following repositories. For each, respond with a JSON array of objects.\n\n"
        for i, (rp, summary) in enumerate(summaries):
            batch_prompt += f"### Repo {i + 1}: {Path(rp).name}\n{summary}\n\n"

        batch_prompt += (
            "\nRespond with a JSON array of objects, one per repo, in order:\n"
            '[{"decision": "...", "reason": "..."}, ...]\n'
        )

        try:
            resp = completion(
                messages=[
                    {"role": "system", "content": _TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": batch_prompt},
                ],
                **self.llm_config,
            )
            raw = resp.choices[0].message.content.strip()
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end > start:
                decisions = json.loads(raw[start : end + 1])
            else:
                decisions = [json.loads(raw)]

            for (rp, _), dec in zip(summaries, decisions):
                results.append(
                    TriageResult(
                        rp,
                        dec.get("decision", "needs-clone"),
                        dec.get("reason", ""),
                        "agentic",
                    )
                )
        except Exception:
            _LOGGER.warning("Agentic triage failed; defaulting to needs-clone", exc_info=True)
            results = [
                TriageResult(rp, "needs-clone", "agentic triage failed", "deterministic")
                for rp in repo_paths
            ]

        return results

    def _build_repo_summary(self, root: Path) -> str:
        """Build a compact summary of the repo for the LLM."""
        parts: list[str] = []

        readme_names = ["README.md", "readme.md", "README.rst", "README.txt", "README"]
        for rn in readme_names:
            rp = root / rn
            if rp.is_file():
                try:
                    text = rp.read_text(encoding="utf-8", errors="replace")
                    parts.append(f"**README** (first 500 chars):\n{text[:500]}")
                except OSError:
                    pass
                break

        manifest_names = [
            "requirements.txt", "pyproject.toml", "package.json", "go.mod",
            "Cargo.toml", "Gemfile", "pom.xml", "build.gradle",
        ]
        for mn in manifest_names:
            mp = root / mn
            if mp.is_file():
                try:
                    text = mp.read_text(encoding="utf-8", errors="replace")
                    parts.append(f"**{mn}** (first 1000 chars):\n{text[:1000]}")
                except OSError:
                    pass

        if not parts:
            try:
                top = sorted(root.iterdir())[:30]
                names = [f.name for f in top]
                parts.append(f"**Top-level files**: {', '.join(names)}")
            except OSError:
                parts.append("(empty or unreadable)")

        return "\n".join(parts)
