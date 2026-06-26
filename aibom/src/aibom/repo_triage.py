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

The triage agent explores each repository with tools (directory listing,
file reading, codebase search, package-registry lookup) and decides:
**deep-scan**, **skip**, or **needs-clone**.

No hardcoded package list is used as a gate.  The agent is the sole
authority on whether a repository contains AI/ML assets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_LOGGER = logging.getLogger(__name__)

_TRIAGE_RECURSION_LIMIT = 20
_TRIAGE_TIMEOUT_S = 30


@dataclass
class TriageResult:
    repo_path: str
    decision: str  # "deep-scan" | "skip" | "needs-clone"
    reason: str
    evidence: list[str] = field(default_factory=list)
    method: str = ""  # "agentic"


@dataclass
class RepoTriager:
    """Triage repos for AI relevance using a tool-equipped agent."""

    llm_config: Optional[dict[str, Any]] = None
    results: list[TriageResult] = field(default_factory=list)

    def triage_repos(self, repo_paths: list[str]) -> list[TriageResult]:
        results: list[TriageResult] = []

        for repo_path in repo_paths:
            root = Path(repo_path)
            if not root.exists():
                results.append(
                    TriageResult(
                        repo_path,
                        "skip",
                        "path does not exist",
                        method="deterministic",
                    )
                )
                continue
            result = self._triage_single(repo_path)
            results.append(result)

        self.results = results
        return self.results

    def _triage_single(self, repo_path: str) -> TriageResult:
        """Run the triage agent on a single repository."""
        if not self.llm_config:
            return TriageResult(
                repo_path,
                "deep-scan",
                "no LLM config — defaulting to deep-scan (safe bias)",
                method="deterministic",
            )

        try:
            from deepagents import create_deep_agent

            from .agentic.prompts import TRIAGE_AGENT_SYSTEM_PROMPT
            from .agentic.tools import build_triage_tools
            from .llm_factory import build_chat_model
        except ImportError:
            _LOGGER.warning(
                "Agentic triage requires 'cisco-aibom[agentic]'; "
                "defaulting to deep-scan"
            )
            return TriageResult(
                repo_path,
                "deep-scan",
                "agentic extras unavailable — defaulting to deep-scan",
                method="deterministic",
            )

        try:
            cfg = dict(self.llm_config)
            model_string = cfg["model"]

            model = build_chat_model(
                model_string,
                provider=cfg.get("provider"),
                api_key=cfg.get("api_key"),
                api_base=cfg.get("api_base"),
                api_version=cfg.get("api_version"),
                max_tokens=cfg.get("max_tokens"),
            )

            tools = build_triage_tools(repo_path)

            agent = create_deep_agent(
                model=model,
                tools=tools,
                system_prompt=TRIAGE_AGENT_SYSTEM_PROMPT,
                response_format=None,
                name="aibom-triage",
            )

            repo_name = Path(repo_path).name
            user_msg = (
                f"Triage repository: {repo_name}\n"
                f"Path: {repo_path}\n"
                f"Decide if this repo contains AI/ML assets worth scanning."
            )

            result = self._invoke_with_timeout(agent, user_msg)
            return self._parse_result(repo_path, result)

        except Exception:
            _LOGGER.warning(
                "Agentic triage failed for %s; defaulting to deep-scan",
                repo_path,
                exc_info=True,
            )
            return TriageResult(
                repo_path,
                "deep-scan",
                "agentic triage failed — defaulting to deep-scan",
                method="agentic",
            )

    def _invoke_with_timeout(self, agent: Any, user_msg: str) -> Any:
        """Invoke the agent with a hard wall-clock timeout.

        Delegates to the shared daemon-thread deadline helper: a blocking
        ``agent.invoke`` that hangs is abandoned after ``_TRIAGE_TIMEOUT_S``
        rather than wedging the scan at event-loop/process shutdown.
        The helper raises ``_InvokeTimeout`` (an ``Exception``),
        which the caller's broad ``except Exception`` maps to a deep-scan
        fallback. Imported lazily to keep the agentic extras optional.
        """
        from .agentic.agent import _invoke_agent_bounded

        return _invoke_agent_bounded(
            agent,
            user_msg,
            _TRIAGE_TIMEOUT_S,
            recursion_limit=_TRIAGE_RECURSION_LIMIT,
        )

    def _parse_result(self, repo_path: str, result: Any) -> TriageResult:
        """Extract the triage decision from the agent response."""
        content = None
        sr = result.get("structured_response") if isinstance(result, dict) else None
        if sr is not None:
            if hasattr(sr, "model_dump"):
                content = sr.model_dump()
            elif isinstance(sr, dict):
                content = sr

        if content is None:
            messages = result.get("messages", []) if isinstance(result, dict) else []
            for msg in reversed(messages):
                text = (
                    getattr(msg, "content", "") if hasattr(msg, "content") else str(msg)
                )
                if not text:
                    continue
                start = text.find("{")
                end = text.rfind("}")
                if start >= 0 and end > start:
                    try:
                        content = json.loads(text[start : end + 1])
                        break
                    except json.JSONDecodeError:
                        continue

        if not content or not isinstance(content, dict):
            return TriageResult(
                repo_path,
                "deep-scan",
                "could not parse agent response — defaulting to deep-scan",
                method="agentic",
            )

        decision = content.get("decision", "deep-scan")
        if decision not in ("deep-scan", "skip", "needs-clone"):
            decision = "deep-scan"

        return TriageResult(
            repo_path=repo_path,
            decision=decision,
            reason=content.get("reason", ""),
            evidence=content.get("evidence", []),
            method="agentic",
        )
