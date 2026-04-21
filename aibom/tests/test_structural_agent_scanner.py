# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`aibom.scanners.structural_agent_scanner`.

The scanner must emit :class:`AIComponentType.AGENT` candidates for
classes with a structurally plausible ReAct-style loop while honoring
anti-patterns, test-file guards, size limits, and existing framework
coverage. All test fixtures use synthetic class / module names so no
real production class names appear in the suite.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from aibom.models import ScanContext
from aibom.models.enums import AIComponentType, DetectionSource
from aibom.scanners import structural_agent_scanner
from aibom.scanners.base import scanner_registry
from aibom.scanners.structural_agent_scanner import (
    StructuralAgentScanner,
    _is_test_file,
    iter_structural_agent_candidates,
)


# Synthetic source fixtures. None of these class names correspond to a
# real production class. They exist solely to exercise the detector.

_REACT_LOOP_SOURCE = textwrap.dedent(
    """\
    from openai import OpenAI

    class PlannerRouterLoop:
        '''Iterative planner-router loop over a router model.'''

        def __init__(self, router, tools):
            self.router = router
            self.tools = tools

        def run(self, task):
            context = {"task": task}
            while not context.get("done"):
                plan = self.router.invoke(context)
                if plan.get("action") == "tool":
                    observation = self.tools.call(plan["tool"], plan["args"])
                    context = self.router.update(context, observation)
                else:
                    context = self.router.finalize(context)
            return context
    """
)


# Two distinct callees in the loop body — below the bumped
# ``min_react_loop_distinct_callees`` threshold, so the scanner must
# reject this even though it has a loop, a branch, and an LLM hint.
_TWO_CALLEE_LOOP_SOURCE = textwrap.dedent(
    """\
    from openai import OpenAI

    class TwoCalleeLoop:
        '''Loop with only two distinct callees.'''

        def run(self, task):
            context = {"task": task}
            while not context.get("done"):
                plan = self.llm.invoke(context)
                if plan:
                    context = self.llm.invoke(plan)
                else:
                    context = self.llm.invoke(None)
            return context
    """
)


# Structurally-ReAct-shaped loop but with NO LLM SDK import at module
# level — historically a dominant source of false positives such as
# HTTP pollers and ETL retry wrappers.
_LOOP_WITHOUT_LLM_IMPORT_SOURCE = textwrap.dedent(
    """\
    import httpx

    class HttpPoller:
        '''Polls an HTTP endpoint until done — not an agent.'''

        def __init__(self, client, handler, metrics):
            self.client = client
            self.handler = handler
            self.metrics = metrics

        def run(self, task):
            status = None
            while status != "done":
                response = self.client.fetch(task)
                if response.ok:
                    status = self.handler.process(response)
                    self.metrics.incr("ok")
                else:
                    status = self.handler.retry(response)
            return status
    """
)


# Loop whose body callees are only legacy "loose" verbs that previously
# matched (run / stream / generate / complete). With the tightened
# ``_LLM_CALL_HINTS`` these are now rejected because nothing actually
# looks like an LLM call.
_LOOSE_HINT_ONLY_LOOP_SOURCE = textwrap.dedent(
    """\
    from openai import OpenAI

    class LooseHintLoop:
        '''Loop over non-LLM verbs that happen to appear in old hints.'''

        def run(self, items):
            out = []
            while items:
                head = items.pop(0)
                if head.kind == "a":
                    out.append(self.stream.read(head))
                else:
                    out.append(self.generator.next_chunk(head))
                self.metrics.complete(head)
            return out
    """
)


_TEMPORAL_WORKFLOW_SOURCE = textwrap.dedent(
    """\
    from temporalio import workflow

    @workflow.defn
    class OrchestrationWorkflow:
        '''Looks like a loop but is a Temporal workflow \u2014 anti-pattern.'''

        @workflow.run
        async def run(self, items):
            results = []
            while items:
                head = items.pop(0)
                if head.kind == "a":
                    results.append(await workflow.execute_activity(head))
                else:
                    results.append(await workflow.execute_local_activity(head))
            return results
    """
)


_FRAMEWORK_AGENT_SOURCE = textwrap.dedent(
    """\
    from langchain.agents import BaseSingleActionAgent

    class FrameworkAgent(BaseSingleActionAgent):
        '''A LangChain agent \u2014 already handled by KB / config scanners.'''

        def run(self, query):
            while not self.done:
                action = self.plan(query)
                if action.kind == "tool":
                    observation = self.execute(action)
                    self.remember(observation)
                else:
                    self.finalize()
            return self.result
    """
)


_NOT_A_LOOP_SOURCE = textwrap.dedent(
    """\
    class SimpleTransformer:
        '''Does not iterate or dispatch \u2014 not a candidate.'''

        def transform(self, value):
            return value.upper()

        def describe(self):
            return "transformer"
    """
)


# ---------------------------------------------------------------------------
# Test-file detection helper
# ---------------------------------------------------------------------------


class TestIsTestFile:
    def test_test_prefix_filename_is_skipped(self, tmp_path: Path) -> None:
        assert _is_test_file(tmp_path / "test_loop.py") is True

    def test_test_suffix_filename_is_skipped(self, tmp_path: Path) -> None:
        assert _is_test_file(tmp_path / "loop_test.py") is True

    def test_conftest_is_skipped(self, tmp_path: Path) -> None:
        assert _is_test_file(tmp_path / "conftest.py") is True

    def test_tests_parent_dir_is_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "tests" / "module_a.py"
        p.parent.mkdir(parents=True)
        assert _is_test_file(p) is True

    def test_examples_parent_dir_is_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "examples" / "demo.py"
        p.parent.mkdir(parents=True)
        assert _is_test_file(p) is True

    def test_production_file_is_not_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "src" / "production.py"
        p.parent.mkdir(parents=True)
        assert _is_test_file(p) is False


# ---------------------------------------------------------------------------
# Candidate emission
# ---------------------------------------------------------------------------


def _write_module(root: Path, name: str, source: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _context(root: Path) -> ScanContext:
    return ScanContext(paths=[str(root)])


class TestEmission:
    def test_react_loop_produces_one_agent_candidate(
        self, tmp_path: Path
    ) -> None:
        _write_module(tmp_path, "planner.py", _REACT_LOOP_SOURCE)

        comps = iter_structural_agent_candidates(_context(tmp_path))

        assert len(comps) == 1
        c = comps[0]
        assert c.name == "PlannerRouterLoop"
        assert c.component_type == AIComponentType.AGENT
        assert c.detection_source == DetectionSource.CODE_ANALYSIS
        assert c.framework == "unknown"
        assert c.metadata["discovery"] == "structural_react_loop"
        assert isinstance(c.metadata["react_loop_start_line"], int)
        assert c.metadata["react_loop_start_line"] <= c.metadata["react_loop_end_line"]
        assert "Structural ReAct" in c.agentic_hint

    def test_temporal_anti_pattern_suppresses_candidate(
        self, tmp_path: Path
    ) -> None:
        _write_module(
            tmp_path, "workflow_module.py", _TEMPORAL_WORKFLOW_SOURCE
        )

        comps = iter_structural_agent_candidates(_context(tmp_path))

        assert comps == []

    def test_framework_match_is_left_to_kb_config_scanners(
        self, tmp_path: Path
    ) -> None:
        _write_module(tmp_path, "framework_agent.py", _FRAMEWORK_AGENT_SOURCE)

        comps = iter_structural_agent_candidates(_context(tmp_path))

        assert comps == []

    def test_no_loop_no_candidate(self, tmp_path: Path) -> None:
        _write_module(tmp_path, "simple.py", _NOT_A_LOOP_SOURCE)

        comps = iter_structural_agent_candidates(_context(tmp_path))

        assert comps == []

    def test_test_file_is_skipped_even_with_valid_loop(
        self, tmp_path: Path
    ) -> None:
        _write_module(
            tmp_path / "tests", "test_planner_loop.py", _REACT_LOOP_SOURCE
        )
        _write_module(tmp_path, "conftest.py", _REACT_LOOP_SOURCE)

        comps = iter_structural_agent_candidates(_context(tmp_path))

        assert comps == []

    def test_duplicate_files_do_not_emit_twice(self, tmp_path: Path) -> None:
        _write_module(tmp_path / "pkg", "planner.py", _REACT_LOOP_SOURCE)

        ctx = ScanContext(
            paths=[str(tmp_path), str(tmp_path / "pkg")]
        )
        comps = iter_structural_agent_candidates(ctx)

        assert len({(c.file_path, c.line_number, c.name) for c in comps}) == len(comps)
        assert len(comps) == 1

    def test_oversize_python_file_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _write_module(tmp_path, "planner.py", _REACT_LOOP_SOURCE)
        real_size = mod.stat().st_size
        assert real_size > 0
        monkeypatch.setattr(
            structural_agent_scanner,
            "_MAX_PY_FILE_SIZE_BYTES",
            real_size - 1,
        )

        comps = iter_structural_agent_candidates(_context(tmp_path))

        assert comps == []


class TestTightenedEmissionGate:
    """Regression tests for the tightened structural-agent emission.

    Historically the scanner accepted any loop with ≥2 callees, a
    branch, and an advisory LLM-hint match — which routinely tagged
    HTTP pollers, ETL retry wrappers, and plain state machines as
    agents. After the tightening:

    * The file must import at least one LLM / agent-runtime SDK.
    * The loop body must contain at least one call whose qualified
      name matches a high-precision LLM hint (see
      ``_LLM_CALL_HINTS``).
    * The loop must have at least three distinct callees.
    """

    def test_file_without_llm_sdk_import_is_rejected(
        self, tmp_path: Path
    ) -> None:
        _write_module(
            tmp_path, "poller.py", _LOOP_WITHOUT_LLM_IMPORT_SOURCE
        )

        comps = iter_structural_agent_candidates(_context(tmp_path))

        assert comps == [], (
            "Files with no LLM SDK import must not emit structural "
            "agent candidates — this was the FP source for HTTP "
            "pollers and ETL runners."
        )

    def test_strands_import_passes_llm_sdk_hint_gate(
        self, tmp_path: Path
    ) -> None:
        """A file whose ONLY LLM SDK import is ``strands`` must pass the
        file-level hint gate so a plausible ReAct loop can emit an agent
        candidate. Prior to adding ``strands`` / ``strands_tools`` to
        ``_BUILTIN_LLM_SDK_IMPORT_HINTS`` this was silently filtered out.
        """
        strands_react_source = _REACT_LOOP_SOURCE.replace(
            "from openai import OpenAI",
            "from strands import Agent",
        )
        _write_module(tmp_path, "strands_planner.py", strands_react_source)

        comps = iter_structural_agent_candidates(_context(tmp_path))

        assert len(comps) == 1, (
            "``from strands import Agent`` must satisfy the LLM SDK import "
            "hint gate so the structural scanner considers the file."
        )
        assert comps[0].name == "PlannerRouterLoop"

    def test_strands_tools_import_passes_llm_sdk_hint_gate(
        self, tmp_path: Path
    ) -> None:
        """``strands_tools`` alone also satisfies the hint gate."""
        source = _REACT_LOOP_SOURCE.replace(
            "from openai import OpenAI",
            "from strands_tools import mcp_client",
        )
        _write_module(tmp_path, "mcp_planner.py", source)

        comps = iter_structural_agent_candidates(_context(tmp_path))

        assert len(comps) == 1

    def test_two_callee_loop_is_rejected(self, tmp_path: Path) -> None:
        _write_module(tmp_path, "two_callee.py", _TWO_CALLEE_LOOP_SOURCE)

        comps = iter_structural_agent_candidates(_context(tmp_path))

        assert comps == [], (
            "min_react_loop_distinct_callees is 3; a loop with only "
            "two distinct callees must not be emitted."
        )

    def test_loose_hint_only_loop_is_rejected(
        self, tmp_path: Path
    ) -> None:
        _write_module(tmp_path, "loose.py", _LOOSE_HINT_ONLY_LOOP_SOURCE)

        comps = iter_structural_agent_candidates(_context(tmp_path))

        assert comps == [], (
            "Loops whose only hint matches are loose verbs (run / "
            "stream / generate / complete) must not be emitted after "
            "the hint set was tightened."
        )


class TestEmissionAttachedEvidence:
    """Regression tests for machine-generated ``agent_evidence``.

    The structural scanner must attach a verifiable ``agent_evidence``
    payload at emission time so the symmetric evidence gate can drop
    candidates whose citation no longer resolves (e.g. source file
    moved / truncated) on the same terms it uses for LLM-generated
    verdicts.
    """

    def test_evidence_attached_with_expected_schema(
        self, tmp_path: Path
    ) -> None:
        _write_module(tmp_path, "planner.py", _REACT_LOOP_SOURCE)

        comps = iter_structural_agent_candidates(_context(tmp_path))

        assert len(comps) == 1
        ev = comps[0].metadata.get("agent_evidence")
        assert isinstance(ev, dict), (
            "agent_evidence must be attached at emission"
        )
        assert ev["pattern"] == "react_loop"
        assert ev["definition_file"] == comps[0].file_path
        assert isinstance(ev["definition_start_line"], int)
        assert isinstance(ev["definition_end_line"], int)
        assert (
            ev["definition_start_line"]
            <= comps[0].metadata["react_loop_start_line"]
        )
        assert (
            ev["definition_end_line"]
            >= comps[0].metadata["react_loop_end_line"]
        )
        assert "evidence_snippet" in ev
        assert ev["evidence_snippet"].strip()
        assert ev["justification"]

    def test_emitted_evidence_passes_middleware_verification(
        self, tmp_path: Path
    ) -> None:
        """The emitted evidence must round-trip through the gate used
        to validate LLM-generated ``agent_evidence`` blocks, so both
        detection paths share the same contract.
        """
        from aibom.agentic.middleware import _verify_agent_evidence

        _write_module(tmp_path, "planner.py", _REACT_LOOP_SOURCE)

        comps = iter_structural_agent_candidates(_context(tmp_path))
        assert len(comps) == 1

        ok, reason = _verify_agent_evidence(
            comps[0].metadata["agent_evidence"],
            allowed_roots=[str(tmp_path)],
        )
        assert ok, (
            f"Machine-generated evidence must pass the same gate the "
            f"middleware applies to LLM verdicts: {reason}"
        )

    def test_evidence_omitted_when_file_unreadable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the file cannot be read to produce a snippet, the
        scanner must *omit* ``agent_evidence`` rather than attach an
        unverifiable citation. The symmetric evidence gate will then
        drop this component downstream.
        """
        _write_module(tmp_path, "planner.py", _REACT_LOOP_SOURCE)

        def _fail_read(*_args: object, **_kwargs: object) -> str:
            raise OSError("simulated unreadable file")

        monkeypatch.setattr(
            structural_agent_scanner,
            "_read_loop_snippet",
            lambda *_a, **_kw: "",
        )

        comps = iter_structural_agent_candidates(_context(tmp_path))

        assert len(comps) == 1
        assert "agent_evidence" not in comps[0].metadata, (
            "An empty snippet must prevent agent_evidence attachment, "
            "not produce a citation that cannot be verified."
        )


# ---------------------------------------------------------------------------
# Scanner registration / integration
# ---------------------------------------------------------------------------


class TestScannerRegistration:
    def test_scanner_auto_registered_on_package_import(self) -> None:
        import aibom.scanners  # noqa: F401

        assert StructuralAgentScanner in scanner_registry

    def test_supports_any_context(self, tmp_path: Path) -> None:
        scanner = StructuralAgentScanner()
        assert scanner.supports(_context(tmp_path))

    def test_scan_returns_empty_relationships(self, tmp_path: Path) -> None:
        _write_module(tmp_path, "planner.py", _REACT_LOOP_SOURCE)
        scanner = StructuralAgentScanner()
        comps, rels = scanner.scan(_context(tmp_path))

        assert rels == []
        assert len(comps) == 1
        assert comps[0].name == "PlannerRouterLoop"
