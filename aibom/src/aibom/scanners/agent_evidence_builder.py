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

"""Per-class agent-evidence dossier builder.

Phase 2 of the agent-detection rework. Consumes:

* a merged :class:`aibom.agent_signatures.AgentSignatureCatalog`
* a collection of :class:`aibom.structures.CodeAnalysisResult` (one per
  parsed Python file)

and produces a list of :class:`AgentEvidenceDossier` entries, keyed by
``(file_path, qualified_class_name)``, that the Phase 5 LLM prompt
injects and the Phase 6 verification gate checks.

The builder is intentionally **declarative and strict**:

* It does **not** read source files from disk. All raw source it cites
  comes from :attr:`ClassBodyFactsObservation.body_source` (captured by
  the libcst visitor in Phase 1).
* It does **not** infer agency on its own. It surfaces structured facts
  (framework match, ReAct loop shape, protocol signals, anti-patterns)
  and leaves the final classification to the LLM + verification gate.
* It emits **line-bounded** evidence so the verification gate can
  independently re-verify each claim.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..agent_signatures import (
    AgentAntiPatternSignature,
    AgentFrameworkSignature,
    AgentProtocolSignature,
    AgentSignatureCatalog,
    VerificationPolicy,
)
from ..structures import (
    ClassBodyFactsObservation,
    CodeAnalysisResult,
    ControlFlowObservation,
    MethodBodyShapeObservation,
    StringLiteralObservation,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dossier dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AgentEvidenceMatch:
    """A single framework/protocol/ReAct-loop match inside a class.

    ``evidence_pattern`` aligns with :class:`aibom.agentic.agent.AgentEvidence`
    so the LLM's structured response can cite this match verbatim.
    """

    signature_id: str
    evidence_pattern: str
    file_path: str
    start_line: int
    end_line: int
    rationale: str


@dataclass
class AgentAntiPatternMatch:
    """A signature that EXCLUDES agent classification for a class."""

    signature_id: str
    label: str
    file_path: str
    line_number: int
    rationale: str


@dataclass
class AgentEvidenceDossier:
    """All evidence discovered for a given class.

    The dossier is consumed by:

    * the LLM prompt (Phase 5) — as structured context
    * the verification gate (Phase 6) — to verify the LLM's citations
    * the evidence-aware scanner (future) — to short-circuit obvious cases

    ``class_body_source`` is the verbatim text of the class definition
    captured by the libcst visitor; it is safe to include in prompts
    because it is bounded to the class body (not the whole file).
    """

    class_name: str
    qualified_name: str | None
    file_path: str
    class_start_line: int
    class_end_line: int
    class_body_source: str = ""
    framework_matches: list[AgentEvidenceMatch] = field(default_factory=list)
    protocol_matches: list[AgentEvidenceMatch] = field(default_factory=list)
    react_loop_matches: list[AgentEvidenceMatch] = field(default_factory=list)
    anti_pattern_matches: list[AgentAntiPatternMatch] = field(default_factory=list)

    @property
    def has_direct_agent_evidence(self) -> bool:
        """True when at least one match is positive, standalone agent evidence.

        * framework_agent / framework_inheritance (from ``framework_matches``)
        * a2a_server (from ``protocol_matches``)
        * react_loop (from ``react_loop_matches``)

        Protocol matches with pattern ``remote_proxy`` or ``other`` do NOT
        count as direct evidence; they require cross-repo verification or
        are informational.
        """
        if self.framework_matches:
            return True
        if self.react_loop_matches:
            return True
        for match in self.protocol_matches:
            if match.evidence_pattern == "a2a_server":
                return True
        return False

    @property
    def has_remote_proxy_evidence(self) -> bool:
        """True when the class appears to call a remote agent.

        Remote proxy classifications require independent confirmation
        (Phase 4 — cross-repo resolver or parsed Agent Card).
        """
        return any(m.evidence_pattern == "remote_proxy" for m in self.protocol_matches)

    @property
    def is_excluded_by_anti_pattern(self) -> bool:
        return bool(self.anti_pattern_matches)

    @property
    def preferred_pattern(self) -> str | None:
        """The highest-priority ``AgentEvidence.pattern`` this dossier
        supports. Used by the LLM prompt to pre-select a pattern suggestion.
        """
        priority = (
            "framework_agent",
            "framework_inheritance",
            "a2a_server",
            "react_loop",
            "remote_proxy",
            "other",
        )
        seen: set[str] = set()
        for match in (
            self.framework_matches + self.protocol_matches + self.react_loop_matches
        ):
            seen.add(match.evidence_pattern)
        for pattern in priority:
            if pattern in seen:
                return pattern
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_lines(result: CodeAnalysisResult) -> list[str]:
    """Return the raw import statement strings from a file-level analysis."""
    return [stmt for _, stmt in result.imports]


def _normalize_import_line(line: str) -> list[str]:
    """Return the raw line plus a dotted-module alias.

    Signatures declare import substrings in dotted form
    (e.g. ``"temporalio.workflow"``), but Python accepts both
    ``import temporalio.workflow`` *and* ``from temporalio import workflow``.
    Only the first form literally contains the dotted substring in its raw
    text. To match both without tightening the signature file, we also emit
    a normalized view of ``from X import Y`` as the dotted string ``X.Y``
    (handling ``as`` aliases and comma-separated ``import A, B``).

    The original line is always retained so existing substring patterns that
    intentionally reference non-dotted text still match.
    """
    candidates = [line]
    stripped = line.strip()
    if stripped.startswith("from ") and " import " in stripped:
        try:
            after_from = stripped[len("from "):]
            module_part, imports_part = after_from.split(" import ", 1)
            module_part = module_part.strip()
            imports_part = imports_part.strip().rstrip(";")
            if imports_part.startswith("(") and imports_part.endswith(")"):
                imports_part = imports_part[1:-1]
            for raw_name in imports_part.split(","):
                name = raw_name.strip()
                if not name or name == "*":
                    continue
                if " as " in name:
                    name = name.split(" as ", 1)[0].strip()
                if name:
                    candidates.append(f"{module_part}.{name}")
        except ValueError:
            pass
    elif stripped.startswith("import "):
        body = stripped[len("import "):].strip().rstrip(";")
        for raw_name in body.split(","):
            name = raw_name.strip()
            if " as " in name:
                name = name.split(" as ", 1)[0].strip()
            if name:
                candidates.append(name)
    return candidates


def _file_imports_any_substring(
    result: CodeAnalysisResult, substrings: Sequence[str]
) -> bool:
    """True if any import statement contains any of *substrings*."""
    if not substrings:
        return False
    for line in _import_lines(result):
        for candidate in _normalize_import_line(line):
            if any(sub in candidate for sub in substrings):
                return True
    return False


def _within(line_number: int, start: int, end: int) -> bool:
    return start <= line_number <= end


def _qualified_name_matches(
    haystack: str, needle: str
) -> bool:
    """Match a candidate against a signature's qualified-name pattern.

    A signature pattern matches when *needle* equals *haystack* or is a
    dotted suffix of it. This lets us match ``langchain.agents.AgentExecutor``
    when the user code only sees ``AgentExecutor`` locally, and vice versa.
    """
    if not haystack or not needle:
        return False
    if haystack == needle:
        return True
    # suffix on dotted boundary
    if haystack.endswith("." + needle):
        return True
    if needle.endswith("." + haystack):
        return True
    return False


def _base_class_matches(
    class_bases: Sequence[str], needle: str
) -> bool:
    return any(_qualified_name_matches(base, needle) for base in class_bases)


def _class_matches_imports(
    result: CodeAnalysisResult, import_substrings: Sequence[str]
) -> bool:
    """A framework's ``import_substrings`` list acts as a required narrowing
    filter when the entrypoint short names are ambiguous (e.g. ``Agent``).
    An empty list means 'no narrowing' and always matches.
    """
    if not import_substrings:
        return True
    return _file_imports_any_substring(result, import_substrings)


def _calls_in_class(
    result: CodeAnalysisResult,
    class_obs: ClassBodyFactsObservation,
) -> list[tuple[str, int]]:
    """Return ``(qualified_name, line)`` pairs for calls inside the class's
    line range. Uses the file-level ``calls`` list, not ``method_shapes``'
    bounded-to-method lists, because protocol hits can appear in method
    bodies or constructor default values alike.
    """
    return [
        (call.qualified_name, call.line_number)
        for call in result.calls
        if _within(call.line_number, class_obs.start_line, class_obs.end_line)
    ]


def _assignments_in_class(
    result: CodeAnalysisResult,
    class_obs: ClassBodyFactsObservation,
) -> list[tuple[str, int]]:
    """Return ``(assignment_call_qualified_name, line)`` for class-scoped
    assignments whose RHS is a call. Captures patterns like
    ``self.agent = AgentExecutor(...)``.
    """
    return [
        (assign.call.qualified_name, assign.line_number)
        for assign in result.assignments
        if _within(assign.line_number, class_obs.start_line, class_obs.end_line)
    ]


def _method_shapes_in_class(
    result: CodeAnalysisResult,
    class_obs: ClassBodyFactsObservation,
) -> list[MethodBodyShapeObservation]:
    return [
        shape
        for shape in result.method_shapes
        if shape.owner_class_name == class_obs.class_name
        and _within(shape.start_line, class_obs.start_line, class_obs.end_line)
    ]


def _loops_in_class(
    result: CodeAnalysisResult,
    class_obs: ClassBodyFactsObservation,
) -> list[ControlFlowObservation]:
    return [
        loop
        for loop in result.control_flows
        if loop.owner_class_name == class_obs.class_name
        and _within(loop.start_line, class_obs.start_line, class_obs.end_line)
    ]


def _strings_in_class(
    result: CodeAnalysisResult,
    class_obs: ClassBodyFactsObservation,
) -> list[StringLiteralObservation]:
    return [
        s
        for s in result.protocol_strings
        if s.owner_class_name == class_obs.class_name
        and _within(s.line_number, class_obs.start_line, class_obs.end_line)
    ]


def _decorators_in_class(
    result: CodeAnalysisResult,
    class_obs: ClassBodyFactsObservation,
) -> list[tuple[str, int]]:
    """Decorators applied to the class itself *or* to any method inside it.

    Class-level decorators (e.g. ``@workflow.defn`` above a Temporal
    workflow class) come from :attr:`ClassBodyFactsObservation.class_decorators`
    and are reported at the class start line since libcst does not expose a
    separate line number for each class decorator in that field.
    """
    method_decorators = [
        (dec.decorator_qualified_name, dec.line_number)
        for dec in result.decorators
        if _within(dec.line_number, class_obs.start_line, class_obs.end_line)
    ]
    class_decorators = [
        (name, class_obs.start_line) for name in class_obs.class_decorators
    ]
    return class_decorators + method_decorators


# ---------------------------------------------------------------------------
# Match: frameworks
# ---------------------------------------------------------------------------


def _match_framework(
    sig: AgentFrameworkSignature,
    result: CodeAnalysisResult,
    class_obs: ClassBodyFactsObservation,
) -> AgentEvidenceMatch | None:
    """Produce a framework match if *sig* fires for *class_obs*."""
    if not _class_matches_imports(result, sig.import_substrings):
        return None

    # 1. Entrypoint call or assignment inside the class body.
    for needle in sig.entrypoint_qualified_names:
        for qname, line in _calls_in_class(result, class_obs):
            if _qualified_name_matches(qname, needle):
                return AgentEvidenceMatch(
                    signature_id=sig.id,
                    evidence_pattern=sig.evidence_pattern,
                    file_path=result.file_path,
                    start_line=line,
                    end_line=line,
                    rationale=(
                        f"Call to framework entrypoint '{qname}' at line {line} "
                        f"(matches signature '{sig.id}', framework='{sig.framework}')."
                    ),
                )
        for qname, line in _assignments_in_class(result, class_obs):
            if _qualified_name_matches(qname, needle):
                return AgentEvidenceMatch(
                    signature_id=sig.id,
                    evidence_pattern=sig.evidence_pattern,
                    file_path=result.file_path,
                    start_line=line,
                    end_line=line,
                    rationale=(
                        f"Assignment from framework entrypoint '{qname}' at line {line} "
                        f"(matches signature '{sig.id}', framework='{sig.framework}')."
                    ),
                )

    # 2. Base-class inheritance.
    for needle in sig.base_class_names:
        if _base_class_matches(class_obs.base_classes, needle):
            return AgentEvidenceMatch(
                signature_id=sig.id,
                evidence_pattern=sig.evidence_pattern,
                file_path=result.file_path,
                start_line=class_obs.start_line,
                end_line=class_obs.start_line,
                rationale=(
                    f"Class '{class_obs.class_name}' inherits from "
                    f"'{needle}' (signature '{sig.id}', framework='{sig.framework}')."
                ),
            )

    return None


# ---------------------------------------------------------------------------
# Match: protocols
# ---------------------------------------------------------------------------


def _match_protocol(
    sig: AgentProtocolSignature,
    result: CodeAnalysisResult,
    class_obs: ClassBodyFactsObservation,
) -> AgentEvidenceMatch | None:
    """Produce a protocol match if *sig* fires for *class_obs*.

    Protocol signatures use substring matching (not exact/suffix) because
    protocol identifiers are often embedded in longer strings or call
    chains — e.g. ``app.client.beta.assistants.create``.
    """
    if sig.import_substrings and not _file_imports_any_substring(
        result, sig.import_substrings
    ):
        # Import narrowing is optional for protocols (a JSON-RPC string alone
        # can be enough for server endpoints), but if the signature declared
        # import substrings they act as an additional, soft narrowing layer.
        # We only treat missing imports as disqualifying when the signature
        # has NO other matching surface.
        has_other_surface = bool(
            sig.qualified_name_substrings or sig.string_literal_substrings
        )
        if not has_other_surface:
            return None

    # 1. Qualified-name substring on calls or assignments.
    for needle in sig.qualified_name_substrings:
        for qname, line in _calls_in_class(result, class_obs):
            if needle in qname:
                return AgentEvidenceMatch(
                    signature_id=sig.id,
                    evidence_pattern=sig.evidence_pattern,
                    file_path=result.file_path,
                    start_line=line,
                    end_line=line,
                    rationale=(
                        f"Protocol call '{qname}' at line {line} "
                        f"(signature '{sig.id}', protocol='{sig.protocol}', "
                        f"role='{sig.role}')."
                    ),
                )
        for qname, line in _assignments_in_class(result, class_obs):
            if needle in qname:
                return AgentEvidenceMatch(
                    signature_id=sig.id,
                    evidence_pattern=sig.evidence_pattern,
                    file_path=result.file_path,
                    start_line=line,
                    end_line=line,
                    rationale=(
                        f"Protocol assignment '{qname}' at line {line} "
                        f"(signature '{sig.id}', protocol='{sig.protocol}', "
                        f"role='{sig.role}')."
                    ),
                )

    # 2. Protocol-relevant string literal inside a method body of this class.
    for needle in sig.string_literal_substrings:
        for s in _strings_in_class(result, class_obs):
            if needle in s.value:
                return AgentEvidenceMatch(
                    signature_id=sig.id,
                    evidence_pattern=sig.evidence_pattern,
                    file_path=result.file_path,
                    start_line=s.line_number,
                    end_line=s.line_number,
                    rationale=(
                        f"Protocol string literal '{s.value}' at line {s.line_number} "
                        f"(signature '{sig.id}', protocol='{sig.protocol}', "
                        f"role='{sig.role}')."
                    ),
                )

    return None


# ---------------------------------------------------------------------------
# Match: anti-patterns
# ---------------------------------------------------------------------------


def _match_anti_pattern(
    sig: AgentAntiPatternSignature,
    result: CodeAnalysisResult,
    class_obs: ClassBodyFactsObservation,
) -> AgentAntiPatternMatch | None:
    # Narrow by import substrings if declared. Unlike framework matching,
    # missing imports here are disqualifying — we never want to flag a
    # class as "temporal_workflow" without the temporal import.
    if sig.import_substrings and not _file_imports_any_substring(
        result, sig.import_substrings
    ):
        return None

    # 1. Base-class exclusion.
    for needle in sig.base_class_names:
        if _base_class_matches(class_obs.base_classes, needle):
            return AgentAntiPatternMatch(
                signature_id=sig.id,
                label=sig.label,
                file_path=result.file_path,
                line_number=class_obs.start_line,
                rationale=(
                    f"Class '{class_obs.class_name}' inherits from '{needle}' "
                    f"(anti-pattern '{sig.label}')."
                ),
            )

    # 2. Decorator exclusion — any method decorated with one of the
    # declared decorators counts.
    for needle in sig.decorator_qualified_names:
        for qname, line in _decorators_in_class(result, class_obs):
            if _qualified_name_matches(qname, needle):
                return AgentAntiPatternMatch(
                    signature_id=sig.id,
                    label=sig.label,
                    file_path=result.file_path,
                    line_number=line,
                    rationale=(
                        f"Decorator '@{qname}' at line {line} "
                        f"(anti-pattern '{sig.label}')."
                    ),
                )

    return None


# ---------------------------------------------------------------------------
# Match: ReAct-style orchestration loops
# ---------------------------------------------------------------------------


# High-precision substrings that identify an LLM invocation call. The
# match is against the *lower-cased* fully qualified callee name (e.g.
# ``openai.chat.completions.create``, ``client.messages.create``,
# ``self.llm.invoke``). Loose verbs such as ``run``, ``stream``,
# ``generate``, ``complete``, ``completion``, ``predict`` were removed
# because they match legitimate non-LLM code (``loop.run``,
# ``stream.read``, ``random.generate``, ``transaction.complete``,
# ``sklearn.predict``) and were the dominant cause of false-positive
# ReAct-loop matches. A loop that does not contain one of these tokens
# is rejected as not-a-ReAct-loop.
_LLM_CALL_HINTS: tuple[str, ...] = (
    "invoke",          # LangChain / LangGraph: chain.invoke, llm.invoke
    "ainvoke",         # async variant
    "chat.completion", # OpenAI classic + .completions
    "chat_completion", # SDK wrappers
    "messages.create", # Anthropic
    "responses.create",# OpenAI Responses API
    ".llm.call",       # LangChain legacy LLM.call
    "llm_call",        # common wrapper name
    "llm_invoke",      # common wrapper name
    "acomplet",        # acompletion / acomplete
)


def _loop_has_llm_like_call(loop: ControlFlowObservation) -> bool:
    """Return True iff at least one body callee looks like an LLM call.

    High-precision check: see :data:`_LLM_CALL_HINTS` for the vetted
    substring set. Matching is done against the *lower-cased* qualified
    callee name, so ``client.chat.completions.create`` matches
    ``chat.completion``.
    """
    lowered = tuple(name.lower() for name in loop.body_call_qualified_names)
    return any(
        any(hint in name for hint in _LLM_CALL_HINTS) for name in lowered
    )


def _match_react_loop(
    policy: VerificationPolicy,
    result: CodeAnalysisResult,
    class_obs: ClassBodyFactsObservation,
) -> list[AgentEvidenceMatch]:
    """Find in-class loops whose structure resembles a ReAct orchestration.

    Structural requirements (all must hold):

    * Loop's owner is a method of this class.
    * Loop body has at least ``policy.min_react_loop_call_count`` calls
      total and ``policy.min_react_loop_distinct_callees`` distinct
      callees.
    * Loop body contains at least one branch (if/else) — dispatch.
    * Loop body contains at least one call whose qualified name matches
      a high-precision LLM-call hint (see :data:`_LLM_CALL_HINTS`).
      Previously this check was advisory; e2e results showed that
      without it the scanner tagged HTTP pollers and retry wrappers as
      agents. An agent that never calls an LLM is not an agent.
    """
    matches: list[AgentEvidenceMatch] = []
    for loop in _loops_in_class(result, class_obs):
        body_calls = loop.body_call_qualified_names
        distinct = len({name for name in body_calls if name})
        if len(body_calls) < policy.min_react_loop_call_count:
            continue
        if distinct < policy.min_react_loop_distinct_callees:
            continue
        if not loop.has_branch:
            continue
        if not _loop_has_llm_like_call(loop):
            continue

        matches.append(
            AgentEvidenceMatch(
                signature_id="structural.react_loop",
                evidence_pattern="react_loop",
                file_path=result.file_path,
                start_line=loop.start_line,
                end_line=loop.end_line,
                rationale=(
                    f"{loop.loop_kind} loop in method "
                    f"'{loop.owner_method_name or '?'}' "
                    f"(lines {loop.start_line}–{loop.end_line}) has "
                    f"{len(body_calls)} calls / {distinct} distinct callees "
                    f"with a conditional branch and an LLM-like call name."
                ),
            )
        )
    return matches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_dossier_for_class(
    catalog: AgentSignatureCatalog,
    result: CodeAnalysisResult,
    class_obs: ClassBodyFactsObservation,
) -> AgentEvidenceDossier:
    """Build a single dossier for one class in one file."""
    dossier = AgentEvidenceDossier(
        class_name=class_obs.class_name,
        qualified_name=class_obs.qualified_name,
        file_path=result.file_path,
        class_start_line=class_obs.start_line,
        class_end_line=class_obs.end_line,
        class_body_source=class_obs.body_source,
    )

    for sig in catalog.frameworks:
        match = _match_framework(sig, result, class_obs)
        if match is not None:
            dossier.framework_matches.append(match)

    for sig in catalog.protocols:
        match = _match_protocol(sig, result, class_obs)
        if match is not None:
            dossier.protocol_matches.append(match)

    dossier.react_loop_matches.extend(
        _match_react_loop(catalog.verification_policy, result, class_obs)
    )

    for sig in catalog.anti_patterns:
        anti = _match_anti_pattern(sig, result, class_obs)
        if anti is not None:
            dossier.anti_pattern_matches.append(anti)

    return dossier


def build_dossiers(
    catalog: AgentSignatureCatalog,
    results: Iterable[CodeAnalysisResult],
) -> list[AgentEvidenceDossier]:
    """Build dossiers for every class across a collection of files.

    Files with no :class:`ClassBodyFactsObservation` entries contribute
    nothing. Callers may filter the returned list by
    :attr:`AgentEvidenceDossier.has_direct_agent_evidence`, etc.
    """
    out: list[AgentEvidenceDossier] = []
    for result in results:
        for class_obs in result.class_bodies:
            out.append(build_dossier_for_class(catalog, result, class_obs))
    LOGGER.debug("Built %d agent-evidence dossier(s).", len(out))
    return out


# ---------------------------------------------------------------------------
# Rendering helpers for the LLM prompt / debug output
# ---------------------------------------------------------------------------


def render_dossier_for_prompt(dossier: AgentEvidenceDossier) -> dict[str, Any]:
    """Render a dossier into a JSON-safe dict for prompt injection.

    The shape is intentionally small and flat — the LLM gets structured
    facts, not a free-form narrative.
    """

    def _match(m: AgentEvidenceMatch) -> dict[str, Any]:
        return {
            "signature_id": m.signature_id,
            "pattern": m.evidence_pattern,
            "file_path": m.file_path,
            "start_line": m.start_line,
            "end_line": m.end_line,
            "rationale": m.rationale,
        }

    def _anti(m: AgentAntiPatternMatch) -> dict[str, Any]:
        return {
            "signature_id": m.signature_id,
            "label": m.label,
            "file_path": m.file_path,
            "line_number": m.line_number,
            "rationale": m.rationale,
        }

    return {
        "class_name": dossier.class_name,
        "qualified_name": dossier.qualified_name,
        "file_path": dossier.file_path,
        "class_start_line": dossier.class_start_line,
        "class_end_line": dossier.class_end_line,
        "preferred_pattern": dossier.preferred_pattern,
        "has_direct_agent_evidence": dossier.has_direct_agent_evidence,
        "has_remote_proxy_evidence": dossier.has_remote_proxy_evidence,
        "is_excluded_by_anti_pattern": dossier.is_excluded_by_anti_pattern,
        "framework_matches": [_match(m) for m in dossier.framework_matches],
        "protocol_matches": [_match(m) for m in dossier.protocol_matches],
        "react_loop_matches": [_match(m) for m in dossier.react_loop_matches],
        "anti_pattern_matches": [_anti(m) for m in dossier.anti_pattern_matches],
    }
