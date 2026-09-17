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

"""Derive component relationships from code structure.

Before this module, every ``USES_*`` edge between two components detected
in Python source came from the LLM, and no gate re-read the source to
confirm one. That matters because ``_propagate_model_from_relationships``
copies an edge's target name straight into a component's ``model_name``,
so a hallucinated edge becomes a reported model.

Edges come from two shapes. A constructor keyword is emitted only when all
three hold:

1. the source component records the constructor kwargs it was built with;
2. a kwarg whose name implies a dependency (``llm=``, ``tools=``) refers to
   a variable;
3. that variable is the binding site of another detected component whose
   type is valid for that kwarg.

The second shape is a framework wiring call. LangGraph attaches nodes with
``builder.add_node(...)`` rather than constructor keywords, so keyword rules
alone see nothing in the template repositories that make up most modern
LangChain code. There the edge kind comes from the target's own type,
because the call site does not name a role.

Each edge carries the source location of the call and of the target's
binding, so it can be audited the same way agent evidence is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

from ..models.enums import (
    AIComponentType,
    DetectionSource,
    EvidenceStrength,
    RelationshipType,
)
from ..models.scan import (
    AIComponent,
    ComponentRelationship,
    DecisionAnnotation,
    EvidenceLocation,
)
from .models import CodeGraph, MethodCall

_LOGGER = logging.getLogger(__name__)

_VARIABLE_TAG = "VARIABLE:"
_ATTRIBUTE_TAG = "ATTRIBUTE:"

# Methods that attach one component to another. ``add_edge`` is absent on
# purpose: it links two node *names*, which says nothing about components.
_WIRING_METHODS = frozenset({"add_node"})

# How far to follow call edges out of a node function. Two hops covers the
# usual "node -> factory helper -> construction" shape; beyond that the
# reachable set describes the program rather than the node.
MAX_WIRING_DEPTH = 2

STATED = EvidenceStrength.STATED

# Past a handful of candidates the reference is not "ambiguous between a
# few things", it is a common word, and listing them all is noise.
MAX_AMBIGUOUS_CANDIDATES = 4

# Keywords whose value is the model itself rather than a component that
# holds one, so the string they resolve to is the model name.
_MODEL_KWARGS = frozenset(
    {
        "model",
        "model_name",
        "model_id",
        "model_version",
        "deployment_name",
        "azure_deployment",
        "embed_model",
        "embedding_model",
    }
)


@dataclass(frozen=True)
class EdgeRule:
    """A constructor kwarg family and the target types it may point at.

    The kwarg name alone is not enough: ``store=`` means a memory in one
    framework and a vector store in another. Requiring the target's
    detected type to match keeps the derivation honest, and lets two rules
    share a kwarg without producing a wrong edge.
    """

    kwargs: frozenset[str]
    targets: frozenset[AIComponentType]
    relationship: RelationshipType


_RULES: tuple[EdgeRule, ...] = (
    EdgeRule(
        frozenset({"tool", "tools", "toolkit", "toolkits", "abilities"}),
        frozenset({AIComponentType.TOOL}),
        RelationshipType.USES_TOOL,
    ),
    EdgeRule(
        frozenset({"llm", "model", "language_model", "chat_model", "llm_model"}),
        frozenset({AIComponentType.MODEL}),
        RelationshipType.USES_MODEL,
    ),
    EdgeRule(
        frozenset({"llm", "model", "language_model", "chat_model", "llm_model"}),
        frozenset({AIComponentType.LLM_ENDPOINT, AIComponentType.MODEL_ENDPOINT}),
        RelationshipType.USES_LLM_ENDPOINT,
    ),
    EdgeRule(
        frozenset({"memory", "checkpointer", "saver", "chat_history", "store"}),
        frozenset({AIComponentType.MEMORY}),
        RelationshipType.USES_MEMORY,
    ),
    EdgeRule(
        frozenset({"retriever", "retrievers"}),
        frozenset({AIComponentType.RETRIEVER}),
        RelationshipType.USES_RETRIEVER,
    ),
    EdgeRule(
        frozenset(
            {"embedding", "embeddings", "embedding_function", "embed_model", "embedder"}
        ),
        frozenset({AIComponentType.EMBEDDING}),
        RelationshipType.USES_EMBEDDING,
    ),
    EdgeRule(
        frozenset(
            {"agent", "agents", "sub_agents", "subagents", "handoffs", "delegates"}
        ),
        frozenset({AIComponentType.AGENT, AIComponentType.AGENT_PROXY}),
        RelationshipType.USES_AGENT,
    ),
    EdgeRule(
        frozenset({"skill", "skills", "plugins"}),
        frozenset({AIComponentType.SKILL}),
        RelationshipType.USES_SKILL,
    ),
    EdgeRule(
        frozenset({"vectorstore", "vector_store", "vectordb", "vector_db", "store"}),
        frozenset({AIComponentType.VECTOR_STORE}),
        RelationshipType.USES_VECTOR_STORE,
    ),
    EdgeRule(
        frozenset({"guardrail", "guardrails", "input_guardrails"}),
        frozenset({AIComponentType.GUARDRAIL}),
        RelationshipType.USES_GUARDRAIL,
    ),
    EdgeRule(
        frozenset({"dataset", "datasets", "train_dataset", "eval_dataset"}),
        frozenset({AIComponentType.DATASET}),
        RelationshipType.USES_DATASET,
    ),
    EdgeRule(
        frozenset({"knowledge_base", "knowledge_bases", "kb"}),
        frozenset({AIComponentType.KNOWLEDGE_BASE}),
        RelationshipType.USES_KNOWLEDGE_BASE,
    ),
)


def _extract_references(value: Any) -> set[str]:
    """Pull ``VARIABLE:``/``ATTRIBUTE:`` names out of a kwarg value.

    Containers are walked because ``tools=[search, calc]`` is the common
    shape and each element is a separate reference.
    """
    refs: set[str] = set()
    if isinstance(value, str):
        if value.startswith(_VARIABLE_TAG):
            refs.add(value[len(_VARIABLE_TAG) :])
        elif value.startswith(_ATTRIBUTE_TAG):
            refs.add(value[len(_ATTRIBUTE_TAG) :])
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            refs |= _extract_references(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == "_call":
                continue
            refs |= _extract_references(item)
    return refs


def _inline_call_names(value: Any) -> set[str]:
    """Pull class names out of constructions nested in an argument.

    ``add_node("tools", ToolNode(TOOLS))`` builds its component inside the
    call, so there is no variable to follow — the only handle on it is the
    ``_call`` name the parser records.
    """
    names: set[str] = set()
    if isinstance(value, dict):
        called = value.get("_call")
        if isinstance(called, str) and called:
            names.add(called)
        for key, item in value.items():
            if key == "_call":
                continue
            names |= _inline_call_names(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            names |= _inline_call_names(item)
    return names


class _ComponentIndex:
    """Resolves a variable reference to the component bound to it."""

    def __init__(self, components: Iterable[AIComponent]) -> None:
        self._by_var: dict[tuple[str, str], AIComponent] = {}
        self._by_name: dict[str, list[AIComponent]] = {}
        self._by_file: dict[str, list[AIComponent]] = {}
        for comp in components:
            target = (comp.metadata or {}).get("assigned_target")
            if isinstance(target, str) and target:
                bare = target.split(".")[-1]
                self._by_var.setdefault((comp.file_path, bare), comp)
            if comp.name:
                self._by_name.setdefault(comp.name, []).append(comp)
            self._by_file.setdefault(comp.file_path, []).append(comp)

    def in_range(self, file_path: str, start: int, end: int) -> list[AIComponent]:
        """Components declared inside a line span of one file."""
        return [
            comp
            for comp in self._by_file.get(file_path, ())
            if start <= comp.line_number <= end
        ]

    def same_file_candidates(self, reference: str, file_path: str) -> list[AIComponent]:
        """Components in this file that a colliding name could refer to.

        Only consulted once :meth:`resolve` has already declined, so this
        is the set behind an ambiguous reference rather than a second
        guess at the right answer.
        """
        bare = reference.split(".")[-1]
        return [
            comp
            for comp in self._by_file.get(file_path, ())
            if comp.name in (reference, bare)
        ]

    def resolve(self, reference: str, file_path: str) -> AIComponent | None:
        """Same-file binding first, then an unambiguous global name.

        A name matching several components resolves to nothing rather than
        to a guess; picking one would reintroduce exactly the unverified
        edge this module exists to remove. Callers that want to record the
        near-miss ask :meth:`same_file_candidates` afterwards.
        """
        bare = reference.split(".")[-1]
        for key in ((file_path, reference), (file_path, bare)):
            found = self._by_var.get(key)
            if found is not None:
                return found
        for candidate in (reference, bare):
            matches = self._by_name.get(candidate, [])
            if len(matches) == 1:
                return matches[0]
        return None


def _annotation(
    source: AIComponent, target: AIComponent, kwarg: str
) -> DecisionAnnotation:
    return DecisionAnnotation(
        decision="KEEP",
        justification=(
            f"Constructor keyword '{kwarg}' of '{source.name}' references "
            f"'{target.name}' at {source.file_path}:{source.line_number}. "
            f"Derived from code structure, not inferred."
        ),
        evidence_kinds=["code_graph", "constructor_argument"],
        evidence_locations=[
            EvidenceLocation(
                file_path=source.file_path,
                start_line=source.line_number,
                end_line=source.line_number,
                role="constructor_call",
            ),
            EvidenceLocation(
                file_path=target.file_path,
                start_line=target.line_number,
                end_line=target.line_number,
                role="referenced_component",
            ),
        ],
    )


def _in_scope(
    graph: CodeGraph | None, source: AIComponent, target: AIComponent
) -> bool:
    """Reject a reference that crosses a function boundary.

    ``agent = Agent(llm=client)`` in one function and ``client = ...`` in
    another are two different variables that happen to share a name.
    Without the graph there is no way to tell them apart, so this check is
    skipped when no graph is available rather than guessing.
    """
    if graph is None or source.file_path != target.file_path:
        return True
    target_scope = graph.enclosing_function(target.file_path, target.line_number)
    if target_scope is None:
        # Module-level binding is visible everywhere in the file.
        return True
    source_scope = graph.enclosing_function(source.file_path, source.line_number)
    if source_scope is None:
        return False
    return source_scope.node_id == target_scope.node_id


def resolve_literal_model_names(
    components: list[AIComponent],
    graph: CodeGraph,
) -> dict[str, str]:
    """Map components to the model string their ``model=`` kwarg holds.

    ``ChatOpenAI(model=expt_llm)`` records the kwarg as a reference, so the
    model a scan reports is a variable name or nothing at all. The binding
    table already knows what that variable holds; this reads it.

    Only literals are accepted. A name that resolves to another reference,
    a subscript, or an attribute is left alone rather than reported as a
    model, because that is a value this cannot evaluate.
    """
    found: dict[str, str] = {}
    for comp in components:
        if comp.model_name:
            continue
        arguments = (comp.metadata or {}).get("arguments")
        if not isinstance(arguments, dict):
            continue
        scope = graph.enclosing_function(comp.file_path, comp.line_number)
        for kwarg, value in arguments.items():
            if not isinstance(kwarg, str) or kwarg.lower() not in _MODEL_KWARGS:
                continue
            if not isinstance(value, str) or not value.startswith(_VARIABLE_TAG):
                continue
            binding = graph.resolve_value(
                comp.file_path,
                value[len(_VARIABLE_TAG) :],
                owner=scope.qualified_name if scope else None,
                before_line=comp.line_number,
            )
            if binding is None or binding.kind != "literal":
                continue
            if isinstance(binding.value, str) and binding.value.strip():
                found[comp.instance_id] = binding.value
                break
    if found:
        _LOGGER.info("Resolved %d model name(s) from literal bindings", len(found))
    return found


def _relationship_for(component_type: AIComponentType) -> RelationshipType | None:
    """The edge kind implied by what sits at the far end.

    Constructor keywords name the role directly (``llm=``), but a wiring
    call does not: ``add_node`` says only that something was attached, so
    the target's own type is the only thing that can classify the edge.
    """
    for rule in _RULES:
        if component_type in rule.targets:
            return rule.relationship
    return None


def _edge(
    source: AIComponent,
    target: AIComponent,
    relationship: RelationshipType,
    annotation: DecisionAnnotation,
    strength: EvidenceStrength,
) -> ComponentRelationship:
    return ComponentRelationship(
        source_instance_id=source.instance_id,
        target_instance_id=target.instance_id,
        relationship_type=relationship,
        source_name=source.name,
        target_name=target.name,
        source_type=source.component_type,
        target_type=target.component_type,
        detection_source=DetectionSource.CODE_ANALYSIS,
        evidence_strength=strength,
        decision_annotation=annotation,
    )


def _wiring_annotation(
    source: AIComponent, target: AIComponent, call: MethodCall, how: str
) -> DecisionAnnotation:
    return DecisionAnnotation(
        decision="KEEP",
        justification=(
            f"'{source.name}' is wired to '{target.name}' by "
            f"{call.receiver}.{call.method}() at "
            f"{call.file_path}:{call.line_number} ({how}). "
            f"Derived from code structure, not inferred."
        ),
        evidence_kinds=["code_graph", "graph_wiring"],
        evidence_locations=[
            EvidenceLocation(
                file_path=call.file_path,
                start_line=call.line_number,
                end_line=call.line_number,
                role="wiring_call",
            ),
            EvidenceLocation(
                file_path=target.file_path,
                start_line=target.line_number,
                end_line=target.line_number,
                role="attached_component",
            ),
        ],
    )


def _node_body_targets(
    reference: str,
    call: MethodCall,
    graph: CodeGraph,
    index: _ComponentIndex,
) -> list[tuple[AIComponent, str]]:
    """Components built inside a function attached as a graph node.

    ``builder.add_node(call_model)`` attaches a function, not a component,
    so what the graph uses is whatever that function constructs. The call
    edges are followed a short distance because the construction is
    routinely one helper away (``call_model`` -> ``load_chat_model``), but
    only a short distance: past a couple of hops the reachable set stops
    describing this node and starts describing the whole program.
    """
    node = next(
        (
            fn
            for fn in graph.functions_in_file(call.file_path)
            if fn.method_name == reference
        ),
        None,
    )
    if node is None:
        return []

    found: list[tuple[AIComponent, str]] = []
    reached = [(node, "built in the attached node function")] + [
        (callee, f"built in {callee.method_name}(), reached from {node.method_name}()")
        for callee in graph.transitive_callees(node.node_id, MAX_WIRING_DEPTH)
    ]
    for fn, how in reached:
        for comp in index.in_range(fn.file_path, fn.start_line, fn.end_line):
            found.append((comp, how))
    return found


def _derive_wiring_edges(
    graph: CodeGraph,
    index: _ComponentIndex,
    emit: Any,
) -> None:
    """Read edges off framework wiring calls such as ``add_node``."""
    for file_path, calls in graph.method_calls.items():
        for call in calls:
            if call.method not in _WIRING_METHODS:
                continue
            source = index.resolve(call.receiver, file_path)
            if source is None:
                continue
            for value in call.arguments.values():
                candidates: list[tuple[AIComponent, str, EvidenceStrength]] = []
                for name in sorted(_inline_call_names(value)):
                    target = index.resolve(name, file_path)
                    if target is not None:
                        candidates.append((target, "constructed in the call", STATED))
                for reference in sorted(_extract_references(value)):
                    target = index.resolve(reference, file_path)
                    if target is not None:
                        candidates.append((target, "passed by name", STATED))
                    else:
                        # The argument is a function, not a component, so
                        # the link runs through what that function builds.
                        candidates.extend(
                            (comp, how, EvidenceStrength.REACHED)
                            for comp, how in _node_body_targets(
                                reference, call, graph, index
                            )
                        )
                for target, how, strength in candidates:
                    if target is source:
                        continue
                    relationship = _relationship_for(target.component_type)
                    if relationship is None:
                        continue
                    emit(
                        source,
                        target,
                        relationship,
                        _wiring_annotation(source, target, call, how),
                        strength,
                    )


def _ambiguous_annotation(
    source: AIComponent, target: AIComponent, kwarg: str, rivals: int
) -> DecisionAnnotation:
    return DecisionAnnotation(
        decision="REVIEW",
        justification=(
            f"Constructor keyword '{kwarg}' of '{source.name}' at "
            f"{source.file_path}:{source.line_number} names a symbol that "
            f"matches {rivals} components of a valid type. '{target.name}' "
            f"is one of them; which one is meant cannot be decided from "
            f"code structure alone."
        ),
        evidence_kinds=["code_graph", "ambiguous_reference"],
        evidence_locations=[
            EvidenceLocation(
                file_path=source.file_path,
                start_line=source.line_number,
                end_line=source.line_number,
                role="constructor_call",
            ),
            EvidenceLocation(
                file_path=target.file_path,
                start_line=target.line_number,
                end_line=target.line_number,
                role="candidate_component",
            ),
        ],
    )


def _emit_ambiguous(
    source: AIComponent,
    kwarg: str,
    reference: str,
    rule: EdgeRule,
    index: _ComponentIndex,
    graph: CodeGraph | None,
    emit: Any,
) -> None:
    """Record a reference that matched several components.

    Dropping these outright loses a relationship that is genuinely in the
    code and leaves nothing for anyone to look at. Emitting every
    candidate marked ``AMBIGUOUS`` keeps the finding visible while making
    it obvious that no single answer was established.
    """
    candidates = [
        comp
        for comp in index.same_file_candidates(reference, source.file_path)
        if comp is not source
        and comp.component_type in rule.targets
        and _in_scope(graph, source, comp)
    ]
    if len(candidates) < 2 or len(candidates) > MAX_AMBIGUOUS_CANDIDATES:
        return
    for candidate in candidates:
        emit(
            source,
            candidate,
            rule.relationship,
            _ambiguous_annotation(source, candidate, kwarg, len(candidates)),
            EvidenceStrength.AMBIGUOUS,
        )


def derive_relationships(
    components: list[AIComponent],
    graph: CodeGraph | None = None,
) -> list[ComponentRelationship]:
    """Derive ``USES_*`` edges from constructor keywords and graph wiring.

    Passing *graph* enables the scope check, which drops references whose
    binding lives in a different function than the constructor call, and
    unlocks wiring calls, which exist only in the parsed call sites.
    """
    index = _ComponentIndex(components)
    edges: list[ComponentRelationship] = []
    seen: set[tuple[str, str, str]] = set()

    def emit(source, target, relationship, annotation, strength=STATED) -> None:
        dedup_key = (source.instance_id, target.instance_id, relationship.value)
        if dedup_key in seen:
            return
        seen.add(dedup_key)
        edges.append(_edge(source, target, relationship, annotation, strength))

    for source in components:
        arguments = (source.metadata or {}).get("arguments")
        if not isinstance(arguments, dict):
            continue
        for kwarg, value in arguments.items():
            if not isinstance(kwarg, str) or kwarg.startswith("_pos_"):
                continue
            key = kwarg.lower()
            references = _extract_references(value)
            if not references:
                continue
            for rule in _RULES:
                if key not in rule.kwargs:
                    continue
                for reference in sorted(references):
                    target = index.resolve(reference, source.file_path)
                    if target is None:
                        _emit_ambiguous(
                            source, kwarg, reference, rule, index, graph, emit
                        )
                        continue
                    if target is source:
                        continue
                    if target.component_type not in rule.targets:
                        continue
                    if not _in_scope(graph, source, target):
                        continue
                    emit(
                        source,
                        target,
                        rule.relationship,
                        _annotation(source, target, kwarg),
                    )

    if graph is not None:
        _derive_wiring_edges(graph, index, emit)

    if edges:
        _LOGGER.info("Derived %d relationship(s) from code structure", len(edges))
    return edges
