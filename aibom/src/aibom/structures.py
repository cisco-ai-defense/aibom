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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class CallObservation:
    """Represents a function or method call."""
    qualified_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    line_number: int = 0
    raw_code: str = ""  # Raw code snippet for LLM analysis


@dataclass
class AssignmentObservation:
    """Represents an assignment of a class instantiation."""
    target_qualified_name: str
    call: CallObservation
    line_number: int = 0


@dataclass
class DecoratorObservation:
    """Represents a decorator applied to a function."""
    decorator_qualified_name: str
    decorated_function_name: str
    line_number: int = 0
    instance_variable: Optional[str] = None  # The variable if decorator is an attribute, e.g., 'app' in @app.get(...)


@dataclass
class TypeAnnotationObservation:
    """Represents an annotated assignment, capturing the annotation and the target."""
    target_qualified_name: str
    annotation_qualified_name: str
    line_number: int = 0


@dataclass
class ContextManagerObservation:
    """Represents usage of a callable within a with or async with block."""
    context_expr_qualified_name: str
    as_target: Optional[str]
    line_number: int = 0


@dataclass
class ClassDefObservation:
    """Represents a class definition with its base classes and optional aibom annotation."""
    class_name: str
    qualified_name: Optional[str] = None
    base_classes: List[str] = field(default_factory=list)
    line_number: int = 0
    aibom_annotation: Optional[Dict[str, str]] = None


@dataclass
class FunctionAnnotationObservation:
    """Represents a function/method tagged with an ``# aibom:`` inline annotation."""
    function_name: str
    qualified_name: Optional[str] = None
    line_number: int = 0
    aibom_annotation: Dict[str, str] = field(default_factory=dict)


@dataclass
class ControlFlowObservation:
    """A ``while`` / ``for`` / ``async for`` loop inside a function or method.

    Captures enough structure to let downstream matchers recognize a
    ReAct-style orchestration loop (loop body that calls an LLM and
    dispatches to a tool based on the result) without the parser itself
    encoding any framework names.
    """

    owner_qualified_name: Optional[str]
    owner_class_name: Optional[str]
    owner_method_name: Optional[str]
    loop_kind: str
    start_line: int = 0
    end_line: int = 0
    body_call_qualified_names: List[str] = field(default_factory=list)
    has_branch: bool = False


@dataclass
class MethodBodyShapeObservation:
    """Per-function/method structural summary.

    The YAML signature matcher (phase 2) uses these counts to distinguish a
    single-LLM-call wrapper (``call_count <= 2``, ``loop_count == 0``) from
    a tool-using orchestrator (``loop_count >= 1`` and at least two distinct
    called qualified names inside the loop body).

    ``called_qualified_names`` only covers calls in *this* function's body
    and stops at nested ``def``/``class``/``lambda`` boundaries so that
    inner helpers do not inflate the outer method's call count.
    """

    owner_qualified_name: Optional[str]
    owner_class_name: Optional[str]
    method_name: str
    start_line: int = 0
    end_line: int = 0
    statement_count: int = 0
    call_count: int = 0
    loop_count: int = 0
    branch_count: int = 0
    return_count: int = 0
    called_qualified_names: List[str] = field(default_factory=list)


@dataclass
class StringLiteralObservation:
    """A protocol-relevant string literal inside a function or method body.

    Only literals that look like URLs, filesystem/HTTP paths, or JSON-RPC
    method identifiers (e.g. ``message/send``) are recorded. All other
    strings are filtered out at parse time to keep the observation volume
    bounded — A2A / MCP / endpoint matchers in downstream scanners decide
    whether a given captured value is actually interesting.
    """

    owner_class_name: Optional[str]
    owner_method_name: Optional[str]
    line_number: int
    value: str


@dataclass
class ClassBodyFactsObservation:
    """Per-class aggregation of line range, base classes, method names, and
    the full class source text.

    The LLM prompt (phase 5) and the evidence-builder (phase 2) inject
    ``body_source`` verbatim so the LLM reasons over the entire class
    without re-reading the source file from disk.
    """

    class_name: str
    qualified_name: Optional[str]
    start_line: int = 0
    end_line: int = 0
    base_classes: List[str] = field(default_factory=list)
    method_names: List[str] = field(default_factory=list)
    body_source: str = ""
    class_decorators: List[str] = field(default_factory=list)


@dataclass
class CodeAnalysisResult:
    """Holds all observations from a single source file analysis."""
    file_path: str
    assignments: List[AssignmentObservation] = field(default_factory=list)
    calls: List[CallObservation] = field(default_factory=list)
    decorators: List[DecoratorObservation] = field(default_factory=list)
    type_annotations: List[TypeAnnotationObservation] = field(default_factory=list)
    context_managers: List[ContextManagerObservation] = field(default_factory=list)
    class_defs: List[ClassDefObservation] = field(default_factory=list)
    function_annotations: List[FunctionAnnotationObservation] = field(default_factory=list)
    imports: List[tuple[int, str]] = field(default_factory=list)  # (line_number, import_stmt)
    control_flows: List[ControlFlowObservation] = field(default_factory=list)
    method_shapes: List[MethodBodyShapeObservation] = field(default_factory=list)
    protocol_strings: List[StringLiteralObservation] = field(default_factory=list)
    class_bodies: List[ClassBodyFactsObservation] = field(default_factory=list)

    def get_all_qualified_names(self) -> Set[str]:
        """Returns a set of all unique qualified names found in the file."""
        names: Set[str] = set()
        for assignment in self.assignments:
            names.add(assignment.call.qualified_name)
        for call in self.calls:
            names.add(call.qualified_name)
        for decorator in self.decorators:
            names.add(decorator.decorator_qualified_name)
        for annotation in self.type_annotations:
            names.add(annotation.annotation_qualified_name)
        for ctx in self.context_managers:
            names.add(ctx.context_expr_qualified_name)
        return names


@dataclass
class ComponentRelationship:
    """Represents a relationship between two component instances."""
    source_instance_id: str
    target_instance_id: str
    label: str
    source_name: str
    target_name: str
    source_category: str
    target_category: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the relationship for JSON/plaintext outputs."""
        return {
            "source_instance_id": self.source_instance_id,
            "target_instance_id": self.target_instance_id,
            "label": self.label,
            "source_name": self.source_name,
            "target_name": self.target_name,
            "source_category": self.source_category,
            "target_category": self.target_category,
        }


@dataclass
class CategorizationOutput:
    """Container for categorized components and their relationships."""
    components: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    relationships: List[ComponentRelationship] = field(default_factory=list)
