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
class CodeAnalysisResult:
    """Holds all observations from a single source file analysis."""
    file_path: str
    assignments: List[AssignmentObservation] = field(default_factory=list)
    calls: List[CallObservation] = field(default_factory=list)
    decorators: List[DecoratorObservation] = field(default_factory=list)
    type_annotations: List[TypeAnnotationObservation] = field(default_factory=list)
    context_managers: List[ContextManagerObservation] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)  # Import statements for disambiguation

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
