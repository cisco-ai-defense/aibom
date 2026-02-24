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

import ast
import libcst as cst
from typing import Any, Dict, List, Optional, Set, Union

from .structures import (
    AssignmentObservation,
    CallObservation,
    ClassDefObservation,
    CodeAnalysisResult,
    ContextManagerObservation,
    DecoratorObservation,
    FunctionAnnotationObservation,
    TypeAnnotationObservation,
)
from .custom_catalog import parse_inline_annotation


def _format_attribute_name(node: cst.Attribute) -> str:
    """Create a dotted string representation for an attribute expression."""
    parts: List[str] = []
    current: Union[cst.Attribute, cst.BaseExpression] = node
    while isinstance(current, cst.Attribute):
        attr = current.attr
        attr_name = attr.value if isinstance(attr, cst.Name) else str(attr)
        parts.append(attr_name)
        current = current.value
    if isinstance(current, cst.Name):
        parts.append(current.value)
    elif isinstance(current, cst.Attribute):
        # Should never happen due to loop, but keep for safety
        parts.append(str(current))
    else:
        parts.append(str(current))
    return ".".join(reversed(parts))


def _extract_argument_value(node: cst.BaseExpression) -> Any:
    """Tries to extract a literal or structured value from a CST node."""
    if isinstance(node, (cst.List, cst.Tuple, cst.Set)):
        values: List[Any] = []
        for element in node.elements:
            if element is None:
                continue
            value_node = element.value if hasattr(element, "value") else None
            if value_node is not None:
                values.append(_extract_argument_value(value_node))
        return values

    if isinstance(node, cst.Dict):
        dict_value: Dict[str, Any] = {}
        for element in node.elements:
            if not isinstance(element, cst.DictElement):
                continue
            key = _extract_argument_value(element.key) if element.key else None
            value = _extract_argument_value(element.value)
            key_str = str(key)
            dict_value[key_str] = value
        return dict_value

    if isinstance(node, cst.SimpleString):
        # Use ast.literal_eval to safely evaluate the string, handling quotes
        try:
            return ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return node.value  # Return raw string if eval fails
    if isinstance(node, cst.Integer | cst.Float):
        return ast.literal_eval(node.value)
    if isinstance(node, cst.Name) and node.value in ["True", "False", "None"]:
        return ast.literal_eval(node.value)

    if isinstance(node, cst.Name):
        # For complex nodes (lists, dicts, calls, etc.), we can't easily get a literal value.
        return f"VARIABLE:{node.value}"

    if isinstance(node, cst.Attribute):
        attr_name = _format_attribute_name(node)
        return f"ATTRIBUTE:{attr_name}"

    if isinstance(node, cst.Call):
        # Extract the call target name and flatten inner variable references
        # so that e.g. ToolNode(tools) → "CALL:ToolNode(VARIABLE:tools)"
        func_name = None
        if isinstance(node.func, cst.Name):
            func_name = node.func.value
        elif isinstance(node.func, cst.Attribute):
            func_name = _format_attribute_name(node.func)
        inner_parts = []
        for arg in node.args:
            inner_parts.append(_extract_argument_value(arg.value))
        if func_name:
            return {"_call": func_name, "_args": inner_parts}
        return {"_call": "unknown", "_args": inner_parts}

    return f"COMPLEX_TYPE:{type(node).__name__}"


class SymbolVisitor(cst.CSTVisitor):
    """A CST visitor that extracts assignments and calls, resolving qualified names."""

    METADATA_DEPENDENCIES = (
        cst.metadata.ParentNodeProvider,
        cst.metadata.PositionProvider,
        cst.metadata.QualifiedNameProvider,
    )

    def __init__(self, file_path: str, source_code: str = ""):
        self.file_path = file_path
        self.source_code = source_code
        self.result = CodeAnalysisResult(file_path=self.file_path)

    def _unwrap_call_expression(self, node: cst.BaseExpression) -> Optional[cst.Call]:
        """Return the underlying Call node, unwrapping Await expressions when needed."""
        if node is None:
            return None

        if isinstance(node, cst.Await):
            node = node.expression

        if isinstance(node, cst.Call):
            return node

        return None

    def _extract_target_name(self, node: cst.CSTNode) -> Optional[str]:
        """Retrieve a qualified name for a target node, falling back to simple names."""
        target_name = self._get_qualified_name_for_node(node)
        if target_name:
            return target_name

        if isinstance(node, cst.Name):
            return node.value

        return None

    def _get_qualified_name_for_node(self, node: cst.CSTNode) -> Optional[str]:
        """Safely retrieves the most likely qualified name for a given node."""
        try:
            qnames = self.get_metadata(cst.metadata.QualifiedNameProvider, node)
            if qnames:
                # A node can resolve to multiple names, e.g., if imported with an alias.
                # We prefer the one that seems most canonical.
                return sorted(list(qnames), key=lambda n: n.name)[0].name
        except (KeyError, IndexError):
            return None
        return None
    
    def _extract_raw_code(self, node: cst.CSTNode) -> str:
        """Extract raw code snippet from a CST node."""
        try:
            position = self.get_metadata(cst.metadata.PositionProvider, node)
            if self.source_code and position:
                lines = self.source_code.split('\n')
                start_line = position.start.line - 1  # Convert to 0-based indexing
                end_line = position.end.line - 1
                
                if start_line == end_line:
                    # Single line
                    line = lines[start_line] if start_line < len(lines) else ""
                    return line[position.start.column:position.end.column]
                else:
                    # Multi-line
                    result_lines = []
                    for i in range(start_line, min(end_line + 1, len(lines))):
                        if i == start_line:
                            result_lines.append(lines[i][position.start.column:])
                        elif i == end_line:
                            result_lines.append(lines[i][:position.end.column])
                        else:
                            result_lines.append(lines[i])
                    return '\n'.join(result_lines)
        except (KeyError, IndexError, AttributeError):
            pass
        return ""
    
    def _extract_aibom_annotation(self, node: cst.CSTNode) -> Optional[Dict[str, str]]:
        """Extract an ``# aibom: ...`` annotation from leading lines or trailing comment.

        Checks two places:
        1. Leading comment lines immediately above the statement.
        2. A trailing inline comment on the same line.
        """
        # Check leading lines (for ClassDef / FunctionDef these are in .leading_lines
        # or attached to decorators).  We use the source code directly for reliability.
        try:
            position = self.get_metadata(cst.metadata.PositionProvider, node)
            if self.source_code and position:
                lines = self.source_code.split("\n")
                start_line_idx = position.start.line - 1  # 0-based

                # Check the line itself for a trailing comment
                if 0 <= start_line_idx < len(lines):
                    annotation = parse_inline_annotation(lines[start_line_idx])
                    if annotation:
                        return annotation

                # Check the line immediately above for a leading comment
                prev_idx = start_line_idx - 1
                if 0 <= prev_idx < len(lines):
                    annotation = parse_inline_annotation(lines[prev_idx])
                    if annotation:
                        return annotation
        except (KeyError, IndexError, AttributeError):
            pass

        return None

    def leave_Import(self, original_node: cst.Import) -> None:
        """Capture import statements."""
        for alias in original_node.names:
            if isinstance(alias, cst.ImportAlias):
                import_name = alias.name.value if isinstance(alias.name, cst.Name) else str(alias.name)
                import_stmt = f"import {import_name}"
                if alias.asname:
                    import_stmt += f" as {alias.asname.name.value}"
                self.result.imports.append(import_stmt)
    
    def leave_ImportFrom(self, original_node: cst.ImportFrom) -> None:
        """Capture from...import statements."""
        if original_node.module:
            module_name = self._extract_raw_code(original_node.module)
            if isinstance(original_node.names, cst.ImportStar):
                import_stmt = f"from {module_name} import *"
                self.result.imports.append(import_stmt)
            elif hasattr(original_node.names, '__iter__'):
                imported_items = []
                for alias in original_node.names:
                    if isinstance(alias, cst.ImportAlias):
                        item_name = alias.name.value if isinstance(alias.name, cst.Name) else str(alias.name)
                        if alias.asname:
                            item_name += f" as {alias.asname.name.value}"
                        imported_items.append(item_name)
                if imported_items:
                    import_stmt = f"from {module_name} import {', '.join(imported_items)}"
                    self.result.imports.append(import_stmt)

    @staticmethod
    def _extract_all_arguments(args) -> Dict[str, Any]:
        """Extract both keyword and positional arguments from a call.

        Keyword arguments are stored under their name.  Positional arguments
        are stored as ``_pos_0``, ``_pos_1``, etc.
        """
        result: Dict[str, Any] = {}
        pos_idx = 0
        for arg in args:
            if arg.keyword:
                result[arg.keyword.value] = _extract_argument_value(arg.value)
            else:
                result[f"_pos_{pos_idx}"] = _extract_argument_value(arg.value)
                pos_idx += 1
        return result

    def leave_Call(self, original_node: cst.Call) -> None:
        """Captures standalone calls that are not part of an assignment."""
        try:
            parent = self.get_metadata(cst.metadata.ParentNodeProvider, original_node)
            if isinstance(parent, cst.Assign):
                return
        except KeyError:
            pass

        qualified_name = self._get_qualified_name_for_node(original_node.func)
        if not qualified_name and isinstance(original_node.func, cst.Attribute):
            qualified_name = _format_attribute_name(original_node.func)
        if not qualified_name:
            return

        args = self._extract_all_arguments(original_node.args)

        call_obs = CallObservation(
            qualified_name=qualified_name,
            arguments=args,
            line_number=self.get_metadata(cst.metadata.PositionProvider, original_node).start.line,
            raw_code=self._extract_raw_code(original_node),
        )
        self.result.calls.append(call_obs)

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        """Capture class definitions: base classes and ``# aibom:`` annotations."""
        class_name = original_node.name.value

        qualified_name = self._get_qualified_name_for_node(original_node.name)

        base_classes: List[str] = []
        for arg in original_node.bases:
            base_expr = arg.value
            base_name = self._get_qualified_name_for_node(base_expr)
            if not base_name and isinstance(base_expr, cst.Attribute):
                base_name = _format_attribute_name(base_expr)
            if not base_name and isinstance(base_expr, cst.Name):
                base_name = base_expr.value
            if base_name:
                base_classes.append(base_name)

        annotation = self._extract_aibom_annotation(original_node)

        try:
            line_number = self.get_metadata(
                cst.metadata.PositionProvider, original_node
            ).start.line
        except (KeyError, AttributeError):
            line_number = 0

        obs = ClassDefObservation(
            class_name=class_name,
            qualified_name=qualified_name,
            base_classes=base_classes,
            line_number=line_number,
            aibom_annotation=annotation,
        )
        self.result.class_defs.append(obs)

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        """Captures functions that have decorators and/or ``# aibom:`` annotations."""
        # Check for inline aibom annotation
        annotation = self._extract_aibom_annotation(original_node)
        if annotation:
            func_name = original_node.name.value
            qualified_name = self._get_qualified_name_for_node(original_node.name)
            try:
                line_number = self.get_metadata(
                    cst.metadata.PositionProvider, original_node
                ).start.line
            except (KeyError, AttributeError):
                line_number = 0

            self.result.function_annotations.append(
                FunctionAnnotationObservation(
                    function_name=func_name,
                    qualified_name=qualified_name,
                    line_number=line_number,
                    aibom_annotation=annotation,
                )
            )

        # Original decorator handling
        if not original_node.decorators:
            return

        for decorator_node in original_node.decorators:
            decorator_expr: cst.CSTNode = decorator_node.decorator

            decorator_name = self._get_qualified_name_for_node(decorator_expr)
            if not decorator_name and isinstance(decorator_expr, cst.Call):
                decorator_name = self._get_qualified_name_for_node(decorator_expr.func)
            if not decorator_name:
                continue

            instance_variable = None
            target_expr = decorator_expr.func if isinstance(decorator_expr, cst.Call) else decorator_expr
            if isinstance(target_expr, cst.Attribute):
                if isinstance(target_expr.value, cst.Name):
                    instance_variable = target_expr.value.value

            obs = DecoratorObservation(
                decorator_qualified_name=decorator_name,
                decorated_function_name=original_node.name.value,
                line_number=self.get_metadata(cst.metadata.PositionProvider, decorator_node).start.line,
                instance_variable=instance_variable,
            )
            self.result.decorators.append(obs)

    def leave_Assign(self, original_node: cst.Assign) -> None:
        """Captures assignments where the value is a class instantiation (a Call)."""
        call_node = self._unwrap_call_expression(original_node.value)
        if call_node is None:
            return

        qualified_name = self._get_qualified_name_for_node(call_node.func)
        if not qualified_name and isinstance(call_node.func, cst.Attribute):
            qualified_name = _format_attribute_name(call_node.func)
        if not qualified_name:
            return

        # Process arguments (keyword + positional)
        args = self._extract_all_arguments(call_node.args)

        call_obs = CallObservation(
            qualified_name=qualified_name,
            arguments=args,
            line_number=self.get_metadata(cst.metadata.PositionProvider, call_node).start.line,
            raw_code=self._extract_raw_code(original_node),
        )

        # Process assignment target
        target_node = original_node.targets[0].target
        target_name = self._extract_target_name(target_node)
        if not target_name:
            return

        assignment_obs = AssignmentObservation(
            target_qualified_name=target_name,
            call=call_obs,
            line_number=self.get_metadata(cst.metadata.PositionProvider, original_node).start.line,
        )
        self.result.assignments.append(assignment_obs)

    def leave_AnnAssign(self, original_node: cst.AnnAssign) -> None:
        """Captures annotated assignments, recording both annotation and calls."""
        annotation_node = original_node.annotation.annotation if original_node.annotation else None
        if annotation_node:
            annotation_name = self._get_qualified_name_for_node(annotation_node)
            if annotation_name:
                target_name = self._extract_target_name(original_node.target)
                if target_name:
                    ann_obs = TypeAnnotationObservation(
                        target_qualified_name=target_name,
                        annotation_qualified_name=annotation_name,
                        line_number=self.get_metadata(cst.metadata.PositionProvider, original_node).start.line,
                    )
                    self.result.type_annotations.append(ann_obs)

        call_node = self._unwrap_call_expression(original_node.value)
        if call_node is None:
            return

        qualified_name = self._get_qualified_name_for_node(call_node.func)
        if not qualified_name:
            return

        args = self._extract_all_arguments(call_node.args)

        call_obs = CallObservation(
            qualified_name=qualified_name,
            arguments=args,
            line_number=self.get_metadata(cst.metadata.PositionProvider, call_node).start.line,
            raw_code=self._extract_raw_code(original_node),
        )

        target_name = self._extract_target_name(original_node.target)
        if not target_name:
            return

        assignment_obs = AssignmentObservation(
            target_qualified_name=target_name,
            call=call_obs,
            line_number=self.get_metadata(cst.metadata.PositionProvider, original_node).start.line,
        )
        self.result.assignments.append(assignment_obs)

    def _handle_with_items(self, items) -> None:
        for item in items:
            context_expr = getattr(item, "context_expr", None)
            if context_expr is None:
                context_expr = getattr(item, "item", None)
            call_node = None
            if isinstance(context_expr, cst.Call):
                call_node = context_expr
            else:
                call_node = self._unwrap_call_expression(context_expr)

            if call_node is None:
                continue

            qualified_name = self._get_qualified_name_for_node(call_node.func)
            if not qualified_name:
                continue

            target_name = None
            optional_vars = getattr(item, "optional_vars", None)
            if optional_vars:
                target_name = self._extract_target_name(optional_vars)
            else:
                asname = getattr(item, "asname", None)
                if asname and hasattr(asname, "name"):
                    target_name = self._extract_target_name(asname.name)

            position = self.get_metadata(cst.metadata.PositionProvider, context_expr)
            ctx_obs = ContextManagerObservation(
                context_expr_qualified_name=qualified_name,
                as_target=target_name,
                line_number=position.start.line if position else 0,
            )
            self.result.context_managers.append(ctx_obs)

    def leave_With(self, original_node) -> None:
        self._handle_with_items(original_node.items)

    def leave_AsyncWith(self, original_node) -> None:
        self._handle_with_items(original_node.items)

def parse_source_code(file_path: str, source_code: str) -> CodeAnalysisResult:
    """
    Orchestrates the parsing of a single source file.

    Args:
        file_path: The path to the file being parsed.
        source_code: The string content of the Python file.

    Returns:
        A CodeAnalysisResult object containing all observations from the file.
    """
    try:
        tree = cst.parse_module(source_code)
    except cst.ParserSyntaxError:
        # If there's a syntax error, return an empty result for this file.
        return CodeAnalysisResult(file_path=file_path)

    wrapper = cst.metadata.MetadataWrapper(tree)
    visitor = SymbolVisitor(file_path=file_path, source_code=source_code)
    wrapper.visit(visitor)

    return visitor.result
