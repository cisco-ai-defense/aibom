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
from typing import Any, Dict, List, Optional, Union

from .structures import (
    AssignmentObservation,
    CallObservation,
    ClassBodyFactsObservation,
    ClassDefObservation,
    CodeAnalysisResult,
    ContextManagerObservation,
    ControlFlowObservation,
    DecoratorObservation,
    FunctionAnnotationObservation,
    MethodBodyShapeObservation,
    StringLiteralObservation,
    TypeAnnotationObservation,
)
from .custom_catalog import parse_inline_annotation


_MAX_INTERESTING_STRING_LEN = 500


def _looks_like_jsonrpc_method(s: str) -> bool:
    """True if *s* matches ``namespace/method`` — e.g. ``message/send``.

    The namespace must be an identifier, and the method may contain dots
    (for sub-methods like ``tasks/messageId/cancel``).
    """
    if "/" not in s:
        return False
    ns, _, method = s.partition("/")
    if not ns or not method:
        return False
    if not (ns[0] == "_" or ns[0].isalpha()):
        return False
    if not all(c.isalnum() or c == "_" for c in ns):
        return False
    if not (method[0] == "_" or method[0].isalpha()):
        return False
    if not all(c.isalnum() or c in ("_", ".") for c in method):
        return False
    return True


def _is_protocol_relevant_string(value: str) -> bool:
    """True if *value* looks like a URL, HTTP/filesystem path, or JSON-RPC method."""
    if not value or len(value) > _MAX_INTERESTING_STRING_LEN:
        return False
    s = value.strip()
    if not s:
        return False
    if s.startswith(("http://", "https://", "ws://", "wss://")):
        return True
    if s.startswith("/") and "/" in s[1:]:
        return True
    if _looks_like_jsonrpc_method(s):
        return True
    return False


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
        self._scope_stack: List[Dict[str, Any]] = []

    def _current_function_frame(self) -> Optional[Dict[str, Any]]:
        """Return the innermost ``function`` frame, or ``None`` if a lambda or
        class body lies between us and any enclosing function."""
        for frame in reversed(self._scope_stack):
            kind = frame["kind"]
            if kind == "function":
                return frame
            if kind in ("lambda", "class"):
                return None
        return None

    def _innermost_class_name(self) -> Optional[str]:
        for frame in reversed(self._scope_stack):
            if frame["kind"] == "class":
                return frame["class_name"]
        return None

    def _attribute_call(self, qname: str) -> None:
        """Attribute a call to the innermost function and any enclosing loops.

        Loop frames are transparent for function attribution; lambda and
        class frames block attribution entirely so nested helpers do not
        inflate an outer function's counters.
        """
        loops: List[Dict[str, Any]] = []
        for frame in reversed(self._scope_stack):
            kind = frame["kind"]
            if kind == "loop":
                loops.append(frame)
                continue
            if kind == "function":
                frame["call_count"] += 1
                if qname not in frame["called_qualified_names"]:
                    frame["called_qualified_names"].append(qname)
                for loop in loops:
                    # Only count calls that appear inside the loop body, not
                    # inside its iter (e.g. ``range(10)``) or test expression
                    # (e.g. ``while done():``). Otherwise the ReAct-loop
                    # detector would treat ``range`` as a distinct callee.
                    #
                    # Duplicates are retained so the matcher can see both the
                    # total call count (e.g. two ``llm.invoke`` calls in a
                    # loop body) and the distinct-callee count.
                    if not loop.get("in_body", False):
                        continue
                    loop["body_call_qualified_names"].append(qname)
                return
            return

    def _attribute_loop(self) -> None:
        for frame in reversed(self._scope_stack):
            kind = frame["kind"]
            if kind == "function":
                frame["loop_count"] += 1
                return
            if kind in ("lambda", "class"):
                return

    def _attribute_branch(self) -> None:
        frame = self._current_function_frame()
        if frame is not None:
            frame["branch_count"] += 1

    def _attribute_return(self) -> None:
        frame = self._current_function_frame()
        if frame is not None:
            frame["return_count"] += 1

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

        Checks:
        1. A trailing inline comment on the definition line itself.
        2. Comment lines above the definition, scanning upward past blank lines
           and decorator lines (``@...``) so that annotations above decorators
           are captured.
        """
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

                # Scan upward past blank lines and decorator lines
                idx = start_line_idx - 1
                while 0 <= idx < len(lines):
                    line = lines[idx].strip()
                    if not line:
                        idx -= 1
                        continue
                    annotation = parse_inline_annotation(lines[idx])
                    if annotation:
                        return annotation
                    if line.startswith("@"):
                        idx -= 1
                        continue
                    break
        except (KeyError, IndexError, AttributeError):
            pass

        return None

    def leave_Import(self, original_node: cst.Import) -> None:
        """Capture import statements with line numbers."""
        try:
            line = self.get_metadata(cst.metadata.PositionProvider, original_node).start.line
        except (KeyError, AttributeError):
            line = 0
        for alias in original_node.names:
            if isinstance(alias, cst.ImportAlias):
                if isinstance(alias.name, cst.Name):
                    import_name = alias.name.value
                elif isinstance(alias.name, cst.Attribute):
                    # Dotted imports such as ``import langchain.agents`` come
                    # through as Attribute nodes. Flatten to dotted text so
                    # downstream substring matchers see ``langchain.agents``.
                    import_name = _format_attribute_name(alias.name)
                else:
                    import_name = str(alias.name)
                import_stmt = f"import {import_name}"
                if alias.asname:
                    import_stmt += f" as {alias.asname.name.value}"
                self.result.imports.append((line, import_stmt))

    def leave_ImportFrom(self, original_node: cst.ImportFrom) -> None:
        """Capture from...import statements with line numbers."""
        try:
            line = self.get_metadata(cst.metadata.PositionProvider, original_node).start.line
        except (KeyError, AttributeError):
            line = 0
        if original_node.module:
            module_name = self._extract_raw_code(original_node.module)
            if isinstance(original_node.names, cst.ImportStar):
                import_stmt = f"from {module_name} import *"
                self.result.imports.append((line, import_stmt))
            elif hasattr(original_node.names, '__iter__'):
                imported_items = []
                for alias in original_node.names:
                    if isinstance(alias, cst.ImportAlias):
                        if isinstance(alias.name, cst.Name):
                            item_name = alias.name.value
                        elif isinstance(alias.name, cst.Attribute):
                            item_name = _format_attribute_name(alias.name)
                        else:
                            item_name = str(alias.name)
                        if alias.asname:
                            item_name += f" as {alias.asname.name.value}"
                        imported_items.append(item_name)
                if imported_items:
                    import_stmt = f"from {module_name} import {', '.join(imported_items)}"
                    self.result.imports.append((line, import_stmt))

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
        """Captures standalone calls that are not part of an assignment.

        Attribution (for :class:`MethodBodyShapeObservation` and
        :class:`ControlFlowObservation`) is best-effort and also covers
        bare unresolved names like ``done()`` — the structural matchers
        care about "what is called inside this function body", even when
        the symbol has no binding in the local module.

        ``result.calls`` keeps the stricter behavior of the original
        visitor: only calls whose target resolves to a qualified or
        attribute name are recorded.
        """
        attribution_name = self._get_qualified_name_for_node(original_node.func)
        if not attribution_name and isinstance(original_node.func, cst.Attribute):
            attribution_name = _format_attribute_name(original_node.func)
        if not attribution_name and isinstance(original_node.func, cst.Name):
            attribution_name = original_node.func.value

        if attribution_name:
            self._attribute_call(attribution_name)

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

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Push a class scope frame so nested calls/loops are attributed correctly."""
        class_name = node.name.value
        qname = self._get_qualified_name_for_node(node.name)
        try:
            pos = self.get_metadata(cst.metadata.PositionProvider, node)
            start_line = pos.start.line
            end_line = pos.end.line
        except (KeyError, AttributeError):
            start_line = 0
            end_line = 0
        self._scope_stack.append(
            {
                "kind": "class",
                "class_name": class_name,
                "qname": qname,
                "start_line": start_line,
                "end_line": end_line,
            }
        )

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        """Capture class definitions: base classes and ``# aibom:`` annotations.

        Also emits a :class:`ClassBodyFactsObservation` that carries the
        entire class source so downstream consumers (evidence builder,
        LLM prompt) do not need to re-read the file from disk.
        """
        if self._scope_stack and self._scope_stack[-1].get("kind") == "class":
            self._scope_stack.pop()

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
            pos = self.get_metadata(cst.metadata.PositionProvider, original_node)
            line_number = pos.start.line
            end_line = pos.end.line
        except (KeyError, AttributeError):
            line_number = 0
            end_line = 0

        obs = ClassDefObservation(
            class_name=class_name,
            qualified_name=qualified_name,
            base_classes=base_classes,
            line_number=line_number,
            aibom_annotation=annotation,
        )
        self.result.class_defs.append(obs)

        method_names: List[str] = []
        body = getattr(original_node, "body", None)
        body_stmts = getattr(body, "body", []) or []
        for stmt in body_stmts:
            if isinstance(stmt, cst.FunctionDef):
                method_names.append(stmt.name.value)

        body_source = self._extract_raw_code(original_node)

        class_decorator_names: List[str] = []
        for decorator_node in original_node.decorators:
            decorator_expr: cst.CSTNode = decorator_node.decorator
            decorator_name = self._get_qualified_name_for_node(decorator_expr)
            if not decorator_name and isinstance(decorator_expr, cst.Call):
                decorator_name = self._get_qualified_name_for_node(decorator_expr.func)
            if not decorator_name:
                target_expr = (
                    decorator_expr.func
                    if isinstance(decorator_expr, cst.Call)
                    else decorator_expr
                )
                if isinstance(target_expr, cst.Attribute):
                    decorator_name = _format_attribute_name(target_expr)
                elif isinstance(target_expr, cst.Name):
                    decorator_name = target_expr.value
            if decorator_name:
                class_decorator_names.append(decorator_name)

        self.result.class_bodies.append(
            ClassBodyFactsObservation(
                class_name=class_name,
                qualified_name=qualified_name,
                start_line=line_number,
                end_line=end_line,
                base_classes=list(base_classes),
                method_names=method_names,
                body_source=body_source,
                class_decorators=class_decorator_names,
            )
        )

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Push a function scope frame so we can aggregate structural counts.

        Both ``def`` and ``async def`` are represented by
        :class:`cst.FunctionDef` (async is expressed via the ``asynchronous``
        field), so a single visitor handles both.
        """
        method_name = node.name.value
        qname = self._get_qualified_name_for_node(node.name)
        class_name = self._innermost_class_name()
        try:
            pos = self.get_metadata(cst.metadata.PositionProvider, node)
            start_line = pos.start.line
            end_line = pos.end.line
        except (KeyError, AttributeError):
            start_line = 0
            end_line = 0
        self._scope_stack.append(
            {
                "kind": "function",
                "class_name": class_name,
                "method_name": method_name,
                "qname": qname,
                "start_line": start_line,
                "end_line": end_line,
                "call_count": 0,
                "loop_count": 0,
                "branch_count": 0,
                "return_count": 0,
                "called_qualified_names": [],
            }
        )

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        """Captures functions that have decorators and/or ``# aibom:`` annotations.

        Also pops the scope frame and emits a
        :class:`MethodBodyShapeObservation` summarizing the function body.
        """
        frame: Optional[Dict[str, Any]] = None
        if self._scope_stack and self._scope_stack[-1].get("kind") == "function":
            frame = self._scope_stack.pop()

        if frame is not None:
            body = getattr(original_node, "body", None)
            body_stmts = getattr(body, "body", []) or []
            statement_count = len(body_stmts) if isinstance(body_stmts, (list, tuple)) else 0
            self.result.method_shapes.append(
                MethodBodyShapeObservation(
                    owner_qualified_name=frame["qname"],
                    owner_class_name=frame["class_name"],
                    method_name=frame["method_name"],
                    start_line=frame["start_line"],
                    end_line=frame["end_line"],
                    statement_count=statement_count,
                    call_count=frame["call_count"],
                    loop_count=frame["loop_count"],
                    branch_count=frame["branch_count"],
                    return_count=frame["return_count"],
                    called_qualified_names=list(frame["called_qualified_names"]),
                )
            )

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

    def _push_loop_frame(self, node: cst.CSTNode, loop_kind: str) -> None:
        owner = self._current_function_frame()
        try:
            pos = self.get_metadata(cst.metadata.PositionProvider, node)
            start_line = pos.start.line
            end_line = pos.end.line
        except (KeyError, AttributeError):
            start_line = 0
            end_line = 0
        branch_at_entry = 0
        if owner is not None:
            branch_at_entry = owner.get("branch_count", 0)
        self._scope_stack.append(
            {
                "kind": "loop",
                "loop_kind": loop_kind,
                "start_line": start_line,
                "end_line": end_line,
                "owner": owner,
                "branch_at_entry": branch_at_entry,
                "body_call_qualified_names": [],
                # Flipped on by ``visit_<For|While>_body`` and off again by
                # the matching ``leave`` hook so iter/test expressions don't
                # contribute to the loop's body-call set.
                "in_body": False,
            }
        )

    def _pop_loop_frame(self) -> None:
        """Pop a loop frame and emit an observation only if the loop has an
        owning function.

        Module-level loops (e.g. CLI scripts or top-level ``while True``
        daemons) are intentionally ignored: the matcher consumes
        ``ControlFlowObservation`` to identify ReAct-style orchestration
        loops inside methods, and attaching that signal to a ``None``
        owner would both dilute precision and complicate attribution.
        """
        if not self._scope_stack or self._scope_stack[-1].get("kind") != "loop":
            return
        frame = self._scope_stack.pop()
        owner = frame.get("owner")
        if owner is None:
            return
        has_branch = owner.get("branch_count", 0) > frame.get("branch_at_entry", 0)
        self.result.control_flows.append(
            ControlFlowObservation(
                owner_qualified_name=owner.get("qname"),
                owner_class_name=owner.get("class_name"),
                owner_method_name=owner.get("method_name"),
                loop_kind=frame["loop_kind"],
                start_line=frame["start_line"],
                end_line=frame["end_line"],
                body_call_qualified_names=list(frame["body_call_qualified_names"]),
                has_branch=has_branch,
            )
        )

    def visit_While(self, node: cst.While) -> None:
        self._attribute_loop()
        self._push_loop_frame(node, "while")

    def visit_While_body(self, node: cst.While) -> None:
        self._mark_loop_body(True)

    def leave_While_body(self, original_node: cst.While) -> None:
        self._mark_loop_body(False)

    def leave_While(self, original_node: cst.While) -> None:
        self._pop_loop_frame()

    def visit_For(self, node: cst.For) -> None:
        self._attribute_loop()
        kind = "async for" if getattr(node, "asynchronous", None) is not None else "for"
        self._push_loop_frame(node, kind)

    def visit_For_body(self, node: cst.For) -> None:
        self._mark_loop_body(True)

    def leave_For_body(self, original_node: cst.For) -> None:
        self._mark_loop_body(False)

    def leave_For(self, original_node: cst.For) -> None:
        self._pop_loop_frame()

    def _mark_loop_body(self, in_body: bool) -> None:
        """Toggle the ``in_body`` flag on the innermost active loop frame.

        No-op if no loop frame is on the stack (e.g. if libcst ever delivers
        a body visit without a matching loop frame, which should not happen).
        """
        for frame in reversed(self._scope_stack):
            if frame.get("kind") == "loop":
                frame["in_body"] = in_body
                return

    def visit_Lambda(self, node: cst.Lambda) -> None:
        self._scope_stack.append({"kind": "lambda"})

    def leave_Lambda(self, original_node: cst.Lambda) -> None:
        if self._scope_stack and self._scope_stack[-1].get("kind") == "lambda":
            self._scope_stack.pop()

    def leave_If(self, original_node: cst.If) -> None:
        self._attribute_branch()

    def leave_IfExp(self, original_node: cst.IfExp) -> None:
        self._attribute_branch()

    def leave_Return(self, original_node: cst.Return) -> None:
        self._attribute_return()

    def leave_SimpleString(self, original_node: cst.SimpleString) -> None:
        """Emit a :class:`StringLiteralObservation` for protocol-relevant literals
        that appear inside a class body (including methods).

        A literal is attributed to its innermost enclosing function when one
        exists (so ``owner_method_name`` names the method). If the literal
        lives directly in the class body — e.g. ``card_path = "/.well-known/
        agent.json"`` as a class attribute — it is still captured, with
        ``owner_method_name = None`` and ``owner_class_name`` set to the
        enclosing class. Strings at pure module scope are ignored.
        """
        fn_frame = self._current_function_frame()
        class_name = self._innermost_class_name()
        if fn_frame is None and class_name is None:
            return
        try:
            value = ast.literal_eval(original_node.value)
        except (ValueError, SyntaxError):
            return
        if not isinstance(value, str):
            return
        if not _is_protocol_relevant_string(value):
            return
        try:
            line = self.get_metadata(cst.metadata.PositionProvider, original_node).start.line
        except (KeyError, AttributeError):
            line = 0
        if fn_frame is not None:
            owner_class = fn_frame.get("class_name")
            owner_method = fn_frame.get("method_name")
        else:
            owner_class = class_name
            owner_method = None
        self.result.protocol_strings.append(
            StringLiteralObservation(
                owner_class_name=owner_class,
                owner_method_name=owner_method,
                line_number=line,
                value=value,
            )
        )

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
