# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow and call-graph analysis utilities.

This module performs a lightweight static analysis pass over Python source files
to build a best-effort call graph. The analyzer focuses on identifying the
function ("workflow") that contains a particular line as well as the ancestor
functions that may eventually invoke it. The resulting structure is used to
annotate detected AI assets with the workflows that reach them.
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


_LOGGER = logging.getLogger(__name__)


@dataclass
class FunctionNode:
    """Metadata for a single function or method."""

    function_id: str
    file_path: str
    qualname: str
    start_line: int
    end_line: int
    decorators: List[str] = field(default_factory=list)
    class_name: Optional[str] = None
    parameters: List[str] = field(default_factory=list)

    def contains_line(self, line_number: int) -> bool:
        return self.start_line <= line_number <= self.end_line


class WorkflowIndex:
    """Holds function metadata and call graph relationships."""

    def __init__(self) -> None:
        self.functions: Dict[str, FunctionNode] = {}
        self.functions_by_file: Dict[str, List[FunctionNode]] = {}
        self.call_graph: Dict[str, Set[str]] = {}
        self.reverse_call_graph: Dict[str, Set[str]] = {}
        self.call_metadata: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    def add_function(self, node: FunctionNode) -> None:
        self.functions[node.function_id] = node
        self.functions_by_file.setdefault(node.file_path, []).append(node)

    def add_call(
        self,
        caller_id: str,
        callee_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if caller_id == callee_id:
            return
        if caller_id not in self.functions:
            return
        self.call_graph.setdefault(caller_id, set()).add(callee_id)
        if metadata:
            key = (caller_id, callee_id)
            self.call_metadata.setdefault(key, []).append(metadata)

    def finalize(self) -> None:
        for file_nodes in self.functions_by_file.values():
            file_nodes.sort(key=lambda n: (n.start_line, n.end_line))
        for caller, callees in self.call_graph.items():
            if caller not in self.functions:
                continue
            for callee in callees:
                if callee not in self.functions:
                    continue
                self.reverse_call_graph.setdefault(callee, set()).add(caller)

    # ------------------------------------------------------------------
    # Lookup helpers
    def find_function(self, file_path: str, line_number: int) -> Optional[FunctionNode]:
        nodes = self.functions_by_file.get(file_path, [])
        for node in nodes:
            if node.contains_line(line_number):
                return node
        return None

    def _workflow_entry(
        self,
        node: FunctionNode,
        distance: int,
        child_function_id: Optional[str],
        call_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Optional[str]]:
        workflow_id = workflow_identifier(node.qualname, node.file_path, node.start_line)
        return {
            "function": node.qualname,
            "file_path": node.file_path,
            "line": node.start_line,
            "distance": distance,
            "function_id": node.function_id,
            "child_function_id": child_function_id,
            "workflow_id": workflow_id,
            "parameters": list(node.parameters),
            "callsite_line": call_metadata.get("line") if call_metadata else None,
            "call_arguments": call_metadata.get("arguments") if call_metadata else None,
        }

    def _get_call_metadata(
        self, caller_id: str, callee_id: str
    ) -> Optional[Dict[str, Any]]:
        entries = self.call_metadata.get((caller_id, callee_id))
        if not entries:
            return None
        return entries[0]

    def get_workflow_context(
        self,
        file_path: str,
        line_number: int,
        max_depth: int = 4,
        max_entries: int = 10,
    ) -> List[Dict[str, Optional[str]]]:
        """Return workflow metadata for the code location if known."""

        node = self.find_function(file_path, line_number)
        if not node:
            return []

        workflows: List[Dict[str, Optional[str]]] = [
            self._workflow_entry(
                node,
                distance=0,
                child_function_id=None,
                call_metadata=None,
            )
        ]

        visited: Set[str] = {node.function_id}
        frontier: List[Tuple[str, int]] = [(node.function_id, 0)]

        while frontier and len(workflows) < max_entries:
            current, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for caller in self.reverse_call_graph.get(current, set()):
                if caller in visited:
                    continue
                visited.add(caller)
                caller_node = self.functions.get(caller)
                if caller_node:
                    workflows.append(
                        self._workflow_entry(
                            caller_node,
                            distance=depth + 1,
                            child_function_id=current,
                            call_metadata=self._get_call_metadata(caller, current),
                        )
                    )
                    frontier.append((caller, depth + 1))
                if len(workflows) >= max_entries:
                    break

        return workflows


# ----------------------------------------------------------------------
# Builders

def build_workflow_index(python_files: List[Path]) -> WorkflowIndex:
    """Create a WorkflowIndex for the provided Python files."""

    index = WorkflowIndex()
    for file_path in python_files:
        try:
            source = Path(file_path)
            text = source.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(file_path))
        except Exception as exc:  # pragma: no cover - parsing best-effort
            _LOGGER.debug("Skipping %s during workflow index build: %s", file_path, exc)
            continue

        module_name = _module_name(source)
        collector = _FunctionCollector(module_name, source, index)
        collector.visit(tree)

    index.finalize()
    return index


class _FunctionCollector(ast.NodeVisitor):
    """Collect function metadata and call relationships."""

    def __init__(self, module_name: str, file_path: Path, index: WorkflowIndex) -> None:
        self.module_name = module_name
        self.file_path = str(file_path)
        self.index = index
        self.class_stack: List[str] = []
        self.current_function: Optional[FunctionNode] = None

    # ------------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._register_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._register_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    # ------------------------------------------------------------------
    def _register_function(self, node: ast.AST) -> None:
        qualname = self._build_qualname(node.name)  # type: ignore[attr-defined]
        function_id = f"{self.file_path}::{qualname}"
        decorators = [self._decorator_name(d) for d in getattr(node, "decorator_list", [])]
        func_node = FunctionNode(
            function_id=function_id,
            file_path=self.file_path,
            qualname=f"{self.module_name}.{qualname}" if self.module_name else qualname,
            start_line=getattr(node, "lineno", 0),
            end_line=getattr(node, "end_lineno", getattr(node, "lineno", 0)),
            decorators=[d for d in decorators if d],
            class_name=self.class_stack[-1] if self.class_stack else None,
            parameters=_extract_parameters(getattr(node, "args", None)),
        )

        self.index.add_function(func_node)

        previous_function = self.current_function
        self.current_function = func_node
        self.generic_visit(node)
        self.current_function = previous_function

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if not self.current_function:
            return
        callee_name = _call_name(node)
        callee_id = self._resolve_callee(callee_name)
        if callee_id:
            metadata = {
                "line": getattr(node, "lineno", None),
                "arguments": _call_arguments(node),
            }
            self.index.add_call(self.current_function.function_id, callee_id, metadata=metadata)
        self.generic_visit(node)

    # ------------------------------------------------------------------
    def _build_qualname(self, func_name: str) -> str:
        if self.class_stack:
            return f"{self.class_stack[-1]}.{func_name}"
        return func_name

    def _decorator_name(self, decorator: ast.expr) -> str:
        if isinstance(decorator, ast.Call):
            decorator = decorator.func
        if isinstance(decorator, ast.Attribute):
            parts = []
            current = decorator
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        if isinstance(decorator, ast.Name):
            return decorator.id
        return ""

    def _resolve_callee(self, callee: str) -> Optional[str]:
        if not callee:
            return None

        candidates = []
        if "." not in callee:
            candidates.append(f"{self.file_path}::{callee}")
            if self.class_stack:
                candidates.append(f"{self.file_path}::{self.class_stack[-1]}.{callee}")
        else:
            parts = callee.split(".")
            if parts[0] in {"self", "cls"} and self.class_stack:
                method = parts[-1]
                candidates.append(f"{self.file_path}::{self.class_stack[-1]}.{method}")
            if len(parts) >= 2:
                last_two = ".".join(parts[-2:])
                candidates.append(f"{self.file_path}::{last_two}")
            candidates.append(f"{self.file_path}::{callee}")

        for candidate in candidates:
            if candidate in self.index.functions:
                return candidate
        if candidates:
            return candidates[0]
        return None


# ----------------------------------------------------------------------
# Helper utilities

def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: List[str] = []
        current = func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        return ".".join(parts)
    return ""


def _module_name(path: Path) -> str:
    try:
        relative = path.relative_to(os.getcwd())
    except ValueError:
        relative = path
    parts = list(relative.with_suffix("").parts)
    return ".".join(parts)


def _call_arguments(node: ast.Call) -> Dict[str, Any]:
    positional = [_expr_to_source(arg) for arg in getattr(node, "args", [])]
    keywords = {}
    for keyword in getattr(node, "keywords", []):
        key = keyword.arg or ""
        keywords[key] = _expr_to_source(keyword.value)
    return {"args": positional, "keywords": keywords}


def _expr_to_source(expr: ast.AST) -> str:
    try:
        return ast.unparse(expr)
    except Exception:  # pragma: no cover - best effort serialization
        return ""


def _extract_parameters(args: Optional[ast.arguments]) -> List[str]:
    if not args:
        return []
    parameters: List[str] = []
    for param in getattr(args, "posonlyargs", []):
        parameters.append(param.arg)
    for param in getattr(args, "args", []):
        parameters.append(param.arg)
    vararg = getattr(args, "vararg", None)
    if vararg:
        parameters.append(f"*{vararg.arg}")
    for param in getattr(args, "kwonlyargs", []):
        parameters.append(param.arg)
    kwarg = getattr(args, "kwarg", None)
    if kwarg:
        parameters.append(f"**{kwarg.arg}")
    return parameters


def workflow_identifier(
    function_name: Optional[str],
    file_path: Optional[str],
    line: Optional[int],
) -> str:
    """Create a stable identifier for a workflow node."""

    normalized_name = (function_name or "workflow").replace(" ", "_")
    normalized_file = (file_path or "").replace(" ", "_")
    line_value: int
    if isinstance(line, int):
        line_value = line
    else:
        try:
            line_value = int(line) if line is not None else 0
        except (TypeError, ValueError):
            line_value = 0
    return f"{normalized_name}|{normalized_file}|{line_value}"
