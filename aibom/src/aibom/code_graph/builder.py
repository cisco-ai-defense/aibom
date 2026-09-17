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

"""Build a :class:`CodeGraph` from per-file parser observations.

The parser already records, for each function, the qualified names it
calls. Those are strings, not links: ``self._step`` stays the literal text
``self._step`` and nothing maps it to the method of that name. This module
does the resolution, which is what turns a pile of name lists into a graph.
"""

from __future__ import annotations

import logging
import posixpath

from ..structures import CodeAnalysisResult
from .models import CodeGraph, FunctionNode, MethodCall, ValueBinding, make_node_id

_LOGGER = logging.getLogger(__name__)

_LOCALS_MARKER = ".<locals>."
_SELF_PREFIXES = ("self.", "cls.")


def _strip_locals(name: str) -> str:
    """Reduce ``Agent.run.<locals>.self._step`` to ``self._step``.

    The parser qualifies names relative to the enclosing scope, so a call
    to ``self._step`` inside a method arrives wrapped in the caller's own
    path. Only the part after the last ``<locals>`` is the callee.
    """
    if _LOCALS_MARKER in name:
        return name.rsplit(_LOCALS_MARKER, 1)[1]
    return name


def _module_keys(file_path: str) -> list[str]:
    """Dotted module names a file could be imported as, longest first."""
    normalized = file_path.replace("\\", "/")
    if normalized.endswith(".py"):
        normalized = normalized[: -len(".py")]
    parts = [p for p in normalized.split("/") if p and p != "."]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return []
    return [".".join(parts[i:]) for i in range(len(parts))]


class _ModuleIndex:
    """Maps dotted module names to the files that define them.

    A short key such as ``tools`` is frequently ambiguous in a monorepo, so
    ambiguous keys are recorded and then only used when the importing file
    sits in the same directory as exactly one candidate.
    """

    def __init__(self, file_paths: list[str]) -> None:
        self._by_key: dict[str, set[str]] = {}
        for path in file_paths:
            for key in _module_keys(path):
                self._by_key.setdefault(key, set()).add(path)

    def resolve(self, module: str, importing_file: str) -> str | None:
        candidates = self._by_key.get(module)
        if not candidates:
            return None
        if len(candidates) == 1:
            return next(iter(candidates))
        same_dir = posixpath.dirname(importing_file.replace("\\", "/"))
        local = [
            c for c in candidates if posixpath.dirname(c.replace("\\", "/")) == same_dir
        ]
        if len(local) == 1:
            return local[0]
        return None


class _SymbolIndex:
    """Function lookup by file, by qualified name, and by bare name."""

    def __init__(self) -> None:
        self.by_node_id: dict[str, FunctionNode] = {}
        self.by_file_qname: dict[tuple[str, str], FunctionNode] = {}
        self.by_file_method: dict[tuple[str, str], list[FunctionNode]] = {}
        self.class_files: dict[str, set[str]] = {}

    def add(self, node: FunctionNode) -> None:
        self.by_node_id[node.node_id] = node
        self.by_file_qname[(node.file_path, node.qualified_name)] = node
        self.by_file_method.setdefault((node.file_path, node.method_name), []).append(
            node
        )
        if node.class_name:
            bare = node.class_name.split(".")[-1]
            self.class_files.setdefault(bare, set()).add(node.file_path)

    def in_file(self, file_path: str, qualified_name: str) -> FunctionNode | None:
        return self.by_file_qname.get((file_path, qualified_name))

    def method_in_file(self, file_path: str, method: str) -> FunctionNode | None:
        matches = self.by_file_method.get((file_path, method), [])
        if len(matches) == 1:
            return matches[0]
        return None

    def method_on_class(self, class_name: str, method: str) -> FunctionNode | None:
        """Look up ``Class.method`` when exactly one file defines the class.

        Two files defining a class of the same name make the lookup a coin
        flip, so it declines instead.
        """
        files = self.class_files.get(class_name, set())
        if len(files) != 1:
            return None
        return self.in_file(next(iter(files)), f"{class_name}.{method}")


class _ReceiverTypes:
    """Per-file table of ``variable -> class it was instantiated from``.

    This is what makes ``eng.start()`` resolvable without guessing: the
    receiver's type is read from ``eng = Engine()`` rather than inferred
    from the method name. A variable rebound to two different classes in
    one file is dropped, because then the call site genuinely could go to
    either.
    """

    def __init__(self, results: list[CodeAnalysisResult]) -> None:
        self._by_file: dict[str, dict[str, str]] = {}
        for result in results:
            seen: dict[str, set[str]] = {}
            for obs in result.assignments:
                variable = _strip_locals(obs.target_qualified_name).split(".")[-1]
                class_name = _strip_locals(obs.call.qualified_name).split(".")[-1]
                if variable and class_name:
                    seen.setdefault(variable, set()).add(class_name)
            table = {v: next(iter(c)) for v, c in seen.items() if len(c) == 1}
            if table:
                self._by_file[result.file_path] = table

    def type_of(self, file_path: str, variable: str) -> str | None:
        return self._by_file.get(file_path, {}).get(variable)


def _collect_functions(
    results: list[CodeAnalysisResult], index: _SymbolIndex, graph: CodeGraph
) -> None:
    for result in results:
        for shape in result.method_shapes:
            qname = shape.owner_qualified_name or shape.method_name
            if not qname:
                continue
            node = FunctionNode(
                node_id=make_node_id(result.file_path, qname),
                file_path=result.file_path,
                qualified_name=qname,
                method_name=shape.method_name,
                class_name=shape.owner_class_name,
                start_line=shape.start_line,
                end_line=shape.end_line,
            )
            graph.add_function(node)
            index.add(node)


def _collect_bindings(results: list[CodeAnalysisResult], graph: CodeGraph) -> None:
    for result in results:
        bindings: list[ValueBinding] = []
        for obs in result.value_assignments:
            # Targets arrive scope-qualified (``build.<locals>.model``); the
            # resolver looks names up as written in source.
            bare = _strip_locals(obs.target_qualified_name).split(".")[-1]
            bindings.append(
                ValueBinding(
                    name=bare,
                    value=obs.value,
                    kind=obs.value_kind,
                    owner=obs.owner_qualified_name,
                    line_number=obs.line_number,
                )
            )
        if bindings:
            graph.bindings[result.file_path] = bindings


def _collect_method_calls(results: list[CodeAnalysisResult], graph: CodeGraph) -> None:
    """Record ``receiver.method(...)`` sites that carry arguments.

    Bare function calls are already covered by the call edges; only the
    dotted form can express "attach this component to that one", which is
    how graph-building frameworks wire things together.
    """
    for result in results:
        for obs in result.calls:
            if not obs.arguments:
                continue
            receiver, _, method = _strip_locals(obs.qualified_name).rpartition(".")
            if not receiver or not method:
                continue
            graph.add_method_call(
                MethodCall(
                    file_path=result.file_path,
                    receiver=receiver.split(".")[-1],
                    method=method,
                    arguments=obs.arguments,
                    line_number=obs.line_number,
                )
            )


def _resolve_callee(
    raw_name: str,
    caller: FunctionNode,
    index: _SymbolIndex,
    modules: _ModuleIndex,
    receivers: _ReceiverTypes,
) -> str | None:
    """Resolve one callee name to a node id, or ``None`` if external."""
    name = _strip_locals(raw_name)
    if not name:
        return None

    for prefix in _SELF_PREFIXES:
        if name.startswith(prefix) and caller.class_name:
            method = name[len(prefix) :]
            target = index.in_file(
                caller.file_path, f"{caller.class_name}.{method}"
            ) or index.method_in_file(caller.file_path, method)
            return target.node_id if target else None

    same_file = index.in_file(caller.file_path, name)
    if same_file is not None:
        return same_file.node_id

    if "." not in name:
        local = index.method_in_file(caller.file_path, name)
        return local.node_id if local else None

    # Dotted: try every module/symbol split, longest module first, so
    # ``pkg.tools.search`` prefers module ``pkg.tools`` over module ``pkg``.
    stripped = name.lstrip(".")
    parts = stripped.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:cut])
        symbol = ".".join(parts[cut:])
        target_file = modules.resolve(module, caller.file_path)
        if target_file is None:
            continue
        target = index.in_file(target_file, symbol) or index.method_in_file(
            target_file, symbol.split(".")[-1]
        )
        if target is not None:
            return target.node_id

    # ``obj.method()`` resolves through the receiver's recorded type, never
    # through the method name alone. Matching ``a.b.method()`` to a same-file
    # function called ``method`` was wrong essentially every time it fired on
    # real repositories: ``meeting_flow.kickoff()`` calls the flow object's
    # method, not the module-level ``kickoff()`` next to it. An unresolved
    # external call is recorded as such, which is accurate; a wrong edge
    # propagates into every relationship derived by walking reachability.
    if len(parts) >= 2:
        class_name = receivers.type_of(caller.file_path, parts[-2])
        if class_name:
            target = index.method_on_class(class_name, parts[-1])
            if target is not None:
                return target.node_id
    return None


def build_code_graph(results: list[CodeAnalysisResult]) -> CodeGraph:
    """Build a call graph and value-binding table from parsed files."""
    graph = CodeGraph()
    if not results:
        return graph

    index = _SymbolIndex()
    _collect_functions(results, index, graph)
    _collect_bindings(results, graph)
    _collect_method_calls(results, graph)
    modules = _ModuleIndex([r.file_path for r in results])
    receivers = _ReceiverTypes(results)

    for result in results:
        for shape in result.method_shapes:
            qname = shape.owner_qualified_name or shape.method_name
            caller = index.in_file(result.file_path, qname)
            if caller is None:
                continue
            for raw_name in shape.called_qualified_names:
                callee_id = _resolve_callee(raw_name, caller, index, modules, receivers)
                if callee_id is None:
                    graph.add_unresolved(caller.node_id, _strip_locals(raw_name))
                elif callee_id != caller.node_id:
                    graph.add_call(caller.node_id, callee_id)

    _LOGGER.debug("Code graph built: %s", graph.stats())
    return graph
