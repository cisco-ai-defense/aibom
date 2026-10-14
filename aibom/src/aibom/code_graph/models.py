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

"""Data model for the deterministic code graph.

The graph answers three questions that per-file observations cannot:

* which function calls which, across file boundaries;
* which functions can reach a given line (used to widen LLM evidence
  beyond the candidate's own 31-line window);
* what concrete value a name holds, following assignment chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

MAX_CALLER_DEPTH = 4


@dataclass(frozen=True)
class FunctionNode:
    """One function or method, addressable across the whole scan."""

    node_id: str
    file_path: str
    qualified_name: str
    method_name: str
    class_name: str | None = None
    start_line: int = 0
    end_line: int = 0

    def contains_line(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line


@dataclass(frozen=True)
class ValueBinding:
    """A name bound to a value inside one scope.

    ``owner`` is the enclosing function's qualified name, or ``None`` for a
    module-level binding. Scope is kept because two functions in one file
    routinely bind the same name to different values, and matching on bare
    name across a file would silently pick whichever came last.
    """

    name: str
    value: Any
    kind: str
    owner: str | None
    line_number: int


@dataclass(frozen=True)
class MethodCall:
    """A ``receiver.method(...)`` call site with its arguments.

    Graph-building frameworks wire components through method calls rather
    than constructor keywords: LangGraph's ``builder.add_node(...)`` is the
    only place a node is attached to its graph. Those call sites are not
    assignments, so nothing else in the parse result retains them.
    """

    file_path: str
    receiver: str
    method: str
    arguments: dict[str, Any]
    line_number: int


@dataclass
class CodeGraph:
    """Call edges plus value bindings for a set of analyzed files."""

    functions: dict[str, FunctionNode] = field(default_factory=dict)
    call_edges: dict[str, set[str]] = field(default_factory=dict)
    reverse_call_edges: dict[str, set[str]] = field(default_factory=dict)
    unresolved_calls: dict[str, set[str]] = field(default_factory=dict)
    bindings: dict[str, list[ValueBinding]] = field(default_factory=dict)
    method_calls: dict[str, list[MethodCall]] = field(default_factory=dict)

    def add_function(self, node: FunctionNode) -> None:
        self.functions[node.node_id] = node

    def add_method_call(self, call: MethodCall) -> None:
        self.method_calls.setdefault(call.file_path, []).append(call)

    def functions_in_file(self, file_path: str) -> list[FunctionNode]:
        return [n for n in self.functions.values() if n.file_path == file_path]

    def add_call(self, caller_id: str, callee_id: str) -> None:
        self.call_edges.setdefault(caller_id, set()).add(callee_id)
        self.reverse_call_edges.setdefault(callee_id, set()).add(caller_id)

    def add_unresolved(self, caller_id: str, callee_name: str) -> None:
        self.unresolved_calls.setdefault(caller_id, set()).add(callee_name)

    def enclosing_function(self, file_path: str, line: int) -> FunctionNode | None:
        """Return the tightest function containing *line* in *file_path*.

        Tightest rather than first, so a nested helper wins over the outer
        function that lexically contains it.
        """
        best: FunctionNode | None = None
        for node in self.functions.values():
            if node.file_path != file_path or not node.contains_line(line):
                continue
            if best is None or node.start_line > best.start_line:
                best = node
        return best

    def callees_of(self, node_id: str) -> set[str]:
        return set(self.call_edges.get(node_id, ()))

    def callers_of(
        self, node_id: str, max_depth: int = MAX_CALLER_DEPTH
    ) -> list[FunctionNode]:
        """Breadth-first walk up the reverse edges, nearest callers first.

        Bounded because a utility called from everywhere would otherwise
        pull the entire repository into an LLM prompt.
        """
        seen = {node_id}
        frontier = [node_id]
        found: list[FunctionNode] = []
        for _ in range(max_depth):
            if not frontier:
                break
            nxt: list[str] = []
            for current in frontier:
                for caller in sorted(self.reverse_call_edges.get(current, ())):
                    if caller in seen:
                        continue
                    seen.add(caller)
                    nxt.append(caller)
                    node = self.functions.get(caller)
                    if node is not None:
                        found.append(node)
            frontier = nxt
        return found

    def transitive_callees(
        self, node_id: str, max_depth: int = MAX_CALLER_DEPTH
    ) -> list[FunctionNode]:
        """Bounded walk down the call edges.

        This is what defeats the one-level-of-indirection problem: a loop
        calling ``self._step()`` looks empty until you can see that
        ``_step`` is what actually calls the model.
        """
        seen = {node_id}
        frontier = [node_id]
        found: list[FunctionNode] = []
        for _ in range(max_depth):
            if not frontier:
                break
            nxt: list[str] = []
            for current in frontier:
                for callee in sorted(self.call_edges.get(current, ())):
                    if callee in seen:
                        continue
                    seen.add(callee)
                    nxt.append(callee)
                    node = self.functions.get(callee)
                    if node is not None:
                        found.append(node)
            frontier = nxt
        return found

    def resolve_value(
        self,
        file_path: str,
        name: str,
        owner: str | None = None,
        before_line: int | None = None,
    ) -> ValueBinding | None:
        """Follow ``name`` through its assignment chain to a literal.

        Function scope is searched before module scope. A ``VARIABLE:``
        binding is followed; anything else terminates the walk and is
        returned as-is, so the caller can still see that the name resolved
        to (say) a subscript it cannot evaluate.

        ``before_line`` is the line doing the reading. Without it a name
        rebound later in the file resolves to its final value for every
        use, which reads ``expt_llm = "gpt-4o"`` ... ``expt_llm =
        "claude-3-opus"`` as though the first model were the second.
        """
        chain_guard: set[str] = set()
        current = name
        current_owner = owner
        cutoff = before_line
        last: ValueBinding | None = None

        while current and current not in chain_guard:
            chain_guard.add(current)
            binding = self._lookup_binding(file_path, current, current_owner, cutoff)
            if binding is None:
                return last
            last = binding
            if binding.kind != "variable":
                return binding
            current = str(binding.value).split("VARIABLE:", 1)[-1]
            current_owner = binding.owner
            # The next link was written before this one, never after it.
            cutoff = binding.line_number
        return last

    def _lookup_binding(
        self,
        file_path: str,
        name: str,
        owner: str | None,
        before_line: int | None = None,
    ) -> ValueBinding | None:
        candidates = self.bindings.get(file_path, ())

        def latest(pool: list[ValueBinding]) -> ValueBinding | None:
            if before_line is not None:
                pool = [b for b in pool if b.line_number <= before_line]
            if not pool:
                return None
            return max(pool, key=lambda b: b.line_number)

        scoped = latest([b for b in candidates if b.name == name and b.owner == owner])
        if scoped is not None:
            return scoped
        return latest([b for b in candidates if b.name == name and b.owner is None])

    def stats(self) -> dict[str, int]:
        return {
            "functions": len(self.functions),
            "call_edges": sum(len(v) for v in self.call_edges.values()),
            "unresolved_calls": sum(len(v) for v in self.unresolved_calls.values()),
            "files_with_bindings": len(self.bindings),
            "bindings": sum(len(v) for v in self.bindings.values()),
            "method_calls": sum(len(v) for v in self.method_calls.values()),
        }


def make_node_id(file_path: str, qualified_name: str) -> str:
    return f"{file_path}::{qualified_name}"


def iter_nodes(graph: CodeGraph) -> Iterable[FunctionNode]:
    return graph.functions.values()
