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

"""Resolve tools registered in a loop to the functions they register.

MCP servers often register their tools in a loop over an inventory declared
elsewhere::

    TOOL_SET = ToolSet(read_only=(search, lookup), write=(update,))
    ...
    for tool in spec.tools.tools_for(read_only_mode=ro):
        mcp.add_tool(FunctionTool.from_function(tool))

The scanner sees one registration call whose argument is the loop variable.
This module follows the loop's iterable backwards through the repository -
assignments, ``append``/``extend`` calls, function parameters (via their call
sites), return values, constructor arguments, attribute access and imports -
until it reaches literal collections of plain function references. The walk is
an over-approximation bounded by depth and by an evaluation budget; anything it
cannot follow contributes nothing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .utils.path_filter import should_skip_dir

_MAX_DEPTH = 24
_MAX_EVALUATIONS = 20_000
_MAX_FILES = 3_000
_TEST_DIRS = frozenset({"tests", "test", "testing", "__tests__"})
_PASSTHROUGH_BUILTINS = frozenset(
    {"list", "tuple", "set", "frozenset", "sorted", "reversed", "iter"}
)
_ACCUMULATORS = frozenset({"append", "extend", "add", "update", "insert"})
_FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPE_NODES = _FUNC_NODES + (ast.ClassDef, ast.Lambda)


@dataclass(frozen=True)
class FuncRef:
    """A module-level function definition: the asset a registration refers to."""

    path: Path
    name: str
    line: int


@dataclass(frozen=True)
class _Ctx:
    path: Path
    scope: ast.AST  # ast.Module or a function definition


def _is_test_path(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    name = parts[-1] if parts else ""
    return (
        bool({p.lower() for p in parts[:-1]} & _TEST_DIRS)
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _own_nodes(scope: ast.AST):
    """Nodes of *scope*'s own body, not descending into nested scopes."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPE_NODES):
            stack.extend(ast.iter_child_nodes(node))


def _names_in_target(target: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


class _RepoIndex:
    """Parsed source files of one repository, with lazy lookups."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: list[Path] = []
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root).parts
            if any(should_skip_dir(p) for p in rel[:-1]):
                continue
            if _is_test_path(path, root):
                continue
            self.files.append(path)
            if len(self.files) >= _MAX_FILES:
                break
        self._trees: dict[Path, ast.Module | None] = {}
        self._parents: dict[Path, dict[ast.AST, ast.AST]] = {}
        self._calls: dict[str, list[tuple[Path, ast.Call]]] | None = None

    def tree(self, path: Path) -> ast.Module | None:
        if path not in self._trees:
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
                self._trees[path] = ast.parse(source)
            except (OSError, SyntaxError, ValueError):
                self._trees[path] = None
        return self._trees[path]

    def parents(self, path: Path) -> dict[ast.AST, ast.AST]:
        if path not in self._parents:
            tree = self.tree(path)
            parents: dict[ast.AST, ast.AST] = {}
            if tree is not None:
                for node in ast.walk(tree):
                    for child in ast.iter_child_nodes(node):
                        parents[child] = node
            self._parents[path] = parents
        return self._parents[path]

    def enclosing_scope(self, path: Path, node: ast.AST) -> ast.AST:
        parents = self.parents(path)
        cur = parents.get(node)
        while cur is not None and not isinstance(cur, _FUNC_NODES):
            cur = parents.get(cur)
        return cur if cur is not None else self.tree(path)

    def calls_named(self, name: str) -> list[tuple[Path, ast.Call]]:
        if self._calls is None:
            self._calls = {}
            for path in self.files:
                tree = self.tree(path)
                if tree is None:
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    callee = (
                        func.id
                        if isinstance(func, ast.Name)
                        else func.attr if isinstance(func, ast.Attribute) else None
                    )
                    if callee:
                        self._calls.setdefault(callee, []).append((path, node))
        return self._calls.get(name, [])

    def module_file(self, importer: Path, module: str, level: int) -> Path | None:
        parts = module.split(".") if module else []
        if level:
            base = importer.parent
            for _ in range(level - 1):
                base = base.parent
            bases = [base]
        else:
            bases = [p for p in importer.parents if self._within_root(p)]
        for base in bases:
            target = base.joinpath(*parts)
            candidates = [target / "__init__.py"]
            if parts:
                candidates.insert(0, target.with_suffix(".py"))
            for cand in candidates:
                if cand.is_file() and self._within_root(cand):
                    return cand
        return None

    def _within_root(self, path: Path) -> bool:
        try:
            path.relative_to(self.root)
        except ValueError:
            return False
        return True


class _Resolver:
    def __init__(self, index: _RepoIndex) -> None:
        self.index = index
        self.budget = _MAX_EVALUATIONS
        self._active: set[tuple[str, int, int]] = set()

    # -- bindings -----------------------------------------------------------
    def _bindings(self, name: str, ctx: _Ctx, depth: int) -> list[tuple[str, object]]:
        """How *name* gets its value in *ctx*: (kind, payload) pairs."""
        out: list[tuple[str, object]] = []
        scope = ctx.scope
        for node in _own_nodes(scope):
            if isinstance(node, ast.Assign):
                if any(name in _names_in_target(t) for t in node.targets):
                    out.append(("expr", node.value))
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                if name in _names_in_target(node.target) and node.value is not None:
                    out.append(("expr", node.value))
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                if name in _names_in_target(node.target):
                    out.append(("expr", node.iter))
            elif isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in _ACCUMULATORS
                    and isinstance(func.value, ast.Name)
                    and func.value.id == name
                ):
                    out.extend(("expr", a) for a in node.args)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) == name:
                        out.append(("import", (node, alias.name)))
            elif isinstance(node, _FUNC_NODES) and node.name == name:
                out.append(("def", node))
            elif isinstance(node, ast.ClassDef) and node.name == name:
                out.append(("class", node))
        if isinstance(scope, _FUNC_NODES):
            params = scope.args
            positional = [*params.posonlyargs, *params.args]
            names = [a.arg for a in positional]
            kwonly = [a.arg for a in params.kwonlyargs]
            if name in names or name in kwonly:
                out.append(("param", (names, name)))
            if not out:
                module = self.index.tree(ctx.path)
                if module is not None:
                    out = self._bindings(name, _Ctx(ctx.path, module), depth)
        return out

    def _import_target(
        self, ctx: _Ctx, node: ast.ImportFrom, name: str
    ) -> tuple[_Ctx, str] | None:
        target = self.index.module_file(ctx.path, node.module or "", node.level)
        if target is None:
            return None
        tree = self.index.tree(target)
        if tree is None:
            return None
        return _Ctx(target, tree), name

    def _param_args(
        self, ctx: _Ctx, names: list[str], name: str
    ) -> list[tuple[ast.AST, _Ctx]]:
        """Argument expressions passed for parameter *name* at every call site."""
        func = ctx.scope
        assert isinstance(func, _FUNC_NODES)  # noqa: S101
        is_method = isinstance(
            self.index.parents(ctx.path).get(func), ast.ClassDef
        ) and names[:1] in (["self"], ["cls"])
        index = names.index(name) if name in names else None
        if index is not None and is_method:
            index -= 1
        out: list[tuple[ast.AST, _Ctx]] = []
        for path, call in self.index.calls_named(func.name):
            caller = _Ctx(path, self.index.enclosing_scope(path, call))
            matched = False
            for kw in call.keywords:
                if kw.arg == name:
                    out.append((kw.value, caller))
                    matched = True
            if matched or index is None or index < 0:
                continue
            for pos, arg in enumerate(call.args):
                if isinstance(arg, ast.Starred):
                    out.append((arg.value, caller))
                    break
                if pos == index:
                    out.append((arg, caller))
                    break
        return out

    def _lookup(self, func: ast.AST, ctx: _Ctx, depth: int):
        """Repo definition a callee expression names: ('def'|'class', node, ctx)."""
        if not isinstance(func, ast.Name):
            return None
        for kind, payload in self._bindings(func.id, ctx, depth):
            if kind in ("def", "class"):
                return kind, payload, _Ctx(ctx.path, self.index.tree(ctx.path))
            if kind == "import":
                node, orig = payload
                hit = self._import_target(ctx, node, orig)
                if hit is not None and depth < _MAX_DEPTH:
                    return self._lookup(ast.Name(id=hit[1]), hit[0], depth + 1)
        return None

    def _returns(self, func: ast.AST) -> list[ast.AST]:
        return [
            n.value
            for n in _own_nodes(func)
            if isinstance(n, ast.Return) and n.value is not None
        ]

    def _returns_closure(self, func: ast.AST) -> bool:
        nested = {n.name for n in _own_nodes(func) if isinstance(n, _FUNC_NODES)}
        return any(
            isinstance(r, ast.Name) and r.id in nested for r in self._returns(func)
        )

    def _guard(self, expr: ast.AST, ctx: _Ctx, tag: int):
        key = (str(ctx.path), id(expr), tag)
        if key in self._active or self.budget <= 0:
            return None
        self.budget -= 1
        self._active.add(key)
        return key

    # -- function references -------------------------------------------------
    def funcs(self, expr: ast.AST, ctx: _Ctx, depth: int = 0) -> set[FuncRef]:
        if depth > _MAX_DEPTH:
            return set()
        key = self._guard(expr, ctx, 0)
        if key is None:
            return set()
        try:
            return self._funcs(expr, ctx, depth + 1)
        finally:
            self._active.discard(key)

    def _funcs(self, expr: ast.AST, ctx: _Ctx, depth: int) -> set[FuncRef]:
        out: set[FuncRef] = set()
        if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
            for elt in expr.elts:
                out |= self.funcs(elt, ctx, depth)
        elif isinstance(expr, ast.Starred):
            out |= self.funcs(expr.value, ctx, depth)
        elif isinstance(expr, ast.Name):
            out |= self._funcs_name(expr.id, ctx, depth)
        elif isinstance(expr, ast.Attribute):
            for call, cctx in self.objs(expr.value, ctx, depth):
                out |= self._constructor_field(call, expr.attr, cctx, depth)
        elif isinstance(expr, ast.Call):
            out |= self._funcs_call(expr, ctx, depth)
        elif isinstance(expr, ast.Subscript):
            out |= self.funcs(expr.value, ctx, depth)
        elif isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            for gen in expr.generators:
                out |= self.funcs(gen.iter, ctx, depth)
        elif isinstance(expr, ast.IfExp):
            out |= self.funcs(expr.body, ctx, depth)
            out |= self.funcs(expr.orelse, ctx, depth)
        elif isinstance(expr, ast.BoolOp):
            for value in expr.values:
                out |= self.funcs(value, ctx, depth)
        elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            out |= self.funcs(expr.left, ctx, depth)
            out |= self.funcs(expr.right, ctx, depth)
        return out

    def _funcs_name(self, name: str, ctx: _Ctx, depth: int) -> set[FuncRef]:
        out: set[FuncRef] = set()
        for kind, payload in self._bindings(name, ctx, depth):
            if kind == "expr":
                out |= self.funcs(payload, ctx, depth)  # type: ignore[arg-type]
            elif kind == "def":
                node = payload
                if isinstance(ctx.scope, ast.Module) or self._is_module_level(
                    ctx, node
                ):
                    out.add(FuncRef(ctx.path, node.name, node.lineno))
            elif kind == "import":
                node, orig = payload  # type: ignore[misc]
                hit = self._import_target(ctx, node, orig)
                if hit is not None:
                    out |= self._funcs_name(hit[1], hit[0], depth + 1)
            elif kind == "param" and isinstance(ctx.scope, _FUNC_NODES):
                names, pname = payload  # type: ignore[misc]
                for arg, cctx in self._param_args(ctx, names, pname):
                    out |= self.funcs(arg, cctx, depth)
        return out

    def _is_module_level(self, ctx: _Ctx, node: ast.AST) -> bool:
        return isinstance(self.index.parents(ctx.path).get(node), ast.Module)

    def _funcs_call(self, call: ast.Call, ctx: _Ctx, depth: int) -> set[FuncRef]:
        func = call.func
        out: set[FuncRef] = set()
        if isinstance(func, ast.Name) and func.id in _PASSTHROUGH_BUILTINS:
            if call.args:
                out |= self.funcs(call.args[0], ctx, depth)
            return out
        if isinstance(func, ast.Attribute):
            objs = self.objs(func.value, ctx, depth)
            if objs:
                # a method of a repo-defined container returns its members
                for ocall, octx in objs:
                    out |= self._constructor_args(ocall, octx, depth)
                return out
            if func.attr in ("copy", "values"):
                return self.funcs(func.value, ctx, depth)
        found = self._lookup(func, ctx, depth)
        if found is None:
            # external wrapper (functools.partial, FunctionTool.from_function, ...)
            if call.args:
                out |= self.funcs(call.args[0], ctx, depth)
            return out
        kind, node, dctx = found
        if kind == "class":
            return self._constructor_args(call, ctx, depth)
        if self._returns_closure(node):
            if call.args:
                out |= self.funcs(call.args[0], ctx, depth)
            return out
        callee = _Ctx(dctx.path, node)
        for value in self._returns(node):
            out |= self.funcs(value, callee, depth)
        return out

    def _constructor_args(self, call: ast.Call, ctx: _Ctx, depth: int) -> set[FuncRef]:
        out: set[FuncRef] = set()
        for arg in call.args:
            out |= self.funcs(arg, ctx, depth)
        for kw in call.keywords:
            out |= self.funcs(kw.value, ctx, depth)
        return out

    def _constructor_field(
        self, call: ast.Call, field: str, ctx: _Ctx, depth: int
    ) -> set[FuncRef]:
        for kw in call.keywords:
            if kw.arg == field:
                return self.funcs(kw.value, ctx, depth)
        if call.args:
            return self._constructor_args(call, ctx, depth)
        return set()

    # -- constructed objects -------------------------------------------------
    def objs(self, expr: ast.AST, ctx: _Ctx, depth: int) -> list[tuple[ast.Call, _Ctx]]:
        """Constructor calls of repo classes that may produce *expr*'s value."""
        if depth > _MAX_DEPTH:
            return []
        key = self._guard(expr, ctx, 1)
        if key is None:
            return []
        try:
            return self._objs(expr, ctx, depth + 1)
        finally:
            self._active.discard(key)

    def _objs(
        self, expr: ast.AST, ctx: _Ctx, depth: int
    ) -> list[tuple[ast.Call, _Ctx]]:
        out: list[tuple[ast.Call, _Ctx]] = []
        if isinstance(expr, ast.Call):
            func = expr.func
            if (isinstance(func, ast.Name) and func.id == "replace") or (
                isinstance(func, ast.Attribute) and func.attr == "replace"
            ):
                if expr.args:
                    out.extend(self.objs(expr.args[0], ctx, depth))
                return out
            found = self._lookup(func, ctx, depth)
            if found is None:
                return out
            kind, node, dctx = found
            if kind == "class":
                out.append((expr, ctx))
            else:
                callee = _Ctx(dctx.path, node)
                for value in self._returns(node):
                    out.extend(self.objs(value, callee, depth))
        elif isinstance(expr, ast.Name):
            for kind, payload in self._bindings(expr.id, ctx, depth):
                if kind == "expr":
                    out.extend(self.objs(payload, ctx, depth))  # type: ignore[arg-type]
                elif kind == "import":
                    node, orig = payload  # type: ignore[misc]
                    hit = self._import_target(ctx, node, orig)
                    if hit is not None:
                        out.extend(self.objs(ast.Name(id=hit[1]), hit[0], depth))
                elif kind == "param" and isinstance(ctx.scope, _FUNC_NODES):
                    names, pname = payload  # type: ignore[misc]
                    for arg, cctx in self._param_args(ctx, names, pname):
                        out.extend(self.objs(arg, cctx, depth))
        elif isinstance(expr, ast.Attribute):
            for call, cctx in self.objs(expr.value, ctx, depth):
                for kw in call.keywords:
                    if kw.arg == expr.attr:
                        out.extend(self.objs(kw.value, cctx, depth))
        return out


def loop_iterable(path: Path, line: int, name: str) -> tuple[ast.AST, ast.AST] | None:
    """(for-node, enclosing scope) when *name* at *line* is a loop variable."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError, ValueError):
        return None
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        end = getattr(node, "end_lineno", None) or node.lineno
        if node.lineno <= line <= end and name in _names_in_target(node.target):
            if best is None or node.lineno > best.lineno:
                best = node
    if best is None:
        return None
    return best, tree


class LoopRegistrationResolver:
    """Expand loop-variable tool registrations into the functions they register."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._index: _RepoIndex | None = None

    def resolve(self, path: Path, line: int, name: str) -> list[FuncRef] | None:
        """Functions registered by the loop at (*path*, *line*), or None.

        Returns None when *name* is not a loop variable there, and an empty list
        when it is one but its iterable could not be followed to functions.
        """
        found = loop_iterable(path, line, name)
        if found is None:
            return None
        if self._index is None:
            self._index = _RepoIndex(self.root)
        index = self._index
        tree = index.tree(path)
        if tree is None:
            return []
        loop = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, (ast.For, ast.AsyncFor))
                and n.lineno == found[0].lineno
            ),
            None,
        )
        if loop is None:
            return []
        ctx = _Ctx(path, index.enclosing_scope(path, loop))
        refs = _Resolver(index).funcs(loop.iter, ctx)
        return sorted(
            (r for r in refs if not _is_test_path(r.path, self.root)),
            key=lambda r: (str(r.path), r.line),
        )
