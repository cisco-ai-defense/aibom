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

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional

from pathspec import PathSpec

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource
from .base import BaseScanner

try:
    from tree_sitter import Language as _TSLanguage
    from tree_sitter import Parser as _TSParser
except ImportError:
    _TSLanguage = None  # type: ignore[misc, assignment]
    _TSParser = None  # type: ignore[misc, assignment]

_EXTENSION_TO_LANG: dict[str, str] = {
    ".js": "jsts",
    ".mjs": "jsts",
    ".ts": "jsts",
    ".tsx": "jsts",
    ".jsx": "jsts",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".cs": "csharp",
}

_ECOSYSTEM_PACKAGES: dict[str, tuple[str, tuple[str, ...]]] = {
    "jsts": (
        "npm",
        (
            "openai",
            "@langchain/core",
            "@langchain/openai",
            "@anthropic-ai/sdk",
            "ai",
            "@ai-sdk/openai",
            "@google/generative-ai",
            "@modelcontextprotocol/sdk",
            "llamaindex",
            "chromadb",
        ),
    ),
    "java": (
        "maven",
        (
            "dev.langchain4j",
            "com.azure.ai.openai",
            "com.google.cloud.aiplatform",
            "io.github.sashirestela.simpleopenai",
        ),
    ),
    "go": (
        "go",
        (
            "github.com/sashabaranov/go-openai",
            "github.com/anthropics/anthropic-sdk-go",
            "github.com/tmc/langchaingo",
        ),
    ),
    "rust": (
        "cargo",
        (
            "async_openai",
            "llm",
            "candle_core",
            "candle_nn",
            "burn",
        ),
    ),
    "ruby": (
        "rubygems",
        (
            "ruby-openai",
            "langchainrb",
        ),
    ),
    "csharp": (
        "nuget",
        (
            "Azure.AI.OpenAI",
            "Microsoft.SemanticKernel",
            "Microsoft.ML",
        ),
    ),
}

_SKIP_MODEL_STRINGS = frozenset(
    {
        "",
        "true",
        "false",
        "null",
        "none",
        "auto",
        "default",
        "undefined",
    }
)


def _build_pathspec(patterns: list[str]) -> Optional[PathSpec]:
    if not patterns:
        return None
    return PathSpec.from_lines("gitwildmatch", patterns)


def _line_at(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _normalize_model_literal(raw: str) -> str:
    s = raw.strip().strip('"').strip("'")
    if s.startswith("${") or "{{" in s:
        return ""
    return s


def _is_plausible_model_id(s: str) -> bool:
    if len(s) < 2 or len(s) > 256:
        return False
    low = s.lower()
    if low in _SKIP_MODEL_STRINGS or low.startswith("${"):
        return False
    if not re.search(r"[a-zA-Z]", s):
        return False
    return True


def _longest_prefix_match(path: str, packages: tuple[str, ...]) -> Optional[str]:
    best: Optional[str] = None
    for p in packages:
        if path == p or path.startswith(p + "/") or path.startswith(p + "."):
            if best is None or len(p) > len(best):
                best = p
    return best


def _match_known_package(
    spec: str, packages: tuple[str, ...]
) -> Optional[str]:
    s = spec.strip().strip('"').strip("'").rstrip(";")
    if not s:
        return None
    for p in sorted(packages, key=len, reverse=True):
        if s == p:
            return p
        if p.startswith("@") or "/" in p or "." in p:
            if s.startswith(p) and (len(s) == len(p) or s[len(p)] in "/."):
                return p
        elif s.startswith(p + ".") or s == p:
            return p
    return _longest_prefix_match(s.replace("\\", "/"), packages)


def _js_ts_alt(packages: tuple[str, ...]) -> str:
    return "|".join(re.escape(p) for p in sorted(packages, key=len, reverse=True))


def _compile_js_ts_patterns(packages: tuple[str, ...]) -> list[re.Pattern[str]]:
    alt = _js_ts_alt(packages)
    return [
        re.compile(rf'require\s*\(\s*["\'](?P<pkg>{alt})["\']', re.MULTILINE),
        re.compile(
            rf"""import\s+(?:[\w{{}},\s*]+\s+from\s+)?["\'](?P<pkg>{alt})["']""",
            re.MULTILINE,
        ),
        re.compile(rf"""from\s+["\'](?P<pkg>{alt})["']""", re.MULTILINE),
        re.compile(r"\bnew\s+OpenAI\s*\(", re.MULTILINE),
        re.compile(r"\bChatOpenAI\s*\(", re.MULTILINE),
        re.compile(
            r"""(?P<key>\bmodel\b)\s*[:=]\s*["'](?P<val>[^"']+)["']""",
            re.MULTILINE,
        ),
    ]


def _compile_java_patterns(packages: tuple[str, ...]) -> list[re.Pattern[str]]:
    pats: list[re.Pattern[str]] = []
    for p in packages:
        pats.append(
            re.compile(
                rf"import\s+(?P<pkg>{re.escape(p)}(?:\.\w+|\.\*)?)\s*;",
                re.MULTILINE,
            )
        )
    pats.extend(
        [
            re.compile(
                r"""(?P<key>\bmodel\b|\bModel\b)\s*[:=]\s*["'](?P<val>[^"']+)["']""",
                re.MULTILINE,
            ),
            re.compile(
                r"\bnew\s+(?:OpenAIClient|ChatCompletionsClient|ChatClient)\s*\(",
                re.MULTILINE,
            ),
        ]
    )
    return pats


def _compile_go_patterns(packages: tuple[str, ...]) -> str:
    return "|".join(re.escape(p) for p in sorted(packages, key=len, reverse=True))


def _compile_rust_patterns(packages: tuple[str, ...]) -> list[re.Pattern[str]]:
    alt = "|".join(re.escape(p) for p in sorted(packages, key=len, reverse=True))
    return [
        re.compile(rf"\buse\s+(?P<pkg>{alt})::", re.MULTILINE),
        re.compile(
            r"""(?P<key>\bmodel\b)\s*[:=]\s*["'](?P<val>[^"']+)["']""",
            re.MULTILINE,
        ),
    ]


def _compile_ruby_patterns(packages: tuple[str, ...]) -> list[re.Pattern[str]]:
    alt = "|".join(re.escape(p) for p in packages)
    return [
        re.compile(rf'require\s+["\'](?P<pkg>{alt})["\']', re.MULTILINE),
        re.compile(
            r"""(?P<key>\bmodel\b)\s*[:=]\s*["'](?P<val>[^"']+)["']""",
            re.MULTILINE,
        ),
    ]


def _compile_csharp_patterns(packages: tuple[str, ...]) -> list[re.Pattern[str]]:
    pats: list[re.Pattern[str]] = []
    for p in packages:
        pats.append(
            re.compile(rf"using\s+(?P<pkg>{re.escape(p)})\s*;", re.MULTILINE),
        )
    pats.extend(
        [
            re.compile(
                r"""(?P<key>\bModel\b|\bmodel\b)\s*=\s*["'](?P<val>[^"']+)["']""",
                re.MULTILINE,
            ),
            re.compile(r'\.model\s*\(\s*["\'](?P<val>[^"\']+)["\']', re.MULTILINE),
        ]
    )
    return pats


_AGENT_RE = re.compile(r"\bnew\s+Agent\s*\(", re.MULTILINE)
_TOOL_RES = [
    re.compile(r"\bnew\s+\w*Tool\s*\(", re.MULTILINE),
    re.compile(r"@tool\s*\(", re.MULTILINE),
    re.compile(r"\bcreateTool\s*\(", re.MULTILINE),
    re.compile(r"\bFunctionTool\s*\(", re.MULTILINE),
]

_MODEL_GENERIC_RES = [
    re.compile(
        r"""\.model\s*\(\s*["'](?P<val>[^"']+)["']""",
        re.MULTILINE,
    ),
    re.compile(
        r"""(?P<key>\bmodel\b|\bModel\b)\s*[:=]\s*["'](?P<val>[^"']+)["']""",
        re.MULTILINE,
    ),
]

_GO_ALT = _compile_go_patterns(_ECOSYSTEM_PACKAGES["go"][1])

_REGEX_BY_LANG: dict[str, list[re.Pattern[str]]] = {
    "jsts": _compile_js_ts_patterns(_ECOSYSTEM_PACKAGES["jsts"][1]),
    "java": _compile_java_patterns(_ECOSYSTEM_PACKAGES["java"][1]),
    "go": [],
    "rust": _compile_rust_patterns(_ECOSYSTEM_PACKAGES["rust"][1]),
    "ruby": _compile_ruby_patterns(_ECOSYSTEM_PACKAGES["ruby"][1]),
    "csharp": _compile_csharp_patterns(_ECOSYSTEM_PACKAGES["csharp"][1]),
}

_GO_MODEL_RE = re.compile(
    r"""(?P<key>\bmodel\b)\s*[:=]\s*["'](?P<val>[^"']+)["']""",
    re.MULTILINE,
)


def _go_import_strings(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    _, packages = _ECOSYSTEM_PACKAGES["go"]
    for m in re.finditer(rf'import\s+"(?P<pkg>{_GO_ALT})"', text, re.MULTILINE):
        matched = _match_known_package(m.group("pkg"), packages)
        if matched:
            out.append((_line_at(text, m.start()), matched))
    for m in re.finditer(r"import\s*\((?P<block>[\s\S]*?)\)", text):
        block = m.group("block")
        base = m.start("block")
        for sm in re.finditer(rf'"(?P<pkg>{_GO_ALT})"', block):
            matched = _match_known_package(sm.group("pkg"), packages)
            if matched:
                pos = base + sm.start("pkg")
                out.append((_line_at(text, pos), matched))
    return out


def _load_ts_language_factory(ext: str) -> Optional[Callable[[], Any]]:
    if _TSLanguage is None:
        return None
    e = ext.lower()
    if e in (".ts", ".tsx"):
        try:
            import tree_sitter_typescript as tst  # type: ignore[import-untyped]

            if e == ".tsx":
                return tst.language_tsx  # type: ignore[attr-defined]
            return tst.language_typescript  # type: ignore[attr-defined]
        except ImportError:
            return None
    if e in (".js", ".mjs", ".jsx"):
        try:
            import tree_sitter_javascript as tsjs  # type: ignore[import-untyped]

            return tsjs.language  # type: ignore[attr-defined]
        except ImportError:
            return None
    try:
        import tree_sitter_javascript as tsjs  # type: ignore[import-untyped]

        return tsjs.language  # type: ignore[attr-defined]
    except ImportError:
        return None


def _load_language_factory(lang: str, ext: str) -> Optional[Callable[[], Any]]:
    if _TSLanguage is None:
        return None
    if lang == "jsts":
        return _load_ts_language_factory(ext)
    mod_map = {
        "java": ("tree_sitter_java", "language"),
        "go": ("tree_sitter_go", "language"),
        "rust": ("tree_sitter_rust", "language"),
        "ruby": ("tree_sitter_ruby", "language"),
        "csharp": ("tree_sitter_c_sharp", "language"),
    }
    if lang not in mod_map:
        return None
    mod_name, attr = mod_map[lang]
    try:
        mod = __import__(mod_name, fromlist=[attr])
    except ImportError:
        return None
    fn = getattr(mod, attr, None)
    return fn if callable(fn) else None


_parsers: dict[tuple[str, str], Any] = {}


def _parser_for(lang: str, ext: str) -> Optional[Any]:
    if _TSParser is None or _TSLanguage is None:
        return None
    key = (lang, ext)
    if key in _parsers:
        return _parsers[key]
    factory = _load_language_factory(lang, ext)
    if factory is None:
        _parsers[key] = None
        return None
    try:
        lang_obj = _TSLanguage(factory())
        p = _TSParser(lang_obj)
        _parsers[key] = p
        return p
    except Exception:
        _parsers[key] = None
        return None


def _node_text(source: bytes, start: int, end: int) -> str:
    return source[start:end].decode("utf-8", errors="replace")


def _walk_ts_imports_js_like(
    node: Any, source: bytes, packages: tuple[str, ...], out: list[tuple[int, str]]
) -> None:
    dec = source.decode("utf-8", errors="replace")
    stack = [node]
    while stack:
        n = stack.pop()
        t = n.type
        if t == "import_statement":
            for c in n.children:
                if c.type == "string":
                    for sc in c.children:
                        if sc.type == "string_fragment":
                            raw = _node_text(source, sc.start_byte, sc.end_byte)
                            m = _match_known_package(raw, packages)
                            if m:
                                out.append((_line_at(dec, sc.start_byte), m))
        elif t == "call_expression":
            fn = (
                n.child_by_field_name("function")
                if hasattr(n, "child_by_field_name")
                else None
            )
            if fn is None and n.children:
                fn = n.children[0]
            if fn is not None and _node_text(source, fn.start_byte, fn.end_byte) == "require":
                args = (
                    n.child_by_field_name("arguments")
                    if hasattr(n, "child_by_field_name")
                    else None
                )
                if args is None and len(n.children) > 1:
                    args = n.children[1]
                if args is not None:
                    for ac in getattr(args, "children", []) or []:
                        if ac.type == "string":
                            for sc in ac.children:
                                if sc.type == "string_fragment":
                                    raw = _node_text(source, sc.start_byte, sc.end_byte)
                                    matched = _match_known_package(raw, packages)
                                    if matched:
                                        out.append((_line_at(dec, sc.start_byte), matched))
        for c in reversed(n.children):
            stack.append(c)


def _walk_java_imports(
    node: Any, source: bytes, packages: tuple[str, ...], out: list[tuple[int, str]]
) -> None:
    dec = source.decode("utf-8", errors="replace")
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "import_declaration":
            frag = _node_text(source, n.start_byte, n.end_byte)
            m = re.search(
                r"import\s+(?:static\s+)?([\w.]+)\s*;",
                frag,
            )
            if m:
                matched = _match_known_package(m.group(1), packages)
                if matched:
                    out.append((_line_at(dec, n.start_byte), matched))
        for c in reversed(n.children):
            stack.append(c)


def _walk_go_imports(
    node: Any, source: bytes, packages: tuple[str, ...], out: list[tuple[int, str]]
) -> None:
    dec = source.decode("utf-8", errors="replace")
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "import_spec":
            path_node = (
                n.child_by_field_name("path")
                if hasattr(n, "child_by_field_name")
                else None
            )
            if path_node is None:
                for c in n.children:
                    if c.type == "interpreted_string_literal":
                        path_node = c
                        break
            if path_node is not None:
                raw = _node_text(source, path_node.start_byte, path_node.end_byte).strip(
                    '"'
                )
                matched = _match_known_package(raw, packages)
                if matched:
                    out.append((_line_at(dec, path_node.start_byte), matched))
        for c in reversed(n.children):
            stack.append(c)


def _walk_rust_uses(
    node: Any, source: bytes, packages: tuple[str, ...], out: list[tuple[int, str]]
) -> None:
    dec = source.decode("utf-8", errors="replace")
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "use_declaration":
            frag = _node_text(source, n.start_byte, n.end_byte)
            m = re.match(r"\s*use\s+([\w:]+)", frag)
            if m:
                path = m.group(1).split("::")[0].removeprefix("crate::")
                matched = _match_known_package(path, packages)
                if matched:
                    out.append((_line_at(dec, n.start_byte), matched))
        for c in reversed(n.children):
            stack.append(c)


def _walk_ruby_requires(
    node: Any, source: bytes, packages: tuple[str, ...], out: list[tuple[int, str]]
) -> None:
    dec = source.decode("utf-8", errors="replace")
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "call":
            fn = (
                n.child_by_field_name("method")
                if hasattr(n, "child_by_field_name")
                else None
            )
            if fn is None and n.children:
                fn = n.children[0]
            if fn is not None and _node_text(source, fn.start_byte, fn.end_byte) == "require":
                args = (
                    n.child_by_field_name("arguments")
                    if hasattr(n, "child_by_field_name")
                    else None
                )
                if args is None and len(n.children) > 1:
                    args = n.children[1]
                if args is not None:
                    for ac in getattr(args, "children", []) or []:
                        if ac.type == "string":
                            for sc in ac.children:
                                if sc.type in ("string_content", "string_fragment"):
                                    raw = _node_text(source, sc.start_byte, sc.end_byte)
                                    matched = _match_known_package(raw, packages)
                                    if matched:
                                        out.append((_line_at(dec, sc.start_byte), matched))
        for c in reversed(n.children):
            stack.append(c)


def _walk_csharp_usings(
    node: Any, source: bytes, packages: tuple[str, ...], out: list[tuple[int, str]]
) -> None:
    dec = source.decode("utf-8", errors="replace")
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "using_directive":
            name = (
                n.child_by_field_name("name")
                if hasattr(n, "child_by_field_name")
                else None
            )
            if name is not None:
                raw = _node_text(source, name.start_byte, name.end_byte)
                matched = _match_known_package(raw, packages)
                if matched:
                    out.append((_line_at(dec, name.start_byte), matched))
        for c in reversed(n.children):
            stack.append(c)


def _tree_sitter_imports(
    lang: str, ext: str, text: str, packages: tuple[str, ...]
) -> list[tuple[int, str]]:
    parser = _parser_for(lang, ext)
    if parser is None:
        return []
    source = text.encode("utf-8")
    try:
        tree = parser.parse(source)
    except Exception:
        return []
    root = tree.root_node
    out: list[tuple[int, str]] = []
    if lang == "jsts":
        _walk_ts_imports_js_like(root, source, packages, out)
    elif lang == "java":
        _walk_java_imports(root, source, packages, out)
    elif lang == "go":
        _walk_go_imports(root, source, packages, out)
    elif lang == "rust":
        _walk_rust_uses(root, source, packages, out)
    elif lang == "ruby":
        _walk_ruby_requires(root, source, packages, out)
    elif lang == "csharp":
        _walk_csharp_usings(root, source, packages, out)
    return out


def _tree_sitter_model_strings(lang: str, ext: str, text: str) -> list[tuple[int, str]]:
    parser = _parser_for(lang, ext)
    if parser is None:
        return []
    source = text.encode("utf-8")
    try:
        tree = parser.parse(source)
    except Exception:
        return []
    dec = source.decode("utf-8", errors="replace")
    out: list[tuple[int, str]] = []
    stack = [tree.root_node]
    string_types = frozenset(
        {
            "string",
            "string_literal",
            "interpreted_string_literal",
            "verbatim_string_literal",
        }
    )
    while stack:
        n = stack.pop()
        if n.type in string_types:
            raw = _node_text(source, n.start_byte, n.end_byte)
            inner = raw.strip('"\'`')
            if _is_plausible_model_id(inner):
                prefix = dec[max(0, n.start_byte - 48) : n.start_byte]
                if re.search(
                    r"(?:\bmodel\b|\bModel\b)\s*[:=]\s*$|\.model\s*\(\s*$",
                    prefix,
                    re.MULTILINE,
                ):
                    out.append((_line_at(dec, n.start_byte), inner))
        for c in reversed(n.children):
            stack.append(c)
    return out


def _regex_scan_file(
    path: Path,
    rel: str,
    lang: str,
    text: str,
) -> list[AIComponent]:
    components: list[AIComponent] = []
    seen: set[tuple[int, str, str]] = set()
    ecosystem, packages = _ECOSYSTEM_PACKAGES[lang]
    patterns = _REGEX_BY_LANG[lang]

    def add_dep(line: int, pkg: str) -> None:
        key = (line, "dep", pkg)
        if key in seen:
            return
        seen.add(key)
        components.append(
            AIComponent(
                name=pkg,
                component_type=AIComponentType.DEPENDENCY,
                file_path=rel,
                line_number=line,
                framework=ecosystem,
                detection_source=DetectionSource.CODE_ANALYSIS,
                confidence=0.85,
                metadata={
                    "ecosystem": ecosystem,
                    "package": pkg,
                    "scanner": "multi_language_scanner",
                },
            )
        )

    def add_model(line: int, model_id: str) -> None:
        key = (line, "model", model_id)
        if key in seen:
            return
        seen.add(key)
        components.append(
            AIComponent(
                name=model_id,
                component_type=AIComponentType.MODEL,
                file_path=rel,
                line_number=line,
                framework=ecosystem,
                detection_source=DetectionSource.CODE_ANALYSIS,
                confidence=0.75,
                model_name=model_id,
                metadata={"scanner": "multi_language_scanner"},
            )
        )

    def add_agent(line: int) -> None:
        key = (line, "agent", "Agent")
        if key in seen:
            return
        seen.add(key)
        components.append(
            AIComponent(
                name="Agent",
                component_type=AIComponentType.AGENT,
                file_path=rel,
                line_number=line,
                framework=ecosystem,
                detection_source=DetectionSource.CODE_ANALYSIS,
                confidence=0.7,
                metadata={"scanner": "multi_language_scanner"},
            )
        )

    def add_tool(line: int, hint: str) -> None:
        key = (line, "tool", hint)
        if key in seen:
            return
        seen.add(key)
        components.append(
            AIComponent(
                name=hint,
                component_type=AIComponentType.TOOL,
                file_path=rel,
                line_number=line,
                framework=ecosystem,
                detection_source=DetectionSource.CODE_ANALYSIS,
                confidence=0.65,
                metadata={"scanner": "multi_language_scanner"},
            )
        )

    for pat in patterns:
        for m in pat.finditer(text):
            line = _line_at(text, m.start())
            gd = m.groupdict()
            if gd.get("val"):
                val = _normalize_model_literal(gd["val"])
                if _is_plausible_model_id(val):
                    add_model(line, val)
                continue
            if gd.get("pkg"):
                matched = _match_known_package(gd["pkg"], packages)
                if matched:
                    add_dep(line, matched)
                continue
            g0 = m.group(0)
            if "OpenAI" in g0 or "ChatOpenAI" in g0:
                add_dep(line, "openai")
            elif (
                "OpenAIClient" in g0
                or "ChatCompletionsClient" in g0
                or "ChatClient" in g0
            ):
                add_dep(line, "com.azure.ai.openai")

    if lang == "go":
        for line, pkg in _go_import_strings(text):
            add_dep(line, pkg)
        for m in _GO_MODEL_RE.finditer(text):
            if m.group("val"):
                val = _normalize_model_literal(m.group("val"))
                if _is_plausible_model_id(val):
                    add_model(_line_at(text, m.start()), val)

    for pat in _MODEL_GENERIC_RES:
        for m in pat.finditer(text):
            gd = m.groupdict()
            if gd.get("val"):
                val = _normalize_model_literal(gd["val"])
                if _is_plausible_model_id(val):
                    add_model(_line_at(text, m.start()), val)

    for m in _AGENT_RE.finditer(text):
        add_agent(_line_at(text, m.start()))

    for i, tr in enumerate(_TOOL_RES):
        for m in tr.finditer(text):
            add_tool(_line_at(text, m.start()), f"tool_pattern_{i}")

    for line, pkg in _tree_sitter_imports(lang, path.suffix, text, packages):
        add_dep(line, pkg)

    for line, mid in _tree_sitter_model_strings(lang, path.suffix, text):
        norm = _normalize_model_literal(mid)
        if _is_plausible_model_id(norm):
            add_model(line, norm)

    return components


def _iter_scan_files(context: ScanContext) -> list[tuple[Path, str, str]]:
    spec = _build_pathspec(context.exclude_patterns)
    out: list[tuple[Path, str, str]] = []
    for scan_root in context.paths:
        root = Path(scan_root)
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() == ".py":
                continue
            lang = _EXTENSION_TO_LANG.get(root.suffix.lower())
            if lang is None:
                continue
            rel = root.name
            if spec and spec.match_file(rel):
                continue
            out.append((root.resolve(), rel, lang))
            continue
        base = root.resolve()
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() == ".py":
                continue
            lang = _EXTENSION_TO_LANG.get(f.suffix.lower())
            if lang is None:
                continue
            try:
                rel = f.resolve().relative_to(base).as_posix()
            except ValueError:
                rel = f.as_posix()
            if spec and spec.match_file(rel):
                continue
            out.append((f.resolve(), rel, lang))
    return out


class MultiLanguageScanner(BaseScanner):
    name = "multi_language_scanner"

    def supports(self, context: ScanContext) -> bool:
        return bool(_iter_scan_files(context))

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components: list[AIComponent] = []
        for path, rel, lang in _iter_scan_files(context):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            components.extend(_regex_scan_file(path, rel, lang, text))
        return components, []
