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

"""Scanner that detects env var references in AI-related keyword arguments.

Tier 2 of the three-tier detection architecture.  This scanner finds patterns
like ``model=os.getenv("LLM_MODEL")`` and emits components with
``metadata.env`` set so that the cross-ref resolution pass can fill in the
actual value from config files (.env, docker-compose, Helm values, etc.).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

from pathspec import PathSpec

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource
from .base import BaseScanner
from .file_cache import is_python_source, read_python_source, read_text_cached

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env-var access patterns per language
# ---------------------------------------------------------------------------

_PY_GETENV = re.compile(
    r"""os\.getenv\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']"""
)
_PY_ENVIRON_BRACKET = re.compile(
    r"""os\.environ\[\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*\]"""
)
_PY_ENVIRON_GET = re.compile(
    r"""os\.environ\.get\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']"""
)
_PY_ENV_PATTERNS = [_PY_GETENV, _PY_ENVIRON_BRACKET, _PY_ENVIRON_GET]

_JS_PROCESS_ENV_DOT = re.compile(
    r"""process\.env\.([A-Za-z_][A-Za-z0-9_]*)"""
)
_JS_PROCESS_ENV_BRACKET = re.compile(
    r"""process\.env\[\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*\]"""
)
_JS_ENV_PATTERNS = [_JS_PROCESS_ENV_DOT, _JS_PROCESS_ENV_BRACKET]

_GO_GETENV = re.compile(
    r"""os\.Getenv\(\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\)"""
)

_JAVA_GETENV = re.compile(
    r"""System\.getenv\(\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\)"""
)

_RUBY_ENV_BRACKET = re.compile(
    r"""ENV\[\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*\]"""
)
_RUBY_ENV_FETCH = re.compile(
    r"""ENV\.fetch\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']"""
)
_RUBY_ENV_PATTERNS = [_RUBY_ENV_BRACKET, _RUBY_ENV_FETCH]

_LANG_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    ".py": _PY_ENV_PATTERNS,
    ".ipynb": _PY_ENV_PATTERNS,
    ".js": _JS_ENV_PATTERNS,
    ".ts": _JS_ENV_PATTERNS,
    ".jsx": _JS_ENV_PATTERNS,
    ".tsx": _JS_ENV_PATTERNS,
    ".mjs": _JS_ENV_PATTERNS,
    ".go": [_GO_GETENV],
    ".java": [_JAVA_GETENV],
    ".rb": _RUBY_ENV_PATTERNS,
}

# ---------------------------------------------------------------------------
# AI-related kwarg contexts
# ---------------------------------------------------------------------------

_MODEL_KWARGS = frozenset({
    "model", "model_name", "model_id", "deployment_name",
})
_API_KEY_KWARGS = frozenset({
    "api_key", "openai_api_key", "anthropic_api_key",
    "huggingface_api_key", "hf_token", "cohere_api_key",
    "google_api_key", "replicate_api_token",
})
_ENDPOINT_KWARGS = frozenset({
    "base_url", "endpoint", "azure_endpoint", "api_base",
})

_ALL_AI_KWARGS = _MODEL_KWARGS | _API_KEY_KWARGS | _ENDPOINT_KWARGS

_KWARG_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ALL_AI_KWARGS)) + r")"
    r"""\s*[:=]\s*""",
)

_MODEL_VAR_NAMES = frozenset({
    "model", "model_name", "model_id", "llm_model",
    "deployment_name", "engine",
})

_AI_KEY_ENV_RE = re.compile(
    r"(OPENAI|ANTHROPIC|HUGGING_?FACE|HF_|COHERE|GOOGLE_AI|REPLICATE|"
    r"MISTRAL|TOGETHER|AZURE_OPENAI|BEDROCK|VERTEX|GROQ|FIREWORKS|"
    r"ANYSCALE|DEEPINFRA|PERPLEXITY)"
    r".*(?:KEY|TOKEN|SECRET)\b",
    re.IGNORECASE,
)

_ASSIGN_LHS_RE = re.compile(
    r"""(?:^|\n)\s*(?:(?:const|let|var|final)\s+)?"""
    r"""([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*""",
)

_INFRA_ENV_PREFIXES: tuple[str, ...] = (
    "TEMPORAL_", "OTEL_", "KAFKA_", "REDIS_", "POSTGRES_", "PG_",
    "MYSQL_", "MONGO_", "RABBIT", "GRPC_", "HTTP_", "HTTPS_",
    "LOG_", "DEBUG_", "NODE_", "K8S_", "KUBE_", "DOCKER_",
    "VAULT_", "CONSUL_", "ETCD_", "JAEGER_", "ZIPKIN_",
    "DATADOG_", "DD_", "PROMETHEUS_", "GRAFANA_", "ELASTIC_",
    "SPLUNK_", "SENTRY_", "MY_NODE_", "CLUSTER_", "POD_",
)

# ---------------------------------------------------------------------------
# Vault / Secret-manager patterns  (Fix 3: generalized secret fetch)
#
# Matches programmatic secret retrieval calls across providers:
#   - HashiCorp Vault:  client.read("secret/..."), vault_client.read(...)
#   - Conjur:           conjur_client.get_secret("alias")
#   - AWS Secrets Mgr:  client.get_secret_value(SecretId="...")
#   - Azure Key Vault:  client.get_secret("name")
#   - GCP Secret Mgr:   client.access_secret_version(request=...)
#   - Generic:          read_secret("name"), get_secret("name")
# ---------------------------------------------------------------------------

_VAULT_SECRET_CALL_RE = re.compile(
    r"""(?:"""
    r"""\.get_secret(?:_value)?\s*\("""       # conjur, Azure KV, generic
    r"""|\.read_secret\s*\("""                # generic
    r"""|\.access_secret_version\s*\("""      # GCP Secret Manager
    r"""|vault[_\.].*\.read\s*\("""           # HashiCorp Vault
    r"""|secrets_manager.*\.get_secret_value\s*\("""  # AWS explicit
    r""")""",
    re.IGNORECASE,
)

_VAULT_SECRET_STRING_RE = re.compile(
    r"""["'](secret[/:][\w/.-]+|"""
    r"""SecretId[=:]\s*[\w/.-]+)["']""",
)

_VAULT_IMPORT_RE = re.compile(
    r"""(?:"""
    r"""from\s+(?:hvac|conjur|azure\.keyvault|google\.cloud\.secretmanager"""
    r"""|botocore|boto3)"""
    r"""|import\s+(?:hvac|conjur))""",
)

_VAULT_STRING_LITERAL_RE = re.compile(
    r"""["'](?:secret/|conjur/|vault:|arn:aws:secretsmanager:)[\w/.:-]+["']""",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# File iteration (shared pattern with other scanners)
# ---------------------------------------------------------------------------


def _build_pathspec(patterns: list[str] | tuple[str, ...]) -> PathSpec | None:
    if not patterns:
        return None
    return PathSpec.from_lines("gitwildmatch", patterns)


def _iter_files(context: ScanContext) -> Iterator[tuple[Path, str]]:
    idx = context.file_index()
    if idx:
        for entries in idx.values():
            for entry in entries:
                try:
                    rel = entry.path.relative_to(entry.root).as_posix()
                except ValueError:
                    rel = entry.path.as_posix()
                yield entry.path, rel
        return

    spec = _build_pathspec(context.exclude_patterns)
    for scan_root in context.paths:
        root = Path(scan_root)
        if not root.exists():
            continue
        if root.is_file():
            rel = root.name
            if spec and spec.match_file(rel):
                continue
            yield root, rel
            continue
        base = root.resolve()
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            try:
                rel = f.resolve().relative_to(base).as_posix()
            except ValueError:
                rel = f.as_posix()
            if spec and spec.match_file(rel):
                continue
            yield f, rel


def _line_number(text: str, pos: int) -> int:
    return text[:pos].count("\n") + 1


_COMMENT_PREFIXES = {
    ".py": "#",
    ".ipynb": "#",
    ".rb": "#",
    ".js": "//",
    ".ts": "//",
    ".jsx": "//",
    ".tsx": "//",
    ".mjs": "//",
    ".go": "//",
    ".java": "//",
}


def _is_commented(text: str, pos: int, suffix: str) -> bool:
    """Return True if *pos* falls on a line that is a comment."""
    prefix = _COMMENT_PREFIXES.get(suffix)
    if not prefix:
        return False
    line_start = text.rfind("\n", 0, pos) + 1
    line_text = text[line_start:pos + 80].split("\n", 1)[0]
    return line_text.lstrip().startswith(prefix)


# ---------------------------------------------------------------------------
# Context classification
# ---------------------------------------------------------------------------


def _classify_kwarg(text: str, env_match_start: int) -> str | None:
    """Look backward from an env-var match to find the nearest AI kwarg assignment."""
    window_start = max(0, env_match_start - 120)
    window = text[window_start:env_match_start]

    best_pos = -1
    best_ctx = None
    for m in _KWARG_RE.finditer(window):
        kwarg_name = m.group(1)
        if m.end() > best_pos:
            best_pos = m.end()
            if kwarg_name in _MODEL_KWARGS:
                best_ctx = "model_kwarg"
            elif kwarg_name in _API_KEY_KWARGS:
                best_ctx = "api_key_kwarg"
            elif kwarg_name in _ENDPOINT_KWARGS:
                best_ctx = "endpoint_kwarg"
    return best_ctx


def _classify_assignment(text: str, env_match_start: int) -> str | None:
    """Check if the env var is assigned to a model-like variable name."""
    window_start = max(0, env_match_start - 80)
    window = text[window_start:env_match_start]

    for m in _ASSIGN_LHS_RE.finditer(window):
        var_name = m.group(1).lower()
        if var_name in _MODEL_VAR_NAMES:
            return "model_assignment"
    return None


def _env_context_to_component_type(ctx: str) -> AIComponentType:
    if ctx in ("model_kwarg", "model_assignment"):
        return AIComponentType.MODEL
    if ctx in ("api_key_kwarg", "api_key_envname"):
        return AIComponentType.SECRET
    if ctx == "endpoint_kwarg":
        return AIComponentType.DEPENDENCY
    return AIComponentType.MODEL


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class EnvVarResolver(BaseScanner):
    name = "env_var_resolver"

    def supports(self, context: ScanContext) -> bool:
        return any(Path(p).exists() for p in context.paths)

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components: list[AIComponent] = []
        seen: set[tuple[str, str, str]] = set()

        for fpath, _rel in _iter_files(context):
            suffix = fpath.suffix.lower()
            patterns = _LANG_PATTERNS.get(suffix)
            if not patterns:
                continue

            try:
                text = (
                    read_python_source(fpath)
                    if is_python_source(fpath)
                    else read_text_cached(fpath)
                )
            except OSError:
                continue

            fp_str = str(fpath)

            for rx in patterns:
                for m in rx.finditer(text):
                    env_name = m.group(1)
                    match_start = m.start()

                    if _is_commented(text, match_start, suffix):
                        continue

                    ctx = _classify_kwarg(text, match_start)
                    if ctx is None:
                        ctx = _classify_assignment(text, match_start)
                    if ctx is None and _AI_KEY_ENV_RE.match(env_name):
                        ctx = "api_key_envname"
                    if ctx is None:
                        continue

                    env_upper = env_name.upper()
                    if any(env_upper.startswith(p) for p in _INFRA_ENV_PREFIXES):
                        continue

                    dedup_key = (fp_str, env_name, ctx)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    comp_type = _env_context_to_component_type(ctx)
                    line_no = _line_number(text, match_start)

                    components.append(AIComponent(
                        name=f"env:{env_name}",
                        component_type=comp_type,
                        file_path=fp_str,
                        line_number=line_no,
                        framework="",
                        detection_source=DetectionSource.CODE_ANALYSIS,
                        model_name=None,
                        confidence=0.3,
                        needs_agentic=True,
                        agentic_hint=(
                            f"env var {env_name} used as {ctx}; "
                            f"value unknown until cross-ref resolution"
                        ),
                        metadata={
                            "env": env_name,
                            "env_context": ctx,
                        },
                    ))

        vault_comps = self._detect_vault_secrets(context, seen)
        components.extend(vault_comps)

        return components, []

    def _detect_vault_secrets(
        self,
        context: ScanContext,
        already_seen: set[tuple[str, str, str]],
    ) -> list[AIComponent]:
        """Detect secrets fetched via vault/secret-manager SDKs."""
        results: list[AIComponent] = []
        for fpath, _rel in _iter_files(context):
            suffix = fpath.suffix.lower()
            if suffix not in (".py", ".ipynb", ".go", ".java", ".js", ".ts", ".rb"):
                continue

            try:
                text = (
                    read_python_source(fpath)
                    if is_python_source(fpath)
                    else read_text_cached(fpath)
                )
            except OSError:
                continue

            fp_str = str(fpath)
            has_vault_import = bool(_VAULT_IMPORT_RE.search(text))

            for m in _VAULT_SECRET_CALL_RE.finditer(text):
                if _is_commented(text, m.start(), suffix):
                    continue
                line_no = _line_number(text, m.start())
                dedup = (fp_str, f"vault_call_L{line_no}", "vault")
                if dedup in already_seen:
                    continue
                already_seen.add(dedup)

                secret_name = ""
                str_m = _VAULT_SECRET_STRING_RE.search(
                    text[m.start():m.start() + 200]
                )
                if str_m:
                    secret_name = str_m.group(1)

                conf = 0.8 if has_vault_import else 0.4
                results.append(AIComponent(
                    name=secret_name or "vault-secret",
                    component_type=AIComponentType.SECRET,
                    file_path=fp_str,
                    line_number=line_no,
                    detection_source=DetectionSource.CODE_ANALYSIS,
                    confidence=conf,
                    needs_agentic=conf < 0.6,
                    agentic_hint=(
                        "Secret fetched via vault/secret-manager SDK; "
                        "agent should verify the secret path/alias."
                    ) if conf < 0.6 else "",
                    metadata={
                        "secret_source": "vault_sdk",
                        "secret_path": secret_name,
                        "has_vault_import": has_vault_import,
                    },
                ))

            if has_vault_import and not any(
                d[0] == fp_str and d[2] == "vault" for d in already_seen
            ):
                line_no = _line_number(text, _VAULT_IMPORT_RE.search(text).start())  # type: ignore[union-attr]
                dedup = (fp_str, "vault_import_hint", "vault")
                if dedup not in already_seen:
                    already_seen.add(dedup)
                    results.append(AIComponent(
                        name="vault-secret-candidate",
                        component_type=AIComponentType.SECRET,
                        file_path=fp_str,
                        line_number=line_no,
                        detection_source=DetectionSource.CODE_ANALYSIS,
                        confidence=0.3,
                        needs_agentic=True,
                        agentic_hint=(
                            "vault_secret_pattern: File imports a vault/secret-manager "
                            "SDK but no explicit get_secret call was matched. "
                            "Use analyze_imports to trace secret access patterns."
                        ),
                        metadata={
                            "secret_source": "vault_import_only",
                            "has_vault_import": True,
                        },
                    ))

            for m in _VAULT_STRING_LITERAL_RE.finditer(text):
                if _is_commented(text, m.start(), suffix):
                    continue
                line_no = _line_number(text, m.start())
                dedup = (fp_str, f"vault_str_L{line_no}", "vault")
                if dedup in already_seen:
                    continue
                already_seen.add(dedup)
                secret_path = m.group(0).strip("\"'")
                results.append(AIComponent(
                    name=secret_path,
                    component_type=AIComponentType.SECRET,
                    file_path=fp_str,
                    line_number=line_no,
                    detection_source=DetectionSource.CODE_ANALYSIS,
                    confidence=0.5 if has_vault_import else 0.3,
                    needs_agentic=True,
                    agentic_hint=(
                        "vault_secret_pattern: String literal references a "
                        "secret path. Use analyze_imports to confirm this is "
                        "a vault/secret-manager access."
                    ),
                    metadata={
                        "secret_source": "string_literal",
                        "secret_path": secret_path,
                        "has_vault_import": has_vault_import,
                    },
                ))

        return results
