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

"""KB Enrichment Scanner -- combines LibCST parsing with the DuckDB knowledge
base to detect and classify AI framework usage in Python source code.

Unlike the legacy ``categorize_symbols`` path which emits every KB match, this
scanner filters results to only high-signal AI asset concepts: agents, models,
tools, vector stores, embeddings, prompts, memory, and retrievers.  When no KB
is installed the scanner gracefully no-ops, letting the other v2 detectors
carry the workload.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..catalog_db import CatalogDB
from ..cst_parser import parse_source_code
from ..db_loader import (
    DatabaseLoadError,
    UnsupportedDatabaseSchemaError,
    ensure_local_database,
    require_supported_manifest_schema,
)
from ..models import (
    AIComponent,
    AIComponentType,
    ComponentRelationship,
    DetectionSource,
    ScanContext,
)
from ..structures import CodeAnalysisResult
from .base import BaseScanner
from .file_cache import read_python_source, read_text_cached

_LOGGER = logging.getLogger(__name__)


def _import_strings(imports: list) -> list[str]:
    """Extract plain import statement strings from a list that may contain
    ``(line_number, stmt)`` tuples or bare strings."""
    return [entry[1] if isinstance(entry, tuple) else entry for entry in imports]


# Concept -> component type mapping.
#
# This is the single source of truth for which knowledge-base concepts the
# CLI is willing to surface; ``ALLOWED_CONCEPTS`` is derived from its keys so
# the two never drift. The original set covered ten concepts. The expanded
# vocabulary adds the operational and MCP concepts (guardrail, mcp_server,
# mcp_client, skill, observability) plus the ML-lifecycle and data concepts
# (dataset, training_run, model_artifact, vector_store) so that knowledge-base
# rows carrying those concepts are mapped to a component type instead of being
# silently dropped at lookup time.
#
# Concepts that have no dedicated ``AIComponentType`` yet (reranker, evaluator,
# framework_core) are mapped to ``OTHER`` rather than dropped, so the evidence
# survives with the original concept preserved in ``kb_concept`` for downstream
# refinement. ``datastore`` remains an alias for the vector-store type.
_CONCEPT_TO_TYPE: dict[str, AIComponentType] = {
    "agent": AIComponentType.AGENT,
    "model": AIComponentType.MODEL,
    "tool": AIComponentType.TOOL,
    "datastore": AIComponentType.VECTOR_STORE,
    "vector_store": AIComponentType.VECTOR_STORE,
    "embedding": AIComponentType.EMBEDDING,
    "prompt": AIComponentType.PROMPT,
    "memory": AIComponentType.MEMORY,
    "retriever": AIComponentType.RETRIEVER,
    "knowledge_base": AIComponentType.KNOWLEDGE_BASE,
    "feature_store": AIComponentType.FEATURE_STORE,
    # Operational + MCP concepts.
    "guardrail": AIComponentType.GUARDRAIL,
    "mcp_server": AIComponentType.MCP_SERVER,
    "mcp_client": AIComponentType.MCP_CLIENT,
    "skill": AIComponentType.SKILL,
    "observability": AIComponentType.OBSERVABILITY,
    # Data + ML-lifecycle concepts.
    "dataset": AIComponentType.DATASET,
    "training_run": AIComponentType.TRAINING_RUN,
    "model_artifact": AIComponentType.MODEL_ARTIFACT,
    # Concepts without a dedicated type yet: keep the evidence as OTHER and
    # preserve the original concept in ``kb_concept`` instead of dropping.
    "reranker": AIComponentType.OTHER,
    "evaluator": AIComponentType.OTHER,
    "framework_core": AIComponentType.OTHER,
}

# Concepts the CLI will surface. Derived from the mapping above so a new
# concept only has to be added in one place.
ALLOWED_CONCEPTS: frozenset[str] = frozenset(_CONCEPT_TO_TYPE)

# KB id path segments that override or suppress the raw concept.
# Checked in order; first match wins.  ``None`` means "exclude this entry".
_ID_PATH_OVERRIDES: list[tuple[str, AIComponentType | None]] = [
    (".chains.", None),
    (".document_loaders.", None),
    (".text_splitter.", None),
    (".text_splitters.", None),
    (".retrievers.", AIComponentType.RETRIEVER),
    (".memory.", AIComponentType.MEMORY),
    (".vectorstores.", AIComponentType.VECTOR_STORE),
    (".embeddings.", AIComponentType.EMBEDDING),
    (".agents.", AIComponentType.AGENT),
    (".tools.", AIComponentType.TOOL),
    (".prompts.", AIComponentType.PROMPT),
]


_EXCLUDED_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "RecursiveCharacterTextSplitter",
        "CharacterTextSplitter",
        "TokenTextSplitter",
        "TextSplitter",
        "SentenceTransformersTokenTextSplitter",
        "SpacyTextSplitter",
        "NLTKTextSplitter",
        "TextLoader",
        "DirectoryLoader",
        "WebBaseLoader",
        "PyPDFLoader",
        "CSVLoader",
        "UnstructuredFileLoader",
        "LLMChain",
        "RetrievalQA",
        "ConversationalRetrievalChain",
        "SequentialChain",
        "SimpleSequentialChain",
        "ConversationChain",
    }
)


def _refine_type_from_kb_id(
    kb_id: str, concept_type: AIComponentType
) -> AIComponentType | None:
    """Apply path-based and class-name overrides to correct KB misclassifications.

    Returns ``None`` when the entry should be excluded entirely (e.g.
    chains, document loaders, text splitters that are not true AI assets).
    """
    class_name = kb_id.rsplit(".", 1)[-1] if "." in kb_id else kb_id
    if class_name in _EXCLUDED_CLASS_NAMES:
        return None
    for segment, override in _ID_PATH_OVERRIDES:
        if segment in kb_id:
            return override
    return concept_type


_AGENT_CREATION_PATTERNS: frozenset[str] = frozenset(
    {
        "initialize_agent",
        "AgentExecutor",
        "create_react_agent",
        "create_openai_functions_agent",
        "create_openai_tools_agent",
        "create_structured_chat_agent",
        "create_tool_calling_agent",
        "create_json_chat_agent",
        "create_xml_agent",
        "Agent",
        "Crew",
        "AssistantAgent",
        "UserProxyAgent",
        "GroupChat",
        "GroupChatManager",
    }
)

_AGENT_FRAMEWORK_PREFIXES: frozenset[str] = frozenset(
    {
        "langchain",
        # LangGraph ships the prebuilt ReAct agent factory
        # ``langgraph.prebuilt.create_react_agent`` (catalogued in the KB as an
        # agent). The call-pattern gate derives the prefix from the qualified
        # name's first segment, so ``langgraph`` must be listed here for those
        # module-level factory calls to be emitted as agents.
        "langgraph",
        "crewai",
        "autogen",
        # Strands Agents (https://strandsagents.com/) uses module-level
        # ``from strands import Agent`` followed by ``agent = Agent(...)``
        # (no class wrapper). Without ``strands`` here, the call-pattern
        # gate below rejects the call and the agent is never emitted, which
        # matches the symptom observed on published open-source Strands
        # sample repositories.
        "strands",
    }
)


def _is_agent_creation_call(qualified_name: str, imports: list) -> bool:
    """Return True if *qualified_name* is a known agent creation pattern
    from a recognized framework."""
    short = (
        qualified_name.rsplit(".", 1)[-1] if "." in qualified_name else qualified_name
    )
    if short not in _AGENT_CREATION_PATTERNS:
        return False
    prefix = qualified_name.split(".")[0].split("_")[0]
    if prefix in _AGENT_FRAMEWORK_PREFIXES:
        return True
    for imp in _import_strings(imports):
        parts = imp.split()
        mod = parts[1] if len(parts) > 1 else ""
        mod_prefix = mod.split(".")[0].split("_")[0]
        if mod_prefix in _AGENT_FRAMEWORK_PREFIXES and short in imp:
            return True
    return False


_STATIC_TOOL_PATTERNS: frozenset[str] = frozenset(
    {
        "Tool",
        "StructuredTool",
    }
)
_STATIC_MEMORY_PATTERNS: frozenset[str] = frozenset(
    {
        "ConversationBufferMemory",
        "ConversationSummaryMemory",
        "ConversationBufferWindowMemory",
        "ConversationKGMemory",
        "ConversationEntityMemory",
        "ConversationTokenBufferMemory",
        "VectorStoreRetrieverMemory",
        "ReadOnlySharedMemory",
    }
)
_STATIC_PROMPT_PATTERNS: frozenset[str] = frozenset(
    {
        "PromptTemplate",
        "ChatPromptTemplate",
        "FewShotPromptTemplate",
        "FewShotChatMessagePromptTemplate",
    }
)


@dataclass(frozen=True, slots=True)
class _MatchResult:
    """Outcome of matching an observation against the KB."""

    entry: dict[str, Any] | None = None
    partial_kb_id: str | None = None
    partial_kb_framework: str | None = None
    obs_module: str = ""

    @property
    def is_confirmed(self) -> bool:
        return self.entry is not None and self.partial_kb_id is None

    @property
    def is_partial(self) -> bool:
        return self.entry is None and self.partial_kb_id is not None


_AI_SUGGESTIVE_DIR_SEGMENTS: frozenset[str] = frozenset(
    {
        "models",
        "agents",
        "tools",
        "prompts",
        "embeddings",
        "llm",
        "inference",
        "ai",
        "ml",
        "vectorstores",
        "vector_stores",
        "retrievers",
        "memory",
        "chains",
        "chatbots",
    }
)

_AI_SUGGESTIVE_IMPORT_SEGMENTS: frozenset[str] = frozenset(
    {
        "llm",
        "openai",
        "anthropic",
        "embedding",
        "vector",
        "agent",
        "model",
        "inference",
        "chat",
        "completion",
        "bedrock",
        "vertex",
        "huggingface",
        "transformers",
        "torch",
        "tensorflow",
        "keras",
        "ollama",
        "cohere",
        "mistral",
        "gemini",
    }
)

_AI_INDICATIVE_CLASS_RE: re.Pattern[str] = re.compile(
    r"(LLM|Model|Agent|Embed(?:ding|der)|Vector|Chain|Tool|Prompt|Memory|Chat|"
    r"Retriever|Completion|Inference|Tokenizer|History|Conversation|ChatBuffer|"
    r"Guard(?:rail)?|Inspector|Rails|MCPClient|Traceloop)",
    re.IGNORECASE,
)


_CLASS_NAME_TYPE_MAP: list[tuple[re.Pattern[str], AIComponentType]] = [
    (re.compile(r"Embed(?:ding|der)", re.IGNORECASE), AIComponentType.EMBEDDING),
    (
        re.compile(r"Guard(?:rail)?|Inspector|Rails", re.IGNORECASE),
        AIComponentType.GUARDRAIL,
    ),
    (re.compile(r"MCPClient", re.IGNORECASE), AIComponentType.MCP_CLIENT),
    (re.compile(r"Traceloop", re.IGNORECASE), AIComponentType.OBSERVABILITY),
    (re.compile(r"Agent"), AIComponentType.AGENT),
    (re.compile(r"Tool"), AIComponentType.TOOL),
    (re.compile(r"Prompt"), AIComponentType.PROMPT),
    (re.compile(r"Memory|History|Conversation|ChatBuffer"), AIComponentType.MEMORY),
    (re.compile(r"KnowledgeBase", re.IGNORECASE), AIComponentType.KNOWLEDGE_BASE),
    (re.compile(r"FeatureStore", re.IGNORECASE), AIComponentType.FEATURE_STORE),
    (re.compile(r"Retriever", re.IGNORECASE), AIComponentType.RETRIEVER),
    (re.compile(r"Vector(?:Store|DB|Database)"), AIComponentType.VECTOR_STORE),
]


@dataclass(frozen=True)
class PlatformEntry:
    """A known AI platform with a primary type and optional additional types."""

    primary: AIComponentType
    additional: frozenset[AIComponentType] = field(default_factory=frozenset)

    @property
    def all_types(self) -> frozenset[AIComponentType]:
        return frozenset({self.primary}) | self.additional


_OBS = AIComponentType.OBSERVABILITY
_GRD = AIComponentType.GUARDRAIL
_REG = AIComponentType.MODEL_REGISTRY

_IMPORT_MODULE_TYPE_MAP: dict[str, PlatformEntry] = {
    # Pure observability
    "traceloop": PlatformEntry(_OBS),
    "openllmetry": PlatformEntry(_OBS),
    "langsmith": PlatformEntry(_OBS),
    "langfuse": PlatformEntry(_OBS),
    "arize": PlatformEntry(_OBS),
    "phoenix": PlatformEntry(_OBS),
    "opik": PlatformEntry(_OBS),
    "helicone": PlatformEntry(_OBS),
    "freeplay": PlatformEntry(_OBS),
    "tracia": PlatformEntry(_OBS),
    "llmetry": PlatformEntry(_OBS),
    "galileo": PlatformEntry(_OBS),
    "honeyhive": PlatformEntry(_OBS),
    "promptlayer": PlatformEntry(_OBS),
    "humanloop": PlatformEntry(_OBS),
    "braintrust": PlatformEntry(_OBS),
    "whylabs": PlatformEntry(_OBS),
    # Observability + model registry
    "wandb": PlatformEntry(_OBS, frozenset({_REG})),
    "weights_biases": PlatformEntry(_OBS, frozenset({_REG})),
    "mlflow": PlatformEntry(_OBS, frozenset({_REG})),
    "neptune": PlatformEntry(_OBS, frozenset({_REG})),
    "comet": PlatformEntry(_OBS, frozenset({_REG})),
    "deepchecks": PlatformEntry(_OBS),
    # Guardrails
    "nemoguardrails": PlatformEntry(_GRD),
    "guardrails": PlatformEntry(_GRD),
    "llm_guard": PlatformEntry(_GRD),
    "llm_guardrails": PlatformEntry(_GRD),
    "lakera_guard": PlatformEntry(_GRD),
    "rebuff": PlatformEntry(_GRD),
    "guardrails_ai": PlatformEntry(_GRD),
    # Cisco AI Defense runtime protection (agentsec).
    "aidefense": PlatformEntry(_GRD),
    "agentsec": PlatformEntry(_GRD),
}

OBSERVABILITY_PLATFORM_TOKENS: frozenset[str] = frozenset(
    k for k, entry in _IMPORT_MODULE_TYPE_MAP.items() if _OBS in entry.all_types
)


_NON_AI_CLASS_SUFFIXES: tuple[str, ...] = (
    "Response",
    "Request",
    "Schema",
    "Config",
    "Spec",
    "Params",
    "DTO",
    "DML",
    "DDL",
    "Enum",
    "Type",
    "Base",
    "Abstract",
    "Interface",
    "Mixin",
    "Factory",
    "Builder",
    "Validator",
    "Serializer",
    "Deserializer",
    "Mapper",
    "Converter",
    "Exception",
    "Error",
    "Test",
    "Mock",
    "Stub",
    "Fake",
    "Code",
    "Status",
    "Flag",
)

_AMBIGUOUS_NAME_TYPES: frozenset[AIComponentType] = frozenset(
    {
        AIComponentType.EMBEDDING,
        AIComponentType.MEMORY,
        AIComponentType.TOOL,
        AIComponentType.AGENT,
        AIComponentType.VECTOR_STORE,
        AIComponentType.RETRIEVER,
        AIComponentType.PROMPT,
        AIComponentType.GUARDRAIL,
    }
)

_EMBEDDING_IMPORT_SIGNALS: frozenset[str] = frozenset(
    {
        "embedding",
        "embedder",
        "embedders",
        "openai",
        "langchain",
        "haystack",
        "llama_index",
        "sentence_transformers",
        "transformers",
        "chromadb",
    }
)

_EMBEDDING_CALL_SIGNALS: frozenset[str] = frozenset(
    {
        "model=",
        "model_name=",
        "deployment=",
        "deployment_name=",
        "api_key=",
        "client=",
        "embedding_function=",
    }
)


def _is_data_class_name(name: str) -> bool:
    """Return True when the class name looks like a data/schema class."""
    return any(name.endswith(s) for s in _NON_AI_CLASS_SUFFIXES)


def _is_class_like_name(name: str) -> bool:
    return bool(name) and name[0].isupper() and not name.isupper()


def _has_embedding_type_name(name: str) -> bool:
    lower = name.lower()
    return (
        lower.endswith("embedding")
        or lower.endswith("embeddings")
        or lower.endswith("embedder")
    )


def _has_embedding_import_evidence(text: str) -> bool:
    lower = text.lower()
    return any(signal in lower for signal in _EMBEDDING_IMPORT_SIGNALS)


def _has_embedding_call_evidence(line: str, source: str) -> bool:
    lower_line = line.lower()
    return any(
        signal in lower_line for signal in _EMBEDDING_CALL_SIGNALS
    ) or _has_embedding_import_evidence(source)


def _infer_type_from_name(name: str) -> tuple[AIComponentType, bool]:
    """Infer the component type from a class/import name using pattern matching.

    Returns ``(type, needs_agentic)`` — ambiguous name-only matches are
    marked ``needs_agentic=True`` so the LLM can confirm or reject them.
    """
    if _is_data_class_name(name):
        return AIComponentType.MODEL, False
    for pattern, comp_type in _CLASS_NAME_TYPE_MAP:
        if pattern.search(name):
            ambiguous = comp_type in _AMBIGUOUS_NAME_TYPES
            return comp_type, ambiguous
    return AIComponentType.MODEL, False


_GENERIC_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "ABC",
        "Any",
        "Dict",
        "List",
        "Optional",
        "Tuple",
        "Set",
        "Type",
        "Union",
        "Callable",
        "Generator",
        "AsyncGenerator",
        "Iterator",
        "Sequence",
        "Mapping",
        "MutableMapping",
        "Annotated",
        "ClassVar",
        "Protocol",
        "BaseModel",
        "Field",
        "PrivateAttr",
        "ConfigDict",
        "Awaitable",
        "AsyncIterator",
        "Path",
        "Formatter",
        "ErrorCode",
    }
)

_NON_AI_PACKAGES: frozenset[str] = frozenset(
    {
        "more_itertools",
        "itertools",
        "functools",
        "collections",
        "dataclasses",
        "typing_extensions",
        "pydantic",
        "attrs",
        "pytest",
        "unittest",
        "mock",
        "faker",
    }
)

_DATA_CLASS_SUFFIXES: tuple[str, ...] = (
    "Action",
    "Step",
    "Finish",
    "Message",
    "Output",
    "Input",
    "Schema",
    "Config",
    "Event",
    "Error",
    "Exception",
    "Result",
    "Response",
    "Request",
    "Callback",
    "Handler",
    "Parser",
    "Serializer",
    "Kwargs",
    "Meta",
    "State",
    "Log",
    "Mixin",
    "Interface",
    "Value",
    "Wrapper",
    "Item",
    "Record",
    "Encoder",
    "Decoder",
    "Triple",
)


def _extract_leaf_class(kb_id: str) -> Optional[str]:
    """Extract a usable class name from a KB id, or None if unsuitable."""
    if "." not in kb_id:
        return None
    parts = kb_id.split(".")
    leaf = parts[-1]
    if not leaf or not leaf[0].isupper() or leaf.isupper() or len(leaf) <= 2:
        return None
    if leaf in _GENERIC_CLASS_NAMES or leaf in _EXCLUDED_CLASS_NAMES:
        return None
    if any(leaf.endswith(s) for s in _DATA_CLASS_SUFFIXES):
        return None
    if len(parts) >= 2 and parts[-2] and parts[-2][0].isupper():
        return None
    return leaf


def _build_kb_patterns(
    db: CatalogDB,
) -> dict[AIComponentType, frozenset[str]]:
    """Query the KB for class names under canonical path segments.

    Returns a mapping from component type to the set of class names
    that can be matched in ``call`` observations.  Falls back to
    static lists when the KB yields nothing for a category.
    """
    _PATH_CONCEPT_MAP: list[
        tuple[str, tuple[str, ...], AIComponentType, frozenset[str]]
    ] = [
        (".tools.", ("tool",), AIComponentType.TOOL, _STATIC_TOOL_PATTERNS),
        (
            ".memory.",
            ("memory", "datastore"),
            AIComponentType.MEMORY,
            _STATIC_MEMORY_PATTERNS,
        ),
        (".prompts.", ("prompt",), AIComponentType.PROMPT, _STATIC_PROMPT_PATTERNS),
        (".agents.", ("agent",), AIComponentType.AGENT, _AGENT_CREATION_PATTERNS),
    ]

    result: dict[AIComponentType, frozenset[str]] = {}
    for path_seg, concepts, comp_type, static_fallback in _PATH_CONCEPT_MAP:
        try:
            ids = db.find_ids_by_path_and_concept(path_seg, concepts)
        except Exception:  # noqa: BLE001
            _LOGGER.debug("KB pattern query failed for %s", path_seg, exc_info=True)
            ids = []

        try:
            custom_ids = db.find_ids_in_custom_by_path_and_concept(path_seg, concepts)
        except Exception:  # noqa: BLE001
            _LOGGER.debug(
                "KB pattern custom query failed for %s", path_seg, exc_info=True
            )
            custom_ids = []

        kb_classes: set[str] = set()
        for kid in (*ids, *custom_ids):
            cls = _extract_leaf_class(kid)
            if cls:
                kb_classes.add(cls)

        combined = kb_classes | set(static_fallback)
        result[comp_type] = frozenset(combined)
        _LOGGER.debug(
            "KB patterns for %s: %d from KB + %d custom + %d static = %d total",
            comp_type.value,
            len(kb_classes),
            len(custom_ids),
            len(static_fallback),
            len(combined),
        )

    return result


def _build_kb_framework_prefixes(db: CatalogDB) -> frozenset[str]:
    """Extract unique framework root prefixes from the KB.

    Always unions the DuckDB prefixes with :data:`_AGENT_FRAMEWORK_PREFIXES`
    and any frameworks registered via custom/built-in catalog supplements so
    we don't miss modern SDKs (e.g. Strands) that post-date the DuckDB
    snapshot.
    """
    roots: set[str] = set(_AGENT_FRAMEWORK_PREFIXES)
    try:
        for fw in db.distinct_frameworks():
            if fw:
                roots.add(fw.split("_")[0].split("-")[0])
    except Exception:  # noqa: BLE001
        _LOGGER.debug("KB framework query failed", exc_info=True)
    return frozenset(roots)


def _is_known_call(
    qualified_name: str,
    imports: list,
    patterns: frozenset[str],
    framework_prefixes: frozenset[str],
) -> bool:
    """Return True if *qualified_name* matches a known creation pattern
    from a recognized framework."""
    short = (
        qualified_name.rsplit(".", 1)[-1] if "." in qualified_name else qualified_name
    )
    if short not in patterns:
        return False
    prefix = qualified_name.split(".")[0].split("_")[0]
    if prefix in framework_prefixes:
        return True
    for imp in _import_strings(imports):
        parts = imp.split()
        mod = parts[1] if len(parts) > 1 else ""
        mod_prefix = mod.split(".")[0].split("_")[0]
        if mod_prefix in framework_prefixes and short in imp:
            return True
    return False


def _resolve_kb_path(context: ScanContext) -> Optional[Path]:
    """Locate the KB DuckDB file.  Returns ``None`` when unavailable."""
    try:
        require_supported_manifest_schema()
    except UnsupportedDatabaseSchemaError as exc:
        _LOGGER.warning("%s", exc)
        return None

    if context.kb_path:
        p = Path(context.kb_path)
        if p.is_file():
            return p

    try:
        return ensure_local_database()
    except UnsupportedDatabaseSchemaError as exc:
        _LOGGER.warning("%s", exc)
        return None
    except (DatabaseLoadError, Exception):  # noqa: BLE001
        pass

    catalogs_dir = Path.home() / ".aibom" / "catalogs"
    if catalogs_dir.is_dir():
        dbs = sorted(catalogs_dir.glob("*.duckdb"), reverse=True)
        if dbs:
            return dbs[0]

    return None


def _extract_frameworks_from_imports(
    imports: list[str] | list[tuple[int, str]],
) -> set[str]:
    """Derive top-level package names from import statements.

    e.g. ``"from langchain_openai import ChatOpenAI"`` → ``{"langchain_openai"}``
    """
    frameworks: set[str] = set()
    for entry in imports:
        stmt = entry[1] if isinstance(entry, tuple) else entry
        if stmt.startswith("from "):
            module = stmt.split()[1] if len(stmt.split()) > 1 else ""
            top = module.split(".")[0]
            if top:
                frameworks.add(top)
        elif stmt.startswith("import "):
            module = stmt.split()[1] if len(stmt.split()) > 1 else ""
            top = module.split(".")[0]
            if top:
                frameworks.add(top)
    return frameworks


class KBEnrichmentScanner(BaseScanner):
    """Detect AI framework usage by matching LibCST observations against the KB.

    Only emits for concepts in :data:`ALLOWED_CONCEPTS`, suppressing the noise
    that made the legacy categorizer output hard to use.
    """

    name = "kb_enrichment"

    def supports(self, context: ScanContext) -> bool:
        return _resolve_kb_path(context) is not None

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        kb_path = _resolve_kb_path(context)
        if not kb_path:
            return [], []

        _LOGGER.info("KB enrichment: using %s", kb_path)
        components: list[AIComponent] = []

        with CatalogDB(kb_path) as db:
            try:
                from ..builtin_catalog import BUILTIN_CATALOG_ENTRIES

                db.add_custom_entries(BUILTIN_CATALOG_ENTRIES)
                _LOGGER.debug(
                    "KB enrichment: loaded %d built-in catalog supplement(s)",
                    len(BUILTIN_CATALOG_ENTRIES),
                )
            except Exception:  # noqa: BLE001
                _LOGGER.debug(
                    "KB enrichment: built-in catalog load failed", exc_info=True
                )

            custom_cfg = context.config.get("custom_catalog")
            if custom_cfg is not None:
                try:
                    from ..custom_catalog import CustomCatalogConfig

                    if (
                        isinstance(custom_cfg, CustomCatalogConfig)
                        and not custom_cfg.is_empty
                    ):
                        db.add_custom_entries(
                            [c.to_catalog_dict() for c in custom_cfg.components]
                        )
                        if custom_cfg.excludes:
                            db.add_excludes(custom_cfg.excludes)
                except Exception:  # noqa: BLE001
                    _LOGGER.debug(
                        "KB enrichment: custom catalog load failed", exc_info=True
                    )

            kb_patterns = _build_kb_patterns(db)
            kb_fw_prefixes = _build_kb_framework_prefixes(db)
            kb_fw_names = _build_kb_framework_names(db)

            if context.ai_package_set:
                combined = set(kb_fw_names)
                for pkg in context.ai_package_set:
                    combined.add(pkg)
                    combined.add(pkg.replace("-", "_"))
                filter_set = frozenset(combined)
            else:
                filter_set = kb_fw_names

            all_py_files = _find_python_files(context)
            tier1_files: list[tuple[Path, str]] = []
            suggestive_files: list[tuple[Path, str]] = []

            for py_file in all_py_files:
                try:
                    source = read_python_source(py_file)
                except Exception:  # noqa: BLE001
                    continue
                if _file_has_kb_framework_import(source, filter_set):
                    tier1_files.append((py_file, source))
                elif _has_suggestive_signal(py_file, source):
                    suggestive_files.append((py_file, source))

            _LOGGER.debug(
                "KB enrichment: %d Tier 1 (framework import), "
                "%d suggestive (wrapper), %d skipped of %d total",
                len(tier1_files),
                len(suggestive_files),
                len(all_py_files) - len(tier1_files) - len(suggestive_files),
                len(all_py_files),
            )

            parsed_results: list[CodeAnalysisResult] = []
            all_symbols: set[str] = set()
            all_files = tier1_files + suggestive_files
            for py_file, source in all_files:
                try:
                    result = parse_source_code(str(py_file), source)
                    parsed_results.append(result)
                    _collect_symbols(result, all_symbols)
                except Exception:  # noqa: BLE001
                    _LOGGER.debug(
                        "KB enrichment: failed to parse %s", py_file, exc_info=True
                    )

            kb_entries = _batch_kb_lookup(db, all_symbols) if all_symbols else {}

            for result in parsed_results:
                components.extend(
                    _process_file_with_cache(
                        result,
                        kb_entries,
                        kb_patterns,
                        kb_fw_prefixes,
                    )
                )

            for result in parsed_results:
                components.extend(_detect_tool_schemas(result))
                components.extend(_detect_prompt_kwargs(result))
                components.extend(_detect_model_kwargs(result))
                components.extend(_detect_import_based_assets(result))
                components.extend(_detect_guardrail_calls(result))

            for py_file, source in suggestive_files:
                components.extend(_emit_suggestive_candidates(py_file, source))

            components.extend(_detect_cache_ai_co_occurrence(tier1_files))

        components = [
            c
            for c in components
            if c.name.lower().replace("-", "_") not in _NON_AI_PACKAGES
        ]

        return components, []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_kb_framework_names(db: CatalogDB) -> frozenset[str]:
    """Query the KB for the full set of framework package names.

    Returns both the raw framework values (``langchain_community``) **and**
    their root prefixes (``langchain``), giving us a complete allowlist
    derived from the KB and from any custom/built-in catalog supplements.
    """
    names: set[str] = set()
    try:
        for fw in db.distinct_frameworks():
            if fw:
                names.add(fw)
                names.add(fw.split("_")[0].split("-")[0])
    except Exception:  # noqa: BLE001
        _LOGGER.debug("KB framework name query failed", exc_info=True)
    return frozenset(names)


_AMBIGUOUS_TOP_LEVEL = frozenset(
    {
        "google",
        "aws",
        "azure",
        "microsoft",
        "com",
        "org",
        "io",
        "ai",
    }
)


def _import_matches_framework(dotted_path: str, kb_frameworks: frozenset[str]) -> bool:
    """Check if a dotted import path matches any known framework.

    Strategy:
    - Single-segment (``torch``): direct check + root-prefix check.
    - Multi-dot (``google.cloud.aiplatform``): build progressive underscore
      joins (``google_cloud``, ``google_cloud_aiplatform``) and check each.
      The bare first segment is only accepted if it is NOT in
      ``_AMBIGUOUS_TOP_LEVEL`` — so ``langchain.agents`` matches (first
      segment ``langchain`` is specific) but ``google.protobuf`` does not
      (``google`` is ambiguous without deeper context).
    """
    segments = dotted_path.split(".")

    if len(segments) == 1:
        pkg = segments[0]
        if pkg in kb_frameworks:
            return True
        root = pkg.split("_")[0].split("-")[0]
        return root in kb_frameworks

    first = segments[0]
    if first not in _AMBIGUOUS_TOP_LEVEL and first in kb_frameworks:
        return True

    accumulated = ""
    for i, seg in enumerate(segments):
        accumulated = f"{accumulated}_{seg}" if accumulated else seg
        if i == 0:
            continue
        if accumulated in kb_frameworks:
            return True

    return False


def _file_has_kb_framework_import(
    source: str,
    kb_frameworks: frozenset[str],
) -> bool:
    """Return True if *source* contains an ``import`` or ``from`` statement
    referencing any framework known to the KB.

    This is a cheap structural check — it only looks at lines that start with
    ``import `` or ``from `` (after stripping whitespace) and checks the full
    dotted import path progressively against the framework set.
    """
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("from "):
            parts = stripped.split(None, 2)
            if len(parts) >= 2:
                if _import_matches_framework(parts[1], kb_frameworks):
                    return True
        elif stripped.startswith("import "):
            parts = stripped.split(None, 2)
            if len(parts) >= 2:
                for mod in parts[1].split(","):
                    dotted = mod.strip().split(" ")[0]
                    if _import_matches_framework(dotted, kb_frameworks):
                        return True
    return False


def _has_suggestive_signal(py_file: Path, source: str) -> bool:
    """Return True if the file has AI-suggestive directory path or import path.

    This is the bypass for wrapper libraries that don't directly import known
    frameworks but live in directories or import modules whose names suggest
    AI involvement (e.g., ``models/llm/openai.py``).
    """
    parts = py_file.parts
    for part in parts:
        if part.lower() in _AI_SUGGESTIVE_DIR_SEGMENTS:
            return True

    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("from ", "import ")):
            lower = stripped.lower()
            for seg in _AI_SUGGESTIVE_IMPORT_SEGMENTS:
                if seg in lower:
                    return True
    return False


def _emit_suggestive_candidates(
    py_file: Path,
    source: str,
) -> list[AIComponent]:
    """Emit low-confidence agentic candidates for AI-indicative class names.

    Only fires for files that passed the suggestive-signal check but failed
    the framework import check.  The agent will trace the wrapper chain.
    """
    candidates: list[AIComponent] = []
    seen_lines: set[int] = set()
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assign_match = re.match(
            r"(\w+)\s*=\s*([A-Z]\w+)\s*\(",
            stripped,
        )
        if not assign_match:
            continue
        _target, class_name = assign_match.group(1), assign_match.group(2)
        if class_name in _GENERIC_CLASS_NAMES:
            continue
        if _is_data_class_name(class_name):
            continue
        if not _AI_INDICATIVE_CLASS_RE.search(class_name):
            continue
        if i in seen_lines:
            continue
        seen_lines.add(i)
        inferred_type, _name_ambiguous = _infer_type_from_name(class_name)
        if inferred_type == AIComponentType.MODEL:
            continue
        if inferred_type == AIComponentType.EMBEDDING and not (
            _has_embedding_type_name(class_name)
            and _has_embedding_call_evidence(stripped, source)
        ):
            continue
        candidates.append(
            AIComponent(
                name=class_name,
                component_type=inferred_type,
                file_path=str(py_file),
                line_number=i,
                framework="",
                detection_source=DetectionSource.KB_ENRICHMENT,
                heuristic_confidence=0.2,
                needs_agentic=True,
                agentic_hint=(
                    f"Class '{class_name}' in path "
                    f"'{py_file.parent.name}/'. A class name alone is "
                    f"NOT proof of an AI component. REMOVE unless "
                    f"code_context proves this is a genuine AI "
                    f"{inferred_type.value}. Classes in AI-adjacent "
                    f"directories are often ordinary handlers, "
                    f"utilities, or DTOs."
                ),
                metadata={
                    "suggestive_signal": True,
                    "parent_dir": py_file.parent.name,
                },
            )
        )
    return candidates


def _detect_cache_ai_co_occurrence(
    tier1_files: list[tuple[Path, str]],
) -> list[AIComponent]:
    """Emit agentic candidates when a file imports both an AI framework and a caching library.

    Redis/Memcached alone is not an AI asset, but co-occurring with an AI
    framework import suggests LLM response caching or conversation memory.
    """
    from .import_context import has_cache_imports

    candidates: list[AIComponent] = []
    seen: set[str] = set()
    for py_file, source in tier1_files:
        if not has_cache_imports(source):
            continue
        file_key = str(py_file)
        if file_key in seen:
            continue
        seen.add(file_key)
        cache_lib = "unknown"
        for lib in (
            "redis",
            "memcache",
            "pymemcache",
            "cachetools",
            "diskcache",
            "aiocache",
        ):
            if re.search(rf"(?:^|\n)\s*(?:from|import)\s+{lib}\b", source):
                cache_lib = lib
                break
        candidates.append(
            AIComponent(
                name=f"{cache_lib} (AI cache co-occurrence)",
                component_type=AIComponentType.MEMORY,
                file_path=file_key,
                line_number=0,
                framework=cache_lib,
                detection_source=DetectionSource.CODE_ANALYSIS,
                heuristic_confidence=0.4,
                needs_agentic=True,
                agentic_hint=(
                    f"File imports both an AI framework and '{cache_lib}'. "
                    f"REMOVE unless code_context shows the cache stores "
                    f"LLM responses or conversation history — generic app "
                    f"caching is not an AI component."
                ),
                metadata={"cache_ai_co_occurrence": True, "cache_library": cache_lib},
            )
        )
    return candidates


def _detect_import_based_assets(
    result: "CodeAnalysisResult",
) -> list[AIComponent]:
    """Detect AI assets from import statements that match known modules or name patterns.

    Handles: embedder/embedding imports, MCP client imports, guardrail framework
    imports, and observability framework imports.
    """
    from ..structures import CodeAnalysisResult as _CAR  # noqa: F811

    if not isinstance(result, _CAR):
        return []

    candidates: list[AIComponent] = []
    seen: set[tuple[str, int]] = set()

    for imp_entry in result.imports:
        if isinstance(imp_entry, tuple):
            stored_line, imp_line = imp_entry
        else:
            stored_line, imp_line = 0, imp_entry
        imp_lower = imp_line.lower()

        comp_type: AIComponentType | None = None
        extra_types: frozenset[AIComponentType] = frozenset()
        matched_name = ""
        name_ambiguous = False

        for mod_key, entry in _IMPORT_MODULE_TYPE_MAP.items():
            if mod_key in imp_lower:
                comp_type = entry.primary
                extra_types = entry.additional
                matched_name = mod_key
                break

        if comp_type is None:
            tokens = imp_line.split()
            try:
                import_idx = tokens.index("import")
                symbols = tokens[import_idx + 1 :]
            except ValueError:
                symbols = tokens
            for part in symbols:
                cleaned = part.strip(",").strip("(").strip(")")
                if not cleaned or cleaned in ("from", "import", "as"):
                    continue
                inferred, name_ambiguous = _infer_type_from_name(cleaned)
                if inferred == AIComponentType.EMBEDDING and not (
                    _is_class_like_name(cleaned)
                    and _has_embedding_type_name(cleaned)
                    and _has_embedding_import_evidence(imp_lower)
                ):
                    continue
                if inferred != AIComponentType.MODEL:
                    comp_type = inferred
                    matched_name = cleaned
                    break

        if comp_type is None:
            continue

        line_no = stored_line

        dedup = (result.file_path, line_no)
        if dedup in seen:
            continue
        seen.add(dedup)

        for ct in (comp_type, *extra_types):
            is_additional = ct != comp_type
            if ct == AIComponentType.AGENT:
                hint = (
                    f"Import-inferred as 'agent' from "
                    f"'{imp_line.strip()}'. An agent requires: "
                    f"(1) LLM-driven control flow, (2) tool/action "
                    f"execution, (3) iterative loop. Use "
                    f"`read_file_snippet` on the source module to "
                    f"check for these patterns. If the class only "
                    f"wraps a single LLM call with no loop or tool "
                    f"dispatch, REMOVE it. Reclassify to `tool` only "
                    f"if it is registered as a callable tool for "
                    f"another agent (`@tool`, `tools=[...]`)."
                )
            else:
                hint = (
                    f"Import-inferred as '{ct.value}' from "
                    f"'{imp_line.strip()}'. Import alone is weak "
                    f"evidence — REMOVE unless surrounding code proves "
                    f"this is a genuine AI {ct.value}."
                )
            candidates.append(
                AIComponent(
                    name=matched_name,
                    component_type=ct,
                    file_path=result.file_path,
                    line_number=line_no,
                    framework="",
                    detection_source=DetectionSource.CODE_ANALYSIS,
                    heuristic_confidence=(
                        0.35 if name_ambiguous else 0.40 if is_additional else 0.55
                    ),
                    needs_agentic=True,
                    agentic_hint=hint,
                    metadata={"import_statement": imp_line.strip()},
                )
            )

    return candidates


# Guardrail protection call sites. These are unambiguous: a call to one of
# these qualified names installs runtime safety/guardrail protection over LLM
# or MCP traffic, so the call site itself is a guardrail component. Keyed on
# the ``module.method`` tail so e.g. ``agentsec.protect(...)`` matches whether
# imported as ``from aidefense.runtime import agentsec`` or ``import agentsec``.
_GUARDRAIL_CALL_PATTERNS: frozenset[str] = frozenset(
    {
        "agentsec.protect",
        "RailsConfig.from_path",  # NeMo Guardrails
        "LLMRails",  # NeMo Guardrails
        "Guard.for_string",  # Guardrails AI
        "Guard.for_pydantic",  # Guardrails AI
    }
)


def _detect_guardrail_calls(result: "CodeAnalysisResult") -> list[AIComponent]:
    """Detect guardrail components from protection call sites.

    The clearest guardrail signal is a call that installs runtime protection,
    e.g. Cisco AI Defense's ``agentsec.protect(...)`` or a NeMo Guardrails /
    Guardrails AI entry point. Import-based detection alone misses these when
    the protection is wired through a call rather than a class import, and the
    call site carries the precise file/line evidence.
    """
    from ..structures import CodeAnalysisResult as _CAR  # noqa: F811

    if not isinstance(result, _CAR):
        return []

    components: list[AIComponent] = []
    seen: set[tuple[str, int]] = set()

    all_calls = [(c.qualified_name or "", c.line_number) for c in result.calls]
    all_calls += [
        (a.call.qualified_name or "", a.line_number) for a in result.assignments
    ]

    for qn, line in all_calls:
        if not qn:
            continue
        # Match either the full ``module.method`` tail or a bare class name.
        tail2 = ".".join(qn.split(".")[-2:]) if "." in qn else qn
        short = qn.rsplit(".", 1)[-1] if "." in qn else qn
        if tail2 not in _GUARDRAIL_CALL_PATTERNS and short not in (
            _GUARDRAIL_CALL_PATTERNS
        ):
            continue

        key = (result.file_path, line)
        if key in seen:
            continue
        seen.add(key)

        framework = qn.split(".")[0] if "." in qn else qn
        components.append(
            AIComponent(
                name=tail2 if tail2 in _GUARDRAIL_CALL_PATTERNS else short,
                component_type=AIComponentType.GUARDRAIL,
                file_path=result.file_path,
                line_number=line,
                framework=framework,
                detection_source=DetectionSource.CODE_ANALYSIS,
                metadata={"call_pattern": qn, "guardrail_protection": True},
            )
        )

    return components


_TOOL_KWARG_NAMES: frozenset[str] = frozenset(
    {
        "tools",
        "functions",
        "tool_choice",
    }
)

_TOOL_DECORATOR_NAMES: frozenset[str] = frozenset(
    {
        "tool",
        "register_tool",
    }
)

_TOOL_DECORATOR_FRAMEWORKS: frozenset[str] = frozenset(
    {
        "langchain",
        "langchain_core",
        "crewai",
        "smolagents",
        "pydantic_ai",
        "autogen",
        "deepagents",
        "llama_index",
        "agno",
        "phidata",
        "strands",
    }
)

_TOOL_CONVERSION_CALLS: frozenset[str] = frozenset(
    {
        "function_to_schema",
        "convert_to_openai_tool",
        "convert_to_openai_function",
        "format_tool_to_openai_function",
        "tool_to_function_definition",
    }
)

_AI_CLIENT_CALLS: frozenset[str] = frozenset(
    {
        "create",
        "chat",
        "completions",
        "invoke",
        "ainvoke",
        "bind_tools",
        "with_structured_output",
    }
)

_PROMPT_KWARG_NAMES: frozenset[str] = frozenset(
    {
        "system_prompt",
        "system_message",
        "system",
        "instructions",
        "prompt",
        "template",
        "messages",
        "few_shot_examples",
        "examples",
    }
)


def _detect_tool_schemas(result: "CodeAnalysisResult") -> list[AIComponent]:
    """Detect tools via structural analysis of calls and decorators.

    Three paths:
    1. ``tools=``/``functions=`` kwargs on confirmed AI client calls
    2. ``@tool`` decorators from known AI frameworks
    3. Tool conversion function calls (``function_to_schema()``, etc.)
    """
    from ..structures import CodeAnalysisResult as _CAR  # noqa: F811

    components: list[AIComponent] = []
    seen: set[tuple[str, int]] = set()
    imports = _import_strings(getattr(result, "imports", []) or [])
    import_text = "\n".join(imports)

    for call_obs in result.calls:
        qn = call_obs.qualified_name or ""
        short = qn.rsplit(".", 1)[-1] if "." in qn else qn

        if short in _AI_CLIENT_CALLS or short in {"create_deep_agent"}:
            for kwarg_name in _TOOL_KWARG_NAMES:
                val = call_obs.arguments.get(kwarg_name)
                if val is None:
                    continue
                tool_names = _extract_tool_names_from_arg(val)
                for tname in tool_names:
                    key = (result.file_path, call_obs.line_number)
                    if key in seen:
                        continue
                    seen.add(key)
                    components.append(
                        AIComponent(
                            name=tname,
                            component_type=AIComponentType.TOOL,
                            file_path=result.file_path,
                            line_number=call_obs.line_number,
                            framework=qn.split(".")[0] if "." in qn else "",
                            detection_source=DetectionSource.CODE_ANALYSIS,
                            metadata={"tool_kwarg": kwarg_name, "enclosing_call": qn},
                        )
                    )

        if short in _TOOL_CONVERSION_CALLS:
            arg0 = call_obs.arguments.get("_pos_0", "")
            tname = _variable_name(arg0) or short
            key = (result.file_path, call_obs.line_number)
            if key not in seen:
                seen.add(key)
                components.append(
                    AIComponent(
                        name=tname,
                        component_type=AIComponentType.TOOL,
                        file_path=result.file_path,
                        line_number=call_obs.line_number,
                        framework="",
                        detection_source=DetectionSource.CODE_ANALYSIS,
                        metadata={"tool_conversion": short},
                    )
                )

    for assignment in result.assignments:
        qn = assignment.call.qualified_name or ""
        short = qn.rsplit(".", 1)[-1] if "." in qn else qn
        if short in _TOOL_CONVERSION_CALLS:
            target = assignment.target_qualified_name or short
            key = (result.file_path, assignment.line_number)
            if key not in seen:
                seen.add(key)
                components.append(
                    AIComponent(
                        name=target.rsplit(".", 1)[-1] if "." in target else target,
                        component_type=AIComponentType.TOOL,
                        file_path=result.file_path,
                        line_number=assignment.line_number,
                        framework="",
                        detection_source=DetectionSource.CODE_ANALYSIS,
                        metadata={"tool_conversion": short, "assigned_to": target},
                    )
                )

    for dec in result.decorators:
        dname = dec.decorator_qualified_name or ""
        short_dec = dname.rsplit(".", 1)[-1] if "." in dname else dname
        if short_dec not in _TOOL_DECORATOR_NAMES:
            continue
        fw_confirmed = False
        prefix = dname.split(".")[0].split("_")[0] if "." in dname else ""
        if prefix in _TOOL_DECORATOR_FRAMEWORKS:
            fw_confirmed = True
        else:
            for imp in imports:
                for fw in _TOOL_DECORATOR_FRAMEWORKS:
                    if fw in imp and "tool" in imp:
                        fw_confirmed = True
                        break
                if fw_confirmed:
                    break
        if not fw_confirmed:
            continue
        key = (result.file_path, dec.line_number)
        if key not in seen:
            seen.add(key)
            components.append(
                AIComponent(
                    name=dec.decorated_function_name,
                    component_type=AIComponentType.TOOL,
                    file_path=result.file_path,
                    line_number=dec.line_number,
                    framework=prefix or "",
                    detection_source=DetectionSource.CODE_ANALYSIS,
                    metadata={"tool_decorator": dname},
                )
            )

    return components


def _extract_tool_names_from_arg(val: Any) -> list[str]:
    """Extract tool names from a ``tools=`` kwarg value."""
    names: list[str] = []
    if isinstance(val, list):
        for item in val:
            n = _variable_name(item)
            if n:
                names.append(n)
            elif isinstance(item, dict) and "name" in item:
                names.append(str(item["name"]))
            elif isinstance(item, dict):
                fn = item.get("function", {})
                if isinstance(fn, dict) and "name" in fn:
                    names.append(str(fn["name"]))
    elif isinstance(val, str):
        n = _variable_name(val)
        if n:
            names.append(n)
    return names


def _variable_name(val: Any) -> str | None:
    """Extract a variable/attribute name from CST argument sentinel values."""
    if isinstance(val, str):
        if val.startswith("VARIABLE:"):
            return val.removeprefix("VARIABLE:")
        if val.startswith("ATTRIBUTE:"):
            return val.removeprefix("ATTRIBUTE:").rsplit(".", 1)[-1]
    return None


_MODEL_KWARG_NAMES: frozenset[str] = frozenset(
    {
        "model",
        "model_name",
        "model_id",
        "deployment_name",
        "engine",
    }
)

_ENDPOINT_KWARG_NAMES: frozenset[str] = frozenset(
    {
        "base_url",
        "azure_endpoint",
        "api_base",
        "endpoint_url",
    }
)

_KNOWN_AI_CLIENT_CLASSES: frozenset[str] = frozenset(
    {
        "ChatOpenAI",
        "ChatAnthropic",
        "ChatGoogleGenerativeAI",
        "AzureChatOpenAI",
        "OpenAI",
        "Anthropic",
        "GenerativeModel",
        "AnthropicBedrock",
        "ChatBedrock",
        "BedrockChat",
        "ChatVertexAI",
        "ChatCohere",
        "ChatMistralAI",
        "ChatOllama",
        "ChatLiteLLM",
        "ChatFireworks",
        "ChatGroq",
        "ChatTogether",
        "VLLMOpenAI",
        "AzureOpenAI",
        # Strands built-in model provider classes (``strands.models.*``). Each
        # accepts a ``model_id=`` kwarg at construction time and is the primary
        # way to wire a Strands ``Agent`` to a provider. Covering these here
        # lets ``_detect_model_kwargs`` surface both the concrete model ID and,
        # when the constructor is bare, a needs-agentic bare-client component.
        "BedrockModel",
        "AnthropicModel",
        "OpenAIModel",
        "OllamaModel",
        "LiteLLMModel",
        "GeminiModel",
        "MistralModel",
        "SageMakerModel",
        "LlamaAPIModel",
        "WriterModel",
        "LlamaCppModel",
        "CohereModel",
    }
)

_BARE_CLIENT_HINTS: dict[str, str] = {
    "AnthropicBedrock": "Resolve model ID from .messages.create(model=...) calls. Remove if no model ID found.",
    "BedrockChat": "Resolve model ID from model_id= kwarg or nearby invoke_model calls. Remove if no model ID found.",
    "ChatBedrock": "Resolve model ID from model_id= kwarg or nearby invoke_model calls. Remove if no model ID found.",
    # Strands model providers: the model ID is normally passed to the
    # constructor or supplied via environment variable. If neither surfaces,
    # the bare client is not a reliable model evidence and should be pruned.
    "BedrockModel": "Resolve model ID from model_id= kwarg, BEDROCK_MODEL_ID env, or nearby Agent(model=...) wiring. Remove if no model ID found.",
    "AnthropicModel": "Resolve model ID from model_id= kwarg or ANTHROPIC_MODEL_ID env. Remove if no model ID found.",
    "OpenAIModel": "Resolve model ID from model_id= kwarg or OPENAI_MODEL_ID env. Remove if no model ID found.",
    "OllamaModel": "Resolve model ID from model_id= kwarg or OLLAMA_MODEL env. Remove if no model ID found.",
    "LiteLLMModel": "Resolve model ID from model_id= kwarg. Remove if no model ID found.",
    "GeminiModel": "Resolve model ID from model_id= kwarg or GEMINI_MODEL_ID env. Remove if no model ID found.",
    "MistralModel": "Resolve model ID from model_id= kwarg or MISTRAL_MODEL_ID env. Remove if no model ID found.",
    "SageMakerModel": "Resolve endpoint_name= and model_id= kwargs. Remove if neither found.",
    "LlamaAPIModel": "Resolve model_id= kwarg. Remove if no model ID found.",
    "WriterModel": "Resolve model_id= kwarg or WRITER_MODEL_ID env. Remove if no model ID found.",
    "LlamaCppModel": "Resolve model_path= or model_id= kwarg. Remove if neither found.",
    "CohereModel": "Resolve model_id= kwarg or COHERE_MODEL_ID env. Remove if no model ID found.",
}


def _is_confirmed_ai_prompt_call(qn: str) -> bool:
    """Return True only for prompt kwargs on confirmed AI call chains.

    Generic helpers frequently accept ``prompt=`` or ``messages=`` kwargs, so
    we require the enclosing call to resolve to a known AI constructor or an AI
    client method chain before emitting a prompt asset.
    """
    if not qn:
        return False

    short = qn.rsplit(".", 1)[-1] if "." in qn else qn
    if short == "create_deep_agent":
        return True
    if short in _KNOWN_AI_CLIENT_CLASSES:
        return True
    if short not in _AI_CLIENT_CALLS:
        return False

    class_segment = _extract_class_segment(qn)
    return bool(class_segment and class_segment in _KNOWN_AI_CLIENT_CLASSES)


def _detect_model_kwargs(result: "CodeAnalysisResult") -> list[AIComponent]:
    """Extract models, LLM endpoints, and bare clients from AI constructor kwargs via LibCST.

    Classification rules (no dual emission):
    1. Constructor has ``base_url`` / ``azure_endpoint`` / ``api_base`` / ``endpoint_url``
       with a concrete URL -> **LLM_ENDPOINT**.
    2. Constructor has ``model=`` with a registry-known value -> **MODEL**.
    3. Known client class with neither model nor URL kwargs -> **needs_agentic MODEL**
       (bare client; prune if agentic layer cannot resolve a model ID).
    """
    from .model_detector import registry_lookup

    components: list[AIComponent] = []
    seen: set[tuple[str, int]] = set()

    all_calls = [(c, c.line_number) for c in result.calls]
    all_calls += [(a.call, a.line_number) for a in result.assignments]

    for call_obs, line in all_calls:
        qn = call_obs.qualified_name or ""
        short = qn.rsplit(".", 1)[-1] if "." in qn else qn
        if short not in _KNOWN_AI_CLIENT_CLASSES:
            continue

        key = (result.file_path, line)
        if key in seen:
            continue

        endpoint_url: str | None = None
        for ek in _ENDPOINT_KWARG_NAMES:
            raw = call_obs.arguments.get(ek)
            if isinstance(raw, str) and not raw.startswith(
                ("VARIABLE:", "ATTRIBUTE:", "COMPLEX_TYPE:")
            ):
                cleaned = raw.strip().strip("'\"")
                if cleaned.startswith(("http://", "https://")):
                    endpoint_url = cleaned
                    break

        if endpoint_url:
            seen.add(key)
            components.append(
                AIComponent(
                    name=endpoint_url,
                    component_type=AIComponentType.LLM_ENDPOINT,
                    file_path=result.file_path,
                    line_number=line,
                    framework=short,
                    detection_source=DetectionSource.CODE_ANALYSIS,
                    heuristic_confidence=0.9,
                    metadata={
                        "endpoint_url": endpoint_url,
                        "constructor": short,
                    },
                )
            )
            continue

        found_model = False
        for kwarg_name in _MODEL_KWARG_NAMES:
            val = call_obs.arguments.get(kwarg_name)
            if not isinstance(val, str):
                continue
            if val.startswith(("VARIABLE:", "ATTRIBUTE:", "COMPLEX_TYPE:")):
                continue
            val = val.strip().strip("'\"")
            if not val or len(val) < 3:
                continue
            reg = registry_lookup(val)
            if reg:
                seen.add(key)
                components.append(
                    AIComponent(
                        name=val,
                        component_type=AIComponentType.MODEL,
                        file_path=result.file_path,
                        line_number=line,
                        model_name=val,
                        framework=short,
                        detection_source=DetectionSource.CODE_ANALYSIS,
                        metadata={
                            "provider": reg.get("provider", ""),
                            "extracted_from_kwarg": kwarg_name,
                            "constructor": short,
                        },
                    )
                )
                found_model = True
                break

        if not found_model and short in _BARE_CLIENT_HINTS:
            seen.add(key)
            components.append(
                AIComponent(
                    name=short,
                    component_type=AIComponentType.MODEL,
                    file_path=result.file_path,
                    line_number=line,
                    framework=short,
                    detection_source=DetectionSource.CODE_ANALYSIS,
                    heuristic_confidence=0.3,
                    needs_agentic=True,
                    agentic_hint=_BARE_CLIENT_HINTS[short],
                    metadata={
                        "detection_method": "bare_provider_client",
                        "constructor": short,
                    },
                )
            )

    return components


def _detect_prompt_kwargs(result: "CodeAnalysisResult") -> list[AIComponent]:
    """Detect prompts consumed by confirmed AI framework calls.

    When a confirmed AI call has a prompt-accepting kwarg (``system_prompt=``,
    ``system=``, ``instructions=``, etc.), the value is an AI prompt asset.
    """
    components: list[AIComponent] = []
    seen: set[tuple[str, int]] = set()
    variable_map: dict[str, str] = {}

    for assignment in result.assignments:
        if assignment.target_qualified_name and assignment.call.qualified_name:
            variable_map[assignment.target_qualified_name] = (
                assignment.call.qualified_name
            )

    all_calls = [(c, c.line_number) for c in result.calls]
    all_calls += [(a.call, a.line_number) for a in result.assignments]

    for call_obs, line in all_calls:
        qn = _resolve_chain(call_obs.qualified_name or "", variable_map)
        if not _is_confirmed_ai_prompt_call(qn):
            continue

        for kwarg_name in _PROMPT_KWARG_NAMES:
            val = call_obs.arguments.get(kwarg_name)
            if val is None:
                continue
            key = (result.file_path, line)
            if key in seen:
                continue

            if isinstance(val, str) and not val.startswith(
                ("VARIABLE:", "ATTRIBUTE:", "COMPLEX_TYPE:")
            ):
                seen.add(key)
                display = val[:80] + "..." if len(val) > 80 else val
                components.append(
                    AIComponent(
                        name=f"prompt ({kwarg_name})",
                        component_type=AIComponentType.PROMPT,
                        file_path=result.file_path,
                        line_number=line,
                        text=val,
                        framework=qn.split(".")[0] if "." in qn else "",
                        detection_source=DetectionSource.CODE_ANALYSIS,
                        metadata={"prompt_kwarg": kwarg_name, "enclosing_call": qn},
                    )
                )
            elif isinstance(val, str) and val.startswith("VARIABLE:"):
                var_name = val.removeprefix("VARIABLE:")
                seen.add(key)
                components.append(
                    AIComponent(
                        name=var_name,
                        component_type=AIComponentType.PROMPT,
                        file_path=result.file_path,
                        line_number=line,
                        framework=qn.split(".")[0] if "." in qn else "",
                        detection_source=DetectionSource.CODE_ANALYSIS,
                        heuristic_confidence=0.7,
                        metadata={
                            "prompt_kwarg": kwarg_name,
                            "enclosing_call": qn,
                            "variable_ref": var_name,
                        },
                    )
                )
    return components


def _kb_entry_describes_method(kb_entry: dict[str, Any]) -> bool:
    """KB method rows describe operations, not durable AI assets."""
    return str(kb_entry.get("label") or "").strip().lower() == "method"


def _find_python_files(context: ScanContext) -> list[Path]:
    idx = context.file_index()
    if idx:
        py = [e.path for e in idx.get(".py", [])]
        py.extend(e.path for e in idx.get(".ipynb", []))
        return py

    files: list[Path] = []
    for p in context.paths:
        path = Path(p)
        if path.is_file() and path.suffix in (".py", ".ipynb"):
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
            files.extend(sorted(path.rglob("*.ipynb")))
    return files


def _collect_symbols(result: CodeAnalysisResult, symbols: set[str]) -> None:
    """Gather all symbols from a parsed file result into the shared set."""
    variable_map: dict[str, str] = {}
    for assignment in result.assignments:
        if assignment.target_qualified_name and assignment.call.qualified_name:
            variable_map[assignment.target_qualified_name] = (
                assignment.call.qualified_name
            )

    for assignment in result.assignments:
        name = _resolve_chain(assignment.call.qualified_name, variable_map)
        symbols.add(name)
    for dec in result.decorators:
        name = dec.decorator_qualified_name
        if dec.instance_variable and dec.instance_variable in variable_map:
            base = variable_map[dec.instance_variable]
            attr = name.split(".", 1)[-1] if "." in name else name
            name = f"{base}.{attr}"
        name = _resolve_chain(name, variable_map)
        symbols.add(name)
    for ctx in result.context_managers:
        if ctx.context_expr_qualified_name:
            symbols.add(_resolve_chain(ctx.context_expr_qualified_name, variable_map))


def _batch_kb_lookup(
    db: CatalogDB,
    symbols: set[str],
) -> dict[str, dict[str, Any]]:
    """Query DuckDB for symbols via the token-based hash-join index.

    Returns a filtered ``kb_by_id`` dict containing only entries whose concept
    is in :data:`ALLOWED_CONCEPTS`.
    """
    query_suffixes: set[str] = set(symbols)
    for s in symbols:
        if "." in s:
            query_suffixes.add(s.rsplit(".", 1)[-1])
            cls = _extract_class_segment(s)
            if cls:
                query_suffixes.add(cls)

    matched = db.find_components_by_suffixes(list(query_suffixes))

    kb_by_id: dict[str, dict[str, Any]] = {}
    for entry in matched:
        concept = (entry.get("concept") or "").lower()
        if concept not in ALLOWED_CONCEPTS:
            continue
        eid = entry["id"]
        if eid not in kb_by_id:
            kb_by_id[eid] = entry

    return kb_by_id


def _process_file_with_cache(
    result: CodeAnalysisResult,
    kb_by_id: dict[str, dict[str, Any]],
    kb_patterns: dict[AIComponentType, frozenset[str]] | None = None,
    kb_fw_prefixes: frozenset[str] | None = None,
) -> list[AIComponent]:
    """Match one file using pre-fetched KB entries (no per-file DB query)."""
    variable_map: dict[str, str] = {}
    for assignment in result.assignments:
        if assignment.target_qualified_name and assignment.call.qualified_name:
            variable_map[assignment.target_qualified_name] = (
                assignment.call.qualified_name
            )

    observations: list[dict[str, Any]] = []
    components: list[AIComponent] = []
    seen: set[tuple[str, int]] = set()

    for assignment in result.assignments:
        name = _resolve_chain(assignment.call.qualified_name, variable_map)
        observations.append(
            _obs(
                name,
                result.file_path,
                assignment.line_number,
                "assignment",
                args=assignment.call.arguments,
                assigned_target=assignment.target_qualified_name,
            )
        )

    for dec in result.decorators:
        name = dec.decorator_qualified_name
        if dec.instance_variable and dec.instance_variable in variable_map:
            base = variable_map[dec.instance_variable]
            attr = name.split(".", 1)[-1] if "." in name else name
            name = f"{base}.{attr}"
        name = _resolve_chain(name, variable_map)
        observations.append(
            _obs(
                name,
                result.file_path,
                dec.line_number,
                "decorator",
                decorated=dec.decorated_function_name,
            )
        )

    for ctx in result.context_managers:
        if not ctx.context_expr_qualified_name:
            continue
        name = _resolve_chain(ctx.context_expr_qualified_name, variable_map)
        observations.append(
            _obs(name, result.file_path, ctx.line_number, "context_manager")
        )

    imports = _import_strings(getattr(result, "imports", []) or [])

    fw_prefixes = kb_fw_prefixes or _AGENT_FRAMEWORK_PREFIXES
    pat = kb_patterns or {}
    _call_pattern_map: list[tuple[frozenset[str], frozenset[str], AIComponentType]] = [
        (
            pat.get(AIComponentType.AGENT, _AGENT_CREATION_PATTERNS),
            fw_prefixes,
            AIComponentType.AGENT,
        ),
        (
            pat.get(AIComponentType.TOOL, _STATIC_TOOL_PATTERNS),
            fw_prefixes,
            AIComponentType.TOOL,
        ),
        (
            pat.get(AIComponentType.MEMORY, _STATIC_MEMORY_PATTERNS),
            fw_prefixes,
            AIComponentType.MEMORY,
        ),
        (
            pat.get(AIComponentType.PROMPT, _STATIC_PROMPT_PATTERNS),
            fw_prefixes,
            AIComponentType.PROMPT,
        ),
    ]

    for call_obs in result.calls:
        qn = _resolve_chain(call_obs.qualified_name, variable_map)
        for patterns, fw_pref, comp_type in _call_pattern_map:
            if _is_known_call(qn, imports, patterns, fw_pref):
                key = (result.file_path, call_obs.line_number)
                if key not in seen:
                    short = qn.rsplit(".", 1)[-1] if "." in qn else qn
                    components.append(
                        AIComponent(
                            name=short,
                            component_type=comp_type,
                            file_path=result.file_path,
                            line_number=call_obs.line_number,
                            framework=qn.split(".")[0] if "." in qn else "",
                            detection_source=DetectionSource.CODE_ANALYSIS,
                            metadata={"call_pattern": qn},
                        )
                    )
                    seen.add(key)
                break

    for assignment in result.assignments:
        qn = _resolve_chain(assignment.call.qualified_name, variable_map)
        for patterns, fw_pref, comp_type in _call_pattern_map:
            if _is_known_call(qn, imports, patterns, fw_pref):
                key = (result.file_path, assignment.line_number)
                if key not in seen:
                    target = assignment.target_qualified_name or qn
                    short = target.rsplit(".", 1)[-1] if "." in target else target
                    components.append(
                        AIComponent(
                            name=short,
                            component_type=comp_type,
                            file_path=result.file_path,
                            line_number=assignment.line_number,
                            framework=qn.split(".")[0] if "." in qn else "",
                            detection_source=DetectionSource.CODE_ANALYSIS,
                            metadata={"call_pattern": qn, "assigned_to": target},
                        )
                    )
                    seen.add(key)
                break

    suffix_idx = _build_suffix_index(kb_by_id)
    imported_frameworks = _extract_frameworks_from_imports(imports)

    for obs_data in observations:
        key = (obs_data["file"], obs_data["line"])
        if key in seen:
            continue

        match = _match_observation_rich(
            obs_data["name"],
            kb_by_id,
            imported_frameworks,
            suffix_idx,
        )

        if match.is_confirmed:
            kb_entry = match.entry
            assert kb_entry is not None  # noqa: S101
            if _kb_entry_describes_method(kb_entry):
                continue
            concept = kb_entry["concept"].lower()
            comp_type = _CONCEPT_TO_TYPE.get(concept)
            if not comp_type:
                continue
            comp_type = _refine_type_from_kb_id(kb_entry["id"], comp_type)
            if comp_type is None:
                continue

            meta: dict[str, Any] = {"kb_id": kb_entry["id"]}
            if obs_data.get("assigned_target"):
                meta["assigned_target"] = obs_data["assigned_target"]
            if obs_data.get("decorated"):
                meta["decorated_function"] = obs_data["decorated"]
            if obs_data.get("args"):
                meta["arguments"] = obs_data["args"]

            display_name = obs_data["name"]
            if obs_data.get("type") == "decorator" and obs_data.get("decorated"):
                display_name = obs_data["decorated"]

            components.append(
                AIComponent(
                    name=display_name,
                    component_type=comp_type,
                    file_path=obs_data["file"],
                    line_number=obs_data["line"],
                    framework=kb_entry.get("framework", ""),
                    detection_source=DetectionSource.KB_ENRICHMENT,
                    kb_concept=concept,
                    kb_label=kb_entry.get("label", ""),
                    needs_agentic=True,
                    agentic_hint=(
                        f"KB catalog matched '{display_name}' as "
                        f"'{concept}' (type={comp_type.value}, "
                        f"id={kb_entry['id']}). Default: REMOVE if "
                        f"code_context reveals a CRUD handler, DTO, "
                        f"utility, or test artifact. Only KEEP if it is "
                        f"a genuine AI component — then (1) verify the "
                        f"assigned type matches what the code does, "
                        f"(2) enrich with concrete identifiers (model "
                        f"names, endpoint URLs, config values) from "
                        f"nearby lines."
                    ),
                    metadata=meta,
                )
            )
            seen.add(key)

        elif match.is_partial:
            if match.obs_module in imported_frameworks:
                continue
            class_seg = _extract_class_segment(obs_data["name"])
            display_name = class_seg or obs_data["name"]
            if obs_data.get("type") == "decorator" and obs_data.get("decorated"):
                display_name = obs_data["decorated"]
            hint = (
                f"Class '{display_name}' partially matches KB entry "
                f"'{match.partial_kb_id}' but import module "
                f"'{match.obs_module}' differs from KB framework "
                f"'{match.partial_kb_framework}'. Module mismatch is a "
                f"red flag — REMOVE unless wrapper chain in code_context "
                f"proves this resolves to a genuine model identifier."
            )
            meta_partial: dict[str, Any] = {
                "partial_kb_id": match.partial_kb_id,
                "obs_module": match.obs_module,
            }
            if obs_data.get("assigned_target"):
                meta_partial["assigned_target"] = obs_data["assigned_target"]
            if obs_data.get("args"):
                meta_partial["arguments"] = obs_data["args"]
            components.append(
                AIComponent(
                    name=display_name,
                    component_type=AIComponentType.MODEL,
                    file_path=obs_data["file"],
                    line_number=obs_data["line"],
                    framework=match.obs_module,
                    detection_source=DetectionSource.KB_ENRICHMENT,
                    heuristic_confidence=0.3,
                    needs_agentic=True,
                    agentic_hint=hint,
                    metadata=meta_partial,
                )
            )
            seen.add(key)

    return components


def _process_file(
    result: CodeAnalysisResult,
    db: CatalogDB,
    kb_patterns: dict[AIComponentType, frozenset[str]] | None = None,
    kb_fw_prefixes: frozenset[str] | None = None,
) -> list[AIComponent]:
    """Match one file's parsed observations against the KB."""

    variable_map: dict[str, str] = {}
    for assignment in result.assignments:
        if assignment.target_qualified_name and assignment.call.qualified_name:
            variable_map[assignment.target_qualified_name] = (
                assignment.call.qualified_name
            )

    observations: list[dict[str, Any]] = []
    symbols: set[str] = set()
    components: list[AIComponent] = []
    seen: set[tuple[str, int]] = set()

    for assignment in result.assignments:
        name = _resolve_chain(assignment.call.qualified_name, variable_map)
        observations.append(
            _obs(
                name,
                result.file_path,
                assignment.line_number,
                "assignment",
                args=assignment.call.arguments,
                assigned_target=assignment.target_qualified_name,
            )
        )
        symbols.add(name)

    for dec in result.decorators:
        name = dec.decorator_qualified_name
        if dec.instance_variable and dec.instance_variable in variable_map:
            base = variable_map[dec.instance_variable]
            attr = name.split(".", 1)[-1] if "." in name else name
            name = f"{base}.{attr}"
        name = _resolve_chain(name, variable_map)
        observations.append(
            _obs(
                name,
                result.file_path,
                dec.line_number,
                "decorator",
                decorated=dec.decorated_function_name,
            )
        )
        symbols.add(name)

    for ctx in result.context_managers:
        if not ctx.context_expr_qualified_name:
            continue
        name = _resolve_chain(ctx.context_expr_qualified_name, variable_map)
        observations.append(
            _obs(name, result.file_path, ctx.line_number, "context_manager")
        )
        symbols.add(name)

    imports = _import_strings(getattr(result, "imports", []) or [])

    fw_prefixes = kb_fw_prefixes or _AGENT_FRAMEWORK_PREFIXES
    pat = kb_patterns or {}
    _call_pattern_map: list[tuple[frozenset[str], frozenset[str], AIComponentType]] = [
        (
            pat.get(AIComponentType.AGENT, _AGENT_CREATION_PATTERNS),
            fw_prefixes,
            AIComponentType.AGENT,
        ),
        (
            pat.get(AIComponentType.TOOL, _STATIC_TOOL_PATTERNS),
            fw_prefixes,
            AIComponentType.TOOL,
        ),
        (
            pat.get(AIComponentType.MEMORY, _STATIC_MEMORY_PATTERNS),
            fw_prefixes,
            AIComponentType.MEMORY,
        ),
        (
            pat.get(AIComponentType.PROMPT, _STATIC_PROMPT_PATTERNS),
            fw_prefixes,
            AIComponentType.PROMPT,
        ),
    ]

    for call_obs in result.calls:
        qn = _resolve_chain(call_obs.qualified_name, variable_map)
        for patterns, fw_prefixes, comp_type in _call_pattern_map:
            if _is_known_call(qn, imports, patterns, fw_prefixes):
                key = (result.file_path, call_obs.line_number)
                if key not in seen:
                    short = qn.rsplit(".", 1)[-1] if "." in qn else qn
                    components.append(
                        AIComponent(
                            name=short,
                            component_type=comp_type,
                            file_path=result.file_path,
                            line_number=call_obs.line_number,
                            framework=qn.split(".")[0] if "." in qn else "",
                            detection_source=DetectionSource.CODE_ANALYSIS,
                            metadata={"call_pattern": qn},
                        )
                    )
                    seen.add(key)
                break

    for assignment in result.assignments:
        qn = _resolve_chain(assignment.call.qualified_name, variable_map)
        for patterns, fw_prefixes, comp_type in _call_pattern_map:
            if _is_known_call(qn, imports, patterns, fw_prefixes):
                key = (result.file_path, assignment.line_number)
                if key not in seen:
                    target = assignment.target_qualified_name or qn
                    short = target.rsplit(".", 1)[-1] if "." in target else target
                    components.append(
                        AIComponent(
                            name=short,
                            component_type=comp_type,
                            file_path=result.file_path,
                            line_number=assignment.line_number,
                            framework=qn.split(".")[0] if "." in qn else "",
                            detection_source=DetectionSource.CODE_ANALYSIS,
                            metadata={"call_pattern": qn, "assigned_to": target},
                        )
                    )
                    seen.add(key)
                break

    if not symbols:
        return components

    query_suffixes = set(symbols)
    for s in symbols:
        if "." in s:
            query_suffixes.add(s.rsplit(".", 1)[-1])
            cls = _extract_class_segment(s)
            if cls:
                query_suffixes.add(cls)

    matched = db.find_components_by_suffixes(list(query_suffixes))

    kb_by_id: dict[str, dict[str, Any]] = {}
    for entry in matched:
        concept = (entry.get("concept") or "").lower()
        if concept not in ALLOWED_CONCEPTS:
            continue
        eid = entry["id"]
        if eid not in kb_by_id:
            kb_by_id[eid] = entry

    suffix_idx = _build_suffix_index(kb_by_id)
    imported_frameworks = _extract_frameworks_from_imports(imports)

    for obs_data in observations:
        key = (obs_data["file"], obs_data["line"])
        if key in seen:
            continue

        match = _match_observation_rich(
            obs_data["name"], kb_by_id, imported_frameworks, suffix_idx
        )

        if match.is_confirmed:
            kb_entry = match.entry
            assert kb_entry is not None  # noqa: S101
            if _kb_entry_describes_method(kb_entry):
                continue
            concept = kb_entry["concept"].lower()
            comp_type = _CONCEPT_TO_TYPE.get(concept)
            if not comp_type:
                continue
            comp_type = _refine_type_from_kb_id(kb_entry["id"], comp_type)
            if comp_type is None:
                continue

            seen.add(key)

            display_name = obs_data["name"]
            if obs_data["type"] == "decorator" and obs_data.get("decorated"):
                display_name = obs_data["decorated"]

            components.append(
                AIComponent(
                    name=display_name,
                    component_type=comp_type,
                    file_path=obs_data["file"],
                    line_number=obs_data["line"],
                    framework=kb_entry.get("framework", ""),
                    detection_source=DetectionSource.KB_ENRICHMENT,
                    kb_concept=concept,
                    kb_label=kb_entry.get("label", ""),
                    needs_agentic=True,
                    agentic_hint=(
                        f"KB catalog matched '{display_name}' as "
                        f"'{concept}' (type={comp_type.value}, "
                        f"id={kb_entry['id']}). Default: REMOVE if "
                        f"code_context reveals a CRUD handler, DTO, "
                        f"utility, or test artifact. Only KEEP if it is "
                        f"a genuine AI component — then (1) verify the "
                        f"assigned type matches what the code does, "
                        f"(2) enrich with concrete identifiers (model "
                        f"names, endpoint URLs, config values) from "
                        f"nearby lines."
                    ),
                    metadata={
                        "kb_id": kb_entry["id"],
                        "observation_type": obs_data["type"],
                    },
                )
            )

        elif match.is_partial:
            if match.obs_module in imported_frameworks:
                continue
            class_seg = _extract_class_segment(obs_data["name"])
            display_name = class_seg or obs_data["name"]
            if obs_data["type"] == "decorator" and obs_data.get("decorated"):
                display_name = obs_data["decorated"]
            hint = (
                f"Class '{display_name}' partially matches KB entry "
                f"'{match.partial_kb_id}' but import module "
                f"'{match.obs_module}' differs from KB framework "
                f"'{match.partial_kb_framework}'. Module mismatch is a "
                f"red flag — REMOVE unless wrapper chain in code_context "
                f"proves this resolves to a genuine model identifier."
            )
            components.append(
                AIComponent(
                    name=display_name,
                    component_type=AIComponentType.MODEL,
                    file_path=obs_data["file"],
                    line_number=obs_data["line"],
                    framework=match.obs_module,
                    detection_source=DetectionSource.KB_ENRICHMENT,
                    heuristic_confidence=0.3,
                    needs_agentic=True,
                    agentic_hint=hint,
                    metadata={
                        "partial_kb_id": match.partial_kb_id,
                        "obs_module": match.obs_module,
                        "observation_type": obs_data["type"],
                    },
                )
            )
            seen.add(key)

    return components


def _obs(
    name: str,
    file_path: str,
    line: int,
    obs_type: str,
    *,
    args: Optional[dict[str, Any]] = None,
    assigned_target: Optional[str] = None,
    decorated: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "file": file_path,
        "line": line,
        "type": obs_type,
        "args": args or {},
        "assigned_target": assigned_target,
        "decorated": decorated,
    }


def _resolve_chain(name: str, variable_map: dict[str, str]) -> str:
    """Resolve ``var.method`` → ``FullyQualified.method`` via the variable map."""
    if "." in name:
        head, tail = name.split(".", 1)
        if head in variable_map:
            return f"{variable_map[head]}.{tail}"
    return name


def _build_suffix_index(
    kb_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build a suffix→entries index to avoid O(n) scans in _match_observation.

    Keys are each dot-delimited suffix of the KB id.  E.g. for
    ``langchain_community.vectorstores.faiss.FAISS`` we index entries under
    ``FAISS``, ``faiss.FAISS``, ``vectorstores.faiss.FAISS``, etc.
    """
    idx: dict[str, list[dict[str, Any]]] = {}
    for kb_id, entry in kb_by_id.items():
        parts = kb_id.split(".")
        for i in range(1, len(parts) + 1):
            suffix = ".".join(parts[-i:])
            idx.setdefault(suffix, []).append(entry)
    return idx


def _match_observation_rich(
    obs_name: str,
    kb_by_id: dict[str, dict[str, Any]],
    imported_frameworks: set[str],
    suffix_index: dict[str, list[dict[str, Any]]] | None = None,
) -> _MatchResult:
    """Match an observation against the KB, returning confirmed, partial, or empty.

    Tier 1: exact match on KB id → confirmed.
    Tier 2: suffix match on full qualified name, framework-related → confirmed.
    Tier 3: suffix match on SHORT class name, framework-related → confirmed.
    Partial: leaf class exists in KB suffix index but framework guard rejected.
    Empty: no KB overlap at all.

    Bare, lowercase observations (e.g. ``agent`` from ``result = agent(...)``)
    are variable invocations, not AI-component identifiers.  They carry no
    module context so the suffix index cannot reason about framework
    relatedness and they produce spurious matches against attribute KB
    entries that share the same leaf token.  We short-circuit matching
    for that shape after the exact-id check so callers still recover the
    tier-1 path if the observation happens to be a top-level catalog id.
    """
    if obs_name in kb_by_id:
        return _MatchResult(entry=kb_by_id[obs_name])

    if "." not in obs_name and obs_name[:1].islower():
        return _MatchResult()

    obs_module = obs_name.split(".")[0] if "." in obs_name else ""

    if suffix_index is not None:
        all_candidates = suffix_index.get(obs_name, [])
        fw_candidates = [
            e
            for e in all_candidates
            if not obs_module or _frameworks_related(obs_module, e.get("framework", ""))
        ]
        if fw_candidates:
            return _MatchResult(
                entry=_pick_best(fw_candidates, imported_frameworks, obs_module)
            )

        if all_candidates and obs_module:
            best = _pick_best(all_candidates, imported_frameworks, obs_module)
            return _MatchResult(
                partial_kb_id=best.get("id", ""),
                partial_kb_framework=best.get("framework", ""),
                obs_module=obs_module,
            )

        class_name = _extract_class_segment(obs_name)
        if class_name:
            all_cls_candidates = suffix_index.get(class_name, [])
            if all_cls_candidates:
                best = _pick_best(all_cls_candidates, imported_frameworks, obs_module)
                if not obs_module or _frameworks_related(
                    obs_module, best.get("framework", "")
                ):
                    return _MatchResult(entry=best)
                return _MatchResult(
                    partial_kb_id=best.get("id", ""),
                    partial_kb_framework=best.get("framework", ""),
                    obs_module=obs_module,
                )
        return _MatchResult()

    candidates: list[dict[str, Any]] = []
    all_suffix_candidates: list[dict[str, Any]] = []
    for kb_id, entry in kb_by_id.items():
        if kb_id.endswith("." + obs_name):
            all_suffix_candidates.append(entry)
            if not obs_module or _frameworks_related(
                obs_module, entry.get("framework", "")
            ):
                candidates.append(entry)

    if candidates:
        return _MatchResult(
            entry=_pick_best(candidates, imported_frameworks, obs_module)
        )
    if all_suffix_candidates and obs_module:
        best = _pick_best(all_suffix_candidates, imported_frameworks, obs_module)
        return _MatchResult(
            partial_kb_id=best.get("id", ""),
            partial_kb_framework=best.get("framework", ""),
            obs_module=obs_module,
        )

    class_name = _extract_class_segment(obs_name)
    if not class_name:
        return _MatchResult()

    all_cls: list[dict[str, Any]] = []
    for kb_id, entry in kb_by_id.items():
        if kb_id.endswith("." + class_name):
            all_cls.append(entry)

    if not all_cls:
        return _MatchResult()

    best = _pick_best(all_cls, imported_frameworks, obs_module)
    if not obs_module or _frameworks_related(obs_module, best.get("framework", "")):
        return _MatchResult(entry=best)
    return _MatchResult(
        partial_kb_id=best.get("id", ""),
        partial_kb_framework=best.get("framework", ""),
        obs_module=obs_module,
    )


def _extract_class_segment(obs_name: str) -> Optional[str]:
    """Return the nearest class-like (uppercase-start) segment from a dotted name.

    Scans right-to-left so that ``pkg.FAISS.from_texts`` yields ``FAISS`` and
    ``openai.OpenAI.chat.completions.create`` yields ``OpenAI``.  Returns
    ``None`` when no uppercase segment is found or the result would be the
    entire *obs_name* (already covered by Tier 1).
    """
    if "." not in obs_name:
        return None
    parts = obs_name.split(".")
    for segment in reversed(parts):
        if segment and segment[0].isupper():
            if segment == obs_name:
                return None
            return segment
    return None


def _frameworks_related(obs_module: str, kb_framework: str) -> bool:
    """Return ``True`` when the observation module and KB framework belong to
    the same package family.

    ``langchain_openai`` and ``langchain_community`` share the ``langchain``
    prefix; ``crewai`` matches ``crewai``; ``openai`` does not match
    ``langchain_community``.
    """
    if not obs_module or not kb_framework:
        return True
    if obs_module == kb_framework:
        return True
    obs_top = obs_module.split("_")[0].split("-")[0]
    kb_top = kb_framework.split("_")[0].split("-")[0]
    return obs_top == kb_top


def _pick_best(
    candidates: list[dict[str, Any]],
    imported_frameworks: set[str],
    obs_module: str = "",
) -> dict[str, Any]:
    """From a set of KB candidates, prefer the one whose framework matches best.

    Priority: framework matches the observation's module prefix > framework is
    imported > deterministic fallback (prefer entries where path override agrees
    with KB concept).
    """
    if len(candidates) == 1:
        return candidates[0]

    # Prefer the candidate whose framework matches the observation module prefix.
    if obs_module:
        for c in candidates:
            fw = c.get("framework") or ""
            if fw == obs_module or fw.startswith(obs_module):
                return c

    for c in candidates:
        fw = c.get("framework") or ""
        fw_top = fw.split("_")[0].split("-")[0]
        if fw in imported_frameworks or fw_top in imported_frameworks:
            return c

    def _path_override_score(c: dict[str, Any]) -> tuple[int, str]:
        kid = c.get("id", "")
        concept = c.get("concept", "")
        concept_type = _CONCEPT_TO_TYPE.get(concept)
        if concept_type:
            refined = _refine_type_from_kb_id(kid, concept_type)
            if refined is None:
                return (2, kid)
            if refined != concept_type:
                return (1, kid)
        return (0, kid)

    candidates.sort(key=_path_override_score)
    return candidates[0]
