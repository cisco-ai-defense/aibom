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

from enum import Enum


class AIComponentType(str, Enum):
    """All recognized AI asset types that AIBOM can detect.

    SYNC NOTE: The agentic system prompt in ``agentic/prompts.py``
    lists these categories as protected (must-not-prune). When adding
    or removing values here, update that prompt to match.
    """

    # Model-related (identity + endpoints)
    MODEL = "model"
    LLM_ENDPOINT = "llm_endpoint"
    MODEL_ENDPOINT = "model_endpoint"

    @property
    def is_model_related(self) -> bool:
        return self in (
            AIComponentType.MODEL,
            AIComponentType.LLM_ENDPOINT,
            AIComponentType.MODEL_ENDPOINT,
        )

    # Agentic
    AGENT = "agent"
    # Local code that invokes a *remote* A2A agent. The proxy is not
    # itself an agent — it wraps one. Paired with ``INVOKES_A2A_AGENT``
    # relationships and ``A2A_AGENT_CLIENT_SERVER`` cross-repo links.
    AGENT_PROXY = "agent_proxy"
    TOOL = "tool"
    MCP_SERVER = "mcp_server"
    MCP_CLIENT = "mcp_client"
    MCP_GATEWAY = "mcp_gateway"

    # Data / retrieval
    EMBEDDING = "embedding"
    VECTOR_STORE = "vector_store"
    DATASET = "dataset"
    RETRIEVER = "retriever"
    KNOWLEDGE_BASE = "knowledge_base"
    FEATURE_STORE = "feature_store"

    # Memory / state
    MEMORY = "memory"
    PROMPT = "prompt"

    # ML lifecycle
    TRAINING_RUN = "training_run"
    HYPERPARAMETER = "hyperparameter"
    MODEL_ARTIFACT = "model_artifact"
    EXPERIMENT_TRACKER = "experiment_tracker"
    MODEL_REGISTRY = "model_registry"
    DATA_VERSIONING = "data_versioning"
    ML_PIPELINE = "ml_pipeline"

    # Operational
    GUARDRAIL = "guardrail"
    SKILL = "skill"
    OBSERVABILITY = "observability"
    SECRET = "secret"
    DEPENDENCY = "dependency"

    # Catch-all
    OTHER = "other"


class Severity(str, Enum):
    """Risk severity levels, ordered from most to least severe."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def numeric(self) -> int:
        return _SEVERITY_ORDER[self]

    def __lt__(self, other: "Severity") -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric < other.numeric

    def __le__(self, other: "Severity") -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric <= other.numeric

    def __gt__(self, other: "Severity") -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric > other.numeric

    def __ge__(self, other: "Severity") -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric >= other.numeric


_SEVERITY_ORDER: dict["Severity", int] = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
    Severity.INFO: 0,
}


class DetectionSource(str, Enum):
    """How an AI component was detected."""

    CODE_ANALYSIS = "code_analysis"
    CONFIG_FILE = "config_file"
    DEPENDENCY_MANIFEST = "dependency_manifest"
    INLINE_ANNOTATION = "inline_annotation"
    BASE_CLASS_RULE = "base_class_rule"
    KB_ENRICHMENT = "kb_enrichment"
    AGENTIC = "agentic"
    API = "api"


class CrossRepoLinkType(str, Enum):
    """Types of cross-repository links in a multi-repo scan."""

    ENV_VAR_BINDING = "ENV_VAR_BINDING"
    SHARED_MODEL = "SHARED_MODEL"
    SHARED_DEPENDENCY = "SHARED_DEPENDENCY"
    SHARED_BASE_IMAGE = "SHARED_BASE_IMAGE"
    DERIVED_FROM_REPO = "DERIVED_FROM_REPO"
    MCP_SERVER_CLIENT = "MCP_SERVER_CLIENT"
    A2A_AGENT_CLIENT_SERVER = "A2A_AGENT_CLIENT_SERVER"
    SHARED_LIBRARY = "SHARED_LIBRARY"
    CUSTOM = "CUSTOM"


class RelationshipType(str, Enum):
    """Recognized relationships between AI components."""

    USES_MODEL = "USES_MODEL"
    USES_TOOL = "USES_TOOL"
    USES_LLM = "USES_LLM"
    USES_MEMORY = "USES_MEMORY"
    USES_RETRIEVER = "USES_RETRIEVER"
    USES_EMBEDDING = "USES_EMBEDDING"
    USES_DATASET = "USES_DATASET"
    USES_VECTOR_STORE = "USES_VECTOR_STORE"
    USES_MCP_SERVER = "USES_MCP_SERVER"
    USES_MCP_GATEWAY = "USES_MCP_GATEWAY"
    USES_KNOWLEDGE_BASE = "USES_KNOWLEDGE_BASE"
    USES_FEATURE_STORE = "USES_FEATURE_STORE"
    USES_LLM_ENDPOINT = "USES_LLM_ENDPOINT"
    TRAINS_MODEL = "TRAINS_MODEL"
    LOGS_TO = "LOGS_TO"
    USES_GUARDRAIL = "USES_GUARDRAIL"
    OBSERVES = "OBSERVES"
    OBSERVED_BY = "OBSERVED_BY"
    EXPOSES_TOOL = "EXPOSES_TOOL"
    USES_MCP_CLIENT = "USES_MCP_CLIENT"
    HOSTS_MODEL = "HOSTS_MODEL"
    INVOKES_A2A_AGENT = "INVOKES_A2A_AGENT"
    EXPOSES_A2A_AGENT = "EXPOSES_A2A_AGENT"
    # An agent invoking / delegating to another agent.
    USES_AGENT = "USES_AGENT"
    # An agent using a skill (Semantic Kernel / Copilot-style plugin or skill).
    USES_SKILL = "USES_SKILL"
    CUSTOM = "CUSTOM"
