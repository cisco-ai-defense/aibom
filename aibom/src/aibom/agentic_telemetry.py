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

"""Privacy-preserving Galileo telemetry for the agentic classifier.

The Galileo SDK is deliberately imported only when an enabled, sampled trace
is started.  This keeps observability an optional dependency and makes every
failure in this module a no-op from the scanner's perspective.

Only controlled counters/labels, content-free per-call timing and ordering, and
HMAC pseudonyms leave this module. Raw source, paths, prompts, model text, tool
arguments/results, environment values, exceptions, and component names are
intentionally absent from the public API and rejected again at flush time.
"""

from __future__ import annotations

import atexit
import hashlib
import hmac
import importlib
import json
import logging
import math
import os
import re
import secrets
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Literal, Mapping, Sequence
from urllib.parse import urlparse
from uuid import UUID

from aibom.agentic.telemetry_tool_names import TOOL_NAMES as _TOOL_NAMES
from aibom.models.enums import AIComponentType

_LOGGER = logging.getLogger(__name__)

_PROCESS_HMAC_KEY = secrets.token_bytes(32)
_HMAC_KEY_ENV = "AIBOM_GALILEO_HMAC_KEY"
# Explicit, off-by-default egress opt-in for sanitized telemetry to Galileo's
# hosted console. Only the canonical app.galileo.ai console origin is accepted.
_ALLOW_PUBLIC_CLOUD_ENV = "AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD"
_MAX_SAFE_INTEGER = (1 << 63) - 1
_MAX_SETUP_BUDGET_S = 10.0
_HOSTED_CONSOLE_HOST = "app.galileo.ai"
_HOSTED_API_HOST = "api.galileo.ai"
_HOSTED_API_ORIGIN = f"https://{_HOSTED_API_HOST}"

_COMPONENT_TYPES = frozenset(item.value for item in AIComponentType)
_LANGUAGES = frozenset(
    {
        "c",
        "cpp",
        "csharp",
        "css",
        "go",
        "html",
        "java",
        "javascript",
        "julia",
        "kotlin",
        "php",
        "python",
        "r",
        "ruby",
        "rust",
        "scala",
        "shell",
        "sql",
        "swift",
        "typescript",
        "unknown",
        "other",
    }
)
_CONFIDENCE_BUCKETS = frozenset({"low", "medium", "high", "unknown", "other"})
_PROVIDERS = frozenset(
    {
        "anthropic",
        "aws",
        "azure",
        "azure_openai",
        "bedrock",
        "gemini",
        "google",
        "google_genai",
        "google_vertexai",
        "ollama",
        "openai",
        "unknown",
        "other",
    }
)
_TIERS = frozenset(
    {"simple", "complex", "fallback", "retry", "mixed", "unknown", "other"}
)
_ATTEMPT_KINDS = frozenset(
    {
        "initial",
        "retry",
        "fallback",
        "coercion",
        "middleware_validation",
        "unknown",
        "other",
    }
)
_STATUSES = frozenset(
    {
        "success",
        "degraded",
        "failed",
        "timeout",
        "rate_limited",
        "provider_outage",
        "refused",
        "parse_error",
        "cache_hit",
        "circuit_breaker",
        "skipped",
        "unknown",
        "other",
    }
)
_FAILURE_HINTS = frozenset(
    {
        "",
        "batch_timeout",
        "batch_recursion_limit",
        "rate_limited",
        "provider_outage",
        "structured_output_parse_error",
        "model_refused",
        "no_usable_output",
        "circuit_breaker_tripped",
        "retry_budget_exhausted",
        "retry_failed",
        "total_agentic_degradation",
        "unknown",
        "other",
    }
)
_SOURCE_KINDS = frozenset(
    {
        "archive",
        "cloud",
        "container",
        "directory",
        "git",
        "repository",
        "unknown",
        "other",
    }
)
_DECISION_KEYS = frozenset(
    {
        "kept",
        "removed",
        "reclassified",
        "discovered",
        "relationships",
        "risk_findings",
        "degraded",
        "enriched",
    }
)
_COMPONENT_DECISION_KEYS = frozenset(
    {"kept", "removed", "reclassified", "discovered", "enriched"}
)
LoggerFactory = Callable[[str, str], Any]
FlushSubmitter = Callable[[Any], bool]
DeferredTraceStarter = Callable[[], Any | None]


@dataclass(slots=True)
class _BufferedAttempt:
    """Sanitized child spans retained for an unsampled batch.

    A single ordered collection is required: separate LLM/tool buffers replayed
    every LLM before every tool and destroyed the actual agent trajectory.
    """

    input: str
    metadata: dict[str, Any]
    name: str
    child_spans: list[tuple[Literal["llm", "tool"], dict[str, Any]]] = field(
        default_factory=list
    )
    output: str = ""
    duration_ns: int | None = None
    status_code: int = 200
    force_retain: bool = False
    finished: bool = False


def _resolve_setup_budget_s() -> float:
    # Logger/session setup does a networked healthcheck + login on first use.
    # Hosted Galileo receives a bounded allowance for first-use authentication.
    default = 2.0
    raw = os.getenv("AIBOM_GALILEO_SETUP_BUDGET_S", "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return (
        value if math.isfinite(value) and 0 < value <= _MAX_SETUP_BUDGET_S else default
    )


_FLUSH_QUEUE_CAPACITY = 256

_TERMINAL_RETENTION_STATUSES = frozenset(
    {
        "circuit_breaker",
        "degraded",
        "failed",
        "parse_error",
        "provider_outage",
        "rate_limited",
        "refused",
        "timeout",
    }
)
_TERMINAL_FAILURE_HINTS = _FAILURE_HINTS - {"", "unknown", "other"}

_MAX_BUFFERED_ATTEMPTS = 16
_MAX_BUFFERED_LLM_SPANS = 16
_MAX_BUFFERED_TOOL_SPANS = 32

_TRACE_NAMES = frozenset({"aibom.agentic.batch", "aibom.agentic.source_summary"})
_WORKFLOW_NAMES = frozenset(f"aibom.agentic.{kind}" for kind in _ATTEMPT_KINDS)
_LLM_NAMES = frozenset({"aibom.agentic.llm"})
_TOOL_SPAN_NAMES = frozenset(
    f"aibom.tool.{name}" for name in (*sorted(_TOOL_NAMES), "other")
)
_ALLOWED_SPAN_NAMES = {
    "trace": _TRACE_NAMES,
    "workflow": _WORKFLOW_NAMES,
    "llm": _LLM_NAMES,
    "tool": _TOOL_SPAN_NAMES,
}


def _bounded_call(
    callback: Callable[[], Any],
    *,
    budget_s: float | None = None,
) -> tuple[bool, Any]:
    """Run a potentially network-bound SDK setup call without blocking a scan.

    Galileo resolves project/log-stream names and creates sessions through the
    network. A failed destination must not inherit the SDK's HTTP timeout as
    scanner latency, so setup runs in a daemon thread with a bounded budget.
    The returned boolean distinguishes a completed call (whose value may be
    ``None``) from a timed-out one.
    """

    box: dict[str, Any] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            box["value"] = callback()
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller
            box["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(
        target=_worker,
        name="aibom-galileo-setup",
        daemon=True,
    )
    worker.start()
    if not done.wait(_resolve_setup_budget_s() if budget_s is None else budget_s):
        return False, None
    if "error" in box:
        raise box["error"]
    return True, box.get("value")


def _disable_logger_atexit_flush(logger: Any) -> None:
    """Prevent Galileo's synchronous atexit flush from delaying CLI shutdown.

    This integration owns ingestion through a bounded dispatcher. Leaving the
    SDK's per-logger atexit callback registered would reintroduce an unbounded
    network wait after the scan had already completed.
    """

    terminate = getattr(logger, "terminate", None)
    if callable(terminate):
        try:
            atexit.unregister(terminate)
        except Exception:
            _LOGGER.debug("Unable to unregister Galileo logger shutdown hook")


def _disable_agent_control(logger: Any) -> bool:
    """Disable SDK auto-instrumentation before any AIBOM span is created."""
    disable = getattr(logger, "disable_agent_control", None)
    if not callable(disable):
        return True
    try:
        disable()
        return True
    except Exception:
        _LOGGER.debug("Unable to disable Galileo Agent Control bridge")
        return False


def _harden_logger(logger: Any) -> bool:
    """Remove autonomous SDK behaviors that could ingest unsanitized data."""
    # Unregister first so even a failing bridge shutdown cannot leave a
    # synchronous process-exit flush behind.
    _disable_logger_atexit_flush(logger)
    return _disable_agent_control(logger)


def _discard_logger(logger: Any) -> None:
    """Drop in-memory traces without invoking SDK terminate(), which flushes."""
    _disable_logger_atexit_flush(logger)
    _disable_agent_control(logger)
    try:
        traces = getattr(logger, "traces", None)
        if isinstance(traces, list):
            traces.clear()
    except Exception:
        _LOGGER.debug("Unable to clear rejected Galileo traces")


def _payload_text(value: Any) -> str | None:
    """Extract the JSON text from Galileo's string/message representations."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        return content if isinstance(content, str) else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 1:
            return None
        return _payload_text(value[0])
    content = getattr(value, "content", None)
    return content if isinstance(content, str) else None


def _json_object(value: Any) -> dict[str, Any] | None:
    text = _payload_text(value)
    if text is None or len(text) > 32_768:
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_non_negative_json_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_count_object(value: Any, allowed: frozenset[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value).issubset(allowed | {"other"})
        and all(_is_non_negative_json_int(item) for item in value.values())
    )


def _is_decision_object(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == _DECISION_KEYS
        and all(_is_non_negative_json_int(item) for item in value.values())
    )


def _is_decision_breakdown(value: Any, allowed: frozenset[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == _COMPONENT_DECISION_KEYS
        and all(_is_count_object(counts, allowed) for counts in value.values())
    )


def _is_tool_stats(value: Any) -> bool:
    if not isinstance(value, dict) or not set(value).issubset(_TOOL_NAMES | {"other"}):
        return False
    required = {"calls", "duration_ns", "errors", "guard_denials"}
    return all(
        isinstance(stats, dict)
        and set(stats) == required
        and all(_is_non_negative_json_int(item) for item in stats.values())
        for stats in value.values()
    )


def _is_token(value: Any, prefix: str) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(rf"{re.escape(prefix)}_[0-9a-f]{{24}}", value)
    )


def _valid_input_payload(kind: str, name: str, value: Any) -> bool:
    payload = _json_object(value)
    if payload is None:
        return False
    if kind == "trace" and name == "aibom.agentic.batch":
        return (
            set(payload)
            == {
                "attempt_kind",
                "batch_id",
                "batch_num",
                "batch_size",
                "cache_hit",
                "component_ids",
                "component_type_counts",
                "decision_chain_ids",
                "event",
                "language_counts",
                "source_id",
                "tier",
                "total_batches",
            }
            and payload["event"] == "agentic_batch"
            and payload["attempt_kind"] in _ATTEMPT_KINDS
            and _is_token(payload["batch_id"], "batch")
            and _is_non_negative_json_int(payload["batch_num"])
            and _is_non_negative_json_int(payload["batch_size"])
            and isinstance(payload["cache_hit"], bool)
            and isinstance(payload["component_ids"], list)
            and len(payload["component_ids"]) <= 128
            and all(_is_token(item, "component") for item in payload["component_ids"])
            and _is_count_object(payload["component_type_counts"], _COMPONENT_TYPES)
            and isinstance(payload["decision_chain_ids"], list)
            and len(payload["decision_chain_ids"]) == len(payload["component_ids"])
            and all(_is_token(item, "chain") for item in payload["decision_chain_ids"])
            and _is_count_object(payload["language_counts"], _LANGUAGES)
            and _is_token(payload["source_id"], "source")
            and payload["tier"] in _TIERS
            and _is_non_negative_json_int(payload["total_batches"])
        )
    if kind == "trace" and name == "aibom.agentic.source_summary":
        return (
            set(payload)
            == {
                "candidate_count",
                "candidate_count_available",
                "event",
                "source_id",
                "source_kind",
            }
            and payload["event"] == "agentic_source_summary"
            and _is_non_negative_json_int(payload["candidate_count"])
            and isinstance(payload["candidate_count_available"], bool)
            and _is_token(payload["source_id"], "source")
            and payload["source_kind"] in _SOURCE_KINDS
        )
    if kind == "workflow":
        return (
            set(payload) == {"attempt_number", "event", "kind"}
            and payload["event"] == "agentic_attempt"
            and _is_non_negative_json_int(payload["attempt_number"])
            and payload["kind"] in _ATTEMPT_KINDS
            and name == f"aibom.agentic.{payload['kind']}"
        )
    if kind == "llm":
        return (
            set(payload)
            == {
                "call_id",
                "event",
                "mode",
                "provider",
                "schema_expected",
                "sequence",
            }
            and payload["event"] == "agentic_llm_call"
            and _is_token(payload["call_id"], "call")
            and payload["mode"] in {"aggregate", "per_call"}
            and payload["provider"] in _PROVIDERS
            and isinstance(payload["schema_expected"], bool)
            and _is_non_negative_json_int(payload["sequence"])
        )
    if kind == "tool":
        if payload.get("event") == "tool_aggregate":
            return (
                set(payload) == {"calls", "event", "mode"}
                and payload["mode"] == "aggregate"
                and _is_non_negative_json_int(payload["calls"])
            )
        return (
            set(payload) == {"call_id", "event", "mode", "sequence"}
            and payload["event"] == "tool_call"
            and payload["mode"] == "per_call"
            and _is_token(payload["call_id"], "call")
            and _is_non_negative_json_int(payload["sequence"])
        )
    return False


def _valid_output_payload(kind: str, name: str, value: Any) -> bool:
    payload = _json_object(value)
    if payload is None:
        return False
    if kind == "trace" and name == "aibom.agentic.batch":
        return (
            set(payload)
            == {
                "decisions",
                "decisions_by_component_type",
                "decisions_by_confidence",
                "decisions_by_language",
                "degraded_candidates",
                "failure_hint",
                "middleware_guard_triggered",
                "schema_valid",
                "status",
            }
            and _is_decision_object(payload["decisions"])
            and _is_decision_breakdown(
                payload["decisions_by_component_type"], _COMPONENT_TYPES
            )
            and _is_decision_breakdown(
                payload["decisions_by_confidence"], _CONFIDENCE_BUCKETS
            )
            and _is_decision_breakdown(payload["decisions_by_language"], _LANGUAGES)
            and _is_non_negative_json_int(payload["degraded_candidates"])
            and payload["failure_hint"] in _FAILURE_HINTS
            and isinstance(payload["middleware_guard_triggered"], bool)
            and isinstance(payload["schema_valid"], bool)
            and payload["status"] in _STATUSES
        )
    if kind == "trace" and name == "aibom.agentic.source_summary":
        return (
            set(payload)
            == {
                "cached_tokens",
                "completion_tokens",
                "agentic_component_count",
                "decision_boundary",
                "decisions",
                "degraded_candidate_count",
                "final_component_count",
                "prompt_tokens",
                "status",
            }
            and _is_non_negative_json_int(payload["cached_tokens"])
            and _is_non_negative_json_int(payload["completion_tokens"])
            and _is_non_negative_json_int(payload["agentic_component_count"])
            and payload["decision_boundary"] == "agentic_stage_output"
            and _is_decision_object(payload["decisions"])
            and _is_non_negative_json_int(payload["degraded_candidate_count"])
            and _is_non_negative_json_int(payload["final_component_count"])
            and _is_non_negative_json_int(payload["prompt_tokens"])
            and payload["status"] in _STATUSES
        )
    if kind == "workflow":
        return (
            set(payload)
            == {
                "blocked_actions",
                "final_actions",
                "raw_actions",
                "recovered",
                "status",
                "tool_stats",
            }
            and _is_decision_object(payload["blocked_actions"])
            and _is_decision_object(payload["final_actions"])
            and _is_decision_object(payload["raw_actions"])
            and isinstance(payload["recovered"], bool)
            and payload["status"] in _STATUSES
            and _is_tool_stats(payload["tool_stats"])
        )
    if kind == "llm":
        return (
            set(payload)
            == {
                "decision_carrier",
                "decisions",
                "schema_valid",
                "status",
                "token_usage_missing",
            }
            and isinstance(payload["decision_carrier"], bool)
            and _is_decision_object(payload["decisions"])
            and isinstance(payload["schema_valid"], bool)
            and payload["status"] in _STATUSES
            and isinstance(payload["token_usage_missing"], bool)
        )
    if kind == "tool":
        return (
            set(payload) == {"errors", "guard_denials"}
            and _is_non_negative_json_int(payload["errors"])
            and _is_non_negative_json_int(payload["guard_denials"])
        )
    return False


def _safe_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return 0 <= value <= _MAX_SAFE_INTEGER
    if isinstance(value, float):
        return math.isfinite(value) and 0 <= value <= _MAX_SAFE_INTEGER
    if isinstance(value, str):
        return (
            len(value) <= 96
            and "\n" not in value
            and "\r" not in value
            and "\x00" not in value
            and not value.startswith(("/", "~", "\\"))
            and "://" not in value
        )
    return False


def _valid_metadata(kind: str, metadata: Any) -> bool:
    if metadata is None:
        return True
    if not isinstance(metadata, Mapping):
        return False
    allowed = {
        "trace": {
            "analyzer_version",
            "attempt_kind",
            "batch_size",
            "cache_hit",
            "model",
            "prompt_version",
            "provider",
            "schema_version",
            "source_id",
            "source_kind",
            "status",
            "tier",
        },
        "workflow": {"attempt_number", "kind"},
        "llm": {
            "call_id",
            "cached_tokens",
            "decision_carrier",
            "mode",
            "provider",
            "schema_valid",
            "sequence",
            "status",
            "token_usage_missing",
        },
        "tool": {
            "call_id",
            "calls",
            "errors",
            "guard_denials",
            "mode",
            "sequence",
        },
    }.get(kind, set())
    if not set(metadata).issubset(allowed) or not all(
        _safe_scalar(value) for value in metadata.values()
    ):
        return False
    numeric_fields = {
        "attempt_number",
        "batch_size",
        "cached_tokens",
        "calls",
        "errors",
        "guard_denials",
        "sequence",
    }
    boolean_fields = {
        "cache_hit",
        "decision_carrier",
        "schema_valid",
        "token_usage_missing",
    }
    enum_fields = {
        "attempt_kind": _ATTEMPT_KINDS,
        "kind": _ATTEMPT_KINDS,
        "provider": _PROVIDERS,
        "source_kind": _SOURCE_KINDS,
        "status": _STATUSES,
        "tier": _TIERS,
    }
    for key, value in metadata.items():
        if key in numeric_fields:
            if isinstance(value, bool):
                return False
            if isinstance(value, int):
                if not 0 <= value <= _MAX_SAFE_INTEGER:
                    return False
            elif not (
                isinstance(value, str)
                and bool(re.fullmatch(r"0|[1-9][0-9]{0,18}", value))
                and int(value) <= _MAX_SAFE_INTEGER
            ):
                return False
        elif key in boolean_fields:
            if not (
                isinstance(value, bool)
                or (isinstance(value, str) and value in {"True", "False"})
            ):
                return False
        elif key in enum_fields and value not in enum_fields[key]:
            return False
        elif key == "call_id" and not _is_token(value, "call"):
            return False
        elif key == "mode" and value not in {"aggregate", "per_call"}:
            return False
        elif key == "source_id" and not _is_token(value, "source"):
            return False
        elif key == "model" and not _safe_model_value(value):
            return False
        elif key in {"prompt_version", "schema_version"} and not (
            isinstance(value, str)
            and (
                value in {"unknown", "other"}
                or bool(re.fullmatch(r"[0-9a-f]{20}", value))
            )
        ):
            return False
        elif key == "analyzer_version" and not (
            isinstance(value, str)
            and (
                value in {"unknown", "other"}
                or bool(re.fullmatch(r"[0-9][0-9A-Za-z._+-]{0,63}", value))
            )
        ):
            return False
    return True


def _valid_external_id(kind: str, name: str, value: Any) -> bool:
    if kind != "trace":
        return value is None
    if name == "aibom.agentic.batch":
        return _is_token(value, "batch")
    if name == "aibom.agentic.source_summary":
        return _is_token(value, "source")
    return False


def _valid_tags(tags: Any) -> bool:
    if tags is None:
        return True
    if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes, bytearray)):
        return False
    return set(tags).issubset(
        {"aibom", "agentic", "sanitized", "summary", "tool-aggregate"}
    )


def _node_kind(node: Any) -> str:
    raw_type = getattr(node, "type", "")
    if isinstance(raw_type, Enum):
        raw_type = raw_type.value
    return str(raw_type).strip().lower()


_SDK_COMMON_NODE_FIELDS = {
    "created_at",
    "dataset_input",
    "dataset_metadata",
    "dataset_output",
    "external_id",
    "id",
    "input",
    "metrics",
    "name",
    "output",
    "parent_id",
    "redacted_input",
    "redacted_output",
    "session_id",
    "status_code",
    "step_number",
    "tags",
    "trace_id",
    "type",
    "user_metadata",
}
_SDK_NODE_FIELDS = {
    "trace": _SDK_COMMON_NODE_FIELDS | {"spans"},
    "workflow": _SDK_COMMON_NODE_FIELDS | {"spans"},
    "tool": _SDK_COMMON_NODE_FIELDS | {"spans", "tool_call_id"},
    "llm": _SDK_COMMON_NODE_FIELDS
    | {"events", "finish_reason", "model", "temperature", "tools"},
}
_SDK_LLM_METRIC_FIELDS = {
    "duration_ns",
    "num_audio_input_tokens",
    "num_audio_output_tokens",
    "num_image_input_tokens",
    "num_image_output_tokens",
    "num_input_tokens",
    "num_output_tokens",
    "num_total_tokens",
    "time_to_first_token_ns",
}


def _is_bounded_optional_int(value: Any) -> bool:
    return value is None or (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= _MAX_SAFE_INTEGER
    )


def _is_uuid(value: Any) -> bool:
    if isinstance(value, UUID):
        return True
    if not isinstance(value, str):
        return False
    try:
        UUID(value)
    except (ValueError, AttributeError):
        return False
    return True


def _is_uuid4(value: Any) -> bool:
    """Return whether *value* is an RFC 4122 UUIDv4.

    Galileo's session ingestion contract requires version 4 UUIDs. Trace and
    span identifiers are validated separately with :func:`_is_uuid` because
    they are SDK-owned and need only be structurally valid here.
    """

    try:
        parsed = value if isinstance(value, UUID) else UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return parsed.version == 4


def _valid_sdk_metrics(kind: str, metrics: Any) -> bool:
    if getattr(metrics, "model_extra", None) not in (None, {}):
        return False
    try:
        values = vars(metrics)
    except TypeError:
        return False
    expected = _SDK_LLM_METRIC_FIELDS if kind == "llm" else {"duration_ns"}
    if set(values) != expected:
        return False
    if not all(_is_bounded_optional_int(value) for value in values.values()):
        return False
    if kind == "llm":
        # Manual sanitized logging sets only these four fields. Reject future
        # or mutated modality/event metrics rather than accidentally ingesting
        # data outside the declared contract.
        for field_name in (
            "num_audio_input_tokens",
            "num_audio_output_tokens",
            "num_image_input_tokens",
            "num_image_output_tokens",
            "time_to_first_token_ns",
        ):
            if values[field_name] is not None:
                return False
        token_fields = (
            "num_input_tokens",
            "num_output_tokens",
            "num_total_tokens",
        )
        token_values = [values[field_name] for field_name in token_fields]
        return all(value is None for value in token_values) or all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in token_values
        )
    return True


def _valid_sdk_message(value: Any, *, role: str) -> bool:
    if getattr(value, "model_extra", None) not in (None, {}):
        return False
    try:
        fields = set(vars(value))
    except TypeError:
        return False
    if fields != {"content", "role", "tool_call_id", "tool_calls"}:
        return False
    actual_role = getattr(value, "role", None)
    if isinstance(actual_role, Enum):
        actual_role = actual_role.value
    return (
        isinstance(getattr(value, "content", None), str)
        and actual_role == role
        and getattr(value, "tool_call_id", None) is None
        and getattr(value, "tool_calls", None) is None
    )


def _valid_sdk_content_envelope(node: Any, kind: str) -> bool:
    raw_input = getattr(node, "input", None)
    redacted_input = getattr(node, "redacted_input", None)
    raw_output = getattr(node, "output", None)
    redacted_output = getattr(node, "redacted_output", None)
    if kind != "llm":
        return all(
            isinstance(value, str)
            for value in (raw_input, redacted_input, raw_output, redacted_output)
        )
    for value in (raw_input, redacted_input):
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or len(value) != 1
            or not _valid_sdk_message(value[0], role="user")
        ):
            return False
    return _valid_sdk_message(raw_output, role="assistant") and _valid_sdk_message(
        redacted_output, role="assistant"
    )


def _valid_sdk_envelope(node: Any, kind: str) -> bool:
    if getattr(node, "model_extra", None) not in (None, {}):
        return False
    try:
        fields = set(vars(node))
    except TypeError:
        return False
    if fields != _SDK_NODE_FIELDS[kind]:
        return False
    if not isinstance(getattr(node, "created_at", None), datetime):
        return False
    if not _is_uuid(getattr(node, "id", None)):
        return False
    for field_name in ("parent_id", "trace_id"):
        identifier = getattr(node, field_name, None)
        if identifier is not None and not _is_uuid(identifier):
            return False
    session_identifier = getattr(node, "session_id", None)
    if session_identifier is not None and not _is_uuid4(session_identifier):
        return False
    if not _is_bounded_optional_int(getattr(node, "step_number", None)):
        return False
    if getattr(node, "status_code", None) not in {200, 500}:
        return False
    if not _valid_sdk_metrics(kind, getattr(node, "metrics", None)):
        return False
    if kind == "llm" and any(
        getattr(node, field_name, None) is not None
        for field_name in ("events", "finish_reason", "temperature", "tools")
    ):
        return False
    if kind == "tool":
        tool_call_id = getattr(node, "tool_call_id", None)
        if tool_call_id is not None and not _is_token(tool_call_id, "call"):
            return False
    return True


def _validate_sdk_node(node: Any, *, root: bool = False) -> bool:
    kind = _node_kind(node)
    name = getattr(node, "name", None)
    if kind not in _ALLOWED_SPAN_NAMES or name not in _ALLOWED_SPAN_NAMES[kind]:
        return False
    if not _valid_sdk_envelope(node, kind):
        return False
    if root != (kind == "trace"):
        return False
    if not _valid_sdk_content_envelope(node, kind):
        return False
    dataset_metadata = getattr(node, "dataset_metadata", None)
    dataset_metadata_empty = dataset_metadata is None or (
        isinstance(dataset_metadata, Mapping) and not dataset_metadata
    )
    if (
        getattr(node, "dataset_input", None) is not None
        or getattr(node, "dataset_output", None) is not None
        or not dataset_metadata_empty
        or not _valid_external_id(kind, str(name), getattr(node, "external_id", None))
    ):
        return False

    raw_input = getattr(node, "input", None)
    redacted_input = getattr(node, "redacted_input", None)
    raw_output = getattr(node, "output", None)
    redacted_output = getattr(node, "redacted_output", None)
    if (
        _payload_text(raw_input) is None
        or _payload_text(raw_input) != _payload_text(redacted_input)
        or _payload_text(raw_output) is None
        or _payload_text(raw_output) != _payload_text(redacted_output)
        or not _valid_input_payload(kind, str(name), raw_input)
        or not _valid_output_payload(kind, str(name), raw_output)
    ):
        return False

    metadata = getattr(node, "user_metadata", getattr(node, "metadata", None))
    if not _valid_metadata(kind, metadata) or not _valid_tags(
        getattr(node, "tags", None)
    ):
        return False
    if kind == "llm" and not _safe_model_value(getattr(node, "model", None)):
        return False

    children = getattr(node, "spans", ())
    if children is None:
        children = ()
    if not isinstance(children, Sequence) or isinstance(
        children, (str, bytes, bytearray)
    ):
        return False
    child_kinds = {_node_kind(child) for child in children}
    if kind == "trace":
        if name == "aibom.agentic.source_summary" and children:
            return False
        if not child_kinds.issubset({"workflow"}):
            return False
    elif kind == "workflow":
        if not child_kinds.issubset({"llm", "tool"}):
            return False
    elif children:
        return False
    return all(_validate_sdk_node(child) for child in children)


def _valid_sdk_logger_envelope(logger: Any) -> bool:
    session_id = getattr(logger, "session_id", None)
    session_external_id = getattr(logger, "_session_external_id", None)
    local_metrics = getattr(logger, "local_metrics", None)
    return (
        getattr(logger, "mode", None) == "batch"
        and getattr(logger, "experiment_id", None) is None
        and getattr(logger, "trace_id", None) is None
        and getattr(logger, "span_id", None) is None
        and (session_id is None or _is_uuid4(session_id))
        and (session_external_id is None or _is_token(session_external_id, "session"))
        and (
            local_metrics is None
            or (
                isinstance(local_metrics, Sequence)
                and not isinstance(local_metrics, (str, bytes, bytearray))
                and not local_metrics
            )
        )
    )


def _validate_fake_creation(kind: str, values: Mapping[str, Any]) -> bool:
    name = values.get("name")
    if name not in _ALLOWED_SPAN_NAMES[kind]:
        return False
    if any(
        values.get(field_name) is not None
        for field_name in ("dataset_input", "dataset_output", "dataset_metadata")
    ) or not _valid_external_id(kind, str(name), values.get("external_id")):
        return False
    raw_input = values.get("input")
    redacted_input = values.get("redacted_input")
    if (
        _payload_text(raw_input) is None
        or _payload_text(raw_input) != _payload_text(redacted_input)
        or not _valid_input_payload(kind, str(name), raw_input)
        or not _valid_metadata(kind, values.get("metadata"))
        or not _valid_tags(values.get("tags"))
    ):
        return False
    if kind in {"llm", "tool"}:
        raw_output = values.get("output")
        return (
            _payload_text(raw_output) is not None
            and _payload_text(raw_output)
            == _payload_text(values.get("redacted_output"))
            and _valid_output_payload(kind, str(name), raw_output)
            and (kind != "llm" or _safe_model_value(values.get("model")))
        )
    return True


def _validate_fake_logger(logger: Any) -> bool:
    calls = getattr(logger, "calls", None)
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
        return False
    creation_events = {
        "start_trace": "trace",
        "add_workflow_span": "workflow",
        "add_llm_span": "llm",
        "add_tool_span": "tool",
    }
    trace_count = 0
    for item in calls:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], Mapping)
        ):
            return False
        event, values = item
        if event in creation_events:
            kind = creation_events[event]
            if kind == "trace":
                trace_count += 1
            if not _validate_fake_creation(kind, values):
                return False
        elif event == "conclude":
            raw_output = values.get("output")
            if (
                _payload_text(raw_output) is None
                or _payload_text(raw_output)
                != _payload_text(values.get("redacted_output"))
                or not any(
                    _valid_output_payload(kind, name, raw_output)
                    for kind, names in _ALLOWED_SPAN_NAMES.items()
                    for name in names
                )
            ):
                return False
        elif (
            event.startswith("add_") and event.endswith("_span")
        ) or event == "start_trace":
            return False
    return trace_count == 1


def _logger_contains_only_sanitized_aibom_spans(logger: Any) -> bool:
    """Fail closed if the SDK contains an automatic or raw span."""
    try:
        if hasattr(logger, "traces"):
            traces = getattr(logger, "traces")
            return (
                _valid_sdk_logger_envelope(logger)
                and isinstance(traces, Sequence)
                and not isinstance(traces, (str, bytes, bytearray))
                and len(traces) == 1
                and _validate_sdk_node(traces[0], root=True)
            )
        return _validate_fake_logger(logger)
    except Exception:
        _LOGGER.debug("Unable to audit Galileo trace; ingestion rejected")
        return False


class _FlushDispatcher:
    """Bounded, single-worker ingestion queue with an explicit finite drain."""

    def __init__(self, *, capacity: int = _FLUSH_QUEUE_CAPACITY) -> None:
        self._capacity = max(1, int(capacity))
        self._queue: deque[Any] = deque()
        self._pending = 0
        self._closed = False
        self._condition = threading.Condition()
        self._worker: threading.Thread | None = None

    def submit(self, logger: Any) -> bool:
        if not _logger_contains_only_sanitized_aibom_spans(logger):
            _LOGGER.warning("Rejected non-AIBOM or unsanitized Galileo trace")
            _discard_logger(logger)
            return False
        with self._condition:
            if self._closed:
                _LOGGER.warning("Galileo flush dispatcher is closed; trace dropped")
                accepted = False
            elif self._pending >= self._capacity:
                _LOGGER.warning("Galileo flush queue is full; telemetry trace dropped")
                accepted = False
            else:
                self._queue.append(logger)
                self._pending += 1
                accepted = True
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(
                        target=self._run,
                        name="aibom-galileo-flush",
                        daemon=True,
                    )
                    self._worker.start()
                self._condition.notify_all()
        if not accepted:
            _discard_logger(logger)
        return accepted

    def _run(self) -> None:
        while True:
            with self._condition:
                if not self._queue:
                    self._worker = None
                    self._condition.notify_all()
                    return
                logger = self._queue.popleft()
            try:
                # Re-audit immediately before the network boundary in case the
                # logger was mutated while waiting behind another flush.
                if not _disable_agent_control(logger):
                    _LOGGER.warning(
                        "Galileo Agent Control could not be disabled; trace dropped"
                    )
                    _discard_logger(logger)
                elif not _logger_contains_only_sanitized_aibom_spans(logger):
                    _LOGGER.warning("Rejected mutated Galileo trace before flush")
                    _discard_logger(logger)
                else:
                    logger.flush(on_error=lambda _error: None)
            except Exception:
                _LOGGER.debug("Unable to flush Galileo telemetry")
            finally:
                with self._condition:
                    self._pending -= 1
                    self._condition.notify_all()

    def drain(self, timeout_s: float) -> bool:
        """Close ingestion and wait up to one deadline for accepted flushes.

        Once drain begins, new submissions are rejected. If the deadline is
        reached, queued loggers that have not started are discarded. A flush
        already executing inside the SDK cannot be cancelled safely, but it
        remains on the daemon worker and cannot delay process shutdown.
        """
        try:
            timeout = float(timeout_s)
        except (TypeError, ValueError, OverflowError):
            timeout = 0.0
        if not math.isfinite(timeout) or timeout < 0:
            timeout = 0.0
        deadline = time.monotonic() + timeout
        dropped: list[Any] = []
        with self._condition:
            self._closed = True
            while self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    dropped = list(self._queue)
                    self._queue.clear()
                    self._pending -= len(dropped)
                    if dropped:
                        self._condition.notify_all()
                    break
                self._condition.wait(timeout=remaining)
            completed = self._pending == 0 and not dropped
        for logger in dropped:
            _discard_logger(logger)
        return completed


def _parse_enabled(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_sample_rate(value: str | None) -> float:
    if value is None:
        return 1.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return -1.0
    return parsed if math.isfinite(parsed) else -1.0


def _as_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return min(_MAX_SAFE_INTEGER, max(0, int(value)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _duration_ns(seconds: float | int | None) -> int | None:
    if seconds is None or isinstance(seconds, bool):
        return None
    try:
        value = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    if value >= _MAX_SAFE_INTEGER / 1_000_000_000:
        return _MAX_SAFE_INTEGER
    return min(_MAX_SAFE_INTEGER, int(value * 1_000_000_000))


def _enum_value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _allowed_label(value: Any, allowed: frozenset[str]) -> str:
    candidate = _enum_value(value).strip().lower()
    return candidate if candidate in allowed else "other"


def _safe_model_value(value: Any) -> bool:
    """Accept only sentinels or an AIBOM-generated model pseudonym."""
    return isinstance(value, str) and (
        value in {"unknown", "other"}
        or bool(re.fullmatch(r"model_[0-9a-f]{24}", value))
    )


def _safe_model_label(
    value: str | None,
    pseudonymizer: "Pseudonymizer | None" = None,
) -> str:
    """Return a stable HMAC token, never a raw model or deployment name."""
    if not value:
        return "unknown"
    candidate = str(value).strip()
    if candidate.lower() in {"unknown", "other"}:
        return candidate.lower()
    lowered = candidate.lower()
    forbidden = (
        "api_key",
        "apikey",
        "bearer ",
        "private_key",
        "secret",
        "token=",
    )
    if (
        len(candidate) > 96
        or candidate.startswith(("/", "~", "\\"))
        or "://" in candidate
        or bool(re.match(r"^[a-zA-Z]:[\\/]", candidate))
        or ".." in candidate
        or "@" in candidate
        or "\n" in candidate
        or "\r" in candidate
        or lowered.startswith(("sk-", "ghp_", "xox"))
        or any(marker in lowered for marker in forbidden)
    ):
        return "other"
    # Registry-style model identifiers may contain a slash. They are safe only
    # after pseudonymization; absolute/traversing paths were rejected above.
    if not all(ch.isalnum() or ch in "-_.:/" for ch in candidate):
        return "other"
    if pseudonymizer is None:
        return "other"
    return pseudonymizer.token(candidate, prefix="model")


def _safe_version(value: str | None) -> str:
    if not value:
        return "unknown"
    candidate = str(value).strip()
    if len(candidate) > 64 or not all(ch.isalnum() or ch in "-_.+" for ch in candidate):
        return "other"
    return candidate


def _count_map(
    values: Mapping[Any, Any] | None,
    allowed: frozenset[str],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw_key, raw_value in (values or {}).items():
        key = _allowed_label(raw_key, allowed)
        result[key] = result.get(key, 0) + _as_non_negative_int(raw_value)
    return dict(sorted(result.items()))


def _decision_counts(values: Mapping[str, Any] | None) -> dict[str, int]:
    result = {key: 0 for key in sorted(_DECISION_KEYS)}
    for key, value in (values or {}).items():
        normalized = str(key).strip().lower()
        if normalized in _DECISION_KEYS:
            result[normalized] = _as_non_negative_int(value)
    return result


def _decision_breakdown(
    values: Mapping[str, Mapping[Any, Any]] | None,
    allowed: frozenset[str],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {
        key: {} for key in sorted(_COMPONENT_DECISION_KEYS)
    }
    for raw_action, counts in (values or {}).items():
        action = str(raw_action).strip().lower()
        if action in _COMPONENT_DECISION_KEYS and isinstance(counts, Mapping):
            result[action] = _count_map(counts, allowed)
    return result


def _json_payload(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _status_code(status: str) -> int:
    if status in {"success", "cache_hit", "skipped"}:
        return 200
    return 500


def _attempt_span_values(kind: Any, attempt_number: Any) -> dict[str, Any]:
    safe_kind = _allowed_label(kind, _ATTEMPT_KINDS)
    safe_number = _as_non_negative_int(attempt_number)
    payload = _json_payload(
        {
            "attempt_number": safe_number,
            "event": "agentic_attempt",
            "kind": safe_kind,
        }
    )
    return {
        "input": payload,
        "redacted_input": payload,
        "name": f"aibom.agentic.{safe_kind}",
        "metadata": {"attempt_number": safe_number, "kind": safe_kind},
        "tags": ["aibom", "agentic", "sanitized"],
    }


def _call_token(value: Any, pseudonymizer: "Pseudonymizer | None") -> str:
    if pseudonymizer is not None:
        return pseudonymizer.token(value, prefix="call")
    digest = hmac.new(
        _PROCESS_HMAC_KEY, str(value).encode("utf-8", errors="replace"), hashlib.sha256
    ).hexdigest()[:24]
    return f"call_{digest}"


def _llm_span_values(
    *,
    provider: Any,
    model: Any,
    status: Any,
    duration_s: Any,
    prompt_tokens: Any,
    completion_tokens: Any,
    total_tokens: Any,
    cached_tokens: Any,
    schema_valid: Any,
    decisions: Mapping[str, Any] | None,
    pseudonymizer: "Pseudonymizer | None",
    call_id: Any = "aggregate",
    sequence: Any = 0,
    created_at: datetime | None = None,
    mode: Any = "aggregate",
    decision_carrier: Any = True,
    schema_expected: Any = True,
) -> tuple[dict[str, Any], bool]:
    safe_provider = _allowed_label(provider, _PROVIDERS)
    safe_model = _safe_model_label(model, pseudonymizer)
    safe_status = _allowed_label(status, _STATUSES)
    safe_decisions = _decision_counts(decisions)
    safe_mode = str(mode) if str(mode) in {"aggregate", "per_call"} else "aggregate"
    safe_sequence = _as_non_negative_int(sequence)
    safe_call_id = _call_token(call_id, pseudonymizer)
    token_usage_missing = not any(
        _as_non_negative_int(value)
        for value in (
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cached_tokens,
        )
    )
    encoded_input = _json_payload(
        {
            "call_id": safe_call_id,
            "event": "agentic_llm_call",
            "mode": safe_mode,
            "provider": safe_provider,
            "schema_expected": bool(schema_expected),
            "sequence": safe_sequence,
        }
    )
    encoded_output = _json_payload(
        {
            "decision_carrier": bool(decision_carrier),
            "decisions": safe_decisions,
            "schema_valid": bool(schema_valid),
            "status": safe_status,
            "token_usage_missing": token_usage_missing,
        }
    )
    values = {
        "input": encoded_input,
        "redacted_input": encoded_input,
        "output": encoded_output,
        "redacted_output": encoded_output,
        "model": safe_model,
        "name": "aibom.agentic.llm",
        "created_at": created_at,
        "duration_ns": _duration_ns(duration_s),
        "metadata": {
            "call_id": safe_call_id,
            "cached_tokens": _as_non_negative_int(cached_tokens),
            "decision_carrier": bool(decision_carrier),
            "mode": safe_mode,
            "provider": safe_provider,
            "schema_valid": bool(schema_valid),
            "sequence": safe_sequence,
            "status": safe_status,
            "token_usage_missing": token_usage_missing,
        },
        "tags": ["aibom", "agentic", "sanitized"],
        "num_input_tokens": (
            None if token_usage_missing else _as_non_negative_int(prompt_tokens)
        ),
        "num_output_tokens": (
            None if token_usage_missing else _as_non_negative_int(completion_tokens)
        ),
        "total_tokens": (
            None if token_usage_missing else _as_non_negative_int(total_tokens)
        ),
        "status_code": _status_code(safe_status),
        "step_number": safe_sequence or None,
    }
    force_retain = (
        token_usage_missing
        or not bool(schema_valid)
        or safe_status in _TERMINAL_RETENTION_STATUSES
        or safe_decisions["discovered"] > 0
        or safe_decisions["risk_findings"] > 0
        or safe_decisions["degraded"] > 0
    )
    return values, force_retain


def _tool_span_values(
    stats: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], bool]:
    normalized: dict[str, dict[str, float | int]] = {}
    for raw_name, raw_stats in (stats or {}).items():
        name = str(raw_name) if str(raw_name) in _TOOL_NAMES else "other"
        target = normalized.setdefault(
            name,
            {"calls": 0, "errors": 0, "guard_denials": 0, "total_s": 0.0},
        )
        target["calls"] = int(target["calls"]) + _as_non_negative_int(
            raw_stats.get("calls", 0)
        )
        target["errors"] = int(target["errors"]) + _as_non_negative_int(
            raw_stats.get("errors", 0)
        )
        target["guard_denials"] = int(target["guard_denials"]) + _as_non_negative_int(
            raw_stats.get("guard_denials", 0)
        )
        try:
            elapsed = float(raw_stats.get("total_s", 0.0))
        except (TypeError, ValueError, OverflowError):
            elapsed = 0.0
        if not math.isfinite(elapsed) or elapsed < 0:
            elapsed = 0.0
        target["total_s"] = float(target["total_s"]) + elapsed

    values: list[dict[str, Any]] = []
    force_retain = False
    for name, aggregate in sorted(normalized.items()):
        calls = _as_non_negative_int(aggregate["calls"])
        errors = _as_non_negative_int(aggregate["errors"])
        guard_denials = _as_non_negative_int(aggregate["guard_denials"])
        elapsed = float(aggregate["total_s"])
        encoded_input = _json_payload(
            {"calls": calls, "event": "tool_aggregate", "mode": "aggregate"}
        )
        encoded_output = _json_payload(
            {"errors": errors, "guard_denials": guard_denials}
        )
        values.append(
            {
                "input": encoded_input,
                "redacted_input": encoded_input,
                "output": encoded_output,
                "redacted_output": encoded_output,
                "name": f"aibom.tool.{name}",
                "duration_ns": _duration_ns(elapsed),
                "metadata": {
                    "calls": calls,
                    "errors": errors,
                    "guard_denials": guard_denials,
                },
                "tags": ["aibom", "agentic", "tool-aggregate", "sanitized"],
                "status_code": 500 if errors or guard_denials else 200,
            }
        )
        force_retain = force_retain or bool(errors or guard_denials)
    return values, force_retain


def _sanitized_tool_stats(
    stats: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for raw_name, raw_stats in (stats or {}).items():
        name = str(raw_name) if str(raw_name) in _TOOL_NAMES else "other"
        target = result.setdefault(
            name,
            {"calls": 0, "duration_ns": 0, "errors": 0, "guard_denials": 0},
        )
        target["calls"] += _as_non_negative_int(raw_stats.get("calls", 0))
        target["errors"] += _as_non_negative_int(raw_stats.get("errors", 0))
        target["guard_denials"] += _as_non_negative_int(
            raw_stats.get("guard_denials", 0)
        )
        target["duration_ns"] = min(
            _MAX_SAFE_INTEGER,
            target["duration_ns"] + (_duration_ns(raw_stats.get("total_s", 0.0)) or 0),
        )
    return dict(sorted(result.items()))


def _tool_call_span_values(
    *,
    name: Any,
    call_id: Any,
    sequence: Any,
    created_at: datetime | None,
    duration_s: Any,
    status: Any,
    pseudonymizer: "Pseudonymizer | None",
) -> tuple[dict[str, Any], bool]:
    safe_name = str(name) if str(name) in _TOOL_NAMES else "other"
    safe_status = _allowed_label(status, _STATUSES)
    safe_sequence = _as_non_negative_int(sequence)
    safe_call_id = _call_token(call_id, pseudonymizer)
    errors = 0 if safe_status == "success" else 1
    encoded_input = _json_payload(
        {
            "call_id": safe_call_id,
            "event": "tool_call",
            "mode": "per_call",
            "sequence": safe_sequence,
        }
    )
    encoded_output = _json_payload({"errors": errors, "guard_denials": 0})
    values = {
        "input": encoded_input,
        "redacted_input": encoded_input,
        "output": encoded_output,
        "redacted_output": encoded_output,
        "name": f"aibom.tool.{safe_name}",
        "created_at": created_at,
        "duration_ns": _duration_ns(duration_s),
        "metadata": {
            "call_id": safe_call_id,
            "calls": 1,
            "errors": errors,
            "guard_denials": 0,
            "mode": "per_call",
            "sequence": safe_sequence,
        },
        "tags": ["aibom", "agentic", "sanitized"],
        "status_code": _status_code(safe_status),
        "step_number": safe_sequence or None,
        "tool_call_id": safe_call_id,
    }
    return values, bool(errors)


class Pseudonymizer:
    """Create stable keyed pseudonyms for telemetry correlation."""

    __slots__ = ("_key",)

    def __init__(self, key: str | bytes | None = None) -> None:
        if key is None:
            configured = os.getenv(_HMAC_KEY_ENV)
            resolved = configured.encode("utf-8") if configured else _PROCESS_HMAC_KEY
        elif isinstance(key, str):
            resolved = key.encode("utf-8")
        else:
            resolved = bytes(key)
        self._key = resolved or _PROCESS_HMAC_KEY

    def token(self, value: Any, *, prefix: str = "id") -> str:
        safe_prefix = (
            prefix
            if prefix
            in {
                "batch",
                "call",
                "chain",
                "component",
                "model",
                "session",
                "source",
            }
            else "id"
        )
        digest = hmac.new(
            self._key,
            f"token:{safe_prefix}:{value}".encode("utf-8", errors="replace"),
            hashlib.sha256,
        ).hexdigest()[:24]
        return f"{safe_prefix}_{digest}"

    def selected(self, value: Any, sample_rate: float) -> bool:
        if sample_rate <= 0:
            return False
        if sample_rate >= 1:
            return True
        digest = hmac.new(
            self._key,
            f"sample:{value}".encode("utf-8", errors="replace"),
            hashlib.sha256,
        ).digest()
        fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return fraction < sample_rate


@dataclass(frozen=True, slots=True)
class GalileoTelemetryConfig:
    """Non-secret configuration for optional Galileo telemetry."""

    enabled: bool = False
    sample_rate: float = 1.0
    project: str = ""
    log_stream: str = ""

    @property
    def configured(self) -> bool:
        return (
            self.enabled
            and 0.0 <= self.sample_rate <= 1.0
            and bool(self.project.strip())
            and bool(self.log_stream.strip())
        )

    @classmethod
    def from_env(
        cls,
        *,
        enabled: bool | None = None,
        sample_rate: float | None = None,
    ) -> "GalileoTelemetryConfig":
        """Build configuration without retaining the Galileo API key."""
        resolved_enabled = (
            _parse_enabled(os.getenv("AIBOM_GALILEO_ENABLED"))
            if enabled is None
            else bool(enabled)
        )
        resolved_rate = (
            _parse_sample_rate(os.getenv("AIBOM_GALILEO_SAMPLE_RATE"))
            if sample_rate is None
            else float(sample_rate)
        )
        return cls(
            enabled=resolved_enabled,
            sample_rate=resolved_rate,
            project=os.getenv("GALILEO_PROJECT", "").strip(),
            log_stream=os.getenv("GALILEO_LOG_STREAM", "").strip(),
        )


def _normalized_exact_https_origin(candidate: str, expected_host: str) -> str | None:
    """Canonicalize an exact HTTPS origin with no URL payload."""
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        parsed = urlparse(candidate)
        hostname = (parsed.hostname or "").rstrip(".").lower()
        # Accessing ``port`` validates malformed/non-numeric port syntax.
        _ = parsed.port
    except (TypeError, ValueError):
        return None
    if (
        parsed.scheme.lower() != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        return None
    port = parsed.port
    if hostname != expected_host or port not in {None, 443}:
        return None
    return f"https://{expected_host}"


def _normalized_console_origin(value: str | None = None) -> str | None:
    """Return the explicitly approved hosted console origin, if configured."""
    candidate = os.getenv("GALILEO_CONSOLE_URL", "") if value is None else value
    normalized = _normalized_exact_https_origin(candidate, _HOSTED_CONSOLE_HOST)
    if normalized is None or not _parse_enabled(os.getenv(_ALLOW_PUBLIC_CLOUD_ENV)):
        return None
    return normalized


def _normalized_api_origin(value: str | None = None) -> str | None:
    """Return the hosted API origin, rejecting every explicit override."""
    if value is None:
        candidate = os.getenv("GALILEO_API_URL", "").strip() or _HOSTED_API_ORIGIN
    else:
        candidate = value
    return _normalized_exact_https_origin(candidate, _HOSTED_API_HOST)


def _console_destination_configured(value: str | None = None) -> bool:
    return _normalized_console_origin(value) is not None


def _api_destination_configured(value: str | None = None) -> bool:
    return _normalized_api_origin(value) is not None


def _tls_verification_configured() -> bool:
    """Reject SDK configuration that explicitly disables TLS verification."""
    candidate = os.getenv("GALILEO_SSL_CONTEXT", "").strip().casefold()
    return not candidate or candidate in {"1", "true", "yes", "on", "t", "y"}


def _sdk_tls_verification_enabled(value: Any) -> bool:
    """Accept the SDK default or a hostname-checking, validating SSL context."""
    if value is True:
        return True
    if isinstance(value, ssl.SSLContext):
        return value.check_hostname and value.verify_mode == ssl.CERT_REQUIRED
    return False


def _loaded_galileo_destination_matches() -> bool:
    """Verify a pre-existing SDK singleton cannot override the allowed origin."""
    expected = _normalized_console_origin()
    expected_api = _normalized_api_origin()
    if expected is None or expected_api is None or not _tls_verification_configured():
        return False
    try:
        config_module = importlib.import_module("galileo.config")
        config_class = getattr(config_module, "GalileoPythonConfig")
        instance = getattr(config_class, "_instance", None)
    except Exception:
        return False
    if instance is None:
        return True
    actual = str(getattr(instance, "console_url", "")).strip()
    actual_api = str(getattr(instance, "api_url", "")).strip()
    return (
        _normalized_console_origin(actual) == expected
        and _normalized_api_origin(actual_api) == expected_api
        and _sdk_tls_verification_enabled(getattr(instance, "ssl_context", True))
    )


def _default_logger_factory(project: str, log_stream: str) -> Any:
    """Construct a logger only after resolving already-provisioned resources.

    GalileoLogger's name-based constructor creates a missing project or log
    stream. Production observability must never mutate cluster configuration,
    so resolution is explicit and the logger is always constructed by IDs.
    """
    if (
        not _console_destination_configured()
        or not _api_destination_configured()
        or not _tls_verification_configured()
    ):
        raise RuntimeError(
            "Sanitized telemetry requires the approved hosted Galileo console "
            "and API origins"
        )
    if not _loaded_galileo_destination_matches():
        raise RuntimeError(
            "Loaded Galileo SDK configuration does not match the explicit "
            "GALILEO_CONSOLE_URL; restart the process before enabling telemetry"
        )
    projects_module = importlib.import_module("galileo.projects")
    log_streams_module = importlib.import_module("galileo.log_streams")
    project_obj = getattr(projects_module, "Projects")().get(name=project)
    project_id = getattr(project_obj, "id", None)
    if not isinstance(project_id, str) or not project_id:
        raise LookupError(f"Galileo project is not provisioned: {project!r}")
    log_stream_obj = getattr(log_streams_module, "LogStreams")().get(
        name=log_stream,
        project_id=project_id,
    )
    log_stream_id = getattr(log_stream_obj, "id", None)
    if not isinstance(log_stream_id, str) or not log_stream_id:
        raise LookupError(f"Galileo log stream is not provisioned: {log_stream!r}")
    return _default_logger_id_factory(project_id, log_stream_id)


def _default_logger_id_factory(project_id: str, log_stream_id: str) -> Any:
    if (
        not _console_destination_configured()
        or not _api_destination_configured()
        or not _tls_verification_configured()
        or not _loaded_galileo_destination_matches()
    ):
        raise RuntimeError("Galileo SDK destination changed after resource resolution")
    module = importlib.import_module("galileo")
    logger_class = getattr(module, "GalileoLogger")
    return logger_class(
        project_id=project_id,
        log_stream_id=log_stream_id,
        mode="batch",
    )


class AgenticTelemetry:
    """Factory and session coordinator for independent batch loggers."""

    def __init__(
        self,
        config: GalileoTelemetryConfig,
        *,
        logger_factory: LoggerFactory | None,
        pseudonymizer: Pseudonymizer,
        session_external_id: str | None = None,
        galileo_session_id: str | None = None,
    ) -> None:
        self.config = config
        self._logger_factory = logger_factory
        self._pseudonymizer = pseudonymizer
        self._session_external_token = (
            pseudonymizer.token(session_external_id, prefix="session")
            if session_external_id
            else None
        )
        self._galileo_session_id = (
            galileo_session_id if _is_uuid4(galileo_session_id) else None
        )
        self._session_attempted = bool(galileo_session_id)
        # Reject an explicitly supplied non-v4 session identifier before any
        # logger or network setup. Galileo ingestion requires UUIDv4 session
        # IDs, and silently creating a replacement session would break the
        # caller's intended correlation boundary.
        self._factory_failed = bool(
            galileo_session_id and not _is_uuid4(galileo_session_id)
        )
        self._resolved_project_id: str | None = None
        self._resolved_log_stream_id: str | None = None
        self._lock = threading.Lock()
        self._flush_dispatcher = _FlushDispatcher()

    @property
    def enabled(self) -> bool:
        return (
            self.config.configured
            and self._logger_factory is not None
            and not self._factory_failed
        )

    def _new_logger(self) -> Any | None:
        if not self.enabled or self._logger_factory is None:
            return None
        factory = self._logger_factory

        def _construct() -> Any:
            if (
                factory is _default_logger_factory
                and self._resolved_project_id
                and self._resolved_log_stream_id
            ):
                logger = _default_logger_id_factory(
                    self._resolved_project_id,
                    self._resolved_log_stream_id,
                )
            else:
                logger = factory(self.config.project, self.config.log_stream)
            # This must happen in the worker as well as the caller: if logger
            # construction exceeds the budget and later finishes, its SDK
            # atexit handler must still not delay process shutdown.
            if not _harden_logger(logger):
                _discard_logger(logger)
                raise RuntimeError("Galileo Agent Control bridge remained enabled")
            return logger

        try:
            completed, logger = _bounded_call(_construct)
            if not completed or logger is None:
                with self._lock:
                    self._factory_failed = True
                _LOGGER.warning(
                    "Galileo logger initialization exceeded its budget; "
                    "telemetry disabled"
                )
                return None
            if factory is _default_logger_factory:
                project_id = getattr(logger, "project_id", None)
                log_stream_id = getattr(logger, "log_stream_id", None)
                if isinstance(project_id, str) and project_id:
                    self._resolved_project_id = project_id
                if isinstance(log_stream_id, str) and log_stream_id:
                    self._resolved_log_stream_id = log_stream_id
            return logger
        except Exception:  # Galileo must never affect scanning.
            with self._lock:
                self._factory_failed = True
            _LOGGER.warning("Galileo logger initialization failed; telemetry disabled")
            return None

    def _attach_session(self, logger: Any) -> bool:
        if not self._galileo_session_id and not self._session_external_token:
            return True
        with self._lock:
            session_id = self._galileo_session_id
            if session_id:
                if not _is_uuid4(session_id):
                    self._factory_failed = True
                    return False
                try:
                    logger.set_session(session_id)
                except Exception:
                    _LOGGER.debug("Unable to attach Galileo session")
                    self._factory_failed = True
                    return False
                return True
            if self._session_attempted:
                return False
            self._session_attempted = True
            try:
                # Timestamp the display name so repeated scans are visually
                # distinguishable in the Galileo Sessions list. Correlation
                # still rides on the content-free external_id token, and the
                # UTC time here is not sensitive telemetry.
                session_name = "aibom-agentic-scan-" + datetime.now(
                    timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
                completed, session_id = _bounded_call(
                    lambda: logger.start_session(
                        name=session_name,
                        external_id=self._session_external_token,
                        metadata={"component": "agentic-classifier"},
                    )
                )
                if completed and _is_uuid4(session_id):
                    self._galileo_session_id = session_id
                    return True
                _LOGGER.debug(
                    "Galileo session creation exceeded its budget; trace skipped"
                )
                self._factory_failed = True
                return False
            except Exception:
                _LOGGER.debug("Unable to create Galileo session")
                self._factory_failed = True
                return False

    def _sampled(self, identifier: str) -> bool:
        # Do not include the invocation/session identifier: the same source or
        # batch must make the same sampling decision across repeated scans.
        return self._pseudonymizer.selected(
            f"aibom-agentic:{identifier}", self.config.sample_rate
        )

    def _start_prepared_batch(
        self,
        *,
        encoded: str,
        metadata: Mapping[str, Any],
        batch_token: str,
    ) -> Any | None:
        logger = self._new_logger()
        if logger is None or not self._attach_session(logger):
            if logger is not None:
                _discard_logger(logger)
            return None
        try:
            trace = logger.start_trace(
                input=encoded,
                redacted_input=encoded,
                name="aibom.agentic.batch",
                metadata=dict(metadata),
                tags=["aibom", "agentic", "sanitized"],
                external_id=batch_token,
            )
        except Exception:
            _LOGGER.debug("Unable to start Galileo batch trace")
            _discard_logger(logger)
            return None
        if trace is None:
            _discard_logger(logger)
            return None
        return logger

    def start_batch(
        self,
        *,
        batch_id: str,
        sample_key: str | None = None,
        source_id: str | None = None,
        attempt_kind: str = "initial",
        tier: str = "unknown",
        batch_num: int | None = None,
        total_batches: int | None = None,
        component_ids: Sequence[str] = (),
        component_type_counts: Mapping[Any, Any] | None = None,
        language_counts: Mapping[Any, Any] | None = None,
        provider: str = "unknown",
        model: str = "unknown",
        analyzer_version: str | None = None,
        prompt_version: str | None = None,
        schema_version: str | None = None,
        cache_hit: bool = False,
    ) -> "BatchTrace":
        """Start one sanitized trace backed by a fresh Galileo logger."""
        if not self.enabled:
            return BatchTrace.noop()

        safe_tier = _allowed_label(tier, _TIERS)
        safe_provider = _allowed_label(provider, _PROVIDERS)
        safe_attempt_kind = _allowed_label(attempt_kind, _ATTEMPT_KINDS)
        # Scope every correlation identifier to the source. Repository-relative
        # component IDs and batch ordinals can repeat across a multi-source CLI
        # invocation; without this namespace their traces become ambiguous.
        # ``sample_key`` remains a backward-compatible source hint for direct
        # callers, while AIBOM's pipeline always provides ``source_id``.
        source_scope = source_id if source_id is not None else sample_key
        source_scope = source_scope or "unknown"
        source_token = self._pseudonymizer.token(source_scope, prefix="source")
        batch_token = self._pseudonymizer.token(
            f"{source_scope}\0{batch_id}", prefix="batch"
        )
        identifiers = [
            self._pseudonymizer.token(f"{source_scope}\0{item}", prefix="component")
            for item in list(component_ids)[:128]
        ]
        # A retry may split an initial batch and a fallback may merge candidates
        # from several failed batches. Per-component chain IDs remain stable
        # through those regroupings and provide lossless attempt lineage without
        # exposing component names, paths, or repository identities.
        decision_chain_ids = [
            self._pseudonymizer.token(f"{source_scope}\0{item}", prefix="chain")
            for item in list(component_ids)[:128]
        ]
        payload = {
            "attempt_kind": safe_attempt_kind,
            "batch_id": batch_token,
            "batch_num": _as_non_negative_int(batch_num),
            "batch_size": len(component_ids),
            "cache_hit": bool(cache_hit),
            "component_ids": identifiers,
            "component_type_counts": _count_map(
                component_type_counts, _COMPONENT_TYPES
            ),
            "decision_chain_ids": decision_chain_ids,
            "event": "agentic_batch",
            "language_counts": _count_map(language_counts, _LANGUAGES),
            "source_id": source_token,
            "tier": safe_tier,
            "total_batches": _as_non_negative_int(total_batches),
        }
        encoded = _json_payload(payload)
        metadata = {
            "analyzer_version": _safe_version(analyzer_version),
            "attempt_kind": safe_attempt_kind,
            "batch_size": len(component_ids),
            "cache_hit": bool(cache_hit),
            "model": _safe_model_label(model, self._pseudonymizer),
            "prompt_version": _safe_version(prompt_version),
            "provider": safe_provider,
            "schema_version": _safe_version(schema_version),
            "source_id": source_token,
            "tier": safe_tier,
        }
        sampling_identity = (
            f"source:{source_scope}:tier:{safe_tier}:"
            f"batch:{_as_non_negative_int(batch_num)}"
            if source_id is not None or sample_key is not None
            else f"batch:{batch_id}"
        )
        if not self._sampled(sampling_identity):
            # Capture only sanitized aggregate JSON. If the batch later ends in
            # an operationally significant state, finish() can emit a minimal
            # trace without ever buffering prompts, source, or tool content.
            def _deferred_start(
                *,
                safe_encoded: str = encoded,
                safe_metadata: Mapping[str, Any] = dict(metadata),
                safe_batch_token: str = batch_token,
            ) -> Any | None:
                return self._start_prepared_batch(
                    encoded=safe_encoded,
                    metadata=safe_metadata,
                    batch_token=safe_batch_token,
                )

            return BatchTrace(
                deferred_start=_deferred_start,
                flush_submit=self._flush_dispatcher.submit,
                pseudonymizer=self._pseudonymizer,
            )

        logger = self._start_prepared_batch(
            encoded=encoded,
            metadata=metadata,
            batch_token=batch_token,
        )
        if logger is None:
            return BatchTrace.noop()
        return BatchTrace(
            logger=logger,
            flush_submit=self._flush_dispatcher.submit,
            pseudonymizer=self._pseudonymizer,
        )

    def record_summary(
        self,
        *,
        source_id: str,
        source_kind: str = "unknown",
        status: str = "success",
        candidate_count: int = 0,
        candidate_count_available: bool = False,
        agentic_component_count: int = 0,
        final_component_count: int = 0,
        degraded_candidate_count: int = 0,
        duration_s: float | None = None,
        created_at: datetime | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        decisions: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit one sanitized source summary without affecting the scan."""
        try:
            self._record_summary(
                source_id=source_id,
                source_kind=source_kind,
                status=status,
                candidate_count=candidate_count,
                candidate_count_available=candidate_count_available,
                agentic_component_count=agentic_component_count,
                final_component_count=final_component_count,
                degraded_candidate_count=degraded_candidate_count,
                duration_s=duration_s,
                created_at=created_at,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                decisions=decisions,
            )
        except Exception:
            _LOGGER.debug("Unable to prepare Galileo source summary")

    def _record_summary(
        self,
        *,
        source_id: str,
        source_kind: str = "unknown",
        status: str = "success",
        candidate_count: int = 0,
        candidate_count_available: bool = False,
        agentic_component_count: int = 0,
        final_component_count: int = 0,
        degraded_candidate_count: int = 0,
        duration_s: float | None = None,
        created_at: datetime | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        decisions: Mapping[str, Any] | None = None,
    ) -> None:
        safe_status = _allowed_label(status, _STATUSES)
        force_retain = (
            safe_status in _TERMINAL_RETENTION_STATUSES
            or _as_non_negative_int(degraded_candidate_count) > 0
        )
        if not self.enabled or (
            not force_retain and not self._sampled(f"summary:{source_id}")
        ):
            return
        logger = self._new_logger()
        if logger is None:
            return
        if not self._attach_session(logger):
            _discard_logger(logger)
            return
        source_token = self._pseudonymizer.token(source_id, prefix="source")
        safe_kind = _allowed_label(source_kind, _SOURCE_KINDS)
        payload = {
            "candidate_count": _as_non_negative_int(candidate_count),
            "candidate_count_available": bool(candidate_count_available),
            "event": "agentic_source_summary",
            "source_id": source_token,
            "source_kind": safe_kind,
        }
        result = {
            "agentic_component_count": _as_non_negative_int(agentic_component_count),
            "cached_tokens": _as_non_negative_int(cached_tokens),
            "completion_tokens": _as_non_negative_int(completion_tokens),
            "decision_boundary": "agentic_stage_output",
            "decisions": _decision_counts(decisions),
            "degraded_candidate_count": _as_non_negative_int(degraded_candidate_count),
            "final_component_count": _as_non_negative_int(final_component_count),
            "prompt_tokens": _as_non_negative_int(prompt_tokens),
            "status": safe_status,
        }
        encoded_input = _json_payload(payload)
        encoded_output = _json_payload(result)
        try:
            trace = logger.start_trace(
                input=encoded_input,
                redacted_input=encoded_input,
                name="aibom.agentic.source_summary",
                created_at=created_at,
                metadata={"source_kind": safe_kind, "status": safe_status},
                tags=["aibom", "agentic", "summary", "sanitized"],
                external_id=source_token,
            )
            if trace is None:
                return
            logger.conclude(
                output=encoded_output,
                redacted_output=encoded_output,
                duration_ns=_duration_ns(duration_s),
                status_code=_status_code(safe_status),
                conclude_all=True,
            )
            self._flush_dispatcher.submit(logger)
        except Exception:
            _LOGGER.debug("Unable to emit Galileo source summary")
            _discard_logger(logger)

    def drain(self, timeout_s: float = 2.0) -> bool:
        """Wait up to ``timeout_s`` for accepted telemetry flushes to finish."""
        return self._flush_dispatcher.drain(timeout_s)


class BatchTrace:
    """A fail-open handle for one agentic batch trace."""

    def __init__(
        self,
        *,
        logger: Any | None = None,
        flush_submit: FlushSubmitter | None = None,
        deferred_start: DeferredTraceStarter | None = None,
        pseudonymizer: Pseudonymizer | None = None,
    ) -> None:
        self._logger = logger
        self._flush_submit = flush_submit
        self._deferred_start = deferred_start
        self._pseudonymizer = pseudonymizer
        self._buffered_attempts: list[_BufferedAttempt] = []
        self._finished = False

    @classmethod
    def noop(cls) -> "BatchTrace":
        return cls()

    @property
    def active(self) -> bool:
        return (
            self._logger is not None or self._deferred_start is not None
        ) and not self._finished

    def start_attempt(
        self,
        *,
        kind: str = "initial",
        attempt_number: int = 1,
    ) -> "AttemptTrace":
        logger = self._logger
        if self._finished:
            return AttemptTrace.noop()
        values = _attempt_span_values(kind, attempt_number)
        if logger is None:
            if (
                self._deferred_start is None
                or len(self._buffered_attempts) >= _MAX_BUFFERED_ATTEMPTS
            ):
                return AttemptTrace.noop()
            buffered = _BufferedAttempt(
                input=values["input"],
                metadata=dict(values["metadata"]),
                name=values["name"],
            )
            self._buffered_attempts.append(buffered)
            return AttemptTrace(
                buffer=buffered,
                pseudonymizer=self._pseudonymizer,
            )
        try:
            span = logger.add_workflow_span(**values)
        except Exception:
            _LOGGER.debug("Unable to start Galileo attempt span")
            return AttemptTrace.noop()
        if span is None:
            return AttemptTrace.noop()
        return AttemptTrace(logger=logger, pseudonymizer=self._pseudonymizer)

    def finish(
        self,
        *,
        status: str = "success",
        duration_s: float | None = None,
        decisions: Mapping[str, Any] | None = None,
        decisions_by_component_type: Mapping[str, Mapping[Any, Any]] | None = None,
        decisions_by_confidence: Mapping[str, Mapping[Any, Any]] | None = None,
        decisions_by_language: Mapping[str, Mapping[Any, Any]] | None = None,
        degraded_candidates: int = 0,
        failure_hint: str = "",
        schema_valid: bool = True,
        middleware_guard_triggered: bool = False,
    ) -> None:
        logger = self._logger
        if self._finished:
            return
        self._finished = True
        safe_status = _allowed_label(status, _STATUSES)
        safe_hint = _allowed_label(failure_hint, _FAILURE_HINTS)
        safe_decisions = _decision_counts(decisions)
        safe_type_breakdown = _decision_breakdown(
            decisions_by_component_type, _COMPONENT_TYPES
        )
        safe_confidence_breakdown = _decision_breakdown(
            decisions_by_confidence, _CONFIDENCE_BUCKETS
        )
        safe_language_breakdown = _decision_breakdown(decisions_by_language, _LANGUAGES)
        force_retain = (
            safe_status in _TERMINAL_RETENTION_STATUSES
            or safe_hint in _TERMINAL_FAILURE_HINTS
            or _as_non_negative_int(degraded_candidates) > 0
            or not bool(schema_valid)
            or bool(middleware_guard_triggered)
            or safe_decisions["discovered"] > 0
            or safe_decisions["risk_findings"] > 0
            or any(item.force_retain for item in self._buffered_attempts)
        )
        if logger is None and force_retain and self._deferred_start is not None:
            try:
                logger = self._deferred_start()
                self._logger = logger
                if logger is not None and not self._replay_buffered_attempts(logger):
                    _discard_logger(logger)
                    logger = None
                    self._logger = None
            except Exception:
                _LOGGER.debug("Unable to start retained terminal Galileo trace")
                logger = None
        # Release the closure as soon as the terminal decision is known. It
        # contains only sanitized data, but there is no reason to retain it.
        self._deferred_start = None
        if logger is None:
            self._buffered_attempts.clear()
            return
        output = _json_payload(
            {
                "decisions": safe_decisions,
                "decisions_by_component_type": safe_type_breakdown,
                "decisions_by_confidence": safe_confidence_breakdown,
                "decisions_by_language": safe_language_breakdown,
                "degraded_candidates": _as_non_negative_int(degraded_candidates),
                "failure_hint": safe_hint,
                "middleware_guard_triggered": bool(middleware_guard_triggered),
                "schema_valid": bool(schema_valid),
                "status": safe_status,
            }
        )
        try:
            logger.conclude(
                output=output,
                redacted_output=output,
                duration_ns=_duration_ns(duration_s),
                status_code=_status_code(safe_status),
                conclude_all=True,
            )
            if self._flush_submit is not None:
                self._flush_submit(logger)
            else:
                _discard_logger(logger)
        except Exception:
            _LOGGER.debug("Unable to conclude Galileo batch trace")
            _discard_logger(logger)

    def _replay_buffered_attempts(self, logger: Any) -> bool:
        """Materialize sanitized deferred spans under their workflow parents."""
        try:
            for attempt in self._buffered_attempts:
                workflow = logger.add_workflow_span(
                    input=attempt.input,
                    redacted_input=attempt.input,
                    name=attempt.name,
                    metadata=dict(attempt.metadata),
                    tags=["aibom", "agentic", "sanitized"],
                )
                if workflow is None:
                    return False
                for kind, values in attempt.child_spans:
                    add_span = (
                        logger.add_llm_span if kind == "llm" else logger.add_tool_span
                    )
                    if add_span(**values) is None:
                        return False
                output = attempt.output or _json_payload(
                    {
                        "blocked_actions": _decision_counts(None),
                        "final_actions": _decision_counts(None),
                        "raw_actions": _decision_counts(None),
                        "recovered": False,
                        "status": "unknown",
                        "tool_stats": {},
                    }
                )
                logger.conclude(
                    output=output,
                    redacted_output=output,
                    duration_ns=attempt.duration_ns,
                    status_code=attempt.status_code,
                )
            return True
        except Exception:
            _LOGGER.debug("Unable to replay retained Galileo attempt spans")
            return False
        finally:
            self._buffered_attempts.clear()


class AttemptTrace:
    """A nested attempt workflow containing sanitized LLM and tool spans."""

    def __init__(
        self,
        *,
        logger: Any | None = None,
        buffer: _BufferedAttempt | None = None,
        pseudonymizer: Pseudonymizer | None = None,
    ) -> None:
        self._logger = logger
        self._buffer = buffer
        self._pseudonymizer = pseudonymizer
        self._finished = False

    @classmethod
    def noop(cls) -> "AttemptTrace":
        return cls()

    @property
    def active(self) -> bool:
        return (
            self._logger is not None or self._buffer is not None
        ) and not self._finished

    def record_llm(
        self,
        *,
        provider: str = "unknown",
        model: str = "unknown",
        status: str = "success",
        duration_s: float | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int = 0,
        schema_valid: bool = True,
        decisions: Mapping[str, Any] | None = None,
        call_id: str = "aggregate",
        sequence: int = 0,
        created_at: datetime | None = None,
        mode: Literal["aggregate", "per_call"] = "aggregate",
        decision_carrier: bool = True,
        schema_expected: bool = True,
    ) -> None:
        logger = self._logger
        buffer = self._buffer
        if (logger is None and buffer is None) or self._finished:
            return
        try:
            values, force_retain = _llm_span_values(
                provider=provider,
                model=model,
                status=status,
                duration_s=duration_s,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                schema_valid=schema_valid,
                decisions=decisions,
                pseudonymizer=self._pseudonymizer,
                call_id=call_id,
                sequence=sequence,
                created_at=created_at,
                mode=mode,
                decision_carrier=decision_carrier,
                schema_expected=schema_expected,
            )
        except Exception:
            _LOGGER.debug("Unable to prepare Galileo LLM span")
            return
        if buffer is not None:
            llm_count = sum(1 for kind, _ in buffer.child_spans if kind == "llm")
            if llm_count < _MAX_BUFFERED_LLM_SPANS:
                buffer.child_spans.append(("llm", values))
            elif bool(schema_expected):
                # A long agent loop can exceed the deferred-trace cap. Always
                # retain the terminal/schema-bearing model call: it owns the
                # raw decision counts, schema result, and missing-token alert.
                # Drop the oldest LLM entry and append the terminal call so the
                # remaining child order is still chronological (sequence gaps
                # explicitly show that bounded truncation occurred).
                for index, (kind, _) in enumerate(buffer.child_spans):
                    if kind == "llm":
                        del buffer.child_spans[index]
                        buffer.child_spans.append(("llm", values))
                        break
            buffer.force_retain = buffer.force_retain or force_retain
            return
        assert logger is not None
        try:
            logger.add_llm_span(**values)
        except Exception:
            _LOGGER.debug("Unable to emit Galileo LLM span")

    def record_tools(self, stats: Mapping[str, Mapping[str, Any]] | None) -> None:
        """Record aggregate fallback stats when call callbacks were unavailable."""
        logger = self._logger
        buffer = self._buffer
        if (logger is None and buffer is None) or self._finished:
            return
        try:
            span_values, force_retain = _tool_span_values(stats)
        except Exception:
            _LOGGER.debug("Unable to prepare Galileo tool spans")
            return
        if buffer is not None:
            tool_count = sum(1 for kind, _ in buffer.child_spans if kind == "tool")
            available = max(0, _MAX_BUFFERED_TOOL_SPANS - tool_count)
            buffer.child_spans.extend(
                ("tool", values) for values in span_values[:available]
            )
            buffer.force_retain = buffer.force_retain or force_retain
            return
        assert logger is not None
        for values in span_values:
            try:
                logger.add_tool_span(**values)
            except Exception:
                _LOGGER.debug("Unable to emit Galileo tool span")

    def record_tool_call(
        self,
        *,
        name: str,
        call_id: str,
        sequence: int,
        created_at: datetime,
        duration_s: float,
        status: str = "success",
    ) -> None:
        """Record one content-free tool invocation in trajectory order."""

        logger = self._logger
        buffer = self._buffer
        if (logger is None and buffer is None) or self._finished:
            return
        try:
            values, force_retain = _tool_call_span_values(
                name=name,
                call_id=call_id,
                sequence=sequence,
                created_at=created_at,
                duration_s=duration_s,
                status=status,
                pseudonymizer=self._pseudonymizer,
            )
        except Exception:
            _LOGGER.debug("Unable to prepare Galileo tool-call span")
            return
        if buffer is not None:
            tool_count = sum(1 for kind, _ in buffer.child_spans if kind == "tool")
            if tool_count < _MAX_BUFFERED_TOOL_SPANS:
                buffer.child_spans.append(("tool", values))
            buffer.force_retain = buffer.force_retain or force_retain
            return
        assert logger is not None
        try:
            logger.add_tool_span(**values)
        except Exception:
            _LOGGER.debug("Unable to emit Galileo tool-call span")

    def finish(
        self,
        *,
        status: str = "success",
        duration_s: float | None = None,
        recovered: bool = False,
        raw_decisions: Mapping[str, Any] | None = None,
        final_decisions: Mapping[str, Any] | None = None,
        blocked_decisions: Mapping[str, Any] | None = None,
        tool_stats: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        logger = self._logger
        buffer = self._buffer
        if (logger is None and buffer is None) or self._finished:
            return
        self._finished = True
        safe_status = _allowed_label(status, _STATUSES)
        output = _json_payload(
            {
                "blocked_actions": _decision_counts(blocked_decisions),
                "final_actions": _decision_counts(final_decisions),
                "raw_actions": _decision_counts(raw_decisions),
                "recovered": bool(recovered),
                "status": safe_status,
                "tool_stats": _sanitized_tool_stats(tool_stats),
            }
        )
        if buffer is not None:
            buffer.output = output
            buffer.duration_ns = _duration_ns(duration_s)
            buffer.status_code = _status_code(safe_status)
            buffer.force_retain = (
                buffer.force_retain or safe_status in _TERMINAL_RETENTION_STATUSES
            )
            buffer.finished = True
            return
        assert logger is not None
        try:
            logger.conclude(
                output=output,
                redacted_output=output,
                duration_ns=_duration_ns(duration_s),
                status_code=_status_code(safe_status),
            )
        except Exception:
            _LOGGER.debug("Unable to conclude Galileo attempt span")

    def __enter__(self) -> "AttemptTrace":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Literal[False]:
        if exc_type is None:
            self.finish()
        else:
            self.finish(status="failed")
        return False


def create_agentic_telemetry(
    config: GalileoTelemetryConfig | None = None,
    *,
    session_external_id: str | None = None,
    galileo_session_id: str | None = None,
    logger_factory: LoggerFactory | None = None,
    hmac_key: str | bytes | None = None,
) -> AgenticTelemetry:
    """Create telemetry, returning a safe no-op for every invalid setup."""
    resolved = config or GalileoTelemetryConfig.from_env()
    factory = logger_factory
    if not resolved.configured:
        factory = None
    elif factory is None:
        # The SDK reads the key itself.  Checking presence here prevents a known
        # misconfiguration from importing or initializing the optional SDK.
        # The hosted console requires both an explicit URL and the dedicated
        # public-cloud egress opt-in.
        if (
            os.getenv("GALILEO_API_KEY", "").strip()
            and _console_destination_configured()
            and _api_destination_configured()
            and _tls_verification_configured()
        ):
            factory = _default_logger_factory
    return AgenticTelemetry(
        resolved,
        logger_factory=factory,
        pseudonymizer=Pseudonymizer(hmac_key),
        session_external_id=session_external_id,
        galileo_session_id=galileo_session_id,
    )


__all__ = [
    "AgenticTelemetry",
    "AttemptTrace",
    "BatchTrace",
    "GalileoTelemetryConfig",
    "Pseudonymizer",
    "create_agentic_telemetry",
]
