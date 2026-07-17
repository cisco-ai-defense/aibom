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

"""Approval-gated, opt-in Galileo evaluation helpers.

This module has two deliberately separate responsibilities:

* validate a small, versioned, structured decision-evaluation suite and turn it
  into local rows accepted by Galileo's custom-function experiment runner,
  optionally carrying tightly bounded, explicitly approved evidence excerpts;
  and
* run hosted custom-function experiments with per-run random pseudonyms while
  keeping exact fixture identity out of Galileo unless exact mode is separately
  approved; the supplied application receives the exact fixture and controls
  its own model/network egress; and
* construct Galileo's raw LangChain async callback only after explicit fixture,
  identity, content, trajectory, immutable-destination, and egress gates pass.

Label-only suites do not contain prompts, responses, credentials, or other raw
model-content fields, but their exact repository/component/path/case identities
may still be confidential. The hosted runner therefore replaces them by
default. Evidence content is accepted only through the bounded
:class:`ApprovedEvidenceExcerpt` field; it is opaque and is not scanned
probabilistically for secrets. Dataset owners must approve exact/full-content
use. The row builder performs no SDK import and no I/O. Galileo and LangChain
remain optional dependencies and are imported lazily only after the applicable
approval and destination checks.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import posixpath
import re
import secrets
import ssl
import threading
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from typing import Any, Literal, cast
from urllib.parse import urlparse
from uuid import UUID

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

DECISION_SUITE_SCHEMA_VERSION = "aibom.galileo.decision_suite.v1"
GALILEO_DECISION_OUTPUT_SCHEMA_VERSION = "aibom.galileo.decision_output.v1"
FULL_CONTENT_ENV_VAR = "AIBOM_GALILEO_ALLOW_FULL_CONTENT"
FULL_TRAJECTORY_ENV_VAR = "AIBOM_GALILEO_ALLOW_FULL_TRAJECTORY"
EXACT_IDENTITIES_ENV_VAR = "AIBOM_GALILEO_ALLOW_EXACT_IDENTITIES"
EVALUATION_PROJECT_ID_ENV_VAR = "AIBOM_GALILEO_EVALUATION_PROJECT_ID"
EVALUATION_LOG_STREAM_ID_ENV_VAR = "AIBOM_GALILEO_EVALUATION_LOG_STREAM_ID"
# Explicit, off-by-default egress approval for hosted evaluation. Full-content
# evidence still requires the independent fixture and environment gates.
ALLOW_PUBLIC_CLOUD_ENV_VAR = "AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD"
_HOSTED_CONSOLE_HOST = "app.galileo.ai"
_HOSTED_API_HOST = "api.galileo.ai"
_HOSTED_API_ORIGIN = f"https://{_HOSTED_API_HOST}"
AIBOM_EVIDENCE_GROUNDING_METRIC_NAME = "AIBOM Evidence Grounding"
AIBOM_EVIDENCE_GROUNDING_PROMPT = """\
Evaluate only whether the predicted AIBOM decisions are supported by the
explicit approved_evidence excerpts in the evaluation input. Do not use world
knowledge, assumptions about a framework, or evidence that is not present in
approved_evidence.

Evaluation input (candidates and approved_evidence):
{input}

Predicted sanitized decisions (final components, actions, relationships, and
risk flags):
{output}

SME reference labels, supplied only to identify the intended entities and
decision boundaries:
{reference_output}

Check support for every keep, remove, enrich, reclassify, and discover action;
every relationship edge; and every risk flag. Penalize decisions that cite no
approved evidence, overstate what an excerpt shows, or contradict an excerpt.
Do not reward agreement with the SME labels unless the approved evidence itself
supports that decision. Return only one number from 0.0 to 1.0, where 1.0 means
every predicted decision is directly and sufficiently grounded and 0.0 means
none are grounded.
"""

_MAX_CASES = 5_000
_MAX_RELATIONSHIPS_PER_CASE = 256
_MAX_RISKS_PER_CASE = 128
_MAX_CANDIDATES_PER_CASE = 512
_MAX_EXPECTED_COMPONENTS_PER_CASE = 1_024
_MAX_REASON_CODES = 64
_MAX_METADATA_KEYS = 64
_MAX_METADATA_BYTES = 16_384
_MAX_METADATA_LABEL_CHARS = 96
_MAX_EVIDENCE_EXCERPTS_PER_CASE = 32
_MAX_EVIDENCE_LINE_NUMBER = 10_000_000
_MAX_EVIDENCE_LINES_PER_EXCERPT = 500
_MAX_EVIDENCE_CONTENT_CHARS = 16_384
_MAX_EVIDENCE_CONTENT_BYTES = 16_384
_MAX_EVIDENCE_BYTES_PER_CASE = 131_072

_SENSITIVE_METADATA_KEYS = frozenset(
    {
        "api_key",
        "access_token",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "completion",
        "content",
        "credential",
        "credentials",
        "file_content",
        "file_contents",
        "input",
        "output",
        "password",
        "private_key",
        "prompt",
        "raw_content",
        "raw_input",
        "raw_output",
        "raw_response",
        "response",
        "secret",
        "secret_key",
        "session_token",
        "source_code",
        "token",
    }
)
_SENSITIVE_METADATA_SUFFIXES = (
    "_access_token",
    "_api_key",
    "_auth_token",
    "_bearer_token",
    "_client_secret",
    "_credential",
    "_credentials",
    "_password",
    "_private_key",
    "_prompt",
    "_raw_content",
    "_raw_input",
    "_raw_output",
    "_raw_response",
    "_response",
    "_secret",
    "_secret_key",
    "_session_token",
    "_source_code",
    "_content",
    "_completion",
    "_input",
    "_output",
    "_token",
)

_METADATA_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_METADATA_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SECRET_SHAPED_LABEL_RE = re.compile(
    r"(?i)^(?:"
    r"sk-[a-z0-9_-]{8,}|"
    r"gh[pousr]_[a-z0-9]{8,}|"
    r"github_pat_[a-z0-9_]{8,}|"
    r"xox[baprs]-[a-z0-9-]{8,}|"
    r"akia[a-z0-9]{12,}|"
    r"aiza[a-z0-9_-]{20,}|"
    r"eyj[a-z0-9_-]{8,}\."
    r")"
)

_DecisionAction = Literal["keep", "remove", "enrich", "reclassify", "discover"]
_RiskSeverity = Literal["critical", "high", "medium", "low", "info"]
_ExecutionStatus = Literal[
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
]
_SchemaVersion = Literal["aibom.galileo.decision_suite.v1"]
_OutputSchemaVersion = Literal["aibom.galileo.decision_output.v1"]


class FullContentLoggingDenied(PermissionError):
    """Raised when a full-content fixture or environment approval is absent."""


class ExactIdentityLoggingDenied(PermissionError):
    """Raised when networked evaluation identities were not explicitly approved."""


class GalileoIntegrationUnavailable(RuntimeError):
    """Raised when an approved callback cannot load the optional integration."""


class HostedGalileoDestinationRequired(RuntimeError):
    """Raised before evaluation data could be sent to hosted Galileo safely."""


@dataclass(slots=True)
class _SessionSetupAttempt:
    done: threading.Event = field(default_factory=threading.Event)
    session_id: str | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class _FullTrajectorySessionState:
    session_id: str | None = None
    in_flight: _SessionSetupAttempt | None = None


@dataclass(slots=True)
class _CallbackSetupAttempt:
    """One bounded callback construction owned by a factory invocation."""

    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    callback: Any | None = None
    logger: Any | None = None
    error: BaseException | None = None
    abandoned: bool = False


async def _close_owned_ingest_client(logger: Any) -> None:
    """Close this thread's logger-owned Galileo ingest client, when present.

    Galileo 2.4's ``IngestTraces`` creates one ``httpx.AsyncClient`` lazily per
    logger and thread but does not expose a lifecycle method for it.  Closing
    that exact client after a one-shot flush prevents socket/file-descriptor
    leaks.  The fallback ``Traces`` client delegates to the process-wide
    ``GalileoPythonConfig.api_client`` and must never be closed here.
    """
    traces_client = getattr(logger, "_traces_client", None)
    try:
        from galileo.traces import IngestTraces
    except (ImportError, ModuleNotFoundError):
        return
    if not isinstance(traces_client, IngestTraces):
        return

    thread_local = getattr(traces_client, "_thread_local", None)
    if thread_local is None:
        return
    client = getattr(thread_local, "client", None)
    if client is None:
        return
    try:
        await client.aclose()
    except BaseException:  # noqa: BLE001 - cleanup must preserve flush outcome
        pass
    finally:
        # Do not retain a closed client if an SDK path happens to inspect this
        # logger after its deliberately terminal flush.
        try:
            del thread_local.client
        except (AttributeError, TypeError):
            pass


def _discard_raw_logger(logger: Any) -> None:
    """Disable SDK-global lifecycle hooks and erase retained raw traces."""
    from .agentic_telemetry import _discard_logger

    _discard_logger(logger)


class _OneShotGalileoLoggerMixin:
    """Class-level one-shot flush override accepted by Pydantic loggers.

    This mixin deliberately does not capture a logger in a closure.  Galileo's
    logger is a Pydantic model that rejects assigning ``async_flush`` on an
    instance, while a normal class-level override remains supported.
    """

    async def async_flush(self, *args: Any, **kwargs: Any) -> Any:
        try:
            parent_flush = getattr(super(), "async_flush")
            return await parent_flush(*args, **kwargs)
        finally:
            await _close_owned_ingest_client(self)
            # Galileo 2.4 swallows ordinary async ingestion failures and can
            # leave traces resident.  Clearing here makes this callback's one
            # attempt final and prevents delayed shutdown egress.
            _discard_raw_logger(self)


@lru_cache(maxsize=8)
def _one_shot_logger_class(galileo_logger_class: type[Any]) -> type[Any]:
    """Return a reusable SDK subclass without importing Galileo eagerly."""
    return type(
        "_AIBOMOneShotGalileoLogger",
        (_OneShotGalileoLoggerMixin, galileo_logger_class),
        {"__module__": __name__},
    )


# Backward-compatible import alias for the pre-hosted-only integration name.
PrivateGalileoDestinationRequired = HostedGalileoDestinationRequired


class _StrictSchemaModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        from_attributes=True,
        hide_input_in_errors=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalise_metadata_key(key: str) -> str:
    # Split both ordinary camelCase and acronym boundaries before comparing.
    # This makes apiKey, APIKey, api-key, and api_key equivalent for the
    # deterministic deny-list instead of allowing spelling variants through.
    separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key.strip())
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
    return re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")


def _is_sensitive_metadata_key(key: str) -> bool:
    normalized = _normalise_metadata_key(key)
    collapsed = normalized.replace("_", "")
    sensitive_collapsed = {item.replace("_", "") for item in _SENSITIVE_METADATA_KEYS}
    suffix_collapsed = tuple(
        item.removeprefix("_").replace("_", "") for item in _SENSITIVE_METADATA_SUFFIXES
    )
    return (
        normalized in _SENSITIVE_METADATA_KEYS
        or normalized.endswith(_SENSITIVE_METADATA_SUFFIXES)
        or collapsed in sensitive_collapsed
        or any(collapsed.endswith(item) for item in suffix_collapsed)
    )


def _inspect_json_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        if (
            len(value) > _MAX_METADATA_LABEL_CHARS
            or _METADATA_LABEL_RE.fullmatch(value) is None
            or _SECRET_SHAPED_LABEL_RE.match(value) is not None
        ):
            raise ValueError(
                f"{path} string values must be bounded slice labels using only "
                "letters, digits, '.', '_', and '-'; raw prose, paths, PII, and "
                "secret-shaped values are not permitted"
            )
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _inspect_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{path} keys must be non-empty strings")
            if _METADATA_KEY_RE.fullmatch(key) is None:
                raise ValueError(f"{path} keys must be bounded slice-label identifiers")
            if _is_sensitive_metadata_key(key):
                raise ValueError(
                    f"{path} contains disallowed raw-content or secret key {key!r}"
                )
            _inspect_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(
        f"{path} must contain JSON values only; got {type(value).__name__}"
    )


def _validated_metadata(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    if len(value) > _MAX_METADATA_KEYS:
        raise ValueError(
            f"{field_name} supports at most {_MAX_METADATA_KEYS} top-level keys"
        )
    _inspect_json_value(value, path=field_name)
    normalized = {key: value[key] for key in sorted(value)}
    encoded = _canonical_json(normalized).encode("utf-8")
    if len(encoded) > _MAX_METADATA_BYTES:
        raise ValueError(f"{field_name} exceeds the {_MAX_METADATA_BYTES}-byte limit")
    return normalized


def _normalise_label(value: Any) -> str:
    if value is None:
        raise ValueError("label must not be null")
    if hasattr(value, "value"):
        value = value.value
    return "_".join(str(value).strip().casefold().split())


def _normalise_nonempty(value: Any) -> str:
    if value is None:
        raise ValueError("value must not be null")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("value must not be empty")
    return normalized


def _normalise_repository_relative_path(
    value: Any,
    *,
    field_name: str,
    allow_empty: bool,
) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} must not be empty")
    if any(ord(character) < 32 for character in text):
        raise ValueError(f"{field_name} must not contain control characters")
    if posixpath.isabs(text) or (
        len(text) >= 2 and text[1] == ":" and text[0].isalpha()
    ):
        raise ValueError(f"{field_name} must be repository-relative")
    normalized = posixpath.normpath(text)
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"{field_name} must not escape the repository")
    if normalized == ".":
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} must identify a repository file")
    return normalized


def _full_content_environment_approved() -> bool:
    return os.environ.get(FULL_CONTENT_ENV_VAR, "").strip().casefold() == "true"


def _require_full_content_approval(*, approved_fixture: bool) -> None:
    if approved_fixture is not True or not _full_content_environment_approved():
        raise FullContentLoggingDenied(
            "Approved evidence and full-content Galileo integration require "
            "approved_fixture=True and "
            f"{FULL_CONTENT_ENV_VAR}=true"
        )


def _environment_approved(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _require_exact_identity_approval(*, approved_fixture: bool) -> None:
    """Require deliberate approval before exact fixture identities leave locally."""

    if approved_fixture is not True or not _environment_approved(
        EXACT_IDENTITIES_ENV_VAR
    ):
        raise ExactIdentityLoggingDenied(
            "Networked Galileo evaluation contains exact repository, component, "
            "path, case, relationship, and risk identities; it requires "
            f"approved_fixture=True and {EXACT_IDENTITIES_ENV_VAR}=true"
        )


def _require_full_trajectory_approval(*, approved_fixture: bool) -> None:
    _require_full_content_approval(approved_fixture=approved_fixture)
    _require_exact_identity_approval(approved_fixture=approved_fixture)
    if not _environment_approved(FULL_TRAJECTORY_ENV_VAR):
        raise FullContentLoggingDenied(
            "Full LangChain trajectories include prompts, responses, tool I/O, "
            "metadata, and exception details; additionally set "
            f"{FULL_TRAJECTORY_ENV_VAR}=true"
        )


def _raw_tool_root_guards_approved() -> bool:
    """Return whether raw-mode file guards have every static approval gate."""
    try:
        _require_full_trajectory_approval(approved_fixture=True)
        _require_hosted_galileo_destination()
        _required_resource_id(EVALUATION_PROJECT_ID_ENV_VAR)
        _required_resource_id(EVALUATION_LOG_STREAM_ID_ENV_VAR)
    except (
        ExactIdentityLoggingDenied,
        FullContentLoggingDenied,
        HostedGalileoDestinationRequired,
    ):
        return False
    return True


def _required_resource_id(environment_name: str) -> str:
    value = os.environ.get(environment_name, "").strip()
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise HostedGalileoDestinationRequired(
            f"Hosted evaluation requires an explicit UUID in {environment_name}"
        ) from exc
    return str(parsed)


def _validated_experiment_label(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if (
        not normalized
        or _METADATA_LABEL_RE.fullmatch(normalized) is None
        or _SECRET_SHAPED_LABEL_RE.match(normalized) is not None
    ):
        raise ValueError(
            f"{field_name} must be a bounded non-secret label using only letters, "
            "digits, '.', '_', and '-'"
        )
    return normalized


def _validated_experiment_tags(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or len(value) > _MAX_METADATA_KEYS:
        raise ValueError("experiment_tags must be a bounded string mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        safe_key = _validated_experiment_label(key, field_name="experiment tag key")
        safe_value = _validated_experiment_label(
            item, field_name=f"experiment tag {safe_key!r}"
        )
        result[safe_key] = safe_value
    return dict(sorted(result.items()))


def _normalized_exact_https_origin(value: str, expected_host: str) -> str | None:
    """Canonicalize an exact HTTPS origin with no URL payload."""
    candidate = value.strip()
    if not candidate:
        return None
    try:
        parsed = urlparse(candidate)
        hostname = (parsed.hostname or "").rstrip(".").casefold()
        _ = parsed.port
    except (TypeError, ValueError):
        return None
    if (
        parsed.scheme.casefold() != "https"
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


def _normalized_evaluation_console_origin(value: str) -> str | None:
    """Return the canonical explicitly approved hosted console origin."""
    allow_hosted = os.environ.get(
        ALLOW_PUBLIC_CLOUD_ENV_VAR, ""
    ).strip().casefold() in {"1", "true", "yes", "on"}
    if not allow_hosted:
        return None
    return _normalized_exact_https_origin(value, _HOSTED_CONSOLE_HOST)


def _normalized_evaluation_api_origin(value: str | None = None) -> str | None:
    """Return the hosted API origin, rejecting every explicit override."""
    if value is None:
        candidate = os.environ.get("GALILEO_API_URL", "").strip() or _HOSTED_API_ORIGIN
    else:
        candidate = value
    return _normalized_exact_https_origin(candidate, _HOSTED_API_HOST)


def _evaluation_tls_verification_configured() -> bool:
    """Reject SDK configuration that explicitly disables TLS verification."""
    candidate = os.environ.get("GALILEO_SSL_CONTEXT", "").strip().casefold()
    return not candidate or candidate in {"1", "true", "yes", "on", "t", "y"}


def _sdk_tls_verification_enabled(value: Any) -> bool:
    """Accept the SDK default or a hostname-checking, validating SSL context."""
    if value is True:
        return True
    if isinstance(value, ssl.SSLContext):
        return value.check_hostname and value.verify_mode == ssl.CERT_REQUIRED
    return False


def _require_hosted_galileo_destination() -> str:
    """Require explicit approval for the hosted Galileo HTTPS origin.

    Evaluation rows contain exact component identities and approved runs may
    contain source evidence. Hosted evaluation therefore requires its own
    explicit egress flag in addition to any full-content approvals.
    """
    candidate = os.environ.get("GALILEO_CONSOLE_URL", "").strip()
    normalized = _normalized_evaluation_console_origin(candidate)
    if (
        normalized is None
        or _normalized_evaluation_api_origin() is None
        or not _evaluation_tls_verification_configured()
    ):
        raise HostedGalileoDestinationRequired(
            "Galileo evaluation requires GALILEO_CONSOLE_URL="
            "https://app.galileo.ai, the hosted https://api.galileo.ai API, and "
            "TLS verification plus explicit hosted egress approval via "
            f"{ALLOW_PUBLIC_CLOUD_ENV_VAR}=true"
        )
    return normalized


def _verify_loaded_galileo_destination(expected_url: str) -> None:
    """Fail closed if Galileo's process singleton targets another origin."""
    try:
        from galileo.config import GalileoPythonConfig
    except (ImportError, ModuleNotFoundError) as exc:
        raise GalileoIntegrationUnavailable(
            "The optional Galileo configuration integration is unavailable"
        ) from exc

    instance = getattr(GalileoPythonConfig, "_instance", None)
    if instance is None:
        return
    actual_url = str(getattr(instance, "console_url", "")).strip()
    expected_api_url = _normalized_evaluation_api_origin()
    actual_api_url = str(getattr(instance, "api_url", "")).strip()
    if (
        _normalized_evaluation_console_origin(actual_url) != expected_url
        or expected_api_url is None
        or _normalized_evaluation_api_origin(actual_api_url) != expected_api_url
        or not _sdk_tls_verification_enabled(getattr(instance, "ssl_context", True))
    ):
        raise HostedGalileoDestinationRequired(
            "The loaded Galileo SDK configuration does not match the approved "
            "GALILEO_CONSOLE_URL; restart the process before evaluation"
        )


class ApprovedEvidenceExcerpt(_StrictSchemaModel):
    """A bounded evidence excerpt approved for an evaluation fixture.

    The path is checked lexically and is never opened. Content is required and
    bounded by characters and UTF-8 bytes, but remains opaque: this model does
    not attempt probabilistic secret or PII filtering. The dataset owner must
    inspect and approve the exact excerpt before enabling full-content rows.
    Metadata is JSON-only and rejects raw-content and secret-shaped keys.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        from_attributes=True,
        hide_input_in_errors=True,
        populate_by_name=True,
        str_strip_whitespace=False,
    )

    source_path: str = Field(
        min_length=1,
        max_length=2_048,
        validation_alias=AliasChoices(
            "source_path", "repo_relative_path", "file_path", "path"
        ),
    )
    start_line: int = Field(ge=1, le=_MAX_EVIDENCE_LINE_NUMBER, strict=True)
    end_line: int = Field(ge=1, le=_MAX_EVIDENCE_LINE_NUMBER, strict=True)
    content: str = Field(
        min_length=1,
        max_length=_MAX_EVIDENCE_CONTENT_CHARS,
        strict=True,
    )
    evidence_kind: str = Field(
        min_length=1,
        max_length=64,
        validation_alias=AliasChoices("evidence_kind", "kind"),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_path", mode="before")
    @classmethod
    def normalize_source_path(cls, value: Any) -> str:
        return _normalise_repository_relative_path(
            value,
            field_name="source_path",
            allow_empty=False,
        )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("evidence content must be a string")
        if not value.strip():
            raise ValueError("evidence content must not be empty or whitespace-only")
        if len(value.encode("utf-8")) > _MAX_EVIDENCE_CONTENT_BYTES:
            raise ValueError(
                "evidence content exceeds the "
                f"{_MAX_EVIDENCE_CONTENT_BYTES}-byte UTF-8 limit"
            )
        return value

    @field_validator("evidence_kind", mode="before")
    @classmethod
    def normalize_evidence_kind(cls, value: Any) -> str:
        return _normalise_label(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_evidence_metadata(cls, value: Any) -> dict[str, Any]:
        return _validated_metadata(value, field_name="approved_evidence.metadata")

    @model_validator(mode="after")
    def validate_line_range(self) -> "ApprovedEvidenceExcerpt":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        line_count = self.end_line - self.start_line + 1
        if line_count > _MAX_EVIDENCE_LINES_PER_EXCERPT:
            raise ValueError(
                "approved evidence supports at most "
                f"{_MAX_EVIDENCE_LINES_PER_EXCERPT} lines per excerpt"
            )
        return self


class DecisionCandidate(_StrictSchemaModel):
    """Bounded component candidate without raw source or prompt content."""

    component_type: str = Field(
        min_length=1,
        max_length=64,
        validation_alias=AliasChoices("component_type", "type"),
    )
    name: str = Field(min_length=1, max_length=512)
    repository: str = Field(default="", max_length=512)
    source_path: str = Field(
        default="",
        max_length=2_048,
        validation_alias=AliasChoices("source_path", "file_path", "file"),
    )
    line_number: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("line_number", "line"),
    )
    stable_case_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("stable_case_id", "case_id", "eval_case_id"),
    )
    instance_id: str | None = Field(default=None, min_length=1, max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("component_type", mode="before")
    @classmethod
    def normalize_component_type(cls, value: Any) -> str:
        return _normalise_label(value)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: Any) -> str:
        return _normalise_nonempty(value)

    @field_validator("repository", mode="before")
    @classmethod
    def strip_repository(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()

    @field_validator("stable_case_id", "instance_id", mode="before")
    @classmethod
    def strip_optional_ids(cls, value: Any) -> str | None:
        if value is None:
            return None
        return _normalise_nonempty(value)

    @field_validator("source_path", mode="before")
    @classmethod
    def normalize_relative_source_path(cls, value: Any) -> str:
        return _normalise_repository_relative_path(
            value,
            field_name="source_path",
            allow_empty=True,
        )

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_candidate_metadata(cls, value: Any) -> dict[str, Any]:
        return _validated_metadata(value, field_name="candidate.metadata")


class ExpectedActionLabel(_StrictSchemaModel):
    """Ground-truth decision for one candidate."""

    action: _DecisionAction
    target_type: str | None = Field(default=None, min_length=1, max_length=64)
    reason_codes: list[str] = Field(default_factory=list, max_length=_MAX_REASON_CODES)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, value: Any) -> str:
        return _normalise_label(value)

    @field_validator("target_type", mode="before")
    @classmethod
    def normalize_target_type(cls, value: Any) -> str | None:
        if value is None:
            return None
        return _normalise_label(value)

    @field_validator("reason_codes", mode="before")
    @classmethod
    def normalize_reason_codes(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple, set, frozenset)):
            raise ValueError("reason_codes must be a list of labels")
        normalized = {_normalise_label(item) for item in value}
        if "" in normalized:
            raise ValueError("reason_codes must not contain empty labels")
        return sorted(normalized)

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_action_metadata(cls, value: Any) -> dict[str, Any]:
        return _validated_metadata(value, field_name="expected_action.metadata")

    @model_validator(mode="after")
    def validate_reclassification_target(self) -> "ExpectedActionLabel":
        if self.action == "reclassify" and self.target_type is None:
            raise ValueError("reclassify actions require target_type")
        if self.action != "reclassify" and self.target_type is not None:
            raise ValueError("target_type is valid only for reclassify actions")
        return self


class ExpectedRelationshipLabel(_StrictSchemaModel):
    """Ground-truth directed relationship between stable entity identifiers."""

    relationship_type: str = Field(
        min_length=1,
        max_length=128,
        validation_alias=AliasChoices("relationship_type", "type", "label"),
    )
    source_case_id: str = Field(
        min_length=1,
        max_length=512,
        validation_alias=AliasChoices(
            "source_case_id", "source_id", "source_instance_id", "source"
        ),
    )
    target_case_id: str = Field(
        min_length=1,
        max_length=512,
        validation_alias=AliasChoices(
            "target_case_id", "target_id", "target_instance_id", "target"
        ),
    )
    expected_present: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("relationship_type", mode="before")
    @classmethod
    def normalize_relationship_type(cls, value: Any) -> str:
        return _normalise_label(value)

    @field_validator("source_case_id", "target_case_id", mode="before")
    @classmethod
    def strip_endpoint_ids(cls, value: Any) -> str:
        return _normalise_nonempty(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_relationship_metadata(cls, value: Any) -> dict[str, Any]:
        return _validated_metadata(value, field_name="expected_relationship.metadata")


class ExpectedRiskLabel(_StrictSchemaModel):
    """Ground-truth risk presence or absence for one candidate."""

    case_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        validation_alias=AliasChoices(
            "case_id", "candidate_case_id", "component_case_id", "stable_case_id"
        ),
    )
    risk_type: str = Field(
        min_length=1,
        max_length=128,
        validation_alias=AliasChoices("risk_type", "risk_flag", "flag", "type", "name"),
    )
    severity: _RiskSeverity
    expected_present: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("case_id", mode="before")
    @classmethod
    def normalize_case_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        return _normalise_nonempty(value)

    @field_validator("risk_type", mode="before")
    @classmethod
    def normalize_risk_type(cls, value: Any) -> str:
        return _normalise_label(value)

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, value: Any) -> str:
        if value is None:
            raise ValueError("expected risk severity is required")
        return _normalise_label(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_risk_metadata(cls, value: Any) -> dict[str, Any]:
        return _validated_metadata(value, field_name="expected_risk.metadata")


class ExecutionOutcome(_StrictSchemaModel):
    """Structured batch reliability outcome for deterministic evaluation.

    Every field is optional so a fixture can label only the dimension it can
    authoritatively establish. At least one field is required whenever the
    object is present; a missing prediction for a labeled field scores as a
    mismatch rather than silently disappearing from the aggregate.
    """

    status: _ExecutionStatus | None = None
    schema_valid: bool | None = Field(default=None, strict=True)
    abstained: bool | None = Field(default=None, strict=True)
    degraded_candidate_count: int | None = Field(default=None, ge=0, strict=True)
    retry_count: int | None = Field(default=None, ge=0, strict=True)
    fallback_count: int | None = Field(default=None, ge=0, strict=True)
    cache_hit: bool | None = Field(default=None, strict=True)
    tool_error_count: int | None = Field(default=None, ge=0, strict=True)
    guard_denial_count: int | None = Field(default=None, ge=0, strict=True)

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: Any) -> str | None:
        if value is None:
            return None
        return _normalise_label(value)

    @model_validator(mode="after")
    def require_labeled_dimension(self) -> "ExecutionOutcome":
        if all(value is None for value in self.model_dump(mode="python").values()):
            raise ValueError("execution outcome must label at least one dimension")
        return self


class GalileoDecisionRelationship(_StrictSchemaModel):
    """Sanitized predicted relationship used by local Galileo metrics."""

    relationship_type: str = Field(min_length=1, max_length=128)
    source_id: str = Field(min_length=1, max_length=512)
    target_id: str = Field(min_length=1, max_length=512)
    predicted_present: bool = True

    @field_validator("relationship_type", mode="before")
    @classmethod
    def normalize_relationship_type(cls, value: Any) -> str:
        return _normalise_label(value)

    @field_validator("source_id", "target_id", mode="before")
    @classmethod
    def normalize_endpoint_id(cls, value: Any) -> str:
        return _normalise_nonempty(value)


class GalileoDecisionRisk(_StrictSchemaModel):
    """Sanitized predicted risk used by local Galileo metrics."""

    case_id: str | None = Field(default=None, min_length=1, max_length=512)
    risk_type: str = Field(min_length=1, max_length=128)
    severity: _RiskSeverity | None = None
    predicted_present: bool = True

    @field_validator("case_id", mode="before")
    @classmethod
    def normalize_case_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        return _normalise_nonempty(value)

    @field_validator("risk_type", mode="before")
    @classmethod
    def normalize_risk_type(cls, value: Any) -> str:
        return _normalise_label(value)

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, value: Any) -> str | None:
        if value is None:
            return None
        return _normalise_label(value)


class GalileoDecisionOutput(_StrictSchemaModel):
    """Canonical, content-minimized output logged by custom experiments.

    The custom application may return ``components`` or ``final_components``
    and ``risks`` or ``risk_flags``.  The public sanitizer projects either form
    into this single schema and drops every field not required for exact entity
    scoring.  A schema-invalid application result is represented only by an
    empty ``schema_valid=False`` envelope; the rejected value and validation
    error are never included in the logged output.
    """

    schema_version: _OutputSchemaVersion = "aibom.galileo.decision_output.v1"
    schema_valid: bool = True
    final_components: list[DecisionCandidate] = Field(
        default_factory=list,
        max_length=_MAX_EXPECTED_COMPONENTS_PER_CASE,
    )
    relationships: list[GalileoDecisionRelationship] = Field(
        default_factory=list,
        max_length=_MAX_RELATIONSHIPS_PER_CASE,
    )
    risk_flags: list[GalileoDecisionRisk] = Field(
        default_factory=list,
        max_length=_MAX_RISKS_PER_CASE,
    )
    actions: dict[str, ExpectedActionLabel] | None = None
    execution_outcome: ExecutionOutcome | None = None

    @field_validator("actions", mode="before")
    @classmethod
    def validate_actions(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("actions must be an object keyed by stable case ID")
        if len(value) > _MAX_EXPECTED_COMPONENTS_PER_CASE:
            raise ValueError("actions exceeds the per-output limit")
        return {_normalise_nonempty(key): item for key, item in value.items()}

    @model_validator(mode="after")
    def validate_failure_envelope(self) -> "GalileoDecisionOutput":
        if (
            self.execution_outcome is not None
            and self.execution_outcome.schema_valid is not None
            and self.execution_outcome.schema_valid != self.schema_valid
        ):
            raise ValueError(
                "execution_outcome.schema_valid must match output schema_valid"
            )
        if not self.schema_valid and (
            self.final_components
            or self.relationships
            or self.risk_flags
            or self.actions not in (None, {})
        ):
            raise ValueError(
                "schema-invalid outputs may contain only a sanitized execution outcome"
            )
        return self


def _candidate_case_id(candidate: DecisionCandidate) -> str:
    if candidate.stable_case_id:
        return candidate.stable_case_id
    for key in ("stable_case_id", "case_id", "eval_case_id"):
        metadata_id = candidate.metadata.get(key)
        if metadata_id is not None and str(metadata_id).strip():
            return str(metadata_id).strip()
    return ""


def _candidate_exact_identity(candidate: DecisionCandidate) -> tuple[Any, ...]:
    """Identity fields consumed by entity-level decision evaluation."""
    return (
        _candidate_case_id(candidate),
        candidate.component_type,
        candidate.name,
        candidate.repository,
        candidate.source_path,
        candidate.line_number,
    )


def _candidate_identity_except_type(candidate: DecisionCandidate) -> tuple[Any, ...]:
    """Identity fields that a reclassification is not allowed to change."""
    return (
        _candidate_case_id(candidate),
        candidate.name,
        candidate.repository,
        candidate.source_path,
        candidate.line_number,
    )


def _canonical_candidate_input(
    value: Any,
    *,
    fallback_case_id: str | None = None,
) -> dict[str, Any]:
    candidate = DecisionCandidate.model_validate(value)
    stable_id = _candidate_case_id(candidate) or fallback_case_id
    if stable_id and candidate.stable_case_id != stable_id:
        candidate = candidate.model_copy(update={"stable_case_id": stable_id})
    return candidate.model_dump(mode="json", exclude_none=True)


def _canonical_action_input(value: Any) -> dict[str, Any]:
    raw = {"action": value} if isinstance(value, str) else value
    return ExpectedActionLabel.model_validate(raw).model_dump(
        mode="json", exclude_none=True
    )


class DecisionSuiteCase(_StrictSchemaModel):
    """One deterministic candidate batch and its entity-level golden labels.

    ``candidate``/``expected_action`` remain accepted for legacy single-candidate
    fixtures. They are normalized into the canonical ``candidates`` and
    ``expected_actions`` batch fields before validation.
    """

    case_id: str = Field(
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("case_id", "stable_case_id", "eval_case_id"),
    )
    candidate: DecisionCandidate | None = None
    candidates: list[DecisionCandidate] = Field(
        default_factory=list, min_length=1, max_length=_MAX_CANDIDATES_PER_CASE
    )
    expected_action: ExpectedActionLabel | None = None
    expected_actions: dict[str, ExpectedActionLabel] = Field(default_factory=dict)
    expected_components: list[DecisionCandidate] = Field(
        default_factory=list, max_length=_MAX_EXPECTED_COMPONENTS_PER_CASE
    )
    expected_discovered_components: list[DecisionCandidate] = Field(
        default_factory=list, max_length=_MAX_EXPECTED_COMPONENTS_PER_CASE
    )
    deterministic_relationships: list[ExpectedRelationshipLabel] | None = Field(
        default=None,
        max_length=_MAX_RELATIONSHIPS_PER_CASE,
        validation_alias=AliasChoices(
            "deterministic_relationships", "baseline_relationships"
        ),
    )
    expected_relationships: list[ExpectedRelationshipLabel] = Field(
        default_factory=list, max_length=_MAX_RELATIONSHIPS_PER_CASE
    )
    expected_risks: list[ExpectedRiskLabel] = Field(
        default_factory=list, max_length=_MAX_RISKS_PER_CASE
    )
    expected_execution_outcome: ExecutionOutcome | None = Field(
        default=None,
        validation_alias=AliasChoices("expected_execution_outcome", "expected_outcome"),
    )
    approved_evidence: list[ApprovedEvidenceExcerpt] = Field(
        default_factory=list,
        max_length=_MAX_EVIDENCE_EXCERPTS_PER_CASE,
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_and_batch_contract(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = deepcopy(dict(value))
        raw_case_id = next(
            (
                data[key]
                for key in ("case_id", "stable_case_id", "eval_case_id")
                if key in data
            ),
            None,
        )
        case_id = _normalise_nonempty(raw_case_id) if raw_case_id is not None else None

        legacy_candidate = data.get("candidate")
        raw_candidates = data.get("candidates")
        if raw_candidates is None and legacy_candidate is not None:
            raw_candidates = [legacy_candidate]
        if isinstance(raw_candidates, (list, tuple)):
            canonical_candidates = [
                _canonical_candidate_input(
                    item,
                    fallback_case_id=(
                        case_id
                        if legacy_candidate is not None and len(raw_candidates) == 1
                        else None
                    ),
                )
                for item in raw_candidates
            ]
            data["candidates"] = canonical_candidates
            if legacy_candidate is not None and len(canonical_candidates) == 1:
                data["candidate"] = canonical_candidates[0]
        else:
            canonical_candidates = []

        raw_actions = data.get("expected_actions")
        if raw_actions is None and data.get("expected_action") is not None:
            if not canonical_candidates:
                raise ValueError("legacy expected_action requires candidate")
            action_case_id = _candidate_case_id(
                DecisionCandidate.model_validate(canonical_candidates[0])
            )
            raw_actions = {action_case_id: data["expected_action"]}
        if isinstance(raw_actions, Mapping):
            canonical_actions = {
                _normalise_nonempty(key): _canonical_action_input(raw_actions[key])
                for key in sorted(raw_actions)
            }
            data["expected_actions"] = canonical_actions
            if data.get("expected_action") is not None:
                data["expected_action"] = _canonical_action_input(
                    data["expected_action"]
                )
        else:
            canonical_actions = {}

        if "expected_components" not in data and canonical_candidates:
            inferred_final: list[dict[str, Any]] = []
            for raw_candidate in canonical_candidates:
                candidate_model = DecisionCandidate.model_validate(raw_candidate)
                stable_id = _candidate_case_id(candidate_model)
                action = canonical_actions.get(stable_id)
                action_name = action.get("action") if action else None
                if action_name == "remove":
                    continue
                final_candidate = deepcopy(raw_candidate)
                if action is not None and action_name == "reclassify":
                    final_candidate["component_type"] = action["target_type"]
                inferred_final.append(final_candidate)
            data["expected_components"] = inferred_final

        if "expected_discovered_components" not in data and canonical_candidates:
            data["expected_discovered_components"] = [
                deepcopy(raw_candidate)
                for raw_candidate in canonical_candidates
                if canonical_actions.get(
                    _candidate_case_id(DecisionCandidate.model_validate(raw_candidate)),
                    {},
                ).get("action")
                == "discover"
            ]

        raw_risks = data.get("expected_risks")
        if isinstance(raw_risks, (list, tuple)):
            normalized_risks: list[Any] = []
            sole_candidate_id = (
                _candidate_case_id(
                    DecisionCandidate.model_validate(canonical_candidates[0])
                )
                if len(canonical_candidates) == 1
                else None
            )
            for item in raw_risks:
                normalized = {"risk_type": item} if isinstance(item, str) else item
                if isinstance(normalized, Mapping):
                    normalized = deepcopy(dict(normalized))
                    if (
                        not any(
                            key in normalized
                            for key in (
                                "case_id",
                                "candidate_case_id",
                                "component_case_id",
                                "stable_case_id",
                            )
                        )
                        and sole_candidate_id
                    ):
                        normalized["case_id"] = sole_candidate_id
                normalized_risks.append(normalized)
            data["expected_risks"] = normalized_risks
        return data

    @field_validator("case_id", mode="before")
    @classmethod
    def strip_case_id(cls, value: Any) -> str:
        return _normalise_nonempty(value)

    @field_validator("expected_action", mode="before")
    @classmethod
    def normalize_action_object(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            return {"action": value}
        return value

    @field_validator("expected_actions", mode="before")
    @classmethod
    def normalize_action_mapping(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(
                "expected_actions must be an object keyed by stable case ID"
            )
        if len(value) > _MAX_EXPECTED_COMPONENTS_PER_CASE:
            raise ValueError("expected_actions exceeds the per-case limit")
        return {
            _normalise_nonempty(key): (
                {"action": item} if isinstance(item, str) else item
            )
            for key, item in value.items()
        }

    @field_validator("expected_risks", mode="before")
    @classmethod
    def normalize_risk_objects(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [
                {"risk_type": item} if isinstance(item, str) else item for item in value
            ]
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_case_metadata(cls, value: Any) -> dict[str, Any]:
        return _validated_metadata(value, field_name="case.metadata")

    @model_validator(mode="after")
    def validate_entity_labels(self) -> "DecisionSuiteCase":
        candidate_ids = [_candidate_case_id(candidate) for candidate in self.candidates]
        if any(
            not candidate.repository or not candidate.source_path
            for candidate in self.candidates
        ):
            raise ValueError(
                "every batch candidate requires repository and source_path identity"
            )
        if any(not case_id for case_id in candidate_ids):
            raise ValueError("every batch candidate requires a stable_case_id")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidates contains duplicate stable_case_id values")

        if self.candidate is not None:
            if len(self.candidates) != 1 or self.candidate != self.candidates[0]:
                raise ValueError(
                    "legacy candidate must match the single batch candidate"
                )
            if candidate_ids[0] != self.case_id:
                raise ValueError(
                    "candidate stable case id must match the enclosing case_id"
                )
        if self.expected_action is not None:
            if len(self.candidates) != 1:
                raise ValueError("legacy expected_action supports one candidate only")
            action = self.expected_actions.get(candidate_ids[0])
            if action != self.expected_action:
                raise ValueError(
                    "legacy expected_action must match expected_actions for its candidate"
                )

        expected_by_id: dict[str, DecisionCandidate] = {}
        for component in self.expected_components:
            if not component.repository or not component.source_path:
                raise ValueError(
                    "every expected component requires repository and source_path "
                    "identity"
                )
            stable_id = _candidate_case_id(component)
            if not stable_id:
                raise ValueError(
                    "every expected component requires exact identity fields "
                    "and a stable_case_id"
                )
            if stable_id in expected_by_id:
                raise ValueError(
                    "expected_components contains duplicate stable_case_id values"
                )
            expected_by_id[stable_id] = component

        discovered_by_id: dict[str, DecisionCandidate] = {}
        for component in self.expected_discovered_components:
            stable_id = _candidate_case_id(component)
            if not stable_id:
                raise ValueError(
                    "every expected discovered component requires exact identity "
                    "fields and a stable_case_id"
                )
            if stable_id in discovered_by_id:
                raise ValueError(
                    "expected_discovered_components contains duplicate stable_case_id "
                    "values"
                )
            discovered_by_id[stable_id] = component
            final_component = expected_by_id.get(stable_id)
            if final_component is None or _candidate_exact_identity(
                final_component
            ) != _candidate_exact_identity(component):
                raise ValueError(
                    "expected_discovered_components must be an exact subset of "
                    "expected_components"
                )

        overlapping_discoveries = set(candidate_ids) & set(discovered_by_id)
        if overlapping_discoveries:
            raise ValueError(
                "expected discoveries must be absent from deterministic candidates "
                f"(overlap={sorted(overlapping_discoveries)})"
            )

        required_action_ids = set(candidate_ids) | set(discovered_by_id)
        action_ids = set(self.expected_actions)
        if action_ids != required_action_ids:
            missing = sorted(required_action_ids - action_ids)
            unexpected = sorted(action_ids - required_action_ids)
            raise ValueError(
                "expected_actions must contain exactly one action for every batch "
                f"candidate and discovery (missing={missing}, unexpected={unexpected})"
            )

        candidate_by_id = dict(zip(candidate_ids, self.candidates, strict=True))
        expected_final_ids = set(expected_by_id)
        for stable_id, candidate in candidate_by_id.items():
            action = self.expected_actions[stable_id]
            final_component = expected_by_id.get(stable_id)
            if action.action == "remove":
                if final_component is not None:
                    raise ValueError(
                        "removed candidates must not be expected components"
                    )
                continue
            if final_component is None:
                raise ValueError(
                    f"candidate {stable_id!r} action {action.action!r} requires an "
                    "expected final component"
                )
            if action.action in {"keep", "enrich", "discover"} and (
                _candidate_exact_identity(candidate)
                != _candidate_exact_identity(final_component)
            ):
                raise ValueError(
                    f"{action.action} candidates must preserve exact component identity"
                )
            if action.action == "reclassify":
                if final_component.component_type != action.target_type:
                    raise ValueError(
                        "reclassified expected component type must match target_type"
                    )
                if _candidate_identity_except_type(
                    candidate
                ) != _candidate_identity_except_type(final_component):
                    raise ValueError(
                        "reclassified candidates may change only component_type"
                    )
            if action.action == "discover" and stable_id not in discovered_by_id:
                raise ValueError(
                    "discover actions require an exact expected discovered component"
                )

        for stable_id in discovered_by_id:
            if self.expected_actions[stable_id].action != "discover":
                raise ValueError("expected discoveries require action='discover'")

        allowed_final_ids = set(candidate_ids) | set(discovered_by_id)
        if not expected_final_ids <= allowed_final_ids:
            raise ValueError(
                "expected_components contains entities that are neither batch "
                "candidates nor expected discoveries"
            )

        if self.deterministic_relationships is not None:
            deterministic_relationship_keys: list[tuple[str, str, str]] = []
            valid_endpoint_ids = set(candidate_ids)
            for relationship in self.deterministic_relationships:
                if not relationship.expected_present:
                    raise ValueError(
                        "deterministic_relationships must contain present edges only"
                    )
                if (
                    relationship.source_case_id not in valid_endpoint_ids
                    or relationship.target_case_id not in valid_endpoint_ids
                ):
                    raise ValueError(
                        "deterministic_relationship endpoints must reference candidate "
                        "stable_case_id values"
                    )
                deterministic_relationship_keys.append(
                    (
                        relationship.relationship_type,
                        relationship.source_case_id,
                        relationship.target_case_id,
                    )
                )
            if len(deterministic_relationship_keys) != len(
                set(deterministic_relationship_keys)
            ):
                raise ValueError(
                    "deterministic_relationships contains duplicate present edges"
                )

        relationship_keys = [
            (
                relationship.relationship_type,
                relationship.source_case_id,
                relationship.target_case_id,
            )
            for relationship in self.expected_relationships
        ]
        if len(relationship_keys) != len(set(relationship_keys)):
            raise ValueError("expected_relationships contains duplicate labels")
        labeled_relationship_endpoint_ids = set(candidate_ids) | expected_final_ids
        for relationship in self.expected_relationships:
            valid_relationship_endpoint_ids = (
                expected_final_ids
                if relationship.expected_present
                else labeled_relationship_endpoint_ids
            )
            if (
                relationship.source_case_id not in valid_relationship_endpoint_ids
                or relationship.target_case_id not in valid_relationship_endpoint_ids
            ):
                if relationship.expected_present:
                    raise ValueError(
                        "present expected_relationship endpoints must reference "
                        "expected final stable_case_id values"
                    )
                raise ValueError(
                    "negative expected_relationship endpoints must reference "
                    "candidate or expected final stable_case_id values"
                )

        risk_keys: list[tuple[str, str, str | None]] = []
        allowed_risk_ids = set(candidate_ids) | expected_final_ids
        for risk in self.expected_risks:
            if not risk.case_id:
                raise ValueError(
                    "expected risks in a batch require a candidate/component case_id"
                )
            if risk.case_id not in allowed_risk_ids:
                raise ValueError("expected risk case_id is not present in the batch")
            risk_keys.append((risk.case_id, risk.risk_type, risk.severity))
        if len(risk_keys) != len(set(risk_keys)):
            raise ValueError("expected_risks contains duplicate labels")

        evidence_keys = [
            (
                evidence.source_path,
                evidence.start_line,
                evidence.end_line,
                evidence.evidence_kind,
            )
            for evidence in self.approved_evidence
        ]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("approved_evidence contains duplicate excerpts")
        evidence_bytes = sum(
            len(evidence.content.encode("utf-8")) for evidence in self.approved_evidence
        )
        if evidence_bytes > _MAX_EVIDENCE_BYTES_PER_CASE:
            raise ValueError(
                "approved_evidence exceeds the "
                f"{_MAX_EVIDENCE_BYTES_PER_CASE}-byte per-case limit"
            )
        return self


class DecisionSuite(_StrictSchemaModel):
    """Versioned collection of entity-level decision-evaluation fixtures."""

    schema_version: _SchemaVersion = Field(
        validation_alias=AliasChoices("schema_version", "version")
    )
    cases: list[DecisionSuiteCase] = Field(min_length=1, max_length=_MAX_CASES)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_suite_metadata(cls, value: Any) -> dict[str, Any]:
        return _validated_metadata(value, field_name="suite.metadata")

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> "DecisionSuite":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("decision suite case_id values must be unique")
        return self


def validate_decision_suite(
    payload: DecisionSuite | Mapping[str, Any] | str | bytes,
) -> DecisionSuite:
    """Validate a decision-suite object or JSON document.

    The schema version is mandatory. Unknown fields, absolute/traversing source
    paths, raw-content or secret metadata keys, duplicate labels, mismatched
    stable case IDs, and out-of-bounds evidence are rejected. Evidence content
    is not inspected probabilistically and requires human approval before it
    can be serialized by :func:`build_galileo_experiment_rows`.
    """
    if isinstance(payload, DecisionSuite):
        return payload
    if isinstance(payload, (str, bytes)):
        return DecisionSuite.model_validate_json(payload)
    return DecisionSuite.model_validate(payload)


def _metadata_value(value: Any) -> str:
    return value if isinstance(value, str) else _canonical_json(value)


def _row_metadata(suite: DecisionSuite, case: DecisionSuiteCase) -> dict[str, str]:
    repositories = {item.repository for item in case.candidates if item.repository}
    single_candidate = case.candidates[0] if len(case.candidates) == 1 else None
    metadata = {
        "aibom.candidate_count": str(len(case.candidates)),
        "aibom.case_id": case.case_id,
        "aibom.component_type": (
            single_candidate.component_type if single_candidate else "batch"
        ),
        "aibom.repository": (
            next(iter(repositories))
            if len(repositories) == 1
            else ("multiple" if repositories else "")
        ),
        "aibom.schema_version": suite.schema_version,
        "aibom.source_path": single_candidate.source_path if single_candidate else "",
    }
    for namespace, values in (
        ("suite", suite.metadata),
        ("case", case.metadata),
    ):
        for key in sorted(values):
            metadata[f"aibom.{namespace}.{key}"] = _metadata_value(values[key])
    return dict(sorted(metadata.items()))


def _validated_experiment_rows(suite: DecisionSuite) -> list[dict[str, Any]]:
    """Serialize an already validated suite without performing approval checks."""

    rows: list[dict[str, Any]] = []
    for case in sorted(suite.cases, key=lambda item: item.case_id):
        candidates = [
            candidate.model_dump(mode="json", exclude_none=True)
            for candidate in case.candidates
        ]
        input_payload = {
            "approved_evidence": [
                evidence.model_dump(mode="json", exclude_none=True)
                for evidence in case.approved_evidence
            ],
            "candidates": candidates,
            "case_id": case.case_id,
            "metadata": case.metadata,
            "schema_version": suite.schema_version,
        }
        # Preserve the legacy key only when the fixture actually used the
        # legacy contract. Adding it to every one-item batch would incorrectly
        # impose the legacy invariant that case_id == candidate stable ID.
        if case.candidate is not None:
            input_payload["candidate"] = case.candidate.model_dump(
                mode="json", exclude_none=True
            )
        if case.deterministic_relationships is not None:
            input_payload["deterministic_relationships"] = [
                relationship.model_dump(mode="json", exclude_none=True)
                for relationship in case.deterministic_relationships
            ]

        ground_truth_payload = {
            "case_id": case.case_id,
            "expected_actions": {
                stable_id: case.expected_actions[stable_id].model_dump(
                    mode="json", exclude_none=True
                )
                for stable_id in sorted(case.expected_actions)
            },
            "expected_components": [
                component.model_dump(mode="json", exclude_none=True)
                for component in case.expected_components
            ],
            "expected_discovered_components": [
                component.model_dump(mode="json", exclude_none=True)
                for component in case.expected_discovered_components
            ],
            "expected_relationships": [
                relationship.model_dump(mode="json", exclude_none=True)
                for relationship in case.expected_relationships
            ],
            "expected_risks": [
                risk.model_dump(mode="json", exclude_none=True)
                for risk in case.expected_risks
            ],
            "schema_version": suite.schema_version,
        }
        if case.expected_execution_outcome is not None:
            ground_truth_payload["expected_execution_outcome"] = (
                case.expected_execution_outcome.model_dump(
                    mode="json", exclude_none=True
                )
            )
        if case.expected_action is not None:
            ground_truth_payload["expected_action"] = case.expected_action.model_dump(
                mode="json", exclude_none=True
            )
        rows.append(
            {
                "ground_truth": _canonical_json(ground_truth_payload),
                "input": _canonical_json(input_payload),
                "metadata": _row_metadata(suite, case),
            }
        )
    return rows


def build_galileo_experiment_rows(
    payload: DecisionSuite | Mapping[str, Any] | str | bytes,
    *,
    approved_fixture: bool = False,
) -> list[dict[str, Any]]:
    """Return network-free rows for Galileo custom-function experiments.

    Galileo 2.4 dataset records accept string ``input`` and ground truth fields,
    plus string-valued metadata.  Each input and ground truth is canonical JSON
    so a custom function or local scorer can parse the structured decision data.
    The function performs validation and serialization only: it does not import
    Galileo, construct clients, or perform network calls.

    Label-only suites remain available without approval. If any case contains
    ``approved_evidence``, serialization is denied unless the caller passes the
    literal boolean ``approved_fixture=True`` and
    ``AIBOM_GALILEO_ALLOW_FULL_CONTENT=true`` is present in the environment.
    Evidence content is opaque; callers must approve it before invoking this
    function because no probabilistic secret or PII filter runs here.
    """
    suite = validate_decision_suite(payload)
    if any(case.approved_evidence for case in suite.cases):
        _require_full_content_approval(approved_fixture=approved_fixture)
    return _validated_experiment_rows(suite)


class _HostedPseudonymRegistry:
    """Per-run, local-only opaque identity registry for hosted experiments.

    Random tokens are deliberately used instead of stable hashes. Repository
    names, component names, paths, case IDs, and line numbers often have small
    guessable domains; a stable digest would remain vulnerable to dictionary
    matching. The registry lives only for the duration of one experiment.
    """

    def __init__(self) -> None:
        self._tokens: dict[tuple[str, str], str] = {}
        self._used_tokens: set[str] = set()
        self._line_numbers: dict[str, int] = {}
        self._used_line_numbers: set[int] = set()

    @staticmethod
    def _key(value: Any) -> str:
        return _canonical_json({"value": value})

    def token(self, domain: str, value: Any, *, prefix: str) -> str:
        key = (domain, self._key(value))
        existing = self._tokens.get(key)
        if existing is not None:
            return existing
        for _ in range(128):
            candidate = f"{prefix}_{secrets.token_hex(12)}"
            if candidate not in self._used_tokens:
                break
        else:
            raise RuntimeError("unable to allocate a unique hosted identity token")
        self._tokens[key] = candidate
        self._used_tokens.add(candidate)
        return candidate

    def line_number(self, location: tuple[str, str, int]) -> int:
        key = self._key(location)
        existing = self._line_numbers.get(key)
        if existing is not None:
            return existing
        for _ in range(128):
            candidate = secrets.randbelow(2_147_483_646) + 1
            if candidate not in self._used_line_numbers:
                break
        else:
            raise RuntimeError("unable to allocate a unique hosted line token")
        self._line_numbers[key] = candidate
        self._used_line_numbers.add(candidate)
        return candidate


def _hosted_action(value: Any, registry: _HostedPseudonymRegistry) -> dict[str, Any]:
    action = ExpectedActionLabel.model_validate(value)
    result: dict[str, Any] = {"action": action.action}
    if action.target_type is not None:
        result["target_type"] = registry.token(
            "component-type", action.target_type, prefix="type"
        )
    # Free-form reason codes and metadata are not required by deterministic
    # action metrics and may themselves contain customer terminology.
    return result


def _hosted_candidate(value: Any, registry: _HostedPseudonymRegistry) -> dict[str, Any]:
    candidate = DecisionCandidate.model_validate(value)
    stable_id = _candidate_case_id(candidate)
    result: dict[str, Any] = {
        "component_type": registry.token(
            "component-type", candidate.component_type, prefix="type"
        ),
        "line_number": (
            registry.line_number(
                (candidate.repository, candidate.source_path, candidate.line_number)
            )
            if candidate.line_number > 0
            else 0
        ),
        "name": registry.token("component-name", candidate.name, prefix="name"),
        "repository": (
            registry.token("repository", candidate.repository, prefix="repository")
            if candidate.repository
            else ""
        ),
        "source_path": (
            registry.token("source-path", candidate.source_path, prefix="path")
            if candidate.source_path
            else ""
        ),
    }
    if stable_id:
        result["stable_case_id"] = registry.token(
            "entity-id", stable_id, prefix="entity"
        )
    # Runtime instance IDs and metadata are unnecessary for hosted scoring.
    # The local sanitizer has already used them to resolve exact identities.
    return result


def _hosted_expected_relationship(
    value: Any, registry: _HostedPseudonymRegistry
) -> dict[str, Any]:
    relationship = ExpectedRelationshipLabel.model_validate(value)
    return {
        "expected_present": relationship.expected_present,
        "relationship_type": registry.token(
            "relationship-type", relationship.relationship_type, prefix="relation"
        ),
        "source_case_id": registry.token(
            "entity-id", relationship.source_case_id, prefix="entity"
        ),
        "target_case_id": registry.token(
            "entity-id", relationship.target_case_id, prefix="entity"
        ),
    }


def _hosted_output_relationship(
    value: Any, registry: _HostedPseudonymRegistry
) -> dict[str, Any]:
    relationship = GalileoDecisionRelationship.model_validate(value)
    return {
        "predicted_present": relationship.predicted_present,
        "relationship_type": registry.token(
            "relationship-type", relationship.relationship_type, prefix="relation"
        ),
        "source_id": registry.token(
            "entity-id", relationship.source_id, prefix="entity"
        ),
        "target_id": registry.token(
            "entity-id", relationship.target_id, prefix="entity"
        ),
    }


def _hosted_expected_risk(
    value: Any, registry: _HostedPseudonymRegistry
) -> dict[str, Any]:
    risk = ExpectedRiskLabel.model_validate(value)
    result: dict[str, Any] = {
        "expected_present": risk.expected_present,
        "risk_type": registry.token("risk-type", risk.risk_type, prefix="risk"),
        "severity": risk.severity,
    }
    if risk.case_id is not None:
        result["case_id"] = registry.token("entity-id", risk.case_id, prefix="entity")
    return result


def _hosted_output_risk(
    value: Any, registry: _HostedPseudonymRegistry
) -> dict[str, Any]:
    risk = GalileoDecisionRisk.model_validate(value)
    result: dict[str, Any] = {
        "predicted_present": risk.predicted_present,
        "risk_type": registry.token("risk-type", risk.risk_type, prefix="risk"),
    }
    if risk.case_id is not None:
        result["case_id"] = registry.token("entity-id", risk.case_id, prefix="entity")
    if risk.severity is not None:
        result["severity"] = risk.severity
    return result


def _hosted_experiment_row(
    row: Mapping[str, Any], registry: _HostedPseudonymRegistry
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return a pseudonymous row and its local exact application fixture."""

    exact_input = _json_object(row.get("input"))
    exact_ground_truth = _json_object(row.get("ground_truth"))
    if exact_input is None or exact_ground_truth is None:
        raise ValueError("validated experiment row is not canonical JSON")
    raw_case_id = _normalise_nonempty(exact_input.get("case_id"))
    hosted_case_id = registry.token("entity-id", raw_case_id, prefix="entity")

    hosted_candidates = [
        _hosted_candidate(item, registry) for item in exact_input.get("candidates", [])
    ]
    hosted_input: dict[str, Any] = {
        # Evidence content is intentionally absent in the default hosted mode.
        # The application receives the exact local fixture below, while an
        # evidence judge/full trajectory requires explicit exact mode.
        "approved_evidence": [],
        "candidates": hosted_candidates,
        "case_id": hosted_case_id,
        "metadata": {},
        "schema_version": exact_input["schema_version"],
    }
    if exact_input.get("candidate") is not None:
        hosted_input["candidate"] = _hosted_candidate(
            exact_input["candidate"], registry
        )
    if exact_input.get("deterministic_relationships") is not None:
        hosted_input["deterministic_relationships"] = [
            _hosted_expected_relationship(item, registry)
            for item in exact_input["deterministic_relationships"]
        ]

    hosted_ground_truth: dict[str, Any] = {
        "case_id": hosted_case_id,
        "expected_actions": {
            registry.token("entity-id", stable_id, prefix="entity"): _hosted_action(
                action, registry
            )
            for stable_id, action in sorted(
                exact_ground_truth.get("expected_actions", {}).items()
            )
        },
        "expected_components": [
            _hosted_candidate(item, registry)
            for item in exact_ground_truth.get("expected_components", [])
        ],
        "expected_discovered_components": [
            _hosted_candidate(item, registry)
            for item in exact_ground_truth.get("expected_discovered_components", [])
        ],
        "expected_relationships": [
            _hosted_expected_relationship(item, registry)
            for item in exact_ground_truth.get("expected_relationships", [])
        ],
        "expected_risks": [
            _hosted_expected_risk(item, registry)
            for item in exact_ground_truth.get("expected_risks", [])
        ],
        "schema_version": exact_ground_truth["schema_version"],
    }
    if exact_ground_truth.get("expected_execution_outcome") is not None:
        hosted_ground_truth["expected_execution_outcome"] = deepcopy(
            exact_ground_truth["expected_execution_outcome"]
        )
    if exact_ground_truth.get("expected_action") is not None:
        hosted_ground_truth["expected_action"] = _hosted_action(
            exact_ground_truth["expected_action"], registry
        )

    repositories = {
        item["repository"] for item in hosted_candidates if item.get("repository")
    }
    single_candidate = hosted_candidates[0] if len(hosted_candidates) == 1 else None
    hosted_metadata = {
        "aibom.candidate_count": str(len(hosted_candidates)),
        "aibom.case_id": hosted_case_id,
        "aibom.component_type": (
            str(single_candidate["component_type"])
            if single_candidate is not None
            else "batch"
        ),
        "aibom.mode": "pseudonymous",
        "aibom.repository": (
            next(iter(repositories))
            if len(repositories) == 1
            else ("multiple" if repositories else "")
        ),
        "aibom.schema_version": str(exact_input["schema_version"]),
        "aibom.source_path": (
            str(single_candidate["source_path"]) if single_candidate is not None else ""
        ),
    }
    hosted_row = {
        "ground_truth": _canonical_json(hosted_ground_truth),
        "input": _canonical_json(hosted_input),
        "metadata": dict(sorted(hosted_metadata.items())),
    }
    return (
        hosted_row,
        hosted_case_id,
        deepcopy(dict(exact_input)),
        deepcopy(dict(exact_ground_truth)),
    )


def _hosted_experiment_bundle(
    rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[dict[str, Any], dict[str, Any]]],
    _HostedPseudonymRegistry,
]:
    registry = _HostedPseudonymRegistry()
    hosted_rows: list[dict[str, Any]] = []
    local_fixtures: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in rows:
        hosted_row, hosted_case_id, exact_input, exact_ground_truth = (
            _hosted_experiment_row(row, registry)
        )
        if hosted_case_id in local_fixtures:
            raise ValueError("pseudonymous case ID collision")
        hosted_rows.append(hosted_row)
        local_fixtures[hosted_case_id] = (exact_input, exact_ground_truth)
    hosted_rows.sort(key=lambda item: str(item["metadata"]["aibom.case_id"]))
    return hosted_rows, local_fixtures, registry


def _hosted_decision_output(
    serialized_output: str, registry: _HostedPseudonymRegistry
) -> str:
    parsed = _json_object(serialized_output)
    if parsed is None:
        return _serialized_decision_output(_schema_invalid_output())
    try:
        output = GalileoDecisionOutput.model_validate(parsed)
        if not output.schema_valid:
            return _serialized_decision_output(output)
        hosted_actions = None
        if output.actions is not None:
            hosted_actions = {
                registry.token("entity-id", stable_id, prefix="entity"): _hosted_action(
                    action, registry
                )
                for stable_id, action in sorted(output.actions.items())
            }
        hosted = GalileoDecisionOutput(
            schema_valid=True,
            final_components=[
                DecisionCandidate.model_validate(_hosted_candidate(item, registry))
                for item in output.final_components
            ],
            relationships=[
                GalileoDecisionRelationship.model_validate(
                    _hosted_output_relationship(item, registry)
                )
                for item in output.relationships
            ],
            risk_flags=[
                GalileoDecisionRisk.model_validate(_hosted_output_risk(item, registry))
                for item in output.risk_flags
            ],
            actions=(
                {
                    stable_id: ExpectedActionLabel.model_validate(action)
                    for stable_id, action in hosted_actions.items()
                }
                if hosted_actions is not None
                else None
            ),
            execution_outcome=output.execution_outcome,
        )
        return _serialized_decision_output(hosted)
    except Exception:
        return _serialized_decision_output(_schema_invalid_output())


_MISSING = object()


def _object_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, BaseModel):
        return cast(Mapping[str, Any], value.model_dump(mode="python"))
    return None


def _object_value(value: Any, *names: str, default: Any = _MISSING) -> Any:
    mapping = _object_mapping(value)
    if mapping is not None:
        for name in names:
            if name in mapping:
                return mapping[name]
        return default
    for name in names:
        try:
            if hasattr(value, name):
                return getattr(value, name)
        except Exception:
            continue
    return default


def _output_alias_value(
    value: Any,
    *names: str,
    default: Any = _MISSING,
) -> Any:
    mapping = _object_mapping(value)
    if mapping is not None:
        present = [name for name in names if name in mapping]
        if len(present) > 1:
            raise ValueError(f"output must use only one of {', '.join(names)}")
        return mapping[present[0]] if present else default
    return _object_value(value, *names, default=default)


def _output_items(value: Any, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")
    return list(value)


def _project_output_candidate(value: Any) -> dict[str, Any]:
    component_type = _object_value(value, "component_type", "type")
    name = _object_value(value, "name")
    if component_type is _MISSING or name is _MISSING:
        raise ValueError("output components require component_type and name")

    projected: dict[str, Any] = {
        "component_type": component_type,
        "name": name,
    }
    aliases = {
        "repository": ("repository", "repo", "source_repo"),
        "source_path": ("source_path", "file_path", "file"),
        "line_number": ("line_number", "line"),
        "stable_case_id": ("stable_case_id", "case_id", "eval_case_id"),
        "instance_id": ("instance_id", "id"),
    }
    for canonical_name, source_names in aliases.items():
        item = _object_value(value, *source_names)
        if item is not _MISSING and item is not None:
            projected[canonical_name] = item

    if "stable_case_id" not in projected:
        metadata = _object_value(value, "metadata", default={})
        if isinstance(metadata, Mapping):
            for key in ("stable_case_id", "case_id", "eval_case_id"):
                if metadata.get(key) is not None and str(metadata[key]).strip():
                    projected["stable_case_id"] = metadata[key]
                    break
    agentic_hint = _object_value(value, "agentic_hint", default="")
    if str(agentic_hint or "").strip():
        # Internal-only signal used to avoid crediting a degraded passthrough
        # as a keep. It is removed before the sanitized output is serialized.
        projected["_agentic_degraded"] = True
    return projected


def _project_output_relationship(value: Any) -> dict[str, Any]:
    relationship_type = _object_value(value, "relationship_type", "type", "label")
    source_id = _object_value(
        value,
        "source_case_id",
        "source_instance_id",
        "source_id",
        "source",
    )
    target_id = _object_value(
        value,
        "target_case_id",
        "target_instance_id",
        "target_id",
        "target",
    )
    if _MISSING in (relationship_type, source_id, target_id):
        raise ValueError("output relationships require type, source, and target")
    projected = {
        "relationship_type": relationship_type,
        "source_id": source_id,
        "target_id": target_id,
    }
    present = _object_value(
        value,
        "predicted_present",
        "present",
        "is_present",
    )
    if present is not _MISSING:
        projected["predicted_present"] = present
    return projected


def _project_output_risk(value: Any) -> dict[str, Any]:
    risk_type = _object_value(value, "risk_type", "risk_flag", "flag", "type", "name")
    if risk_type is _MISSING:
        raise ValueError("output risks require risk_type")
    projected: dict[str, Any] = {"risk_type": risk_type}
    case_id = _object_value(
        value,
        "case_id",
        "candidate_case_id",
        "component_case_id",
        "stable_case_id",
    )
    severity = _object_value(value, "severity")
    source_path = _object_value(value, "source_path", "file_path", "file")
    line_number = _object_value(value, "line_number", "line")
    present = _object_value(
        value,
        "predicted_present",
        "present",
        "is_present",
    )
    if case_id is not _MISSING and case_id is not None:
        projected["case_id"] = case_id
    if severity is not _MISSING and severity is not None:
        projected["severity"] = severity
    if source_path is not _MISSING and source_path is not None:
        projected["_source_path"] = source_path
    if line_number is not _MISSING and line_number is not None:
        projected["_line_number"] = line_number
    if present is not _MISSING:
        projected["predicted_present"] = present
    return projected


def _project_output_action(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"action": value}
    action = _object_value(value, "action", "decision")
    if action is _MISSING:
        raise ValueError("output actions require action")
    projected = {"action": action}
    target_type = _object_value(value, "target_type", "new_type")
    if target_type is not _MISSING and target_type is not None:
        projected["target_type"] = target_type
    return projected


def _project_execution_outcome(value: Any) -> dict[str, Any]:
    if _object_mapping(value) is None:
        raise ValueError("execution_outcome must be an object")
    projected: dict[str, Any] = {}
    aliases = {
        "status": ("status", "execution_status"),
        "schema_valid": ("schema_valid",),
        "abstained": ("abstained", "abstention"),
        "degraded_candidate_count": (
            "degraded_candidate_count",
            "degraded_candidates",
        ),
        "retry_count": ("retry_count", "retries"),
        "fallback_count": ("fallback_count", "fallbacks"),
        "cache_hit": ("cache_hit",),
        "tool_error_count": ("tool_error_count", "tool_errors"),
        "guard_denial_count": ("guard_denial_count", "guard_denials"),
    }
    for field_name, source_names in aliases.items():
        item = _object_value(value, *source_names)
        if item is not _MISSING and item is not None:
            projected[field_name] = item
    return ExecutionOutcome.model_validate(projected).model_dump(
        mode="json", exclude_none=True
    )


def _json_object(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    return _object_mapping(value)


def _candidate_records(payload: Any, *field_names: str) -> list[DecisionCandidate]:
    parsed = _json_object(payload)
    if parsed is None:
        return []
    raw_candidates: Any = None
    for field_name in field_names:
        if parsed.get(field_name) is not None:
            raw_candidates = parsed[field_name]
            break
    if raw_candidates is None and "candidates" in field_names:
        candidate = parsed.get("candidate")
        if candidate is not None:
            raw_candidates = [candidate]
    if not isinstance(raw_candidates, (list, tuple)):
        return []
    candidates: list[DecisionCandidate] = []
    for item in raw_candidates:
        try:
            candidates.append(DecisionCandidate.model_validate(item))
        except Exception:
            return []
    return candidates


def _sanitization_candidates(dataset_input: Any) -> list[DecisionCandidate]:
    return _candidate_records(dataset_input, "candidates")


def _sanitization_expected_components(
    dataset_ground_truth: Any,
) -> list[DecisionCandidate]:
    return _candidate_records(dataset_ground_truth, "expected_components")


def _path_matches_registry(raw_path: Any, approved_path: str) -> bool:
    raw = str(raw_path or "").strip().replace("\\", "/")
    approved = approved_path.strip().replace("\\", "/")
    if not raw or not approved:
        return raw == approved
    normalized = posixpath.normpath(raw)
    return normalized == approved or normalized.endswith(f"/{approved}")


def _projected_candidate_matches(
    projected: Mapping[str, Any],
    candidate: DecisionCandidate,
    *,
    include_type: bool,
) -> bool:
    try:
        line_number = int(projected.get("line_number", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    if _normalise_nonempty(projected.get("name", "")).casefold() != (
        candidate.name.casefold()
    ):
        return False
    if include_type and _normalise_label(projected.get("component_type")) != (
        candidate.component_type
    ):
        return False
    if line_number != candidate.line_number or not _path_matches_registry(
        projected.get("source_path", ""), candidate.source_path
    ):
        return False
    raw_repository = str(projected.get("repository", "")).strip()
    return (
        not raw_repository
        or raw_repository.casefold() == candidate.repository.casefold()
    )


def _contextualize_output_candidates(
    projected: list[dict[str, Any]],
    *,
    dataset_input: Any,
    dataset_ground_truth: Any,
) -> tuple[list[dict[str, Any]], dict[str, str], set[str]]:
    deterministic = _sanitization_candidates(dataset_input)
    expected = _sanitization_expected_components(dataset_ground_truth)

    by_stable_id = {
        _candidate_case_id(candidate): candidate
        for candidate in [*deterministic, *expected]
        if _candidate_case_id(candidate)
    }
    by_instance_id = {
        candidate.instance_id: candidate
        for candidate in [*deterministic, *expected]
        if candidate.instance_id
    }
    repositories = {
        candidate.repository
        for candidate in [*deterministic, *expected]
        if candidate.repository
    }
    default_repository = next(iter(repositories)) if len(repositories) == 1 else ""

    contextualized: list[dict[str, Any]] = []
    runtime_to_stable: dict[str, str] = {}
    degraded_case_ids: set[str] = set()
    for component in projected:
        item = dict(component)
        degraded = bool(item.pop("_agentic_degraded", False))
        match = None
        stable_id = str(item.get("stable_case_id", "")).strip()
        instance_id = str(item.get("instance_id", "")).strip()
        if stable_id:
            stable_match = by_stable_id.get(stable_id)
            # A stable ID is not a license to copy identity from the label.
            # Normalize native absolute paths only after every supplied entity
            # field agrees with the registered repository-relative identity.
            if stable_match is not None and _projected_candidate_matches(
                item, stable_match, include_type=True
            ):
                match = stable_match
        elif instance_id:
            instance_match = by_instance_id.get(instance_id)
            if instance_match is not None and _projected_candidate_matches(
                item, instance_match, include_type=False
            ):
                match = instance_match
        if match is None and not stable_id:
            final_matches = [
                candidate
                for candidate in expected
                if _projected_candidate_matches(item, candidate, include_type=True)
            ]
            if len(final_matches) == 1:
                match = final_matches[0]
        if match is None and not stable_id:
            baseline_matches = [
                candidate
                for candidate in deterministic
                if _projected_candidate_matches(item, candidate, include_type=False)
            ]
            if len(baseline_matches) == 1:
                match = baseline_matches[0]
        if match is not None:
            matched_case_id = _candidate_case_id(match)
            item["stable_case_id"] = matched_case_id
            item["repository"] = match.repository
            item["source_path"] = match.source_path
            item["line_number"] = match.line_number
            if instance_id and matched_case_id:
                runtime_to_stable[instance_id] = matched_case_id
            if degraded and matched_case_id:
                degraded_case_ids.add(matched_case_id)
        elif default_repository:
            item["repository"] = default_repository
        raw_path = str(item.get("source_path", "")).strip().replace("\\", "/")
        if posixpath.isabs(raw_path) or raw_path == ".." or raw_path.startswith("../"):
            item["source_path"] = ""
        # Native AIComponent instance IDs embed the source path. They are used
        # only for in-memory relationship alignment and never serialized.
        item.pop("instance_id", None)
        contextualized.append(item)
    return contextualized, runtime_to_stable, degraded_case_ids


def _inferred_output_actions(
    components: list[dict[str, Any]],
    *,
    dataset_input: Any,
    degraded_case_ids: set[str],
) -> dict[str, dict[str, Any]] | None:
    """Infer identity actions while treating degraded passthroughs as abstentions.

    Substantive ``enrich`` cannot be inferred from the content-minimized suite
    identity alone; callers must provide explicit actions for that dimension.
    """

    deterministic = _sanitization_candidates(dataset_input)
    if not deterministic:
        return None
    baseline_by_id = {
        _candidate_case_id(candidate): candidate
        for candidate in deterministic
        if _candidate_case_id(candidate)
    }
    final_models = [DecisionCandidate.model_validate(item) for item in components]
    final_by_id = {
        _candidate_case_id(candidate): candidate
        for candidate in final_models
        if _candidate_case_id(candidate)
    }
    actions: dict[str, dict[str, Any]] = {}
    for stable_id, baseline in sorted(baseline_by_id.items()):
        if stable_id in degraded_case_ids:
            continue
        final = final_by_id.get(stable_id)
        if final is None:
            actions[stable_id] = {"action": "remove"}
        elif final.component_type != baseline.component_type:
            actions[stable_id] = {
                "action": "reclassify",
                "target_type": final.component_type,
            }
        else:
            actions[stable_id] = {"action": "keep"}
    for stable_id in sorted(set(final_by_id) - set(baseline_by_id)):
        if stable_id not in degraded_case_ids:
            actions[stable_id] = {"action": "discover"}
    return actions


def _contextualize_output_relationships(
    relationships: list[dict[str, Any]],
    runtime_to_stable: Mapping[str, str],
) -> list[dict[str, Any]]:
    contextualized: list[dict[str, Any]] = []
    for relationship in relationships:
        item = dict(relationship)
        for endpoint in ("source_id", "target_id"):
            raw_identifier = str(item.get(endpoint, "")).strip()
            if raw_identifier in runtime_to_stable:
                item[endpoint] = runtime_to_stable[raw_identifier]
        contextualized.append(item)
    return contextualized


def _contextualize_output_risks(
    risks: list[dict[str, Any]],
    *,
    dataset_input: Any,
    dataset_ground_truth: Any,
) -> list[dict[str, Any]]:
    candidates = [
        *_sanitization_candidates(dataset_input),
        *_sanitization_expected_components(dataset_ground_truth),
    ]
    contextualized: list[dict[str, Any]] = []
    for risk in risks:
        item = dict(risk)
        if not str(item.get("case_id", "")).strip():
            raw_path = item.get("_source_path", "")
            try:
                raw_line = int(item.get("_line_number", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                raw_line = -1
            location_matches = {
                _candidate_case_id(candidate)
                for candidate in candidates
                if _candidate_case_id(candidate)
                and raw_line == candidate.line_number
                and _path_matches_registry(raw_path, candidate.source_path)
            }
            if len(location_matches) == 1:
                item["case_id"] = next(iter(location_matches))
        item.pop("_source_path", None)
        item.pop("_line_number", None)
        contextualized.append(item)
    return contextualized


def _schema_invalid_output(
    execution_outcome: Mapping[str, Any] | None = None,
) -> GalileoDecisionOutput:
    return GalileoDecisionOutput(
        schema_valid=False,
        execution_outcome=(
            ExecutionOutcome.model_validate(execution_outcome)
            if execution_outcome is not None
            else None
        ),
    )


def adapt_pipeline_result_for_galileo(
    result: Any,
    *,
    execution_outcome: ExecutionOutcome | Mapping[str, Any] | None = None,
    actions: Mapping[str, Any] | None = None,
    dataset_input: Any = None,
) -> dict[str, Any]:
    """Wrap a four-stage pipeline result in the evaluation output contract.

    Entity fields are passed only to the local sanitizer, which projects them
    into the bounded schema before Galileo sees them. The native result's
    authoritative ``agentic_degraded_count`` is added automatically. Callers
    must supply operational facts that the pipeline result does not expose
    (for example retry/fallback/cache/tool-guard counts or a provider-specific
    terminal status) through ``execution_outcome``. Pass explicit ``actions``
    whenever the fixture scores substantive enrichment because identity-only
    output cannot distinguish ``enrich`` from ``keep``. Supplying
    ``dataset_input`` also lets the adapter mark total degradation as an
    abstention without consulting ground-truth labels.
    """

    components = _object_value(result, "components", "final_components")
    if components is _MISSING:
        raise ValueError("pipeline result requires components")
    relationships = _object_value(result, "relationships", default=[])
    risks = _object_value(
        result,
        "agentic_risk_flags",
        "risk_flags",
        "risks",
        default=[],
    )
    result_actions = _object_value(result, "actions", default=_MISSING)
    if result_actions is None:
        result_actions = _MISSING
    if actions is not None and result_actions is not _MISSING:
        raise ValueError("actions were supplied both explicitly and by the result")

    if execution_outcome is None:
        outcome_payload: dict[str, Any] = {}
    elif isinstance(execution_outcome, ExecutionOutcome):
        outcome_payload = execution_outcome.model_dump(mode="python", exclude_none=True)
    elif isinstance(execution_outcome, Mapping):
        outcome_payload = dict(execution_outcome)
    else:
        raise TypeError("execution_outcome must be an object")

    native_degraded = _object_value(result, "agentic_degraded_count", default=_MISSING)
    if native_degraded is not _MISSING:
        outcome_payload.setdefault("degraded_candidate_count", native_degraded)
        try:
            degraded_count = int(native_degraded)
        except (TypeError, ValueError, OverflowError):
            degraded_count = -1
        if degraded_count >= 0:
            outcome_payload.setdefault(
                "status", "degraded" if degraded_count else "success"
            )
            candidate_count = len(_sanitization_candidates(dataset_input))
            if (
                degraded_count > 0
                and candidate_count > 0
                and (degraded_count >= candidate_count)
            ):
                outcome_payload.setdefault("abstained", True)

    validated_outcome = (
        ExecutionOutcome.model_validate(outcome_payload) if outcome_payload else None
    )
    adapted: dict[str, Any] = {
        "final_components": components,
        "relationships": relationships,
        "risk_flags": risks,
        "schema_valid": (
            validated_outcome.schema_valid
            if validated_outcome is not None
            and validated_outcome.schema_valid is not None
            else True
        ),
    }
    if actions is not None:
        adapted["actions"] = dict(actions)
    elif result_actions is not _MISSING:
        adapted["actions"] = result_actions
    if validated_outcome is not None:
        adapted["execution_outcome"] = validated_outcome.model_dump(
            mode="python", exclude_none=True
        )
    return adapted


def _serialized_decision_output(output: GalileoDecisionOutput) -> str:
    payload = output.model_dump(mode="json")
    if output.execution_outcome is None:
        payload.pop("execution_outcome", None)
    else:
        payload["execution_outcome"] = output.execution_outcome.model_dump(
            mode="json", exclude_none=True
        )
    return _canonical_json(payload)


def sanitize_galileo_decision_output(
    value: Any,
    *,
    dataset_input: Any = None,
    dataset_ground_truth: Any = None,
) -> str:
    """Project arbitrary application output into the safe scoring contract.

    Only component identities, directed relationship identities, risk labels,
    and action labels survive. Unknown top-level fields, model descriptions,
    source snippets, prompts, tool payloads, and exception details are never
    serialized. Invalid structures produce a deterministic empty failure
    envelope so the schema-validity metric can report ``0`` without echoing the
    rejected value.
    """
    safe_execution: dict[str, Any] | None = None
    try:
        schema_version = _object_value(value, "schema_version")
        if (
            schema_version is not _MISSING
            and schema_version != GALILEO_DECISION_OUTPUT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported output schema")
        schema_valid = _object_value(value, "schema_valid", default=True)
        execution_value = _output_alias_value(
            value, "execution_outcome", "outcome", default=None
        )
        if execution_value is not None:
            safe_execution = _project_execution_outcome(execution_value)
        if schema_valid is False:
            if safe_execution is not None:
                safe_execution["schema_valid"] = False
            return _serialized_decision_output(_schema_invalid_output(safe_execution))
        if schema_valid is not True:
            raise ValueError("schema_valid must be a boolean")

        component_values = _output_alias_value(value, "final_components", "components")
        if component_values is _MISSING:
            raise ValueError("output requires final_components or components")
        relationship_values = _object_value(value, "relationships", default=[])
        risk_values = _output_alias_value(
            value,
            "risk_flags",
            "risks",
            "agentic_risk_flags",
            default=[],
        )
        action_values = _object_value(value, "actions", default=None)

        components = [
            _project_output_candidate(item)
            for item in _output_items(component_values, field_name="final_components")
        ]
        components, runtime_to_stable, degraded_case_ids = (
            _contextualize_output_candidates(
                components,
                dataset_input=dataset_input,
                dataset_ground_truth=dataset_ground_truth,
            )
        )
        relationships = [
            _project_output_relationship(item)
            for item in _output_items(relationship_values, field_name="relationships")
        ]
        relationships = _contextualize_output_relationships(
            relationships, runtime_to_stable
        )
        risks = [
            _project_output_risk(item)
            for item in _output_items(risk_values, field_name="risk_flags")
        ]
        risks = _contextualize_output_risks(
            risks,
            dataset_input=dataset_input,
            dataset_ground_truth=dataset_ground_truth,
        )
        actions = None
        if action_values is not None:
            if not isinstance(action_values, Mapping):
                raise ValueError("actions must be an object keyed by stable case ID")
            actions = {
                _normalise_nonempty(key): _project_output_action(item)
                for key, item in action_values.items()
            }
            deterministic_ids = {
                _candidate_case_id(candidate)
                for candidate in _sanitization_candidates(dataset_input)
                if _candidate_case_id(candidate)
            }
            final_ids = {
                _candidate_case_id(DecisionCandidate.model_validate(component))
                for component in components
                if _candidate_case_id(DecisionCandidate.model_validate(component))
            }
            if deterministic_ids and not set(actions).issubset(
                deterministic_ids | final_ids
            ):
                # Suite labels are exhaustive. An action for an entity that
                # exists in neither the deterministic input nor final output
                # is a malformed prediction, not a free true negative.
                raise ValueError("actions contain an unknown stable case ID")
        else:
            actions = _inferred_output_actions(
                components,
                dataset_input=dataset_input,
                degraded_case_ids=degraded_case_ids,
            )

        output = GalileoDecisionOutput.model_validate(
            {
                "actions": actions,
                "execution_outcome": safe_execution,
                "final_components": components,
                "relationships": relationships,
                "risk_flags": risks,
                "schema_valid": True,
                "schema_version": GALILEO_DECISION_OUTPUT_SCHEMA_VERSION,
            }
        )
    except Exception:
        if safe_execution is not None:
            safe_execution["schema_valid"] = False
        output = _schema_invalid_output(safe_execution)
    return _serialized_decision_output(output)


_PRIMARY_METRIC_NAMES = (
    "aibom.components.precision",
    "aibom.components.recall",
    "aibom.components.f1",
    "aibom.discoveries.precision",
    "aibom.discoveries.recall",
    "aibom.discoveries.f1",
    "aibom.net_recall_lift",
    "aibom.over_prune_rate",
    "aibom.relationships.precision",
    "aibom.relationships.recall",
    "aibom.relationships.f1",
    "aibom.risks.precision",
    "aibom.risks.recall",
    "aibom.risks.f1",
    "aibom.relationship_recall_lift",
    "aibom.action_accuracy",
    "aibom.action_macro_f1",
    "aibom.decision_coverage",
    "aibom.reclassification_accuracy",
    "aibom.schema_validity",
    "aibom.execution.status_accuracy",
    "aibom.execution.schema_validity_accuracy",
    "aibom.execution.abstention_accuracy",
    "aibom.execution.degraded_count_accuracy",
    "aibom.execution.retry_count_accuracy",
    "aibom.execution.fallback_count_accuracy",
    "aibom.execution.cache_hit_accuracy",
    "aibom.execution.tool_error_count_accuracy",
    "aibom.execution.guard_denial_count_accuracy",
)


def _canonical_trace_output(value: Any) -> GalileoDecisionOutput | None:
    try:
        mapping = _json_object(value)
        if mapping is None:
            return None
        return GalileoDecisionOutput.model_validate(mapping)
    except Exception:
        return None


def _execution_outcome_metrics(
    output: GalileoDecisionOutput | None,
    expected: ExecutionOutcome | None,
) -> dict[str, float]:
    if expected is None:
        return {}
    predicted = output.execution_outcome if output is not None else None
    fields = {
        "status": "aibom.execution.status_accuracy",
        "schema_valid": "aibom.execution.schema_validity_accuracy",
        "abstained": "aibom.execution.abstention_accuracy",
        "degraded_candidate_count": "aibom.execution.degraded_count_accuracy",
        "retry_count": "aibom.execution.retry_count_accuracy",
        "fallback_count": "aibom.execution.fallback_count_accuracy",
        "cache_hit": "aibom.execution.cache_hit_accuracy",
        "tool_error_count": "aibom.execution.tool_error_count_accuracy",
        "guard_denial_count": "aibom.execution.guard_denial_count_accuracy",
    }
    scores: dict[str, float] = {}
    for field_name, metric_name in fields.items():
        expected_value = getattr(expected, field_name)
        if expected_value is None:
            continue
        if field_name == "schema_valid":
            predicted_value: Any = bool(output is not None and output.schema_valid)
        else:
            predicted_value = (
                getattr(predicted, field_name) if predicted is not None else None
            )
        scores[metric_name] = float(predicted_value == expected_value)
    return scores


def _trace_decision_metrics(trace: Any) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {name: None for name in _PRIMARY_METRIC_NAMES}
    output = _canonical_trace_output(getattr(trace, "output", None))
    metrics["aibom.schema_validity"] = float(output is not None and output.schema_valid)

    dataset_input = _json_object(getattr(trace, "dataset_input", None))
    ground_truth = _json_object(getattr(trace, "dataset_output", None))
    if dataset_input is None or ground_truth is None:
        return metrics

    try:
        case_payload: dict[str, Any] = {
            "case_id": ground_truth.get("case_id", dataset_input.get("case_id")),
        }
        for key in (
            "candidate",
            "candidates",
            "deterministic_relationships",
            "baseline_relationships",
        ):
            if key in dataset_input:
                case_payload[key] = dataset_input[key]
        for key in (
            "expected_action",
            "expected_actions",
            "expected_components",
            "expected_discovered_components",
            "expected_relationships",
            "expected_risks",
            "expected_execution_outcome",
            "expected_outcome",
        ):
            if key in ground_truth:
                case_payload[key] = ground_truth[key]
        case = DecisionSuiteCase.model_validate(case_payload)
        metrics.update(
            _execution_outcome_metrics(output, case.expected_execution_outcome)
        )
        if output is None or not output.schema_valid:
            return metrics

        from aibom.decision_evaluation import evaluate_decisions

        evaluation = evaluate_decisions(
            predicted_components=output.final_components,
            expected_components=case.expected_components,
            predicted_relationships=output.relationships,
            expected_relationships=case.expected_relationships,
            predicted_risks=output.risk_flags,
            expected_risks=case.expected_risks,
            deterministic_components=case.candidates,
            deterministic_relationships=case.deterministic_relationships,
            predicted_actions=output.actions,
            expected_actions=case.expected_actions,
        )
        metrics.update(evaluation.to_galileo_metrics())

        # An exact true-empty set is not a zero-quality result.  Keep it out of
        # Galileo's row averages instead of treating the evaluator's neutral
        # 0/0/0 convention as a failure.
        for prefix, entity_metric in (
            ("aibom.components", evaluation.components),
            ("aibom.relationships", evaluation.relationships),
            ("aibom.risks", evaluation.risks),
            ("aibom.discoveries", evaluation.discoveries),
        ):
            if (
                entity_metric is not None
                and entity_metric.predicted_count == 0
                and entity_metric.expected_count == 0
            ):
                for suffix in ("precision", "recall", "f1"):
                    metrics[f"{prefix}.{suffix}"] = None
    except Exception:
        return metrics
    return metrics


def _score_trace_metric(trace: Any, metric_name: str) -> float | None:
    return _trace_decision_metrics(trace)[metric_name]


def build_galileo_decision_metrics() -> list[Any]:
    """Build Galileo 2.4 trace-level ``LocalMetric`` instances lazily.

    Calling this factory is the only deterministic-metric path that imports the
    optional Galileo SDK. Each scorer reads ``trace.dataset_input``,
    ``trace.dataset_output``, and the canonical ``trace.output`` and delegates
    entity matching to :func:`aibom.decision_evaluation.evaluate_decisions`.
    Dimensions that are not represented in a row (notably relationship lift
    without deterministic relationship labels) return ``None``.
    """
    try:
        from galileo import LocalMetric, StepType
    except (ImportError, ModuleNotFoundError) as exc:
        raise GalileoIntegrationUnavailable(
            "The optional Galileo metric integration is unavailable; install "
            "the AIBOM observability extra before evaluation"
        ) from exc

    metrics: list[Any] = []
    for metric_name in _PRIMARY_METRIC_NAMES:
        metrics.append(
            LocalMetric(
                name=metric_name,
                scorer_fn=(
                    lambda trace, name=metric_name: _score_trace_metric(trace, name)
                ),
                scorable_types=[StepType.trace],
                aggregatable_types=[StepType.trace],
                description=(
                    "Deterministic AIBOM entity-level decision metric; unavailable "
                    "dimensions return null."
                ),
                tags=["aibom", "deterministic", "agentic-decision"],
            )
        )
    return metrics


def build_aibom_evidence_grounding_metric(
    *,
    approved_fixture: bool = False,
    judge_model: str,
    judges: int = 1,
) -> Any:
    """Construct the approval-gated diagnostic evidence-grounding judge.

    This helper only constructs an unsynchronized Galileo 2.4 ``LlmMetric``;
    it never calls ``.create()`` and performs no API request. The caller must
    deliberately persist/configure the judge in an access-restricted
    evaluation project. It is not included in the default deterministic metric
    set and must not be used as an authoritative entity-accuracy score.

    Because the trace-level prompt receives approved evidence excerpts, the
    same two full-content gates used by the LangChain callback are required and
    checked before importing the optional Galileo SDK.
    """
    _require_full_content_approval(approved_fixture=approved_fixture)
    if not isinstance(judge_model, str) or not judge_model.strip():
        raise ValueError("judge_model must be an explicit non-empty model alias")
    if isinstance(judges, bool) or not isinstance(judges, int) or judges < 1:
        raise ValueError("judges must be a positive integer")

    try:
        from galileo import LlmMetric, StepType
    except (ImportError, ModuleNotFoundError) as exc:
        raise GalileoIntegrationUnavailable(
            "The optional Galileo LLM metric integration is unavailable; install "
            "the AIBOM observability extra before evaluation"
        ) from exc

    return LlmMetric(
        name=AIBOM_EVIDENCE_GROUNDING_METRIC_NAME,
        prompt=AIBOM_EVIDENCE_GROUNDING_PROMPT,
        model=judge_model.strip(),
        judges=judges,
        node_level=StepType.trace,
        cot_enabled=False,
        output_type="percentage",
        ground_truth=True,
        description=(
            "Diagnostic judge for evidence support of AIBOM component, action, "
            "relationship, and risk decisions."
        ),
        tags=["aibom", "diagnostic", "evidence-grounding"],
    )


def create_galileo_async_callback(
    *,
    approved_fixture: bool = False,
    galileo_logger: Any | None = None,
    start_new_trace: bool = True,
    flush_on_chain_end: bool = True,
    ingestion_hook: Callable[[Any], None] | None = None,
    session_binder: Callable[[Any], None] | None = None,
) -> Any:
    """Construct Galileo 2.4's LangChain async callback after explicit approval.

    The callback captures full prompts, responses, tool inputs, and tool outputs.
    Construction is therefore denied unless all conditions are true:

    1. the caller passes the literal boolean ``approved_fixture=True``; and
    2. ``AIBOM_GALILEO_ALLOW_FULL_CONTENT=true`` is present in the environment;
    3. ``AIBOM_GALILEO_ALLOW_FULL_TRAJECTORY=true`` independently approves
       prompts, responses, tool I/O, callback metadata, and exception details;
    4. ``GALILEO_CONSOLE_URL`` names the hosted HTTPS origin and
       ``AIBOM_GALILEO_ALLOW_PUBLIC_CLOUD=true`` approves that egress; and
    5. immutable evaluation project and log-stream UUIDs are configured.

    The approval checks happen before the optional Galileo/LangChain import.
    This function does not invoke a chain or explicitly flush/send any data.
    Logger initialization and optional session binding run in a daemon worker
    and are bounded by ``AIBOM_GALILEO_SETUP_BUDGET_S`` (2 seconds by default).
    A logger that completes after that deadline is hardened and discarded.

    ``session_binder``, when supplied, receives the freshly built logger and may
    attach it to a shared session so that every callback built for one scan
    groups its trace under a single Galileo session. It is called before the
    logger is wrapped; a failure inside it must not silently drop the approved
    destination, so it is allowed to raise.
    """
    _require_full_trajectory_approval(approved_fixture=approved_fixture)
    hosted_url = _require_hosted_galileo_destination()
    if galileo_logger is not None or ingestion_hook is not None:
        raise HostedGalileoDestinationRequired(
            "Full-content evaluation does not accept an external Galileo logger "
            "or ingestion hook because its destination cannot be verified"
        )
    if start_new_trace is not True or flush_on_chain_end is not True:
        raise HostedGalileoDestinationRequired(
            "The approved callback requires start_new_trace=True and "
            "flush_on_chain_end=True so every trajectory is ingested exactly once"
        )

    project_id = _required_resource_id(EVALUATION_PROJECT_ID_ENV_VAR)
    log_stream_id = _required_resource_id(EVALUATION_LOG_STREAM_ID_ENV_VAR)

    try:
        from galileo import GalileoLogger
        from galileo.handlers.langchain import (
            GalileoAsyncCallback,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise GalileoIntegrationUnavailable(
            "The optional Galileo LangChain integration is unavailable; install "
            "the AIBOM observability extra before approved evaluation"
        ) from exc

    _verify_loaded_galileo_destination(hosted_url)
    from .agentic_telemetry import (
        _disable_agent_control,
        _disable_logger_atexit_flush,
        _resolve_setup_budget_s,
    )

    setup = _CallbackSetupAttempt()

    def _build_callback() -> None:
        logger: Any | None = None
        callback: Any | None = None
        error: BaseException | None = None
        try:
            one_shot_logger_class = _one_shot_logger_class(GalileoLogger)
            logger = one_shot_logger_class(
                project_id=project_id,
                log_stream_id=log_stream_id,
                mode="batch",
            )

            # Remove autonomous SDK lifecycle behavior before any raw span can
            # be created. A logger that completed after the caller's deadline
            # is still hardened before it can start a remote session.
            _disable_logger_atexit_flush(logger)
            if not _disable_agent_control(logger):
                raise HostedGalileoDestinationRequired(
                    "Galileo Agent Control could not be disabled for the approved "
                    "full-trajectory callback"
                )
            if not callable(getattr(logger, "async_flush", None)):
                raise HostedGalileoDestinationRequired(
                    "The Galileo callback logger does not expose async one-shot "
                    "ingestion"
                )
            if (
                str(getattr(logger, "project_id", "")) != project_id
                or str(getattr(logger, "log_stream_id", "")) != log_stream_id
            ):
                raise HostedGalileoDestinationRequired(
                    "The Galileo callback logger did not retain the approved "
                    "project and log-stream IDs"
                )

            with setup.lock:
                abandoned = setup.abandoned
            if abandoned:
                raise GalileoIntegrationUnavailable(
                    "Galileo callback setup exceeded its setup budget"
                )
            if session_binder is not None:
                session_binder(logger)
            callback = GalileoAsyncCallback(
                galileo_logger=logger,
                start_new_trace=True,
                flush_on_chain_end=True,
                ingestion_hook=None,
            )
        except BaseException as exc:  # noqa: BLE001 - relayed to caller
            error = exc
            if logger is not None:
                _discard_raw_logger(logger)

        late_logger: Any | None = None
        with setup.lock:
            if setup.abandoned:
                late_logger = logger
            else:
                setup.callback = callback
                setup.logger = logger
                setup.error = error
            setup.done.set()
        if late_logger is not None:
            _discard_raw_logger(late_logger)

    threading.Thread(
        target=_build_callback,
        name="aibom-galileo-callback-setup",
        daemon=True,
    ).start()

    if not setup.done.wait(_resolve_setup_budget_s()):
        completed_logger: Any | None = None
        with setup.lock:
            setup.abandoned = True
            # Resolve the boundary race where the worker published its callback
            # just as Event.wait reached the deadline.
            completed_logger = setup.logger
            setup.callback = None
            setup.logger = None
        if completed_logger is not None:
            _discard_raw_logger(completed_logger)
        raise GalileoIntegrationUnavailable(
            "Galileo callback setup exceeded its setup budget; raw trajectory "
            "disabled for this invocation"
        )

    with setup.lock:
        callback = setup.callback
        error = setup.error
        setup.callback = None
        setup.logger = None
        setup.error = None
    if error is not None:
        if isinstance(error, Exception):
            raise error
        raise GalileoIntegrationUnavailable(
            "Galileo callback setup failed with a fatal worker error"
        ) from error
    if callback is None:
        raise GalileoIntegrationUnavailable(
            "Galileo callback setup returned no callback"
        )
    return callback


def create_full_trajectory_callback_factory(
    *,
    session_name: str,
    session_external_id: str | None = None,
) -> Callable[[], Any]:
    """Return a scan-scoped factory that groups every full-trajectory callback
    under a single Galileo session.

    The CLI attaches one full-trajectory callback per live agent invocation so
    that each trajectory is flushed independently and no mutable trace state is
    shared across concurrent batches. Without coordination, each of those
    loggers would emit a top-level trace with no session, so one ``analyze`` run
    fans out into many sessions in the Galileo UI.

    This factory keeps that per-invocation isolation — every call still builds a
    fresh logger and callback — while sharing only the immutable resolved
    session id. The first logger creates the session and every later logger
    attaches to it via ``set_session``. The SDK's ``external_id`` de-duplication
    is a non-atomic search-then-create, so only one remote creation may remain in
    flight. Session creation is bounded by the sanitized path's setup budget
    (``AIBOM_GALILEO_SETUP_BUDGET_S``, 2s default). A caller that reaches that
    budget receives no raw callback; later callers wait for the same in-flight
    operation instead of creating a duplicate session.

    The returned callable takes no arguments and satisfies the CLI's
    ``invoke_callback_factory`` contract; it raises exactly like
    :func:`create_galileo_async_callback` when a gate is not satisfied.
    """
    from .agentic_telemetry import _resolve_setup_budget_s

    lock = threading.Lock()
    state = _FullTrajectorySessionState()

    def _start_session(logger: Any) -> Any:
        """Use Galileo 2.4's coroutine on this worker so its client is closable."""
        async_start = getattr(logger, "_start_or_get_session_async", None)
        if not callable(async_start):
            # Test doubles and older compatible SDKs may expose only the public
            # method. They do not have Galileo 2.4's per-thread IngestTraces
            # client, so there is no private client to close here.
            return logger.start_session(
                name=session_name,
                external_id=session_external_id,
                metadata={"component": "agentic-classifier-full-trajectory"},
            )

        async def _run() -> Any:
            try:
                return await async_start(
                    name=session_name,
                    external_id=session_external_id,
                    metadata={"component": "agentic-classifier-full-trajectory"},
                )
            finally:
                await _close_owned_ingest_client(logger)

        return asyncio.run(_run())

    def _attach_session(logger: Any, session_id: str) -> None:
        logger.set_session(session_id)
        if str(getattr(logger, "session_id", "")) != session_id:
            raise GalileoIntegrationUnavailable(
                "Galileo did not retain the approved full-trajectory session ID"
            )

    def _bind_session(logger: Any) -> None:
        owns_attempt = False
        with lock:
            if state.session_id is not None:
                resolved_session_id = state.session_id
                attempt = None
            else:
                resolved_session_id = None
                attempt = state.in_flight
            if resolved_session_id is not None:
                pass
            elif attempt is None:
                attempt = _SessionSetupAttempt()
                state.in_flight = attempt
                owns_attempt = True

        if resolved_session_id is not None:
            _attach_session(logger, resolved_session_id)
            return

        assert attempt is not None
        if owns_attempt:
            try:
                created = _start_session(logger)
                if isinstance(created, str) and created:
                    attempt.session_id = created
                else:
                    attempt.error = GalileoIntegrationUnavailable(
                        "Galileo session creation returned no session ID"
                    )
            except BaseException as exc:  # noqa: BLE001 - relayed to callers
                attempt.error = exc
            finally:
                with lock:
                    if attempt.session_id is not None:
                        state.session_id = attempt.session_id
                    if state.in_flight is attempt:
                        state.in_flight = None
                attempt.done.set()
        else:
            # This wait happens only in the daemon setup worker. The outer
            # callback deadline can abandon this logger without starting a
            # duplicate non-atomic session create.
            if not attempt.done.wait(_resolve_setup_budget_s()):
                raise GalileoIntegrationUnavailable(
                    "Galileo session creation remained in flight beyond the setup "
                    "budget; raw trajectory disabled for this invocation"
                )

        if attempt.error is not None:
            raise GalileoIntegrationUnavailable(
                "Galileo session creation failed; raw trajectory disabled for "
                "this invocation"
            ) from attempt.error
        if attempt.session_id is None:
            raise GalileoIntegrationUnavailable(
                "Galileo session creation returned no session ID; raw trajectory "
                "disabled for this invocation"
            )
        _attach_session(logger, attempt.session_id)

    def _factory() -> Any:
        return create_galileo_async_callback(
            approved_fixture=True,
            session_binder=_bind_session,
        )

    # The agent wrapper uses this private marker to enable direct-file guards
    # only for a statically approved raw Galileo mode. A missing approval must
    # remain fail-open and behavior-identical to an ordinary scan.
    setattr(_factory, "_aibom_strict_tool_roots", _raw_tool_root_guards_approved())
    return _factory


def run_galileo_custom_function_experiment(
    payload: DecisionSuite | Mapping[str, Any] | str | bytes,
    *,
    experiment_name: str,
    function: Callable[[Any], Any],
    project: str | None = None,
    metrics: list[Any] | None = None,
    experiment_tags: dict[str, str] | None = None,
    experiment_group: str | None = None,
    approved_fixture: bool = False,
    exact_identities: bool = False,
    on_error: Callable[[Exception], None] | None = None,
) -> Any:
    """Run Galileo 2.4's custom-function experiment path explicitly.

    This is the evaluation entry point; it never uses AIBOM's legacy
    benchmark harness. Validation, exact-identity approval, and full-content
    approval (when evidence is present) happen before the optional SDK import
    or any network activity. In the default mode, Galileo receives per-run
    random pseudonyms while the supplied function receives a deep copy of the
    exact in-process fixture. The function controls its own model/network
    egress; this helper controls only what it sends to Galileo. The return value
    is first projected through :func:`sanitize_galileo_decision_output` against
    that exact fixture, then pseudonymized before logging. Set
    ``exact_identities=True`` only for an explicitly approved exact/full-content
    run; that mode additionally requires ``AIBOM_GALILEO_ALLOW_EXACT_IDENTITIES``.
    The application is responsible for using an isolated empty agentic cache for
    every experiment variant. When ``metrics`` is omitted, the deterministic
    trace-level metrics from :func:`build_galileo_decision_metrics` are installed.

    Calling this function creates/runs resources in the configured Galileo
    project and therefore requires an explicit approved HTTPS
    ``GALILEO_CONSOLE_URL``. Hosted Galileo additionally requires the public
    cloud egress flag. Merely importing this module, validating a suite, or
    building rows performs no I/O.
    """
    requested_experiment_name = _validated_experiment_label(
        experiment_name, field_name="experiment_name"
    )
    if not callable(function):
        raise TypeError("function must be callable")
    if project is not None:
        raise HostedGalileoDestinationRequired(
            "Caller-supplied Galileo project names are not accepted for hosted "
            f"evaluation; configure {EVALUATION_PROJECT_ID_ENV_VAR}"
        )
    if exact_identities is True:
        _require_exact_identity_approval(approved_fixture=approved_fixture)
    elif exact_identities is False:
        # The default hosted path removes exact identities, arbitrary metadata,
        # and evidence before egress. Destination approval and project pinning
        # remain mandatory, but a full-content fixture approval is unnecessary.
        pass
    else:
        raise TypeError("exact_identities must be a boolean")
    project_id = _required_resource_id(EVALUATION_PROJECT_ID_ENV_VAR)
    requested_tags = _validated_experiment_tags(experiment_tags)
    requested_group = (
        _validated_experiment_label(experiment_group, field_name="experiment_group")
        if experiment_group is not None
        else None
    )

    suite = validate_decision_suite(payload)
    exact_rows = _validated_experiment_rows(suite)
    local_fixtures: dict[str, tuple[dict[str, Any], dict[str, Any]]] | None = None
    hosted_registry: _HostedPseudonymRegistry | None = None
    if exact_identities:
        # Reuse the public serializer so approved evidence receives its
        # independent full-content gate before any SDK import or network call.
        rows = build_galileo_experiment_rows(
            suite,
            approved_fixture=approved_fixture,
        )
        safe_experiment_name = requested_experiment_name
        safe_tags = requested_tags
        safe_group = requested_group
    else:
        rows, local_fixtures, hosted_registry = _hosted_experiment_bundle(exact_rows)
        run_token = hosted_registry.token(
            "experiment-name", requested_experiment_name, prefix="run"
        ).split("_", 1)[1]
        safe_experiment_name = f"aibom-decision-{run_token}"
        safe_tags = {"aibom_mode": "pseudonymous"}
        if requested_tags:
            for key, value in sorted(requested_tags.items()):
                safe_key = hosted_registry.token("tag-key", key, prefix="tag")
                safe_tags[safe_key] = hosted_registry.token(
                    "tag-value", (key, value), prefix="value"
                )
        safe_group = (
            hosted_registry.token("experiment-group", requested_group, prefix="group")
            if requested_group is not None
            else None
        )
    hosted_url = _require_hosted_galileo_destination()
    try:
        from galileo.experiments import run_experiment
    except (ImportError, ModuleNotFoundError) as exc:
        raise GalileoIntegrationUnavailable(
            "The optional Galileo experiment integration is unavailable; "
            "install the AIBOM observability extra before evaluation"
        ) from exc

    _verify_loaded_galileo_destination(hosted_url)

    selected_metrics = build_galileo_decision_metrics() if metrics is None else metrics
    ground_truth_by_case: dict[str, Any] = {}
    if exact_identities:
        for row in rows:
            row_input = _json_object(row.get("input"))
            if row_input is not None and row_input.get("case_id") is not None:
                ground_truth_by_case[str(row_input["case_id"])] = row.get(
                    "ground_truth"
                )

    @wraps(function)
    def sanitized_function(dataset_input: Any) -> str:
        parsed_input = _json_object(dataset_input)
        case_id = (
            str(parsed_input.get("case_id"))
            if parsed_input is not None and parsed_input.get("case_id") is not None
            else ""
        )
        if exact_identities:
            application_input = dataset_input
            sanitization_input = dataset_input
            dataset_ground_truth = ground_truth_by_case.get(case_id)
        else:
            assert local_fixtures is not None
            fixture = local_fixtures.get(case_id)
            if fixture is None:
                return _serialized_decision_output(_schema_invalid_output())
            exact_input, exact_ground_truth = fixture
            application_input = deepcopy(exact_input)
            sanitization_input = exact_input
            dataset_ground_truth = exact_ground_truth
        try:
            application_output = function(application_input)
        except Exception:
            # Application exceptions can contain prompts, source, tool payloads,
            # paths, or provider responses. Do not let the experiment runner log
            # the exception text; represent the failure only as schema-invalid.
            return _serialized_decision_output(_schema_invalid_output())
        exact_output = sanitize_galileo_decision_output(
            application_output,
            dataset_input=sanitization_input,
            dataset_ground_truth=dataset_ground_truth,
        )
        if exact_identities:
            return exact_output
        assert hosted_registry is not None
        return _hosted_decision_output(exact_output, hosted_registry)

    if not exact_identities:
        # Do not let a customer/repository-bearing application function name
        # become experiment metadata through SDK introspection. The original
        # callable remains reachable only through this local closure.
        sanitized_function.__name__ = "aibom_pseudonymous_application"
        sanitized_function.__qualname__ = "aibom_pseudonymous_application"
        sanitized_function.__doc__ = (
            "Run one local fixture and return a pseudonymized result."
        )
        sanitized_function.__module__ = __name__
        try:
            delattr(sanitized_function, "__wrapped__")
        except AttributeError:
            pass

    return run_experiment(
        safe_experiment_name,
        project=None,
        project_id=project_id,
        dataset=rows,
        metrics=selected_metrics,
        function=sanitized_function,
        experiment_tags=safe_tags,
        on_error=on_error,
        experiment_group=safe_group,
    )


__all__ = [
    "AIBOM_EVIDENCE_GROUNDING_METRIC_NAME",
    "AIBOM_EVIDENCE_GROUNDING_PROMPT",
    "ALLOW_PUBLIC_CLOUD_ENV_VAR",
    "DECISION_SUITE_SCHEMA_VERSION",
    "EVALUATION_LOG_STREAM_ID_ENV_VAR",
    "EVALUATION_PROJECT_ID_ENV_VAR",
    "EXACT_IDENTITIES_ENV_VAR",
    "GALILEO_DECISION_OUTPUT_SCHEMA_VERSION",
    "FULL_CONTENT_ENV_VAR",
    "FULL_TRAJECTORY_ENV_VAR",
    "ApprovedEvidenceExcerpt",
    "DecisionCandidate",
    "DecisionSuite",
    "DecisionSuiteCase",
    "ExecutionOutcome",
    "ExpectedActionLabel",
    "ExpectedRelationshipLabel",
    "ExpectedRiskLabel",
    "ExactIdentityLoggingDenied",
    "FullContentLoggingDenied",
    "GalileoDecisionOutput",
    "GalileoDecisionRelationship",
    "GalileoDecisionRisk",
    "GalileoIntegrationUnavailable",
    "HostedGalileoDestinationRequired",
    "PrivateGalileoDestinationRequired",
    "adapt_pipeline_result_for_galileo",
    "build_aibom_evidence_grounding_metric",
    "build_galileo_decision_metrics",
    "build_galileo_experiment_rows",
    "create_galileo_async_callback",
    "create_full_trajectory_callback_factory",
    "run_galileo_custom_function_experiment",
    "sanitize_galileo_decision_output",
    "validate_decision_suite",
]
