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

"""Content-free LangChain callback for detailed production telemetry.

The callback receives prompts, model responses, tool arguments, and tool
results because those values are part of LangChain's callback API.  It never
stores, serializes, hashes, or forwards any of them.  Only timing, ordering,
allowlisted tool names, random callback run IDs, status, and token counters are
retained.

This module deliberately does not import LangChain.  The handler implements the
small callback protocol by duck typing, which keeps the agentic and Galileo
dependencies optional for normal AIBOM installations.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping

_TOOL_NAMES = frozenset(
    {
        "analyze_imports",
        "list_directory_tree",
        "lookup_model",
        "read_file_snippet",
        "resolve_env_var",
        "search_codebase",
        "search_package_info",
        "trace_data_flow",
    }
)


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _nested_int(value: Any, *path: str) -> int:
    current = value
    for key in path:
        if not isinstance(current, Mapping):
            return 0
        current = current.get(key)
    return _non_negative_int(current)


def _usage_from_mapping(value: Any) -> tuple[int, int, int, int]:
    """Return prompt, completion, total, and cache-read tokens from safe counters."""

    if not isinstance(value, Mapping):
        return 0, 0, 0, 0

    prompt = _non_negative_int(value.get("input_tokens"))
    completion = _non_negative_int(value.get("output_tokens"))
    total = _non_negative_int(value.get("total_tokens"))
    cached = max(
        _nested_int(value, "input_token_details", "cache_read"),
        _non_negative_int(value.get("cache_read_input_tokens")),
    )
    if prompt or completion or total or cached:
        return prompt, completion, total or prompt + completion, cached

    prompt = _non_negative_int(value.get("prompt_tokens"))
    completion = _non_negative_int(value.get("completion_tokens"))
    total = _non_negative_int(value.get("total_tokens"))
    cached = _nested_int(value, "prompt_tokens_details", "cached_tokens")
    if prompt or completion or total or cached:
        return prompt, completion, total or prompt + completion, cached

    prompt = _non_negative_int(value.get("inputTokenCount"))
    completion = _non_negative_int(value.get("outputTokenCount"))
    if prompt or completion:
        return prompt, completion, prompt + completion, 0
    return 0, 0, 0, 0


def _usage_from_message(message: Any) -> tuple[int, int, int, int]:
    usage = _usage_from_mapping(getattr(message, "usage_metadata", None))
    if any(usage):
        return usage
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, Mapping):
        return 0, 0, 0, 0
    for key in ("token_usage", "usage", "amazon-bedrock-invocationMetrics"):
        usage = _usage_from_mapping(metadata.get(key))
        if any(usage):
            return usage
    return 0, 0, 0, 0


def _usage_from_llm_result(response: Any) -> tuple[int, int, int, int]:
    prompt = completion = total = cached = 0
    for group in getattr(response, "generations", ()) or ():
        for generation in group or ():
            message = getattr(generation, "message", generation)
            current = _usage_from_message(message)
            prompt += current[0]
            completion += current[1]
            total += current[2]
            cached += current[3]
    if prompt or completion or total or cached:
        return prompt, completion, total or prompt + completion, cached

    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, Mapping):
        for key in ("token_usage", "usage", "amazon-bedrock-invocationMetrics"):
            current = _usage_from_mapping(llm_output.get(key))
            if any(current):
                return current
        return _usage_from_mapping(llm_output)
    return 0, 0, 0, 0


@dataclass(frozen=True, slots=True)
class SanitizedCall:
    """One content-free model or tool invocation captured from callbacks."""

    kind: Literal["llm", "tool"]
    call_id: str
    sequence: int
    created_at: datetime
    duration_s: float
    status: Literal["success", "failed", "timeout"]
    tool_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


@dataclass(slots=True)
class _ActiveCall:
    kind: Literal["llm", "tool"]
    call_id: str
    sequence: int
    created_at: datetime
    monotonic_started: float
    tool_name: str = ""


class SanitizedAgentCallback:
    """Record detailed call shape without retaining callback content.

    ``run_inline`` preserves callback ordering.  Callback failures never escape
    into the scan because ``raise_error`` is false and every method is defensive.
    """

    raise_error = False
    run_inline = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, _ActiveCall] = {}
        self._completed: list[SanitizedCall] = []
        self._next_sequence = 1
        self._sealed = False

    @property
    def ignore_llm(self) -> bool:
        return False

    @property
    def ignore_chat_model(self) -> bool:
        return False

    @property
    def ignore_agent(self) -> bool:
        # LangChain uses this condition for tool callbacks as well.
        return False

    @property
    def ignore_chain(self) -> bool:
        return True

    @property
    def ignore_retriever(self) -> bool:
        return True

    @property
    def ignore_retry(self) -> bool:
        return True

    @property
    def ignore_custom_event(self) -> bool:
        return True

    def _start(
        self,
        kind: Literal["llm", "tool"],
        run_id: Any,
        *,
        tool_name: str = "",
        call_id: Any = None,
    ) -> None:
        identifier = str(run_id)
        if not identifier:
            return
        now = datetime.now(timezone.utc)
        monotonic_started = time.monotonic()
        with self._lock:
            if self._sealed or identifier in self._active:
                return
            sequence = self._next_sequence
            self._next_sequence += 1
            self._active[identifier] = _ActiveCall(
                kind=kind,
                call_id=str(call_id) if call_id is not None else identifier,
                sequence=sequence,
                created_at=now,
                monotonic_started=monotonic_started,
                tool_name=tool_name,
            )

    def _finish(
        self,
        run_id: Any,
        *,
        status: Literal["success", "failed", "timeout"],
        usage: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        identifier = str(run_id)
        now = time.monotonic()
        with self._lock:
            if self._sealed:
                return
            active = self._active.pop(identifier, None)
            if active is None:
                return
            duration = max(0.0, now - active.monotonic_started)
            if not math.isfinite(duration):
                duration = 0.0
            self._completed.append(
                SanitizedCall(
                    kind=active.kind,
                    call_id=active.call_id,
                    sequence=active.sequence,
                    created_at=active.created_at,
                    duration_s=duration,
                    status=status,
                    tool_name=active.tool_name,
                    prompt_tokens=usage[0],
                    completion_tokens=usage[1],
                    total_tokens=usage[2],
                    cached_tokens=usage[3],
                )
            )

    def on_chat_model_start(
        self,
        serialized: Mapping[str, Any],
        messages: Any,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        del serialized, messages, kwargs
        self._start("llm", run_id)

    def on_llm_start(
        self,
        serialized: Mapping[str, Any],
        prompts: Any,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        del serialized, prompts, kwargs
        self._start("llm", run_id)

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        del kwargs
        self._finish(run_id, status="success", usage=_usage_from_llm_result(response))

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        del kwargs
        status: Literal["failed", "timeout"] = (
            "timeout" if isinstance(error, TimeoutError) else "failed"
        )
        self._finish(run_id, status=status)

    def on_tool_start(
        self,
        serialized: Mapping[str, Any] | None,
        input_str: str,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        del input_str
        raw_name = serialized.get("name") if isinstance(serialized, Mapping) else None
        name = str(raw_name) if str(raw_name) in _TOOL_NAMES else "other"
        self._start(
            "tool",
            run_id,
            tool_name=name,
            call_id=kwargs.get("tool_call_id"),
        )

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        del output, kwargs
        self._finish(run_id, status="success")

    def on_tool_error(
        self, error: BaseException, *, run_id: Any, **kwargs: Any
    ) -> None:
        del kwargs
        status: Literal["failed", "timeout"] = (
            "timeout" if isinstance(error, TimeoutError) else "failed"
        )
        self._finish(run_id, status=status)

    # Agent callbacks share ``ignore_agent`` with tool callbacks. They must be
    # accepted as no-ops so detailed tool capture does not produce warnings.
    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        del action, kwargs

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        del finish, kwargs

    def seal(
        self, *, unfinished_status: Literal["failed", "timeout"] = "failed"
    ) -> list[SanitizedCall]:
        """Return an immutable snapshot and reject callbacks arriving later."""

        now = time.monotonic()
        with self._lock:
            if not self._sealed:
                for active in self._active.values():
                    duration = max(0.0, now - active.monotonic_started)
                    if not math.isfinite(duration):
                        duration = 0.0
                    self._completed.append(
                        SanitizedCall(
                            kind=active.kind,
                            call_id=active.call_id,
                            sequence=active.sequence,
                            created_at=active.created_at,
                            duration_s=duration,
                            status=unfinished_status,
                            tool_name=active.tool_name,
                        )
                    )
                self._active.clear()
                self._sealed = True
            return sorted(self._completed, key=lambda call: call.sequence)
