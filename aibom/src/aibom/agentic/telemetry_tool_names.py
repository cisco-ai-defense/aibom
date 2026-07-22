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

"""Canonical allowlist for content-free Galileo tool telemetry."""

from __future__ import annotations

TOOL_NAMES = frozenset(
    {
        # AIBOM detection tools (see agentic/tools.py build_tools()).
        "analyze_imports",
        "list_directory_tree",
        "lookup_model",
        "read_file_snippet",
        "resolve_env_var",
        "search_codebase",
        "search_package_info",
        "trace_data_flow",
        # Deep Agents scaffolding tools injected by create_deep_agent().
        "compact_conversation",
        "edit_file",
        "execute",
        "glob",
        "grep",
        "ls",
        "read_file",
        "task",
        "write_file",
        "write_todos",
    }
)

__all__ = ["TOOL_NAMES"]
