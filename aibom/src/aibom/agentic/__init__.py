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

"""Agentic scanning module for Cisco AIBOM.

All heavy imports (``deepagents``, ``langchain``) are deferred to function
call time so that ``import aibom`` never pulls in optional dependencies.
"""

from __future__ import annotations

from .middleware import AIBOMScannerMiddleware

__all__ = [
    "AIBOMScannerMiddleware",
    "create_aibom_agent",
    "run_agentic_enrichment",
    "AgenticEnrichmentError",
]


def create_aibom_agent(model_string: str, **kwargs):  # type: ignore[no-untyped-def]
    """Lazy re-export — actual import happens on first call."""
    from .agent import create_aibom_agent as _impl

    return _impl(model_string, **kwargs)


def run_agentic_enrichment(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Lazy re-export — actual import happens on first call."""
    from .agent import run_agentic_enrichment as _impl

    return _impl(*args, **kwargs)


def AgenticEnrichmentError(*args, **kwargs):  # type: ignore[no-untyped-def]  # noqa: N802
    """Lazy re-export of the exception class."""
    from .agent import AgenticEnrichmentError as _cls

    return _cls(*args, **kwargs)
