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

import logging
from abc import ABC, abstractmethod
from typing import IO, Optional

from ..models import ScanResult

_LOGGER = logging.getLogger(__name__)

reporter_registry: list[type["BaseReporter"]] = []


class BaseReporter(ABC):
    """Interface that all AIBOM output reporters implement.

    Subclasses auto-register by setting a non-empty ``name`` class variable.
    """

    name: str = ""
    file_extension: str = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            reporter_registry.append(cls)
            _LOGGER.debug("Registered reporter: %s", cls.name)

    @abstractmethod
    def render(self, result: ScanResult, output: IO[str]) -> None:
        """Serialize *result* and write to *output*."""
        ...

    def validate(self, result: ScanResult) -> list[str]:
        """Validate *result* against this format's schema.

        Returns a list of validation error messages (empty if valid).
        The default implementation performs no validation.
        """
        return []


def get_reporter(name: str) -> Optional[BaseReporter]:
    """Look up a reporter by name and return an instance, or *None*."""
    for cls in reporter_registry:
        if cls.name == name:
            return cls()
    return None
