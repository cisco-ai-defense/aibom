# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Module entry point for cisco_aibom."""

from .cli import cli_entry_point

__all__ = ["cli_entry_point"]


if __name__ == "__main__":
    cli_entry_point()
