# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Module entry point for cisco_aibom."""

# Re-export the CLI entry point and report helper expected by tests.
from .cli import _generate_json_report, cli_entry_point

__all__ = ["cli_entry_point", "_generate_json_report"]


if __name__ == "__main__":
    cli_entry_point()
