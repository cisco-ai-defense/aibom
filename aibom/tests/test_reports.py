# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from aibom.cli import _build_submission_payload


def test_submission_payload_wraps_report():
    report = {
        "aibom_analysis": {
            "metadata": {
                "run_id": "run-123",
                "analyzer_version": "1.2.3",
                "completed_at": "2025-01-01T12:00:00Z",
            },
            "sources": {},
            "summary": {},
        }
    }
    source_outcomes = {
        "repo": {
            "source_kind": "local-path",
            "source_path": "/app",
            "source_name": "repo",
        }
    }

    payload = _build_submission_payload(report, source_outcomes)

    assert payload["run_id"] == "run-123"
    assert payload["analyzer_version"] == "1.2.3"
    assert payload["submitted_at"] == "2025-01-01T12:00:00Z"
    assert payload["source_kind"] == "SOURCE_KIND_LOCAL_PATH"
    assert payload["sources"] == [{"name": "repo", "path": "/app"}]
    assert payload["report"] == report
