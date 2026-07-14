# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from aibom.cli import _build_submission_payload, _build_submission_payloads


def test_submission_payload_wraps_report():
    report = {
        "aibom_analysis": {
            "metadata": {
                "run_id": "run-123",
                "scan_batch_id": "batch-abc",
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
    assert payload["scan_batch_id"] == "batch-abc"
    assert payload["analyzer_version"] == "1.2.3"
    assert payload["submitted_at"] == "2025-01-01T12:00:00Z"
    assert payload["source_kind"] == "SOURCE_KIND_LOCAL_PATH"
    assert payload["sources"] == [{"name": "repo", "path": "/app"}]
    assert payload["report"] == report


def _multi_source_report() -> dict:
    return {
        "aibom_analysis": {
            "metadata": {
                "run_id": "run-123",
                "scan_batch_id": "batch-xyz",
                "analyzer_version": "1.2.3",
                "completed_at": "2025-01-01T12:00:00Z",
            },
            "sources": {
                "repo-a": {
                    "source_name": "repo-a",
                    "source_path": "/a",
                    "summary": {"source_kind": "local-path"},
                    "components": [],
                    "relationships": [],
                    "metadata": {},
                },
                "repo-b": {
                    "source_name": "repo-b",
                    "source_path": "/b",
                    "summary": {"source_kind": "local-path"},
                    "components": [],
                    "relationships": [],
                    "metadata": {},
                },
            },
            "summary": {},
        }
    }


def test_fan_out_shares_batch_id_but_distinct_run_ids():
    payloads = _build_submission_payloads(_multi_source_report())

    assert len(payloads) == 2
    # Every upload in the invocation carries the same co-scan batch id...
    batch_ids = {p["scan_batch_id"] for p in payloads}
    assert batch_ids == {"batch-xyz"}
    # ...but each keeps its own distinct run_id.
    run_ids = {p["run_id"] for p in payloads}
    assert len(run_ids) == 2


def test_fan_out_mints_batch_id_for_legacy_report():
    report = _multi_source_report()
    del report["aibom_analysis"]["metadata"]["scan_batch_id"]

    payloads = _build_submission_payloads(report)

    batch_ids = {p["scan_batch_id"] for p in payloads}
    assert len(batch_ids) == 1
    assert batch_ids != {None}


def _legacy_no_sources_report() -> dict:
    return {
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


def test_legacy_no_sources_report_mints_batch_id():
    # A pre-field report with no per-source breakdown still gets a minted batch
    # id on its single upload, consistent with the fan-out path.
    payloads = _build_submission_payloads(_legacy_no_sources_report())

    assert len(payloads) == 1
    assert payloads[0]["scan_batch_id"]


def test_legacy_no_sources_report_preserves_existing_batch_id():
    report = _legacy_no_sources_report()
    report["aibom_analysis"]["metadata"]["scan_batch_id"] = "batch-abc"

    payloads = _build_submission_payloads(report)

    assert len(payloads) == 1
    assert payloads[0]["scan_batch_id"] == "batch-abc"
