# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the report_sender module (retries, backoff, error handling)."""

from __future__ import annotations

from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from aibom.report_sender import post_report_with_retries, _retry_delay_seconds


class TestRetryDelay:
    def test_basic_delay(self):
        delay = _retry_delay_seconds(1, None, base_backoff=1.0, max_backoff=30.0)
        assert 1.0 <= delay <= 1.25

    def test_retry_after_header(self):
        delay = _retry_delay_seconds(1, "5", base_backoff=1.0, max_backoff=30.0)
        assert delay == 5.0

    def test_max_backoff_cap(self):
        delay = _retry_delay_seconds(10, None, base_backoff=1.0, max_backoff=30.0)
        assert delay <= 30.0


class TestPostReportWithRetries:
    @patch("aibom.report_sender.httpx")
    def test_post_success(self, mock_httpx):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response
        mock_httpx.Timeout = MagicMock()
        post_report_with_retries("https://example.com/api", {"key": "value"})
        mock_httpx.post.assert_called_once()

    @patch("aibom.report_sender.httpx")
    def test_post_with_api_key(self, mock_httpx):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response
        mock_httpx.Timeout = MagicMock()
        post_report_with_retries(
            "https://example.com/api",
            {"key": "value"},
            api_key="test-key-123",
        )
        call_args = mock_httpx.post.call_args
        headers = call_args.kwargs.get("headers", {})
        assert "x-cisco-ai-defense-tenant-api-key" in headers
        assert headers["x-cisco-ai-defense-tenant-api-key"] == "test-key-123"

    @patch("aibom.report_sender.time.sleep")
    @patch("aibom.report_sender.httpx")
    def test_post_retry_on_500(self, mock_httpx, mock_sleep):
        fail_resp = MagicMock()
        fail_resp.status_code = 500
        fail_resp.headers = {"retry-after": "1"}
        fail_resp.request = MagicMock()

        class FakeHTTPStatusError(Exception):
            def __init__(self, msg):
                super().__init__(msg)
                self.response = fail_resp

        mock_httpx.HTTPStatusError = FakeHTTPStatusError

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.raise_for_status = MagicMock()

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise FakeHTTPStatusError("fail")
            return ok_resp

        mock_httpx.post.side_effect = side_effect
        mock_httpx.Timeout = MagicMock()
        mock_httpx.TransportError = type("TransportError", (Exception,), {})
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})

        post_report_with_retries(
            "https://example.com/api", {"key": "value"},
            max_attempts=3, base_backoff=0.01,
        )
        assert call_count[0] == 2

    @patch("aibom.report_sender.time.sleep")
    @patch("aibom.report_sender.httpx")
    def test_post_all_retries_exhausted(self, mock_httpx, mock_sleep):
        mock_httpx.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        mock_httpx.TransportError = type("TransportError", (Exception,), {})
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})
        mock_httpx.Timeout = MagicMock()
        mock_httpx.post.side_effect = mock_httpx.TransportError("network error")

        with pytest.raises(RuntimeError, match="Failed to POST"):
            post_report_with_retries(
                "https://example.com/api", {"key": "value"},
                max_attempts=2, base_backoff=0.01,
            )

    @patch("aibom.report_sender.httpx")
    def test_post_verify_tls_disabled(self, mock_httpx):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response
        mock_httpx.Timeout = MagicMock()
        post_report_with_retries(
            "https://example.com/api", {"key": "value"},
            verify_tls=False,
        )
        call_args = mock_httpx.post.call_args
        assert call_args.kwargs.get("verify") is False
