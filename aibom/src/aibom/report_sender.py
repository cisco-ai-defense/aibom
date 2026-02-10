# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""HTTP report sender with resilient retries and backoff."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, Optional, Set

import httpx

RETRYABLE_STATUS_CODES: Set[int] = {429, 500, 502, 503, 504}
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_BACKOFF_SECONDS = 30.0


def _retry_delay_seconds(
    attempt: int,
    retry_after_header: Optional[str],
    *,
    base_backoff: float,
    max_backoff: float,
) -> float:
    """Compute exponential backoff with jitter, honoring Retry-After when numeric."""
    if retry_after_header and retry_after_header.isdigit():
        return max(1.0, min(float(retry_after_header), max_backoff))

    base = base_backoff * (2 ** (attempt - 1))
    jitter = base * random.uniform(0, 0.25)
    return min(max_backoff, base + jitter)


def post_report_with_retries(
    url: str,
    payload: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    verify_tls: bool = True,
    timeout_seconds: float = 30.0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_backoff: float = DEFAULT_BASE_BACKOFF_SECONDS,
    max_backoff: float = DEFAULT_MAX_BACKOFF_SECONDS,
    logger: Optional[logging.Logger] = None,
) -> None:
    """POST the report with retries on transient failures.

    Raises:
        RuntimeError: if the request does not succeed after retries.
    """
    log = logger or logging.getLogger(__name__)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-cisco-ai-defense-tenant-api-key"] = api_key

    timeout = httpx.Timeout(timeout_seconds)
    last_err: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
                verify=verify_tls,
            )
            if resp.status_code in RETRYABLE_STATUS_CODES:
                retry_after = resp.headers.get("retry-after")
                raise httpx.HTTPStatusError(
                    f"retryable status {resp.status_code}",
                    request=resp.request,
                    response=resp,
                ) from None

            resp.raise_for_status()
            log.info("Report posted to %s (status %s)", url, resp.status_code)
            return

        except (httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            last_err = exc
            retry_after = None
            if isinstance(exc, httpx.HTTPStatusError):
                retry_after = exc.response.headers.get("retry-after")
            if attempt == max_attempts:
                break

            sleep_seconds = _retry_delay_seconds(
                attempt,
                retry_after,
                base_backoff=base_backoff,
                max_backoff=max_backoff,
            )
            log.warning(
                "Report POST attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                max_attempts,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to POST report to {url} after {max_attempts} attempts: {last_err}")
