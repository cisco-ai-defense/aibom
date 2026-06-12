# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from aibom.models import AIComponent, AIComponentType


@pytest.fixture(autouse=True)
def _isolate_agentic_cache():
    """Prevent on-disk agentic cache from leaking between tests."""
    with patch("aibom.agentic.agent._default_agentic_cache_dir", return_value=None):
        yield


class TestAgenticTimeout:
    @patch("aibom.agentic.agent._RETRY_COOLDOWN_S", 0)
    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_timeout_marks_components_not_agentic(
        self, mock_create: MagicMock, _mb: MagicMock, _mc: MagicMock
    ) -> None:
        from aibom.agentic.agent import run_agentic_enrichment

        mock_agent = MagicMock()

        def slow_invoke(*_a: object, **_k: object) -> dict[str, object]:
            time.sleep(10)
            return {"messages": []}

        mock_agent.invoke.side_effect = slow_invoke
        mock_create.return_value = mock_agent

        comp = AIComponent(
            name="x",
            component_type=AIComponentType.AGENT,
            file_path="b.py",
            line_number=1,
        )
        comps, _, _, _ = run_agentic_enrichment(
            model_string="m",
            deterministic_components=[comp],
            deterministic_relationships=[],
            scan_paths=["/tmp"],
            timeout_s=1,
        )
        assert len(comps) == 1
        assert comps[0].needs_agentic is False
        assert comps[0].agentic_hint == "batch_timeout"

    @patch("aibom.agentic.agent._RETRY_COOLDOWN_S", 0)
    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_never_returning_invoke_does_not_hang(
        self, mock_create: MagicMock, _mb: MagicMock, _mc: MagicMock
    ) -> None:
        """A truly hung LLM call (invoke that never returns) must NOT wedge the
        scan: the batch must time out and fail open within a bounded wall-clock,
        even though the worker thread cannot be cancelled.

        Regression for the executor-shutdown hang: the previous implementation
        wrapped the blocking call in ``asyncio.run(asyncio.wait_for(
        asyncio.to_thread(...)))``; ``wait_for`` raised TimeoutError but
        ``asyncio.run`` then blocked forever joining the orphaned executor
        thread on loop shutdown.
        """
        never = threading.Event()  # never set -> invoke blocks forever

        def hung_invoke(*_a: object, **_k: object) -> dict[str, object]:
            never.wait()  # blocks until process exit
            return {"messages": []}

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = hung_invoke
        mock_create.return_value = mock_agent

        comp = AIComponent(
            name="x",
            component_type=AIComponentType.AGENT,
            file_path="b.py",
            line_number=1,
        )

        from aibom.agentic.agent import run_agentic_enrichment

        start = time.monotonic()
        comps, _, _, _ = run_agentic_enrichment(
            model_string="m",
            deterministic_components=[comp],
            deterministic_relationships=[],
            scan_paths=["/tmp"],
            timeout_s=1,
        )
        elapsed = time.monotonic() - start

        # Must return shortly after the 1s timeout, not hang on the stuck thread.
        assert (
            elapsed < 20
        ), f"batch hung for {elapsed:.1f}s on a never-returning invoke"
        assert len(comps) == 1
        assert comps[0].needs_agentic is False
        assert comps[0].agentic_hint == "batch_timeout"


class TestAgenticAsyncTimeout:
    @patch("aibom.agentic.agent._RETRY_COOLDOWN_S", 0)
    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_parallel_path_stalled_ainvoke_does_not_hang(
        self, mock_create: MagicMock, _mb: MagicMock, _mc: MagicMock
    ) -> None:
        """The async/parallel batch path (max_concurrent>1, >1 batch) must bound
        a stalled provider call and let the scan complete + the process exit.

        Models a real native-async provider call that never returns (e.g. a
        wedged streaming response): ``ainvoke`` awaits a never-set event.
        ``asyncio.timeout`` cancels it at the coroutine level; the batch fails
        open as ``batch_timeout``. The old ``asyncio.run`` orchestration could
        wedge on ``loop.shutdown_default_executor()``; ``_run_async_bounded``
        runs the loop in a daemon thread and abandons its executor, so the scan
        returns within the timeout and the process exits cleanly.

        Note: a provider that offloads a *blocking* sync call to the loop's
        default executor (``run_in_executor``) cannot be hard-cancelled — that
        is a Python limitation (you cannot kill a thread stuck in a syscall),
        independent of this fix. The fix still lets the scan complete; only the
        orphaned worker lingers until process exit.
        """
        import asyncio

        sync_never = threading.Event()  # for the sequential retry (sync) pass

        async def stalled_ainvoke(*_a: object, **_k: object) -> dict[str, object]:
            # Native-async stall: awaitable + cancellable, like a wedged
            # httpx.AsyncClient streaming read.
            await asyncio.Event().wait()
            return {"messages": []}

        def hung_invoke(*_a: object, **_k: object) -> dict[str, object]:
            # The sequential retry pass uses the sync path; keep it hung too so
            # both passes fail open via timeout rather than on a junk mock.
            sync_never.wait()
            return {"messages": []}

        mock_agent = MagicMock()
        mock_agent.ainvoke = stalled_ainvoke
        mock_agent.invoke.side_effect = hung_invoke
        mock_create.return_value = mock_agent

        comps = [
            AIComponent(
                name=f"c{i}",
                component_type=AIComponentType.AGENT,
                file_path=f"f{i}.py",
                line_number=i,
            )
            for i in range(2)
        ]

        from aibom.agentic.agent import run_agentic_enrichment

        start = time.monotonic()
        out, _, _, _ = run_agentic_enrichment(
            model_string="m",
            deterministic_components=comps,
            deterministic_relationships=[],
            scan_paths=["/tmp"],
            batch_size=1,  # 2 batches -> parallel/async path
            max_concurrent=2,
            timeout_s=1,
        )
        elapsed = time.monotonic() - start

        assert (
            elapsed < 25
        ), f"parallel path hung for {elapsed:.1f}s on a never-returning ainvoke"
        assert len(out) == 2
        assert all(c.needs_agentic is False for c in out)
        assert all(c.agentic_hint == "batch_timeout" for c in out)


class TestAgenticCircuitBreaker:
    @patch("aibom.agentic.agent._RETRY_COOLDOWN_S", 0)
    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_skips_after_three_consecutive_failures(
        self, mock_create: MagicMock, _mb: MagicMock, _mc: MagicMock
    ) -> None:
        from aibom.agentic.agent import run_agentic_enrichment

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("LLM down")
        mock_create.return_value = mock_agent

        comps_in = [
            AIComponent(
                name=f"c{i}",
                component_type=AIComponentType.AGENT,
                file_path=f"f{i}.py",
                line_number=i,
            )
            for i in range(4)
        ]
        comps, _, _, _ = run_agentic_enrichment(
            model_string="m",
            deterministic_components=comps_in,
            deterministic_relationships=[],
            scan_paths=["/tmp"],
            batch_size=1,
            max_concurrent=1,
            timeout_s=120,
            max_consecutive_failures=3,
        )
        # 3 main-pass invocations (breaker trips, 4th skipped) +
        # 3 retry-pass invocations (retry breaker also trips on 4th)
        assert mock_agent.invoke.call_count == 6
        assert all(c.needs_agentic is False for c in comps)

    @patch("aibom.agentic.agent._RETRY_COOLDOWN_S", 0)
    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_non_consecutive_failures_do_not_trip(
        self, mock_create: MagicMock, _mb: MagicMock, _mc: MagicMock
    ) -> None:
        from aibom.agentic.agent import run_agentic_enrichment

        ok = json.dumps(
            {
                "enriched_components": [],
                "new_components": [],
                "new_relationships": [],
                "risk_findings": [],
            }
        )
        mock_msg = MagicMock()
        mock_msg.content = ok

        mock_agent = MagicMock()
        # 5 main-pass calls (fail/ok/fail/ok/fail) + 3 retry calls (all succeed)
        mock_agent.invoke.side_effect = [
            RuntimeError("a"),
            {"messages": [mock_msg]},
            RuntimeError("b"),
            {"messages": [mock_msg]},
            RuntimeError("c"),
            {"messages": [mock_msg]},
            {"messages": [mock_msg]},
            {"messages": [mock_msg]},
        ]
        mock_create.return_value = mock_agent

        comps_in = [
            AIComponent(
                name=f"c{i}",
                component_type=AIComponentType.AGENT,
                file_path=f"f{i}.py",
                line_number=i,
            )
            for i in range(5)
        ]
        out, _, _, _ = run_agentic_enrichment(
            model_string="m",
            deterministic_components=comps_in,
            deterministic_relationships=[],
            scan_paths=["/tmp"],
            batch_size=1,
            max_concurrent=1,
            max_consecutive_failures=3,
        )
        assert mock_agent.invoke.call_count == 8
        assert not any(c.agentic_hint == "circuit_breaker_tripped" for c in out)
