import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from src.application.use_cases import run_whatsapp_agent as module
from src.application.use_cases.run_whatsapp_agent import (
    RunWhatsAppAgentUseCase,
    _retry_transient,
)
from src.infrastructure.adapters.db import checkpointer_pool


class BuildCheckpointerPoolTest(unittest.TestCase):
    def test_pool_receives_required_kwargs_check_and_limits(self) -> None:
        captured: dict = {}

        def fake_pool(**kwargs):
            captured.update(kwargs)
            return "pool"

        settings = SimpleNamespace(
            CHECKPOINTER_POOL_MIN_SIZE=2,
            CHECKPOINTER_POOL_MAX_SIZE=7,
            CHECKPOINTER_POOL_MAX_IDLE=111.0,
            CHECKPOINTER_POOL_MAX_LIFETIME=222.0,
        )
        with patch.object(checkpointer_pool, "AsyncConnectionPool", fake_pool):
            fake_pool.check_connection = AsyncConnectionPool.check_connection
            result = checkpointer_pool.build_checkpointer_pool("postgresql://x", settings)

        self.assertEqual(result, "pool")
        self.assertEqual(captured["conninfo"], "postgresql://x")
        self.assertEqual(captured["min_size"], 2)
        self.assertEqual(captured["max_size"], 7)
        self.assertEqual(captured["max_idle"], 111.0)
        self.assertEqual(captured["max_lifetime"], 222.0)
        self.assertIs(captured["open"], False)
        self.assertIs(captured["check"], AsyncConnectionPool.check_connection)
        kwargs = captured["kwargs"]
        self.assertIs(kwargs["autocommit"], True)
        self.assertEqual(kwargs["prepare_threshold"], 0)
        self.assertIs(kwargs["row_factory"], dict_row)
        self.assertEqual(kwargs["keepalives"], 1)
        self.assertEqual(kwargs["keepalives_idle"], 30)
        self.assertEqual(kwargs["keepalives_interval"], 10)
        self.assertEqual(kwargs["keepalives_count"], 5)


class RetryTransientTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        patcher = patch.object(module, "_CHECKPOINT_RETRY_BACKOFF_SECONDS", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_retries_then_succeeds(self) -> None:
        calls = 0

        async def op():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise psycopg.OperationalError("ssl drop")
            return "ok"

        self.assertEqual(await _retry_transient(op, "t"), "ok")
        self.assertEqual(calls, 3)

    async def test_gives_up_after_max_attempts(self) -> None:
        calls = 0

        async def op():
            nonlocal calls
            calls += 1
            raise PoolTimeout("timeout")

        with self.assertRaises(PoolTimeout):
            await _retry_transient(op, "t")
        self.assertEqual(calls, module._CHECKPOINT_RETRY_ATTEMPTS)

    async def test_does_not_retry_other_exceptions(self) -> None:
        calls = 0

        async def op():
            nonlocal calls
            calls += 1
            raise ValueError("nope")

        with self.assertRaises(ValueError):
            await _retry_transient(op, "t")
        self.assertEqual(calls, 1)

    async def test_propagates_cancelled_error_without_retry(self) -> None:
        calls = 0

        async def op():
            nonlocal calls
            calls += 1
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await _retry_transient(op, "t")
        self.assertEqual(calls, 1)


class FlakyGraph:
    def __init__(self) -> None:
        self.calls = 0

    async def aget_state(self, config: dict) -> SimpleNamespace:
        self.calls += 1
        if self.calls == 1:
            raise psycopg.OperationalError("SSL SYSCALL error")
        return SimpleNamespace(values={"messages": []}, next=())


class LoadBaselineRetryTest(unittest.IsolatedAsyncioTestCase):
    async def test_load_baseline_ids_succeeds_after_one_operational_error(self) -> None:
        graph = FlakyGraph()
        use_case = RunWhatsAppAgentUseCase(graph)
        with patch.object(module, "_CHECKPOINT_RETRY_BACKOFF_SECONDS", 0):
            ids = await use_case._load_baseline_ids({"configurable": {}}, "conv")
        self.assertEqual(ids, set())
        self.assertEqual(graph.calls, 2)


if __name__ == "__main__":
    unittest.main()
