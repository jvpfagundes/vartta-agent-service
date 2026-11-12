import asyncio
import unittest

from src.application.services.conversation_debouncer import ConversationDebouncer, CycleGuard

_DEBOUNCE = 0.05


class Recorder:
    """Registra as execucoes de run_cycle e permite sincronizar com eventos."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.finished: list[str] = []
        self.cancelled: list[str] = []
        self.guards: list[CycleGuard] = []


class ConversationDebouncerTest(unittest.IsolatedAsyncioTestCase):
    async def test_single_fragment_runs_one_cycle_with_original_text(self) -> None:
        debouncer = ConversationDebouncer(_DEBOUNCE)
        recorder = Recorder()

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            recorder.finished.append(text)

        debouncer.submit("chat", "oi", run_cycle)
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(recorder.finished, ["oi"])

    async def test_fragments_within_window_run_single_cycle_joined_in_order(self) -> None:
        debouncer = ConversationDebouncer(_DEBOUNCE)
        recorder = Recorder()

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            recorder.finished.append(text)

        debouncer.submit("chat", "a", run_cycle)
        debouncer.submit("chat", "b", run_cycle)
        debouncer.submit("chat", "c", run_cycle)
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(recorder.finished, ["a\nb\nc"])

    async def test_fragment_during_execution_cancels_cycle_and_aggregates(self) -> None:
        debouncer = ConversationDebouncer(_DEBOUNCE)
        recorder = Recorder()
        first_running = asyncio.Event()

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            recorder.started.append(text)
            if len(recorder.started) == 1:
                first_running.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    recorder.cancelled.append(text)
                    raise
            recorder.finished.append(text)

        debouncer.submit("chat", "primeira", run_cycle)
        await first_running.wait()
        debouncer.submit("chat", "segunda", run_cycle)
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(recorder.started, ["primeira", "primeira\nsegunda"])
        self.assertEqual(recorder.cancelled, ["primeira"])
        self.assertEqual(recorder.finished, ["primeira\nsegunda"])

    async def test_new_cycle_waits_for_cancelled_cycle_cleanup(self) -> None:
        debouncer = ConversationDebouncer(0)
        events: list[str] = []
        first_running = asyncio.Event()

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            if not first_running.is_set():
                first_running.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    # simula o rollback do checkpoint, que leva algum tempo
                    events.append("rollback-start")
                    await asyncio.sleep(_DEBOUNCE)
                    events.append("rollback-end")
                    raise
            events.append("second-start")

        debouncer.submit("chat", "a", run_cycle)
        await first_running.wait()
        debouncer.submit("chat", "b", run_cycle)
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(events, ["rollback-start", "rollback-end", "second-start"])

    async def test_committed_cycle_is_not_cancelled_and_new_fragment_runs_next_cycle(self) -> None:
        debouncer = ConversationDebouncer(_DEBOUNCE)
        recorder = Recorder()
        committed = asyncio.Event()
        release = asyncio.Event()

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            recorder.started.append(text)
            if len(recorder.started) == 1:
                guard.commit()
                committed.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    recorder.cancelled.append(text)
                    raise
            recorder.finished.append(text)

        debouncer.submit("chat", "agendar", run_cycle)
        await committed.wait()
        debouncer.submit("chat", "obrigado", run_cycle)
        # cede o controle algumas vezes: o ciclo committed nao pode ter sido cancelado
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertEqual(recorder.cancelled, [])
        self.assertEqual(recorder.started, ["agendar"])
        release.set()
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(recorder.cancelled, [])
        self.assertEqual(recorder.finished, ["agendar", "obrigado"])

    async def test_exception_consumes_fragments_and_next_submit_works(self) -> None:
        debouncer = ConversationDebouncer(_DEBOUNCE)
        recorder = Recorder()

        async def failing(text: str, guard: CycleGuard) -> None:
            recorder.started.append(text)
            raise RuntimeError("falha")

        async def ok(text: str, guard: CycleGuard) -> None:
            recorder.finished.append(text)

        with self.assertLogs("src.application.services.conversation_debouncer", level="ERROR"):
            debouncer.submit("chat", "quebra", failing)
            await asyncio.wait_for(debouncer.join(), timeout=5)

        debouncer.submit("chat", "depois", ok)
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(recorder.started, ["quebra"])
        # o fragmento que falhou nao e reprocessado junto com o proximo
        self.assertEqual(recorder.finished, ["depois"])

    async def test_different_keys_are_independent(self) -> None:
        debouncer = ConversationDebouncer(_DEBOUNCE)
        recorder = Recorder()
        a_running = asyncio.Event()
        release = asyncio.Event()

        async def run_a(text: str, guard: CycleGuard) -> None:
            a_running.set()
            await release.wait()
            recorder.finished.append(f"A:{text}")

        async def run_b(text: str, guard: CycleGuard) -> None:
            recorder.finished.append(f"B:{text}")

        debouncer.submit("chat-a", "x", run_a)
        await a_running.wait()
        debouncer.submit("chat-b", "y", run_b)
        # B termina enquanto A ainda executa, e nao cancela A
        while "B:y" not in recorder.finished:
            await asyncio.sleep(0.01)
        self.assertEqual(recorder.finished, ["B:y"])
        release.set()
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(recorder.finished, ["B:y", "A:x"])

    async def test_internal_state_is_cleaned_after_everything_finishes(self) -> None:
        debouncer = ConversationDebouncer(_DEBOUNCE)

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            return None

        debouncer.submit("chat-1", "a", run_cycle)
        debouncer.submit("chat-1", "b", run_cycle)
        debouncer.submit("chat-2", "c", run_cycle)
        await asyncio.wait_for(debouncer.join(), timeout=5)
        await asyncio.sleep(0)

        self.assertEqual(debouncer._states, {})

    async def test_zero_debounce_runs_without_waiting(self) -> None:
        debouncer = ConversationDebouncer(0)
        recorder = Recorder()

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            recorder.finished.append(text)

        debouncer.submit("chat", "oi", run_cycle)
        await asyncio.wait_for(debouncer.join(), timeout=1)

        self.assertEqual(recorder.finished, ["oi"])

    async def test_shutdown_cancels_pending_cycles_without_leaving_live_tasks(self) -> None:
        debouncer = ConversationDebouncer(60)
        recorder = Recorder()

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            recorder.finished.append(text)

        debouncer.submit("chat-1", "a", run_cycle)
        debouncer.submit("chat-2", "b", run_cycle)
        tasks = [
            cycle.task for state in debouncer._states.values() for cycle in state.cycles
        ]
        self.assertTrue(tasks)

        await debouncer.shutdown()

        self.assertTrue(all(task is not None and task.done() for task in tasks))
        self.assertEqual(recorder.finished, [])
        self.assertEqual(debouncer._states, {})

    async def test_shutdown_cancels_running_cycle(self) -> None:
        debouncer = ConversationDebouncer(0)
        running = asyncio.Event()
        recorder = Recorder()

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            running.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                recorder.cancelled.append(text)
                raise

        debouncer.submit("chat", "a", run_cycle)
        await running.wait()
        await debouncer.shutdown()

        self.assertEqual(recorder.cancelled, ["a"])



class ReserveSlotTest(unittest.IsolatedAsyncioTestCase):
    async def test_reserved_slot_keeps_arrival_order_when_filled_late(self) -> None:
        debouncer = ConversationDebouncer(_DEBOUNCE)
        finished: list[str] = []

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            finished.append(text)

        slot = debouncer.reserve("chat", run_cycle)  # audio chegou primeiro
        debouncer.submit("chat", "e pra amanha", run_cycle)
        await asyncio.sleep(_DEBOUNCE * 3)
        # enquanto o audio esta pendente nenhum ciclo executa
        self.assertEqual(finished, [])
        slot.fill("quero cortar o cabelo")
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(finished, ["quero cortar o cabelo\ne pra amanha"])

    async def test_reserve_cancels_running_cycle_on_arrival(self) -> None:
        debouncer = ConversationDebouncer(0)
        running = asyncio.Event()
        cancelled: list[str] = []
        finished: list[str] = []

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            if not finished and not cancelled:
                running.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    cancelled.append(text)
                    raise
            finished.append(text)

        debouncer.submit("chat", "primeira", run_cycle)
        await running.wait()
        slot = debouncer.reserve("chat", run_cycle)  # audio chegou, ainda sem transcricao
        await asyncio.sleep(0.02)
        self.assertEqual(cancelled, ["primeira"])
        slot.fill("segunda")
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(finished, ["primeira\nsegunda"])

    async def test_release_removes_slot_and_remaining_fragments_still_run(self) -> None:
        debouncer = ConversationDebouncer(_DEBOUNCE)
        finished: list[str] = []

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            finished.append(text)

        slot = debouncer.reserve("chat", run_cycle)
        debouncer.submit("chat", "texto", run_cycle)
        slot.release()
        slot.release()  # idempotente
        slot.fill("ignorado")  # depois de liberado, nao faz nada
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(finished, ["texto"])

    async def test_release_of_only_fragment_cleans_state_without_running_cycle(self) -> None:
        debouncer = ConversationDebouncer(_DEBOUNCE)
        finished: list[str] = []

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            finished.append(text)

        slot = debouncer.reserve("chat", run_cycle)
        slot.release()
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(finished, [])
        self.assertEqual(debouncer._states, {})


class BufferLimitsTest(unittest.IsolatedAsyncioTestCase):
    async def test_fragment_is_truncated_to_max_chars(self) -> None:
        debouncer = ConversationDebouncer(0, max_fragment_chars=10)
        finished: list[str] = []

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            finished.append(text)

        debouncer.submit("chat", "x" * 50, run_cycle)
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(finished, ["x" * 10])

    async def test_buffer_keeps_only_most_recent_fragments(self) -> None:
        debouncer = ConversationDebouncer(_DEBOUNCE, max_fragments=3)
        finished: list[str] = []

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            finished.append(text)

        for index in range(6):
            debouncer.submit("chat", str(index), run_cycle)
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(finished, ["3\n4\n5"])

    async def test_max_wait_caps_starvation_of_continuous_senders(self) -> None:
        # Mensagens a cada 30ms com debounce de 50ms nunca deixariam a janela fechar sem o teto.
        debouncer = ConversationDebouncer(0.05, max_wait_seconds=0.12)
        started_at_fragment: list[int] = []
        sent = 0

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            started_at_fragment.append(sent)

        for index in range(25):
            sent = index + 1
            debouncer.submit("chat", str(index), run_cycle)
            await asyncio.sleep(0.03)
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertTrue(started_at_fragment)
        self.assertLess(started_at_fragment[0], 25)


class CancelledCycleCleanupFailureTest(unittest.IsolatedAsyncioTestCase):
    async def test_exception_replacing_cancellation_does_not_consume_fragments(self) -> None:
        debouncer = ConversationDebouncer(0)
        running = asyncio.Event()
        finished: list[str] = []
        calls = 0

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                running.set()
                try:
                    await asyncio.sleep(60)
                finally:
                    # cleanup que lanca excecao comum e substitui o CancelledError
                    raise RuntimeError("falha no cleanup")
            finished.append(text)

        debouncer.submit("chat", "a", run_cycle)
        await running.wait()
        debouncer.submit("chat", "b", run_cycle)
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(finished, ["a\nb"])


class ShutdownTimeoutTest(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_times_out_on_stuck_committed_cycle(self) -> None:
        debouncer = ConversationDebouncer(0)
        running = asyncio.Event()

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            guard.commit()
            running.set()
            await asyncio.sleep(60)

        debouncer.submit("chat", "a", run_cycle)
        await running.wait()
        tasks = [c.task for s in debouncer._states.values() for c in s.cycles]

        await asyncio.wait_for(debouncer.shutdown(timeout=0.05), timeout=5)

        self.assertTrue(all(task is not None and task.done() for task in tasks))
        self.assertEqual(debouncer._states, {})

class MaxWaitCommitTest(unittest.IsolatedAsyncioTestCase):
    async def test_cycle_started_after_max_wait_is_not_cancelled_by_new_messages(self) -> None:
        debouncer = ConversationDebouncer(0.05, max_wait_seconds=0.05)
        started = asyncio.Event()
        release = asyncio.Event()
        texts: list[str] = []
        finished: list[str] = []

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            texts.append(text)
            if len(texts) == 1:
                started.set()
                await release.wait()
            finished.append(text)

        debouncer.submit("chat", "a", run_cycle)
        await asyncio.wait_for(started.wait(), timeout=5)
        debouncer.submit("chat", "b", run_cycle)  # chega com o ciclo em execução
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.wait_for(debouncer.join(), timeout=5)

        self.assertEqual(finished, ["a", "b"])
        self.assertEqual(texts, ["a", "b"])

    async def test_cycle_within_normal_window_is_still_cancellable(self) -> None:
        debouncer = ConversationDebouncer(0.05)
        started = asyncio.Event()
        cancelled: list[str] = []

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.append(text)
                raise

        debouncer.submit("chat", "a", run_cycle)
        await asyncio.wait_for(started.wait(), timeout=5)
        debouncer.submit("chat", "b", run_cycle)
        await asyncio.sleep(0.01)
        self.assertEqual(cancelled, ["a"])
        await debouncer.shutdown(timeout=0.05)


class UseAfterShutdownTest(unittest.IsolatedAsyncioTestCase):
    async def test_submit_after_shutdown_schedules_nothing(self) -> None:
        debouncer = ConversationDebouncer(0)
        ran: list[str] = []

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            ran.append(text)

        await debouncer.shutdown()
        debouncer.submit("chat", "a", run_cycle)
        slot = debouncer.reserve("chat", run_cycle)
        slot.fill("b")
        await asyncio.sleep(0.02)

        self.assertEqual(ran, [])
        self.assertEqual(debouncer._states, {})

    async def test_fill_and_release_of_pending_slot_after_shutdown_schedule_nothing(self) -> None:
        debouncer = ConversationDebouncer(0)
        ran: list[str] = []

        async def run_cycle(text: str, guard: CycleGuard) -> None:
            ran.append(text)

        slot_fill = debouncer.reserve("chat-1", run_cycle)
        slot_release = debouncer.reserve("chat-2", run_cycle)
        await debouncer.shutdown()
        slot_fill.fill("audio transcrito")
        slot_release.release()
        await asyncio.sleep(0.02)

        self.assertEqual(ran, [])
        self.assertFalse(
            any(c.task and not c.task.done() for s in debouncer._states.values() for c in s.cycles)
        )
        self.assertEqual(debouncer._states, {})


if __name__ == "__main__":
    unittest.main()
