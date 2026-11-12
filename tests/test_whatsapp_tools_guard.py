import json
import unittest

from src.application.services.conversation_debouncer import CycleGuard
from src.infrastructure.adapters.agent.whatsapp_tools import schedule_customer_appointment


class FakeClient:
    def __init__(self, guard: CycleGuard) -> None:
        self._guard = guard
        self.committed_at_call: list[bool] = []

    async def create_booking(self, **kwargs) -> dict:
        # registra se o ciclo ja estava committed no momento da criacao do agendamento
        self.committed_at_call.append(self._guard.committed)
        return {"id": "booking-1"}


def _args(**overrides) -> dict:
    args = {
        "service_id": "svc-1",
        "professional_id": "pro-1",
        "date": "2026-11-03",
        "time": "14:00",
        "customer_name": "Joao",
    }
    args.update(overrides)
    return args


class ScheduleToolCycleGuardTest(unittest.IsolatedAsyncioTestCase):
    def _config(self, guard: CycleGuard, client: FakeClient) -> dict:
        return {
            "configurable": {
                "cycle_guard": guard,
                "schedula_client": client,
                "customer_phone": "5511999999999",
            }
        }

    async def test_commit_happens_before_create_booking(self) -> None:
        guard = CycleGuard()
        client = FakeClient(guard)

        result = await schedule_customer_appointment.ainvoke(
            _args(), config=self._config(guard, client)
        )

        self.assertEqual(json.loads(result)["status"], "scheduled")
        self.assertEqual(client.committed_at_call, [True])
        self.assertTrue(guard.committed)

    async def test_validation_failure_does_not_commit_nor_book(self) -> None:
        for bad in ({"date": "03/11/2026"}, {"time": "2pm"}, {"customer_name": "  "}):
            with self.subTest(bad=bad):
                guard = CycleGuard()
                client = FakeClient(guard)

                result = await schedule_customer_appointment.ainvoke(
                    _args(**bad), config=self._config(guard, client)
                )

                self.assertIn("error", json.loads(result))
                self.assertFalse(guard.committed)
                self.assertEqual(client.committed_at_call, [])

    async def test_works_without_guard(self) -> None:
        guard = CycleGuard()
        client = FakeClient(guard)

        result = await schedule_customer_appointment.ainvoke(
            _args(), config={"configurable": {"schedula_client": client}}
        )

        self.assertEqual(json.loads(result)["status"], "scheduled")


if __name__ == "__main__":
    unittest.main()
