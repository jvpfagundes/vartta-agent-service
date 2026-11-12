import json
import unittest

from src.infrastructure.adapters.agent.whatsapp_tools import (
    WHATSAPP_CUSTOMER_TOOLS,
    schedule_customer_appointment,
    update_customer_name,
)


class FakeClient:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.updates: list[tuple[str, str]] = []
        self.bookings: list[dict] = []

    async def update_customer_name(self, customer_id: str, name: str) -> dict:
        if self._error:
            raise self._error
        self.updates.append((customer_id, name))
        return {"costumer_id": customer_id}

    async def create_booking(self, **kwargs) -> dict:
        self.bookings.append(kwargs)
        return {"id": "booking-1"}


def _config(client: FakeClient, **configurable) -> dict:
    return {"configurable": {"schedula_client": client, **configurable}}


class UpdateCustomerNameToolTest(unittest.IsolatedAsyncioTestCase):
    async def test_is_registered_in_customer_tools(self) -> None:
        self.assertIn(update_customer_name, WHATSAPP_CUSTOMER_TOOLS)

    async def test_updates_using_customer_id_from_config(self) -> None:
        client = FakeClient()

        result = await update_customer_name.ainvoke(
            {"full_name": "Joao Silva"}, config=_config(client, customer_id="c-123")
        )

        self.assertEqual(json.loads(result), {"status": "updated", "name": "Joao Silva"})
        self.assertEqual(client.updates, [("c-123", "Joao Silva")])

    async def test_llm_cannot_override_customer_id(self) -> None:
        client = FakeClient()

        await update_customer_name.ainvoke(
            {"full_name": "Joao Silva", "customer_id": "outro-cliente"},
            config=_config(client, customer_id="c-123"),
        )

        self.assertEqual(client.updates, [("c-123", "Joao Silva")])

    async def test_rejects_invalid_names_without_calling_client(self) -> None:
        for bad in ("Joao", "Cliente", "sem nome", "", "   ", "😀", "123"):
            with self.subTest(bad=bad):
                client = FakeClient()
                result = await update_customer_name.ainvoke(
                    {"full_name": bad}, config=_config(client, customer_id="c-123")
                )
                self.assertIn("error", json.loads(result))
                self.assertEqual(client.updates, [])

    async def test_missing_customer_id_returns_error(self) -> None:
        client = FakeClient()

        result = await update_customer_name.ainvoke(
            {"full_name": "Joao Silva"}, config=_config(client)
        )

        self.assertIn("error", json.loads(result))
        self.assertEqual(client.updates, [])

    async def test_client_exception_returns_json_error(self) -> None:
        client = FakeClient(error=RuntimeError("boom"))

        result = await update_customer_name.ainvoke(
            {"full_name": "Joao Silva"}, config=_config(client, customer_id="c-123")
        )

        self.assertIn("error", json.loads(result))
        self.assertNotIn("boom", result)


class BookingNameTest(unittest.IsolatedAsyncioTestCase):
    def _args(self, name: str) -> dict:
        return {
            "service_id": "svc-1",
            "professional_id": "pro-1",
            "date": "2026-11-03",
            "time": "14:00",
            "customer_name": name,
        }

    async def test_keeps_registered_full_name_when_llm_sends_first_name(self) -> None:
        client = FakeClient()

        await schedule_customer_appointment.ainvoke(
            self._args("Joao"),
            config=_config(
                client, customer_phone="5511999999999", customer_registered_name="Joao Silva"
            ),
        )

        self.assertEqual(client.bookings[0]["customer_name"], "Joao Silva")

    async def test_uses_llm_name_when_more_complete_than_registered(self) -> None:
        client = FakeClient()

        await schedule_customer_appointment.ainvoke(
            self._args("Joao Silva"),
            config=_config(
                client, customer_phone="5511999999999", customer_registered_name="Joao"
            ),
        )

        self.assertEqual(client.bookings[0]["customer_name"], "Joao Silva")

    async def test_uses_llm_name_without_registered_name(self) -> None:
        client = FakeClient()

        await schedule_customer_appointment.ainvoke(
            self._args("Joao"), config=_config(client, customer_phone="5511999999999")
        )

        self.assertEqual(client.bookings[0]["customer_name"], "Joao")


if __name__ == "__main__":
    unittest.main()
