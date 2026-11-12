import inspect
import json
import unittest

from src.infrastructure.adapters.agent.whatsapp_tools import (
    WHATSAPP_CUSTOMER_TOOLS,
    register_customer,
    schedule_customer_appointment,
)

_PHONE = "5511999999999"


class FakeClient:
    def __init__(
        self,
        existing: dict | None = None,
        error: Exception | None = None,
    ) -> None:
        self._existing = existing
        self._error = error
        self.find_calls: list[str] = []
        self.create_calls: list[tuple[str, str]] = []
        self.updates: list[tuple[str, str]] = []
        self.bookings: list[dict] = []

    async def find_customer_by_phone(self, phone: str) -> dict | None:
        self.find_calls.append(phone)
        if self._error:
            raise self._error
        return self._existing

    async def create_customer(self, name: str, phone: str) -> dict:
        self.create_calls.append((name, phone))
        return {"id": "new-id", "name": name, "phone_number": phone}

    async def update_customer_name(self, customer_id: str, name: str) -> dict:
        self.updates.append((customer_id, name))
        return {"costumer_id": customer_id}

    async def create_booking(self, **kwargs) -> dict:
        self.bookings.append(kwargs)
        return {"id": "booking-1"}


def _config(client: FakeClient, **configurable) -> dict:
    return {"configurable": {"schedula_client": client, **configurable}}


class RegisterCustomerToolTest(unittest.IsolatedAsyncioTestCase):
    async def test_is_registered_in_customer_tools(self) -> None:
        self.assertIn(register_customer, WHATSAPP_CUSTOMER_TOOLS)

    async def test_creates_with_full_name_and_configurable_phone(self) -> None:
        client = FakeClient()

        result = await register_customer.ainvoke(
            {"full_name": "Maria Souza"}, config=_config(client, customer_phone=_PHONE)
        )

        self.assertEqual(json.loads(result), {"status": "registered", "name": "Maria Souza"})
        self.assertEqual(client.find_calls, [_PHONE])
        self.assertEqual(client.create_calls, [("Maria Souza", _PHONE)])
        self.assertEqual(client.updates, [])

    async def test_phone_is_normalized_to_digits(self) -> None:
        client = FakeClient()

        await register_customer.ainvoke(
            {"full_name": "Maria Souza"}, config=_config(client, customer_phone="+55 (11) 99999-9999")
        )

        self.assertEqual(client.create_calls, [("Maria Souza", _PHONE)])

    async def test_llm_cannot_choose_the_phone(self) -> None:
        self.assertNotIn("phone", inspect.signature(register_customer.coroutine).parameters)
        self.assertNotIn("phone", register_customer.args)
        client = FakeClient()

        await register_customer.ainvoke(
            {"full_name": "Maria Souza", "phone": "5500000000000"},
            config=_config(client, customer_phone=_PHONE),
        )

        self.assertEqual(client.create_calls, [("Maria Souza", _PHONE)])

    async def test_rejects_invalid_names_without_calling_client(self) -> None:
        for bad in ("Maria", "Cliente", "sem nome", "Cliente WhatsApp", "", "   ", "😀", "123"):
            with self.subTest(bad=bad):
                client = FakeClient()
                result = await register_customer.ainvoke(
                    {"full_name": bad}, config=_config(client, customer_phone=_PHONE)
                )
                self.assertIn("error", json.loads(result))
                self.assertEqual(client.find_calls, [])
                self.assertEqual(client.create_calls, [])
                self.assertEqual(client.updates, [])

    async def test_missing_phone_returns_error(self) -> None:
        for phone in (None, "", "abc"):
            with self.subTest(phone=phone):
                client = FakeClient()
                result = await register_customer.ainvoke(
                    {"full_name": "Maria Souza"}, config=_config(client, customer_phone=phone)
                )
                self.assertIn("error", json.loads(result))
                self.assertEqual(client.find_calls, [])
                self.assertEqual(client.create_calls, [])

    async def test_existing_complete_customer_is_not_changed(self) -> None:
        client = FakeClient(existing={"id": "c1", "name": "Joao Silva"})

        result = await register_customer.ainvoke(
            {"full_name": "Outro Nome"}, config=_config(client, customer_phone=_PHONE)
        )

        self.assertEqual(
            json.loads(result), {"status": "already_registered", "name": "Joao Silva"}
        )
        self.assertEqual(client.create_calls, [])
        self.assertEqual(client.updates, [])

    async def test_existing_incomplete_customer_is_updated_not_duplicated(self) -> None:
        for registered in ("Cliente", "Maria"):
            with self.subTest(registered=registered):
                client = FakeClient(existing={"id": "c1", "name": registered})

                result = await register_customer.ainvoke(
                    {"full_name": "Maria Souza"}, config=_config(client, customer_phone=_PHONE)
                )

                self.assertEqual(
                    json.loads(result), {"status": "updated", "name": "Maria Souza"}
                )
                self.assertEqual(client.create_calls, [])
                self.assertEqual(client.updates, [("c1", "Maria Souza")])

    async def test_client_exception_returns_json_error(self) -> None:
        client = FakeClient(error=RuntimeError("boom"))

        result = await register_customer.ainvoke(
            {"full_name": "Maria Souza"}, config=_config(client, customer_phone=_PHONE)
        )

        self.assertIn("error", json.loads(result))
        self.assertNotIn("boom", result)
        self.assertEqual(client.create_calls, [])


class BookingAfterRegistrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_booking_uses_llm_full_name_when_cycle_has_no_registered_name(self) -> None:
        client = FakeClient()

        await schedule_customer_appointment.ainvoke(
            {
                "service_id": "svc-1",
                "professional_id": "pro-1",
                "date": "2026-11-03",
                "time": "14:00",
                "customer_name": "Maria Souza",
            },
            config=_config(
                client,
                customer_phone=_PHONE,
                customer_id=None,
                customer_registered_name="",
            ),
        )

        self.assertEqual(client.bookings[0]["customer_name"], "Maria Souza")


if __name__ == "__main__":
    unittest.main()
