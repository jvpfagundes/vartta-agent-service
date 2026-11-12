import json
import unittest

import httpx

from src.infrastructure.adapters.http.schedula_client import HttpSchedulaClient
from src.infrastructure.adapters.http.simulation_schedula_client import (
    SIMULATED_CUSTOMER_ID,
    SimulationSchedulaClient,
)


def _client_with(handler) -> tuple[HttpSchedulaClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = HttpSchedulaClient(tenant_id="tenant-1", user_id="5511999999999")
    client._client = httpx.AsyncClient(
        base_url="http://backend.test", transport=httpx.MockTransport(recording)
    )
    return client, requests


class HttpCustomerMethodsTest(unittest.IsolatedAsyncioTestCase):
    async def test_find_customer_by_phone_returns_costumer(self) -> None:
        customer = {"id": "c1", "name": "Joao Silva", "phone_number": "5511999999999"}
        client, requests = _client_with(
            lambda r: httpx.Response(200, json={"status": True, "costumer": customer})
        )

        result = await client.find_customer_by_phone("5511999999999")

        self.assertEqual(result, customer)
        self.assertEqual(requests[0].method, "GET")
        self.assertEqual(requests[0].url.path, "/api/costumers/by-phone")
        self.assertEqual(requests[0].url.params["phone_number"], "5511999999999")

    async def test_find_customer_by_phone_returns_none_when_missing(self) -> None:
        client, _ = _client_with(
            lambda r: httpx.Response(200, json={"status": True, "costumer": None})
        )

        self.assertIsNone(await client.find_customer_by_phone("5511999999999"))

    async def test_create_customer_posts_name_and_phone(self) -> None:
        client, requests = _client_with(
            lambda r: httpx.Response(200, json={"status_code": 200, "costumer_id": "new-id"})
        )

        result = await client.create_customer(name="Ana Lima", phone="5511999999999")

        self.assertEqual(
            result, {"id": "new-id", "name": "Ana Lima", "phone_number": "5511999999999"}
        )
        self.assertEqual(requests[0].method, "POST")
        self.assertEqual(requests[0].url.path, "/api/costumers/")
        self.assertEqual(
            json.loads(requests[0].content),
            {"name": "Ana Lima", "phone_number": "5511999999999"},
        )

    async def test_update_customer_name_puts_id_and_name(self) -> None:
        client, requests = _client_with(
            lambda r: httpx.Response(200, json={"status_code": 200, "costumer_id": "c1"})
        )

        await client.update_customer_name("c1", "Joao Silva")

        self.assertEqual(requests[0].method, "PUT")
        self.assertEqual(requests[0].url.path, "/api/costumers/")
        self.assertEqual(json.loads(requests[0].content), {"id": "c1", "name": "Joao Silva"})

    async def test_http_error_raises(self) -> None:
        client, _ = _client_with(lambda r: httpx.Response(403, json={}))

        with self.assertRaises(httpx.HTTPStatusError):
            await client.find_customer_by_phone("5511999999999")


class FakeInner:
    def __init__(self) -> None:
        self.find_calls: list[str] = []
        self.write_calls = 0

    async def find_customer_by_phone(self, phone: str) -> dict | None:
        self.find_calls.append(phone)
        return {"id": "c1", "name": "Joao"}

    async def create_customer(self, name: str, phone: str) -> dict:
        self.write_calls += 1
        return {}

    async def update_customer_name(self, customer_id: str, name: str) -> dict:
        self.write_calls += 1
        return {}


class SimulationCustomerMethodsTest(unittest.IsolatedAsyncioTestCase):
    async def test_find_is_delegated_and_writes_are_simulated(self) -> None:
        inner = FakeInner()
        client = SimulationSchedulaClient(inner)  # type: ignore[arg-type]

        found = await client.find_customer_by_phone("5511999999999")
        created = await client.create_customer(name="Ana Lima", phone="5511999999999")
        updated = await client.update_customer_name("c1", "Joao Silva")

        self.assertEqual(found, {"id": "c1", "name": "Joao"})
        self.assertEqual(inner.find_calls, ["5511999999999"])
        self.assertEqual(created["id"], SIMULATED_CUSTOMER_ID)
        self.assertTrue(updated["simulated"])
        self.assertEqual(inner.write_calls, 0)

    async def test_register_customer_tool_is_simulated_without_writing(self) -> None:
        from src.infrastructure.adapters.agent.whatsapp_tools import register_customer

        class MissingInner(FakeInner):
            async def find_customer_by_phone(self, phone: str) -> dict | None:
                self.find_calls.append(phone)
                return None

        inner = MissingInner()
        client = SimulationSchedulaClient(inner)  # type: ignore[arg-type]

        result = await register_customer.ainvoke(
            {"full_name": "Ana Lima"},
            config={"configurable": {"schedula_client": client, "customer_phone": "5511999999999"}},
        )

        self.assertIn('"registered"', result)
        self.assertEqual(inner.write_calls, 0)


if __name__ == "__main__":
    unittest.main()
