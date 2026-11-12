import json
import unittest

import httpx

from src.infrastructure.adapters.agent.whatsapp_tools import (
    list_available_slots,
    schedule_customer_appointment,
)
from src.infrastructure.adapters.http.schedula_client import HttpSchedulaClient
from src.infrastructure.adapters.http.simulation_schedula_client import (
    SIMULATED_BOOKING_RESULT,
    SimulationSchedulaClient,
)


class RecordingClient:
    def __init__(self) -> None:
        self.booking_calls: list[dict] = []
        self.slot_calls: list[tuple[tuple, dict]] = []

    async def create_booking(self, **kwargs) -> dict:
        self.booking_calls.append(kwargs)
        return {"id": "booking-1"}

    async def list_available_slots(self, *args, **kwargs) -> dict:
        self.slot_calls.append((args, kwargs))
        return {"slots": ["10:00"]}


def _schedule_args(**overrides) -> dict:
    args = {
        "service_id": "svc-1",
        "professional_id": "pro-1",
        "date": "2026-11-03",
        "time": "10:00",
        "customer_name": "Joao",
    }
    args.update(overrides)
    return args


def _config(client) -> dict:
    return {"configurable": {"schedula_client": client, "customer_phone": "5511999999999"}}


class ScheduleToolMultiServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_additional_ids_are_normalized_and_booked_once(self) -> None:
        client = RecordingClient()

        result = await schedule_customer_appointment.ainvoke(
            _schedule_args(additional_service_ids=["svc-2", "svc-2", "svc-1", "", "  "]),
            config=_config(client),
        )

        self.assertEqual(json.loads(result)["status"], "scheduled")
        self.assertEqual(len(client.booking_calls), 1)
        self.assertEqual(client.booking_calls[0]["service_id"], "svc-1")
        self.assertEqual(client.booking_calls[0]["additional_service_ids"], ["svc-2"])

    async def test_additional_ids_are_capped(self) -> None:
        client = RecordingClient()

        await schedule_customer_appointment.ainvoke(
            _schedule_args(additional_service_ids=[f"x{i}" for i in range(10)]),
            config=_config(client),
        )

        self.assertEqual(len(client.booking_calls[0]["additional_service_ids"]), 5)

    async def test_without_additional_ids_kwarg_is_not_sent(self) -> None:
        for extra in ({}, {"additional_service_ids": []}, {"additional_service_ids": ["svc-1", ""]}):
            with self.subTest(extra=extra):
                client = RecordingClient()

                await schedule_customer_appointment.ainvoke(
                    _schedule_args(**extra), config=_config(client)
                )

                self.assertEqual(len(client.booking_calls), 1)
                self.assertNotIn("additional_service_ids", client.booking_calls[0])


class SlotsToolMultiServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_additional_ids_only_when_present(self) -> None:
        client = RecordingClient()
        base = {"service_id": "svc-1", "professional_id": "pro-1", "date": "2026-11-03"}

        await list_available_slots.ainvoke(base, config=_config(client))
        await list_available_slots.ainvoke(
            {**base, "additional_service_ids": ["svc-2", "svc-2"]}, config=_config(client)
        )

        self.assertEqual(client.slot_calls[0], (("svc-1", "pro-1", "2026-11-03"), {}))
        self.assertEqual(
            client.slot_calls[1],
            (("svc-1", "pro-1", "2026-11-03"), {"additional_service_ids": ["svc-2"]}),
        )


def _http_client(requests: list[httpx.Request]) -> HttpSchedulaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "slots": []})

    client = HttpSchedulaClient(tenant_id="tenant-1", user_id="5511999999999")
    client._client = httpx.AsyncClient(
        base_url="http://backend.test", transport=httpx.MockTransport(handler)
    )
    return client


_BOOKING = {
    "service_id": "svc-1",
    "professional_id": "pro-1",
    "date": "2026-11-03",
    "time": "10:00",
    "customer_name": "Joao",
    "customer_phone": "11999999999",
}


class HttpMultiServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_booking_body_includes_additional_ids(self) -> None:
        requests: list[httpx.Request] = []
        client = _http_client(requests)

        await client.create_booking(**_BOOKING, additional_service_ids=["svc-2", "svc-3"])

        body = json.loads(requests[0].content)
        self.assertEqual(requests[0].url.path, "/api/booking/create")
        self.assertEqual(body["additional_service_ids"], ["svc-2", "svc-3"])
        self.assertEqual(body["costumer_name"], "Joao")

    async def test_create_booking_body_omits_additional_ids_when_absent(self) -> None:
        for extra in ({}, {"additional_service_ids": None}, {"additional_service_ids": []}):
            with self.subTest(extra=extra):
                requests: list[httpx.Request] = []
                client = _http_client(requests)

                await client.create_booking(**_BOOKING, **extra)

                self.assertNotIn("additional_service_ids", json.loads(requests[0].content))

    async def test_slots_query_repeats_additional_ids(self) -> None:
        requests: list[httpx.Request] = []
        client = _http_client(requests)

        await client.list_available_slots(
            "svc-1", "pro-1", "2026-11-03", additional_service_ids=["svc-2", "svc-3"]
        )

        params = requests[0].url.params
        self.assertEqual(requests[0].url.path, "/api/booking/slots")
        self.assertEqual(params.get_list("additional_service_ids"), ["svc-2", "svc-3"])
        self.assertEqual(params["service_id"], "svc-1")

    async def test_slots_query_omits_additional_ids_when_absent(self) -> None:
        requests: list[httpx.Request] = []
        client = _http_client(requests)

        await client.list_available_slots("svc-1", "pro-1", "2026-11-03")
        await client.list_available_slots(
            "svc-1", "pro-1", "2026-11-03", additional_service_ids=[]
        )

        for request in requests:
            self.assertNotIn("additional_service_ids", request.url.params)


class SimulationMultiServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_additional_ids(self) -> None:
        requests: list[httpx.Request] = []
        client = SimulationSchedulaClient(_http_client(requests))

        booking = await client.create_booking(**_BOOKING, additional_service_ids=["svc-2"])
        await client.list_available_slots(
            "svc-1", "pro-1", "2026-11-03", additional_service_ids=["svc-2"]
        )

        self.assertEqual(booking, SIMULATED_BOOKING_RESULT)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.params.get_list("additional_service_ids"), ["svc-2"])


if __name__ == "__main__":
    unittest.main()
