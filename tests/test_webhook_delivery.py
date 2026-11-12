import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.application.services.conversation_debouncer import ConversationDebouncer
from src.infrastructure.entrypoints import webhook


class FakeWahaClient:
    instances: list["FakeWahaClient"] = []
    fail_on_call: int | None = None

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.calls = 0
        self.typing_started = 0
        FakeWahaClient.instances.append(self)

    async def start_typing(self, session_name: str, chat_id: str) -> None:
        self.typing_started += 1

    async def send_text(self, session_name: str, chat_id: str, text: str) -> None:
        self.calls += 1
        if FakeWahaClient.fail_on_call == self.calls:
            raise RuntimeError("falha no envio")
        self.sent.append(text)


class FakeSchedulaClient:
    instances: list["FakeSchedulaClient"] = []

    def __init__(self, tenant_id: str, user_id: str) -> None:
        self.closed = False
        FakeSchedulaClient.instances.append(self)

    async def close(self) -> None:
        self.closed = True


class FakeUseCase:
    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.delivered_records: list[list[str]] = []

    async def record_delivered_replies(self, context, delivered: list[str]) -> None:
        self.delivered_records.append(list(delivered))

    async def execute(self, context, user_message, schedula_client, cycle_guard=None) -> list[str]:
        return self._replies


def _extracted() -> dict:
    return {
        "session_name": "sessao",
        "chat_id": "5511999999999@c.us",
        "customer_phone": "5511999999999",
        "customer_name": "Joao",
        "message_text": "oi",
        "is_audio": False,
        "media_url": "",
        "media_base64": "",
        "media_mimetype": "audio/ogg",
    }


def _tenant_config() -> SimpleNamespace:
    return SimpleNamespace(
        tenant_id="tenant-delivery",
        instance_name="sessao",
        ai_enabled=True,
        business_name="Barbearia Teste",
        ai_custom_prompt="",
        ai_settings={},
        addon_active=True,
    )


class ProcessWhatsAppMessageDeliveryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        FakeWahaClient.instances = []
        FakeWahaClient.fail_on_call = None
        FakeSchedulaClient.instances = []

    async def _run(self, replies: list[str]) -> FakeUseCase:
        debouncer = ConversationDebouncer(0)
        use_case = FakeUseCase(replies)
        with (
            patch.object(webhook, "get_debouncer", return_value=debouncer),
            patch.object(
                webhook,
                "get_tenant_config_by_instance",
                AsyncMock(return_value=_tenant_config()),
            ),
            patch.object(webhook, "WahaClient", FakeWahaClient),
            patch.object(webhook, "HttpSchedulaClient", FakeSchedulaClient),
            patch(
                "src.infrastructure.entrypoints.app.get_whatsapp_use_case",
                return_value=use_case,
            ),
        ):
            await webhook._process_whatsapp_message(_extracted())
            await debouncer.join()
        return use_case

    async def test_sends_each_part_in_order(self) -> None:
        use_case = await self._run(["a", "b", "c"])

        waha = FakeWahaClient.instances[0]
        self.assertEqual(waha.sent, ["a", "b", "c"])
        self.assertEqual(use_case.delivered_records, [])
        self.assertEqual(waha.typing_started, 1)
        self.assertTrue(FakeSchedulaClient.instances[0].closed)

    async def test_stops_remaining_parts_when_second_send_fails(self) -> None:
        FakeWahaClient.fail_on_call = 2

        use_case = await self._run(["a", "b", "c"])

        waha = FakeWahaClient.instances[0]
        self.assertEqual(waha.sent, ["a"])
        self.assertEqual(waha.calls, 2)
        # o checkpoint e corrigido para conter apenas a parte entregue
        self.assertEqual(use_case.delivered_records, [["a"]])
        self.assertTrue(FakeSchedulaClient.instances[0].closed)

    async def test_empty_replies_send_nothing_and_close_client(self) -> None:
        await self._run([])

        self.assertEqual(FakeWahaClient.instances[0].sent, [])
        self.assertTrue(FakeSchedulaClient.instances[0].closed)


if __name__ == "__main__":
    unittest.main()
