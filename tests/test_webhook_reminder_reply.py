import unittest
from unittest.mock import AsyncMock, patch

from src.application.services.conversation_debouncer import ConversationDebouncer
from src.domain.conversation.models import TenantWhatsAppConfig
from src.infrastructure.entrypoints import webhook


class FakeWahaClient:
    instances: list["FakeWahaClient"] = []

    def __init__(self) -> None:
        self.sent: list[str] = []
        FakeWahaClient.instances.append(self)

    async def start_typing(self, session_name: str, chat_id: str) -> None:
        return None

    async def send_text(self, session_name: str, chat_id: str, text: str) -> None:
        self.sent.append(text)


class FakeSchedulaClient:
    instances: list["FakeSchedulaClient"] = []
    outcome: dict = {"result": "no_pending"}
    error: Exception | None = None

    def __init__(self, tenant_id: str, user_id: str) -> None:
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.calls: list[tuple[str, str]] = []
        self.closed = False
        FakeSchedulaClient.instances.append(self)

    async def reply_to_reminder(self, phone_number: str, choice: str) -> dict:
        self.calls.append((phone_number, choice))
        if FakeSchedulaClient.error:
            raise FakeSchedulaClient.error
        return FakeSchedulaClient.outcome

    async def close(self) -> None:
        self.closed = True


class FakeUseCase:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def execute(self, context, user_message, schedula_client, cycle_guard=None) -> list[str]:
        self.messages.append(user_message)
        return ["resposta da IA"]

    async def record_delivered_replies(self, context, delivered: list[str]) -> None:
        return None


def _extracted(text: str, is_audio: bool = False, phone: str = "5511999999999") -> dict:
    return {
        "session_name": "sessao",
        "chat_id": f"{phone}@c.us",
        "customer_phone": phone,
        "customer_name": "Joao",
        "message_id": "msg-1",
        "message_text": text,
        "is_audio": is_audio,
        "media_url": "",
        "media_base64": "",
        "media_mimetype": "audio/ogg",
    }


def _config(ai_enabled: bool = True, addon_active: bool = True) -> TenantWhatsAppConfig:
    return TenantWhatsAppConfig(
        tenant_id="tenant-a",
        instance_name="sessao",
        business_name="Barbearia Teste",
        ai_enabled=ai_enabled,
        addon_active=addon_active,
    )


class WebhookReminderReplyTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        FakeWahaClient.instances = []
        FakeSchedulaClient.instances = []
        FakeSchedulaClient.outcome = {"result": "no_pending"}
        FakeSchedulaClient.error = None

    async def _run(self, extracted: dict, config: TenantWhatsAppConfig) -> FakeUseCase:
        debouncer = ConversationDebouncer(0)
        use_case = FakeUseCase()
        with (
            patch.object(webhook, "get_debouncer", return_value=debouncer),
            patch.object(
                webhook, "get_tenant_config_by_instance", AsyncMock(return_value=config)
            ),
            patch.object(webhook, "WahaClient", FakeWahaClient),
            patch.object(webhook, "HttpSchedulaClient", FakeSchedulaClient),
            patch(
                "src.infrastructure.entrypoints.app.get_whatsapp_use_case",
                return_value=use_case,
            ),
        ):
            await webhook._process_whatsapp_message(extracted)
            await debouncer.join()
        return use_case

    def _sent(self) -> list[str]:
        return [text for waha in FakeWahaClient.instances for text in waha.sent]

    async def test_confirmation_is_forwarded_and_answered_without_the_ai(self) -> None:
        FakeSchedulaClient.outcome = {"result": "confirmed", "message": "Agendamento confirmado"}

        use_case = await self._run(_extracted("1"), _config())

        client = FakeSchedulaClient.instances[0]
        self.assertEqual(client.calls, [("5511999999999", "1")])
        self.assertEqual(client.tenant_id, "tenant-a")
        self.assertTrue(client.closed)
        self.assertEqual(self._sent(), ["Agendamento confirmado"])
        self.assertEqual(use_case.messages, [])

    async def test_cancellation_is_forwarded(self) -> None:
        FakeSchedulaClient.outcome = {"result": "canceled", "message": "Cancelado"}

        await self._run(_extracted("2"), _config())

        self.assertEqual(FakeSchedulaClient.instances[0].calls, [("5511999999999", "2")])
        self.assertEqual(self._sent(), ["Cancelado"])

    async def test_reschedule_is_forwarded_and_answered_without_the_ai(self) -> None:
        FakeSchedulaClient.outcome = {"result": "reschedule", "message": "Um atendente responde"}

        use_case = await self._run(_extracted("3"), _config())

        self.assertEqual(FakeSchedulaClient.instances[0].calls, [("5511999999999", "3")])
        self.assertEqual(self._sent(), ["Um atendente responde"])
        self.assertEqual(use_case.messages, [])

    async def test_works_for_tenants_without_the_ai_addon(self) -> None:
        FakeSchedulaClient.outcome = {"result": "confirmed", "message": "Agendamento confirmado"}

        await self._run(_extracted("1"), _config(ai_enabled=False, addon_active=False))

        self.assertEqual(self._sent(), ["Agendamento confirmado"])

    async def test_without_pending_event_nothing_is_sent_and_the_ai_flow_continues(self) -> None:
        use_case = await self._run(_extracted("1"), _config())

        self.assertEqual(FakeSchedulaClient.instances[0].calls, [("5511999999999", "1")])
        self.assertEqual(use_case.messages, ["1"])
        self.assertEqual(self._sent(), ["resposta da IA"])

    async def test_without_pending_event_and_without_ai_the_message_is_dropped(self) -> None:
        use_case = await self._run(_extracted("1"), _config(ai_enabled=False, addon_active=False))

        self.assertEqual(use_case.messages, [])
        self.assertEqual(self._sent(), [])

    async def test_ambiguous_reply_without_message_follows_the_normal_flow(self) -> None:
        # O aviso de "mais de um agendamento" está desativado no backend (TODO): sem mensagem,
        # nada é enviado pelo lembrete e o texto segue o fluxo normal.
        FakeSchedulaClient.outcome = {"result": "ambiguous"}

        use_case = await self._run(_extracted("1"), _config())

        self.assertEqual(use_case.messages, ["1"])
        self.assertEqual(self._sent(), ["resposta da IA"])

    async def test_backend_failure_does_not_answer_and_falls_back_to_the_ai_flow(self) -> None:
        FakeSchedulaClient.error = RuntimeError("boom")

        use_case = await self._run(_extracted("1"), _config())

        self.assertTrue(FakeSchedulaClient.instances[0].closed)
        self.assertEqual(use_case.messages, ["1"])

    async def test_other_texts_never_reach_the_reminder_endpoint(self) -> None:
        for text in ["oi", "12", "1 corte", "sim"]:
            with self.subTest(text=text):
                FakeSchedulaClient.instances = []
                await self._run(_extracted(text), _config())
                self.assertEqual(FakeSchedulaClient.instances and FakeSchedulaClient.instances[0].calls, [])

    async def test_missing_phone_skips_the_reminder_endpoint(self) -> None:
        use_case = await self._run(_extracted("1", phone=""), _config())

        self.assertEqual(use_case.messages, ["1"])
        self.assertTrue(all(not c.calls for c in FakeSchedulaClient.instances))


if __name__ == "__main__":
    unittest.main()
