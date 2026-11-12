import unittest
from unittest.mock import AsyncMock, patch

from src.application.services.conversation_debouncer import ConversationDebouncer
from src.domain.conversation.models import TenantWhatsAppConfig, WhatsAppConversationContext
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

    def __init__(self, tenant_id: str, user_id: str) -> None:
        self.closed = False
        FakeSchedulaClient.instances.append(self)

    async def close(self) -> None:
        self.closed = True


class FakeUseCase:
    def __init__(self) -> None:
        self.contexts: list[WhatsAppConversationContext] = []

    async def execute(self, context, user_message, schedula_client, cycle_guard=None) -> list[str]:
        self.contexts.append(context)
        return ["resposta"]

    async def record_delivered_replies(self, context, delivered: list[str]) -> None:
        return None


def _extracted() -> dict:
    return {
        "session_name": "sessao",
        "chat_id": "5511999999999@c.us",
        "customer_phone": "5511999999999",
        "customer_name": "Joao",
        "message_id": "msg-1",
        "message_text": "oi",
        "is_audio": False,
        "media_url": "",
        "media_base64": "",
        "media_mimetype": "audio/ogg",
    }


class WebhookAddonGateTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        FakeWahaClient.instances = []
        FakeSchedulaClient.instances = []

    async def _run(self, config: TenantWhatsAppConfig) -> FakeUseCase:
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
            await webhook._process_whatsapp_message(_extracted())
            await debouncer.join()
        return use_case

    async def test_inactive_addon_skips_processing(self) -> None:
        config = TenantWhatsAppConfig(
            tenant_id="tenant-addon",
            instance_name="sessao",
            business_name="Barbearia Teste",
            ai_enabled=True,
            addon_active=False,
        )

        use_case = await self._run(config)

        self.assertEqual(use_case.contexts, [])
        self.assertEqual(FakeWahaClient.instances, [])
        self.assertEqual(FakeSchedulaClient.instances, [])

    async def test_active_addon_passes_style_and_custom_instructions(self) -> None:
        config = TenantWhatsAppConfig(
            tenant_id="tenant-addon",
            instance_name="sessao",
            business_name="Barbearia Teste",
            ai_enabled=True,
            ai_custom_prompt="prompt legado",
            ai_settings={
                "custom_instructions": "instrucoes novas",
                "quiz": {
                    "tone": "formal",
                    "address_form": "senhor",
                    "emoji_usage": "nenhum",
                    "response_length": "detalhadas",
                    "greeting": "",
                    "fallback_behavior": "aguardar_atendente",
                    "fallback_contact": "",
                },
            },
            addon_active=True,
        )

        use_case = await self._run(config)

        self.assertEqual(len(use_case.contexts), 1)
        context = use_case.contexts[0]
        self.assertEqual(context.custom_prompt, "instrucoes novas")
        assert context.style is not None
        self.assertEqual(context.style.tone, "formal")
        self.assertEqual(context.style.address_form, "senhor")
        self.assertEqual(FakeWahaClient.instances[0].sent, ["resposta"])


if __name__ == "__main__":
    unittest.main()
