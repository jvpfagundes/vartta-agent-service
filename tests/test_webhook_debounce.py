import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.application.services.conversation_debouncer import ConversationDebouncer, CycleGuard
from src.infrastructure.entrypoints import webhook


class FakeWahaClient:
    instances: list["FakeWahaClient"] = []
    events: list[str] = []

    def __init__(self) -> None:
        self.sent: list[str] = []
        FakeWahaClient.instances.append(self)

    async def start_typing(self, session_name: str, chat_id: str) -> None:
        return None

    async def send_text(self, session_name: str, chat_id: str, text: str) -> None:
        FakeWahaClient.events.append(f"send:{text}")
        self.sent.append(text)


class FakeSchedulaClient:
    instances: list["FakeSchedulaClient"] = []

    def __init__(self, tenant_id: str, user_id: str) -> None:
        self.closed = False
        FakeSchedulaClient.instances.append(self)

    async def close(self) -> None:
        self.closed = True


class RecordingGuard(CycleGuard):
    def commit(self) -> None:
        FakeWahaClient.events.append("commit")
        super().commit()


class FakeUseCase:
    """Use case falso: a primeira chamada pode ficar bloqueada (execucao lenta do grafo)."""

    def __init__(self, replies: list[str], block_first_call: bool = False) -> None:
        self._replies = replies
        self._block_first_call = block_first_call
        self.messages: list[str] = []
        self.cancelled: list[str] = []
        self.first_call_started = asyncio.Event()

    async def record_delivered_replies(self, context, delivered: list[str]) -> None:
        return None

    async def execute(self, context, user_message, schedula_client, cycle_guard=None) -> list[str]:
        self.messages.append(user_message)
        if self._block_first_call and len(self.messages) == 1:
            self.first_call_started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled.append(user_message)
                raise
        return self._replies


def _extracted(text: str) -> dict:
    return {
        "session_name": "sessao",
        "chat_id": "5511999999999@c.us",
        "customer_phone": "5511999999999",
        "customer_name": "Joao",
        "message_text": text,
        "is_audio": False,
        "media_url": "",
        "media_base64": "",
        "media_mimetype": "audio/ogg",
    }


def _tenant_config() -> SimpleNamespace:
    return SimpleNamespace(
        tenant_id="tenant-debounce",
        instance_name="sessao",
        ai_enabled=True,
        business_name="Barbearia Teste",
        ai_custom_prompt="",
        ai_settings={},
        addon_active=True,
    )


class WebhookDebounceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        FakeWahaClient.instances = []
        FakeWahaClient.events = []
        FakeSchedulaClient.instances = []

    def _patches(self, debouncer: ConversationDebouncer, use_case: FakeUseCase):
        return (
            patch.object(webhook, "get_debouncer", return_value=debouncer),
            patch.object(
                webhook,
                "get_tenant_config_by_instance",
                AsyncMock(return_value=_tenant_config()),
            ),
            patch.object(webhook, "WahaClient", FakeWahaClient),
            patch.object(webhook, "HttpSchedulaClient", FakeSchedulaClient),
            patch("src.infrastructure.entrypoints.app.get_whatsapp_use_case", return_value=use_case),
            patch("src.application.services.conversation_debouncer.CycleGuard", RecordingGuard),
        )

    async def test_fragmented_messages_run_single_cycle_and_send_each_part(self) -> None:
        debouncer = ConversationDebouncer(0.05)
        use_case = FakeUseCase(["parte 1", "parte 2"])
        p = self._patches(debouncer, use_case)
        with p[0], p[1], p[2], p[3], p[4]:
            await webhook._process_whatsapp_message(_extracted("quero"))
            await webhook._process_whatsapp_message(_extracted("cortar o cabelo"))
            await debouncer.join()

        self.assertEqual(use_case.messages, ["quero\ncortar o cabelo"])
        sent = [text for waha in FakeWahaClient.instances for text in waha.sent]
        self.assertEqual(sent, ["parte 1", "parte 2"])
        self.assertTrue(all(client.closed for client in FakeSchedulaClient.instances))

    async def test_new_message_during_slow_execution_cancels_first_cycle(self) -> None:
        debouncer = ConversationDebouncer(0.05)
        use_case = FakeUseCase(["resposta agregada"], block_first_call=True)
        p = self._patches(debouncer, use_case)
        with p[0], p[1], p[2], p[3], p[4]:
            await webhook._process_whatsapp_message(_extracted("primeira"))
            await use_case.first_call_started.wait()
            await webhook._process_whatsapp_message(_extracted("segunda"))
            await debouncer.join()

        self.assertEqual(use_case.messages, ["primeira", "primeira\nsegunda"])
        self.assertEqual(use_case.cancelled, ["primeira"])
        # o ciclo cancelado fechou o cliente do Schedula e nao enviou nada
        self.assertEqual(len(FakeSchedulaClient.instances), 2)
        self.assertTrue(all(client.closed for client in FakeSchedulaClient.instances))
        sent = [text for waha in FakeWahaClient.instances for text in waha.sent]
        self.assertEqual(sent, ["resposta agregada"])

    async def test_commit_happens_before_sending_replies(self) -> None:
        debouncer = ConversationDebouncer(0)
        use_case = FakeUseCase(["a", "b"])
        p = self._patches(debouncer, use_case)
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            await webhook._process_whatsapp_message(_extracted("oi"))
            await debouncer.join()

        self.assertEqual(FakeWahaClient.events, ["commit", "send:a", "send:b"])

    async def test_no_conversation_lock_leftovers(self) -> None:
        self.assertFalse(hasattr(webhook, "_get_conversation_lock"))
        self.assertFalse(hasattr(webhook, "_conversation_locks"))

def _extracted_audio() -> dict:
    data = _extracted("")
    data.update({"is_audio": True, "media_base64": "AAAA"})
    return data


class SlowTranscriber:
    """Transcritor falso: espera um evento antes de devolver o texto (audio lento)."""

    gate: asyncio.Event
    text = "quero cortar o cabelo"
    fail = False

    async def transcribe(self, audio_bytes: bytes, mime_type: str) -> str:
        await SlowTranscriber.gate.wait()
        if SlowTranscriber.fail:
            raise RuntimeError("falha na transcricao")
        return SlowTranscriber.text


class WebhookAudioOrderTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        FakeWahaClient.instances = []
        FakeWahaClient.events = []
        FakeSchedulaClient.instances = []
        SlowTranscriber.gate = asyncio.Event()
        SlowTranscriber.fail = False

    def _patches(self, debouncer: ConversationDebouncer, use_case: FakeUseCase):
        return (
            patch.object(webhook, "get_debouncer", return_value=debouncer),
            patch.object(
                webhook,
                "get_tenant_config_by_instance",
                AsyncMock(return_value=_tenant_config()),
            ),
            patch.object(webhook, "WahaClient", FakeWahaClient),
            patch.object(webhook, "HttpSchedulaClient", FakeSchedulaClient),
            patch.object(webhook, "GroqAudioTranscriber", SlowTranscriber),
            patch("src.infrastructure.entrypoints.app.get_whatsapp_use_case", return_value=use_case),
        )

    async def test_audio_keeps_arrival_order_even_when_text_is_processed_first(self) -> None:
        debouncer = ConversationDebouncer(0.05)
        use_case = FakeUseCase(["ok"])
        p = self._patches(debouncer, use_case)
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            audio_task = asyncio.create_task(webhook._process_whatsapp_message(_extracted_audio()))
            await asyncio.sleep(0.01)  # audio ja reservou a posicao e esta "transcrevendo"
            await webhook._process_whatsapp_message(_extracted("e pra amanha"))
            await asyncio.sleep(0.15)  # mais que o debounce: nenhum ciclo pode rodar
            self.assertEqual(use_case.messages, [])
            SlowTranscriber.gate.set()
            await audio_task
            await debouncer.join()

        self.assertEqual(use_case.messages, ["quero cortar o cabelo\ne pra amanha"])

    async def test_audio_arrival_cancels_running_cycle_before_transcription_ends(self) -> None:
        debouncer = ConversationDebouncer(0.05)
        use_case = FakeUseCase(["resposta"], block_first_call=True)
        p = self._patches(debouncer, use_case)
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            await webhook._process_whatsapp_message(_extracted("primeira"))
            await use_case.first_call_started.wait()
            audio_task = asyncio.create_task(webhook._process_whatsapp_message(_extracted_audio()))
            await asyncio.sleep(0.05)
            # transcricao ainda nao terminou, mas o ciclo em andamento ja foi cancelado
            self.assertEqual(use_case.cancelled, ["primeira"])
            SlowTranscriber.gate.set()
            await audio_task
            await debouncer.join()

        self.assertEqual(use_case.messages, ["primeira", "primeira\nquero cortar o cabelo"])
        sent = [text for waha in FakeWahaClient.instances for text in waha.sent]
        self.assertEqual(sent, ["resposta"])

    async def test_failed_transcription_releases_slot_and_other_messages_still_run(self) -> None:
        debouncer = ConversationDebouncer(0.05)
        use_case = FakeUseCase(["ok"])
        SlowTranscriber.fail = True
        SlowTranscriber.gate.set()
        p = self._patches(debouncer, use_case)
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            await webhook._process_whatsapp_message(_extracted("texto antes"))
            await webhook._process_whatsapp_message(_extracted_audio())
            await debouncer.join()

        self.assertEqual(use_case.messages, ["texto antes"])
        self.assertEqual(debouncer._states, {})

if __name__ == "__main__":
    unittest.main()
