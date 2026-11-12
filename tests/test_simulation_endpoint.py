import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.domain.common.exceptions import AgentError
from src.domain.conversation.models import TenantWhatsAppConfig, WhatsAppConversationContext
from src.infrastructure.adapters.http.simulation_schedula_client import (
    SIMULATED_BOOKING_RESULT,
    SimulationSchedulaClient,
)
from src.infrastructure.entrypoints import simulation
from src.infrastructure.entrypoints.app import app

_SECRET = "segredo-de-teste"
_SESSION = "sessao_teste_01"
_URL = "/api/v1/whatsapp/simulate"


class FakeInnerSchedulaClient:
    instances: list["FakeInnerSchedulaClient"] = []

    def __init__(self, tenant_id: str, user_id: str) -> None:
        self.tenant_id = tenant_id
        self.booking_calls = 0
        self.closed = False
        FakeInnerSchedulaClient.instances.append(self)

    async def get_public_services(self) -> dict:
        return {"services_list": []}

    async def create_booking(self, **kwargs) -> dict:
        self.booking_calls += 1
        return {"id": "real-booking"}

    async def close(self) -> None:
        self.closed = True


class FakeUseCase:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.contexts: list[WhatsAppConversationContext] = []
        self.booking_results: list[dict] = []

    async def execute(self, context, user_message, schedula_client, cycle_guard=None) -> list[str]:
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        self.booking_results.append(
            await schedula_client.create_booking(
                service_id="svc-1",
                professional_id="pro-1",
                date="2026-11-03",
                time="14:00",
                customer_name="Joao",
                customer_phone="5511999999999",
            )
        )
        return ["Oi!", "Tudo certo."]


def _tenant_config(**overrides) -> TenantWhatsAppConfig:
    values = {
        "tenant_id": "goldman",
        "instance_name": "goldman",
        "business_name": "Barbearia Teste",
        "ai_enabled": False,
        "ai_custom_prompt": "Legado.",
        "ai_settings": {
            "custom_instructions": "Instrucoes salvas.",
            "quiz": {
                "tone": "descontraido",
                "address_form": "voce",
                "emoji_usage": "bastante",
                "response_length": "curtas",
                "greeting": "",
                "fallback_behavior": "aguardar_atendente",
                "fallback_contact": "",
            },
        },
        "addon_active": False,
    }
    values.update(overrides)
    return TenantWhatsAppConfig(**values)


def _body(**overrides) -> dict:
    body = {"tenant_id": "goldman", "session_id": _SESSION, "message": "oi", "customer_name": "Joao"}
    body.update(overrides)
    return body


class SimulateEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeInnerSchedulaClient.instances = []
        self.use_case = FakeUseCase()
        self.tenant_lookup = AsyncMock(return_value=_tenant_config())
        patches = [
            patch.object(simulation.settings, "AGENT_SERVICE_SECRET", _SECRET),
            patch.object(simulation, "get_tenant_config_by_tenant_id", self.tenant_lookup),
            patch.object(simulation, "HttpSchedulaClient", FakeInnerSchedulaClient),
            patch(
                "src.infrastructure.entrypoints.app.get_whatsapp_use_case",
                return_value=self.use_case,
            ),
        ]
        for active in patches:
            active.start()
            self.addCleanup(active.stop)
        self.client = TestClient(app)

    def _post(self, body: dict, token: str | None = _SECRET):
        headers = {} if token is None else {"X-Agent-Service-Token": token}
        return self.client.post(_URL, json=body, headers=headers)

    def test_bad_token_returns_401_and_does_not_run_agent(self) -> None:
        response = self._post(_body(), token="errado")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.use_case.contexts, [])

    def test_missing_token_returns_401(self) -> None:
        self.assertEqual(self._post(_body(), token=None).status_code, 401)

    def test_bad_session_returns_422(self) -> None:
        for bad in ("curta", "sessao com espaco!", "x" * 201):
            with self.subTest(session_id=bad):
                response = self._post(_body(session_id=bad))

                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.use_case.contexts, [])

    def test_stored_settings_are_used_without_override(self) -> None:
        response = self._post(_body())

        self.assertEqual(response.status_code, 200)
        context = self.use_case.contexts[0]
        assert context.style is not None
        self.assertEqual(context.style.tone, "descontraido")
        self.assertEqual(context.custom_prompt, "Instrucoes salvas.")

    def test_draft_settings_override_stored_settings(self) -> None:
        draft = {
            "quiz": {
                "tone": "formal",
                "address_form": "senhor",
                "emoji_usage": "nenhum",
                "response_length": "detalhadas",
                "greeting": "Bom dia",
                "fallback_behavior": "informar_contato",
                "fallback_contact": "(11) 3000-0000",
            },
            "custom_instructions": "Rascunho novo.",
        }

        response = self._post(_body(settings=draft))

        self.assertEqual(response.status_code, 200)
        context = self.use_case.contexts[0]
        assert context.style is not None
        self.assertEqual(context.style.tone, "formal")
        self.assertEqual(context.style.address_form, "senhor")
        self.assertEqual(context.style.greeting, "Bom dia")
        self.assertEqual(context.custom_prompt, "Rascunho novo.")

    def test_thread_id_uses_sim_prefix(self) -> None:
        self._post(_body())

        self.assertEqual(
            self.use_case.contexts[0].conversation_id,
            f"sim_goldman_{_SESSION}",
        )

    def test_simulation_runs_without_addon_or_ai_enabled(self) -> None:
        response = self._post(_body())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"replies": ["Oi!", "Tudo certo."]})

    def test_booking_is_never_forwarded_to_schedula_back(self) -> None:
        response = self._post(_body())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.use_case.booking_results, [SIMULATED_BOOKING_RESULT])
        inner = FakeInnerSchedulaClient.instances[0]
        self.assertEqual(inner.booking_calls, 0)
        self.assertTrue(inner.closed)

    def test_unknown_tenant_returns_404(self) -> None:
        self.tenant_lookup.return_value = None

        self.assertEqual(self._post(_body()).status_code, 404)

    def test_agent_error_returns_502(self) -> None:
        self.use_case.error = AgentError("falha interna")

        response = self._post(_body())

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "Não foi possível falar com o agente agora. Tente novamente.",
        )
        self.assertTrue(FakeInnerSchedulaClient.instances[0].closed)


class SimulationSchedulaClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_booking_is_faked_and_reads_are_delegated(self) -> None:
        inner = FakeInnerSchedulaClient("goldman", "5500000000000")
        client = SimulationSchedulaClient(inner)

        booking = await client.create_booking(
            service_id="svc",
            professional_id="pro",
            date="2026-11-03",
            time="14:00",
            customer_name="Joao",
            customer_phone="5511999999999",
        )
        services = await client.get_public_services()

        self.assertEqual(booking, SIMULATED_BOOKING_RESULT)
        self.assertEqual(services, {"services_list": []})
        self.assertEqual(inner.booking_calls, 0)


if __name__ == "__main__":
    unittest.main()
