import hmac
import logging
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from src.domain.common.exceptions import AgentError
from src.domain.conversation.agent_settings import parse_agent_style, resolve_custom_instructions
from src.domain.conversation.models import WhatsAppConversationContext
from src.infrastructure.adapters.db.whatsapp_repository import get_tenant_config_by_tenant_id
from src.infrastructure.adapters.http.schedula_client import HttpSchedulaClient
from src.infrastructure.adapters.http.simulation_schedula_client import SimulationSchedulaClient
from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/whatsapp", tags=["simulation"])

SIMULATION_CUSTOMER_PHONE = "5500000000000"
_SIMULATION_UNAVAILABLE_DETAIL = "Não foi possível falar com o agente agora. Tente novamente."


class SimulationQuiz(BaseModel):
    tone: Literal["amigavel", "descontraido", "formal", "profissional"]
    address_form: Literal["voce", "senhor"]
    emoji_usage: Literal["nenhum", "moderado", "bastante"]
    response_length: Literal["curtas", "detalhadas"]
    greeting: str = Field(default="", max_length=300)
    fallback_behavior: Literal["aguardar_atendente", "informar_contato"]
    fallback_contact: str = Field(default="", max_length=200)


class SimulationSettings(BaseModel):
    quiz: SimulationQuiz | None = None
    custom_instructions: str = Field(default="", max_length=4000)


class SimulateWhatsAppRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,200}$")
    message: str = Field(min_length=1, max_length=2000)
    customer_name: str = Field(default="Cliente Teste", max_length=120)
    settings: SimulationSettings | None = None


class SimulateWhatsAppResponse(BaseModel):
    replies: list[str]


def _token_is_valid(provided: str | None) -> bool:
    expected = settings.AGENT_SERVICE_SECRET
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode(), expected.encode())


@router.post("/simulate", response_model=SimulateWhatsAppResponse)
async def simulate_whatsapp(
    body: SimulateWhatsAppRequest,
    x_agent_service_token: str | None = Header(default=None),
) -> SimulateWhatsAppResponse:
    if not _token_is_valid(x_agent_service_token):
        raise HTTPException(status_code=401, detail="Unauthorized")

    tenant_config = await get_tenant_config_by_tenant_id(body.tenant_id)
    if tenant_config is None:
        raise HTTPException(status_code=404, detail="Tenant not found")

    ai_settings = (
        body.settings.model_dump() if body.settings is not None else tenant_config.ai_settings
    )
    context = WhatsAppConversationContext(
        tenant_id=tenant_config.tenant_id,
        instance_name=tenant_config.instance_name,
        chat_id=body.session_id,
        customer_phone=SIMULATION_CUSTOMER_PHONE,
        customer_name=body.customer_name,
        conversation_id=f"sim_{tenant_config.tenant_id}_{body.session_id}",
        business_name=tenant_config.business_name,
        custom_prompt=resolve_custom_instructions(ai_settings, tenant_config.ai_custom_prompt),
        style=parse_agent_style(ai_settings),
    )
    client = SimulationSchedulaClient(
        HttpSchedulaClient(
            tenant_id=tenant_config.tenant_id,
            user_id=SIMULATION_CUSTOMER_PHONE,
        )
    )
    try:
        from src.infrastructure.entrypoints.app import get_whatsapp_use_case

        use_case = get_whatsapp_use_case()
        replies = await use_case.execute(
            context=context,
            user_message=body.message,
            schedula_client=client,
        )
    except AgentError as exc:
        logger.exception("Agent error in WhatsApp simulation for tenant=%s", tenant_config.tenant_id)
        raise HTTPException(status_code=502, detail=_SIMULATION_UNAVAILABLE_DETAIL) from exc
    finally:
        await client.close()

    return SimulateWhatsAppResponse(replies=replies)
