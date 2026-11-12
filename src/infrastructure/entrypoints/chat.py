import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.domain.common.exceptions import AgentError
from src.domain.conversation.models import ConversationContext
from src.infrastructure.adapters.http.schedula_client import HttpSchedulaClient
from src.infrastructure.entrypoints.app import get_use_case

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    tenant_id: str
    user_id: str


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    context = ConversationContext(
        tenant_id=body.tenant_id,
        user_id=body.user_id,
        conversation_id=body.conversation_id,
    )

    client = HttpSchedulaClient(tenant_id=body.tenant_id, user_id=body.user_id)
    try:
        use_case = get_use_case()
        reply = await use_case.execute(context, body.message, schedula_client=client)
    except AgentError as exc:
        logger.exception("Agent error in /chat")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        await client.close()

    return ChatResponse(reply=reply, conversation_id=body.conversation_id)
