from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.domain.conversation.agent_settings import AgentStyle


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class ConversationContext:
    tenant_id: str
    user_id: str
    conversation_id: str


@dataclass(frozen=True, slots=True)
class TenantWhatsAppConfig:
    tenant_id: str
    instance_name: str
    business_name: str
    ai_enabled: bool
    ai_custom_prompt: str = ""
    ai_settings: dict[str, Any] = field(default_factory=dict)
    addon_active: bool = False


@dataclass(frozen=True, slots=True)
class WhatsAppConversationContext:
    tenant_id: str
    instance_name: str
    chat_id: str
    customer_phone: str
    customer_name: str
    conversation_id: str
    business_name: str
    custom_prompt: str = ""
    style: AgentStyle | None = None
