from abc import ABC, abstractmethod

from src.domain.conversation.models import Message


class ConversationRepository(ABC):
    @abstractmethod
    async def save_message(self, conversation_id: str, message: Message) -> None: ...

    @abstractmethod
    async def get_messages(self, conversation_id: str) -> list[Message]: ...
