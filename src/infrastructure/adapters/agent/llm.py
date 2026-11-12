from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from src.infrastructure.config.settings import settings


def init_llm() -> BaseChatModel:
    return init_chat_model(
        model=settings.LLM_MODEL,
        model_provider=settings.LLM_PROVIDER,
    )
