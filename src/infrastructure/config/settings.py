from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../../../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER")
    LLM_MODEL: str = os.getenv("LLM_MODEL")

    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_WHISPER_MODEL: str = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

    WAHA_API_URL: str = os.getenv("WAHA_API_URL", "")
    WAHA_API_KEY: str = os.getenv("WAHA_API_KEY", "")
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")

    SCHEDULA_BACK_URL: str = "http://localhost:8000"
    AGENT_SERVICE_SECRET: str = "dev_agent_secret_key"

    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Máximo de mensagens enviadas por resposta do agente (excedente é concatenado na última)
    REPLY_MAX_MESSAGES: int = 5

    # Segundos de espera (debounce) por novas mensagens do mesmo chat antes de rodar o agente.
    # Mensagens que chegam dentro da janela são agrupadas em um único ciclo. 0 desativa a espera.
    MESSAGE_DEBOUNCE_SECONDS: float = 10.0

    # Pool de conexões do checkpointer do LangGraph (psycopg)
    CHECKPOINTER_POOL_MIN_SIZE: int = 1
    CHECKPOINTER_POOL_MAX_SIZE: int = 10
    # Segundos ocioso antes de fechar uma conexão do pool
    CHECKPOINTER_POOL_MAX_IDLE: float = 300.0
    # Segundos de vida máxima de uma conexão (renovada depois disso)
    CHECKPOINTER_POOL_MAX_LIFETIME: float = 1800.0

    HOST: str = "0.0.0.0"
    PORT: int = 8001


settings = Settings()

