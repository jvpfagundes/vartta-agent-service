import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from src.infrastructure.adapters.db.checkpointer_pool import build_checkpointer_pool
from src.infrastructure.adapters.db.connection import close_pool, init_pool
from src.infrastructure.adapters.agent.graph import build_agent_graph
from src.infrastructure.adapters.agent.whatsapp_graph import build_whatsapp_agent_graph
from src.application.use_cases.run_agent import RunAgentUseCase
from src.application.use_cases.run_whatsapp_agent import RunWhatsAppAgentUseCase
from src.infrastructure.config.settings import settings
from src.infrastructure.entrypoints.webhook import get_debouncer
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_use_case: RunAgentUseCase | None = None
_whatsapp_use_case: RunWhatsAppAgentUseCase | None = None


def get_use_case() -> RunAgentUseCase:
    if _use_case is None:
        raise RuntimeError("RunAgentUseCase not initialized")
    return _use_case


def get_whatsapp_use_case() -> RunWhatsAppAgentUseCase:
    if _whatsapp_use_case is None:
        raise RuntimeError("RunWhatsAppAgentUseCase not initialized")
    return _whatsapp_use_case


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    global _use_case, _whatsapp_use_case

    logger.info("Starting agent_service…")
    await init_pool()
    logger.info("Database pool ready")

    checkpointer_pool = build_checkpointer_pool(settings.DATABASE_URL, settings)
    await checkpointer_pool.open(wait=True)
    try:
        checkpointer = AsyncPostgresSaver(checkpointer_pool)
        await checkpointer.setup()
        logger.info("LangGraph checkpointer ready (connection pool)")

        graph = build_agent_graph(checkpointer)
        _use_case = RunAgentUseCase(graph)

        whatsapp_graph = build_whatsapp_agent_graph(checkpointer)
        _whatsapp_use_case = RunWhatsAppAgentUseCase(
            whatsapp_graph,
            max_reply_messages=settings.REPLY_MAX_MESSAGES,
        )
        logger.info("Agent graphs (internal + WhatsApp) compiled — service is ready")

        yield

        # Cancela os ciclos pendentes (com rollback) enquanto o pool ainda está aberto
        logger.info("Shutting down agent_service…")
        await get_debouncer().shutdown()
    finally:
        await checkpointer_pool.close()

    await close_pool()
    logger.info("Database pool closed")


app = FastAPI(
    title="Vartta Agent Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "agent_service"}


from src.infrastructure.entrypoints.chat import router as chat_router
from src.infrastructure.entrypoints.simulation import router as simulation_router
from src.infrastructure.entrypoints.webhook import router as webhook_router

app.include_router(chat_router)
app.include_router(simulation_router)
app.include_router(webhook_router)
