from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


def build_checkpointer_pool(dsn: str, settings: Any) -> AsyncConnectionPool:
    """Cria (sem abrir) o pool de conexões usado pelo AsyncPostgresSaver.

    - autocommit/prepare_threshold/row_factory são exigidos pelo AsyncPostgresSaver;
    - `check` valida a conexão ao retirá-la do pool, descartando conexões mortas (ex.: SSL drop);
    - keepalives TCP detectam conexões silenciosamente derrubadas.
    """
    return AsyncConnectionPool(
        conninfo=dsn,
        min_size=settings.CHECKPOINTER_POOL_MIN_SIZE,
        max_size=settings.CHECKPOINTER_POOL_MAX_SIZE,
        open=False,
        max_idle=settings.CHECKPOINTER_POOL_MAX_IDLE,
        max_lifetime=settings.CHECKPOINTER_POOL_MAX_LIFETIME,
        check=AsyncConnectionPool.check_connection,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
    )
