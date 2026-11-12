import json
import logging
from typing import Any

from src.domain.conversation.models import TenantWhatsAppConfig
from src.infrastructure.adapters.db.connection import get_pool

logger = logging.getLogger(__name__)

_AI_ADDON_JOINS_SQL = """
        LEFT JOIN public.addons ad ON ad.code = 'ai_agent' AND ad.deleted_at IS NULL
        LEFT JOIN public.tenant_addons ta
            ON ta.tenant_id = t.id AND ta.addon_id = ad.id AND ta.deleted_at IS NULL
"""

_AI_ADDON_COLUMNS_SQL = """
        ta.settings AS ai_settings,
        COALESCE(
            ta.addon_status = 'active' AND ta.status = true
            AND (ta.addon_end IS NULL OR ta.addon_end > now()),
            false
        ) AS addon_active
"""


def _decode_settings(raw: Any) -> dict[str, Any]:
    """asyncpg devolve colunas json como str; aceita também dict já decodificado."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, (str, bytes)) or not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid JSON in tenant_addons.settings; ignoring it")
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _to_tenant_config(row: Any) -> TenantWhatsAppConfig:
    return TenantWhatsAppConfig(
        tenant_id=row["tenant_id"],
        instance_name=row["instance_name"] or "",
        business_name=row["business_name"],
        ai_enabled=bool(row["ai_enabled"]),
        ai_custom_prompt=row["ai_custom_prompt"] or "",
        ai_settings=_decode_settings(row["ai_settings"]),
        addon_active=bool(row["addon_active"]),
    )


async def get_tenant_config_by_instance(instance_name: str) -> TenantWhatsAppConfig | None:
    """Busca a configuração de WhatsApp/IA do tenant pelo nome da sessão/instância do WAHA."""
    if not instance_name:
        return None

    pool = await get_pool()
    query = f"""
        SELECT
            t.schema_name AS tenant_id,
            wi.instance_name AS instance_name,
            COALESCE(NULLIF(e.display_name, ''), t.name, t.schema_name) AS business_name,
            COALESCE(wi.ai_enabled, false) AS ai_enabled,
            COALESCE(wi.ai_custom_prompt, '') AS ai_custom_prompt,
            {_AI_ADDON_COLUMNS_SQL}
        FROM public.whatsapp_instances wi
        JOIN public.tenants t ON t.id = wi.tenant_id
        LEFT JOIN public.establishment e ON e.tenant_id = t.id AND e.status = true
        {_AI_ADDON_JOINS_SQL}
        WHERE (wi.instance_name = $1 OR t.schema_name = $1)
          AND wi.status = true
          AND t.status = true
        ORDER BY (wi.instance_name = $1) DESC, wi.updated_at DESC NULLS LAST
        LIMIT 1
    """
    try:
        row = await pool.fetchrow(query, instance_name)
        if not row:
            return None
        return _to_tenant_config(row)
    except Exception:
        logger.exception("Error querying tenant WhatsApp config for instance=%s", instance_name)
        return None


async def get_tenant_config_by_tenant_id(tenant_id: str) -> TenantWhatsAppConfig | None:
    """Busca a configuração do tenant pelo schema, sem exigir WhatsApp ativo (usado na simulação)."""
    if not tenant_id:
        return None

    pool = await get_pool()
    query = f"""
        SELECT
            t.schema_name AS tenant_id,
            COALESCE(wi.instance_name, '') AS instance_name,
            COALESCE(NULLIF(e.display_name, ''), t.name, t.schema_name) AS business_name,
            COALESCE(wi.ai_enabled, false) AS ai_enabled,
            COALESCE(wi.ai_custom_prompt, '') AS ai_custom_prompt,
            {_AI_ADDON_COLUMNS_SQL}
        FROM public.tenants t
        LEFT JOIN public.whatsapp_instances wi
            ON wi.tenant_id = t.id AND wi.status = true
        LEFT JOIN public.establishment e ON e.tenant_id = t.id AND e.status = true
        {_AI_ADDON_JOINS_SQL}
        WHERE t.schema_name = $1
          AND t.status = true
        ORDER BY wi.updated_at DESC NULLS LAST
        LIMIT 1
    """
    try:
        row = await pool.fetchrow(query, tenant_id)
        if not row:
            return None
        return _to_tenant_config(row)
    except Exception:
        logger.exception("Error querying tenant config for tenant=%s", tenant_id)
        return None


async def get_waha_credentials_from_db() -> tuple[str, str] | None:
    """Fallback para buscar URL e API Key do WAHA na tabela public.system_integration_keys."""
    pool = await get_pool()
    query = """
        SELECT url, api_key
        FROM public.system_integration_keys
        WHERE code = 'waha' AND status = true
        ORDER BY updated_at DESC NULLS LAST
        LIMIT 1
    """
    try:
        row = await pool.fetchrow(query)
        if row and row["url"] and row["api_key"]:
            return str(row["url"]), str(row["api_key"])
        return None
    except Exception:
        logger.exception("Error querying WAHA credentials from system_integration_keys")
        return None
