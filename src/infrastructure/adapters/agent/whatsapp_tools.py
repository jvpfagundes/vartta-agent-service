import json
import logging
import re

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.application.services.customer_profile import (
    name_is_complete,
    name_word_count,
    sanitize_customer_name,
)
from src.infrastructure.adapters.http.schedula_client import HttpSchedulaClient

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}$")
_MAX_ADDITIONAL_SERVICES = 5


def _normalize_additional_service_ids(
    additional_service_ids: list[str] | str | None, service_id: str,
) -> list[str]:
    """Limpa os IDs dos serviços extras: remove vazios, duplicados e o próprio serviço principal."""
    if not additional_service_ids:
        return []
    if isinstance(additional_service_ids, str):
        additional_service_ids = [additional_service_ids]
    main_id = str(service_id or "").strip()
    cleaned: list[str] = []
    for raw in additional_service_ids:
        value = str(raw or "").strip()
        if not value or value == main_id or value in cleaned:
            continue
        cleaned.append(value)
    return cleaned[:_MAX_ADDITIONAL_SERVICES]


def _get_client(config: RunnableConfig) -> HttpSchedulaClient:
    configurable = config.get("configurable") or {}
    client = configurable.get("schedula_client") or config.get("schedula_client")
    if client is None:
        tenant_id = configurable.get("tenant_id") or config.get("tenant_id")
        user_id = configurable.get("user_id") or config.get("user_id") or "whatsapp_bot"
        if not tenant_id:
            raise ValueError("Tenant context missing for tool execution")
        client = HttpSchedulaClient(tenant_id=tenant_id, user_id=user_id)
    return client


def _professional_display_name(p: dict) -> str:
    """Nome da agenda do profissional; cai no nome do profissional se a agenda não tiver nome."""
    schedule = p.get("schedule") or p.get("agenda") or {}
    candidates = [
        p.get("schedule_name"),
        p.get("agenda_name"),
        schedule.get("name") if isinstance(schedule, dict) else None,
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return f"{p.get('first_name') or ''} {p.get('last_name') or ''}".strip()


def _choose_booking_name(registered_name: str | None, llm_name: str) -> str:
    """Nome enviado ao agendamento: o cadastrado, se completo e não menos completo que o do LLM."""
    registered = sanitize_customer_name(registered_name)
    if registered and name_is_complete(registered) and (
        name_word_count(registered) >= name_word_count(llm_name)
    ):
        return registered
    return llm_name


@tool
async def get_public_services(config: RunnableConfig) -> str:
    """Lista todos os serviços oferecidos pelo estabelecimento (ex: corte, barba, sobrancelha, manicure), com ID, nome, descrição, duração e preço. Chame sempre que o cliente mencionar qualquer serviço ou intenção de agendamento.

    Returns:
        Lista de serviços disponíveis em JSON.
    """
    try:
        client = _get_client(config)
        result = await client.get_public_services()
        services = result.get("services_list", []) if isinstance(result, dict) else []
        formatted = []
        for s in services:
            dur_raw = s.get("duration") or 0
            # No banco, duration costuma ser salvo em segundos (ex: 1800 = 30 min)
            dur_minutes = round(dur_raw / 60) if dur_raw >= 180 else dur_raw
            formatted.append(
                {
                    "id": s.get("id"),
                    "name": s.get("name"),
                    "description": s.get("description") or "",
                    "duration_minutes": dur_minutes,
                    "price": s.get("price"),
                }
            )
        return json.dumps({"services": formatted}, ensure_ascii=False)
    except Exception:
        logger.exception("WhatsApp tool get_public_services failed")
        return json.dumps(
            {"error": "Não foi possível consultar a lista de serviços no momento."},
            ensure_ascii=False,
        )


@tool
async def get_public_professionals(config: RunnableConfig) -> str:
    """Lista os profissionais que atendem no estabelecimento.

    Returns:
        Lista de profissionais com ID e nome.
    """
    try:
        client = _get_client(config)
        result = await client.get_public_professionals()
        professionals = result.get("professionals_list", []) if isinstance(result, dict) else []
        formatted = [
            {
                "id": p.get("id"),
                "name": _professional_display_name(p),
            }
            for p in professionals
        ]
        return json.dumps({"professionals": formatted}, ensure_ascii=False)
    except Exception:
        logger.exception("WhatsApp tool get_public_professionals failed")
        return json.dumps(
            {"error": "Não foi possível consultar os profissionais no momento."},
            ensure_ascii=False,
        )


@tool
async def get_service_professionals(service_id: str, config: RunnableConfig) -> str:
    """Lista os profissionais habilitados para realizar um serviço específico.

    Args:
        service_id: ID do serviço desejado (obtido via get_public_services).

    Returns:
        Lista de profissionais que realizam o serviço.
    """
    try:
        client = _get_client(config)
        result = await client.get_service_professionals(service_id)
        professionals = result.get("professionals_list", []) if isinstance(result, dict) else []
        formatted = [
            {
                "id": p.get("id"),
                "name": _professional_display_name(p),
            }
            for p in professionals
        ]
        return json.dumps({"professionals": formatted}, ensure_ascii=False)
    except Exception:
        logger.exception("WhatsApp tool get_service_professionals failed")
        return json.dumps(
            {"error": "Não foi possível consultar os profissionais para este serviço."},
            ensure_ascii=False,
        )


@tool
async def list_available_slots(
    service_id: str,
    professional_id: str,
    date: str,
    config: RunnableConfig,
    additional_service_ids: list[str] | None = None,
) -> str:
    """Lista os horários livres de um profissional para realizar um serviço em uma data específica.

    Args:
        service_id: ID do serviço desejado (o primeiro serviço, se forem vários).
        professional_id: ID do profissional escolhido.
        date: Data desejada no formato YYYY-MM-DD.
        additional_service_ids: IDs dos demais serviços do MESMO cliente com o MESMO profissional; o primeiro serviço vai em service_id. Os horários retornados comportam a duração somada de todos os serviços.

    Returns:
        Lista de horários disponíveis (HH:MM) na data informada.
    """
    if not _DATE_RE.match(str(date or "").strip()):
        return json.dumps(
            {"error": "Formato de data inválido. Use YYYY-MM-DD."},
            ensure_ascii=False,
        )
    try:
        client = _get_client(config)
        extra_ids = _normalize_additional_service_ids(additional_service_ids, service_id)
        if extra_ids:
            result = await client.list_available_slots(
                service_id, professional_id, date.strip(), additional_service_ids=extra_ids,
            )
        else:
            result = await client.list_available_slots(service_id, professional_id, date.strip())
        slots = result.get("slots", []) if isinstance(result, dict) else result
        return json.dumps({"date": date.strip(), "available_slots": slots}, ensure_ascii=False)
    except Exception:
        logger.exception("WhatsApp tool list_available_slots failed")
        return json.dumps(
            {"error": "Não foi possível consultar os horários disponíveis para essa data."},
            ensure_ascii=False,
        )


@tool
async def schedule_customer_appointment(
    service_id: str,
    professional_id: str,
    date: str,
    time: str,
    customer_name: str,
    config: RunnableConfig,
    additional_service_ids: list[str] | None = None,
) -> str:
    """Agenda um atendimento para o cliente atual do WhatsApp após confirmação explícita de serviço, profissional, data e horário. Para vários serviços da mesma pessoa com o mesmo profissional, chame UMA única vez usando additional_service_ids.

    Args:
        service_id: ID do serviço a ser agendado (o primeiro serviço, se forem vários).
        professional_id: ID do profissional que realizará o atendimento.
        date: Data do atendimento no formato YYYY-MM-DD.
        time: Horário do atendimento no formato HH:MM.
        customer_name: Nome do cliente que será atendido.
        additional_service_ids: IDs dos demais serviços do MESMO cliente com o MESMO profissional; o primeiro serviço vai em service_id. Os serviços são feitos em sequência em um único agendamento.

    Returns:
        Confirmação do agendamento.
    """
    date_clean = str(date or "").strip()
    time_clean = str(time or "").strip()
    name_clean = re.sub(r"[<>]", "", str(customer_name or "").strip())[:80]

    if not _DATE_RE.match(date_clean):
        return json.dumps({"error": "Data inválida. Use o formato YYYY-MM-DD."}, ensure_ascii=False)
    if not _TIME_RE.match(time_clean):
        return json.dumps({"error": "Horário inválido. Use o formato HH:MM."}, ensure_ascii=False)
    if not name_clean:
        return json.dumps({"error": "O nome do cliente é obrigatório para agendar."}, ensure_ascii=False)

    configurable = config.get("configurable") or {}
    # Não rebaixa um nome completo já cadastrado: o backend sobrescreve o cadastro quando o nome
    # do agendamento difere do registrado.
    booking_name = _choose_booking_name(configurable.get("customer_registered_name"), name_clean)
    # Usa sempre o telefone autenticado da sessão do WhatsApp para impedir falsificação de identidade
    customer_phone = (
        configurable.get("customer_phone")
        or config.get("customer_phone")
        or ""
    )
    customer_phone = re.sub(r"\D", "", str(customer_phone))
    # Se começar com 55 (DDI Brasil) e tiver 12 ou 13 dígitos, mantém formato limpo
    if customer_phone.startswith("55") and len(customer_phone) in (12, 13):
        customer_phone_local = customer_phone[2:]
    else:
        customer_phone_local = customer_phone

    # A partir daqui o ciclo não pode mais ser cancelado por uma mensagem nova: interromper
    # durante/depois da criação do agendamento causaria agendamento duplicado (o novo ciclo
    # repetiria a tool) ou estado inconsistente entre o checkpoint e o backend.
    guard = configurable.get("cycle_guard")
    if guard:
        guard.commit()

    try:
        client = _get_client(config)
        booking_kwargs: dict = {}
        extra_ids = _normalize_additional_service_ids(additional_service_ids, service_id)
        if extra_ids:
            booking_kwargs["additional_service_ids"] = extra_ids
        result = await client.create_booking(
            service_id=service_id,
            professional_id=professional_id,
            date=date_clean,
            time=time_clean,
            customer_name=booking_name,
            customer_phone=customer_phone_local or customer_phone,
            **booking_kwargs,
        )
        return json.dumps(
            {
                "status": "scheduled",
                "details": result,
            },
            ensure_ascii=False,
        )
    except Exception:
        logger.exception("WhatsApp tool schedule_customer_appointment failed")
        return json.dumps(
            {
                "error": (
                    "Não foi possível concluir o agendamento para esse horário. "
                    "Verifique se o horário ainda está disponível."
                )
            },
            ensure_ascii=False,
        )


@tool
async def update_customer_name(full_name: str, config: RunnableConfig) -> str:
    """Atualiza o cadastro do cliente atual com o nome completo (nome e sobrenome) informado por ele. Chame assim que o cliente disser o sobrenome ou o nome completo.

    Args:
        full_name: Nome completo do cliente, com nome e sobrenome (ex: "João Silva").

    Returns:
        Confirmação da atualização do cadastro.
    """
    name_clean = sanitize_customer_name(full_name)
    if not name_is_complete(name_clean):
        return json.dumps(
            {"error": "Informe o nome completo do cliente, com nome e sobrenome."},
            ensure_ascii=False,
        )

    configurable = config.get("configurable") or {}
    # O cadastro alterado é sempre o resolvido pelo servidor a partir do telefone autenticado.
    customer_id = configurable.get("customer_id")
    if not customer_id:
        return json.dumps(
            {"error": "Não foi possível localizar o cadastro do cliente para atualizar o nome."},
            ensure_ascii=False,
        )

    try:
        client = _get_client(config)
        await client.update_customer_name(str(customer_id), name_clean)
        return json.dumps({"status": "updated", "name": name_clean}, ensure_ascii=False)
    except Exception:
        logger.exception("WhatsApp tool update_customer_name failed")
        return json.dumps(
            {"error": "Não foi possível atualizar o cadastro do cliente no momento."},
            ensure_ascii=False,
        )


@tool
async def register_customer(full_name: str, config: RunnableConfig) -> str:
    """Cadastra o cliente atual com o nome completo (nome e sobrenome) informado por ele. Use somente quando o cliente ainda não estiver cadastrado, assim que ele disser o nome e o sobrenome.

    Args:
        full_name: Nome completo do cliente, com nome e sobrenome (ex: "Maria Souza").

    Returns:
        Confirmação do cadastro.
    """
    name_clean = sanitize_customer_name(full_name)
    if not name_is_complete(name_clean):
        return json.dumps(
            {"error": "Informe o nome completo do cliente, com nome e sobrenome."},
            ensure_ascii=False,
        )

    configurable = config.get("configurable") or {}
    # O telefone vem sempre da sessão autenticada do WhatsApp, nunca do LLM.
    phone = re.sub(r"\D", "", str(configurable.get("customer_phone") or ""))
    if not phone:
        return json.dumps(
            {"error": "Não foi possível identificar o telefone do cliente para o cadastro."},
            ensure_ascii=False,
        )

    try:
        client = _get_client(config)
        # Evita duplicar: o cadastro pode ter sido criado depois do início do ciclo.
        existing = await client.find_customer_by_phone(phone)
        if existing:
            existing_name = str(
                existing.get("name")
                or f"{existing.get('first_name') or ''} {existing.get('last_name') or ''}"
            ).strip()
            if name_is_complete(existing_name):
                return json.dumps(
                    {"status": "already_registered", "name": existing_name},
                    ensure_ascii=False,
                )
            existing_id = existing.get("id")
            if not existing_id:
                return json.dumps(
                    {"error": "Não foi possível atualizar o cadastro do cliente no momento."},
                    ensure_ascii=False,
                )
            await client.update_customer_name(str(existing_id), name_clean)
            return json.dumps({"status": "updated", "name": name_clean}, ensure_ascii=False)
        await client.create_customer(name=name_clean, phone=phone)
        return json.dumps({"status": "registered", "name": name_clean}, ensure_ascii=False)
    except Exception:
        logger.exception("WhatsApp tool register_customer failed")
        return json.dumps(
            {"error": "Não foi possível cadastrar o cliente no momento."},
            ensure_ascii=False,
        )


WHATSAPP_CUSTOMER_TOOLS = [
    get_public_services,
    get_public_professionals,
    get_service_professionals,
    list_available_slots,
    schedule_customer_appointment,
    update_customer_name,
    register_customer,
]
