import json
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.domain.common.exceptions import ToolExecutionError
from src.infrastructure.adapters.http.schedula_client import HttpSchedulaClient

logger = logging.getLogger(__name__)


def _get_client(config: RunnableConfig) -> HttpSchedulaClient:
    client = config.get('configurable').get("schedula_client")
    if client is None:
        tenant_id = config.get('configurable').get("tenant_id")
        user_id = config.get('configurable').get("user_id")
        if not tenant_id or not user_id:
            raise ToolExecutionError(
                "schedula_client",
                "Client not found in config and tenant_id/user_id are missing",
            )
        logger.warning(
            "schedula_client not found in config, creating a new instance "
            "(tenant_id=%s, user_id=%s)",
            tenant_id,
            user_id,
        )
        client = HttpSchedulaClient(tenant_id=tenant_id, user_id=user_id)
    return client


# ── Booking ───────────────────────────────────────────────────────────────────

@tool
async def list_available_slots(
    service_id: str, user_id: str, date: str, config: RunnableConfig,
) -> str:
    """Lista os horários disponíveis de um profissional para um serviço em uma data específica.

    Args:
        service_id: ID do serviço desejado.
        user_id: ID do profissional (prestador de serviço).
        date: Data no formato YYYY-MM-DD.

    Returns:
        Lista de horários disponíveis para agendamento.
    """
    try:
        client = _get_client(config)
        result = await client.list_available_slots(service_id, user_id, date)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("list_available_slots failed")
        raise ToolExecutionError("list_available_slots", str(exc)) from exc


@tool
async def schedule_appointment(
    service_id: str,
    professional_id: str,
    date: str,
    time: str,
    customer_name: str,
    customer_phone: str,
    config: RunnableConfig,
) -> str:
    """Agenda um atendimento para um cliente com um profissional específico.

    Args:
        service_id: ID do serviço a ser agendado.
        professional_id: ID do profissional que realizará o atendimento.
        date: Data do atendimento no formato YYYY-MM-DD.
        time: Horário do atendimento no formato HH:MM.
        customer_name: Nome completo do cliente.
        customer_phone: Telefone do cliente com DDD (ex: 11999998888).

    Returns:
        Confirmação do agendamento com detalhes.
    """
    try:
        client = _get_client(config)
        result = await client.create_booking(
            service_id=service_id,
            professional_id=professional_id,
            date=date,
            time=time,
            customer_name=customer_name,
            customer_phone=customer_phone,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("schedule_appointment failed")
        raise ToolExecutionError("schedule_appointment", str(exc)) from exc


# ── Schedule / Agenda ─────────────────────────────────────────────────────────

@tool
async def get_events(
    date_start: str, date_end: str, config: RunnableConfig,
) -> str:
    """Consulta os eventos e agendamentos em um período de datas.

    Args:
        date_start: Data de início no formato YYYY-MM-DD.
        date_end: Data de fim no formato YYYY-MM-DD.

    Returns:
        Lista de eventos agendados no período.
    """
    try:
        client = _get_client(config)
        result = await client.get_events(date_start, date_end)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_events failed")
        raise ToolExecutionError("get_events", str(exc)) from exc


@tool
async def get_events_count(
    date_start: str, date_end: str, config: RunnableConfig,
) -> str:
    """Conta o total de eventos/agendamentos em um período.

    Args:
        date_start: Data de início no formato YYYY-MM-DD.
        date_end: Data de fim no formato YYYY-MM-DD.

    Returns:
        Total de eventos no período.
    """
    try:
        client = _get_client(config)
        result = await client.get_events_count(date_start, date_end)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_events_count failed")
        raise ToolExecutionError("get_events_count", str(exc)) from exc


@tool
async def get_dashboard_total(
    date_start: str, date_end: str, config: RunnableConfig,
    user_id: str | None = None,
) -> str:
    """Retorna os totais do dashboard de agenda: atendimentos, faturamento bruto e ticket médio.

    Args:
        date_start: Data de início no formato YYYY-MM-DD.
        date_end: Data de fim no formato YYYY-MM-DD.
        user_id: ID do profissional para filtrar (opcional).

    Returns:
        Totais de atendimentos e faturamento do período.
    """
    try:
        client = _get_client(config)
        result = await client.get_dashboard_total(date_start, date_end, user_id)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_dashboard_total failed")
        raise ToolExecutionError("get_dashboard_total", str(exc)) from exc


@tool
async def get_dashboard_services(
    date_start: str, date_end: str, config: RunnableConfig,
    user_id: str | None = None,
) -> str:
    """Retorna o ranking de serviços mais realizados no período.

    Args:
        date_start: Data de início no formato YYYY-MM-DD.
        date_end: Data de fim no formato YYYY-MM-DD.
        user_id: ID do profissional para filtrar (opcional).

    Returns:
        Lista de serviços ordenados por quantidade de atendimentos.
    """
    try:
        client = _get_client(config)
        result = await client.get_dashboard_services(date_start, date_end, user_id)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_dashboard_services failed")
        raise ToolExecutionError("get_dashboard_services", str(exc)) from exc


@tool
async def get_dashboard_customers(
    date_start: str, date_end: str, config: RunnableConfig,
    user_id: str | None = None,
    sort_by: str = "visits",
) -> str:
    """Retorna o ranking de clientes do período (mais frequentes ou maior gasto).

    Args:
        date_start: Data de início no formato YYYY-MM-DD.
        date_end: Data de fim no formato YYYY-MM-DD.
        user_id: ID do profissional para filtrar (opcional).
        sort_by: Critério de ordenação — "visits" (visitas) ou "revenue" (receita).

    Returns:
        Lista de clientes com visitas e valor gasto.
    """
    try:
        client = _get_client(config)
        result = await client.get_dashboard_customers(date_start, date_end, user_id, sort_by)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_dashboard_customers failed")
        raise ToolExecutionError("get_dashboard_customers", str(exc)) from exc


# ── Services ──────────────────────────────────────────────────────────────────

@tool
async def get_services(config: RunnableConfig) -> str:
    """Lista todos os serviços oferecidos pelo estabelecimento, incluindo nome, duração e preço.

    Returns:
        Lista completa de serviços disponíveis.
    """
    try:
        client = _get_client(config)
        result = await client.get_services()
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_services failed")
        raise ToolExecutionError("get_services", str(exc)) from exc


# ── Customers / Clientes ──────────────────────────────────────────────────────

@tool
async def search_customers(
    query: str, config: RunnableConfig,
) -> str:
    """Busca clientes pelo nome, telefone ou e-mail.

    Args:
        query: Texto para pesquisa (nome, telefone ou e-mail).

    Returns:
        Lista de clientes que correspondem à pesquisa.
    """
    try:
        client = _get_client(config)
        result = await client.search_customers(query)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("search_customers failed")
        raise ToolExecutionError("search_customers", str(exc)) from exc


@tool
async def get_customers(
    config: RunnableConfig,
    limit: int = 10,
    offset: int = 0,
    term: str | None = None,
) -> str:
    """Lista os clientes cadastrados com paginação e busca opcional.

    Args:
        limit: Quantidade máxima de registros a retornar (padrão: 10).
        offset: Quantidade de registros a pular para paginação (padrão: 0).
        term: Texto opcional para filtrar clientes.

    Returns:
        Lista paginada de clientes.
    """
    try:
        client = _get_client(config)
        result = await client.get_customers(limit=limit, offset=offset, term=term)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_customers failed")
        raise ToolExecutionError("get_customers", str(exc)) from exc


@tool
async def get_customer_history(
    customer_id: str, config: RunnableConfig,
) -> str:
    """Consulta o histórico de atendimentos de um cliente específico.

    Args:
        customer_id: ID do cliente.

    Returns:
        Histórico completo de agendamentos e atendimentos do cliente.
    """
    try:
        client = _get_client(config)
        result = await client.get_customer_history(customer_id)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_customer_history failed")
        raise ToolExecutionError("get_customer_history", str(exc)) from exc


@tool
async def get_missing_customers(
    config: RunnableConfig,
    last_schedule: str | None = None,
    minimum_scheduling: int = 0,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """Lista clientes que estão sumidos / sem agendamento há algum tempo.

    Args:
        last_schedule: Data limite do último agendamento no formato YYYY-MM-DD.
                       Retorna clientes sem agendamento desde antes dessa data.
        minimum_scheduling: Quantidade mínima de agendamentos anteriores (padrão: 0).
        limit: Quantidade máxima de registros a retornar (padrão: 10).
        offset: Quantidade de registros a pular para paginação (padrão: 0).

    Returns:
        Lista de clientes ausentes com informações de contato.
    """
    try:
        client = _get_client(config)
        result = await client.get_missing_customers(
            last_schedule=last_schedule,
            minimum_scheduling=minimum_scheduling,
            limit=limit,
            offset=offset,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_missing_customers failed")
        raise ToolExecutionError("get_missing_customers", str(exc)) from exc


# ── Financial ─────────────────────────────────────────────────────────────────

@tool
async def list_bill_categories(config: RunnableConfig) -> str:
    """Lista todas as categorias de contas financeiras disponíveis (ex: aluguel, produtos, salário).

    Returns:
        Lista de categorias para classificar contas.
    """
    try:
        client = _get_client(config)
        result = await client.list_bill_categories()
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("list_bill_categories failed")
        raise ToolExecutionError("list_bill_categories", str(exc)) from exc


@tool
async def create_bill(
    type: str,
    bill_date: str,
    description: str,
    value: float,
    config: RunnableConfig,
    category: str | None = None,
    is_paid: bool = False,
) -> str:
    """Cria uma nova conta a pagar ou a receber no sistema financeiro.

    Args:
        type: Tipo da conta — "payable" (a pagar) ou "receivable" (a receber).
        bill_date: Data de vencimento no formato YYYY-MM-DD.
        description: Descrição da conta (ex: "Aluguel do salão - Junho").
        value: Valor da conta em reais (ex: 1500.00).
        category: ID da categoria da conta (opcional, use list_bill_categories para ver opções).
        is_paid: Se a conta já está paga (padrão: false).

    Returns:
        Confirmação da criação da conta com detalhes.
    """
    try:
        client = _get_client(config)
        result = await client.create_bill(
            type=type,
            bill_date=bill_date,
            description=description,
            value=value,
            category=category,
            is_paid=is_paid,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("create_bill failed")
        raise ToolExecutionError("create_bill", str(exc)) from exc


@tool
async def get_bills(
    config: RunnableConfig,
    date_ini: str | None = None,
    date_end: str | None = None,
    limit: int = 50,
    offset: int = 0,
    term: str | None = None,
) -> str:
    """Lista contas a pagar e a receber com filtros de período e busca.

    Args:
        date_ini: Data de início no formato YYYY-MM-DD (opcional).
        date_end: Data de fim no formato YYYY-MM-DD (opcional).
        limit: Quantidade máxima de registros (padrão: 50).
        offset: Registros a pular para paginação (padrão: 0).
        term: Texto para buscar na descrição das contas (opcional).

    Returns:
        Lista de contas com status de pagamento e valores.
    """
    try:
        client = _get_client(config)
        result = await client.get_bills(
            date_ini=date_ini,
            date_end=date_end,
            limit=limit,
            offset=offset,
            term=term,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_bills failed")
        raise ToolExecutionError("get_bills", str(exc)) from exc


@tool
async def get_financial_summary(
    date_ini: str, date_end: str, config: RunnableConfig,
) -> str:
    """Consulta o resumo financeiro do período com cards de receitas, despesas e saldo.

    Args:
        date_ini: Data de início no formato YYYY-MM-DD.
        date_end: Data de fim no formato YYYY-MM-DD.

    Returns:
        Cards de resumo financeiro com totais de receitas, despesas e saldo.
    """
    try:
        client = _get_client(config)
        result = await client.get_financial_cards(date_ini, date_end)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_financial_summary failed")
        raise ToolExecutionError("get_financial_summary", str(exc)) from exc


@tool
async def get_financial_statement(
    date_ini: str, date_end: str, config: RunnableConfig,
    limit: int = 50,
    offset: int = 0,
    term: str | None = None,
) -> str:
    """Retorna o extrato financeiro detalhado (tabela de movimentações) do período.

    Args:
        date_ini: Data de início no formato YYYY-MM-DD.
        date_end: Data de fim no formato YYYY-MM-DD.
        limit: Quantidade máxima de registros (padrão: 50).
        offset: Registros a pular para paginação (padrão: 0).
        term: Texto para buscar nas movimentações (opcional).

    Returns:
        Lista detalhada de entradas e saídas financeiras do período.
    """
    try:
        client = _get_client(config)
        result = await client.get_financial_statement(
            date_ini=date_ini,
            date_end=date_end,
            limit=limit,
            offset=offset,
            term=term,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_financial_statement failed")
        raise ToolExecutionError("get_financial_statement", str(exc)) from exc


@tool
async def get_expenses_by_category(
    date_ini: str, date_end: str, config: RunnableConfig,
) -> str:
    """Retorna as despesas agrupadas por categoria no período.

    Args:
        date_ini: Data de início no formato YYYY-MM-DD.
        date_end: Data de fim no formato YYYY-MM-DD.

    Returns:
        Despesas totalizadas por categoria (ex: aluguel, produtos, salários).
    """
    try:
        client = _get_client(config)
        result = await client.get_expenses_by_category(date_ini, date_end)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_expenses_by_category failed")
        raise ToolExecutionError("get_expenses_by_category", str(exc)) from exc


@tool
async def get_payment_method_graph(
    date_ini: str, date_end: str, config: RunnableConfig,
) -> str:
    """Retorna a distribuição de receitas por forma de pagamento (dinheiro, cartão, PIX, etc).

    Args:
        date_ini: Data de início no formato YYYY-MM-DD.
        date_end: Data de fim no formato YYYY-MM-DD.

    Returns:
        Receitas totalizadas por método de pagamento.
    """
    try:
        client = _get_client(config)
        result = await client.get_payment_method_graph(date_ini, date_end)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_payment_method_graph failed")
        raise ToolExecutionError("get_payment_method_graph", str(exc)) from exc


# ── Users / Professionals ─────────────────────────────────────────────────────

@tool
async def get_professionals(config: RunnableConfig) -> str:
    """Lista os profissionais cadastrados no estabelecimento.

    Returns:
        Lista de profissionais com nome e ID.
    """
    try:
        client = _get_client(config)
        result = await client.get_professionals()
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("get_professionals failed")
        raise ToolExecutionError("get_professionals", str(exc)) from exc


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_TOOLS = [
    # Booking
    list_available_slots,
    schedule_appointment,
    # Schedule
    get_events,
    get_events_count,
    get_dashboard_total,
    get_dashboard_services,
    get_dashboard_customers,
    # Services
    get_services,
    # Customers
    search_customers,
    get_customers,
    get_customer_history,
    get_missing_customers,
    # Financial
    list_bill_categories,
    create_bill,
    get_bills,
    get_financial_summary,
    get_financial_statement,
    get_expenses_by_category,
    get_payment_method_graph,
    # Professionals
    get_professionals,
]
