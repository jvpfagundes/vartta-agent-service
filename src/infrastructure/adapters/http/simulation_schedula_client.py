from src.application.ports.schedula_client import SchedulaBackClient
from src.infrastructure.adapters.http.schedula_client import HttpSchedulaClient

SIMULATED_BOOKING_RESULT = {
    "simulated": True,
    "message": "Agendamento simulado (nenhum horário foi reservado).",
}

SIMULATED_CUSTOMER_ID = "simulated-customer"
SIMULATED_CUSTOMER_UPDATE_RESULT = {
    "simulated": True,
    "message": "Cadastro simulado (nenhum cliente foi alterado).",
}


class SimulationSchedulaClient(SchedulaBackClient):
    """Repassa as leituras ao schedula_back e simula a criação de agendamentos."""

    def __init__(self, inner: HttpSchedulaClient) -> None:
        self._inner = inner

    async def close(self) -> None:
        await self._inner.close()

    # ── Booking ───────────────────────────────────────────────────────────────

    async def get_public_services(self) -> dict:
        return await self._inner.get_public_services()

    async def get_public_professionals(self) -> dict:
        return await self._inner.get_public_professionals()

    async def get_service_professionals(self, service_id: str) -> dict:
        return await self._inner.get_service_professionals(service_id)

    async def list_available_slots(
        self,
        service_id: str,
        user_id: str,
        date: str,
        additional_service_ids: list[str] | None = None,
    ) -> dict:
        return await self._inner.list_available_slots(
            service_id, user_id, date, additional_service_ids=additional_service_ids,
        )

    async def create_booking(
        self,
        service_id: str,
        professional_id: str,
        date: str,
        time: str,
        customer_name: str,
        customer_phone: str,
        additional_service_ids: list[str] | None = None,
    ) -> dict:
        return dict(SIMULATED_BOOKING_RESULT)

    # ── Schedule ──────────────────────────────────────────────────────────────

    async def get_events(self, date_start: str, date_end: str) -> dict:
        return await self._inner.get_events(date_start, date_end)

    async def get_events_count(self, date_start: str, date_end: str) -> dict:
        return await self._inner.get_events_count(date_start, date_end)

    async def get_dashboard_total(
        self, date_start: str, date_end: str, user_id: str | None = None,
    ) -> dict:
        return await self._inner.get_dashboard_total(date_start, date_end, user_id)

    async def get_dashboard_services(
        self, date_start: str, date_end: str, user_id: str | None = None,
    ) -> dict:
        return await self._inner.get_dashboard_services(date_start, date_end, user_id)

    async def get_dashboard_customers(
        self,
        date_start: str,
        date_end: str,
        user_id: str | None = None,
        sort_by: str = "visits",
    ) -> dict:
        return await self._inner.get_dashboard_customers(date_start, date_end, user_id, sort_by)

    # ── Services ──────────────────────────────────────────────────────────────

    async def get_services(self) -> dict:
        return await self._inner.get_services()

    # ── Customers ─────────────────────────────────────────────────────────────

    async def find_customer_by_phone(self, phone: str) -> dict | None:
        return await self._inner.find_customer_by_phone(phone)

    async def create_customer(self, name: str, phone: str) -> dict:
        return {"id": SIMULATED_CUSTOMER_ID, "name": name, "phone_number": phone}

    async def update_customer_name(self, customer_id: str, name: str) -> dict:
        return {**SIMULATED_CUSTOMER_UPDATE_RESULT, "costumer_id": customer_id}

    async def search_customers(self, query: str) -> dict:
        return await self._inner.search_customers(query)

    async def get_customers(
        self,
        limit: int = 10,
        offset: int = 0,
        term: str | None = None,
    ) -> dict:
        return await self._inner.get_customers(limit, offset, term)

    async def get_customer_history(self, customer_id: str) -> dict:
        return await self._inner.get_customer_history(customer_id)

    async def get_missing_customers(
        self,
        last_schedule: str | None = None,
        minimum_scheduling: int = 0,
        limit: int = 10,
        offset: int = 0,
    ) -> dict:
        return await self._inner.get_missing_customers(
            last_schedule, minimum_scheduling, limit, offset
        )

    # ── Financial ─────────────────────────────────────────────────────────────

    async def list_bill_categories(self) -> dict:
        return await self._inner.list_bill_categories()

    async def create_bill(
        self,
        type: str,
        bill_date: str,
        description: str,
        value: float,
        category: str | None = None,
        is_paid: bool = False,
    ) -> dict:
        return await self._inner.create_bill(
            type, bill_date, description, value, category, is_paid
        )

    async def get_bills(
        self,
        date_ini: str | None = None,
        date_end: str | None = None,
        limit: int = 50,
        offset: int = 0,
        term: str | None = None,
    ) -> dict:
        return await self._inner.get_bills(date_ini, date_end, limit, offset, term)

    async def get_financial_cards(self, date_ini: str, date_end: str) -> dict:
        return await self._inner.get_financial_cards(date_ini, date_end)

    async def get_financial_statement(
        self,
        date_ini: str,
        date_end: str,
        limit: int = 50,
        offset: int = 0,
        term: str | None = None,
    ) -> dict:
        return await self._inner.get_financial_statement(date_ini, date_end, limit, offset, term)

    async def get_expenses_by_category(
        self, date_ini: str, date_end: str,
    ) -> dict:
        return await self._inner.get_expenses_by_category(date_ini, date_end)

    async def get_payment_method_graph(
        self, date_ini: str, date_end: str,
    ) -> dict:
        return await self._inner.get_payment_method_graph(date_ini, date_end)

    # ── Users / Professionals ─────────────────────────────────────────────────

    async def get_professionals(self) -> dict:
        return await self._inner.get_professionals()
