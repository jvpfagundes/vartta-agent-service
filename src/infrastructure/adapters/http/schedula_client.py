import logging
from typing import Any

import httpx

from src.application.ports.schedula_client import SchedulaBackClient
from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


class HttpSchedulaClient(SchedulaBackClient):
    def __init__(self, tenant_id: str, user_id: str) -> None:
        self._base_url = settings.SCHEDULA_BACK_URL.rstrip("/")
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "X-Agent-Service-Token": settings.AGENT_SERVICE_SECRET,
                "X-Tenant": self._tenant_id,
                "X-User-Id": self._user_id,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        response = await self._client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    # ── Booking ───────────────────────────────────────────────────────────────

    async def get_public_services(self) -> dict:
        return await self._request("GET", "/api/booking/services")

    async def get_public_professionals(self) -> dict:
        return await self._request("GET", "/api/booking/professionals")

    async def get_service_professionals(self, service_id: str) -> dict:
        return await self._request(
            "GET", f"/api/booking/services/{service_id}/professionals"
        )

    async def list_available_slots(
        self,
        service_id: str,
        user_id: str,
        date: str,
        additional_service_ids: list[str] | None = None,
    ) -> dict:
        params: dict[str, Any] = {"service_id": service_id, "user_id": user_id, "date": date}
        if additional_service_ids:
            # httpx codifica a lista como parâmetro repetido (additional_service_ids=a&additional_service_ids=b)
            params["additional_service_ids"] = list(additional_service_ids)
        return await self._request("GET", "/api/booking/slots", params=params)

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
        body: dict[str, Any] = {
            "service_id": service_id,
            "professional_id": professional_id,
            "date": date,
            "time": time,
            "costumer_name": customer_name,
            "costumer_phone": customer_phone,
        }
        if additional_service_ids:
            body["additional_service_ids"] = list(additional_service_ids)
        return await self._request("POST", "/api/booking/create", json=body)

    # ── Schedule ──────────────────────────────────────────────────────────────

    async def get_events(self, date_start: str, date_end: str) -> dict:
        return await self._request(
            "GET",
            "/api/schedule/events",
            params={"date_start": date_start, "date_end": date_end},
        )

    async def get_events_count(self, date_start: str, date_end: str) -> dict:
        return await self._request(
            "GET",
            "/api/schedule/events/count",
            params={"date_start": date_start, "date_end": date_end},
        )

    async def get_dashboard_total(
        self, date_start: str, date_end: str, user_id: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"date_start": date_start, "date_end": date_end}
        if user_id:
            params["user_id"] = user_id
        return await self._request("GET", "/api/schedule/dashboard/total", params=params)

    async def get_dashboard_services(
        self, date_start: str, date_end: str, user_id: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"date_start": date_start, "date_end": date_end}
        if user_id:
            params["user_id"] = user_id
        return await self._request("GET", "/api/schedule/dashboard/services", params=params)

    async def get_dashboard_customers(
        self,
        date_start: str,
        date_end: str,
        user_id: str | None = None,
        sort_by: str = "visits",
    ) -> dict:
        params: dict[str, Any] = {
            "date_start": date_start,
            "date_end": date_end,
            "sort_by": sort_by,
        }
        if user_id:
            params["user_id"] = user_id
        return await self._request("GET", "/api/schedule/dashboard/costumers", params=params)

    # ── Services ──────────────────────────────────────────────────────────────

    async def get_services(self) -> dict:
        return await self._request("GET", "/api/services/")

    async def reply_to_reminder(self, phone_number: str, choice: str) -> dict:
        """Resposta "1" (confirmar) / "2" (cancelar) a um lembrete. Não faz parte da porta de
        ferramentas da IA: só o webhook a usa, e a simulação nunca deve alterar agendamentos."""
        return await self._request(
            "POST",
            "/api/schedule/reminder-reply",
            json={"phone_number": phone_number, "choice": choice},
        )

    # ── Customers ─────────────────────────────────────────────────────────────

    async def find_customer_by_phone(self, phone: str) -> dict | None:
        result = await self._request(
            "GET", "/api/costumers/by-phone", params={"phone_number": phone},
        )
        customer = result.get("costumer") if isinstance(result, dict) else None
        return customer or None

    async def create_customer(self, name: str, phone: str) -> dict:
        result = await self._request(
            "POST",
            "/api/costumers/",
            json={"name": name, "phone_number": phone},
        )
        return {"id": result["costumer_id"], "name": name, "phone_number": phone}

    async def update_customer_name(self, customer_id: str, name: str) -> dict:
        return await self._request(
            "PUT",
            "/api/costumers/",
            json={"id": customer_id, "name": name},
        )

    async def search_customers(self, query: str) -> dict:
        return await self._request(
            "GET", "/api/costumers/search", params={"search": query},
        )

    async def get_customers(
        self,
        limit: int = 10,
        offset: int = 0,
        term: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if term:
            params["term"] = term
        return await self._request("GET", "/api/costumers/", params=params)

    async def get_customer_history(self, customer_id: str) -> dict:
        return await self._request("GET", f"/api/costumers/{customer_id}/history")

    async def get_missing_customers(
        self,
        last_schedule: str | None = None,
        minimum_scheduling: int = 0,
        limit: int = 10,
        offset: int = 0,
    ) -> dict:
        params: dict[str, Any] = {
            "minimum_scheduling": minimum_scheduling,
            "limit": limit,
            "offset": offset,
        }
        if last_schedule:
            params["last_schedule"] = last_schedule
        return await self._request("GET", "/api/costumers/missing", params=params)

    # ── Financial ─────────────────────────────────────────────────────────────

    async def list_bill_categories(self) -> dict:
        return await self._request("GET", "/api/financial/categories")

    async def create_bill(
        self,
        type: str,
        bill_date: str,
        description: str,
        value: float,
        category: str | None = None,
        is_paid: bool = False,
    ) -> dict:
        payload: dict[str, Any] = {
            "type": type,
            "bill_date": bill_date,
            "description": description,
            "value": value,
            "is_paid": is_paid,
        }
        if category is not None:
            payload["category"] = category
        return await self._request("POST", "/api/financial/bills", json=payload)

    async def get_bills(
        self,
        date_ini: str | None = None,
        date_end: str | None = None,
        limit: int = 50,
        offset: int = 0,
        term: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if date_ini:
            params["date_ini"] = date_ini
        if date_end:
            params["date_end"] = date_end
        if term:
            params["term"] = term
        return await self._request("GET", "/api/financial/bills", params=params)

    async def get_financial_cards(self, date_ini: str, date_end: str) -> dict:
        return await self._request(
            "GET",
            "/api/financial/report/cards",
            params={"date_ini": date_ini, "date_end": date_end},
        )

    async def get_financial_statement(
        self,
        date_ini: str,
        date_end: str,
        limit: int = 50,
        offset: int = 0,
        term: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {
            "date_ini": date_ini,
            "date_end": date_end,
            "limit": limit,
            "offset": offset,
        }
        if term:
            params["term"] = term
        return await self._request("GET", "/api/financial/report/table", params=params)

    async def get_expenses_by_category(
        self, date_ini: str, date_end: str,
    ) -> dict:
        return await self._request(
            "GET",
            "/api/financial/bills/expenses-by-category",
            params={"date_ini": date_ini, "date_end": date_end},
        )

    async def get_payment_method_graph(
        self, date_ini: str, date_end: str,
    ) -> dict:
        return await self._request(
            "GET",
            "/api/financial/report/payment/methods",
            params={"date_ini": date_ini, "date_end": date_end},
        )

    # ── Users / Professionals ─────────────────────────────────────────────────

    async def get_professionals(self) -> dict:
        return await self._request("GET", "/api/users/")
