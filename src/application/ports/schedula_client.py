from abc import ABC, abstractmethod


class SchedulaBackClient(ABC):
    # ── Booking ──────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_public_services(self) -> dict: ...

    @abstractmethod
    async def get_public_professionals(self) -> dict: ...

    @abstractmethod
    async def get_service_professionals(self, service_id: str) -> dict: ...

    @abstractmethod
    async def list_available_slots(
        self,
        service_id: str,
        user_id: str,
        date: str,
        additional_service_ids: list[str] | None = None,
    ) -> dict: ...

    @abstractmethod
    async def create_booking(
        self,
        service_id: str,
        professional_id: str,
        date: str,
        time: str,
        customer_name: str,
        customer_phone: str,
        additional_service_ids: list[str] | None = None,
    ) -> dict: ...

    # ── Schedule ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_events(self, date_start: str, date_end: str) -> dict: ...

    @abstractmethod
    async def get_events_count(self, date_start: str, date_end: str) -> dict: ...

    @abstractmethod
    async def get_dashboard_total(
        self, date_start: str, date_end: str, user_id: str | None = None,
    ) -> dict: ...

    @abstractmethod
    async def get_dashboard_services(
        self, date_start: str, date_end: str, user_id: str | None = None,
    ) -> dict: ...

    @abstractmethod
    async def get_dashboard_customers(
        self, date_start: str, date_end: str, user_id: str | None = None,
        sort_by: str = "visits",
    ) -> dict: ...

    # ── Services ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_services(self) -> dict: ...

    # ── Customers ────────────────────────────────────────────────────────────

    @abstractmethod
    async def find_customer_by_phone(self, phone: str) -> dict | None: ...

    @abstractmethod
    async def create_customer(self, name: str, phone: str) -> dict: ...

    @abstractmethod
    async def update_customer_name(self, customer_id: str, name: str) -> dict: ...

    @abstractmethod
    async def search_customers(self, query: str) -> dict: ...

    @abstractmethod
    async def get_customers(
        self,
        limit: int = 10,
        offset: int = 0,
        term: str | None = None,
    ) -> dict: ...

    @abstractmethod
    async def get_customer_history(self, customer_id: str) -> dict: ...

    @abstractmethod
    async def get_missing_customers(
        self,
        last_schedule: str | None = None,
        minimum_scheduling: int = 0,
        limit: int = 10,
        offset: int = 0,
    ) -> dict: ...

    # ── Financial ────────────────────────────────────────────────────────────

    @abstractmethod
    async def list_bill_categories(self) -> dict: ...

    @abstractmethod
    async def create_bill(
        self,
        type: str,
        bill_date: str,
        description: str,
        value: float,
        category: str | None = None,
        is_paid: bool = False,
    ) -> dict: ...

    @abstractmethod
    async def get_bills(
        self,
        date_ini: str | None = None,
        date_end: str | None = None,
        limit: int = 50,
        offset: int = 0,
        term: str | None = None,
    ) -> dict: ...

    @abstractmethod
    async def get_financial_cards(self, date_ini: str, date_end: str) -> dict: ...

    @abstractmethod
    async def get_financial_statement(
        self,
        date_ini: str,
        date_end: str,
        limit: int = 50,
        offset: int = 0,
        term: str | None = None,
    ) -> dict: ...

    @abstractmethod
    async def get_expenses_by_category(
        self, date_ini: str, date_end: str,
    ) -> dict: ...

    @abstractmethod
    async def get_payment_method_graph(
        self, date_ini: str, date_end: str,
    ) -> dict: ...

    # ── Users / Professionals ────────────────────────────────────────────────

    @abstractmethod
    async def get_professionals(self) -> dict: ...
