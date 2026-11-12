import json
import unittest

from src.infrastructure.adapters.agent.whatsapp_tools import (
    get_public_professionals,
    get_service_professionals,
)


class _Client:
    async def get_public_professionals(self):
        return self._data()

    async def get_service_professionals(self, service_id):
        return self._data()

    @staticmethod
    def _data():
        return {
            "professionals_list": [
                {"id": "1", "first_name": "joao", "last_name": "rteste", "schedule_name": "Agenda Barba"},
                {"id": "2", "first_name": "Murilo", "last_name": "Mendes"},
            ]
        }


class ProfessionalNamesTest(unittest.IsolatedAsyncioTestCase):
    async def test_uses_schedule_name_with_fallback(self):
        cfg = {"configurable": {"schedula_client": _Client()}}
        for tool in (get_public_professionals, get_service_professionals):
            args = {"service_id": "s"} if tool is get_service_professionals else {}
            out = json.loads(await tool.ainvoke(args, config=cfg))
            self.assertEqual(
                out["professionals"],
                [{"id": "1", "name": "Agenda Barba"}, {"id": "2", "name": "Murilo Mendes"}],
            )


if __name__ == "__main__":
    unittest.main()
