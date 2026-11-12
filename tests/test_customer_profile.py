import asyncio
import unittest

from src.application.services.customer_profile import (
    CustomerProfile,
    first_name_for_address,
    is_placeholder_name,
    name_is_complete,
    resolve_customer_profile,
    sanitize_customer_name,
)


class FakeClient:
    def __init__(self, found: dict | None = None, find_error: Exception | None = None) -> None:
        self._found = found
        self._find_error = find_error
        self.find_calls: list[str] = []
        self.create_calls: list[tuple[str, str]] = []

    async def find_customer_by_phone(self, phone: str) -> dict | None:
        self.find_calls.append(phone)
        if self._find_error:
            raise self._find_error
        return self._found

    async def create_customer(self, name: str, phone: str) -> dict:
        self.create_calls.append((name, phone))
        return {"id": "new-id", "name": name, "phone_number": phone}


class NameRulesTest(unittest.TestCase):
    def test_placeholders(self) -> None:
        for name in ("cliente", "Cliente ", "sem nome", "", "123", "😀", None, "Desconhecido", "Usuário"):
            with self.subTest(name=name):
                self.assertTrue(is_placeholder_name(name))
                self.assertFalse(name_is_complete(name))

    def test_single_word_is_incomplete(self) -> None:
        self.assertFalse(is_placeholder_name("Joao"))
        self.assertFalse(name_is_complete("Joao"))

    def test_full_names_are_complete(self) -> None:
        self.assertTrue(name_is_complete("Joao Silva"))
        self.assertTrue(name_is_complete("Maria de Souza"))
        self.assertTrue(name_is_complete("  joão   da  silva "))

    def test_sanitize_removes_markup_and_emoji(self) -> None:
        self.assertEqual(sanitize_customer_name("<b>Joao</b> 😀  Silva!!"), "bJoaob Silva")
        self.assertEqual(sanitize_customer_name("😀 123"), "")

    def test_sanitize_limits_length(self) -> None:
        self.assertLessEqual(len(sanitize_customer_name("a" * 200)), 80)

    def test_needs_full_name_property(self) -> None:
        self.assertTrue(CustomerProfile("1", "Cliente", True).needs_full_name)
        self.assertTrue(CustomerProfile("1", "Joao", True).needs_full_name)
        self.assertFalse(CustomerProfile("1", "Joao Silva", True).needs_full_name)
        # Cliente inexistente nao "precisa de sobrenome": precisa ser cadastrado.
        self.assertFalse(CustomerProfile(None, "", False).needs_full_name)

    def test_is_new_property(self) -> None:
        self.assertTrue(CustomerProfile(None, "", False).is_new)
        self.assertFalse(CustomerProfile("1", "Joao", True).is_new)


class FirstNameForAddressTest(unittest.TestCase):
    def test_extracts_first_name_with_proper_case_and_accents(self) -> None:
        cases = {
            "Maria Souza": "Maria",
            "maria souza": "Maria",
            "MARIA SOUZA": "Maria",
            "JOSÉ SILVA": "José",
            "joão": "João",
            "José da Silva": "José",
            "Maria de Souza": "Maria",
            "Ana Clara Souza": "Ana",
            "ana-clara souza": "Ana-Clara",
            "Dr. Carlos Lima": "Carlos",
            "Dra Paula Mendes": "Paula",
            "Seu Jorge": "Jorge",
            "McDonald Silva": "McDonald",
            "DeAndre Costa": "DeAndre",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(first_name_for_address(name), expected)

    def test_single_valid_word_is_used_even_when_uncommon(self) -> None:
        for name in ("Joao", "Wellington", "Xuxa"):
            with self.subTest(name=name):
                self.assertEqual(first_name_for_address(name), name)

    def test_leading_initial_is_skipped(self) -> None:
        # Regra implementada: iniciais soltas ('J.') não são nome; usa a próxima palavra.
        self.assertEqual(first_name_for_address("J. Silva"), "Silva")
        self.assertIsNone(first_name_for_address("J."))

    def test_placeholders_and_empty_have_no_first_name(self) -> None:
        for name in ("Cliente", "cliente ", "sem nome", "", None, "123", "😀", "Dr.", "de da"):
            with self.subTest(name=name):
                self.assertIsNone(first_name_for_address(name))

    def test_placeholder_first_word_is_never_used(self) -> None:
        # 'Cliente Teste' não é placeholder pela regra existente, mas nunca chamamos de 'Cliente'.
        self.assertFalse(is_placeholder_name("Cliente Teste"))
        self.assertIsNone(first_name_for_address("Cliente Teste"))

    def test_unsafe_characters_are_sanitized(self) -> None:
        result = first_name_for_address("<script>Maria</script> 😀 Souza")
        assert result is not None
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)
        self.assertNotIn("/", result)
        self.assertEqual(first_name_for_address("😀 Maria 😀"), "Maria")
        self.assertEqual(first_name_for_address("Maria<> Souza"), "Maria")


class AddressNameTest(unittest.TestCase):
    def test_address_name_when_exists(self) -> None:
        self.assertEqual(CustomerProfile("1", "Maria Souza", True).address_name, "Maria")
        self.assertEqual(CustomerProfile("1", "Joao", True).address_name, "Joao")
        self.assertIsNone(CustomerProfile("1", "Cliente", True).address_name)

    def test_address_name_is_none_when_not_exists(self) -> None:
        self.assertIsNone(CustomerProfile(None, "", False).address_name)
        self.assertIsNone(CustomerProfile(None, "Maria Souza", False).address_name)

    def test_existing_flags_are_unchanged(self) -> None:
        self.assertTrue(CustomerProfile("1", "Joao", True).needs_full_name)
        self.assertFalse(CustomerProfile("1", "Maria Souza", True).needs_full_name)
        self.assertTrue(CustomerProfile(None, "", False).is_new)


class ResolveCustomerProfileTest(unittest.IsolatedAsyncioTestCase):
    async def test_found_customer_exists_and_is_not_created(self) -> None:
        client = FakeClient(found={"id": "c1", "name": "Joao Silva"})

        profile = await resolve_customer_profile(client, "5511999999999")

        self.assertEqual(profile, CustomerProfile("c1", "Joao Silva", True))
        self.assertEqual(client.find_calls, ["5511999999999"])
        self.assertEqual(client.create_calls, [])

    async def test_missing_customer_is_not_created(self) -> None:
        client = FakeClient()

        profile = await resolve_customer_profile(client, "5511999999999")

        self.assertEqual(profile, CustomerProfile(None, "", False))
        assert profile is not None
        self.assertTrue(profile.is_new)
        self.assertIsNone(profile.customer_id)
        self.assertEqual(profile.name, "")
        self.assertEqual(client.create_calls, [])

    async def test_nothing_is_ever_written_by_resolve(self) -> None:
        for found in (None, {"id": "c1", "name": "Cliente"}, {"id": "c2", "name": "Joao"}):
            with self.subTest(found=found):
                client = FakeClient(found=found)
                await resolve_customer_profile(client, "5511999999999")
                self.assertEqual(client.create_calls, [])

    async def test_find_error_returns_none(self) -> None:
        client = FakeClient(find_error=RuntimeError("boom"))

        self.assertIsNone(await resolve_customer_profile(client, "5511999999999"))
        self.assertEqual(client.create_calls, [])

    async def test_none_client_or_empty_phone_returns_none(self) -> None:
        self.assertIsNone(await resolve_customer_profile(None, "5511999999999"))
        client = FakeClient()
        self.assertIsNone(await resolve_customer_profile(client, ""))
        self.assertEqual(client.find_calls, [])

    async def test_cancelled_error_propagates(self) -> None:
        client = FakeClient(find_error=asyncio.CancelledError())

        with self.assertRaises(asyncio.CancelledError):
            await resolve_customer_profile(client, "5511999999999")


if __name__ == "__main__":
    unittest.main()
