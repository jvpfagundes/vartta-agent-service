import unittest

from src.application.services.reminder_choice import parse_reminder_choice


class ParseReminderChoiceTest(unittest.TestCase):
    def test_accepts_plain_and_lightly_decorated_answers(self) -> None:
        for text, expected in [
            ("1", "1"),
            ("2", "2"),
            (" 1 ", "1"),
            ("2\n", "2"),
            ("1.", "1"),
            ("2!", "2"),
            ("1\ufe0f\u20e3", "1"),
            ("2\u20e3", "2"),
            ("3", "3"),
            ("3\ufe0f\u20e3", "3"),
            ("confirmo", "1"),
            ("Confirmo!", "1"),
            ("CONFIRMAR", "1"),
            ("confirmar.", "1"),
            ("confirmado", "1"),
            ("Confirmada!", "1"),
            ("1 confirmar", "1"),
            ("1 - confirmar", "1"),
            ("1. confirmar", "1"),
            ("1 e confirmar", "1"),
            ("1\ufe0f\u20e3 confirmar", "1"),
            ("1 confirmado", "1"),
            ("um confirmar", "1"),
            ("Um confirmado", "1"),
            ("sim confirmo", "1"),
            ("Sim, confirmo!", "1"),
            ("sim confirmado", "1"),
            ("sim confirmar", "1"),
            ("Opção 1 - Confirmar", "1"),
            ("2 cancelar", "2"),
            ("2 - cancelar", "2"),
            ("dois cancelar", "2"),
            ("3 reagendar", "3"),
            ("três remarcar", "3"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(parse_reminder_choice(text), expected)

    def test_rejects_anything_else(self) -> None:
        for text in [
            "",
            "   ",
            "0",
            "4",
            "12",
            "1 2",
            "sim",
            "oi",
            "quero 1 corte",
            "1 por favor",
            "11",
            "-1",
            "não confirmo",
            "quero confirmar outro horário",
            None,
        ]:
            with self.subTest(text=text):
                self.assertIsNone(parse_reminder_choice(text))


if __name__ == "__main__":
    unittest.main()
