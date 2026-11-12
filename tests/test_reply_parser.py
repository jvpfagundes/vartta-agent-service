import unittest

from src.application.services.reply_parser import parse_reply_messages


class ParseReplyMessagesTest(unittest.TestCase):
    def test_valid_json_array_with_three_items(self) -> None:
        raw = '["Oi, Joao!", "Temos horario as 14h.", "Qual prefere?"]'
        self.assertEqual(
            parse_reply_messages(raw, 5),
            ["Oi, Joao!", "Temos horario as 14h.", "Qual prefere?"],
        )

    def test_array_inside_json_code_fence(self) -> None:
        raw = '```json\n["a", "b"]\n```'
        self.assertEqual(parse_reply_messages(raw, 5), ["a", "b"])

    def test_text_before_and_after_array(self) -> None:
        raw = 'Aqui esta a resposta: ["primeira", "segunda"] espero ter ajudado'
        self.assertEqual(parse_reply_messages(raw, 5), ["primeira", "segunda"])

    def test_plain_text_falls_back_to_single_message(self) -> None:
        raw = "Ola! Tudo bem? [Data] sem json aqui"
        self.assertEqual(parse_reply_messages(raw, 5), [raw])

    def test_json_object_is_single_message(self) -> None:
        raw = '{"mensagem": "oi"}'
        self.assertEqual(parse_reply_messages(raw, 5), [raw])

    def test_json_string_is_single_message(self) -> None:
        raw = '"oi"'
        self.assertEqual(parse_reply_messages(raw, 5), [raw])

    def test_array_of_non_strings_is_single_message(self) -> None:
        raw = "Opcoes: [1, 2, 3]"
        self.assertEqual(parse_reply_messages(raw, 5), [raw])

    def test_empty_items_are_discarded(self) -> None:
        raw = '["a", "", "   ", "b"]'
        self.assertEqual(parse_reply_messages(raw, 5), ["a", "b"])

    def test_items_with_surrounding_spaces_are_trimmed(self) -> None:
        self.assertEqual(parse_reply_messages('["  oi  "]', 5), ["oi"])

    def test_excess_is_concatenated_in_last_part(self) -> None:
        raw = '["1", "2", "3", "4", "5"]'
        self.assertEqual(parse_reply_messages(raw, 3), ["1", "2", "3\n\n4\n\n5"])

    def test_exactly_max_messages_is_untouched(self) -> None:
        raw = '["1", "2", "3"]'
        self.assertEqual(parse_reply_messages(raw, 3), ["1", "2", "3"])

    def test_bold_markdown_converted_in_each_part(self) -> None:
        raw = '["Ola **Joao**", "Horario **14h**"]'
        self.assertEqual(parse_reply_messages(raw, 5), ["Ola *Joao*", "Horario *14h*"])

    def test_bold_markdown_converted_in_fallback(self) -> None:
        self.assertEqual(parse_reply_messages("**oi**", 5), ["*oi*"])

    def test_empty_list_returns_empty(self) -> None:
        self.assertEqual(parse_reply_messages("[]", 5), [])

    def test_escaped_newlines_and_quotes(self) -> None:
        raw = '["Linha 1\\n• 14:00\\n• 15:00", "Ele disse \\"oi\\""]'
        self.assertEqual(
            parse_reply_messages(raw, 5),
            ["Linha 1\n• 14:00\n• 15:00", 'Ele disse "oi"'],
        )

    def test_truncated_array_sends_only_complete_items(self) -> None:
        self.assertEqual(
            parse_reply_messages('["Oi, João!", "Para amanhã tem', 5),
            ["Oi, João!"],
        )

    def test_truncated_array_with_no_complete_item_returns_empty(self) -> None:
        self.assertEqual(parse_reply_messages('["Para amanhã tem', 5), [])

    def test_truncated_array_with_bracket_inside_item_is_not_sent_raw(self) -> None:
        self.assertEqual(
            parse_reply_messages('["Horários [1] e [2]", "Para am', 5),
            ["Horários [1] e [2]"],
        )

    def test_malformed_array_never_leaks_json_syntax(self) -> None:
        parts = parse_reply_messages('["a", "b" "c"]', 5)
        self.assertTrue(all('"' not in part and "[" not in part for part in parts))

    def test_text_outside_array_is_logged(self) -> None:
        with self.assertLogs("src.application.services.reply_parser", level="WARNING") as logs:
            parts = parse_reply_messages('Claro! ["Oi"] até logo', 5)
        self.assertEqual(parts, ["Oi"])
        self.assertIn("outside the JSON array", logs.output[0])

    def test_plain_bracket_text_still_falls_back_to_single_message(self) -> None:
        self.assertEqual(parse_reply_messages("[Aviso] horário lotado", 5), ["[Aviso] horário lotado"])

    def test_literal_newline_inside_json_string_is_kept(self) -> None:
        raw = '["Oi", "Horários:\n• 10h\n• 11h"]'
        self.assertEqual(parse_reply_messages(raw, 5), ["Oi", "Horários:\n• 10h\n• 11h"])

    def test_literal_newline_in_truncated_array_keeps_complete_items(self) -> None:
        raw = '["Oi", "Horários:\n• 10h\n• 11h", "Qual prefe'
        self.assertEqual(parse_reply_messages(raw, 5), ["Oi", "Horários:\n• 10h\n• 11h"])

    def test_mixed_array_keeps_only_string_items(self) -> None:
        self.assertEqual(parse_reply_messages('["Oi", 1]', 5), ["Oi"])
        self.assertEqual(parse_reply_messages('["Oi", {"a": 1}, "Tchau"]', 5), ["Oi", "Tchau"])

    def test_mixed_array_never_leaks_json_syntax(self) -> None:
        parts = parse_reply_messages('["Oi", 1]', 5)
        self.assertTrue(all('"' not in part and "[" not in part for part in parts))

    def test_array_of_objects_is_single_message(self) -> None:
        raw = '[{"a": 1}]'
        self.assertEqual(parse_reply_messages(raw, 5), [raw])

if __name__ == "__main__":
    unittest.main()
