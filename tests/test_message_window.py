import unittest

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.infrastructure.adapters.agent.message_window import sanitize_messages_for_llm


def _call(call_id: str, name: str = "list_available_slots") -> dict:
    return {"name": name, "args": {}, "id": call_id, "type": "tool_call"}


def _ai_calls(*call_ids: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[_call(call_id) for call_id in call_ids])


def _tool(call_id: str) -> ToolMessage:
    return ToolMessage(content="{}", tool_call_id=call_id)


def _tool_cycle(index: int) -> list[BaseMessage]:
    """Human -> AI(tool_calls) -> Tool -> AI final (4 mensagens)."""
    call_id = f"call-{index}"
    return [
        HumanMessage(content=f"pergunta {index}"),
        _ai_calls(call_id),
        _tool(call_id),
        AIMessage(content=f"resposta {index}"),
    ]


def assert_gemini_valid(test: unittest.TestCase, messages: list[BaseMessage]) -> None:
    test.assertTrue(messages, "lista vazia")
    test.assertIsInstance(messages[0], HumanMessage)
    for index, message in enumerate(messages):
        test.assertNotIsInstance(message, SystemMessage)
        if isinstance(message, ToolMessage):
            previous_calls = [
                m for m in messages[:index] if isinstance(m, AIMessage) and m.tool_calls
            ]
            ids = {call["id"] for m in previous_calls for call in m.tool_calls}
            test.assertIn(message.tool_call_id, ids)
        if isinstance(message, AIMessage) and message.tool_calls:
            test.assertIsInstance(messages[index - 1], (HumanMessage, ToolMessage))
            following = {
                m.tool_call_id for m in messages[index + 1 :] if isinstance(m, ToolMessage)
            }
            test.assertLessEqual({call["id"] for call in message.tool_calls}, following)


class SanitizeMessagesForLlmTest(unittest.TestCase):
    def test_short_valid_history_is_unchanged(self) -> None:
        history = _tool_cycle(1) + [HumanMessage(content="obrigado")]
        result = sanitize_messages_for_llm(history)
        self.assertEqual(result, history)
        assert_gemini_valid(self, result)

    def test_system_message_is_removed(self) -> None:
        history = [SystemMessage(content="prompt"), HumanMessage(content="oi")]
        result = sanitize_messages_for_llm(history)
        self.assertEqual(result, [history[1]])

    def test_window_starting_at_tool_message_advances_to_human(self) -> None:
        history = [
            HumanMessage(content="antigo"),
            _ai_calls("a"),
            _tool("a"),
            AIMessage(content="ok"),
            HumanMessage(content="novo"),
            AIMessage(content="resposta"),
        ]
        # a janela de 4 comecaria em ToolMessage
        result = sanitize_messages_for_llm(history, window=4)
        self.assertEqual(result, history[4:])
        assert_gemini_valid(self, result)

    def test_window_starting_at_ai_with_tool_calls_advances_to_human(self) -> None:
        history = [
            HumanMessage(content="antigo"),
            _ai_calls("a"),
            _tool("a"),
            AIMessage(content="ok"),
            HumanMessage(content="novo"),
        ]
        result = sanitize_messages_for_llm(history, window=4)
        self.assertEqual(result, [history[4]])
        assert_gemini_valid(self, result)

    def test_window_without_human_walks_back_in_full_history(self) -> None:
        history = [
            HumanMessage(content="pedido"),
            _ai_calls("a"),
            _tool("a"),
            _ai_calls("b"),
            _tool("b"),
            _ai_calls("c"),
            _tool("c"),
        ]
        result = sanitize_messages_for_llm(history, window=3)
        self.assertEqual(result, history)
        assert_gemini_valid(self, result)

    def test_no_human_anywhere_returns_empty(self) -> None:
        history = [AIMessage(content="oi"), _ai_calls("a"), _tool("a")]
        self.assertEqual(sanitize_messages_for_llm(history), [])

    def test_ai_with_tool_calls_without_tool_message_is_dropped(self) -> None:
        history = [
            HumanMessage(content="oi"),
            _ai_calls("a"),
            HumanMessage(content="quero cortar"),
            AIMessage(content="claro"),
        ]
        result = sanitize_messages_for_llm(history)
        self.assertEqual(result, [history[0], history[2], history[3]])
        assert_gemini_valid(self, result)

    def test_partially_answered_parallel_calls_are_dropped_with_their_tool_messages(self) -> None:
        history = [
            HumanMessage(content="oi"),
            _ai_calls("a", "b"),
            _tool("a"),
            HumanMessage(content="e ai?"),
        ]
        result = sanitize_messages_for_llm(history)
        self.assertEqual(result, [history[0], history[3]])
        assert_gemini_valid(self, result)

    def test_orphan_tool_message_at_start_is_dropped(self) -> None:
        history = [_tool("zzz"), HumanMessage(content="oi"), AIMessage(content="ola")]
        result = sanitize_messages_for_llm(history)
        self.assertEqual(result, history[1:])
        assert_gemini_valid(self, result)

    def test_orphan_tool_message_in_the_middle_is_dropped(self) -> None:
        history = [
            HumanMessage(content="oi"),
            AIMessage(content="ola"),
            _tool("zzz"),
            HumanMessage(content="tudo bem?"),
        ]
        result = sanitize_messages_for_llm(history)
        self.assertEqual(result, [history[0], history[1], history[3]])

    def test_parallel_tool_calls_are_preserved_intact(self) -> None:
        history = [
            HumanMessage(content="horarios?"),
            _ai_calls("a", "b"),
            _tool("a"),
            _tool("b"),
            AIMessage(content="aqui estao"),
        ]
        result = sanitize_messages_for_llm(history)
        self.assertEqual(result, history)
        assert_gemini_valid(self, result)

    def test_ai_with_tool_calls_after_plain_ai_is_dropped(self) -> None:
        history = [
            HumanMessage(content="oi"),
            AIMessage(content="ola"),
            _ai_calls("a"),
            _tool("a"),
            HumanMessage(content="e ai?"),
        ]
        result = sanitize_messages_for_llm(history)
        self.assertEqual(result, [history[0], history[1], history[4]])
        assert_gemini_valid(self, result)

    def test_ai_with_tool_calls_after_tool_message_is_valid(self) -> None:
        history = [
            HumanMessage(content="oi"),
            _ai_calls("a"),
            _tool("a"),
            _ai_calls("b"),
            _tool("b"),
        ]
        result = sanitize_messages_for_llm(history)
        self.assertEqual(result, history)
        assert_gemini_valid(self, result)

    def test_empty_ai_messages_are_removed(self) -> None:
        history = [HumanMessage(content="oi"), AIMessage(content=""), AIMessage(content="ola")]
        result = sanitize_messages_for_llm(history)
        self.assertEqual(result, [history[0], history[2]])

    def test_long_history_always_starts_with_human_for_any_window(self) -> None:
        history: list[BaseMessage] = []
        for index in range(12):
            history.extend(_tool_cycle(index))
        history.append(HumanMessage(content="final"))
        for window in range(1, 30):
            with self.subTest(window=window):
                result = sanitize_messages_for_llm(history, window=window)
                assert_gemini_valid(self, result)
                self.assertEqual(result[-1], history[-1])

    def test_default_window_cuts_in_middle_of_tool_cycle_without_invalid_start(self) -> None:
        # 5 ciclos de 4 mensagens + Human final = 21; a janela [-16:] antiga comecaria em AIMessage com tool_calls
        history: list[BaseMessage] = []
        for index in range(5):
            history.extend(_tool_cycle(index))
        history.append(HumanMessage(content="final"))
        self.assertIsInstance(history[-16], AIMessage)
        self.assertTrue(history[-16].tool_calls)
        result = sanitize_messages_for_llm(history)
        assert_gemini_valid(self, result)
        self.assertLessEqual(len(result), 16)


class SanitizeLoggingTest(unittest.TestCase):
    _LOGGER = "src.infrastructure.adapters.agent.message_window"

    def test_normal_window_cut_does_not_warn(self) -> None:
        history: list[BaseMessage] = []
        for index in range(9):
            history.extend([HumanMessage(content=f"p{index}"), AIMessage(content=f"r{index}")])
        history.append(HumanMessage(content="final"))

        with self.assertLogs(self._LOGGER, level="DEBUG") as captured:
            result = sanitize_messages_for_llm(history)

        self.assertEqual(len(result), 15)
        self.assertEqual([record.levelname for record in captured.records], ["DEBUG"])

    def test_removed_inconsistent_messages_warn(self) -> None:
        history = [HumanMessage(content="oi"), _ai_calls("a"), HumanMessage(content="de novo")]

        with self.assertLogs(self._LOGGER, level="DEBUG") as captured:
            sanitize_messages_for_llm(history)

        self.assertEqual([record.levelname for record in captured.records], ["WARNING"])
        self.assertIn("removed 1", captured.records[0].getMessage())

    def test_valid_short_history_does_not_log(self) -> None:
        with self.assertNoLogs(self._LOGGER, level="DEBUG"):
            sanitize_messages_for_llm([HumanMessage(content="oi"), AIMessage(content="ola")])

    def test_window_overflow_far_beyond_limit_warns(self) -> None:
        history: list[BaseMessage] = [HumanMessage(content="pedido")]
        for index in range(5):
            history.extend([_ai_calls(f"c{index}"), _tool(f"c{index}")])

        with self.assertLogs(self._LOGGER, level="WARNING") as captured:
            result = sanitize_messages_for_llm(history, window=3)

        self.assertEqual(len(result), len(history))
        self.assertIn("grew well beyond", captured.records[-1].getMessage())


if __name__ == "__main__":
    unittest.main()
