import unittest
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from src.infrastructure.adapters.agent.whatsapp_graph import build_whatsapp_agent_graph

_THREAD_ID = "wa_tenant-1_5511999999999@c.us"
_CONFIG = {"configurable": {"thread_id": _THREAD_ID, "system_prompt": "prompt de teste"}}

_TOOL_TRIGGER = "USE_TOOL"


class GeminiTurnError(Exception):
    """Equivalente ao 400 INVALID_ARGUMENT do Gemini para turnos de function call inválidos."""


def validate_gemini_turns(messages: list[BaseMessage]) -> None:
    """Valida as regras de turnos do Gemini, levantando erro como a API faria."""
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    if not non_system:
        raise GeminiTurnError("contents is not specified")
    if not isinstance(non_system[0], HumanMessage):
        raise GeminiTurnError(
            "Please ensure that function call turn comes immediately after a user turn or after "
            f"a function response turn (first turn is {type(non_system[0]).__name__})."
        )
    seen_call_ids: set[str] = set()
    for index, message in enumerate(non_system):
        if isinstance(message, AIMessage) and message.tool_calls:
            if not isinstance(non_system[index - 1], (HumanMessage, ToolMessage)):
                raise GeminiTurnError(
                    "Please ensure that function call turn comes immediately after a user turn "
                    "or after a function response turn."
                )
            call_ids = {call["id"] for call in message.tool_calls}
            seen_call_ids |= call_ids
            answered = set()
            for following in non_system[index + 1 :]:
                if not isinstance(following, ToolMessage):
                    break
                answered.add(following.tool_call_id)
            if not call_ids <= answered:
                raise GeminiTurnError(
                    "Please ensure that the number of function response parts is equal to the "
                    "number of function call parts of the function call turn."
                )
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in seen_call_ids:
                raise GeminiTurnError("function response without a preceding function call")
            if not isinstance(non_system[index - 1], (AIMessage, ToolMessage)):
                raise GeminiTurnError("function response must follow a function call turn")


class ValidatingFakeChatModel(BaseChatModel):
    """LLM falso que valida os turnos recebidos e responde de forma determinística.

    Se a última mensagem for uma HumanMessage contendo `USE_TOOL`, pede uma tool call; se for uma
    ToolMessage, responde com texto final.
    """

    received: list[list[BaseMessage]] = Field(default_factory=list)
    call_counter: int = 0

    @property
    def _llm_type(self) -> str:
        return "validating-fake-chat-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ValidatingFakeChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        validate_gemini_turns(messages)
        last = messages[-1]
        if isinstance(last, HumanMessage) and _TOOL_TRIGGER in str(last.content):
            self.call_counter += 1
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fake_lookup",
                        "args": {"query": "x"},
                        "id": f"call-{self.call_counter}",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            response = AIMessage(content='["ok"]')
        return ChatResult(generations=[ChatGeneration(message=response)])


@tool
def fake_lookup(query: str) -> str:
    """Tool falsa de consulta usada nos testes."""
    return f"resultado de {query}"


def _build_graph(llm: ValidatingFakeChatModel) -> Any:
    return build_whatsapp_agent_graph(InMemorySaver(), llm=llm, tools=[fake_lookup])


class WhatsAppGraphSanitizationTest(unittest.IsolatedAsyncioTestCase):
    async def test_full_tool_flow_sends_valid_turns_to_llm(self) -> None:
        llm = ValidatingFakeChatModel()
        graph = _build_graph(llm)

        result = await graph.ainvoke({"messages": [HumanMessage(content=f"oi {_TOOL_TRIGGER}")]}, _CONFIG)

        types = [type(m).__name__ for m in result["messages"]]
        self.assertEqual(types, ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"])
        self.assertEqual(len(llm.received), 2)
        # SystemMessage sempre no topo, seguida do histórico começando em HumanMessage
        for call in llm.received:
            self.assertIsInstance(call[0], SystemMessage)
            self.assertIsInstance(call[1], HumanMessage)

    async def test_many_tool_cycles_never_cut_window_in_the_middle_of_tool_calls(self) -> None:
        llm = ValidatingFakeChatModel()
        graph = _build_graph(llm)

        # Cada turno gera 4 mensagens; após 6 turnos o histórico passa muito de 16 mensagens e a
        # janela [-16:] antiga cortava no meio de um ciclo AI(tool_calls)/ToolMessage
        for turn in range(6):
            await graph.ainvoke(
                {"messages": [HumanMessage(content=f"turno {turn} {_TOOL_TRIGGER}")]}, _CONFIG
            )

        state = await graph.aget_state(_CONFIG)
        self.assertEqual(len(state.values["messages"]), 6 * 4)
        self.assertEqual(len(llm.received), 6 * 2)
        for call in llm.received:
            self.assertLessEqual(len(call) - 1, 16)

    async def test_inconsistent_persisted_history_is_sanitized(self) -> None:
        llm = ValidatingFakeChatModel()
        graph = _build_graph(llm)

        # Histórico persistido com ToolMessage órfã no início, AIMessage com tool_calls sem
        # ToolMessage no meio e ToolMessage parcial de uma chamada paralela
        inconsistent = [
            ToolMessage(content="{}", tool_call_id="orphan-tool"),
            HumanMessage(content="oi"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "fake_lookup", "args": {"query": "a"}, "id": "lost-1", "type": "tool_call"}
                ],
            ),
            HumanMessage(content="quero cortar"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "fake_lookup", "args": {"query": "a"}, "id": "par-1", "type": "tool_call"},
                    {"name": "fake_lookup", "args": {"query": "b"}, "id": "par-2", "type": "tool_call"},
                ],
            ),
            ToolMessage(content="{}", tool_call_id="par-1"),
            HumanMessage(content="e ai?"),
            AIMessage(content='["ola"]'),
        ]
        await graph.aupdate_state(_CONFIG, {"messages": inconsistent}, as_node="agent")

        result = await graph.ainvoke({"messages": [HumanMessage(content="obrigado")]}, _CONFIG)

        self.assertEqual(result["messages"][-1].content, '["ok"]')
        self.assertEqual(len(llm.received), 1)

    async def test_inconsistent_history_followed_by_tool_flow(self) -> None:
        llm = ValidatingFakeChatModel()
        graph = _build_graph(llm)

        inconsistent = [
            AIMessage(content='["ola"]'),
            ToolMessage(content="{}", tool_call_id="orphan-tool"),
            HumanMessage(content="oi"),
            AIMessage(content='["tudo bem"]'),
        ]
        await graph.aupdate_state(_CONFIG, {"messages": inconsistent}, as_node="agent")

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=f"horarios {_TOOL_TRIGGER}")]}, _CONFIG
        )

        self.assertEqual(result["messages"][-1].content, '["ok"]')
        self.assertEqual(len(llm.received), 2)

    async def test_validator_rejects_the_old_window_cut(self) -> None:
        """Garante que o validador falso reproduz o erro do Gemini para uma janela mal cortada."""
        history: list[BaseMessage] = [
            ToolMessage(content="{}", tool_call_id="call-1"),
            AIMessage(content="ok"),
            HumanMessage(content="oi"),
        ]
        with self.assertRaises(GeminiTurnError):
            validate_gemini_turns([SystemMessage(content="p"), *history])


if __name__ == "__main__":
    unittest.main()
