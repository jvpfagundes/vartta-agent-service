import asyncio
import unittest
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.application.use_cases.run_whatsapp_agent import RunWhatsAppAgentUseCase
from src.domain.conversation.models import WhatsAppConversationContext

_CONVERSATION_ID = "wa_tenant-1_5511999999999@c.us"
_CONFIG = {"configurable": {"thread_id": _CONVERSATION_ID}}

_TOOL_CALL = {"name": "list_available_slots", "args": {}, "id": "call-1", "type": "tool_call"}

AgentBehavior = Callable[[list[Any]], Awaitable[AIMessage]]


def _make_context() -> WhatsAppConversationContext:
    return WhatsAppConversationContext(
        tenant_id="tenant-1",
        instance_name="instancia",
        chat_id="5511999999999@c.us",
        customer_phone="5511999999999",
        customer_name="Joao",
        conversation_id=_CONVERSATION_ID,
        business_name="Barbearia Teste",
        custom_prompt="",
    )


def _should_continue(state: MessagesState) -> str:
    # mesma regra do grafo real (whatsapp_graph._should_continue)
    if not state["messages"]:
        return END
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


class GraphHarness:
    """Grafo pequeno, com a mesma estrutura do whatsapp_graph, e comportamento controlavel."""

    def __init__(self) -> None:
        self.agent_behaviors: list[AgentBehavior] = []
        self.agent_calls = 0
        self.agent_inputs: list[list[Any]] = []
        self.tools_gate: asyncio.Event | None = None
        self.tools_started = asyncio.Event()
        self.agent_started = asyncio.Event()
        self.graph = self._build()

    def _build(self) -> CompiledStateGraph:
        async def agent_node(state: MessagesState) -> dict:
            self.agent_inputs.append(list(state["messages"]))
            behavior = self.agent_behaviors[self.agent_calls]
            self.agent_calls += 1
            self.agent_started.set()
            return {"messages": [await behavior(state["messages"])]}

        async def tools_node(state: MessagesState) -> dict:
            self.tools_started.set()
            if self.tools_gate is not None:
                await self.tools_gate.wait()
            last = state["messages"][-1]
            return {
                "messages": [
                    ToolMessage(content="{}", tool_call_id=call["id"]) for call in last.tool_calls
                ]
            }

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tools_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        return graph.compile(checkpointer=InMemorySaver())


async def _answer(text: str) -> AIMessage:
    return AIMessage(content=f'["{text}"]')


def _answer_behavior(text: str) -> AgentBehavior:
    async def behavior(messages: list[Any]) -> AIMessage:
        return await _answer(text)

    return behavior


async def _hang(messages: list[Any]) -> AIMessage:
    await asyncio.sleep(60)
    raise AssertionError("nao deveria chegar aqui")


async def _call_tool(messages: list[Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=[_TOOL_CALL])


async def _stored_messages(graph: CompiledStateGraph) -> list[Any]:
    state = await graph.aget_state(_CONFIG)
    return list((state.values or {}).get("messages", []))


def _kinds(messages: list[Any]) -> list[str]:
    result = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            result.append("ai_tool_call")
        else:
            result.append(type(message).__name__)
    return result


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with_cancel = False
    try:
        await task
    except asyncio.CancelledError:
        with_cancel = True
    assert with_cancel, "a task deveria terminar com CancelledError"


class CancellationRollbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_in_agent_node_returns_thread_to_baseline_and_next_cycle_is_clean(
        self,
    ) -> None:
        harness = GraphHarness()
        use_case = RunWhatsAppAgentUseCase(harness.graph)
        harness.agent_behaviors = [_answer_behavior("primeira"), _hang, _answer_behavior("agregada")]

        await use_case.execute(_make_context(), "oi", schedula_client=None)
        baseline = await _stored_messages(harness.graph)
        self.assertEqual(_kinds(baseline), ["HumanMessage", "AIMessage"])

        harness.agent_started.clear()
        task = asyncio.create_task(use_case.execute(_make_context(), "quero cortar", None))
        await harness.agent_started.wait()
        # o checkpoint ja contem a HumanMessage do ciclo em andamento
        self.assertEqual(len(await _stored_messages(harness.graph)), 3)
        await _cancel(task)

        state = await harness.graph.aget_state(_CONFIG)
        self.assertEqual([m.id for m in state.values["messages"]], [m.id for m in baseline])
        self.assertEqual(state.next, ())

        replies = await use_case.execute(
            _make_context(), "quero cortar\namanha cedo", schedula_client=None
        )

        self.assertEqual(replies, ["agregada"])
        final = await _stored_messages(harness.graph)
        self.assertEqual(
            _kinds(final), ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]
        )
        self.assertIn("quero cortar\namanha cedo", final[2].content)
        self.assertEqual(sum("quero cortar" in str(m.content) for m in final), 1)

    async def test_cancel_between_tool_call_and_tool_message_rolls_back_completely(self) -> None:
        harness = GraphHarness()
        harness.tools_gate = asyncio.Event()
        use_case = RunWhatsAppAgentUseCase(harness.graph)
        harness.agent_behaviors = [
            _answer_behavior("primeira"),
            _call_tool,
            _answer_behavior("final"),
        ]
        await use_case.execute(_make_context(), "oi", schedula_client=None)
        baseline_ids = [m.id for m in await _stored_messages(harness.graph)]

        task = asyncio.create_task(use_case.execute(_make_context(), "horarios?", None))
        await harness.tools_started.wait()
        # o checkpoint tem a AIMessage com tool_calls, ainda sem a ToolMessage
        partial = await _stored_messages(harness.graph)
        self.assertEqual(_kinds(partial)[-1], "ai_tool_call")
        await _cancel(task)

        state = await harness.graph.aget_state(_CONFIG)
        self.assertEqual([m.id for m in state.values["messages"]], baseline_ids)
        self.assertEqual(state.next, ())
        self.assertFalse(
            any(isinstance(m, AIMessage) and m.tool_calls for m in state.values["messages"])
        )

    async def test_orphan_tool_call_in_history_is_removed_before_invoking(self) -> None:
        harness = GraphHarness()
        harness.agent_behaviors = [_answer_behavior("ok")]
        use_case = RunWhatsAppAgentUseCase(harness.graph)
        await harness.graph.aupdate_state(
            _CONFIG,
            {
                "messages": [
                    HumanMessage(content="antiga", id="old-human"),
                    AIMessage(content="", tool_calls=[_TOOL_CALL], id="orphan-ai"),
                ]
            },
            as_node="agent",
        )

        replies = await use_case.execute(_make_context(), "nova", schedula_client=None)

        self.assertEqual(replies, ["ok"])
        seen_by_agent = harness.agent_inputs[0]
        self.assertFalse(
            any(isinstance(m, AIMessage) and m.tool_calls for m in seen_by_agent),
            "o LLM nao pode ver a AIMessage com tool_calls orfa",
        )
        final = await _stored_messages(harness.graph)
        self.assertEqual(_kinds(final), ["HumanMessage", "HumanMessage", "AIMessage"])
        self.assertEqual(final[0].id, "old-human")

    async def test_orphan_tool_message_in_history_is_removed_before_invoking(self) -> None:
        harness = GraphHarness()
        harness.agent_behaviors = [_answer_behavior("ok")]
        use_case = RunWhatsAppAgentUseCase(harness.graph)
        await harness.graph.aupdate_state(
            _CONFIG,
            {
                "messages": [
                    HumanMessage(content="antiga", id="old-human"),
                    ToolMessage(content="{}", tool_call_id="sem-ai", id="orphan-tool"),
                ]
            },
            as_node="agent",
        )

        replies = await use_case.execute(_make_context(), "nova", schedula_client=None)

        self.assertEqual(replies, ["ok"])
        self.assertFalse(
            any(isinstance(m, ToolMessage) for m in harness.agent_inputs[0]),
            "o LLM nao pode ver a ToolMessage orfa",
        )
        final = await _stored_messages(harness.graph)
        self.assertEqual(_kinds(final), ["HumanMessage", "HumanMessage", "AIMessage"])
        self.assertNotIn("orphan-tool", [m.id for m in final])

    async def test_answered_tool_calls_are_kept(self) -> None:
        harness = GraphHarness()
        harness.agent_behaviors = [_call_tool, _answer_behavior("com tool"), _answer_behavior("2")]
        use_case = RunWhatsAppAgentUseCase(harness.graph)
        await use_case.execute(_make_context(), "horarios?", schedula_client=None)
        before = [m.id for m in await _stored_messages(harness.graph)]

        await use_case.execute(_make_context(), "obrigado", schedula_client=None)

        after = await _stored_messages(harness.graph)
        self.assertEqual([m.id for m in after[: len(before)]], before)
        self.assertEqual(
            _kinds(after),
            ["HumanMessage", "ai_tool_call", "ToolMessage", "AIMessage", "HumanMessage", "AIMessage"],
        )

    async def test_cancel_on_new_thread_does_not_break(self) -> None:
        harness = GraphHarness()
        harness.agent_behaviors = [_hang, _answer_behavior("depois")]
        use_case = RunWhatsAppAgentUseCase(harness.graph)

        task = asyncio.create_task(use_case.execute(_make_context(), "oi", None))
        await harness.agent_started.wait()
        await _cancel(task)

        state = await harness.graph.aget_state(_CONFIG)
        self.assertEqual(list((state.values or {}).get("messages", [])), [])
        self.assertEqual(state.next, ())

        replies = await use_case.execute(_make_context(), "oi\ntudo bem?", schedula_client=None)
        self.assertEqual(replies, ["depois"])
        self.assertEqual(_kinds(await _stored_messages(harness.graph)), ["HumanMessage", "AIMessage"])

    async def test_rollback_failure_is_logged_and_does_not_mask_cancellation(self) -> None:
        harness = GraphHarness()
        harness.agent_behaviors = [_hang]
        use_case = RunWhatsAppAgentUseCase(harness.graph)
        real_update = harness.graph.aupdate_state

        async def failing_update(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("banco indisponivel")

        task = asyncio.create_task(use_case.execute(_make_context(), "oi", None))
        await harness.agent_started.wait()
        harness.graph.aupdate_state = failing_update  # type: ignore[method-assign]
        try:
            with self.assertLogs("src.application.use_cases.run_whatsapp_agent", level="ERROR"):
                await _cancel(task)
        finally:
            harness.graph.aupdate_state = real_update  # type: ignore[method-assign]

    async def test_second_cancel_during_rollback_still_waits_for_rollback_to_finish(self) -> None:
        harness = GraphHarness()
        harness.agent_behaviors = [_hang]
        use_case = RunWhatsAppAgentUseCase(harness.graph)
        real_update = harness.graph.aupdate_state
        update_started = asyncio.Event()
        release_update = asyncio.Event()
        completed: list[bool] = []

        async def slow_update(*args: Any, **kwargs: Any) -> Any:
            update_started.set()
            await release_update.wait()
            result = await real_update(*args, **kwargs)
            completed.append(True)
            return result

        task = asyncio.create_task(use_case.execute(_make_context(), "oi", None))
        await harness.agent_started.wait()
        harness.graph.aupdate_state = slow_update  # type: ignore[method-assign]
        try:
            task.cancel()
            await asyncio.wait_for(update_started.wait(), timeout=5)
            task.cancel()  # segundo cancelamento enquanto o rollback roda
            for _ in range(5):
                await asyncio.sleep(0)
            self.assertFalse(task.done(), "a task deve esperar o rollback terminar")
            release_update.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            harness.graph.aupdate_state = real_update  # type: ignore[method-assign]

        self.assertEqual(completed, [True])
        self.assertEqual(await _stored_messages(harness.graph), [])

class RecordDeliveredRepliesTest(unittest.IsolatedAsyncioTestCase):
    async def _run_one_cycle(self, reply_json: str) -> tuple[GraphHarness, RunWhatsAppAgentUseCase]:
        harness = GraphHarness()
        harness.agent_behaviors = [
            lambda messages: _async_value(AIMessage(content=reply_json)),
        ]
        use_case = RunWhatsAppAgentUseCase(harness.graph)
        await use_case.execute(_make_context(), "oi", schedula_client=None)
        return harness, use_case

    async def test_partial_delivery_rewrites_last_ai_message_with_delivered_parts(self) -> None:
        harness, use_case = await self._run_one_cycle('["a", "b", "c"]')

        await use_case.record_delivered_replies(_make_context(), ["a"])

        messages = await _stored_messages(harness.graph)
        self.assertEqual(_kinds(messages), ["HumanMessage", "AIMessage"])
        self.assertEqual(messages[-1].content, '["a"]')

    async def test_nothing_delivered_removes_last_ai_message(self) -> None:
        harness, use_case = await self._run_one_cycle('["a", "b"]')

        await use_case.record_delivered_replies(_make_context(), [])

        messages = await _stored_messages(harness.graph)
        self.assertEqual(_kinds(messages), ["HumanMessage"])
        state = await harness.graph.aget_state(_CONFIG)
        self.assertEqual(state.next, ())

    async def test_failure_to_read_checkpoint_does_not_raise(self) -> None:
        harness, use_case = await self._run_one_cycle('["a"]')

        async def broken(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("checkpoint indisponivel")

        harness.graph.aget_state = broken  # type: ignore[method-assign]
        with self.assertLogs("src.application.use_cases.run_whatsapp_agent", level="ERROR"):
            await use_case.record_delivered_replies(_make_context(), [])


async def _async_value(value: Any) -> Any:
    return value

if __name__ == "__main__":
    unittest.main()
