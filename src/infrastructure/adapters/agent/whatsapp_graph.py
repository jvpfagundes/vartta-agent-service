from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_config
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.infrastructure.adapters.agent.llm import init_llm
from src.infrastructure.adapters.agent.message_window import sanitize_messages_for_llm
from src.infrastructure.adapters.agent.whatsapp_tools import WHATSAPP_CUSTOMER_TOOLS


def _should_continue(state: MessagesState) -> str:
    # Histórico vazio acontece após o rollback de um ciclo cancelado em um thread novo
    if not state["messages"]:
        return END
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def build_whatsapp_agent_graph(
    checkpointer: BaseCheckpointSaver,
    llm: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
) -> CompiledStateGraph:
    """Compila um grafo LangGraph isolado para atendimento ao cliente final no WhatsApp.

    Possui acesso estritamente limitado às tools públicas de consulta de serviços,
    profissionais, horários livres, criação de agendamento e cadastro/atualização do nome
    completo do cliente atual (`register_customer` e `update_customer_name`, que só atuam sobre
    o telefone e o cadastro resolvidos pelo servidor).

    `llm` e `tools` são opcionais (padrão: `init_llm()` e `WHATSAPP_CUSTOMER_TOOLS`) e existem
    para permitir injetar um LLM/tools falsos nos testes.
    """
    llm = llm if llm is not None else init_llm()
    tools = list(tools) if tools is not None else WHATSAPP_CUSTOMER_TOOLS
    llm_with_tools = llm.bind_tools(tools)

    async def whatsapp_agent_node(state: MessagesState) -> dict:
        config = get_config()
        configurable = config.get("configurable") or {}
        system_prompt = configurable.get("system_prompt") or ""

        # Sempre substitui/garante que o SystemMessage no topo seja o prompt blindado atualizado.
        # O histórico é recortado (janela recente, evita estouro de contexto e diluição de guardrails)
        # e saneado para respeitar as regras de turnos do Gemini (início em HumanMessage, tool calls
        # sempre seguidas de todas as suas ToolMessages).
        recent_messages = sanitize_messages_for_llm(state["messages"])

        messages = [SystemMessage(content=system_prompt), *recent_messages]
        response = await llm_with_tools.ainvoke(messages, config=config)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(MessagesState)
    graph.add_node("agent", whatsapp_agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
