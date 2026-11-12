from langchain_core.messages import SystemMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.config import get_config
from langgraph.prebuilt import ToolNode

from src.application.use_cases.run_agent import _SYSTEM_PROMPT_TEMPLATE as SYSTEM_PROMPT
from src.infrastructure.adapters.agent.llm import init_llm
from src.infrastructure.adapters.agent.tools import ALL_TOOLS
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


def _should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def build_agent_graph(checkpointer: AsyncPostgresSaver) -> CompiledStateGraph:
    llm = init_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    async def agent_node(state: MessagesState) -> dict:
        config = get_config()
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = await llm_with_tools.ainvoke(messages, config=config)
        return {"messages": [response]}

    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
