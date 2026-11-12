import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

DEFAULT_WINDOW = 16
# Acima de window * fator o recuo da janela é considerado anormal e gera um aviso
WINDOW_OVERFLOW_WARNING_FACTOR = 2


def _tool_call_ids(message: AIMessage) -> list[str]:
    return [call.get("id") for call in message.tool_calls]


def _is_empty_ai(message: AIMessage) -> bool:
    if message.tool_calls:
        return False
    content = message.content
    if isinstance(content, str):
        return not content.strip()
    return not content


def _clean_sequence(messages: list[Any]) -> list[Any]:
    """Mantém apenas mensagens que formam turnos válidos para o Gemini.

    Regras aplicadas em ordem cronológica:
    - AIMessage com tool_calls só é válida se vier logo após uma HumanMessage ou ToolMessage e se
      TODOS os seus tool_call ids tiverem ToolMessage em seguida (bloco contíguo de respostas);
    - ToolMessage fora de um bloco assim (órfã, duplicada ou parcial) é descartada;
    - AIMessage sem conteúdo e sem tool_calls é descartada.
    """
    result: list[Any] = []
    index = 0
    total = len(messages)
    while index < total:
        message = messages[index]

        if isinstance(message, AIMessage) and message.tool_calls:
            # Bloco contíguo de ToolMessages que sucede esta AIMessage
            end = index + 1
            while end < total and isinstance(messages[end], ToolMessage):
                end += 1
            block = messages[index + 1 : end]

            call_ids = _tool_call_ids(message)
            answers: dict[str, ToolMessage] = {}
            for tool_message in block:
                answers.setdefault(tool_message.tool_call_id, tool_message)

            previous = result[-1] if result else None
            valid_position = isinstance(previous, (HumanMessage, ToolMessage))
            fully_answered = all(call_id and call_id in answers for call_id in call_ids)

            if valid_position and fully_answered:
                result.append(message)
                seen: set[str] = set()
                for tool_message in block:
                    tool_call_id = tool_message.tool_call_id
                    if tool_call_id in call_ids and tool_call_id not in seen:
                        seen.add(tool_call_id)
                        result.append(tool_message)
            # Caso contrário descarta a AIMessage junto com as respostas parciais dela
            index = end
            continue

        if isinstance(message, ToolMessage):
            # ToolMessage que não foi consumida por uma AIMessage válida acima é órfã
            index += 1
            continue

        if isinstance(message, AIMessage) and _is_empty_ai(message):
            index += 1
            continue

        result.append(message)
        index += 1
    return result


def _align_window_start(cleaned: list[Any], window: int) -> list[Any]:
    """Recorta as últimas `window` mensagens garantindo que o início seja uma HumanMessage."""
    if window <= 0 or not cleaned:
        return []
    offset = max(len(cleaned) - window, 0)
    tail = cleaned[offset:]
    if isinstance(tail[0], HumanMessage):
        return tail

    for position, message in enumerate(tail):
        if isinstance(message, HumanMessage):
            return tail[position:]

    # Nenhuma HumanMessage dentro da janela: recua no histórico até a anterior mais próxima
    for position in range(offset - 1, -1, -1):
        if isinstance(cleaned[position], HumanMessage):
            return cleaned[position:]
    return []


def sanitize_messages_for_llm(messages: list[BaseMessage], window: int = DEFAULT_WINDOW) -> list[BaseMessage]:
    """Prepara o histórico para o LLM respeitando as regras de turnos do Gemini.

    Remove SystemMessage, descarta tool calls inconsistentes (órfãs ou sem resposta completa),
    recorta as últimas `window` mensagens e ajusta o início da janela para uma HumanMessage, de
    modo que o primeiro turno nunca seja AIMessage/ToolMessage. Loga em `warning` apenas quando
    mensagens inconsistentes são removidas (sinal de histórico corrompido) ou quando o recuo da
    janela passa muito de `window`; o ajuste normal do início da janela é logado em `debug`.
    """
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    cleaned = _clean_sequence(non_system)
    removed = len(non_system) - len(cleaned)

    windowed = _align_window_start(cleaned, window)
    expected_len = min(len(cleaned), max(window, 0))
    # Janela maior que o esperado: recuou no histórico; menor: avançou até uma HumanMessage
    adjusted = len(windowed) != expected_len

    if removed:
        logger.warning(
            "Sanitized LLM history: removed %d inconsistent message(s), window adjusted=%s "
            "(input=%d, output=%d)",
            removed,
            adjusted,
            len(non_system),
            len(windowed),
        )
    elif adjusted:
        logger.debug(
            "LLM history window adjusted to start at a HumanMessage (input=%d, output=%d)",
            len(non_system),
            len(windowed),
        )

    if window > 0 and len(windowed) > window * WINDOW_OVERFLOW_WARNING_FACTOR:
        # Ciclo de tool calls muito longo: o recuo até a HumanMessage ultrapassou bastante a janela
        logger.warning(
            "LLM history window grew well beyond the limit (window=%d, output=%d)",
            window,
            len(windowed),
        )
    return windowed
