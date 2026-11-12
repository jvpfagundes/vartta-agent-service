import json
import logging
import re

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```[a-zA-Z]*")
# Saída que começa como array de strings (`["`), mas não é um JSON válido (truncada/malformada)
_TRUNCATED_ARRAY_RE = re.compile(r'^\[\s*"')


def format_for_whatsapp(text: str) -> str:
    """Converte eventuais marcações Markdown (`**negrito**`) para o padrão nativo do WhatsApp (`*negrito*`)."""
    # Substitui **texto** por *texto*
    formatted = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    # Remove cabeçalhos ### no início de linhas
    formatted = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", formatted, flags=re.M)
    return formatted.strip()


def _find_json_array(text: str) -> tuple[list, int, int] | None:
    """Retorna (array, início, fim) do primeiro array JSON válido encontrado no texto, ou None."""
    decoder = json.JSONDecoder(strict=False)  # aceita quebras de linha literais dentro de strings
    start = text.find("[")
    while start != -1:
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            value = None
            end = start
        if isinstance(value, list):
            return value, start, end
        start = text.find("[", start + 1)
    return None


def _extract_complete_strings(text: str) -> list[str]:
    """Extrai as strings JSON completas de um array truncado ou malformado (ex.: `["Oi", "Para am`).

    Para na primeira string incompleta ou item que não seja string: o último item cortado
    não é enviado, para o cliente nunca receber sintaxe JSON nem frases pela metade.
    """
    decoder = json.JSONDecoder(strict=False)  # aceita quebras de linha literais dentro de strings
    items: list[str] = []
    index = text.find("[") + 1
    length = len(text)
    while index < length:
        while index < length and text[index] in " \t\r\n,":
            index += 1
        if index >= length or text[index] != '"':
            break
        try:
            value, index = decoder.raw_decode(text, index)
        except ValueError:
            break
        if not isinstance(value, str):
            break
        items.append(value)
    return items


def parse_reply_messages(raw_text: str, max_messages: int) -> list[str]:
    """Extrai a lista de mensagens a enviar ao cliente a partir da saída do LLM.

    O LLM é instruído a responder um array JSON de strings. O parser é tolerante:
    ignora cercas de código e texto antes/depois do array (o texto fora do array é descartado
    e registrado em log). Se a saída for um array JSON truncado ou malformado (começa com `["`),
    aproveita as strings completas; se não houver nenhuma, devolve lista vazia para o chamador
    usar a mensagem de fallback, em vez de enviar sintaxe JSON crua. Se a saída não contiver um
    array JSON de strings, o texto inteiro vira UMA mensagem. Em arrays mistos, itens que não são
    string são descartados (com log). Quebras de linha literais dentro das strings são aceitas. Partes vazias são descartadas e o
    excedente além de `max_messages` é concatenado na última parte.
    """
    if raw_text is None:
        return []

    stripped = _CODE_FENCE_RE.sub("", raw_text).strip()
    found = _find_json_array(stripped)

    starts_as_string_array = bool(_TRUNCATED_ARRAY_RE.match(stripped))
    if starts_as_string_array and (found is None or found[1] != 0):
        items: list = _extract_complete_strings(stripped)
        logger.warning(
            "LLM reply looks like a truncated/malformed JSON array; recovered %d complete item(s)",
            len(items),
        )
    elif found is None or (found[0] and not any(isinstance(item, str) for item in found[0])):
        single = format_for_whatsapp(raw_text)
        return [single] if single else []
    else:
        items, start, end = found
        if not all(isinstance(item, str) for item in items):
            # Array misto (ex.: `["Oi", 1]`): só os itens string são mensagens; nunca envia JSON cru.
            logger.warning("Non-string items in the LLM reply array were discarded")
            items = [item for item in items if isinstance(item, str)]
        outside = (stripped[:start] + " " + stripped[end:]).strip()
        if outside:
            logger.warning("Text outside the JSON array in LLM reply was discarded: %.120r", outside)

    parts = [formatted for item in items if (formatted := format_for_whatsapp(item))]

    limit = max(1, max_messages)
    if len(parts) > limit:
        tail = "\n\n".join(parts[limit - 1 :])
        parts = [*parts[: limit - 1], tail]

    return parts
