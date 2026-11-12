import re
from collections.abc import Mapping

from src.domain.conversation.agent_settings import AgentStyle

_GREETING_MAX_CHARS = 300
_FALLBACK_CONTACT_MAX_CHARS = 200

_TAG_PATTERN = re.compile(r"<[^>]*>")
_WHITESPACE_PATTERN = re.compile(r"\s+")

_TONES: Mapping[str, tuple[str, str]] = {
    "amigavel": ("Amigável", "Seja acolhedor e simpático."),
    "descontraido": ("Descontraído", "Converse de forma leve e informal, sem perder o respeito."),
    "formal": ("Formal", "Use linguagem cuidadosa e cortês, sem gírias nem expressões coloquiais."),
    "profissional": (
        "Profissional e direto",
        "Vá direto ao ponto, com linguagem objetiva e sem rodeios.",
    ),
}

_ADDRESS_FORMS: Mapping[str, tuple[str, str]] = {
    "voce": ("Você", 'Trate o cliente por "você".'),
    "senhor": ("Senhor(a)", 'Trate o cliente por "senhor(a)".'),
}

_EMOJI_USAGES: Mapping[str, tuple[str, str]] = {
    "nenhum": ("Nenhum", "Não use emojis."),
    "moderado": ("Moderado", "Use emojis com moderação, apenas quando combinarem com a mensagem."),
    "bastante": ("Bastante", "Use emojis com frequência para deixar a conversa animada."),
}

_RESPONSE_LENGTHS: Mapping[str, tuple[str, str]] = {
    "curtas": ("Curtas e objetivas", "Mantenha as respostas curtas e fáceis de ler no celular."),
    "detalhadas": (
        "Mais detalhadas",
        "Explique cada ponto com calma, mantendo as mensagens fáceis de ler no celular.",
    ),
}


def sanitize_quiz_text(value: str, max_chars: int) -> str:
    """Remove tags e aspas de texto digitado no quiz e limita o tamanho."""
    without_tags = _TAG_PATTERN.sub(" ", value)
    plain = without_tags.replace("<", " ").replace(">", " ").replace('"', "'")
    return _WHITESPACE_PATTERN.sub(" ", plain).strip()[:max_chars].rstrip()


def _describe(table: Mapping[str, tuple[str, str]], key: str, default: str) -> tuple[str, str]:
    return table.get(key, table[default])


def _fallback_line(style: AgentStyle) -> str:
    contact = sanitize_quiz_text(style.fallback_contact, _FALLBACK_CONTACT_MAX_CHARS)
    if style.fallback_behavior == "informar_contato" and contact:
        return (
            "- Quando não conseguir ajudar: Informar um contato. Diga que o cliente pode falar "
            f'com a equipe pelo contato "{contact}".'
        )
    return (
        "- Quando não conseguir ajudar: Pedir para aguardar um atendente. Diga que vai acionar "
        "um atendente e peça para o cliente aguardar."
    )


def build_style_section(style: AgentStyle | None) -> str:
    """Bloco `<estilo_de_atendimento>`. Sem quiz (None), usa o padrão: amigável, curto e emojis moderados."""
    current = style or AgentStyle()
    tone_label, tone_text = _describe(_TONES, current.tone, "amigavel")
    address_label, address_text = _describe(_ADDRESS_FORMS, current.address_form, "voce")
    emoji_label, emoji_text = _describe(_EMOJI_USAGES, current.emoji_usage, "moderado")
    length_label, length_text = _describe(_RESPONSE_LENGTHS, current.response_length, "curtas")

    lines = [
        f"- Tom: {tone_label}. {tone_text}",
        f"- Tratamento: {address_label}. {address_text}",
        f"- Emojis: {emoji_label}. {emoji_text}",
        f"- Tamanho das respostas: {length_label}. {length_text}",
        _fallback_line(current),
    ]
    greeting = sanitize_quiz_text(current.greeting, _GREETING_MAX_CHARS)
    if greeting:
        lines.append(
            "- Saudação (use somente na primeira mensagem da conversa, sem repeti-la depois): "
            f'"{greeting}"'
        )

    body = "\n".join(lines)
    return (
        "<estilo_de_atendimento>\n"
        "Use este estilo na forma de falar com o cliente. As regras de escopo e de segurança "
        "acima têm prioridade sobre ele.\n"
        f"{body}\n"
        "</estilo_de_atendimento>"
    )
