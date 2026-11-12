from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

TONES = ("amigavel", "descontraido", "formal", "profissional")
ADDRESS_FORMS = ("voce", "senhor")
EMOJI_USAGES = ("nenhum", "moderado", "bastante")
RESPONSE_LENGTHS = ("curtas", "detalhadas")
FALLBACK_BEHAVIORS = ("aguardar_atendente", "informar_contato")


@dataclass(frozen=True, slots=True)
class AgentStyle:
    tone: str = "amigavel"
    address_form: str = "voce"
    emoji_usage: str = "moderado"
    response_length: str = "curtas"
    greeting: str = ""
    fallback_behavior: str = "aguardar_atendente"
    fallback_contact: str = ""


def _choice(value: Any, allowed: tuple[str, ...], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_agent_style(settings: Mapping[str, Any] | None) -> AgentStyle | None:
    """Converte o bloco `quiz` do settings do ai_agent. Retorna None se o quiz não foi respondido."""
    quiz = settings.get("quiz") if settings else None
    if not isinstance(quiz, Mapping):
        return None
    defaults = AgentStyle()
    return AgentStyle(
        tone=_choice(quiz.get("tone"), TONES, defaults.tone),
        address_form=_choice(quiz.get("address_form"), ADDRESS_FORMS, defaults.address_form),
        emoji_usage=_choice(quiz.get("emoji_usage"), EMOJI_USAGES, defaults.emoji_usage),
        response_length=_choice(
            quiz.get("response_length"), RESPONSE_LENGTHS, defaults.response_length
        ),
        greeting=_text(quiz.get("greeting")),
        fallback_behavior=_choice(
            quiz.get("fallback_behavior"), FALLBACK_BEHAVIORS, defaults.fallback_behavior
        ),
        fallback_contact=_text(quiz.get("fallback_contact")),
    )


def resolve_custom_instructions(settings: Mapping[str, Any] | None, legacy_prompt: str) -> str:
    """Usa `custom_instructions` do settings; sem ele, cai no `ai_custom_prompt` legado."""
    value = settings.get("custom_instructions") if settings else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return legacy_prompt or ""
