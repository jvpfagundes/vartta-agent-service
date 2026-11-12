import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_MAX_NAME_CHARS = 80

# Nomes genéricos que não identificam a pessoa (comparados sem acento e em minúsculas).
_PLACEHOLDER_WORDS = frozenset(
    {
        "cliente",
        "client",
        "customer",
        "sem",
        "nome",
        "desconhecido",
        "contato",
        "whatsapp",
        "unknown",
        "usuario",
        "user",
        "anonimo",
    }
)


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def sanitize_customer_name(raw: str | None) -> str:
    """Limpa um nome informado por terceiros (WhatsApp ou LLM).

    Remove `<` e `>`, emojis, símbolos e dígitos, mantendo letras, espaços, hífen, apóstrofo e
    ponto. Colapsa espaços e limita a 80 caracteres.
    """
    text = str(raw or "").replace("<", "").replace(">", "")
    kept = []
    for ch in text:
        category = unicodedata.category(ch)
        if category[0] in ("L", "M") or ch in " -'.":
            kept.append(ch)
        elif ch.isspace():
            kept.append(" ")
    cleaned = re.sub(r"\s+", " ", "".join(kept)).strip(" -'.")
    return cleaned[:_MAX_NAME_CHARS].strip(" -'.")


def _words(name: str) -> list[str]:
    return [w for w in sanitize_customer_name(name).split(" ") if any(c.isalpha() for c in w)]


def name_word_count(name: str | None) -> int:
    """Quantidade de palavras com letras no nome (após a limpeza)."""
    return len(_words(name or ""))


def is_placeholder_name(name: str | None) -> bool:
    """True se o nome é vazio, sem letras (ex.: '123', emoji) ou apenas termos genéricos."""
    words = _words(name or "")
    if not words:
        return True
    normalized = [_strip_accents(w).lower().strip(".-'") for w in words]
    return all(w in _PLACEHOLDER_WORDS for w in normalized)


def name_is_complete(name: str | None) -> bool:
    """True se o nome tem 2 ou mais palavras e não é placeholder."""
    return not is_placeholder_name(name) and name_word_count(name) >= 2


# Tratamentos iniciais que não fazem parte do primeiro nome (sem acento, minúsculas, sem ponto).
_TITLE_WORDS = frozenset({"sr", "sra", "srta", "dr", "dra", "prof", "profa", "dona", "seu"})

# Partículas que só aparecem no meio do nome e nunca são o primeiro nome.
_NAME_PARTICLES = frozenset({"de", "da", "do", "dos", "das", "e", "van", "von"})

_MIN_FIRST_NAME_LETTERS = 2


def _letters_count(word: str) -> int:
    return sum(1 for ch in word if ch.isalpha())


def _format_first_name(word: str) -> str:
    """Ajusta a caixa: tudo minúsculo ou tudo maiúsculo vira capitalizado; caixa mista é mantida."""
    if word.islower() or word.isupper():
        return "-".join(part.capitalize() for part in word.split("-"))
    return word


def first_name_for_address(name: str | None) -> str | None:
    """Primeiro nome para chamar o cliente, ou None quando não há um nome utilizável.

    Ignora tratamentos iniciais (Sr., Dra., Dona...), iniciais soltas ('J.') e partículas
    ('de', 'da'). Nome de uma palavra só é devolvido como está, e acentos são preservados.
    Placeholders ('Cliente') e nomes sem letras resultam em None.
    """
    if is_placeholder_name(name):
        return None
    for word in _words(name or ""):
        core = word.strip(".-'")
        normalized = _strip_accents(core).lower()
        if (
            _letters_count(core) < _MIN_FIRST_NAME_LETTERS
            or normalized in _TITLE_WORDS
            or normalized in _NAME_PARTICLES
        ):
            continue
        if normalized in _PLACEHOLDER_WORDS:
            return None
        return _format_first_name(core)
    return None


@dataclass(frozen=True)
class CustomerProfile:
    """Resultado da consulta do cliente por telefone.

    `exists` é False quando o telefone ainda não tem cadastro (id None, nome vazio).
    """

    customer_id: str | None
    name: str
    exists: bool

    @property
    def is_new(self) -> bool:
        return not self.exists

    @property
    def address_name(self) -> str | None:
        """Primeiro nome para chamar o cliente cadastrado; None se não houver nome utilizável."""
        return first_name_for_address(self.name) if self.exists else None

    @property
    def needs_full_name(self) -> bool:
        """Cadastro existente cujo nome é placeholder ou só o primeiro nome."""
        return self.exists and not name_is_complete(self.name)


async def resolve_customer_profile(client: Any, phone: str) -> CustomerProfile | None:
    """Localiza o cliente pelo telefone. Somente leitura: nunca cadastra nem altera nada.

    Cliente inexistente resulta em `CustomerProfile(None, "", exists=False)`; o cadastro é feito
    pela ferramenta `register_customer` depois que o cliente informa nome e sobrenome.
    Nunca levanta erro de rede ou do backend: falhas são logadas e resultam em None, para o
    atendimento seguir sem perfil. `asyncio.CancelledError` sempre propaga.
    """
    if client is None or not phone:
        return None
    try:
        found = await client.find_customer_by_phone(phone)
        if not found:
            return CustomerProfile(customer_id=None, name="", exists=False)
        name = str(
            found.get("name")
            or f"{found.get('first_name') or ''} {found.get('last_name') or ''}"
        ).strip()
        customer_id = found.get("id")
        return CustomerProfile(
            customer_id=str(customer_id) if customer_id else None,
            name=name,
            exists=True,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Failed to resolve customer profile by phone")
        return None
