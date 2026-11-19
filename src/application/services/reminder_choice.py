import re
import unicodedata

# Aceita apenas a resposta curta ao lembrete: "1", "2" ou "3" (opcionalmente com ponto/exclamação
# ou o emoji de tecla 1️⃣/2️⃣/3️⃣), ou palavras curtas de confirmação ("confirmo", "confirmar",
# "confirmado", "confirmada"). Qualquer outra frase segue para o atendimento normal.
_CHOICE_RE = re.compile(r"^\s*([123])(?:\ufe0f?\u20e3)?\s*[.!]?\s*$")
_CONFIRM_WORDS_RE = re.compile(r"^(?:confirmo|confirmar|confirmad[oa])(?:\s*[.!]+)?$")


def _normalize_word(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", (text or "").strip().lower())
    no_accents = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", no_accents)


def parse_reminder_choice(text: str) -> str | None:
    """Devolve "1" (confirmar), "2" (cancelar) ou "3" (reagendar) se a mensagem for só essa
    resposta; senão None."""
    match = _CHOICE_RE.match(text or "")
    if match:
        return match.group(1)

    if _CONFIRM_WORDS_RE.match(_normalize_word(text or "")):
        return "1"

    return None
