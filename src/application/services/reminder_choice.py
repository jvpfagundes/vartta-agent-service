import re
import unicodedata

# Aceita a resposta curta ao lembrete: "1", "2" ou "3" (opcionalmente com "opção",
# ponto/exclamação ou emoji de tecla 1️⃣/2️⃣/3️⃣), palavras curtas de confirmação/cancelamento/
# reagendamento ("confirmo", "confirmar", "confirmado"), ou combinações como "1 confirmar",
# "1 - confirmado", "um confirmar", "um confirmado", "sim confirmo", "sim confirmado",
# "sim confirmar". Qualquer outra frase segue para o atendimento normal.
_CHOICE_RE = re.compile(r"^(?:opca?o\s*)?([123])(?:\ufe0f?\u20e3)?\s*[.!]?\s*$")
_CONFIRM_WORDS_RE = re.compile(
    r"^(?:(?:sim\s*[,.:!-]*\s*)?(?:opca?o\s*)?(?:1(?:\ufe0f?\u20e3)?|um)\s*[-:.,e]*\s*|sim\s*[,.:!-]*\s*)?"
    r"(?:confirmo|confirmar|confirmad[oa])(?:\s+sim)?(?:\s*[.!]+)?$"
)
_CANCEL_WORDS_RE = re.compile(
    r"^(?:(?:opca?o\s*)?(?:2(?:\ufe0f?\u20e3)?|dois)\s*[-:.,e]*\s*)?"
    r"(?:cancelar|cancelo|cancelad[oa])(?:\s*[.!]+)?$"
)
_RESCHEDULE_WORDS_RE = re.compile(
    r"^(?:(?:opca?o\s*)?(?:3(?:\ufe0f?\u20e3)?|tres)\s*[-:.,e]*\s*)?"
    r"(?:reagendar|reagendo|remarcar|remarco)(?:\s*[.!]+)?$"
)


def _normalize_word(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", (text or "").strip().lower())
    no_accents = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", no_accents)


def parse_reminder_choice(text: str) -> str | None:
    """Devolve "1" (confirmar), "2" (cancelar) ou "3" (reagendar) se a mensagem for só essa
    resposta; senão None."""
    normalized = _normalize_word(text or "")
    if not normalized:
        return None

    match = _CHOICE_RE.match(normalized)
    if match:
        return match.group(1)

    if _CONFIRM_WORDS_RE.match(normalized):
        return "1"
    if _CANCEL_WORDS_RE.match(normalized):
        return "2"
    if _RESCHEDULE_WORDS_RE.match(normalized):
        return "3"

    return None
