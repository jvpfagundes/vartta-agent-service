import re

# Aceita apenas a resposta curta ao lembrete: "1", "2" ou "3" (opcionalmente com ponto/exclamação
# ou o emoji de tecla 1️⃣/2️⃣/3️⃣). Qualquer outra frase segue para o atendimento normal.
_CHOICE_RE = re.compile(r"^\s*([123])(?:\ufe0f?\u20e3)?\s*[.!]?\s*$")


def parse_reminder_choice(text: str) -> str | None:
    """Devolve "1" (confirmar), "2" (cancelar) ou "3" (reagendar) se a mensagem for só essa
    resposta; senão None."""
    match = _CHOICE_RE.match(text or "")
    return match.group(1) if match else None
